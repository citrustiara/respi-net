#!/usr/bin/env python3
"""Analyse a simultaneous LSM6DS3--iPhone guided chest-IMU session.

The script deliberately treats the two devices independently.  It reports the
quality of the BLE recording before comparing signal features, so a locally
regular one-second iPhone burst cannot be mistaken for a complete trial.

Example:

    uv run python tools/analyze_imu_lsm6ds3_iphone_session.py \\
      data/raw/imu/lsm6ds3_iphone_guided/imu_chest_... \\
      --figure docs/thesis/figures/imu_lsm6ds3_klatka_piersiowa.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, periodogram, sosfiltfilt

from respi_net.imu import summarize_lsm6ds3_capture_rows
from respi_net.imu_guided_protocol import stream_coverage_summary


RESP_BAND_HZ = (0.07, 0.60)
PACED_RATE_HZ = 0.20


def _pca_first_component(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return centered @ vectors[0]


def _respiration_from_frame(
    frame: pd.DataFrame,
    *,
    origin_ms: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the host-aligned time and band-limited accelerometer PCA trace."""

    absolute_time_s = frame["Time_ms"].to_numpy(dtype=float) / 1000.0
    reference_s = absolute_time_s[0] if origin_ms is None else float(origin_ms) / 1000.0
    time_s = absolute_time_s - reference_s
    dt = np.diff(time_s)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(valid_dt):
        raise ValueError("IMU recording does not contain a usable time axis.")
    fs = float(1.0 / np.median(valid_dt))
    accelerometer = frame[["ax", "ay", "az"]].to_numpy(dtype=float)
    sos = butter(4, RESP_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    filtered = np.column_stack([sosfiltfilt(sos, accelerometer[:, index]) for index in range(3)])
    return time_s, _pca_first_component(filtered), fs


def lsm_respiration(
    csv_path: Path,
    *,
    origin_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Return a host-aligned LSM6DS3 PCA trace and UART diagnostics."""

    frame = pd.read_csv(csv_path)
    columns = ["Time_ms", "HostTime_ms", "DeviceTime_us", "SampleIndex", "ax", "ay", "az", "gx", "gy", "gz"]
    rows = frame[columns].to_numpy(dtype=float).tolist()
    diagnostics = summarize_lsm6ds3_capture_rows(rows)
    time_s, signal, fs = _respiration_from_frame(frame, origin_ms=origin_ms)
    return time_s, signal, fs, diagnostics


def iphone_respiration(
    csv_path: Path,
    *,
    origin_ms: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a host-aligned iPhone PCA trace in the same processing band."""

    frame = pd.read_csv(csv_path)
    return _respiration_from_frame(frame, origin_ms=origin_ms)


def _segment(time_s: np.ndarray, signal: np.ndarray, start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    mask = (time_s >= start_s) & (time_s < end_s)
    return time_s[mask], signal[mask]


def dominant_frequency_hz(time_s: np.ndarray, signal: np.ndarray, low_hz: float = 0.07, high_hz: float = 0.50) -> float | None:
    """Refine the strongest sinusoidal component on a dense, fixed frequency grid."""

    if len(time_s) < 20:
        return None
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.mean(signal)
    if not np.any(np.abs(signal) > 0):
        return None
    frequencies = np.linspace(low_hz, high_hz, 4301)
    phase = 2.0 * np.pi * np.outer(frequencies, time_s - time_s[0])
    cosine = np.cos(phase) @ signal
    sine = np.sin(phase) @ signal
    power = cosine * cosine + sine * sine
    return float(frequencies[int(np.argmax(power))])


def _cue_direction(cues: pd.DataFrame, time_s: np.ndarray) -> np.ndarray:
    direction = np.full(len(time_s), np.nan, dtype=float)
    for cue in cues.itertuples(index=False):
        if cue.kind not in {"inhale", "exhale"}:
            continue
        mask = (time_s >= float(cue.start_s)) & (time_s < float(cue.end_s))
        direction[mask] = 1.0 if cue.kind == "inhale" else -1.0
    return direction


def paced_direction_agreement(time_s: np.ndarray, signal: np.ndarray, cues: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return absolute directional correlation and timing shift for paced segments.

    PCA sign has no physical meaning, hence the absolute correlation is used.
    This is a check of cue-consistency, not a calibrated estimate of latency.
    """

    desired = _cue_direction(cues, time_s)
    derivative = np.gradient(signal, time_s)
    best: tuple[float, float] | None = None
    for shift_s in np.arange(-2.5, 2.51, 0.05):
        shifted = np.interp(time_s - shift_s, time_s, derivative, left=np.nan, right=np.nan)
        valid = np.isfinite(desired) & np.isfinite(shifted)
        if np.sum(valid) < 50:
            continue
        corr = float(np.corrcoef(desired[valid], shifted[valid])[0, 1])
        if np.isfinite(corr) and (best is None or abs(corr) > best[0]):
            best = (abs(corr), float(shift_s))
    return best if best is not None else (None, None)


def _paced_pre_hold_window(cues: pd.DataFrame) -> tuple[float, float]:
    """Return the continuous paced fragment before the deliberate hold."""

    first_inhale = cues.loc[cues["kind"] == "inhale"].iloc[0]
    hold = cues.loc[cues["kind"] == "hold"].iloc[0]
    return float(first_inhale.start_s), float(hold.start_s)


def spectral_peak_snr_db(time_s: np.ndarray, signal: np.ndarray) -> float | None:
    """Peak-to-median spectral ratio in the respiratory band, expressed in dB."""

    if len(time_s) < 40:
        return None
    dt = np.diff(time_s)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(valid_dt):
        return None
    fs = float(1.0 / np.median(valid_dt))
    demeaned = np.asarray(signal, dtype=float) - float(np.mean(signal))
    frequencies, power = periodogram(demeaned, fs=fs, window="hann", scaling="spectrum")
    band = (frequencies >= RESP_BAND_HZ[0]) & (frequencies <= 0.50)
    if np.count_nonzero(band) < 3:
        return None
    band_frequencies = frequencies[band]
    band_power = power[band]
    peak_index = int(np.argmax(band_power))
    peak_frequency = float(band_frequencies[peak_index])
    noise = band_power[np.abs(band_frequencies - peak_frequency) > 0.03]
    if not len(noise):
        return None
    floor = float(np.median(noise))
    if floor <= 0:
        return None
    return float(10.0 * np.log10(float(band_power[peak_index]) / floor))


def signal_agreement(
    lsm_time_s: np.ndarray,
    lsm_signal: np.ndarray,
    iphone_time_s: np.ndarray,
    iphone_signal: np.ndarray,
    *,
    start_s: float,
    end_s: float,
) -> tuple[float | None, float | None]:
    """Find absolute PCA correlation after a small timing registration.

    The iPhone and LSM6DS3 do not share a hardware clock and PCA has arbitrary
    sign.  The returned value is therefore a shape-consistency measure, not an
    absolute phase calibration.
    """

    grid = np.arange(start_s, end_s, 0.01)
    if len(grid) < 20:
        return None, None
    lsm_grid = np.interp(grid, lsm_time_s, lsm_signal, left=np.nan, right=np.nan)
    best: tuple[float, float] | None = None
    for shift_s in np.arange(-2.5, 2.51, 0.05):
        iphone_grid = np.interp(grid + shift_s, iphone_time_s, iphone_signal, left=np.nan, right=np.nan)
        valid = np.isfinite(lsm_grid) & np.isfinite(iphone_grid)
        if np.sum(valid) < 50:
            continue
        correlation = float(np.corrcoef(lsm_grid[valid], iphone_grid[valid])[0, 1])
        if np.isfinite(correlation) and (best is None or abs(correlation) > best[0]):
            best = (abs(correlation), float(shift_s))
    return best if best is not None else (None, None)


def trial_metrics(
    entry: dict[str, Any],
    session_dir: Path,
) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]:
    files = {name: Path(path) for name, path in entry["files"].items()}
    measurement_start_ms = float(entry["measurement_start_wall_ms"])
    time_s, signal, fs, lsm_diagnostics = lsm_respiration(files["lsm6ds3"], origin_ms=measurement_start_ms)
    cues = pd.read_csv(files["cues"])
    iphone = pd.read_csv(files["iphone"])
    iphone_time_s, iphone_signal, iphone_fs = iphone_respiration(files["iphone"], origin_ms=measurement_start_ms)
    duration_s = float(entry["actual_seconds"])
    iphone_rows = iphone.to_numpy(dtype=float).tolist()
    coverage = stream_coverage_summary(
        iphone_rows,
        window_start_ms=float(entry["measurement_start_wall_ms"]),
        expected_duration_s=duration_s,
        expected_sample_rate_hz=100.0,
    )

    normal_windows = cues.loc[cues["kind"] == "normal", ["start_s", "end_s"]]
    hold_windows = cues.loc[cues["kind"] == "hold", ["start_s", "end_s"]]
    paced_mask = cues["kind"].isin(["inhale", "exhale"])
    normal_time, normal_signal = _segment(time_s, signal, float(normal_windows.iloc[0].start_s), float(normal_windows.iloc[0].end_s))
    hold_start_s = float(hold_windows.iloc[0].start_s)
    hold_end_s = float(hold_windows.iloc[0].end_s)
    _, hold_signal = _segment(time_s, signal, max(hold_start_s, hold_end_s - 10.0), hold_end_s)
    rate_hz = dominant_frequency_hz(normal_time, normal_signal)
    if paced_mask.any():
        paced_start_s, paced_end_s = _paced_pre_hold_window(cues)
        paced_time, paced_signal = _segment(time_s, signal, paced_start_s, paced_end_s)
        iphone_paced_time, iphone_paced_signal = _segment(iphone_time_s, iphone_signal, paced_start_s, paced_end_s)
        rate_hz = dominant_frequency_hz(paced_time, paced_signal)
        iphone_rate_hz = dominant_frequency_hz(iphone_paced_time, iphone_paced_signal)
        direction_abs_corr, best_shift_s = paced_direction_agreement(time_s, signal, cues)
        iphone_direction_abs_corr, iphone_best_shift_s = paced_direction_agreement(iphone_time_s, iphone_signal, cues)
        interdevice_abs_corr, interdevice_shift_s = signal_agreement(
            time_s,
            signal,
            iphone_time_s,
            iphone_signal,
            start_s=paced_start_s,
            end_s=paced_end_s,
        )
        breathing_signal = paced_signal
        iphone_breathing_signal = iphone_paced_signal
    else:
        iphone_time_normal, iphone_signal_normal = _segment(
            iphone_time_s,
            iphone_signal,
            float(normal_windows.iloc[0].start_s),
            float(normal_windows.iloc[0].end_s),
        )
        iphone_rate_hz = dominant_frequency_hz(iphone_time_normal, iphone_signal_normal)
        direction_abs_corr, best_shift_s = None, None
        iphone_direction_abs_corr, iphone_best_shift_s = None, None
        interdevice_abs_corr, interdevice_shift_s = signal_agreement(
            time_s,
            signal,
            iphone_time_s,
            iphone_signal,
            start_s=float(normal_windows.iloc[0].start_s),
            end_s=float(normal_windows.iloc[0].end_s),
        )
        breathing_signal = normal_signal
        iphone_breathing_signal = iphone_signal_normal

    breathing_rms = float(np.sqrt(np.mean((breathing_signal - np.mean(breathing_signal)) ** 2)))
    hold_rms = float(np.sqrt(np.mean((hold_signal - np.mean(hold_signal)) ** 2)))
    _, iphone_hold_signal = _segment(iphone_time_s, iphone_signal, max(hold_start_s, hold_end_s - 10.0), hold_end_s)
    iphone_breathing_rms = float(np.sqrt(np.mean((iphone_breathing_signal - np.mean(iphone_breathing_signal)) ** 2)))
    iphone_hold_rms = float(np.sqrt(np.mean((iphone_hold_signal - np.mean(iphone_hold_signal)) ** 2)))
    result = {
        "trial": entry["trial"]["trial_id"],
        "duration_s": duration_s,
        "lsm_rows": int(len(time_s)),
        "lsm_sample_rate_hz": fs,
        "lsm_missing_samples": int(lsm_diagnostics["missing_samples"]),
        "lsm_malformed_lines": int(entry.get("lsm6ds3", {}).get("malformed_lines", 0)),
        "resp_rate_hz": rate_hz,
        "resp_rate_bpm": None if rate_hz is None else 60.0 * rate_hz,
        "lsm_spectral_peak_snr_db": spectral_peak_snr_db(paced_time if paced_mask.any() else normal_time, breathing_signal),
        "steady_hold_to_breath_rms": hold_rms / breathing_rms if breathing_rms else None,
        "paced_direction_abs_corr": direction_abs_corr,
        "paced_direction_best_shift_s": best_shift_s,
        "iphone_rows": int(len(iphone)),
        "iphone_sample_rate_hz": iphone_fs,
        "iphone_resp_rate_hz": iphone_rate_hz,
        "iphone_resp_rate_bpm": None if iphone_rate_hz is None else 60.0 * iphone_rate_hz,
        "iphone_spectral_peak_snr_db": spectral_peak_snr_db(
            iphone_paced_time if paced_mask.any() else iphone_time_normal,
            iphone_breathing_signal,
        ),
        "iphone_steady_hold_to_breath_rms": iphone_hold_rms / iphone_breathing_rms if iphone_breathing_rms else None,
        "iphone_paced_direction_abs_corr": iphone_direction_abs_corr,
        "iphone_paced_direction_best_shift_s": iphone_best_shift_s,
        "interdevice_paced_abs_corr": interdevice_abs_corr,
        "interdevice_best_shift_s": interdevice_shift_s,
        "iphone_time_span_s": float((iphone["Time_ms"].iloc[-1] - iphone["Time_ms"].iloc[0]) / 1000.0) if len(iphone) > 1 else 0.0,
        "iphone_sample_coverage_percent": coverage["sample_coverage_percent"],
        "iphone_time_coverage_percent": coverage["time_coverage_percent"],
        "iphone_first_sample_offset_s": coverage["first_sample_offset_s"],
        "iphone_last_sample_offset_s": coverage["last_sample_offset_s"],
        "iphone_largest_gap_s": coverage["largest_gap_s"],
        "iphone_missing_ble_batches": int(entry.get("iphone", {}).get("ble_batches", {}).get("missing_batches", 0)),
    }
    return result, (time_s, signal, iphone_time_s, iphone_signal, cues)


def plot_lsm_trials(
    traces: list[tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]],
    output_path: Path,
) -> None:
    colours = {
        "settle": "#e5e7eb",
        "normal": "#f3f4f6",
        "inhale": "#bfdbfe",
        "exhale": "#bbf7d0",
        "hold": "#fecaca",
    }
    captions = {"settle": "ułożenie", "normal": "swobodnie", "inhale": "wdech", "exhale": "wydech", "hold": "wstrzymanie"}
    trial_labels = {
        "natural-hold-r1": "swobodny + wstrzymanie, 1",
        "natural-hold-r2": "swobodny + wstrzymanie, 2",
        "paced-r1": "sterowany 12/min + wstrzymanie, 1",
        "paced-r2": "sterowany 12/min + wstrzymanie, 2",
    }
    fig, axes = plt.subplots(len(traces), 1, figsize=(11.0, 10.5), constrained_layout=True)
    for index, (metrics, (time_s, signal, _iphone_time_s, _iphone_signal, cues)) in enumerate(traces):
        axis = axes[index]
        scaled = (signal - np.mean(signal)) / max(float(np.std(signal)), 1e-12)
        for cue in cues.itertuples(index=False):
            axis.axvspan(float(cue.start_s), float(cue.end_s), color=colours.get(cue.kind, "#f3f4f6"), alpha=0.55, lw=0)
        axis.plot(time_s, scaled, color="#1d4ed8", lw=0.85)
        title = f"{trial_labels.get(metrics['trial'], metrics['trial'])}: {metrics['resp_rate_bpm']:.2f} oddechu/min"
        if metrics["steady_hold_to_breath_rms"] is not None:
            title += f", RMS końca wstrzymania / oddechu = {metrics['steady_hold_to_breath_rms']:.3f}"
        if metrics["paced_direction_abs_corr"] is not None:
            title += f", |r| komend kierunku = {metrics['paced_direction_abs_corr']:.3f}"
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("PCA [znorm.]", fontsize=9)
        axis.grid(alpha=0.25)
        if index == 0:
            handles = [plt.Rectangle((0, 0), 1, 1, color=colour, alpha=0.55) for kind, colour in colours.items()]
            axis.legend(handles, [captions[kind] for kind in colours], loc="upper right", ncol=5, fontsize=7)
    axes[-1].set_xlabel("Czas od startu próby [s]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_comparison_trials(
    traces: list[tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]]],
    output_path: Path,
) -> None:
    """Plot standardized, host-aligned PCA traces of both IMUs for each trial."""

    colours = {"normal": "#f3f4f6", "inhale": "#bfdbfe", "exhale": "#bbf7d0", "hold": "#fecaca"}
    captions = {"normal": "swobodnie", "inhale": "wdech", "exhale": "wydech", "hold": "wstrzymanie"}
    fig, axes = plt.subplots(len(traces), 1, figsize=(11.0, 6.7), constrained_layout=True, sharex=True)
    if len(traces) == 1:
        axes = [axes]
    for index, (metrics, (lsm_time_s, lsm_signal, iphone_time_s, iphone_signal, cues)) in enumerate(traces):
        axis = axes[index]
        for cue in cues.itertuples(index=False):
            if cue.kind in colours:
                axis.axvspan(float(cue.start_s), float(cue.end_s), color=colours[cue.kind], alpha=0.50, lw=0)
        lsm_scaled = (lsm_signal - np.mean(lsm_signal)) / max(float(np.std(lsm_signal)), 1e-12)
        iphone_scaled = (iphone_signal - np.mean(iphone_signal)) / max(float(np.std(iphone_signal)), 1e-12)
        axis.plot(lsm_time_s, lsm_scaled, color="#1d4ed8", lw=0.85, label="LSM6DS3")
        axis.plot(iphone_time_s, iphone_scaled, color="#c2410c", lw=0.85, alpha=0.88, label="iPhone")
        title = (
            f"próba {index + 1}: LSM6DS3 {metrics['resp_rate_bpm']:.2f}, "
            f"iPhone {metrics['iphone_resp_rate_bpm']:.2f} oddechu/min"
        )
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("PCA [znorm.]", fontsize=9)
        axis.grid(alpha=0.25)
        if index == 0:
            handles = [plt.Rectangle((0, 0), 1, 1, color=colour, alpha=0.50) for colour in colours.values()]
            handles.extend([plt.Line2D([], [], color="#1d4ed8"), plt.Line2D([], [], color="#c2410c")])
            axis.legend(handles, [captions[kind] for kind in colours] + ["LSM6DS3", "iPhone"], loc="upper right", ncol=3, fontsize=7)
    axes[-1].set_xlabel("Czas od startu próby [s]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Katalog sesji zawierający manifest.json i pliki CSV.")
    parser.add_argument("--figure", type=Path, help="Docelowy wykres PNG z seriami LSM6DS3.")
    parser.add_argument("--comparison-figure", type=Path, help="Docelowy wykres PNG porównujący PCA LSM6DS3 i iPhone.")
    parser.add_argument("--metrics", type=Path, help="Docelowa tabela CSV z metrykami.")
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    accepted = [entry for entry in manifest.get("measurements", []) if entry.get("status") == "accepted"]
    if not accepted:
        raise SystemExit("Brak zaakceptowanych prób w manifeście.")
    results = [trial_metrics(entry, session_dir) for entry in accepted]
    metrics = pd.DataFrame([result for result, _ in results])
    metrics_path = args.metrics or session_dir / "analysis_imu_lsm6ds3_iphone_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    if args.figure:
        plot_lsm_trials(results, args.figure)
    if args.comparison_figure:
        plot_comparison_trials(results, args.comparison_figure)

    columns = [
        "trial",
        "lsm_rows",
        "lsm_sample_rate_hz",
        "lsm_missing_samples",
        "resp_rate_bpm",
        "lsm_spectral_peak_snr_db",
        "steady_hold_to_breath_rms",
        "paced_direction_abs_corr",
        "iphone_rows",
        "iphone_sample_rate_hz",
        "iphone_resp_rate_bpm",
        "iphone_spectral_peak_snr_db",
        "iphone_steady_hold_to_breath_rms",
        "iphone_paced_direction_abs_corr",
        "interdevice_paced_abs_corr",
        "iphone_time_coverage_percent",
        "iphone_missing_ble_batches",
    ]
    print(metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMetryki: {metrics_path}")
    if args.figure:
        print(f"Wykres: {args.figure}")
    if args.comparison_figure:
        print(f"Wykres porównawczy: {args.comparison_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
