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
from scipy.signal import butter, sosfiltfilt

from respi_net.imu import summarize_lsm6ds3_capture_rows
from respi_net.imu_guided_protocol import stream_coverage_summary


RESP_BAND_HZ = (0.07, 0.60)
PACED_RATE_HZ = 0.20


def _pca_first_component(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return centered @ vectors[0]


def lsm_respiration(csv_path: Path) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Return time, a band-limited accelerometer PCA trace and UART diagnostics."""

    frame = pd.read_csv(csv_path)
    columns = ["Time_ms", "HostTime_ms", "DeviceTime_us", "SampleIndex", "ax", "ay", "az", "gx", "gy", "gz"]
    rows = frame[columns].to_numpy(dtype=float).tolist()
    diagnostics = summarize_lsm6ds3_capture_rows(rows)
    device_time = frame["DeviceTime_us"].to_numpy(dtype=float) / 1_000_000.0
    time_s = device_time - device_time[0]
    dt = np.diff(time_s)
    fs = float(1.0 / np.median(dt[np.isfinite(dt) & (dt > 0)]))
    accelerometer = frame[["ax", "ay", "az"]].to_numpy(dtype=float)
    sos = butter(4, RESP_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    filtered = np.column_stack([sosfiltfilt(sos, accelerometer[:, index]) for index in range(3)])
    signal = _pca_first_component(filtered)
    return time_s, signal, fs, diagnostics


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


def trial_metrics(entry: dict[str, Any], session_dir: Path) -> tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    files = {name: Path(path) for name, path in entry["files"].items()}
    time_s, signal, fs, lsm_diagnostics = lsm_respiration(files["lsm6ds3"])
    cues = pd.read_csv(files["cues"])
    iphone = pd.read_csv(files["iphone"])
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
        paced_samples = np.isfinite(_cue_direction(cues, time_s))
        paced_time = time_s[paced_samples]
        paced_signal = signal[paced_samples]
        rate_hz = dominant_frequency_hz(paced_time, paced_signal)
        direction_abs_corr, best_shift_s = paced_direction_agreement(time_s, signal, cues)
        breathing_signal = paced_signal
    else:
        direction_abs_corr, best_shift_s = None, None
        breathing_signal = normal_signal

    breathing_rms = float(np.sqrt(np.mean((breathing_signal - np.mean(breathing_signal)) ** 2)))
    hold_rms = float(np.sqrt(np.mean((hold_signal - np.mean(hold_signal)) ** 2)))
    result = {
        "trial": entry["trial"]["trial_id"],
        "duration_s": duration_s,
        "lsm_rows": int(len(time_s)),
        "lsm_sample_rate_hz": fs,
        "lsm_missing_samples": int(lsm_diagnostics["missing_samples"]),
        "lsm_malformed_lines": int(entry.get("lsm6ds3", {}).get("malformed_lines", 0)),
        "resp_rate_hz": rate_hz,
        "resp_rate_bpm": None if rate_hz is None else 60.0 * rate_hz,
        "steady_hold_to_breath_rms": hold_rms / breathing_rms if breathing_rms else None,
        "paced_direction_abs_corr": direction_abs_corr,
        "paced_direction_best_shift_s": best_shift_s,
        "iphone_rows": int(len(iphone)),
        "iphone_time_span_s": float((iphone["Time_ms"].iloc[-1] - iphone["Time_ms"].iloc[0]) / 1000.0) if len(iphone) > 1 else 0.0,
        "iphone_sample_coverage_percent": coverage["sample_coverage_percent"],
        "iphone_time_coverage_percent": coverage["time_coverage_percent"],
        "iphone_first_sample_offset_s": coverage["first_sample_offset_s"],
        "iphone_last_sample_offset_s": coverage["last_sample_offset_s"],
        "iphone_largest_gap_s": coverage["largest_gap_s"],
        "iphone_missing_ble_batches": int(entry.get("iphone", {}).get("ble_batches", {}).get("missing_batches", 0)),
    }
    return result, (time_s, signal, cues)


def plot_lsm_trials(traces: list[tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, pd.DataFrame]]], output_path: Path) -> None:
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
    for index, (metrics, (time_s, signal, cues)) in enumerate(traces):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Katalog sesji zawierający manifest.json i pliki CSV.")
    parser.add_argument("--figure", type=Path, help="Docelowy wykres PNG z czterema seriami LSM6DS3.")
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

    columns = [
        "trial",
        "lsm_rows",
        "lsm_sample_rate_hz",
        "lsm_missing_samples",
        "resp_rate_bpm",
        "steady_hold_to_breath_rms",
        "paced_direction_abs_corr",
        "iphone_rows",
        "iphone_time_coverage_percent",
        "iphone_missing_ble_batches",
    ]
    print(metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMetryki: {metrics_path}")
    if args.figure:
        print(f"Wykres: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
