#!/usr/bin/env python3
"""Analyse the simultaneous HB100/A121 guided comparison.

The analysis keeps the two questions separate:

* can the inexpensive HB100 path detect that paced breathing is present, and
* does its single real IF channel preserve inhale/exhale direction?

Both sensors are aligned on host timestamps.  A121 Sparse IQ is reduced to a
fixed, phase-coherent range gate chosen from the modal range selected by the
Acconeer breathing reference app.  The cue delay is estimated only from A121
and then applied unchanged to HB100.  HB100 polarity is calibrated on the
first paced block and evaluated without recalibration on the second block.

Run from the repository root with::

    uv run python tools/analyze_hb100_a121_comparison.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, periodogram, sosfiltfilt

from respi_net.a121 import parse_json_array


ROOT = Path(__file__).resolve().parents[1]
GUIDED_DIR = ROOT / "data" / "raw" / "hb100_a121" / "guided"
DEFAULT_FIGURE_DIR = ROOT / "docs" / "thesis" / "figures"
RADAR_CENTRE_FREQUENCY_HZ = 60.5e9
SPEED_OF_LIGHT_M_S = 299_792_458.0
PHASE_TO_DISPLACEMENT_MM = (
    SPEED_OF_LIGHT_M_S / RADAR_CENTRE_FREQUENCY_HZ / (4.0 * np.pi) * 1000.0
)
A121_COLOUR = "#0072B2"
HB100_COLOUR = "#D55E00"
TEMPLATE_COLOUR = "#222222"


@dataclass
class ComparisonResult:
    run: int
    distance_cm: int
    repeat: int
    cues: list[dict[str, Any]]
    hb_time_s: np.ndarray
    hb_raw_mv: np.ndarray
    hb_resp_mv: np.ndarray
    a121_time_s: np.ndarray
    a121_raw_mm: np.ndarray
    a121_resp_mm: np.ndarray
    cue_delay_s: float
    a121_sign: float
    hb100_sign: float
    metrics: dict[str, Any]


def _latest_session() -> Path:
    candidates = sorted(GUIDED_DIR.glob("hb100_a121_range_*/manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No guided HB100/A121 session found below {GUIDED_DIR}")
    return candidates[-1].parent


_RANGE_CUE_NAME = re.compile(r"^run_(?P<run>\d+)_range-(?P<distance>\d+)-r(?P<repeat>\d+)_\d+cm_cues\.csv$")


def discover_unregistered_range_measurements(
    session: Path,
    registered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return complete range files present on disk but absent from the manifest.

    The guided GUI writes sensor files before the operator accepts a run into the
    manifest.  A completed one-off limit retest remains analytically useful, so
    it is recovered here only when all three sidecars are present.  This function
    does not alter the original manifest.
    """

    known_cues = {str(item.get("files", {}).get("cues", "")) for item in registered}
    discovered: list[dict[str, Any]] = []
    for cue_path in sorted(session.glob("run_*_range-*_cues.csv")):
        if str(cue_path) in known_cues:
            continue
        match = _RANGE_CUE_NAME.fullmatch(cue_path.name)
        if match is None:
            continue
        run = int(match.group("run"))
        distance_cm = int(match.group("distance"))
        repeat = int(match.group("repeat"))
        stem = cue_path.name[: -len("_cues.csv")]
        hb100_path = session / f"{stem}_hb100.csv"
        a121_path = session / f"{stem}_a121.csv"
        if not hb100_path.exists() or not a121_path.exists():
            continue
        cue_df = pd.read_csv(cue_path)
        if cue_df.empty or "start_wall_ms" not in cue_df:
            continue
        discovered.append(
            {
                "step": {
                    "number": run,
                    "step_id": f"RANGE-{distance_cm}-R{repeat}",
                    "label": f"Retest zasięgowy — {distance_cm} cm",
                    "kind": "range",
                    "sensor_mode": "both",
                    "duration_s": float(cue_df["end_s"].max()),
                    "distance_cm": float(distance_cm),
                    "repeat_number": repeat,
                    "repeat_total": 1,
                    "extension": True,
                    "retry_after_failure": False,
                },
                "measurement_start_wall_ms": float(cue_df["start_wall_ms"].iloc[0]),
                "files": {"hb100": str(hb100_path), "a121": str(a121_path), "cues": str(cue_path)},
                "status": "discovered_unregistered",
            }
        )
    return discovered


def _load_matrix(series: pd.Series) -> np.ndarray:
    rows = [parse_json_array(value) for value in series]
    if not rows or min((len(row) for row in rows), default=0) == 0:
        return np.empty((len(rows), 0), dtype=float)
    width = min(len(row) for row in rows)
    return np.vstack([row[:width] for row in rows])


def _sample_rate(timestamps_ms: np.ndarray) -> float:
    differences = np.diff(np.asarray(timestamps_ms, dtype=float)) / 1000.0
    valid = differences[np.isfinite(differences) & (differences > 0)]
    if len(valid) == 0:
        raise ValueError("No positive timestamp differences")
    return 1.0 / float(np.median(valid))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    mask = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(mask) < 20:
        return float("nan")
    return float(np.corrcoef(first[mask], second[mask])[0, 1])


def _cue_pairs(cues: Iterable[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    cue_list = list(cues)
    return [
        (cue, cue_list[index + 1])
        for index, cue in enumerate(cue_list[:-1])
        if cue.get("kind") == "inhale" and cue_list[index + 1].get("kind") == "exhale"
    ]


def cue_template(
    time_s: np.ndarray,
    cues: list[dict[str, Any]],
    delay_s: float,
) -> np.ndarray:
    """Triangular paced-breathing template; other intervals remain NaN."""

    template = np.full(len(time_s), np.nan, dtype=float)
    for inhale, exhale in _cue_pairs(cues):
        start = float(inhale["start_s"]) + delay_s
        turn = float(inhale["end_s"]) + delay_s
        end = float(exhale["end_s"]) + delay_s
        inhale_mask = (time_s >= start) & (time_s < turn)
        exhale_mask = (time_s >= turn) & (time_s <= end)
        template[inhale_mask] = 1.0 - 2.0 * (time_s[inhale_mask] - start) / (turn - start)
        template[exhale_mask] = -1.0 + 2.0 * (time_s[exhale_mask] - turn) / (end - turn)
    return template


def estimate_cue_delay(
    time_s: np.ndarray,
    signal: np.ndarray,
    cues: list[dict[str, Any]],
) -> tuple[float, float]:
    """Estimate aggregate human/acquisition delay from A121 only."""

    delays = np.arange(-0.25, 1.5001, 0.025)
    best_delay = float("nan")
    best_correlation = float("nan")
    for delay in delays:
        template = cue_template(time_s, cues, float(delay))
        correlation = _correlation(signal, template)
        if not np.isfinite(correlation):
            continue
        if not np.isfinite(best_correlation) or abs(correlation) > abs(best_correlation):
            best_delay = float(delay)
            best_correlation = float(correlation)
    if not np.isfinite(best_delay):
        raise ValueError("Could not align A121 with the breathing cues")
    return best_delay, best_correlation


def _modal_reference_gate(df: pd.DataFrame, width: int) -> tuple[np.ndarray, float]:
    starts = pd.to_numeric(df.get("AcconeerRangeStartIndex"), errors="coerce")
    ends = pd.to_numeric(df.get("AcconeerRangeEndIndex"), errors="coerce")
    pairs = [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if np.isfinite(start) and np.isfinite(end) and 0 <= int(start) < int(end) <= width
    ]
    if pairs:
        (start, end), count = Counter(pairs).most_common(1)[0]
        return np.arange(start, end, dtype=int), count / len(df)

    target = pd.to_numeric(df.get("AcconeerTargetDistance_m"), errors="coerce")
    target = target[np.isfinite(target) & (target > 0)]
    distances = parse_json_array(df["Distances_m"].iloc[0])[:width]
    centre = int(np.argmin(np.abs(distances - float(np.median(target))))) if len(target) else width // 2
    return np.arange(max(0, centre - 1), min(width, centre + 2), dtype=int), 0.0


def _load_a121(
    path: Path,
    start_wall_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"Timestamp_ms", "Distances_m", "Real", "Imag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing {', '.join(sorted(missing))}")

    timestamps_ms = pd.to_numeric(df["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    time_s = (timestamps_ms - start_wall_ms) / 1000.0
    fs = _sample_rate(timestamps_ms)
    distances = parse_json_array(df["Distances_m"].iloc[0])
    real = _load_matrix(df["Real"])
    imag = _load_matrix(df["Imag"])
    width = min(len(distances), real.shape[1], imag.shape[1])
    if width == 0:
        raise ValueError(f"No complex range bins in {path}")
    distances = distances[:width]
    complex_profile = real[:, :width] + 1j * imag[:, :width]
    median_amplitude = np.median(np.abs(complex_profile), axis=0)

    gate, gate_share = _modal_reference_gate(df, width)
    weights = np.maximum(median_amplitude[gate], 0.0)
    weights /= float(np.sum(weights)) + 1e-12
    unwrapped = detrend(np.unwrap(np.angle(complex_profile), axis=0), axis=0, type="linear")
    phase_trace = unwrapped[:, gate] @ weights
    raw_mm = phase_trace * PHASE_TO_DISPLACEMENT_MM
    respiratory_sos = butter(2, [0.10, 0.50], btype="bandpass", fs=fs, output="sos")
    respiratory_mm = sosfiltfilt(respiratory_sos, phase_trace) * PHASE_TO_DISPLACEMENT_MM

    return time_s, raw_mm, respiratory_mm, fs, {
        "frames": int(len(df)),
        "gate_start_m": float(distances[int(gate[0])]),
        "gate_end_m": float(distances[int(gate[-1])]),
        "gate_bins": int(len(gate)),
        "gate_share_percent": float(gate_share * 100.0),
    }


def _load_hb100(
    path: Path,
    start_wall_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]]:
    df = pd.read_csv(path)
    host_ms = pd.to_numeric(df["HostTimestamp_ms"], errors="coerce").to_numpy(dtype=float)
    device_ms = pd.to_numeric(df["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(df["Voltage_mV"], errors="coerce").to_numpy(dtype=float)
    time_s = (host_ms - start_wall_ms) / 1000.0
    valid = np.isfinite(time_s) & np.isfinite(voltage) & (time_s >= 0.0) & (time_s <= 90.2)
    time_s, voltage = time_s[valid], voltage[valid]
    order = np.argsort(time_s)
    time_s, voltage = time_s[order], voltage[order]
    unique = np.concatenate(([True], np.diff(time_s) > 1e-6))
    time_s, voltage = time_s[unique], voltage[unique]
    grid = np.arange(0.0, min(90.0, float(time_s[-1])), 0.01)
    uniform_voltage = np.interp(grid, time_s, voltage)
    respiratory_sos = butter(3, [0.10, 0.70], btype="bandpass", fs=100.0, output="sos")
    respiratory_mv = sosfiltfilt(respiratory_sos, uniform_voltage - np.mean(uniform_voltage))
    original_voltage = pd.to_numeric(df["Voltage_mV"], errors="coerce").to_numpy(dtype=float)
    return grid, uniform_voltage, respiratory_mv, _sample_rate(device_ms), {
        "rows": int(len(df)),
        "saturation_percent": float(
            np.mean((original_voltage <= 50.0) | (original_voltage >= 3050.0)) * 100.0
        ),
    }


def _block_template_mask(
    time_s: np.ndarray,
    cues: list[dict[str, Any]],
    delay_s: float,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    template = cue_template(time_s, cues, delay_s)
    return template, np.isfinite(template) & (time_s >= start_s) & (time_s <= end_s)


def _balanced_phase_accuracy(
    time_s: np.ndarray,
    signal: np.ndarray,
    cues: list[dict[str, Any]],
    delay_s: float,
    start_s: float,
    end_s: float,
    polarity: float,
) -> float:
    derivative = np.gradient(polarity * signal, time_s)
    accuracies: list[float] = []
    for kind, expected_sign in (("inhale", -1.0), ("exhale", 1.0)):
        matches: list[np.ndarray] = []
        for cue in cues:
            if cue.get("kind") != kind:
                continue
            start = float(cue["start_s"]) + delay_s + 0.30
            end = float(cue["end_s"]) + delay_s - 0.30
            mask = (time_s >= start) & (time_s <= end) & (time_s >= start_s) & (time_s <= end_s)
            if np.any(mask):
                matches.append(np.sign(derivative[mask]) == expected_sign)
        accuracies.append(float(np.mean(np.concatenate(matches))))
    return float(np.mean(accuracies))


def _sinusoid_amplitude(
    time_s: np.ndarray,
    signal: np.ndarray,
    frequency_hz: float,
    start_s: float,
    end_s: float,
) -> float:
    mask = (time_s >= start_s) & (time_s <= end_s)
    time = time_s[mask]
    values = detrend(signal[mask], type="linear")
    design = np.column_stack(
        (
            np.sin(2.0 * np.pi * frequency_hz * time),
            np.cos(2.0 * np.pi * frequency_hz * time),
            np.ones(len(time)),
        )
    )
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(np.hypot(coefficients[0], coefficients[1]))


def _dominant_frequency(
    time_s: np.ndarray,
    signal: np.ndarray,
    start_s: float,
    end_s: float,
) -> float:
    mask = (time_s >= start_s) & (time_s <= end_s)
    values = detrend(signal[mask], type="linear")
    fs = 1.0 / float(np.median(np.diff(time_s)))
    frequencies, power = periodogram(values, fs=fs, window="hann", nfft=16384)
    band = (frequencies >= 0.10) & (frequencies <= 0.65)
    return float(frequencies[np.flatnonzero(band)[int(np.argmax(power[band]))]])


def _rms_ratio_db(
    time_s: np.ndarray,
    signal: np.ndarray,
    cues: list[dict[str, Any]],
    delay_s: float,
) -> float:
    paced = np.zeros(len(time_s), dtype=bool)
    for cue in cues:
        if cue.get("kind") in {"inhale", "exhale"}:
            paced |= (
                (time_s >= float(cue["start_s"]) + delay_s + 0.10)
                & (time_s <= float(cue["end_s"]) + delay_s - 0.10)
            )
    hold = next(cue for cue in cues if cue.get("kind") == "hold")
    hold_mask = (
        (time_s >= float(hold["start_s"]) + delay_s + 3.0)
        & (time_s <= float(hold["end_s"]) + delay_s - 3.0)
    )
    paced_rms = float(np.sqrt(np.mean(np.square(signal[paced]))))
    hold_rms = float(np.sqrt(np.mean(np.square(signal[hold_mask]))))
    return float(20.0 * np.log10((hold_rms + 1e-12) / (paced_rms + 1e-12)))


def analyse_recording(measurement: dict[str, Any]) -> ComparisonResult:
    files = {key: Path(value) for key, value in measurement["files"].items()}
    cue_df = pd.read_csv(files["cues"])
    cues = cue_df.to_dict("records")
    start_wall_ms = float(cue_df["start_wall_ms"].iloc[0])
    a_time, a_raw, a_resp, a_fs, a_metadata = _load_a121(files["a121"], start_wall_ms)
    h_time, h_raw, h_resp, h_fs, h_metadata = _load_hb100(files["hb100"], start_wall_ms)

    delay, delay_correlation = estimate_cue_delay(a_time, a_resp, cues)
    a_template, a_first = _block_template_mask(a_time, cues, delay, 10.0 + delay, 45.0 + delay)
    _, a_second = _block_template_mask(a_time, cues, delay, 60.0 + delay, 90.0)
    h_template, h_first = _block_template_mask(h_time, cues, delay, 10.0 + delay, 45.0 + delay)
    _, h_second = _block_template_mask(h_time, cues, delay, 60.0 + delay, 90.0)
    a_sign = 1.0 if _correlation(a_resp[a_first], a_template[a_first]) >= 0.0 else -1.0
    h_sign = 1.0 if _correlation(h_resp[h_first], h_template[h_first]) >= 0.0 else -1.0

    block_limits = ((10.0 + delay, 45.0 + delay), (60.0 + delay, 89.8))
    harmonic_metrics: list[dict[str, float]] = []
    for start, end in block_limits:
        amplitudes = {
            frequency: _sinusoid_amplitude(h_time, h_raw, frequency, start, end)
            for frequency in (0.20, 0.40, 0.60)
        }
        strongest_harmonic = max(amplitudes[0.40], amplitudes[0.60])
        harmonic_metrics.append(
            {
                "fundamental_amplitude_mv": amplitudes[0.20],
                "second_harmonic_amplitude_mv": amplitudes[0.40],
                "third_harmonic_amplitude_mv": amplitudes[0.60],
                "fundamental_vs_strongest_harmonic_db": float(
                    20.0 * np.log10((amplitudes[0.20] + 1e-12) / (strongest_harmonic + 1e-12))
                ),
                "dominant_frequency_hz": _dominant_frequency(h_time, h_raw, start, end),
            }
        )

    metrics: dict[str, Any] = {
        "run": int(measurement["step"]["number"]),
        "distance_cm": int(round(float(measurement["step"]["distance_cm"]))),
        "repeat": int(measurement["step"]["repeat_number"]),
        "hb100_rows": h_metadata["rows"],
        "hb100_sample_rate_hz": h_fs,
        "hb100_saturation_percent": h_metadata["saturation_percent"],
        "a121_frames": a_metadata["frames"],
        "a121_sample_rate_hz": a_fs,
        "a121_gate_start_m": a_metadata["gate_start_m"],
        "a121_gate_end_m": a_metadata["gate_end_m"],
        "a121_gate_bins": a_metadata["gate_bins"],
        "a121_gate_share_percent": a_metadata["gate_share_percent"],
        "cue_delay_s": delay,
        "a121_delay_fit_r": abs(delay_correlation),
        "a121_template_r_block1": _correlation(a_sign * a_resp[a_first], a_template[a_first]),
        "a121_template_r_block2": _correlation(a_sign * a_resp[a_second], a_template[a_second]),
        "hb100_template_r_block1": _correlation(h_sign * h_resp[h_first], h_template[h_first]),
        "hb100_template_r_block2": _correlation(h_sign * h_resp[h_second], h_template[h_second]),
        "a121_phase_accuracy_block1": _balanced_phase_accuracy(
            a_time, a_resp, cues, delay, 10.0, 45.0 + delay, a_sign
        ),
        "a121_phase_accuracy_block2": _balanced_phase_accuracy(
            a_time, a_resp, cues, delay, 60.0, 90.0, a_sign
        ),
        "hb100_phase_accuracy_block1": _balanced_phase_accuracy(
            h_time, h_resp, cues, delay, 10.0, 45.0 + delay, h_sign
        ),
        "hb100_phase_accuracy_block2": _balanced_phase_accuracy(
            h_time, h_resp, cues, delay, 60.0, 90.0, h_sign
        ),
        "a121_hold_vs_paced_db": _rms_ratio_db(a_time, a_resp, cues, delay),
        "hb100_hold_vs_paced_db": _rms_ratio_db(h_time, h_resp, cues, delay),
    }
    for index, block in enumerate(harmonic_metrics, start=1):
        for name, value in block.items():
            metrics[f"hb100_{name}_block{index}"] = value

    return ComparisonResult(
        run=metrics["run"],
        distance_cm=metrics["distance_cm"],
        repeat=metrics["repeat"],
        cues=cues,
        hb_time_s=h_time,
        hb_raw_mv=h_raw,
        hb_resp_mv=h_resp,
        a121_time_s=a_time,
        a121_raw_mm=a_raw,
        a121_resp_mm=a_resp,
        cue_delay_s=delay,
        a121_sign=a_sign,
        hb100_sign=h_sign,
        metrics=metrics,
    )


def _a121_empty_scene_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    timestamps = pd.to_numeric(df["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    fs = _sample_rate(timestamps)
    distances = parse_json_array(df["Distances_m"].iloc[0])
    real = _load_matrix(df["Real"])
    imag = _load_matrix(df["Imag"])
    width = min(len(distances), real.shape[1], imag.shape[1])
    distances = distances[:width]
    profile = real[:, :width] + 1j * imag[:, :width]
    median_amplitude = np.median(np.abs(profile), axis=0)
    candidates = np.flatnonzero(distances >= 0.25)
    gate = candidates[median_amplitude[candidates] >= np.median(median_amplitude[candidates])]
    complex_residual = profile[:, gate] - np.mean(profile[:, gate], axis=0)
    normalised_complex_rms = np.sqrt(np.mean(np.abs(complex_residual) ** 2, axis=0)) / (
        median_amplitude[gate] + 1e-12
    )
    phase = detrend(np.unwrap(np.angle(profile[:, gate]), axis=0), axis=0, type="linear")
    phase_band = sosfiltfilt(
        butter(2, [0.10, 0.60], btype="bandpass", fs=fs, output="sos"),
        phase,
        axis=0,
    )
    return {
        "frames": int(len(df)),
        "sample_rate_hz": fs,
        "normalised_complex_rms_median": float(np.median(normalised_complex_rms)),
        "phase_band_rms_rad_median": float(
            np.median(np.sqrt(np.mean(np.square(phase_band), axis=0)))
        ),
    }


def analyse_interference(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {item["step"]["step_id"]: item for item in measurements}
    a_only = _a121_empty_scene_metrics(Path(by_id["INT-A"]["files"]["a121"]))
    a_both = _a121_empty_scene_metrics(Path(by_id["INT-B"]["files"]["a121"]))

    def hb_background(path: Path) -> float:
        voltage = pd.to_numeric(pd.read_csv(path)["Voltage_mV"], errors="coerce").to_numpy(dtype=float)
        return float(np.std(detrend(voltage, type="linear")))

    hb_both = hb_background(Path(by_id["INT-B"]["files"]["hb100"]))
    hb_only = hb_background(Path(by_id["INT-C"]["files"]["hb100"]))
    return {
        "a121_only": a_only,
        "a121_with_hb100": a_both,
        "a121_normalised_complex_rms_change_percent": float(
            (a_both["normalised_complex_rms_median"] / a_only["normalised_complex_rms_median"] - 1.0)
            * 100.0
        ),
        "a121_phase_band_rms_change_percent": float(
            (a_both["phase_band_rms_rad_median"] / a_only["phase_band_rms_rad_median"] - 1.0)
            * 100.0
        ),
        "hb100_only_background_rms_mv": hb_only,
        "hb100_with_a121_background_rms_mv": hb_both,
        "hb100_background_change_db": float(20.0 * np.log10(hb_both / hb_only)),
    }


def _robust_normalise(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    centre = float(np.median(values))
    scale = 0.5 * float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    return (values - centre) / max(scale, 1e-12)


def _shade_cues(ax: plt.Axes, cues: list[dict[str, Any]], delay_s: float) -> None:
    colours = {"inhale": "#56B4E9", "exhale": "#E69F00", "hold": "#777777"}
    alpha = {"inhale": 0.080, "exhale": 0.065, "hold": 0.14}
    for cue in cues:
        kind = str(cue.get("kind"))
        if kind in colours:
            ax.axvspan(
                float(cue["start_s"]) + delay_s,
                float(cue["end_s"]) + delay_s,
                color=colours[kind],
                alpha=alpha[kind],
                linewidth=0,
            )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#dedbd2", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _result_grid(results: list[ComparisonResult]) -> dict[tuple[int, int], ComparisonResult]:
    return {(result.distance_cm, result.repeat): result for result in results}


def _core_results(results: list[ComparisonResult]) -> list[ComparisonResult]:
    """The two six-panel diagnostic figures intentionally retain paired 30–100 cm runs."""

    return [result for result in results if result.distance_cm in {30, 60, 100}]


def plot_raw_overlays(results: list[ComparisonResult], output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 10.2), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.07, top=0.90, hspace=0.18, wspace=0.06)
    by_key = _result_grid(_core_results(results))
    for row, distance in enumerate((30, 60, 100)):
        for column, repeat in enumerate((1, 2)):
            result = by_key[(distance, repeat)]
            ax = axes[row, column]
            _shade_cues(ax, result.cues, result.cue_delay_s)
            ax.plot(
                result.a121_time_s,
                _robust_normalise(result.a121_sign * result.a121_raw_mm),
                color=A121_COLOUR,
                linewidth=1.0,
                label="A121: faza IQ",
            )
            ax.plot(
                result.hb_time_s[::5],
                _robust_normalise(result.hb100_sign * result.hb_raw_mv)[::5],
                color=HB100_COLOUR,
                linewidth=0.75,
                alpha=0.82,
                label="HB100: napięcie ADC",
            )
            ax.set_xlim(0.0, 90.0)
            ax.set_ylim(-2.25, 2.25)
            ax.set_title(f"{distance} cm, powtórzenie {repeat}")
            if column == 0 and row == 1:
                ax.set_ylabel("Sygnał surowy, skala znormalizowana")
            if row == 2:
                ax.set_xlabel("Czas od początku zapisu [s]")
            _style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.extend(
        [
            Patch(facecolor="#56B4E9", alpha=0.25, label="wdech"),
            Patch(facecolor="#E69F00", alpha=0.22, label="wydech"),
            Patch(facecolor="#777777", alpha=0.28, label="wstrzymanie"),
        ]
    )
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=5, frameon=False)
    figure.suptitle("Jednoczesne surowe przebiegi A121 i HB100 (bez cyfrowej filtracji oddechowej)", y=0.992)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_respiratory_overlays(results: list[ComparisonResult], output_path: Path) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 10.2), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.07, top=0.90, hspace=0.18, wspace=0.06)
    by_key = _result_grid(_core_results(results))
    for row, distance in enumerate((30, 60, 100)):
        for column, repeat in enumerate((1, 2)):
            result = by_key[(distance, repeat)]
            ax = axes[row, column]
            _shade_cues(ax, result.cues, result.cue_delay_s)
            a_signal = _robust_normalise(result.a121_sign * result.a121_resp_mm)
            h_signal = _robust_normalise(result.hb100_sign * result.hb_resp_mv)
            template = cue_template(result.a121_time_s, result.cues, result.cue_delay_s)
            ax.plot(result.a121_time_s, a_signal, color=A121_COLOUR, linewidth=1.45, label="A121")
            ax.plot(result.hb_time_s, h_signal, color=HB100_COLOUR, linewidth=1.05, alpha=0.88, label="HB100")
            ax.plot(
                result.a121_time_s,
                template,
                color=TEMPLATE_COLOUR,
                linestyle=":",
                linewidth=1.0,
                label="zadany cykl",
            )
            ax.set_xlim(0.0, 90.0)
            ax.set_ylim(-1.75, 1.75)
            ax.set_title(
                f"{distance} cm, powt. {repeat}  ·  δ={result.cue_delay_s:.3f} s  ·  "
                f"rHB₂={result.metrics['hb100_template_r_block2']:.2f}"
            )
            if column == 0 and row == 1:
                ax.set_ylabel("Składowa oddechowa, skala znormalizowana")
            if row == 2:
                ax.set_xlabel("Czas od początku zapisu [s]")
            _style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.extend(
        [
            Patch(facecolor="#56B4E9", alpha=0.25, label="wdech"),
            Patch(facecolor="#E69F00", alpha=0.22, label="wydech"),
            Patch(facecolor="#777777", alpha=0.28, label="wstrzymanie"),
        ]
    )
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=6, frameon=False)
    figure.suptitle(
        "A121 i HB100 względem komend: polaryzacja ustalona w pierwszym bloku i zachowana w drugim",
        y=0.992,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _distance_scatter(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    columns: tuple[str, ...],
    labels: tuple[str, ...],
    colours: tuple[str, ...],
    markers: tuple[str, ...],
) -> None:
    counts = metrics.groupby("distance_cm")["run"].count().to_dict()
    offsets = {1: -1.6, 2: 1.6}
    for column, label, colour, marker in zip(columns, labels, colours, markers, strict=True):
        for row in metrics.itertuples(index=False):
            ax.scatter(
                float(row.distance_cm) + (offsets.get(int(row.repeat), 0.0) if counts[float(row.distance_cm)] > 1 else 0.0),
                float(getattr(row, column)),
                color=colour,
                marker=marker,
                s=54,
                linewidth=1.4,
                label=label if int(row.run) == int(metrics["run"].min()) else None,
                zorder=3,
            )
    distances = sorted(float(value) for value in metrics["distance_cm"].unique())
    ax.set_xticks(distances, [f"{distance:g}" for distance in distances])
    ax.set_xlabel("Odległość [cm]")
    _style_axis(ax)


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    _distance_scatter(
        axes[0, 0],
        metrics,
        ("a121_template_r_block2", "hb100_template_r_block2"),
        ("A121", "HB100"),
        (A121_COLOUR, HB100_COLOUR),
        ("o", "s"),
    )
    axes[0, 0].axhline(0.0, color="#555555", linewidth=0.9)
    axes[0, 0].set_ylim(-0.65, 1.05)
    axes[0, 0].set_ylabel("Korelacja z zadanym cyklem, blok 2")
    axes[0, 0].set_title("Czy kształt sygnału zachowuje fazę cyklu?")
    axes[0, 0].legend(frameon=False)

    _distance_scatter(
        axes[0, 1],
        metrics,
        ("a121_phase_accuracy_block2", "hb100_phase_accuracy_block2"),
        ("A121", "HB100"),
        (A121_COLOUR, HB100_COLOUR),
        ("o", "s"),
    )
    axes[0, 1].axhline(0.5, color="#555555", linestyle=":", linewidth=1.1)
    axes[0, 1].set_ylim(0.30, 1.03)
    axes[0, 1].set_ylabel("Zgodność kierunku wdech/wydech")
    axes[0, 1].set_title("Polaryzacja ustalona na bloku 1, ocena na bloku 2")

    _distance_scatter(
        axes[1, 0],
        metrics,
        (
            "hb100_fundamental_vs_strongest_harmonic_db_block1",
            "hb100_fundamental_vs_strongest_harmonic_db_block2",
        ),
        ("blok 1", "blok 2"),
        ("#7B3294", "#008837"),
        ("^", "v"),
    )
    axes[1, 0].axhline(0.0, color="#555555", linewidth=0.9)
    axes[1, 0].set_ylabel("0,20 Hz / silniejsza harmoniczna [dB]")
    axes[1, 0].set_title("Wartość ujemna: 0,40 lub 0,60 Hz dominuje nad rytmem")
    axes[1, 0].legend(frameon=False)

    _distance_scatter(
        axes[1, 1],
        metrics,
        ("a121_hold_vs_paced_db", "hb100_hold_vs_paced_db"),
        ("A121", "HB100"),
        (A121_COLOUR, HB100_COLOUR),
        ("o", "s"),
    )
    axes[1, 1].axhline(-3.0, color="#555555", linestyle=":", linewidth=1.1)
    axes[1, 1].set_ylabel("RMS wstrzymanie / oddech [dB]")
    axes[1, 1].set_title("Zanik ruchu potwierdza pochodzenie oddechowe")

    figure.suptitle(f"Porównanie HB100 i A121 w {len(metrics)} równoczesnych nagraniach")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_summary(metrics: pd.DataFrame, interference: dict[str, Any]) -> dict[str, Any]:
    dominant_columns = (
        "hb100_dominant_frequency_hz_block1",
        "hb100_dominant_frequency_hz_block2",
    )
    dominant = metrics.loc[:, dominant_columns].to_numpy(dtype=float).ravel()
    harmonic_dominant = int(np.count_nonzero(~((dominant >= 0.18) & (dominant <= 0.22))))
    return {
        "definitions": {
            "cue_delay_s": "Aggregate command-to-recorded-motion delay estimated from A121 only.",
            "phase_accuracy": (
                "Balanced sample-wise agreement between derivative sign and inhale/exhale cue, "
                "excluding 0.30 s around cue transitions. Polarity is calibrated on block 1."
            ),
            "fundamental_vs_strongest_harmonic_db": (
                "20 log10 of fitted 0.20 Hz amplitude divided by the larger fitted 0.40/0.60 Hz amplitude."
            ),
        },
        "overall": {
            "recordings": int(len(metrics)),
            "cue_delay_median_s": float(metrics["cue_delay_s"].median()),
            "cue_delay_min_s": float(metrics["cue_delay_s"].min()),
            "cue_delay_max_s": float(metrics["cue_delay_s"].max()),
            "a121_template_r_block2_min": float(metrics["a121_template_r_block2"].min()),
            "a121_template_r_block2_max": float(metrics["a121_template_r_block2"].max()),
            "hb100_template_r_block2_min": float(metrics["hb100_template_r_block2"].min()),
            "hb100_template_r_block2_max": float(metrics["hb100_template_r_block2"].max()),
            "a121_phase_accuracy_block2_min": float(metrics["a121_phase_accuracy_block2"].min()),
            "a121_phase_accuracy_block2_max": float(metrics["a121_phase_accuracy_block2"].max()),
            "hb100_phase_accuracy_block2_min": float(metrics["hb100_phase_accuracy_block2"].min()),
            "hb100_phase_accuracy_block2_max": float(metrics["hb100_phase_accuracy_block2"].max()),
            "hb100_harmonic_dominant_blocks": harmonic_dominant,
            "paced_blocks": int(len(dominant)),
            "hb100_hold_drop_db_min": float(metrics["hb100_hold_vs_paced_db"].min()),
            "hb100_hold_drop_db_max": float(metrics["hb100_hold_vs_paced_db"].max()),
            "hb100_saturated_recordings": int(
                np.count_nonzero(metrics["hb100_saturation_percent"].to_numpy(dtype=float) > 0.1)
            ),
        },
        "interference": interference,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def print_summary(metrics: pd.DataFrame, summary: dict[str, Any]) -> None:
    columns = [
        "run",
        "distance_cm",
        "repeat",
        "cue_delay_s",
        "a121_template_r_block2",
        "hb100_template_r_block2",
        "a121_phase_accuracy_block2",
        "hb100_phase_accuracy_block2",
        "hb100_dominant_frequency_hz_block1",
        "hb100_dominant_frequency_hz_block2",
        "hb100_hold_vs_paced_db",
        "hb100_saturation_percent",
    ]
    print(metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nOverall:")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print("\nInterference screen:")
    print(json.dumps(summary["interference"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="Session directory; default: latest guided session.")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Figure output directory (default: {DEFAULT_FIGURE_DIR}).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV/JSON analysis files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = (args.session or _latest_session()).resolve()
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    accepted = [item for item in manifest.get("measurements", []) if item.get("status") == "accepted"]
    range_measurements = sorted(
        [item for item in accepted if item["step"].get("kind") == "range"],
        key=lambda item: int(item["step"]["number"]),
    )
    interference_measurements = [
        item for item in accepted if item["step"].get("kind") == "interference"
    ]
    discovered = discover_unregistered_range_measurements(session, range_measurements)
    range_measurements.extend(discovered)
    range_measurements.sort(key=lambda item: int(item["step"]["number"]))
    if len(range_measurements) < 6:
        raise ValueError(f"Expected at least six range recordings, found {len(range_measurements)}")
    if len(interference_measurements) != 3:
        raise ValueError(f"Expected three interference recordings, found {len(interference_measurements)}")

    results = [analyse_recording(measurement) for measurement in range_measurements]
    metrics = pd.DataFrame([result.metrics for result in results]).sort_values("run")
    interference = analyse_interference(interference_measurements)
    summary = build_summary(metrics, interference)

    metrics_path = session / "analysis_hb100_a121_metrics.csv"
    summary_path = session / "analysis_hb100_a121_summary.json"
    metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.no_plots:
        figure_dir = args.figure_dir.resolve()
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_raw_overlays(results, figure_dir / "hb100_a121_surowe.png")
        plot_respiratory_overlays(results, figure_dir / "hb100_a121_oddech.png")
        plot_metrics(metrics, figure_dir / "hb100_a121_metryki.png")

    print_summary(metrics, summary)
    print(f"\nMetrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    if not args.no_plots:
        print(f"Figures: {args.figure_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
