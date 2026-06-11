from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch

from .a121 import parse_json_array
from .a121_vitals import _times_seconds_from_ms, analyze_a121_vitals, sample_rate_from_ms

BreathOrientation = Literal["peak-inhale", "trough-inhale"]


@dataclass(frozen=True)
class BreathAnnotation:
    timestamp_ms: float
    elapsed_s: float
    frame: int
    event: str
    key: str = "auto"


@dataclass(frozen=True)
class AutoBreathResult:
    annotations_path: Path
    annotations: list[BreathAnnotation]
    rows: int
    duration_s: float
    sample_rate_hz: float
    target_distance_m: float
    resp_bpm: float


def default_breath_annotations_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_breath_annotations.csv")


def _load_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack([parse_json_array(value) for value in series])


def _psd_resp_bpm(resp_signal: np.ndarray, fs: float) -> float:
    if fs <= 0 or len(resp_signal) < max(32, int(fs * 12.0)):
        return 0.0
    freqs_hz, power = welch(resp_signal, fs=fs, nperseg=min(len(resp_signal), max(64, int(fs * 40.0))))
    bpm_axis = freqs_hz * 60.0
    valid = (bpm_axis >= 6.0) & (bpm_axis <= 30.0)
    if not np.any(valid):
        return 0.0
    idx = int(np.argmax(power[valid]))
    return float(bpm_axis[valid][idx])


def respiration_signal_from_a121_csv(
    csv_path: Path,
    *,
    cluster_half_width_m: float = 0.20,
    strong_bin_fraction: float = 0.25,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, float, float]:
    """Return df, time_s, weighted respiration signal, fs, target_m, resp_bpm."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")
    required = {"Timestamp_ms", "Distances_m", "Real", "Imag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required A121 columns: {', '.join(sorted(missing))}")

    timestamps_ms = df["Timestamp_ms"].to_numpy(dtype=float)
    fs = sample_rate_from_ms(timestamps_ms)
    if fs <= 1.2:
        raise ValueError(f"Could not infer a usable A121 sample rate from {csv_path}")
    time_s = _times_seconds_from_ms(timestamps_ms, fs)

    distances = parse_json_array(df["Distances_m"].iloc[0])
    real = _load_matrix(df["Real"])
    imag = _load_matrix(df["Imag"])
    m = min(real.shape[1], imag.shape[1], len(distances))
    if m < 1:
        raise ValueError(f"{csv_path} does not contain usable A121 range bins")
    distances = distances[:m]
    z = real[:, :m] + 1j * imag[:, :m]
    amplitude = np.abs(z)
    median_amp = np.median(amplitude, axis=0)

    analysis = analyze_a121_vitals(df)
    if np.isfinite(analysis.target_distance_m) and analysis.target_distance_m > 0:
        target_m = float(analysis.target_distance_m)
    else:
        target_m = float(distances[int(np.argmax(median_amp))])

    cluster_mask = (distances >= max(float(distances[0]), target_m - cluster_half_width_m)) & (
        distances <= min(float(distances[-1]), target_m + cluster_half_width_m)
    )
    cluster_idx = np.flatnonzero(cluster_mask if np.any(cluster_mask) else np.ones_like(distances, dtype=bool))
    local_peak = float(np.max(median_amp[cluster_idx])) if len(cluster_idx) else 0.0
    strong = median_amp[cluster_idx] >= max(local_peak * float(strong_bin_fraction), 1.0)
    if np.any(strong):
        cluster_idx = cluster_idx[strong]
    if len(cluster_idx) == 0:
        cluster_idx = np.asarray([int(np.argmax(median_amp))], dtype=int)
    weights = np.maximum(median_amp[cluster_idx], 0.0)
    weights = weights / (float(np.sum(weights)) + 1e-12)

    phase = detrend(np.unwrap(np.angle(z), axis=0), axis=0, type="linear")
    high_hz = min(0.50, fs * 0.45)
    low_hz = min(0.10, high_hz * 0.5)
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError(f"Sample rate {fs:.3f} Hz is too low for respiration classification")
    resp_matrix = sosfiltfilt(butter(2, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos"), phase, axis=0)
    resp_signal = resp_matrix[:, cluster_idx] @ weights

    resp_bpm = float(analysis.resp_bpm) if np.isfinite(analysis.resp_bpm) and analysis.resp_bpm > 0 else 0.0
    if resp_bpm <= 0:
        resp_bpm = _psd_resp_bpm(resp_signal, fs)
    return df, time_s, resp_signal, float(fs), target_m, resp_bpm


def _nearest_annotation(
    time_s: np.ndarray,
    timestamps_ms: np.ndarray,
    frames: np.ndarray,
    t_s: float,
    event: str,
) -> BreathAnnotation:
    idx = int(np.clip(np.searchsorted(time_s, t_s), 0, len(time_s) - 1))
    if idx > 0 and abs(float(time_s[idx - 1]) - t_s) < abs(float(time_s[idx]) - t_s):
        idx -= 1
    return BreathAnnotation(
        timestamp_ms=float(timestamps_ms[idx]),
        elapsed_s=float(time_s[idx]),
        frame=int(frames[idx]),
        event=event,
        key="auto",
    )


def _append_unique_event(events: list[BreathAnnotation], annotation: BreathAnnotation) -> None:
    if events and events[-1].event == annotation.event and abs(events[-1].elapsed_s - annotation.elapsed_s) < 1e-6:
        return
    events.append(annotation)


def auto_classify_breaths(
    csv_path: Path,
    *,
    orientation: BreathOrientation = "peak-inhale",
    prominence: float = 0.30,
    cluster_half_width_m: float = 0.20,
    strong_bin_fraction: float = 0.25,
) -> tuple[list[BreathAnnotation], int, float, float, float, float]:
    """Classify breath phases from an A121 recording.

    The default orientation matches the current A121 setup/manual labels: the respiration trace
    falls from a local maximum to a local minimum during inhale, then rises during exhale.
    """
    if orientation not in {"peak-inhale", "trough-inhale"}:
        raise ValueError("orientation must be 'peak-inhale' or 'trough-inhale'")

    df, time_s, resp_signal, fs, target_m, resp_bpm = respiration_signal_from_a121_csv(
        csv_path,
        cluster_half_width_m=cluster_half_width_m,
        strong_bin_fraction=strong_bin_fraction,
    )
    timestamps_ms = df["Timestamp_ms"].to_numpy(dtype=float)
    frames = (
        pd.to_numeric(df["Frame"], errors="coerce").fillna(pd.Series(np.arange(len(df)), index=df.index)).to_numpy(dtype=int)
        if "Frame" in df
        else np.arange(len(df), dtype=int)
    )
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0
    if resp_bpm <= 0:
        resp_bpm = 12.0
    period_s = float(np.clip(60.0 / max(resp_bpm, 1e-9), 2.0, 12.0))

    norm = (resp_signal - float(np.median(resp_signal))) / (float(np.std(resp_signal)) + 1e-12)
    min_distance = max(1, int(round(fs * period_s * 0.45)))
    peaks, _ = find_peaks(norm, distance=min_distance, prominence=prominence)
    troughs, _ = find_peaks(-norm, distance=min_distance, prominence=prominence)
    if len(peaks) < 1 or len(troughs) < 1:
        # Retry with a lower prominence for very clean/low-amplitude recordings.
        peaks, _ = find_peaks(norm, distance=min_distance, prominence=max(0.10, prominence * 0.5))
        troughs, _ = find_peaks(-norm, distance=min_distance, prominence=max(0.10, prominence * 0.5))

    if orientation == "peak-inhale":
        inhale_starts = np.asarray(peaks, dtype=int)
        inhale_ends = np.asarray(troughs, dtype=int)
    else:
        inhale_starts = np.asarray(troughs, dtype=int)
        inhale_ends = np.asarray(peaks, dtype=int)

    annotations: list[BreathAnnotation] = []
    for start_idx in inhale_starts:
        next_inhale_end_candidates = inhale_ends[inhale_ends > start_idx]
        if len(next_inhale_end_candidates) == 0:
            continue
        end_idx = int(next_inhale_end_candidates[0])
        next_start_candidates = inhale_starts[inhale_starts > end_idx]
        if len(next_start_candidates) == 0:
            continue
        next_start_idx = int(next_start_candidates[0])

        # Reject extremely short or long segments caused by noisy extrema.
        inhale_s = float(time_s[end_idx] - time_s[start_idx])
        exhale_s = float(time_s[next_start_idx] - time_s[end_idx])
        if not (0.15 * period_s <= inhale_s <= 0.85 * period_s and 0.15 * period_s <= exhale_s <= 0.95 * period_s):
            continue

        _append_unique_event(annotations, _nearest_annotation(time_s, timestamps_ms, frames, float(time_s[start_idx]), "inhale_start"))
        _append_unique_event(annotations, _nearest_annotation(time_s, timestamps_ms, frames, float(time_s[end_idx]), "inhale_end"))
        _append_unique_event(annotations, _nearest_annotation(time_s, timestamps_ms, frames, float(time_s[end_idx]), "exhale_start"))
        _append_unique_event(annotations, _nearest_annotation(time_s, timestamps_ms, frames, float(time_s[next_start_idx]), "exhale_end"))

    # Stable sort by time only; preserve phase-transition order for identical timestamps
    # (inhale_end before exhale_start at troughs, exhale_end before inhale_start at peaks).
    annotations.sort(key=lambda ann: ann.elapsed_s)
    return annotations, len(df), duration_s, fs, target_m, resp_bpm


def write_breath_annotations(path: Path, annotations: list[BreathAnnotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp_ms", "Elapsed_s", "Frame", "Event", "Key"])
        for ann in annotations:
            writer.writerow([f"{ann.timestamp_ms:.3f}", f"{ann.elapsed_s:.6f}", ann.frame, ann.event, ann.key])


def generate_a121_breath_annotations(
    csv_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
    orientation: BreathOrientation = "peak-inhale",
    prominence: float = 0.30,
    cluster_half_width_m: float = 0.20,
    strong_bin_fraction: float = 0.25,
) -> AutoBreathResult:
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = default_breath_annotations_path(csv_path)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass overwrite=True or choose --output")

    annotations, rows, duration_s, fs, target_m, resp_bpm = auto_classify_breaths(
        csv_path,
        orientation=orientation,
        prominence=prominence,
        cluster_half_width_m=cluster_half_width_m,
        strong_bin_fraction=strong_bin_fraction,
    )
    if not annotations:
        raise ValueError("No breath annotations were detected")
    write_breath_annotations(output_path, annotations)
    return AutoBreathResult(
        annotations_path=output_path,
        annotations=annotations,
        rows=rows,
        duration_s=duration_s,
        sample_rate_hz=fs,
        target_distance_m=target_m,
        resp_bpm=resp_bpm,
    )
