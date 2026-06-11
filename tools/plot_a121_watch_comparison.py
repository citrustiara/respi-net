#!/usr/bin/env python
"""Generate A121 watch-comparison diagnostics plots.

Usage example:

    uv run python tools/plot_a121_watch_comparison.py \
        data/raw/a121/a121_test_80s_90hr_2026-06-06_16-27-36.csv \
        --watch-start-bpm 90 --watch-end-bpm 75

The plot is intentionally diagnostic: it overlays the repository's batch A121 analyzer with a
weighted range-bin spectral view and optional watch reference trend/band.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch

from respi_net.a121 import parse_json_array
from respi_net.a121_vitals import _times_seconds_from_ms, analyze_a121_vitals, sample_rate_from_ms


@dataclass(frozen=True)
class PsdPeak:
    bpm: float
    confidence: float
    bpm_axis: np.ndarray
    power: np.ndarray


@dataclass(frozen=True)
class PlotSummary:
    output_path: Path
    rows: int
    duration_s: float
    sample_rate_hz: float
    target_distance_m: float
    avg_peak_amplitude: float
    analyzer_hr_bpm: float
    analyzer_hr_confidence: float
    analyzer_rr_bpm: float
    analyzer_rr_confidence: float
    weighted_hr_bpm: float
    weighted_hr_confidence: float
    weighted_rr_bpm: float
    weighted_rr_confidence: float
    chunk_hr: list[tuple[float, float, float]]
    watch_band_chunk_hr: list[tuple[float, float, float]]
    analyzer_hr_trend: list[tuple[float, float, float]]
    analyzer_rr_trend: list[tuple[float, float, float]]
    breath_annotation_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an A121 HR/RR plot compared with a watch reference.")
    parser.add_argument("csv_path", type=Path, help="A121 Sparse IQ CSV, e.g. data/raw/a121/a121_test_80s_90hr_*.csv")
    parser.add_argument("--output", type=Path, help="Output PNG path. Defaults to data/plots/<csv-stem>_heart_trend.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "plots", help="Default output directory")
    parser.add_argument("--watch-start-bpm", type=float, help="Watch HR near the start of the recording")
    parser.add_argument("--watch-end-bpm", type=float, help="Watch HR near the end of the recording")
    parser.add_argument(
        "--watch-band-bpm",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Watch comparison band. Defaults to min/max of --watch-start-bpm and --watch-end-bpm when available.",
    )
    parser.add_argument("--chunk-s", type=float, default=20.0, help="Seconds per chunk spectrum")
    parser.add_argument("--analyzer-window-s", type=float, default=30.0, help="Sliding batch analyzer window in seconds")
    parser.add_argument("--analyzer-step-s", type=float, default=5.0, help="Sliding analyzer step in seconds")
    parser.add_argument("--cluster-half-width-m", type=float, default=0.20, help="Range half-width around the analyzer target for weighted diagnostics")
    parser.add_argument(
        "--strong-bin-fraction",
        type=float,
        default=0.25,
        help="Keep diagnostic range bins with median amplitude at least this fraction of the local peak",
    )
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument(
        "--breath-annotations",
        type=Path,
        help="Optional *_breath_annotations.csv sidecar. Defaults to <csv-stem>_breath_annotations.csv when present.",
    )
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def _load_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack([parse_json_array(value) for value in series])


def _auto_breath_annotations_path(csv_path: Path) -> Path | None:
    candidate = csv_path.with_name(f"{csv_path.stem}_breath_annotations.csv")
    return candidate if candidate.exists() else None


def _load_breath_annotations(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["Elapsed_s", "Event", "Key"])
    ann = pd.read_csv(path)
    if "Elapsed_s" not in ann.columns or "Event" not in ann.columns:
        raise ValueError(f"{path} is missing required annotation columns: Elapsed_s, Event")
    ann = ann.copy()
    ann["Elapsed_s"] = pd.to_numeric(ann["Elapsed_s"], errors="coerce")
    ann["Event"] = ann["Event"].astype(str)
    if "Key" not in ann.columns:
        ann["Key"] = ""
    return ann.dropna(subset=["Elapsed_s"]).sort_values("Elapsed_s")


def _infer_watch_start_from_name(path: Path) -> float | None:
    match = re.search(r"(?:^|_)(\d+(?:\.\d+)?)hr(?:_|$)", path.stem, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _watch_values(args: argparse.Namespace) -> tuple[float | None, float | None, tuple[float, float] | None]:
    start = args.watch_start_bpm
    end = args.watch_end_bpm
    if start is None:
        start = _infer_watch_start_from_name(args.csv_path)
    if end is None and start is not None:
        end = start

    band: tuple[float, float] | None = None
    if args.watch_band_bpm is not None:
        lo, hi = sorted(float(v) for v in args.watch_band_bpm)
        band = (lo, hi)
    elif start is not None and end is not None:
        lo, hi = sorted((float(start), float(end)))
        if abs(hi - lo) < 1e-9:
            lo, hi = lo - 5.0, hi + 5.0
        band = (lo, hi)
    return start, end, band


def _psd_peak(
    signal: np.ndarray,
    fs: float,
    band_bpm: tuple[float, float],
    *,
    min_seconds: float = 8.0,
    max_nperseg_s: float | None = None,
) -> PsdPeak:
    x = np.asarray(signal, dtype=float)
    if fs <= 0 or len(x) < max(16, int(round(fs * min_seconds))):
        return PsdPeak(0.0, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float))
    duration_s = len(x) / fs
    if max_nperseg_s is None:
        nperseg = len(x)
    else:
        nperseg = min(len(x), max(64, int(round(fs * min(max_nperseg_s, duration_s)))))
    freqs_hz, power = welch(x, fs=fs, nperseg=nperseg, scaling="spectrum")
    bpm_axis = freqs_hz * 60.0
    valid = (bpm_axis >= band_bpm[0]) & (bpm_axis <= band_bpm[1])
    if not np.any(valid):
        return PsdPeak(0.0, 0.0, bpm_axis, power)
    band_power = power[valid]
    band_bpm_axis = bpm_axis[valid]
    idx = int(np.argmax(band_power))
    confidence = float(band_power[idx] / (float(np.median(band_power)) + 1e-18))
    return PsdPeak(float(band_bpm_axis[idx]), confidence, bpm_axis, power)


def _nearest_trend_bpm(t_s: float, trend: list[tuple[float, float, float]], *, max_age_s: float, min_confidence: float) -> float | None:
    valid = [(abs(center_s - t_s), bpm) for center_s, bpm, confidence in trend if bpm > 0 and confidence >= min_confidence]
    if not valid:
        return None
    age_s, bpm = min(valid, key=lambda item: item[0])
    return float(bpm) if age_s <= max_age_s and 45.0 <= bpm <= 130.0 else None


def _clean_heart_peaks(
    time_s: np.ndarray,
    heart_signal: np.ndarray,
    raw_peaks: np.ndarray,
    prominences: np.ndarray,
    *,
    fs: float,
    analyzer_hr: list[tuple[float, float, float]],
    watch_band_chunk_hr: list[tuple[float, float, float]],
    chunk_hr: list[tuple[float, float, float]],
    weighted_hr_bpm: float,
    chunk_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reject obvious extra visual peaks and insert markers for likely missed beats.

    The waveform peak picker is only a visual aid.  Use the more stable spectral/batch trend as a
    local period prior, then mark intervals that are far too short as extras and far too long as
    likely missed peaks.
    """
    if len(raw_peaks) == 0:
        empty_i = np.asarray([], dtype=int)
        empty_f = np.asarray([], dtype=float)
        return empty_i, empty_i, empty_f, empty_f, empty_f

    def expected_bpm(t_s: float) -> float:
        bpm = _nearest_trend_bpm(t_s, analyzer_hr, max_age_s=max(10.0, chunk_s), min_confidence=5.0)
        if bpm is not None:
            return bpm
        bpm = _nearest_trend_bpm(t_s, watch_band_chunk_hr, max_age_s=max(10.0, chunk_s), min_confidence=0.5)
        if bpm is not None:
            return bpm
        bpm = _nearest_trend_bpm(t_s, chunk_hr, max_age_s=max(10.0, chunk_s), min_confidence=8.0)
        if bpm is not None:
            return bpm
        return float(weighted_hr_bpm) if 45.0 <= weighted_hr_bpm <= 130.0 else 75.0

    accepted: list[int] = []
    accepted_prom: list[float] = []
    rejected: list[int] = []
    missed_times: list[float] = []

    for peak, prom in zip(raw_peaks, prominences):
        peak = int(peak)
        prom = float(prom)
        if not accepted:
            accepted.append(peak)
            accepted_prom.append(prom)
            continue
        last = accepted[-1]
        mid_t = 0.5 * (float(time_s[last]) + float(time_s[peak]))
        period_s = 60.0 / max(expected_bpm(mid_t), 1e-9)
        dt_s = float(time_s[peak] - time_s[last])
        if dt_s < 0.52 * period_s:
            # Two peaks too close together: keep the more prominent one if replacement does not
            # create another too-short interval with the previous accepted peak.
            can_replace = prom > accepted_prom[-1] * 1.35
            if can_replace and len(accepted) >= 2:
                prev = accepted[-2]
                prev_dt_s = float(time_s[peak] - time_s[prev])
                prev_period_s = 60.0 / max(expected_bpm(0.5 * (float(time_s[peak]) + float(time_s[prev]))), 1e-9)
                can_replace = prev_dt_s >= 0.52 * prev_period_s
            if can_replace:
                rejected.append(last)
                accepted[-1] = peak
                accepted_prom[-1] = prom
            else:
                rejected.append(peak)
            continue
        if dt_s > 1.55 * period_s:
            missed_count = int(round(dt_s / period_s)) - 1
            missed_count = int(np.clip(missed_count, 1, 3))
            for k in range(1, missed_count + 1):
                missed_t = float(time_s[last]) + k * dt_s / (missed_count + 1)
                if float(time_s[last]) + 0.45 * period_s < missed_t < float(time_s[peak]) - 0.45 * period_s:
                    missed_times.append(missed_t)
        accepted.append(peak)
        accepted_prom.append(prom)

    accepted_arr = np.asarray(accepted, dtype=int)
    rejected_arr = np.asarray(sorted(set(rejected)), dtype=int)
    missed_arr = np.asarray(missed_times, dtype=float)
    sequence_times = np.sort(np.concatenate([time_s[accepted_arr], missed_arr]))
    if len(sequence_times) >= 2:
        intervals = np.diff(sequence_times)
        beat_bpms = 60.0 / np.maximum(intervals, 1e-9)
        valid = (beat_bpms >= 45.0) & (beat_bpms <= 130.0)
        beat_times = sequence_times[:-1][valid]
        beat_bpms = beat_bpms[valid]
    else:
        beat_times = np.asarray([], dtype=float)
        beat_bpms = np.asarray([], dtype=float)
    return accepted_arr, rejected_arr, missed_arr, beat_times, beat_bpms


def create_watch_comparison_plot(
    csv_path: Path,
    *,
    output_path: Path,
    watch_start_bpm: float | None,
    watch_end_bpm: float | None,
    watch_band_bpm: tuple[float, float] | None,
    chunk_s: float = 20.0,
    analyzer_window_s: float = 30.0,
    analyzer_step_s: float = 5.0,
    cluster_half_width_m: float = 0.20,
    strong_bin_fraction: float = 0.25,
    title: str | None = None,
    breath_annotations_path: Path | None = None,
    dpi: int = 140,
) -> PlotSummary:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")
    required = {"Timestamp_ms", "Distances_m", "Real", "Imag", "PeakDistance_m"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required A121 columns: {', '.join(sorted(missing))}")

    timestamps_ms = df["Timestamp_ms"].to_numpy(dtype=float)
    fs = sample_rate_from_ms(timestamps_ms)
    time_s = _times_seconds_from_ms(timestamps_ms, fs)
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0
    annotations = _load_breath_annotations(breath_annotations_path)
    distances = parse_json_array(df["Distances_m"].iloc[0])
    real = _load_matrix(df["Real"])
    imag = _load_matrix(df["Imag"])
    m = min(real.shape[1], imag.shape[1], len(distances))
    distances = distances[:m]
    z = real[:, :m] + 1j * imag[:, :m]
    amplitude = np.abs(z)
    median_amp = np.median(amplitude, axis=0)

    avg_peak_amplitude = float(np.mean(pd.to_numeric(df["PeakAmplitude"], errors="coerce"))) if "PeakAmplitude" in df else 0.0

    analysis = analyze_a121_vitals(df)
    if np.isfinite(analysis.target_distance_m) and analysis.target_distance_m > 0:
        target_m = float(analysis.target_distance_m)
    else:
        target_m = float(distances[int(np.argmax(median_amp))]) if len(distances) else 0.0

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
    resp_matrix = sosfiltfilt(butter(2, [0.10, 0.50], btype="bandpass", fs=fs, output="sos"), phase, axis=0)
    heart_matrix = sosfiltfilt(butter(3, [0.70, 2.00], btype="bandpass", fs=fs, output="sos"), phase, axis=0)
    resp_signal = resp_matrix[:, cluster_idx] @ weights
    heart_signal = heart_matrix[:, cluster_idx] @ weights

    weighted_rr = _psd_peak(resp_signal, fs, (6.0, 30.0), min_seconds=min(20.0, max(8.0, duration_s * 0.35)))
    weighted_hr = _psd_peak(heart_signal, fs, (42.0, 120.0), min_seconds=min(12.0, max(8.0, duration_s * 0.25)))

    chunk_spectra: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []
    chunk_hr: list[tuple[float, float, float]] = []
    watch_band_chunk_hr: list[tuple[float, float, float]] = []
    if chunk_s > 0:
        starts = np.arange(0.0, max(duration_s - chunk_s * 0.5, 0.0) + 1e-9, chunk_s)
        for start_s in starts:
            end_s = min(start_s + chunk_s, float(time_s[-1]))
            mask = (time_s >= start_s) & (time_s < end_s)
            if int(np.sum(mask)) < int(fs * min(12.0, chunk_s * 0.6)):
                continue
            center_s = (start_s + end_s) * 0.5
            peak = _psd_peak(heart_signal[mask], fs, (42.0, 120.0), min_seconds=min(10.0, max(6.0, chunk_s * 0.45)))
            chunk_spectra.append((f"{int(round(start_s))}-{int(round(end_s))}s", peak.bpm_axis, peak.power, peak.bpm, peak.confidence))
            chunk_hr.append((center_s, peak.bpm, peak.confidence))
            if watch_band_bpm is not None:
                watch_peak = _psd_peak(
                    heart_signal[mask],
                    fs,
                    watch_band_bpm,
                    min_seconds=min(10.0, max(6.0, chunk_s * 0.45)),
                )
                watch_band_chunk_hr.append((center_s, watch_peak.bpm, watch_peak.confidence))

    analyzer_hr: list[tuple[float, float, float]] = []
    analyzer_rr: list[tuple[float, float, float]] = []
    if analyzer_window_s > 0 and analyzer_step_s > 0:
        for end_s in np.arange(analyzer_window_s, duration_s + analyzer_step_s * 0.25, analyzer_step_s):
            mask = (time_s >= end_s - analyzer_window_s) & (time_s <= end_s)
            if int(np.sum(mask)) < int(fs * max(10.0, analyzer_window_s * 0.75)):
                continue
            center_s = end_s - analyzer_window_s * 0.5
            window_analysis = analyze_a121_vitals(df.loc[mask].copy(), heart_window_s=analyzer_window_s)
            if window_analysis.heart_bpm > 0:
                analyzer_hr.append((center_s, float(window_analysis.heart_bpm), float(window_analysis.heart_confidence)))
            if window_analysis.resp_bpm > 0:
                analyzer_rr.append((center_s, float(window_analysis.resp_bpm), float(window_analysis.resp_confidence)))

    heart_norm = (heart_signal - float(np.median(heart_signal))) / (float(np.std(heart_signal)) + 1e-12)
    raw_peaks, peak_props = find_peaks(heart_norm, distance=max(1, int(round(fs * 0.20))), prominence=0.22)
    peak_prominences = np.asarray(peak_props.get("prominences", np.ones(len(raw_peaks))), dtype=float)
    peaks, rejected_peaks, missed_peak_times, beat_times, beat_bpms = _clean_heart_peaks(
        time_s,
        heart_signal,
        raw_peaks,
        peak_prominences,
        fs=fs,
        analyzer_hr=analyzer_hr,
        watch_band_chunk_hr=watch_band_chunk_hr,
        chunk_hr=chunk_hr,
        weighted_hr_bpm=weighted_hr.bpm,
        chunk_s=chunk_s,
    )

    if title is None:
        watch_text = ""
        if watch_start_bpm is not None and watch_end_bpm is not None:
            watch_text = f": watch {watch_start_bpm:.0f}→{watch_end_bpm:.0f} BPM"
        elif watch_start_bpm is not None:
            watch_text = f": watch ~{watch_start_bpm:.0f} BPM"
        title = f"A121 {duration_s:.0f} s recording{watch_text}"

    fig = plt.figure(figsize=(16, 15.3), dpi=dpi)
    grid = fig.add_gridspec(5, 1, height_ratios=[1.35, 0.9, 1.05, 1.05, 1.0], hspace=0.35)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.992)
    summary_text = (
        f"Full window\n"
        f"HR {analysis.heart_bpm:.1f} BPM  conf {analysis.heart_confidence:.1f}\n"
        f"RR {analysis.resp_bpm:.1f} BPM  conf {analysis.resp_confidence:.1f}\n"
        f"Weighted HR {weighted_hr.bpm:.1f} BPM\n"
        f"Avg peak amp {avg_peak_amplitude:.0f}"
    )
    fig.text(
        0.985,
        0.965,
        summary_text,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.86},
    )

    ax0 = fig.add_subplot(grid[0])
    mesh = ax0.pcolormesh(time_s, distances, amplitude.T, shading="auto", cmap="viridis")
    ax0.plot(time_s, df["PeakDistance_m"].to_numpy(dtype=float), color="white", lw=0.8, label="Peak distance")
    ax0.axhline(target_m, color="#ff4d4d", ls="--", lw=1.3, label=f"Target {target_m:.3f} m")
    ax0.axhspan(float(distances[cluster_idx[0]]), float(distances[cluster_idx[-1]]), color="white", alpha=0.08, label="Weighted range cluster")
    ax0.set_title("Range amplitude context")
    ax0.set_ylabel("Distance (m)")
    ax0.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax0.grid(True, color="white", alpha=0.25)
    ax0.legend(loc="upper right")
    fig.colorbar(mesh, ax=ax0, pad=0.015, label="Amplitude")

    ax1 = fig.add_subplot(grid[1], sharex=ax0)
    ax1.plot(time_s, resp_signal, color="#159a22", lw=1.3)
    ax1.axhline(0, color="black", lw=0.7, alpha=0.6)
    if not annotations.empty:
        phase_styles = {
            "inhale": {"color": "#3b82f6", "alpha": 0.13, "label": "annotated inhale"},
            "exhale": {"color": "#f97316", "alpha": 0.13, "label": "annotated exhale"},
        }
        used_labels: set[str] = set()
        for phase, style in phase_styles.items():
            open_start: float | None = None
            phase_rows = annotations[annotations["Event"].str.startswith(phase)]
            for _, ann in phase_rows.iterrows():
                event = str(ann["Event"])
                t_ann = float(ann["Elapsed_s"])
                if event.endswith("_start"):
                    open_start = t_ann
                    label = f"{phase} start" if f"{phase} start" not in used_labels else None
                    used_labels.add(f"{phase} start")
                    ax1.axvline(t_ann, color=style["color"], lw=0.9, ls="--", alpha=0.85, label=label)
                elif event.endswith("_end"):
                    if open_start is not None and t_ann > open_start:
                        label = style["label"] if style["label"] not in used_labels else None
                        used_labels.add(style["label"])
                        ax1.axvspan(open_start, t_ann, color=style["color"], alpha=style["alpha"], label=label)
                    label = f"{phase} end" if f"{phase} end" not in used_labels else None
                    used_labels.add(f"{phase} end")
                    ax1.axvline(t_ann, color=style["color"], lw=0.9, ls=":", alpha=0.9, label=label)
                    open_start = None
        ax1.legend(loc="upper right", ncol=2, fontsize=8)
    ax1.set_title(f"Respiration diagnostic: weighted PSD {weighted_rr.bpm:.1f} BPM; batch analyzer {analysis.resp_bpm:.1f} BPM")
    ax1.set_ylabel("Resp signal")
    ax1.grid(True, alpha=0.65)

    ax2 = fig.add_subplot(grid[2], sharex=ax0)
    ax2.plot(time_s, heart_signal, color="#1577c7", lw=1.0, label="Weighted A121 heart-band signal")
    if len(peaks):
        ax2.scatter(time_s[peaks], heart_signal[peaks], s=14, color="#16a34a", zorder=4, label=f"Accepted visual peaks: {len(peaks)}")
    if len(rejected_peaks):
        ax2.scatter(time_s[rejected_peaks], heart_signal[rejected_peaks], s=22, marker="x", color="#dc2626", zorder=5, label=f"Rejected extras: {len(rejected_peaks)}")
    if len(missed_peak_times):
        ax2.scatter(missed_peak_times, np.zeros_like(missed_peak_times), s=28, marker="D", facecolors="none", edgecolors="#f97316", zorder=5, label=f"Likely missed beats: {len(missed_peak_times)}")
    ax2.axhline(0, color="black", lw=0.7, alpha=0.55)
    ax2.set_title("Heart-band signal with period-guided visual peak cleanup")
    ax2.set_ylabel("Heart signal")
    ax2.grid(True, alpha=0.65)
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(grid[3], sharex=ax0)
    if watch_band_bpm is not None:
        ax3.axhspan(watch_band_bpm[0], watch_band_bpm[1], color="#2ca02c", alpha=0.15, label=f"watch band {watch_band_bpm[0]:.0f}-{watch_band_bpm[1]:.0f}")
    if watch_start_bpm is not None and watch_end_bpm is not None:
        ax3.plot([float(time_s[0]), float(time_s[-1])], [watch_start_bpm, watch_end_bpm], color="#444444", lw=1.0, ls=":", label=f"watch trend {watch_start_bpm:.0f}→{watch_end_bpm:.0f}")
    ax3.axhline(float(analysis.heart_bpm), color="black", lw=1.1, ls="--", label=f"batch full {analysis.heart_bpm:.1f}")
    if chunk_hr:
        chunk_arr = np.asarray(chunk_hr, dtype=float)
        ax3.plot(chunk_arr[:, 0], chunk_arr[:, 1], "o-", color="#d62728", lw=1.6, label=f"{chunk_s:.0f} s weighted spectral peak")
    if watch_band_chunk_hr:
        watch_arr = np.asarray(watch_band_chunk_hr, dtype=float)
        ax3.plot(watch_arr[:, 0], watch_arr[:, 1], "s--", color="#9467bd", lw=1.2, label=f"{chunk_s:.0f} s peak inside watch band")
    if analyzer_hr:
        analyzer_arr = np.asarray(analyzer_hr, dtype=float)
        ax3.plot(analyzer_arr[:, 0], analyzer_arr[:, 1], "^-", color="#1f77b4", lw=1.3, label=f"{analyzer_window_s:.0f} s batch analyzer")
    if len(beat_times):
        ax3.scatter(beat_times, beat_bpms, s=9, color="#ff7f0e", alpha=0.48, label="cleaned beat-interval HR")
    ax3.set_ylim(45, 120)
    ax3.set_title("HR/RR trend diagnostics")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("HR (BPM)")
    ax3.grid(True, alpha=0.65)

    ax3r = ax3.twinx()
    ax3r.axhline(float(analysis.resp_bpm), color="#15803d", lw=1.0, ls="--", alpha=0.8, label=f"batch full RR {analysis.resp_bpm:.1f}")
    ax3r.axhline(float(weighted_rr.bpm), color="#22c55e", lw=0.9, ls=":", alpha=0.8, label=f"weighted RR {weighted_rr.bpm:.1f}")
    if analyzer_rr:
        rr_arr = np.asarray(analyzer_rr, dtype=float)
        ax3r.plot(rr_arr[:, 0], rr_arr[:, 1], "v-", color="#16a34a", lw=1.1, ms=4, label=f"{analyzer_window_s:.0f} s batch RR")
    ax3r.set_ylim(0, 35)
    ax3r.set_ylabel("RR (BPM)", color="#15803d")
    ax3r.tick_params(axis="y", colors="#15803d")
    lines, labels = ax3.get_legend_handles_labels()
    lines_r, labels_r = ax3r.get_legend_handles_labels()
    ax3.legend(lines + lines_r, labels + labels_r, loc="upper right", ncol=2)

    ax4 = fig.add_subplot(grid[4])
    for label, bpm_axis, power, peak_bpm, _confidence in chunk_spectra:
        valid = (bpm_axis >= 42.0) & (bpm_axis <= 120.0)
        ax4.plot(bpm_axis[valid], power[valid], lw=1.2, label=f"{label} peak {peak_bpm:.0f}")
    if watch_band_bpm is not None:
        ax4.axvspan(watch_band_bpm[0], watch_band_bpm[1], color="#2ca02c", alpha=0.12)
    ax4.axvline(float(analysis.heart_bpm), color="black", lw=1.1, ls="--", label=f"batch full {analysis.heart_bpm:.1f}")
    ax4.axvline(float(weighted_hr.bpm), color="#666666", lw=1.0, ls=":", label=f"weighted full {weighted_hr.bpm:.1f}")
    ax4.set_xlim(42, 120)
    ax4.set_title(f"Heart-band spectra by {chunk_s:.0f} s chunk")
    ax4.set_xlabel("Frequency (BPM)")
    ax4.set_ylabel("Power")
    ax4.grid(True, alpha=0.65)
    ax4.legend(loc="upper right", ncol=3)

    for ax in (ax0, ax1, ax2, ax3):
        ax.set_xlim(float(time_s[0]), float(time_s[-1]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return PlotSummary(
        output_path=output_path,
        rows=len(df),
        duration_s=duration_s,
        sample_rate_hz=fs,
        target_distance_m=target_m,
        avg_peak_amplitude=avg_peak_amplitude,
        analyzer_hr_bpm=float(analysis.heart_bpm),
        analyzer_hr_confidence=float(analysis.heart_confidence),
        analyzer_rr_bpm=float(analysis.resp_bpm),
        analyzer_rr_confidence=float(analysis.resp_confidence),
        weighted_hr_bpm=weighted_hr.bpm,
        weighted_hr_confidence=weighted_hr.confidence,
        weighted_rr_bpm=weighted_rr.bpm,
        weighted_rr_confidence=weighted_rr.confidence,
        chunk_hr=chunk_hr,
        watch_band_chunk_hr=watch_band_chunk_hr,
        analyzer_hr_trend=analyzer_hr,
        analyzer_rr_trend=analyzer_rr,
        breath_annotation_count=len(annotations),
    )


def main() -> None:
    args = _parse_args()
    watch_start, watch_end, watch_band = _watch_values(args)
    breath_annotations = args.breath_annotations or _auto_breath_annotations_path(args.csv_path)
    output = args.output or (args.output_dir / f"{args.csv_path.stem}_heart_trend.png")
    summary = create_watch_comparison_plot(
        args.csv_path,
        output_path=output,
        watch_start_bpm=watch_start,
        watch_end_bpm=watch_end,
        watch_band_bpm=watch_band,
        chunk_s=args.chunk_s,
        analyzer_window_s=args.analyzer_window_s,
        analyzer_step_s=args.analyzer_step_s,
        cluster_half_width_m=args.cluster_half_width_m,
        strong_bin_fraction=args.strong_bin_fraction,
        title=args.title,
        breath_annotations_path=breath_annotations,
        dpi=args.dpi,
    )
    print(f"Saved: {summary.output_path}")
    print(f"Rows: {summary.rows} | duration: {summary.duration_s:.1f}s | fs: {summary.sample_rate_hz:.3f} Hz")
    print(f"Target: {summary.target_distance_m:.3f} m | avg peak amplitude: {summary.avg_peak_amplitude:.0f}")
    if breath_annotations is not None:
        print(f"Breath annotations: {breath_annotations} ({summary.breath_annotation_count} markers)")
    print(
        "Batch analyzer: "
        f"HR {summary.analyzer_hr_bpm:.1f} BPM conf {summary.analyzer_hr_confidence:.1f}; "
        f"RR {summary.analyzer_rr_bpm:.1f} BPM conf {summary.analyzer_rr_confidence:.1f}"
    )
    print(
        "Weighted PSD diagnostic: "
        f"HR {summary.weighted_hr_bpm:.1f} BPM conf {summary.weighted_hr_confidence:.1f}; "
        f"RR {summary.weighted_rr_bpm:.1f} BPM conf {summary.weighted_rr_confidence:.1f}"
    )
    if summary.chunk_hr:
        print("Chunk HR peaks:")
        for center_s, bpm, confidence in summary.chunk_hr:
            print(f"  center {center_s:5.1f}s: {bpm:5.1f} BPM conf {confidence:4.1f}")
    if summary.watch_band_chunk_hr:
        print("Watch-band-constrained chunk HR peaks:")
        for center_s, bpm, confidence in summary.watch_band_chunk_hr:
            print(f"  center {center_s:5.1f}s: {bpm:5.1f} BPM conf {confidence:4.1f}")
    if summary.analyzer_hr_trend:
        print("Sliding batch analyzer HR/RR trend:")
        rr_by_center = {round(center_s, 3): (rr_bpm, rr_conf) for center_s, rr_bpm, rr_conf in summary.analyzer_rr_trend}
        for center_s, bpm, confidence in summary.analyzer_hr_trend:
            rr_bpm, rr_conf = rr_by_center.get(round(center_s, 3), (0.0, 0.0))
            print(f"  center {center_s:5.1f}s: HR {bpm:5.1f} BPM conf {confidence:4.1f}; RR {rr_bpm:5.1f} BPM conf {rr_conf:4.1f}")


if __name__ == "__main__":
    main()
