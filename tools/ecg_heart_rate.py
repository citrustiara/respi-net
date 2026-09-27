#!/usr/bin/env python3
"""Detect R-peaks in an AD8232 ECG capture and export heart-rate labels.

Reads an ESP32 ADC capture (``Timestamp_ms,RawADC,Voltage_mV``, optionally with
``HostTimestamp_ms`` from the host capture tools, or a raw headerless serial dump) and
writes next to the input, or into ``--output-dir``:

* ``<stem>_beats.csv``: one row per beat with its time, the RR interval ending at it,
  instantaneous HR and the validity flag;
* ``<stem>_hr_windows.csv``: mean HR per window (10 s by default), NaN when the window
  holds too few valid RR intervals; these are the labels for a windowed radar HR model;
* ``<stem>_ecg_hr.png`` with ``--plot``: an ECG excerpt with the detected beats and the
  heart-rate trace.

Beat times are R-peaks. Shift them with ``respi_net.ecg.mechanical_beat_times`` before
using them as beat-level radar labels; windowed HR needs no shift.

Run from the repository root::

    uv run python tools/ecg_heart_rate.py data/raw/ecg/ecg_session.csv --plot
"""

from __future__ import annotations

import argparse
import sys
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

from respi_net.ecg import (
    DEFAULT_MAX_RR_CHANGE,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_VALID_BEATS,
    EcgRecording,
    HeartRateSeries,
    detect_r_peaks,
    heart_rate_series,
    hr_labels_for_windows,
    load_ecg_csv,
)

# Light chart palette: blue marks beats in both panels, orange the window labels, gray is
# context (the raw trace) or excluded data (rejected intervals).
SURFACE, INK, INK_SECONDARY, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BEAT_COLOR, WINDOW_COLOR = "#2a78d6", "#eb6834"
EXCERPT_S = 20.0


def beats_table(recording: EcgRecording, peak_times_s: np.ndarray, series: HeartRateSeries) -> pd.DataFrame:
    """One row per beat; the first beat has no preceding RR interval."""
    count = len(peak_times_s)
    rr = np.concatenate(([np.nan], series.rr_s))[:count]
    table = pd.DataFrame(
        {
            "beat_time_s": peak_times_s,
            "rr_s": rr,
            "hr_bpm": 60.0 / rr,
            "valid": np.concatenate(([False], series.valid))[:count],
        }
    )
    if recording.start_epoch_s is not None:
        # Host wall-clock time, for aligning beats with radar captures stamped on the same host.
        table.insert(1, "beat_epoch_s", recording.start_epoch_s + peak_times_s)
    return table


def windows_table(
    recording: EcgRecording,
    peak_times_s: np.ndarray,
    *,
    window_s: float,
    step_s: float,
    min_valid_beats: int,
    min_coverage: float,
    max_rr_change: float,
) -> pd.DataFrame:
    starts = np.arange(0.0, recording.duration_s - window_s + 1e-9, step_s)
    labels = hr_labels_for_windows(
        peak_times_s, starts, window_s, min_valid_beats=min_valid_beats, min_coverage=min_coverage,
        max_rr_change=max_rr_change,
    )
    table = pd.DataFrame({"window_start_s": starts, "window_end_s": starts + window_s, "hr_bpm": labels})
    if recording.start_epoch_s is not None:
        table.insert(1, "window_start_epoch_s", recording.start_epoch_s + starts)
    return table


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    # Legend in the title row, so it never covers data.
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=8,
        labelcolor=INK_SECONDARY, borderaxespad=0.2,
    )


def plot_overview(
    recording: EcgRecording,
    peaks: np.ndarray,
    series: HeartRateSeries,
    windows: pd.DataFrame,
    window_s: float,
    title: str,
    path: Path,
) -> None:
    fig, (ax_ecg, ax_hr) = plt.subplots(2, 1, figsize=(11, 6.5), facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", color=INK_SECONDARY, fontsize=9)

    shown = recording.time_s <= EXCERPT_S
    ax_ecg.plot(recording.time_s[shown], recording.voltage_mv[shown], color=INK_SECONDARY, lw=0.6, label="ECG")
    beats = peaks[recording.time_s[peaks] <= EXCERPT_S]
    ax_ecg.plot(
        recording.time_s[beats], recording.voltage_mv[beats], "o", color=BEAT_COLOR, ms=4.5,
        mec=SURFACE, mew=1.0, label="detected R-peak",
    )
    _style(ax_ecg, f"ECG with detected R-peaks (first {EXCERPT_S:.0f} s)", "Time [s]", "ADC input [mV]")

    valid = series.valid
    low, high = 30.0, 200.0
    if np.any(valid):
        # Rejected intervals can imply absurd rates: scale the axis to the valid beats and
        # pin off-scale rejections to its edge so they stay visible.
        p_low, p_high = np.percentile(series.hr_bpm[valid], [1, 99])
        low, high = max(20.0, p_low - 15.0), min(240.0, p_high + 15.0)
    ax_hr.plot(
        series.beat_times_s[valid], series.hr_bpm[valid], "o", color=BEAT_COLOR, ms=3.5,
        mec=SURFACE, mew=0.6, label="beat-to-beat HR",
    )
    ax_hr.plot(
        series.beat_times_s[~valid], np.clip(series.hr_bpm[~valid], low, high), "x", color=INK_MUTED,
        ms=4, mew=1.0, clip_on=False, label="rejected RR",
    )
    ax_hr.hlines(
        windows["hr_bpm"], windows["window_start_s"], windows["window_end_s"], color=WINDOW_COLOR, lw=1.5,
        label=f"{window_s:g} s window label",
    )
    ax_hr.set_ylim(low, high)
    _style(ax_hr, "Heart rate", "Time [s]", "Heart rate [bpm]")

    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="ECG CSV from the ESP32 ADC firmware or a host capture tool")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: next to the input CSV")
    parser.add_argument("--window-s", type=float, default=10.0, help="HR label window length in seconds")
    parser.add_argument("--step-s", type=float, default=None, help="window step in seconds (default: no overlap)")
    parser.add_argument(
        "--min-valid-beats", type=int, default=DEFAULT_MIN_VALID_BEATS,
        help="valid RR intervals a window needs for a label",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
        help="share of a window its valid RR intervals must span for a label",
    )
    parser.add_argument(
        "--max-rr-change", type=float, default=DEFAULT_MAX_RR_CHANGE,
        help="max relative deviation of an RR interval from its local median",
    )
    parser.add_argument("--plot", action="store_true", help="also save <stem>_ecg_hr.png")
    args = parser.parse_args()
    step_s = args.window_s if args.step_s is None else args.step_s
    if args.window_s <= 0 or step_s <= 0:
        parser.error("--window-s and --step-s must be positive")

    try:
        recording = load_ecg_csv(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ecg_heart_rate: {exc}") from exc
    peaks = detect_r_peaks(recording.voltage_mv, recording.fs)
    peak_times = recording.time_s[peaks]
    series = heart_rate_series(peak_times, max_rr_change=args.max_rr_change)
    windows = windows_table(
        recording, peak_times, window_s=args.window_s, step_s=step_s, min_valid_beats=args.min_valid_beats,
        min_coverage=args.min_coverage, max_rr_change=args.max_rr_change,
    )

    output_dir = args.output_dir or args.csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    beats_path = output_dir / f"{args.csv.stem}_beats.csv"
    windows_path = output_dir / f"{args.csv.stem}_hr_windows.csv"
    beats_table(recording, peak_times, series).to_csv(beats_path, index=False, float_format="%.6f")
    windows.to_csv(windows_path, index=False, float_format="%.6f")

    valid_hr = series.hr_bpm[series.valid]
    summary = f"{len(peaks)} beats, {int(series.valid.sum())}/{len(series.rr_s)} valid RR"
    if valid_hr.size:
        summary += f", median HR {np.median(valid_hr):.1f} bpm"
    labelled = int(np.isfinite(windows["hr_bpm"]).sum())
    print(f"{args.csv.name}: {recording.duration_s:.1f} s at {recording.fs:.1f} Hz ({recording.clock} clock)")
    print(f"{summary}; {labelled}/{len(windows)} windows of {args.window_s:g} s labelled")
    print(f"wrote {beats_path}")
    print(f"wrote {windows_path}")
    if args.plot:
        plot_path = output_dir / f"{args.csv.stem}_ecg_hr.png"
        title = f"{args.csv.name}  |  {recording.duration_s:.0f} s at {recording.fs:.0f} Hz  |  {summary}"
        plot_overview(recording, peaks, series, windows, args.window_s, title, plot_path)
        print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
