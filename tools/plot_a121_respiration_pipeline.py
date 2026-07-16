#!/usr/bin/env python3
"""Render the A121 respiration-processing figures used in the thesis.

The figures intentionally use one fixed recording fragment and a fully
deterministic extrema rule.  They make the path from Sparse IQ to the final
respiration trace inspectable without treating automatically inferred phases
as physiological ground truth.

Run from the repository root::

    uv run python tools/plot_a121_respiration_pipeline.py
"""

from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

from respi_net.a121 import parse_json_array
from respi_net.a121_vitals import (
    A121_RESP_BAND_HZ,
    _center_complex_profile,
    _coherent_differential_phase,
    _differential_phase_matrix,
    _lowpass_static_complex,
    sample_rate_from_ms,
)


DEFAULT_CSV = ROOT / "data" / "raw" / "a121" / "a121_sparse_iq_2026-07-13_22-57-02.csv"
DEFAULT_FIGURE_DIR = ROOT / "docs" / "thesis" / "figures"
PIPELINE_FILENAME = "a121_oddech_etapy_przetwarzania.png"
MARKERS_FILENAME = "a121_oddech_fazy_deterministyczne.png"

IQ_I_COLOUR = "#0072B2"
IQ_Q_COLOUR = "#D55E00"
RESIDUAL_COLOUR = "#6A3D9A"
CENTERED_COLOUR = "#009E73"
DIFFERENTIAL_COLOUR = "#CC79A7"
FUSED_COLOUR = "#56B4E9"
FINAL_COLOUR = "#1F4E79"
INHALE_COLOUR = "#56B4E9"
EXHALE_COLOUR = "#E69F00"


@dataclass(frozen=True)
class PipelineTrace:
    """Signals at the successive, visualized stages of the A121 path."""

    time_s: np.ndarray
    raw_i: np.ndarray
    raw_q: np.ndarray
    static_residual_amplitude: np.ndarray
    centered_phase: np.ndarray
    differential_phase: np.ndarray
    coherent_phase: np.ndarray
    respiration_phase: np.ndarray
    sample_rate_hz: float
    gate_min_m: float
    gate_max_m: float
    reference_distance_m: float
    candidate_bins: int


@dataclass(frozen=True)
class PhaseInterval:
    start_s: float
    end_s: float
    phase: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render A121 respiration pipeline and deterministic phase-marker figures."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=DEFAULT_CSV,
        help="A121 Sparse IQ CSV used as the illustrative recording.",
    )
    parser.add_argument(
        "--start-s",
        type=float,
        default=28.0,
        help="Start of the stable illustrative fragment, relative to the recording start.",
    )
    parser.add_argument(
        "--end-s",
        type=float,
        default=43.0,
        help="End of the stable illustrative fragment, relative to the recording start.",
    )
    parser.add_argument(
        "--context-s",
        type=float,
        default=5.0,
        help="Extra samples on both sides used only to avoid filter-edge artifacts.",
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=0.30,
        help="Minimum deterministic extrema prominence, in standard deviations of the final trace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory receiving the two PNG figures.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    return parser.parse_args()


def _read_recording_window(csv_path: Path, start_s: float, end_s: float, context_s: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Read only the portion of a potentially long Sparse IQ recording needed here."""

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    header = pd.read_csv(csv_path, nrows=0)
    required = {"Timestamp_ms", "Distances_m", "Real", "Imag"}
    missing = required.difference(header.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(sorted(missing))}")

    columns = list(required | {"AcconeerRangeStartIndex", "AcconeerRangeEndIndex"})
    columns = [column for column in columns if column in header.columns]
    first = pd.read_csv(csv_path, usecols=["Timestamp_ms"], nrows=1)
    if first.empty:
        raise ValueError(f"{csv_path} is empty")
    time_zero_ms = float(pd.to_numeric(first["Timestamp_ms"], errors="coerce").iloc[0])
    if not np.isfinite(time_zero_ms):
        raise ValueError(f"{csv_path} has an invalid first timestamp")

    read_from_s = max(0.0, float(start_s) - max(0.0, float(context_s)))
    read_to_s = float(end_s) + max(0.0, float(context_s))
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=columns, chunksize=1500):
        timestamps = pd.to_numeric(chunk["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
        elapsed = (timestamps - time_zero_ms) / 1000.0
        valid_elapsed = elapsed[np.isfinite(elapsed)]
        if len(valid_elapsed) == 0:
            continue
        if float(np.max(valid_elapsed)) < read_from_s:
            continue
        keep = np.isfinite(elapsed) & (elapsed >= read_from_s) & (elapsed <= read_to_s)
        if np.any(keep):
            pieces.append(chunk.loc[keep].copy())
        if float(np.max(valid_elapsed)) >= read_to_s:
            break

    if not pieces:
        raise ValueError(f"No samples in {csv_path} cover {start_s:.2f}-{end_s:.2f} s")
    frame = pd.concat(pieces, ignore_index=True)
    timestamps = pd.to_numeric(frame["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    elapsed = (timestamps - time_zero_ms) / 1000.0
    order = np.argsort(elapsed, kind="mergesort")
    return frame.iloc[order].reset_index(drop=True), elapsed[order]


def _load_matrix(series: pd.Series) -> np.ndarray:
    rows = [parse_json_array(value) for value in series]
    width = min((len(row) for row in rows), default=0)
    if width <= 0:
        return np.empty((len(rows), 0), dtype=float)
    return np.vstack([row[:width] for row in rows])


def _recorded_gate(df: pd.DataFrame, display_mask: np.ndarray, width: int) -> np.ndarray:
    """Recover the modal reference-processor range segment stored in the CSV."""

    gate_columns = {"AcconeerRangeStartIndex", "AcconeerRangeEndIndex"}
    if gate_columns.issubset(df.columns):
        pairs = df.loc[display_mask, list(gate_columns)].apply(pd.to_numeric, errors="coerce").dropna()
        if not pairs.empty:
            pairs = pairs.astype(int)
            valid = pairs[
                (pairs["AcconeerRangeStartIndex"] >= 0)
                & (pairs["AcconeerRangeEndIndex"] > pairs["AcconeerRangeStartIndex"])
                & (pairs["AcconeerRangeEndIndex"] <= width)
            ]
            if not valid.empty:
                start_idx, end_idx = valid.value_counts().index[0]
                return np.arange(int(start_idx), int(end_idx), dtype=int)
    return np.asarray([], dtype=int)


def _dominant_period_s(signal: np.ndarray, fs: float) -> float:
    """Return a fixed FFT-derived period used solely to space extrema candidates."""

    values = np.asarray(signal, dtype=float)
    if len(values) < 16 or fs <= 0:
        return 5.0
    values = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(values * np.hamming(len(values))))
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / fs)
    valid = (frequencies >= A121_RESP_BAND_HZ[0]) & (frequencies <= A121_RESP_BAND_HZ[1])
    if not np.any(valid) or not np.any(np.isfinite(spectrum[valid])):
        return 5.0
    frequency = float(frequencies[valid][int(np.argmax(spectrum[valid]))])
    return 1.0 / frequency if frequency > 0 else 5.0


def deterministic_phase_intervals(
    time_s: np.ndarray,
    respiration_phase: np.ndarray,
    sample_rate_hz: float,
    *,
    prominence: float = 0.30,
) -> tuple[list[PhaseInterval], float]:
    """Label peak-to-trough as inhale and trough-to-peak as exhale.

    This intentionally small baseline has no learned parameters or hand edits:
    the FFT-derived period fixes the minimum extrema separation and ``prominence``
    fixes their required height after z-score normalization.
    """

    values = np.asarray(respiration_phase, dtype=float)
    if len(values) != len(time_s) or len(values) < 8:
        return [], 0.0
    normalized = (values - float(np.median(values))) / (float(np.std(values)) + 1e-12)
    period_s = _dominant_period_s(normalized, sample_rate_hz)
    min_distance = max(1, int(round(0.45 * period_s * sample_rate_hz)))
    peaks, _ = find_peaks(normalized, distance=min_distance, prominence=float(prominence))
    troughs, _ = find_peaks(-normalized, distance=min_distance, prominence=float(prominence))

    extrema = [(int(index), "peak") for index in peaks] + [(int(index), "trough") for index in troughs]
    extrema.sort(key=lambda item: item[0])
    alternating: list[tuple[int, str]] = []
    for index, kind in extrema:
        if not alternating or alternating[-1][1] != kind:
            alternating.append((index, kind))
            continue
        previous_index, previous_kind = alternating[-1]
        is_more_extreme = (
            normalized[index] > normalized[previous_index]
            if kind == "peak"
            else normalized[index] < normalized[previous_index]
        )
        if is_more_extreme:
            alternating[-1] = (index, previous_kind)

    intervals: list[PhaseInterval] = []
    for (first_index, first_kind), (second_index, second_kind) in zip(alternating, alternating[1:]):
        if second_kind == first_kind:
            continue
        intervals.append(
            PhaseInterval(
                start_s=float(time_s[first_index]),
                end_s=float(time_s[second_index]),
                phase="inhale" if first_kind == "peak" else "exhale",
            )
        )
    rate_bpm = 60.0 / period_s if period_s > 0 else 0.0
    return intervals, rate_bpm


def build_pipeline_trace(
    csv_path: Path,
    *,
    start_s: float = 28.0,
    end_s: float = 43.0,
    context_s: float = 5.0,
) -> PipelineTrace:
    """Build the six visual stages for one stable, fixed recording segment."""

    if not (np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s):
        raise ValueError("end_s must be greater than start_s")
    df, elapsed_s = _read_recording_window(csv_path, start_s, end_s, context_s)
    timestamps_ms = pd.to_numeric(df["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    fs = sample_rate_from_ms(timestamps_ms, default=20.0)
    display_mask = (elapsed_s >= float(start_s)) & (elapsed_s <= float(end_s))
    if int(np.count_nonzero(display_mask)) < 32:
        raise ValueError("The requested illustrative fragment contains too few samples")

    distances = parse_json_array(df["Distances_m"].iloc[0])
    real = _load_matrix(df["Real"])
    imag = _load_matrix(df["Imag"])
    width = min(len(distances), real.shape[1], imag.shape[1])
    if width < 1:
        raise ValueError(f"{csv_path} does not contain usable complex range bins")
    distances = distances[:width]
    complex_profile = real[:, :width] + 1j * imag[:, :width]

    static_echo = _lowpass_static_complex(complex_profile, fs, A121_RESP_BAND_HZ[0])
    residual_profile = complex_profile - static_echo
    candidate_idx = _recorded_gate(df, display_mask, width)
    residual_median = np.median(np.abs(residual_profile), axis=0)
    if len(candidate_idx) == 0:
        selected_idx = int(np.argmax(residual_median))
        candidate_idx = np.arange(max(0, selected_idx - 1), min(width, selected_idx + 2), dtype=int)
    weights = np.maximum(residual_median[candidate_idx], 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(candidate_idx), dtype=float)
    weights /= float(np.sum(weights)) + 1e-12
    reference_idx = int(candidate_idx[int(np.argmax(weights))])

    centered_profile, _, _ = _center_complex_profile(real[:, :width], imag[:, :width])
    centered_absolute_phase = detrend(np.unwrap(np.angle(centered_profile), axis=0), axis=0, type="linear")
    differential_phase = _differential_phase_matrix(centered_profile)
    coherent_phase, _, _ = _coherent_differential_phase(complex_profile, np.bincount(candidate_idx, weights=weights, minlength=width))
    respiratory_sos = butter(2, A121_RESP_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    respiration_phase = sosfiltfilt(respiratory_sos, coherent_phase)

    return PipelineTrace(
        time_s=elapsed_s[display_mask],
        raw_i=np.real(complex_profile[display_mask, reference_idx]),
        raw_q=np.imag(complex_profile[display_mask, reference_idx]),
        static_residual_amplitude=np.abs(residual_profile[display_mask, reference_idx]),
        centered_phase=centered_absolute_phase[display_mask, reference_idx],
        differential_phase=differential_phase[display_mask, reference_idx],
        coherent_phase=coherent_phase[display_mask],
        respiration_phase=respiration_phase[display_mask],
        sample_rate_hz=float(fs),
        gate_min_m=float(distances[int(candidate_idx[0])]),
        gate_max_m=float(distances[int(candidate_idx[-1])]),
        reference_distance_m=float(distances[reference_idx]),
        candidate_bins=int(len(candidate_idx)),
    )


def _configure_axis(axis: plt.Axes) -> None:
    axis.axhline(0.0, color="#9A9A9A", linewidth=0.65, zorder=0)
    axis.grid(axis="x", color="#D0D0D0", linewidth=0.55, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8)


def plot_pipeline(trace: PipelineTrace, output_path: Path, *, dpi: int) -> None:
    """Draw six stages in chronological reading order, two columns by three rows."""

    figure, axes = plt.subplots(3, 2, figsize=(9.3, 8.35), sharex=True)
    x = trace.time_s
    panels = [
        ("1. Surowe Sparse IQ", "jedn. ADC", [(trace.raw_i, IQ_I_COLOUR, "$I$"), (trace.raw_q, IQ_Q_COLOUR, "$Q$")]),
        (
            "2. Po usunięciu echa statycznego",
            "jedn. ADC",
            [(trace.static_residual_amplitude, RESIDUAL_COLOUR, "$|z-z_{stat}|$")],
        ),
        (
            "3. Faza po wycentrowaniu IQ",
            "rad",
            [(trace.centered_phase, CENTERED_COLOUR, "faza rozwinięta")],
        ),
        (
            "4. Faza różnicowa między ramkami",
            "rad",
            [(trace.differential_phase, DIFFERENTIAL_COLOUR, "$\\sum\\Delta\\varphi$")],
        ),
        (
            "5. Koherentne łączenie bramek",
            "rad",
            [(trace.coherent_phase, FUSED_COLOUR, "bramki ważone")],
        ),
        (
            "6. Stan końcowy: pasmo 0,10--0,50 Hz",
            "rad",
            [(trace.respiration_phase, FINAL_COLOUR, "sygnał oddechowy")],
        ),
    ]
    for axis, (title, ylabel, series) in zip(axes.flat, panels, strict=True):
        for values, colour, label in series:
            axis.plot(x, values, color=colour, linewidth=1.1, label=label)
        axis.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        axis.set_ylabel(ylabel, fontsize=8.5)
        _configure_axis(axis)
        if len(series) > 1:
            axis.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)

    for axis in axes[-1]:
        axis.set_xlabel("czas od początku nagrania [s]", fontsize=8.5)
    figure.suptitle(
        "Ten sam fragment sygnału A121 od surowego IQ do oddechu",
        fontsize=12,
        fontweight="bold",
        y=0.992,
    )
    figure.text(
        0.5,
        0.006,
        (
            f"Bramka referencyjna: {trace.gate_min_m:.3f}--{trace.gate_max_m:.3f} m "
            f"({trace.candidate_bins} bramki, $f_s$={trace.sample_rate_hz:.2f} Hz). "
            "Kolejność paneli: wierszami od lewej do prawej; skale pionowe są niezależne."
        ),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.02, 0.035, 0.98, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_deterministic_markers(
    trace: PipelineTrace,
    intervals: list[PhaseInterval],
    rate_bpm: float,
    output_path: Path,
    *,
    prominence: float,
    dpi: int,
) -> None:
    """Draw the final trace and immutable peak/trough-derived inhale/exhale labels."""

    figure, axis = plt.subplots(figsize=(9.3, 3.65))
    x = trace.time_s
    y = trace.respiration_phase
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = max(y_max - y_min, 1e-9)

    for interval in intervals:
        colour = INHALE_COLOUR if interval.phase == "inhale" else EXHALE_COLOUR
        label = "WDECH" if interval.phase == "inhale" else "WYDECH"
        axis.axvspan(interval.start_s, interval.end_s, facecolor=colour, alpha=0.20, linewidth=0, zorder=0)
        if interval.end_s - interval.start_s >= 0.65:
            axis.text(
                0.5 * (interval.start_s + interval.end_s),
                y_max + 0.115 * span,
                label,
                color=colour,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        marker = "v" if interval.phase == "inhale" else "^"
        marker_y = float(np.interp(interval.start_s, x, y))
        axis.scatter(interval.start_s, marker_y, marker=marker, s=34, color=colour, edgecolor="white", linewidth=0.45, zorder=4)

    axis.plot(x, y, color=FINAL_COLOUR, linewidth=1.5, label="sygnał po filtracji")
    axis.set_ylim(y_min - 0.16 * span, y_max + 0.28 * span)
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_xlabel("czas od początku nagrania [s]")
    axis.set_ylabel("faza [rad]")
    axis.set_title("Stan końcowy sygnału oddechowego z deterministycznymi znacznikami faz", loc="left", fontsize=11, fontweight="bold")
    _configure_axis(axis)
    legend_handles = [
        Line2D([0], [0], color=FINAL_COLOUR, linewidth=1.5, label="sygnał po filtracji"),
        Patch(facecolor=INHALE_COLOUR, alpha=0.20, label="wdech: maksimum → minimum"),
        Patch(facecolor=EXHALE_COLOUR, alpha=0.20, label="wydech: minimum → maksimum"),
    ]
    axis.legend(handles=legend_handles, loc="lower right", fontsize=8, frameon=True, ncol=1)
    axis.text(
        0.01,
        0.02,
        (
            f"Reguła automatyczna: prominencja ≥ {prominence:.2f} σ, odstęp ≥ 0,45 T; "
            f"T z maksimum FFT ({rate_bpm:.1f} odd./min)."
        ),
        transform=axis.transAxes,
        fontsize=8,
        color="#333333",
        va="bottom",
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    trace = build_pipeline_trace(
        args.csv_path,
        start_s=args.start_s,
        end_s=args.end_s,
        context_s=args.context_s,
    )
    intervals, rate_bpm = deterministic_phase_intervals(
        trace.time_s,
        trace.respiration_phase,
        trace.sample_rate_hz,
        prominence=args.prominence,
    )
    pipeline_path = args.output_dir / PIPELINE_FILENAME
    markers_path = args.output_dir / MARKERS_FILENAME
    plot_pipeline(trace, pipeline_path, dpi=args.dpi)
    plot_deterministic_markers(
        trace,
        intervals,
        rate_bpm,
        markers_path,
        prominence=args.prominence,
        dpi=args.dpi,
    )
    print(f"Pipeline figure: {pipeline_path}")
    print(f"Phase-marker figure: {markers_path}")
    print(
        f"Fragment {trace.time_s[0]:.2f}-{trace.time_s[-1]:.2f} s; "
        f"gate {trace.gate_min_m:.3f}-{trace.gate_max_m:.3f} m; "
        f"{len(intervals)} deterministic phase intervals; FFT rate {rate_bpm:.2f} bpm"
    )


if __name__ == "__main__":
    main()
