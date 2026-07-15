#!/usr/bin/env python3
"""Analyse the guided A121 foil/angle retest recorded at 2 m.

The script reads the accepted measurements and cue timeline from the guided
session manifest, reconstructs a phase-based chest-displacement signal from
the complex A121 range bins, estimates the aggregate cue-to-signal delay for
each recording, and generates the figures used by the thesis.

Run from the repository root with::

    uv run python tools/analyze_a121_foil_2m.py

Use ``--session`` to analyse a session other than the latest
``guided_foil2m_*`` directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

from respi_net.a121 import parse_json_array


ROOT = Path(__file__).resolve().parents[1]
GUIDED_DIR = ROOT / "data" / "raw" / "a121" / "guided"
DEFAULT_FIGURE_DIR = ROOT / "docs" / "thesis" / "figures"
RADAR_CENTRE_FREQUENCY_HZ = 60.5e9
SPEED_OF_LIGHT_M_S = 299_792_458.0
PHASE_TO_DISPLACEMENT_MM = (
    SPEED_OF_LIGHT_M_S / RADAR_CENTRE_FREQUENCY_HZ / (4.0 * np.pi) * 1000.0
)
CONDITION_ORDER = ("N0", "NF", "P0", "PF")
CONDITION_LABELS = {
    "N0": "naturalna\nbez folii",
    "NF": "naturalna\nfolia",
    "P0": "prostopadła\nbez folii",
    "PF": "prostopadła\nfolia",
}
CONDITION_LONG_LABELS = {
    "N0": "N0 — naturalna, bez folii",
    "NF": "NF — naturalna, folia",
    "P0": "P0 — prostopadła, bez folii",
    "PF": "PF — prostopadła, folia",
}
CONDITION_COLOURS = {
    "N0": "#0072B2",
    "NF": "#0072B2",
    "P0": "#D55E00",
    "PF": "#D55E00",
}
CONDITION_LINESTYLES = {"N0": "--", "NF": "-", "P0": "--", "PF": "-"}


@dataclass
class RecordingResult:
    metadata: dict[str, Any]
    time_s: np.ndarray
    distances_m: np.ndarray
    range_profile: np.ndarray
    displacement_mm: np.ndarray
    lowpass_displacement_mm: np.ndarray
    metrics: dict[str, Any]

    @property
    def condition(self) -> str:
        return str(self.metadata["condition_id"])

    @property
    def repeat(self) -> int:
        return int(self.metadata["repeat_number"])

    @property
    def run(self) -> int:
        return int(self.metadata["step"])


def _latest_session() -> Path:
    candidates = sorted(GUIDED_DIR.glob("guided_foil2m_*/manifest.json"))
    if not candidates:
        raise FileNotFoundError(f"No guided_foil2m session found below {GUIDED_DIR}")
    return candidates[-1].parent


def _load_matrix(series: pd.Series) -> np.ndarray:
    rows = [parse_json_array(value) for value in series]
    if not rows or min((len(row) for row in rows), default=0) == 0:
        return np.empty((len(rows), 0), dtype=float)
    width = min(len(row) for row in rows)
    return np.vstack([row[:width] for row in rows])


def _sample_rate_and_time(timestamps_ms: np.ndarray) -> tuple[float, np.ndarray]:
    if len(timestamps_ms) < 2:
        return 0.0, np.zeros(len(timestamps_ms), dtype=float)
    diffs_s = np.diff(timestamps_ms) / 1000.0
    positive = diffs_s[np.isfinite(diffs_s) & (diffs_s > 0)]
    if len(positive) == 0:
        raise ValueError("Recording has no positive timestamp increments")
    fs = 1.0 / float(np.median(positive))
    time_s = (timestamps_ms - float(timestamps_ms[0])) / 1000.0
    return fs, time_s


def _numeric_values(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        return np.full(len(df), np.nan, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _finite_median(values: np.ndarray, default: float = float("nan")) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else float(default)


def _cue_pairs(cues: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, cue in enumerate(cues[:-1]):
        following = cues[index + 1]
        if cue.get("kind") == "inhale" and following.get("kind") == "exhale":
            pairs.append((cue, following))
    return pairs


def cue_template(
    time_s: np.ndarray,
    cues: list[dict[str, Any]],
    delay_s: float = 0.0,
) -> np.ndarray:
    """Return a triangular chest-motion template for the paced cue intervals.

    With the current A121 phase convention, the trace falls during inhale and
    rises during exhale. Values outside paced inhale/exhale pairs are NaN.
    """

    template = np.full(len(time_s), np.nan, dtype=float)
    for inhale, exhale in _cue_pairs(cues):
        start = float(inhale["start_s"]) + delay_s
        turn = float(inhale["end_s"]) + delay_s
        end = float(exhale["end_s"]) + delay_s
        inhale_mask = (time_s >= start) & (time_s < turn)
        exhale_mask = (time_s >= turn) & (time_s <= end)
        if turn > start:
            template[inhale_mask] = 1.0 - 2.0 * (time_s[inhale_mask] - start) / (turn - start)
        if end > turn:
            template[exhale_mask] = -1.0 + 2.0 * (time_s[exhale_mask] - turn) / (end - turn)
    return template


def estimate_cue_delay(
    time_s: np.ndarray,
    signal: np.ndarray,
    cues: list[dict[str, Any]],
    *,
    minimum_delay_s: float = -0.25,
    maximum_delay_s: float = 1.50,
    step_s: float = 0.025,
) -> tuple[float, float]:
    """Estimate aggregate human/acquisition delay by template correlation."""

    normalised = (signal - float(np.median(signal))) / (float(np.std(signal)) + 1e-12)
    delays = np.arange(minimum_delay_s, maximum_delay_s + step_s * 0.5, step_s)
    correlations = np.full(len(delays), np.nan, dtype=float)
    for index, delay in enumerate(delays):
        template = cue_template(time_s, cues, float(delay))
        mask = np.isfinite(template) & np.isfinite(normalised)
        if np.count_nonzero(mask) < 20:
            continue
        correlations[index] = float(np.corrcoef(template[mask], normalised[mask])[0, 1])
    if not np.any(np.isfinite(correlations)):
        raise ValueError("Could not align the signal with the paced-breathing template")
    best = int(np.nanargmax(correlations))
    return float(delays[best]), float(correlations[best])


def _matched_extrema(
    time_s: np.ndarray,
    signal: np.ndarray,
    fs: float,
    cues: list[dict[str, Any]],
    delay_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    normalised = (signal - float(np.median(signal))) / (float(np.std(signal)) + 1e-12)
    minimum_distance = max(1, int(round(fs * 3.2)))
    peaks, _ = find_peaks(normalised, distance=minimum_distance, prominence=0.25)
    troughs, _ = find_peaks(-normalised, distance=minimum_distance, prominence=0.25)

    inhale_starts = np.asarray(
        [float(inhale["start_s"]) for inhale, _ in _cue_pairs(cues)], dtype=float
    )
    inhale_ends = np.asarray(
        [float(inhale["end_s"]) for inhale, _ in _cue_pairs(cues)], dtype=float
    )

    def match(indices: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        event_times: list[float] = []
        event_delays: list[float] = []
        candidates = time_s[indices]
        for cue_time in expected:
            local = candidates[np.abs(candidates - (cue_time + delay_s)) <= 1.2]
            if len(local) == 0:
                continue
            event_time = float(local[int(np.argmin(np.abs(local - (cue_time + delay_s))))])
            event_times.append(event_time)
            event_delays.append(event_time - cue_time)
        return np.asarray(event_times), np.asarray(event_delays)

    peak_times, peak_delays = match(peaks, inhale_starts)
    _, trough_delays = match(troughs, inhale_ends)

    # The two paced blocks are separated by a hold; do not treat that gap as
    # one respiratory period when estimating BPM from successive maxima.
    periods: list[float] = []
    for block_start, block_end in ((10.0, 45.0), (60.0, 90.0)):
        block = peak_times[
            (peak_times >= block_start + delay_s - 1.0)
            & (peak_times <= block_end + delay_s + 1.0)
        ]
        if len(block) >= 2:
            candidate_periods = np.diff(block)
            periods.extend(candidate_periods[(candidate_periods >= 3.5) & (candidate_periods <= 6.5)])
    bpm = 60.0 / float(np.median(periods)) if periods else float("nan")
    return peak_delays, trough_delays, bpm


def _interval_mask(
    time_s: np.ndarray,
    cues: Iterable[dict[str, Any]],
    kinds: set[str],
    delay_s: float,
    margin_s: float = 0.0,
) -> np.ndarray:
    mask = np.zeros(len(time_s), dtype=bool)
    for cue in cues:
        if str(cue.get("kind")) not in kinds:
            continue
        start = float(cue["start_s"]) + delay_s + margin_s
        end = float(cue["end_s"]) + delay_s - margin_s
        if end > start:
            mask |= (time_s >= start) & (time_s <= end)
    return mask


def _hold_interior_mask(
    time_s: np.ndarray,
    cues: list[dict[str, Any]],
    delay_s: float,
    edge_margin_s: float = 3.0,
) -> np.ndarray:
    hold_cues = [cue for cue in cues if cue.get("kind") == "hold"]
    if len(hold_cues) != 1:
        raise ValueError(f"Expected exactly one hold cue, found {len(hold_cues)}")
    hold = hold_cues[0]
    start = float(hold["start_s"]) + delay_s + edge_margin_s
    end = float(hold["end_s"]) + delay_s - edge_margin_s
    return (time_s >= start) & (time_s <= end)


def analyse_recording(
    metadata: dict[str, Any],
    cues: list[dict[str, Any]],
) -> RecordingResult:
    csv_path = Path(str(metadata["csv_path"]))
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty recording: {csv_path}")
    required = {"Timestamp_ms", "Distances_m", "Real", "Imag"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")

    timestamps_ms = _numeric_values(df, "Timestamp_ms")
    fs, time_s = _sample_rate_and_time(timestamps_ms)
    distances = parse_json_array(df["Distances_m"].iloc[0])
    real = _load_matrix(df["Real"])
    imag = _load_matrix(df["Imag"])
    width = min(len(distances), real.shape[1], imag.shape[1])
    if width == 0:
        raise ValueError(f"No complex range bins in {csv_path}")
    distances = distances[:width]
    complex_profile = real[:, :width] + 1j * imag[:, :width]
    amplitude = np.abs(complex_profile)
    range_profile = np.median(amplitude, axis=0)

    target_values = _numeric_values(df, "AcconeerTargetDistance_m")
    target_values = target_values[np.isfinite(target_values) & (target_values > 0)]
    if len(target_values):
        target_m = float(np.median(target_values))
    else:
        target_m = float(distances[int(np.argmax(range_profile))])

    cluster = np.flatnonzero(np.abs(distances - target_m) <= 0.20)
    if len(cluster) == 0:
        cluster = np.asarray([int(np.argmin(np.abs(distances - target_m)))], dtype=int)
    local_peak = float(np.max(range_profile[cluster]))
    strong = cluster[range_profile[cluster] >= max(0.25 * local_peak, 1.0)]
    if len(strong):
        cluster = strong
    weights = np.maximum(range_profile[cluster], 0.0)
    weights /= float(np.sum(weights)) + 1e-12

    phase = detrend(np.unwrap(np.angle(complex_profile), axis=0), axis=0, type="linear")
    phase_trace = phase[:, cluster] @ weights
    respiratory_sos = butter(2, [0.10, 0.50], btype="bandpass", fs=fs, output="sos")
    respiratory_phase = sosfiltfilt(respiratory_sos, phase_trace)
    displacement_mm = respiratory_phase * PHASE_TO_DISPLACEMENT_MM
    lowpass_sos = butter(3, 0.70, btype="lowpass", fs=fs, output="sos")
    lowpass_displacement_mm = sosfiltfilt(lowpass_sos, phase_trace) * PHASE_TO_DISPLACEMENT_MM

    delay_s, template_correlation = estimate_cue_delay(time_s, respiratory_phase, cues)
    peak_delays, trough_delays, extrema_bpm = _matched_extrema(
        time_s, respiratory_phase, fs, cues, delay_s
    )
    paced_mask = _interval_mask(
        time_s, cues, {"inhale", "exhale"}, delay_s, margin_s=0.10
    )
    hold_mask = _hold_interior_mask(time_s, cues, delay_s, edge_margin_s=3.0)
    if not np.any(paced_mask) or not np.any(hold_mask):
        raise ValueError(f"Cue masks are empty for {csv_path}")

    paced_values = displacement_mm[paced_mask]
    hold_values = displacement_mm[hold_mask]
    paced_rms_mm = float(np.sqrt(np.mean(np.square(paced_values))))
    hold_rms_mm = float(np.sqrt(np.mean(np.square(hold_values))))
    breath_amplitude_mm = float(
        0.5 * (np.quantile(paced_values, 0.95) - np.quantile(paced_values, 0.05))
    )
    hold_vs_paced_db = float(20.0 * np.log10((hold_rms_mm + 1e-12) / (paced_rms_mm + 1e-12)))

    velocity_mm_s = np.gradient(lowpass_displacement_mm, time_s)
    paced_velocity_rms = float(np.sqrt(np.mean(np.square(velocity_mm_s[paced_mask]))))
    hold_velocity_rms = float(np.sqrt(np.mean(np.square(velocity_mm_s[hold_mask]))))
    hold_velocity_vs_paced_db = float(
        20.0
        * np.log10((hold_velocity_rms + 1e-12) / (paced_velocity_rms + 1e-12))
    )

    peak_amplitude = _numeric_values(df, "PeakAmplitude")
    peak_distance = _numeric_values(df, "PeakDistance_m")
    presence = _numeric_values(df, "AcconeerPresenceDetected")
    acconeer_bpm = _numeric_values(df, "AcconeerBreathingRate_BPM")
    valid_bpm = acconeer_bpm[np.isfinite(acconeer_bpm) & (acconeer_bpm > 0)]

    target_index = int(np.argmin(np.abs(distances - target_m)))
    metrics: dict[str, Any] = {
        "run": int(metadata["step"]),
        "condition": str(metadata["condition_id"]),
        "repeat": int(metadata["repeat_number"]),
        "foil": str(metadata["patch"]),
        "geometry": str(metadata["geometry"]),
        "frames": int(len(df)),
        "duration_s": float(time_s[-1] - time_s[0]),
        "sample_rate_hz": float(fs),
        "target_distance_m": float(target_m),
        "peak_distance_median_m": _finite_median(peak_distance),
        "peak_distance_std_cm": float(np.nanstd(peak_distance) * 100.0),
        "echo_peak_median": _finite_median(peak_amplitude),
        "echo_at_target_median": float(range_profile[target_index]),
        "cluster_bins": int(len(cluster)),
        "cue_delay_s": float(delay_s),
        "template_correlation": float(template_correlation),
        "inhale_start_delay_s": _finite_median(peak_delays),
        "inhale_end_delay_s": _finite_median(trough_delays),
        "matched_inhale_starts": int(len(peak_delays)),
        "matched_inhale_ends": int(len(trough_delays)),
        "extrema_bpm": float(extrema_bpm),
        "breath_amplitude_mm": float(breath_amplitude_mm),
        "paced_rms_mm": float(paced_rms_mm),
        "hold_rms_mm": float(hold_rms_mm),
        "hold_vs_paced_db": float(hold_vs_paced_db),
        "paced_velocity_rms_mm_s": float(paced_velocity_rms),
        "hold_velocity_rms_mm_s": float(hold_velocity_rms),
        "hold_velocity_vs_paced_db": float(hold_velocity_vs_paced_db),
        "presence_percent": float(np.nanmean(presence > 0) * 100.0),
        "acconeer_bpm_median": _finite_median(valid_bpm),
        "acconeer_bpm_valid_percent": float(len(valid_bpm) / len(df) * 100.0),
        "batch_resp_bpm": float(metadata.get("analysis_resp_bpm", float("nan"))),
        "batch_resp_confidence": float(
            metadata.get("analysis_resp_confidence", float("nan"))
        ),
    }
    return RecordingResult(
        metadata=metadata,
        time_s=time_s,
        distances_m=distances,
        range_profile=range_profile,
        displacement_mm=displacement_mm,
        lowpass_displacement_mm=lowpass_displacement_mm,
        metrics=metrics,
    )


def _polish_number(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#dedbd2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def _condition_scatter(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    column: str,
    ylabel: str,
    *,
    ylim: tuple[float, float] | None = None,
    log: bool = False,
) -> None:
    offsets = {1: -0.12, 2: 0.0, 3: 0.12}
    for x, condition in enumerate(CONDITION_ORDER):
        values = metrics.loc[metrics["condition"] == condition].sort_values("repeat")
        for row in values.itertuples(index=False):
            repeat = int(getattr(row, "repeat"))
            ax.scatter(
                x + offsets.get(repeat, 0.0),
                float(getattr(row, column)),
                s=48,
                color=CONDITION_COLOURS[condition],
                facecolor=(
                    CONDITION_COLOURS[condition] if condition in {"NF", "PF"} else "white"
                ),
                linewidth=1.8,
                zorder=3,
            )
        median = float(values[column].median())
        ax.plot([x - 0.22, x + 0.22], [median, median], color="#333333", linewidth=2.2)
    ax.set_xticks(range(len(CONDITION_ORDER)), [CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")
    ax.axvline(1.5, color="#aaa69d", linewidth=0.8)
    _style_axes(ax)


def plot_range_profiles(
    results: list[RecordingResult],
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.2),
        gridspec_kw={"width_ratios": (2.15, 1.0, 1.0)},
        constrained_layout=True,
    )
    profile_ax, echo_ax, distance_ax = axes
    for condition in CONDITION_ORDER:
        condition_results = sorted(
            [result for result in results if result.condition == condition],
            key=lambda result: result.repeat,
        )
        if not condition_results:
            continue
        grid = condition_results[0].distances_m
        profiles: list[np.ndarray] = []
        for result in condition_results:
            profile = np.interp(grid, result.distances_m, result.range_profile)
            profiles.append(profile)
            profile_ax.plot(
                grid,
                profile,
                color=CONDITION_COLOURS[condition],
                linestyle=CONDITION_LINESTYLES[condition],
                linewidth=1.1,
                alpha=0.27,
            )
        profile_ax.plot(
            grid,
            np.median(np.vstack(profiles), axis=0),
            color=CONDITION_COLOURS[condition],
            linestyle=CONDITION_LINESTYLES[condition],
            linewidth=2.5,
            label=CONDITION_LONG_LABELS[condition],
        )
    profile_ax.axvline(2.0, color="#555555", linestyle=":", linewidth=1.3)
    profile_ax.text(
        2.01,
        0.98,
        "2,00 m",
        transform=profile_ax.get_xaxis_transform(),
        va="top",
    )
    profile_ax.set_yscale("log")
    profile_ax.set_xlim(1.50, 2.50)
    profile_ax.set_xlabel("Odległość bramki [m]")
    profile_ax.set_ylabel("Mediana |IQ| [j.u.] (skala log.)")
    profile_ax.set_title("Profile zasięgowe\n(cienkie: próby, grube: mediany)")
    profile_ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    _style_axes(profile_ax)

    _condition_scatter(
        echo_ax,
        metrics,
        "echo_peak_median",
        "Mediana amplitudy piku [j.u.]",
        log=True,
    )
    echo_ax.set_title("Siła dominującego echa")

    _condition_scatter(
        distance_ax,
        metrics,
        "peak_distance_median_m",
        "Mediana odległości piku [m]",
        ylim=(1.72, 2.10),
    )
    distance_ax.axhline(2.0, color="#555555", linestyle=":", linewidth=1.1)
    distance_ax.set_title("Położenie dominującego echa")

    figure.suptitle("A121, 2 m, soczewka hiperboliczna — echo zależy głównie od geometrii")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _shade_shifted_cues(
    ax: plt.Axes,
    cues: list[dict[str, Any]],
    delay_s: float,
) -> None:
    colours = {"inhale": "#56B4E9", "exhale": "#E69F00", "hold": "#777777"}
    alpha = {"inhale": 0.095, "exhale": 0.075, "hold": 0.13}
    for cue in cues:
        kind = str(cue.get("kind"))
        if kind not in colours:
            continue
        ax.axvspan(
            float(cue["start_s"]) + delay_s,
            float(cue["end_s"]) + delay_s,
            color=colours[kind],
            alpha=alpha[kind],
            linewidth=0,
        )


def plot_aligned_traces(
    results: list[RecordingResult],
    cues: list[dict[str, Any]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        4,
        3,
        figsize=(15.8, 11.6),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.065, top=0.88, hspace=0.19, wspace=0.05)
    absolute_quantiles = [
        float(np.quantile(np.abs(result.displacement_mm), 0.995)) for result in results
    ]
    common_limit = max(0.15, max(absolute_quantiles) * 1.08)
    by_key = {(result.condition, result.repeat): result for result in results}
    for row, condition in enumerate(CONDITION_ORDER):
        for column, repeat in enumerate((1, 2, 3)):
            ax = axes[row, column]
            result = by_key[(condition, repeat)]
            delay = float(result.metrics["cue_delay_s"])
            correlation = float(result.metrics["template_correlation"])
            _shade_shifted_cues(ax, cues, delay)
            ax.plot(
                result.time_s,
                result.displacement_mm,
                color=CONDITION_COLOURS[condition],
                linewidth=1.25,
            )
            ax.axhline(0.0, color="#777777", linewidth=0.5)
            ax.set_xlim(0.0, 90.0)
            ax.set_ylim(-common_limit, common_limit)
            ax.set_title(
                f"powt. {repeat}, run {result.run}  ·  δ={_polish_number(delay, 3)} s"
                f"  ·  r={_polish_number(correlation, 3)}",
                fontsize=9.5,
            )
            if column == 0:
                ax.set_ylabel(CONDITION_LONG_LABELS[condition] + "\nprzemieszczenie [mm]")
            if row == len(CONDITION_ORDER) - 1:
                ax.set_xlabel("Czas od początku zapisu [s]")
            _style_axes(ax)

    legend = [
        Patch(facecolor="#56B4E9", alpha=0.28, label="wdech (znaczniki przesunięte o δ)"),
        Patch(facecolor="#E69F00", alpha=0.24, label="wydech"),
        Patch(facecolor="#777777", alpha=0.25, label="wstrzymanie oddechu"),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.943),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Wszystkie 12 przebiegów fazowych przy 2 m z indywidualnie skorygowaną osią komend",
        y=0.985,
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_signal_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.7), constrained_layout=True)
    _condition_scatter(
        axes[0, 0],
        metrics,
        "template_correlation",
        "Korelacja z komendami r",
        ylim=(0.88, 0.98),
    )
    axes[0, 0].set_title("Zgodność kształtu z zadanym cyklem")
    _condition_scatter(
        axes[0, 1],
        metrics,
        "cue_delay_s",
        "Łączne opóźnienie δ [s]",
        ylim=(0.20, 0.70),
    )
    axes[0, 1].set_title("Komenda → ruch widoczny w zapisie")
    _condition_scatter(
        axes[1, 0],
        metrics,
        "breath_amplitude_mm",
        "Amplituda oddechowa [mm]",
    )
    axes[1, 0].set_title("Połowa zakresu 5.–95. percentyla")
    _condition_scatter(
        axes[1, 1],
        metrics,
        "hold_vs_paced_db",
        "RMS hold / RMS oddechu [dB]",
    )
    axes[1, 1].axhline(0.0, color="#555555", linewidth=0.9)
    axes[1, 1].set_title("Zanik ruchu podczas wstrzymania (niżej = lepiej)")
    figure.suptitle(
        "A121 przy 2 m — jakość śledzenia oddechu (punkty: próby, kreski: mediany)"
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def paired_effects(metrics: pd.DataFrame, column: str) -> dict[str, np.ndarray]:
    indexed = metrics.set_index(["condition", "repeat"])

    def values(numerator: str, denominator: str, *, db: bool) -> np.ndarray:
        result: list[float] = []
        for repeat in (1, 2, 3):
            high = float(indexed.loc[(numerator, repeat), column])
            low = float(indexed.loc[(denominator, repeat), column])
            result.append(20.0 * np.log10(high / low) if db else high - low)
        return np.asarray(result)

    return {
        "angle_no_foil": values("P0", "N0", db=True),
        "angle_foil": values("PF", "NF", db=True),
        "foil_natural": values("NF", "N0", db=True),
        "foil_perpendicular": values("PF", "P0", db=True),
    }


def plot_paired_echo_effects(metrics: pd.DataFrame, output_path: Path) -> None:
    effects = paired_effects(metrics, "echo_peak_median")
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    panels = (
        (
            axes[0],
            ("angle_no_foil", "angle_foil"),
            ("bez folii", "z folią"),
            "Wpływ ustawienia: prostopadłe − naturalne",
        ),
        (
            axes[1],
            ("foil_natural", "foil_perpendicular"),
            ("naturalne", "prostopadłe"),
            "Wpływ folii: folia − brak folii",
        ),
    )
    colours = ("#0072B2", "#D55E00")
    offsets = (-0.10, 0.0, 0.10)
    for panel_index, (ax, keys, labels, title) in enumerate(panels):
        for x, (key, label, colour) in enumerate(zip(keys, labels, colours, strict=True)):
            values = effects[key]
            for repeat, value in enumerate(values):
                ax.scatter(x + offsets[repeat], value, color=colour, s=52, zorder=3)
            median = float(np.median(values))
            ax.plot([x - 0.22, x + 0.22], [median, median], color="#333333", linewidth=2.3)
            ax.annotate(
                f"med. {_polish_number(median, 1)} dB",
                xy=(x, median),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )
        ax.axhline(0.0, color="#555555", linewidth=1.0)
        ax.set_xticks((0, 1), labels)
        ax.set_ylabel("Parowana zmiana amplitudy [dB]")
        ax.set_title(title)
        ax.set_ylim((-0.7, 15.8) if panel_index == 0 else (-3.2, 2.8))
        _style_axes(ax)
    figure.suptitle("Efekty parowane po numerze powtórzenia (n = 3 na porównanie)")
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def build_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    fields = (
        "echo_peak_median",
        "peak_distance_median_m",
        "cue_delay_s",
        "template_correlation",
        "inhale_start_delay_s",
        "inhale_end_delay_s",
        "extrema_bpm",
        "breath_amplitude_mm",
        "hold_vs_paced_db",
        "hold_velocity_vs_paced_db",
        "acconeer_bpm_median",
    )
    groups: dict[str, Any] = {}
    for condition in CONDITION_ORDER:
        subset = metrics[metrics["condition"] == condition]
        groups[condition] = {
            field: {
                "median": float(subset[field].median()),
                "minimum": float(subset[field].min()),
                "maximum": float(subset[field].max()),
            }
            for field in fields
        }
    effects = paired_effects(metrics, "echo_peak_median")
    return {
        "definitions": {
            "cue_delay_s": (
                "Aggregate command-to-recorded-motion delay estimated by maximising "
                "correlation with the 2 s inhale / 3 s exhale template."
            ),
            "template_correlation": "Pearson r after per-recording delay correction.",
            "breath_amplitude_mm": (
                "Half of the 5th-to-95th percentile range of the 0.10-0.50 Hz "
                "phase-derived displacement during paced breathing."
            ),
            "hold_vs_paced_db": (
                "20 log10 of respiration-band RMS in the central 9 s of the hold "
                "relative to paced-breathing RMS."
            ),
        },
        "overall": {
            "recordings": int(len(metrics)),
            "cue_delay_median_s": float(metrics["cue_delay_s"].median()),
            "cue_delay_min_s": float(metrics["cue_delay_s"].min()),
            "cue_delay_max_s": float(metrics["cue_delay_s"].max()),
            "template_correlation_min": float(metrics["template_correlation"].min()),
            "template_correlation_max": float(metrics["template_correlation"].max()),
            "matched_inhale_starts": int(metrics["matched_inhale_starts"].sum()),
            "matched_inhale_ends": int(metrics["matched_inhale_ends"].sum()),
        },
        "condition_groups": groups,
        "paired_echo_effects_db": {
            key: {
                "values": values.tolist(),
                "median": float(np.median(values)),
            }
            for key, values in effects.items()
        },
    }


def print_summary(metrics: pd.DataFrame, summary: dict[str, Any]) -> None:
    columns = [
        "run",
        "condition",
        "repeat",
        "echo_peak_median",
        "peak_distance_median_m",
        "cue_delay_s",
        "template_correlation",
        "breath_amplitude_mm",
        "hold_vs_paced_db",
        "extrema_bpm",
        "acconeer_bpm_median",
    ]
    print(metrics[columns].sort_values("run").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nCondition medians:")
    print(
        metrics.groupby("condition", sort=False)[
            [
                "echo_peak_median",
                "peak_distance_median_m",
                "cue_delay_s",
                "template_correlation",
                "breath_amplitude_mm",
                "hold_vs_paced_db",
            ]
        ]
        .median()
        .reindex(CONDITION_ORDER)
        .to_string(float_format=lambda x: f"{x:.3f}")
    )
    print("\nPaired echo effects [dB]:")
    for key, values in summary["paired_echo_effects_db"].items():
        printable = ", ".join(f"{value:.2f}" for value in values["values"])
        print(f"  {key}: [{printable}], median={values['median']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        help="Guided session directory (default: latest guided_foil2m_* session).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Figure output directory (default: {DEFAULT_FIGURE_DIR}).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only write metrics and summary files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = (args.session or _latest_session()).resolve()
    manifest_path = session / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cues = list(manifest.get("breathing_cues", []))
    accepted = sorted(
        [item for item in manifest.get("measurements", []) if item.get("status") == "accepted"],
        key=lambda item: int(item["step"]),
    )
    if len(accepted) != 12:
        raise ValueError(f"Expected 12 accepted recordings, found {len(accepted)} in {manifest_path}")
    condition_counts = pd.Series([item["condition_id"] for item in accepted]).value_counts()
    if any(int(condition_counts.get(condition, 0)) != 3 for condition in CONDITION_ORDER):
        raise ValueError(f"Expected three recordings per condition, found {condition_counts.to_dict()}")

    results: list[RecordingResult] = []
    for metadata in accepted:
        print(
            f"Analysing run {int(metadata['step']):02d}: "
            f"{metadata['condition_id']} repeat {metadata['repeat_number']}...",
            flush=True,
        )
        results.append(analyse_recording(metadata, cues))

    metrics = pd.DataFrame([result.metrics for result in results]).sort_values("run")
    summary = build_summary(metrics)
    metrics_path = session / "analysis_2m_metrics.csv"
    summary_path = session / "analysis_2m_summary.json"
    metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.no_plots:
        figure_dir = args.figure_dir.resolve()
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_range_profiles(results, metrics, figure_dir / "foil2m_profile_i_echo.png")
        plot_aligned_traces(results, cues, figure_dir / "foil2m_przebiegi_fazy.png")
        plot_signal_metrics(metrics, figure_dir / "foil2m_metryki_sygnalu.png")
        plot_paired_echo_effects(metrics, figure_dir / "foil2m_efekty_parowane.png")

    print_summary(metrics, summary)
    print(f"\nMetrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    if not args.no_plots:
        print(f"Figures: {args.figure_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
