"""Hand annotation of breathing phases on a phone IMU trace, and how such labels are compared.

The phone's lower-rib motion is the reference for the radar network.  A person marks the phase changes on the
phone trace alone (no radar, no detector output); the marks become segments of four classes -- exhale, hold,
inhale, noise -- saved as a small CSV.  The same file format serves three jobs: the reference itself, a second
pass over the same minute (how far two passes of one person differ is the reference's own uncertainty), and the
validation of the deterministic detector against the reference.

Classes use the merged scheme of :mod:`respi_net.breath_phases`: 0 exhale, 1 hold, 2 inhale, 4 noise, -1 no label.
A hold is a stillness of at least :data:`MIN_HOLD_S`; shorter pauses belong to the breath around them.  In the phone
traces recorded so far the natural rests between breaths last 0.8-3.1 s and deliberate holds 4.4-15 s, so the rule sits in that gap.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .breath_phases import EXHALE, HOLD, IGNORE, INHALE, NOISE
from .breath_phases import label_runs
from .chest_signal import a121_chest_signal, imu_chest_signal, light_filter
from .phase_metrics import BoundaryScores, PhaseScores, boundary_errors, score_phases

MIN_HOLD_S = 4.0
GUARD_S = 0.2
HOLD_GUARD_S = 2.5  # where a hold begins is fuzzy by 2-4 s on the phone (the slow slide after a deep breath): not scored

CLASS_NAMES = {EXHALE: "exhale", HOLD: "hold", INHALE: "inhale", NOISE: "noise"}
CLASS_BY_NAME = {name: code for code, name in CLASS_NAMES.items()}
KEY_TO_CLASS = {"e": EXHALE, "h": HOLD, "i": INHALE, "n": NOISE}


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    label: int


def next_label(history: Sequence[int]) -> int:
    """The class that most likely follows ``history`` (the classes marked so far): breaths alternate."""

    breaths = [label for label in history if label in (EXHALE, INHALE)]
    if not history:
        return INHALE
    last = history[-1]
    if last == INHALE:
        return EXHALE
    if last == EXHALE:
        return INHALE
    # after a hold or noise: the opposite of the breath that came before it
    if breaths:
        return EXHALE if breaths[-1] == INHALE else INHALE
    return INHALE


def marks_to_segments(marks: Sequence[tuple[float, int]], end_s: float) -> list[Segment]:
    """Segments from ``(start time, class)`` marks; the last one ends at ``end_s``."""

    ordered = sorted(marks)
    segments = []
    for index, (start, label) in enumerate(ordered):
        stop = ordered[index + 1][0] if index + 1 < len(ordered) else end_s
        if stop > start:
            segments.append(Segment(float(start), float(stop), int(label)))
    return segments


def segments_to_labels(segments: Sequence[Segment], grid: np.ndarray) -> np.ndarray:
    """Per-sample classes on ``grid``; samples outside every segment are :data:`IGNORE`."""

    labels = np.full(len(grid), IGNORE, dtype=np.int8)
    for segment in segments:
        labels[(grid >= segment.start_s) & (grid < segment.end_s)] = segment.label
    return labels


def save_segments(path: Path, segments: Sequence[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_s", "end_s", "class"])
        for segment in segments:
            writer.writerow([f"{segment.start_s:.3f}", f"{segment.end_s:.3f}", CLASS_NAMES[segment.label]])


def load_segments(path: Path) -> list[Segment]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [Segment(float(row["start_s"]), float(row["end_s"]), CLASS_BY_NAME[row["class"]]) for row in csv.DictReader(handle)]


def guard_mask(labels: np.ndarray, grid: np.ndarray, guard_s: float = GUARD_S, hold_guard_s: float = HOLD_GUARD_S) -> np.ndarray:
    """``labels`` with the samples around a class change set to :data:`IGNORE` (timing is scored apart).

    ``guard_s`` on each side of a change; ``hold_guard_s`` where the change enters a hold, because the start of a
    hold is fuzzy by a second or two (a slow settle follows a deep breath).
    """

    out = np.array(labels, copy=True)
    change = np.flatnonzero(np.diff(labels.astype(int)) != 0) + 1
    step = float(np.median(np.diff(grid))) if len(grid) > 1 else 1.0
    for index in change:
        entering_hold = labels[index] == HOLD and labels[index - 1] != IGNORE
        reach = int(round((max(guard_s, hold_guard_s) if entering_hold else guard_s) / step))
        out[max(0, index - reach) : index + reach] = IGNORE
    return out


def compare_labels(
    reference: np.ndarray,
    other: np.ndarray,
    grid: np.ndarray,
    *,
    guard_s: float = GUARD_S,
    hold_guard_s: float = HOLD_GUARD_S,
    tolerance_s: float = 1.0,
) -> dict[str, float]:
    """Sample agreement away from the boundaries, and boundary timing, of ``other`` against ``reference``."""

    both = (reference != IGNORE) & (other != IGNORE)
    ref = np.where(both, reference, IGNORE)
    oth = np.where(both, other, IGNORE)
    phases: PhaseScores = score_phases(guard_mask(ref, grid, guard_s, hold_guard_s), oth)
    bounds: BoundaryScores = boundary_errors(ref, oth, grid, tolerance_s=tolerance_s)
    return {
        "accuracy": float(phases.accuracy),
        "macro_f1": float(phases.macro_f1),
        "samples": int(phases.samples),
        "boundaries": int(bounds.reference_boundaries),
        "boundaries_found": float(bounds.detection_rate),
        "boundary_median_abs_error_s": float(bounds.median_abs_error_s),
        "boundary_median_delay_s": float(bounds.median_delay_s),
        "false_boundaries_per_min": float(bounds.false_per_minute),
    }


def oriented_phone(run_prefix: Path, grid_hz: float = 20.0) -> tuple[np.ndarray, np.ndarray, float]:
    """``(grid, phone trace rising on inhale, sign)`` of one recorder run.

    The sign of the phone's main axis is arbitrary; it is taken from the correlation of the velocities with the
    radar's (one number, so the annotator still never sees the radar).  Falls back to +1 without a radar file.
    """

    import pandas as pd  # local: only the viewer and the scorer need it

    cues = pd.read_csv(f"{run_prefix}_cues.csv")
    origin = float(cues["start_wall_ms"].iloc[0])
    duration = float(cues["end_s"].iloc[-1])
    phone = imu_chest_signal(f"{run_prefix}_iphone.csv", origin_ms=origin, name="iphone")
    grid = np.arange(0.0, duration, 1.0 / grid_hz)
    trace = np.interp(grid, phone.time_s, phone.chest)
    sign = 1.0
    radar_path = Path(f"{run_prefix}_a121.csv")
    if radar_path.exists():
        radar = a121_chest_signal(radar_path, origin_ms=origin)
        radar_trace = np.interp(grid, radar.time_s, radar.chest)
        body = grid >= float(cues["end_s"].iloc[0])
        velocity_r, velocity_p = np.gradient(radar_trace)[body], np.gradient(trace)[body]
        best, sign = 0.0, 1.0
        for shift in range(-int(grid_hz), int(grid_hz) + 1):
            a, b = (velocity_r[: len(velocity_r) - shift], velocity_p[shift:]) if shift >= 0 else (velocity_r[-shift:], velocity_p[: len(velocity_p) + shift])
            if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
                if abs(corr) > abs(best):
                    best = corr
        sign = 1.0 if best >= 0 else -1.0
    return grid, sign * trace, sign


def snap_segments(
    segments: Sequence[Segment], grid: np.ndarray, trace: np.ndarray, *, reach_s: float = 0.8, low_pass_hz: float = 1.0
) -> list[Segment]:
    """Move every inhale/exhale boundary to the nearest turning point of ``trace``.

    A click lands a little after the visible turn (about 0.2 s) and a detector may put the boundary at the start of
    the short rest at the bottom; the turning point -- the crest for inhale->exhale, the trough for exhale->inhale --
    is the one convention both can share.  Boundaries that touch a hold or noise stay where they are.
    """

    if len(segments) < 2:
        return list(segments)
    smooth = light_filter(np.asarray(trace, dtype=float), 1.0 / float(np.median(np.diff(grid))), low_pass_hz)
    starts = [segment.start_s for segment in segments]
    ends = [segment.end_s for segment in segments]
    for index in range(len(segments) - 1):
        before, after = segments[index].label, segments[index + 1].label
        if abs(ends[index] - starts[index + 1]) > 1e-6 or (before, after) not in ((INHALE, EXHALE), (EXHALE, INHALE)):
            continue
        window = (grid >= ends[index] - reach_s) & (grid <= ends[index] + reach_s)
        if window.sum() < 3:
            continue
        times = grid[window]
        moment = float(times[np.argmax(smooth[window]) if before == INHALE else np.argmin(smooth[window])])
        if starts[index] + 0.3 < moment < ends[index + 1] - 0.3:  # never squeeze a neighbour away
            ends[index] = moment
            starts[index + 1] = moment
    return [Segment(a, b, segment.label) for a, b, segment in zip(starts, ends, segments)]


def snap_labels(labels: np.ndarray, grid: np.ndarray, trace: np.ndarray, **kwargs: float) -> np.ndarray:
    """:func:`snap_segments` for a per-sample label array (the detector's output)."""

    segments = [
        Segment(float(grid[start]), float(grid[stop]) if stop < len(grid) else float(grid[-1] + (grid[1] - grid[0])), int(label))
        for start, stop, label in label_runs(labels)
        if label != IGNORE
    ]
    return segments_to_labels(snap_segments(segments, grid, trace, **kwargs), grid)
