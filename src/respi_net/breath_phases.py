"""Breathing-phase classes shared by labels, datasets, baselines and models.

The classes and their numbers follow the respiratory-phase datasets of
Szymański et al. (Scientific Data, 2025, Gdańsk University of Technology), so
labels exported from this project can sit next to that data:

    0   exhale                (breath-out)
    1   hold after exhale     (retention after breath-out)
    2   inhale                (breath-in)
    3   hold after inhale     (retention after breath-in)
    999 noise                 (too contaminated to use)

Inside arrays and models noise is class 4, so the classes are 0-4 without a
gap, and :data:`IGNORE` (-1) marks samples that carry no label at all: stretches
nobody labelled, and the uncertain slice around every phase boundary.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

EXHALE = 0
HOLD_AFTER_EXHALE = 1
INHALE = 2
HOLD_AFTER_INHALE = 3
NOISE = 4
IGNORE = -1

NUM_CLASSES = 5
PHASE_CLASSES = (EXHALE, HOLD_AFTER_EXHALE, INHALE, HOLD_AFTER_INHALE)
CLASS_NAMES = ("exhale", "hold_after_exhale", "inhale", "hold_after_inhale", "noise")
CLASS_NAMES_PL = ("wydech", "pauza po wydechu", "wdech", "pauza po wdechu", "szum")
CLASS_COLOURS = ("#22c55e", "#f59e0b", "#3b82f6", "#dc2626", "#9ca3af")

# The merged scheme: one "hold" whatever lung volume it happens at.  A phone on the ribs, which returns to its
# baseline after each breath, cannot tell the two holds apart; the rest of the pipeline can keep five classes.
HOLD = HOLD_AFTER_EXHALE
MERGED_CLASS_NAMES_PL = ("wydech", "pauza", "wdech")


def merge_holds(labels: np.ndarray) -> np.ndarray:
    """``labels`` with hold-after-inhale turned into the single hold class (:data:`HOLD`); the rest is unchanged."""

    out = np.array(labels, copy=True)
    out[out == HOLD_AFTER_INHALE] = HOLD
    return out


NOISE_EXPORT_CODE = 999
IGNORE_EXPORT_CODE = -1

# What may follow what.  A hold is named after the breath it interrupts, so it
# can only be entered from that breath -- exhale straight into "hold after
# inhale" is impossible by definition.  It can be left either way: a breath
# can be held at any lung volume, and a breath held halfway out is often
# finished before the next inhale.  Noise can interrupt and resume anything.
ALLOWED_TRANSITIONS: dict[int, frozenset[int]] = {
    INHALE: frozenset({EXHALE, HOLD_AFTER_INHALE, NOISE}),
    HOLD_AFTER_INHALE: frozenset({EXHALE, INHALE, NOISE}),
    EXHALE: frozenset({INHALE, HOLD_AFTER_EXHALE, NOISE}),
    HOLD_AFTER_EXHALE: frozenset({INHALE, EXHALE, NOISE}),
    NOISE: frozenset({INHALE, HOLD_AFTER_EXHALE, EXHALE, HOLD_AFTER_INHALE}),
}


def export_code(label: int) -> int:
    """The number written to label files: Szymański's codes, 999 for noise."""

    if label == NOISE:
        return NOISE_EXPORT_CODE
    if label == IGNORE:
        return IGNORE_EXPORT_CODE
    if label in PHASE_CLASSES:
        return int(label)
    raise ValueError(f"Unknown breathing-phase label {label!r}")


def from_export_code(code: int) -> int:
    if code == NOISE_EXPORT_CODE:
        return NOISE
    if code == IGNORE_EXPORT_CODE:
        return IGNORE
    if code in PHASE_CLASSES:
        return int(code)
    raise ValueError(f"Unknown breathing-phase code {code!r}")


def export_codes(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    codes = labels.astype(np.int32, copy=True)
    codes[labels == NOISE] = NOISE_EXPORT_CODE
    return codes


def hold_class_after(previous_breath: int | None) -> int:
    """A hold keeps the lungs where the last breath left them.

    With no breath before it (a hold straight after free breathing) the hold is
    taken as one after exhaling, which is how every protocol here asks for it.
    """

    return HOLD_AFTER_INHALE if previous_breath == INHALE else HOLD_AFTER_EXHALE


def classes_for_cue_kinds(kinds: Sequence[str]) -> list[int]:
    """Phase class for each coach cue kind; free breathing and settling get IGNORE."""

    classes: list[int] = []
    last_breath: int | None = None
    for kind in kinds:
        if kind == "inhale":
            label = last_breath = INHALE
        elif kind == "exhale":
            label = last_breath = EXHALE
        elif kind == "hold":
            label = hold_class_after(last_breath)
        else:
            # normal, settle, interference: the cue prescribes no phase.
            label = IGNORE
            last_breath = None
        classes.append(label)
    return classes


def labels_from_segments(
    time_s: np.ndarray,
    segments: Iterable[tuple[float, float, int]],
) -> np.ndarray:
    """Per-sample labels from ``(start_s, end_s, class)`` segments; IGNORE elsewhere."""

    time_s = np.asarray(time_s, dtype=float)
    labels = np.full(len(time_s), IGNORE, dtype=np.int8)
    for start_s, end_s, label in segments:
        labels[(time_s >= start_s) & (time_s < end_s)] = label
    return labels


def mask_around(
    labels: np.ndarray,
    time_s: np.ndarray,
    moments_s: Iterable[float],
    half_width_s: float | Iterable[float],
) -> np.ndarray:
    """Copy of ``labels`` with IGNORE within ``half_width_s`` of every moment."""

    labels = np.array(labels, copy=True)
    time_s = np.asarray(time_s, dtype=float)
    moments = list(moments_s)
    widths = (
        [float(half_width_s)] * len(moments)
        if isinstance(half_width_s, (int, float))
        else [float(width) for width in half_width_s]
    )
    for moment, width in zip(moments, widths, strict=True):
        labels[np.abs(time_s - moment) <= width] = IGNORE
    return labels


def label_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length encoding: ``(start_index, stop_index, label)`` for each run."""

    labels = np.asarray(labels)
    if not len(labels):
        return []
    change = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.r_[0, change]
    stops = np.r_[change, len(labels)]
    return [(int(start), int(stop), int(labels[start])) for start, stop in zip(starts, stops)]


def transitions_allowed(labels: np.ndarray) -> bool:
    """Whether a label sequence only ever steps along :data:`ALLOWED_TRANSITIONS`.

    IGNORE samples are skipped over, so a masked boundary does not count as a
    transition of its own.
    """

    previous: int | None = None
    for _, _, label in label_runs(labels):
        if label == IGNORE:
            continue
        if previous is not None and label != previous and label not in ALLOWED_TRANSITIONS[previous]:
            return False
        previous = label
    return True
