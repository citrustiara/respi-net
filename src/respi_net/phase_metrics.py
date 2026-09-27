"""Scoring breathing-phase predictions, and turning per-sample scores into phases.

Every score skips samples labelled IGNORE, so masked boundaries and unlabelled
stretches neither help nor hurt a model.  Boundary scores match each reference
phase change to the nearest predicted change of the same kind, because a
model that is right everywhere except 300 ms late at every turn is a
different model from one that is right on average but flickers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .breath_phases import ALLOWED_TRANSITIONS, CLASS_NAMES, IGNORE, INHALE, NUM_CLASSES, label_runs


@dataclass(frozen=True)
class PhaseScores:
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: np.ndarray
    samples: int


@dataclass(frozen=True)
class BoundaryScores:
    reference_boundaries: int
    matched: int
    detection_rate: float
    median_abs_error_s: float
    mean_abs_error_s: float
    false_per_minute: float
    # Median of (predicted - reference) over matched changes: how late a
    # detector announces a change, which a real-time one always does.
    median_delay_s: float = float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Rows are the reference class, columns the prediction; IGNORE rows are skipped."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    keep = (y_true != IGNORE) & (y_pred >= 0) & (y_pred < num_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true[keep], y_pred[keep]), 1)
    return matrix


def score_phases(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> PhaseScores:
    """Accuracy, per-class precision/recall/F1 and macro-F1 over the classes present."""

    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = int(matrix.sum())
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in range(num_classes):
        support = int(matrix[label].sum())
        predicted = int(matrix[:, label].sum())
        hits = int(matrix[label, label])
        precision = hits / predicted if predicted else float("nan")
        recall = hits / support if support else float("nan")
        if support and predicted and hits:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if support else float("nan")
        per_class[CLASS_NAMES[label]] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1_values.append(f1)
    accuracy = float(np.trace(matrix) / total) if total else float("nan")
    macro_f1 = float(np.mean(f1_values)) if f1_values else float("nan")
    return PhaseScores(accuracy, macro_f1, per_class, matrix, total)


def phase_boundaries(labels: np.ndarray, time_s: np.ndarray) -> list[tuple[float, int, int]]:
    """``(time_s, class_before, class_after)`` for every change between labelled classes.

    A change that happens across an IGNORE gap is placed in the middle of the
    gap, which is where a masked boundary was before it was masked.
    """

    time_s = np.asarray(time_s, dtype=float)
    changes: list[tuple[float, int, int]] = []
    previous: tuple[int, int] | None = None  # (label, stop index of its run)
    for start, stop, label in label_runs(np.asarray(labels)):
        if label == IGNORE:
            continue
        if previous is not None and previous[0] != label:
            gap_start = time_s[previous[1] - 1]
            gap_end = time_s[start]
            if previous[1] == start:
                moment = 0.5 * (time_s[start - 1] + time_s[start])
            else:
                moment = 0.5 * (gap_start + gap_end)
            changes.append((float(moment), previous[0], label))
        previous = (label, stop)
    return changes


def boundary_errors(
    reference: np.ndarray,
    predicted: np.ndarray,
    time_s: np.ndarray,
    *,
    tolerance_s: float = 1.0,
) -> BoundaryScores:
    """Timing of predicted phase changes against the reference ones of the same kind."""

    time_s = np.asarray(time_s, dtype=float)
    ref = phase_boundaries(reference, time_s)
    pred = phase_boundaries(predicted, time_s)
    used = np.zeros(len(pred), dtype=bool)
    errors: list[float] = []
    delays: list[float] = []
    for moment, before, after in ref:
        best, best_error = -1, math.inf
        for index, (candidate, cand_before, cand_after) in enumerate(pred):
            if used[index] or cand_before != before or cand_after != after:
                continue
            error = abs(candidate - moment)
            if error < best_error:
                best, best_error = index, error
        if best >= 0 and best_error <= tolerance_s:
            used[best] = True
            errors.append(best_error)
            delays.append(pred[best][0] - moment)
    duration_min = (time_s[-1] - time_s[0]) / 60.0 if len(time_s) > 1 else float("nan")
    # Only predicted changes inside the labelled span can be false: outside it
    # there is nothing to compare against.
    labelled = np.flatnonzero(np.asarray(reference) != IGNORE)
    if len(labelled):
        span = (time_s[labelled[0]], time_s[labelled[-1]])
        extra = sum(1 for index, (moment, _, _) in enumerate(pred) if not used[index] and span[0] <= moment <= span[1])
    else:
        extra = 0
    return BoundaryScores(
        reference_boundaries=len(ref),
        matched=len(errors),
        detection_rate=len(errors) / len(ref) if ref else float("nan"),
        median_abs_error_s=float(np.median(errors)) if errors else float("nan"),
        mean_abs_error_s=float(np.mean(errors)) if errors else float("nan"),
        false_per_minute=extra / duration_min if duration_min and duration_min > 0 else float("nan"),
        median_delay_s=float(np.median(delays)) if delays else float("nan"),
    )


def breath_rate_bpm(labels: np.ndarray, time_s: np.ndarray) -> float:
    """Breaths per minute from the inhale onsets in ``labels``."""

    onsets = [moment for moment, _, after in phase_boundaries(labels, time_s) if after == INHALE]
    if len(onsets) < 2:
        return float("nan")
    return 60.0 * (len(onsets) - 1) / (onsets[-1] - onsets[0])


def transition_log_matrix(
    switch_log_prob: float = math.log(0.02),
    allowed: Mapping[int, frozenset[int]] = ALLOWED_TRANSITIONS,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Log transition scores: 0 to stay, ``switch_log_prob`` to an allowed class, -inf otherwise."""

    matrix = np.full((num_classes, num_classes), -np.inf)
    for source in range(num_classes):
        matrix[source, source] = 0.0
        for target in allowed.get(source, frozenset()):
            matrix[source, target] = switch_log_prob
    return matrix


def viterbi_decode(log_probs: np.ndarray, transitions: np.ndarray | None = None) -> np.ndarray:
    """Most likely class sequence for per-sample log-probabilities ``[T, C]``.

    Staying costs nothing, switching costs a penalty and a disallowed switch
    (exhale straight into a hold after inhale) is impossible, which is what
    stops a network's output flickering at every uncertain sample.
    """

    log_probs = np.asarray(log_probs, dtype=float)
    steps, classes = log_probs.shape
    if transitions is None:
        transitions = transition_log_matrix(num_classes=classes)
    score = log_probs[0].copy()
    back = np.zeros((steps, classes), dtype=np.int64)
    for step in range(1, steps):
        candidates = score[:, None] + transitions
        back[step] = np.argmax(candidates, axis=0)
        score = candidates[back[step], np.arange(classes)] + log_probs[step]
    path = np.empty(steps, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for step in range(steps - 1, 0, -1):
        path[step - 1] = back[step, path[step]]
    return path


def enforce_min_duration(labels: np.ndarray, fs: float, min_duration_s: float) -> np.ndarray:
    """Absorb runs shorter than ``min_duration_s`` into the run before them.

    The first run, having nothing before it, is absorbed into the run after it.
    IGNORE runs are left alone.
    """

    labels = np.array(labels, copy=True)
    min_samples = max(1, int(round(min_duration_s * fs)))
    changed = True
    while changed:
        changed = False
        runs = label_runs(labels)
        for index, (start, stop, label) in enumerate(runs):
            if label == IGNORE or stop - start >= min_samples or len(runs) == 1:
                continue
            neighbour = runs[index - 1][2] if index > 0 else runs[index + 1][2]
            if neighbour == IGNORE:
                continue
            labels[start:stop] = neighbour
            changed = True
            break
    return labels
