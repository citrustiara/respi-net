"""Coach cues -> per-breath phase labels on a measured chest signal.

A cue is an instruction, not a measurement.  People start each breath a few
hundred milliseconds after it, and not by the same amount every time: on the
coach recordings one global lag still left boundaries 0.29 s off (IQR) and
about 7.6 % of paced samples on the wrong side of a boundary.  So:

1. one global lag, from correlating the signal with the shape the cues ask for;
2. each boundary is moved to where the measured motion actually *turns* (a
   breath reversing), *stops* (a breath running into a hold) or *starts* (a
   hold ending), searched in a window around the lag-shifted cue;
3. order and a minimum phase length are kept; a boundary with no clear event
   falls back to the lag-shifted cue and gets a wider mask;
4. labels are written per sample, with IGNORE around every boundary, because
   the exact sample a breath turns on is uncertain and should not be taught.

Free breathing and settling carry no cued phase and stay IGNORE.  The signal
may be the radar's own (fine for clean, deep coach runs) or an independent
reference such as a chest IMU, which is what hard data needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .breath_phases import (
    EXHALE,
    HOLD_AFTER_EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    classes_for_cue_kinds,
    labels_from_segments,
    mask_around,
)

SNAPPED_METHODS = ("turn", "stop", "start")


@dataclass(frozen=True)
class CuePhase:
    start_s: float
    end_s: float
    kind: str
    label: int


@dataclass(frozen=True)
class Boundary:
    cue_s: float
    expected_s: float
    time_s: float
    before: int
    after: int
    method: str

    @property
    def shift_s(self) -> float:
        return self.time_s - self.expected_s


@dataclass(frozen=True)
class AlignmentResult:
    lag_s: float
    correlation: float
    sign: float
    boundaries: tuple[Boundary, ...]
    segments: tuple[tuple[float, float, int], ...]
    labels: np.ndarray
    time_s: np.ndarray

    def summary(self) -> dict[str, float]:
        snappable = [b for b in self.boundaries if b.method != "end"]
        snapped = [b.shift_s for b in snappable if b.method in SNAPPED_METHODS]
        shifts = np.asarray(snapped, dtype=float)
        labelled = self.labels != IGNORE
        return {
            "lag_s": self.lag_s,
            "correlation": self.correlation,
            "sign": self.sign,
            "boundaries": float(len(snappable)),
            "fallback_share": float(1.0 - len(snapped) / len(snappable)) if snappable else float("nan"),
            "shift_median_s": float(np.median(shifts)) if len(shifts) else float("nan"),
            "shift_iqr_s": float(np.subtract(*np.percentile(shifts, [75, 25]))) if len(shifts) else float("nan"),
            "shift_abs_p90_s": float(np.percentile(np.abs(shifts), 90)) if len(shifts) else float("nan"),
            "labelled_share": float(np.mean(labelled)) if len(labelled) else float("nan"),
        }


def _field(cue: Any, name: str) -> Any:
    return cue[name] if isinstance(cue, Mapping) else getattr(cue, name)


def cue_phases(cues: Iterable[Any]) -> list[CuePhase]:
    """Cue records (CSV rows as dicts, or BreathPhase objects) with their phase class."""

    records = list(cues)
    labels = classes_for_cue_kinds([str(_field(cue, "kind")) for cue in records])
    return [
        CuePhase(float(_field(cue, "start_s")), float(_field(cue, "end_s")), str(_field(cue, "kind")), int(label))
        for cue, label in zip(records, labels, strict=True)
    ]


def expected_chest(time_s: np.ndarray, phases: Sequence[CuePhase], lag_s: float) -> np.ndarray:
    """The chest shape the cues ask for: 0 = empty, 1 = full; NaN where no phase is cued."""

    time_s = np.asarray(time_s, dtype=float)
    template = np.full(len(time_s), np.nan)
    for phase in phases:
        start, end = phase.start_s + lag_s, phase.end_s + lag_s
        inside = (time_s >= start) & (time_s < end)
        progress = (time_s[inside] - start) / max(end - start, 1e-9)
        if phase.label == INHALE:
            template[inside] = progress
        elif phase.label == EXHALE:
            template[inside] = 1.0 - progress
        elif phase.label == HOLD_AFTER_INHALE:
            template[inside] = 1.0
        elif phase.label == HOLD_AFTER_EXHALE:
            template[inside] = 0.0
    return template


def expected_velocity(time_s: np.ndarray, phases: Sequence[CuePhase], lag_s: float) -> np.ndarray:
    """How the cues ask the chest to move: +1/duration on inhale, -1/duration on exhale, 0 in holds."""

    time_s = np.asarray(time_s, dtype=float)
    template = np.full(len(time_s), np.nan)
    for phase in phases:
        start, end = phase.start_s + lag_s, phase.end_s + lag_s
        inside = (time_s >= start) & (time_s < end)
        duration = max(end - start, 1e-9)
        if phase.label == INHALE:
            template[inside] = 1.0 / duration
        elif phase.label == EXHALE:
            template[inside] = -1.0 / duration
        elif phase.label in (HOLD_AFTER_INHALE, HOLD_AFTER_EXHALE):
            template[inside] = 0.0
    return template


def estimate_lag(
    time_s: np.ndarray,
    velocity: np.ndarray,
    phases: Sequence[CuePhase],
    lags_s: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """``(lag_s, |correlation|, sign)``: the delay that best matches cues to the motion.

    Matched on velocity rather than position: slow drift barely moves a
    velocity trace, while it can tilt a position trace enough to pull the
    match.  The sign says whether the signal already rises on inhale (+1) or
    had to be flipped (-1) -- an IMU's main axis can point either way.
    """

    lags_s = np.arange(-0.25, 1.5001, 0.025) if lags_s is None else lags_s
    velocity = np.asarray(velocity, dtype=float)
    best = (float("nan"), 0.0, 1.0)
    for lag in lags_s:
        template = expected_velocity(time_s, phases, float(lag))
        valid = np.isfinite(template) & np.isfinite(velocity)
        if valid.sum() < 10 or np.std(template[valid]) == 0 or np.std(velocity[valid]) == 0:
            continue
        correlation = float(np.corrcoef(velocity[valid], template[valid])[0, 1])
        if abs(correlation) > best[1]:
            best = (float(lag), abs(correlation), 1.0 if correlation >= 0 else -1.0)
    if not np.isfinite(best[0]):
        raise ValueError("Could not match the cues to the signal: no cued breathing overlaps it.")
    return best


def _window(time_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    return np.flatnonzero((time_s >= start_s) & (time_s <= end_s))


def _sustained(mask: np.ndarray, start: int, length: int) -> bool:
    return bool(np.all(mask[start : start + length])) and start + length <= len(mask)


def align_cues(
    time_s: np.ndarray,
    chest: np.ndarray,
    cues: Iterable[Any],
    *,
    fs: float,
    lag_s: float | None = None,
    search_before_s: float = 0.6,
    search_after_s: float = 0.9,
    hold_end_before_s: float = 1.5,
    min_phase_s: float = 0.4,
    mask_s: float = 0.2,
    fallback_mask_s: float = 0.5,
    motion_fraction: float = 0.2,
    settle_s: float = 0.3,
) -> AlignmentResult:
    """Per-sample phase labels for ``chest`` from coach ``cues``; see the module docstring."""

    time_s = np.asarray(time_s, dtype=float)
    phases = cue_phases(cues)
    labelled_phases = [phase for phase in phases if phase.label != IGNORE]
    if not labelled_phases:
        raise ValueError("The cues contain no inhale, exhale or hold to label.")
    raw_velocity = np.gradient(np.asarray(chest, dtype=float)) * fs
    found_lag, correlation, sign = estimate_lag(time_s, raw_velocity, phases)
    lag = found_lag if lag_s is None else float(lag_s)
    signal = sign * np.asarray(chest, dtype=float)
    velocity = sign * raw_velocity

    # Typical breath speed from the cued breaths, to judge what counts as
    # moving and as standing still in this recording.
    breathing = np.zeros(len(time_s), dtype=bool)
    for phase in labelled_phases:
        if phase.label in (INHALE, EXHALE):
            breathing |= (time_s >= phase.start_s + lag) & (time_s < phase.end_s + lag)
    speed = float(np.median(np.abs(velocity[breathing]))) if breathing.any() else float(np.median(np.abs(velocity)))
    threshold = motion_fraction * max(speed, 1e-12)
    settle = max(1, int(round(settle_s * fs)))
    start_run = max(1, int(round(0.2 * fs)))

    # Boundaries: one before every cued phase whose label differs from the one
    # before it, plus the end of the last labelled phase.
    sequence = [CuePhase(phases[0].start_s, phases[0].start_s, "start", IGNORE), *phases]
    boundaries: list[Boundary] = []
    previous_time = -np.inf
    for index in range(1, len(sequence)):
        before, after = sequence[index - 1].label, sequence[index].label
        if before == after or (before == IGNORE and after == IGNORE):
            continue
        cue_s = sequence[index].start_s
        expected = cue_s + lag
        upcoming = [phase.start_s + lag for phase in sequence[index + 1 :] if phase.label != after]
        limit_hi = (upcoming[0] - min_phase_s) if upcoming else np.inf
        lo = max(expected - search_before_s, previous_time + min_phase_s)
        hi = min(expected + search_after_s, limit_hi)
        window = _window(time_s, lo, hi)
        method, moment = "fallback", expected

        if after == IGNORE:
            method = "end"
        elif len(window) >= 3:
            if (before, after) in ((INHALE, EXHALE), (IGNORE, EXHALE)) or (before, after) in ((EXHALE, INHALE), (IGNORE, INHALE)):
                # A turn is where the motion really reverses: moving one way
                # with real speed before it and the other way after it.  A
                # rounded top passes; a wobble on a flat stretch does not.
                peak = after == EXHALE
                segment = signal[window]
                speeds = velocity[window] if peak else -velocity[window]
                k = int(np.argmax(segment) if peak else np.argmin(segment))
                if 0 < k < len(segment) - 1 and speeds[:k].max() > threshold and speeds[k + 1 :].min() < -threshold:
                    method, moment = "turn", float(time_s[window[k]])
            elif (before, after) in ((INHALE, HOLD_AFTER_INHALE), (EXHALE, HOLD_AFTER_EXHALE)):
                rising = before == INHALE
                still = (velocity[window] < threshold) if rising else (velocity[window] > -threshold)
                moving = ~still
                for k in range(1, len(window)):
                    if moving[:k].any() and _sustained(still, k, settle):
                        method, moment = "stop", float(time_s[window[k]])
                        break
            elif (before, after) in ((HOLD_AFTER_EXHALE, INHALE), (HOLD_AFTER_INHALE, EXHALE)):
                # A breath held halfway is often finished before the next one
                # starts: exhale, hold, a little more exhale, then inhale.  So
                # the first sustained motion either way ends the hold, and the
                # search starts earlier, because people leave a hold as soon
                # as they feel the cue coming.
                rising = after == INHALE
                window = _window(time_s, max(expected - hold_end_before_s, previous_time + min_phase_s), hi)
                v = velocity[window]
                towards = (v > threshold) if rising else (v < -threshold)
                onward = (v < -threshold) if rising else (v > threshold)
                for k in range(1, len(window)):
                    new_breath = not towards[k - 1] and _sustained(towards, k, start_run)
                    carried_on = not onward[k - 1] and _sustained(onward, k, start_run)
                    if not (new_breath or carried_on):
                        continue
                    # Back up to where the motion began, not where it got fast.
                    weak = (v > 0.3 * threshold) if rising == new_breath else (v < -0.3 * threshold)
                    onset = k
                    while onset > 1 and weak[onset - 1]:
                        onset -= 1
                    if new_breath:
                        method, moment = "start", float(time_s[window[onset]])
                    else:
                        # Finishing the held breath delays the next one, so its
                        # turn is looked for up to 2.5 s past the cue rather
                        # than within the usual window.
                        rest = _window(time_s, float(time_s[window[k]]), min(limit_hi, expected + 2.5))
                        turn = int(np.argmin(signal[rest]) if rising else np.argmax(signal[rest]))
                        if 0 < turn < len(rest) - 1:
                            continued = EXHALE if rising else INHALE
                            boundaries.append(Boundary(cue_s, expected, float(time_s[window[onset]]), before, continued, "start"))
                            before = continued
                            method, moment = "turn", float(time_s[rest[turn]])
                    break

        if method == "fallback":
            moment = max(expected, previous_time + min_phase_s)
        boundaries.append(Boundary(cue_s, expected, float(moment), before, after, method))
        previous_time = float(moment)

    # Segments between consecutive boundaries carry the class that was cued.
    segments: list[tuple[float, float, int]] = []
    for current, following in zip(boundaries[:-1], boundaries[1:]):
        if current.after != IGNORE:
            segments.append((current.time_s, following.time_s, current.after))
    last = boundaries[-1]
    if last.after != IGNORE:
        final = next(phase for phase in reversed(labelled_phases))
        segments.append((last.time_s, final.end_s + lag, last.after))

    labels = labels_from_segments(time_s, segments)
    widths = [fallback_mask_s if b.method == "fallback" else mask_s for b in boundaries]
    labels = mask_around(labels, time_s, [b.time_s for b in boundaries], widths)
    return AlignmentResult(
        lag_s=lag,
        correlation=correlation,
        sign=sign,
        boundaries=tuple(boundaries),
        segments=tuple(segments),
        labels=labels,
        time_s=time_s,
    )
