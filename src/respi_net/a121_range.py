"""Which A121 range bins are the chest: chosen by how well they breathe, not by how strongly they echo.

Acconeer's breathing app, and the strongest-echo fallback, take the range of the strongest reflector.  On a
person that is often not the chest: a wristwatch, a phone or a buckle reflects far more than skin and clothing,
and it moves with the arm and the pulse more than with the breath.  In a 90 s recording from the side at about
0.8 m, a watch at 0.49 m had the strongest echo (about 1000 against 160-250 on the chest at 0.64-0.73 m) and was
chosen, with breathing about 3 times its high-frequency noise against about 10 times on the chest, at the same
12.7 breaths/min.

Each bin is scored on its own unwrapped phase -- its displacement -- in windows of
:data:`~respi_net.phase_baseline.RATE_WINDOW_S` every :data:`SCORE_STEP_S`:

* clarity: rms in the breathing band (:data:`SCORE_BREATH_BAND_HZ`, up to just above a faster rhythm) over rms
  in :data:`SCORE_NOISE_BAND_HZ`, where a chest has only receiver noise, the harmonics of the pulse and fidgeting;
* steadiness: the share of the 0.1-2.5 Hz power in the window's dominant rhythm;
* a window counts only when its rhythm is one the phase detector's fast-breathing test calls breathing
  (:func:`~respi_net.phase_baseline.window_rhythm`): dominant, and above
  :data:`~respi_net.phase_baseline.FAST_BREATHING_HZ` at least
  :data:`~respi_net.phase_baseline.RATE_MIN_DEPTH` of a typical breath -- so the pulse (about 1 Hz, a few % of
  a breath) does not pass, in a hold or on a wrist.  The typical breath is that of the bin that breathes best
  at a normal pace.

A window scores clarity x steadiness (0 when it does not count), and a bin the median over the windows in which
some bin breathes, so holds and settling count against no one.  Bins whose raw IQ hardly moves above the
receiver noise are left out (:func:`motion_snr`): the unwrapped phase of noise is a random walk, whose slow
wander passes for breathing.

A score cannot say which way up a bin breathes: next to the chest there are bins moving against it (a reflection,
or the chest echo mixed with another one), and some breathe as clearly as the chest.  The shape of the breath can
(:func:`polarity`): inhaling is active and quicker than the passive exhale, so a trace rising on inhale rises for
less of the time than it falls -- right way up -- and a trace that rises for more of it is upside down.  On all
recordings with cues or a belt this held for every trace it decided; where breathing is symmetric it stays unsure.

Offline (:func:`choose_chest_range`) the whole recording is scored once and the old choice -- Acconeer's range,
else its target, else the strongest echo -- stays unless the best group of bins is :func:`clearly_better`: moving
in the same direction, the old trace scores under :data:`KEEP_OLD_FRACTION` of it and the group is not upside down; moving against
it, the group is right way up and the old trace is not (from the side at 50 cm Acconeer's range was upside down
against the belt, the bins in front of it right way up; on the 2 m retest the best bins were an upside-down
reflection three times).  Live (:class:`LiveRangeSelector`) the last :data:`LIVE_WINDOW_S` are rescored every
:data:`LIVE_UPDATE_S` and the polarity gathered over minutes; the selection moves only to a range that stays
clearly better for :data:`LIVE_SWITCH_AFTER_S`, and not at all while no range breathes (a hold): then only the
heartbeat and arm movements are left to win.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .chest_signal import A121_CHEST_SIGN, PHASE_TO_MM, RATE_BAND_HZ
from .phase_baseline import FAST_BREATHING_HZ, RATE_DOMINANCE, RATE_MIN_DEPTH, RATE_WINDOW_S, window_rhythm

SCORE_STEP_S = 5.0
SCORE_BREATH_BAND_HZ = (0.10, 0.70)
SCORE_NOISE_BAND_HZ = (1.50, 4.00)
NOISE_FLOOR_MM = 1e-6  # a noiseless (synthetic) trace still gets a finite clarity
# Breathing makes the raw IQ of the chest return change in the breathing band far above the receiver noise
# floor (MOTION_NOISE_BAND_MIN_HZ and up); static scenes and noise stay close to it.
BREATHING_MOTION_SNR_MIN = 10.0
MOTION_NOISE_BAND_MIN_HZ = 2.5

# Offline group: up to GROUP_HALF_WIDTH bins either side of the best one, each scoring at least GROUP_FRACTION of
# it and moving in step with it (breathing-band correlation), so a neighbour in antiphase cannot cancel the trace.
GROUP_HALF_WIDTH = 1
GROUP_FRACTION = 0.7
GROUP_MIN_CORRELATION = 0.5
# The old choice stays unless its trace scores below this share of the best group's (both moving the same way).
# 0.65 rather than 0.5 (2026-11-07, tools/compare_a121_range_selection.py --keep-old-fraction on 38 recordings with
# a reference): 7 recordings change offline, 6 for the better (e.g. belt agreement 0.57 -> 0.74 at 1.3 m, phone
# 0.79 -> 0.90, cues 0.88 -> 0.96 at 2 m) and one even (cues 0.94 -> 0.91, cue correlation 0.72 -> 0.79); 0.8 adds
# only even or unscored ones.  Live (tools/replay_a121_live_range_selection.py) the belt agreement where the
# selection moved goes 0.81 -> 0.84 with no recording worse, at 47 switches over the 38 runs instead of 31.
KEEP_OLD_FRACTION = 0.65

# Polarity -- which way up a trace breathes -- from the shape of the breath, with no other sensor: inhaling is active
# and quicker than the passive exhale, so a chest trace that rises on inhale rises for less of the time than it
# falls.  Per scoring window, of the samples whose velocity (POLARITY_BAND_HZ) exceeds POLARITY_DEAD_FRACTION of the
# window's rms (holds and turns left out), the share in which the chest rises; a trace is right way up when the
# median over the windows in which it breathes is at least POLARITY_MARGIN under one half and at least
# POLARITY_AGREEMENT of those windows are under it, upside down when the same holds above one half, and unsure
# otherwise or with fewer than POLARITY_MIN_WINDOWS such windows.
POLARITY_BAND_HZ = (0.05, 0.90)
POLARITY_DEAD_FRACTION = 0.2
POLARITY_MARGIN = 0.03
POLARITY_AGREEMENT = 0.75
POLARITY_MIN_WINDOWS = 6
# A trace whose polarity says upside down is never turned over by default: only the choice between two ranges
# moving against each other uses the polarity (see :func:`clearly_better`).
POLARITY_FLIP = False

LIVE_WINDOW_S = 30.0
LIVE_UPDATE_S = 2.0
LIVE_SMOOTH_S = 30.0  # time constant of the per-bin scores averaged over updates: one breathing block does not decide
LIVE_SWITCH_AFTER_S = 10.0  # a range clearly better than the current one (as offline) at every update for this long
# Frozen while no range breathes: the current bin moving over the last LIVE_STILL_S (a hold: the offline detector's
# shortest one) less than LIVE_STILL_FRACTION of its usual motion over that long, or no rhythm counting anywhere in
# the last window.
LIVE_STILL_S = 4.0
LIVE_STILL_FRACTION = 0.2
LIVE_MOTION_DECAY = 0.97  # per frozen update, of the usual motion: breathing that stays shallow thaws in time
# Live polarity: each bin's rise share (and whether it is under one half) averaged over the updates in which it
# breathes, with this time constant (a plain mean until then), and judged (:func:`polarity`) only after it has
# breathed for LIVE_POLARITY_MIN_S.  Frozen updates (holds) add nothing.
LIVE_POLARITY_S = 120.0
LIVE_POLARITY_MIN_S = 60.0


def motion_snr(profile: np.ndarray, fs: float, band_hz: tuple[float, float] = RATE_BAND_HZ, noise_min_hz: float = MOTION_NOISE_BAND_MIN_HZ) -> np.ndarray:
    """How far the raw IQ of each bin changes in the breathing band above receiver noise.

    The mean power in ``band_hz`` over the median power above ``noise_min_hz``, where only noise is expected.
    Raw IQ is used because the unwrapped phase of noise is a random walk with strong low-frequency power, which
    looks like slow motion; the ratio does not depend on the IQ amplitude scale.  Inf where the frame rate is too
    low to measure the noise band.
    """

    z = np.asarray(profile, dtype=np.complex128)
    z = z.reshape(len(z), -1)
    if z.shape[0] < 16 or fs <= 0 or fs * 0.46 <= noise_min_hz:
        return np.full(z.shape[1], np.inf)
    nperseg = min(z.shape[0], max(64, int(round(fs * 10.0))))
    freqs, psd = welch(z, fs=fs, nperseg=nperseg, return_onesided=False, axis=0)
    abs_freqs = np.abs(freqs)
    breathing = (abs_freqs >= band_hz[0]) & (abs_freqs <= band_hz[1])
    noise = (abs_freqs >= noise_min_hz) & (abs_freqs <= fs * 0.46)
    if not np.any(breathing) or not np.any(noise):
        return np.full(z.shape[1], np.inf)
    floor = np.maximum(np.median(psd[noise], axis=0), np.finfo(float).tiny)
    return np.mean(psd[breathing], axis=0) / floor


def bin_displacement(profile: np.ndarray) -> np.ndarray:
    """Unwrapped phase of every bin in mm (sign as recorded: it falls on inhale)."""

    return np.unwrap(np.angle(np.asarray(profile)), axis=0) * PHASE_TO_MM


@dataclass(frozen=True)
class RangeScores:
    score: np.ndarray  # per bin: median window score over the windows in which some bin breathes
    clarity: np.ndarray  # median breathing-to-noise rms ratio over the same windows
    steadiness: np.ndarray  # median share of the dominant rhythm
    rate_hz: np.ndarray  # median breathing rate of the windows that count, 0 if none
    moving: np.ndarray  # raw IQ moves above receiver noise
    latest: np.ndarray  # score of the last window alone
    typical_mm: float  # size of a typical breath (5th-95th percentile of the rhythm) a fast rhythm was held to
    slow_typical_mm: float  # the typical breath of these data at a normal pace, 0 if they have none
    windows: int  # windows in which some bin breathes
    rise_share: np.ndarray  # per bin: median share of the moving samples in which the chest rises, over its breathing windows
    rise_below: np.ndarray  # per bin: share of its breathing windows in which the chest rises for less than half
    polarity_windows: np.ndarray  # per bin: its breathing windows (the windows that count for it)

    def polarity(self) -> np.ndarray:
        """:func:`polarity` of every bin."""

        return polarity(self.rise_share, self.rise_below, self.polarity_windows)


def _empty_scores(bins: int, moving: np.ndarray, typical_mm: float = 0.0, slow_typical_mm: float = 0.0) -> RangeScores:
    zeros, halves = np.zeros(bins), np.full(bins, 0.5)
    return RangeScores(zeros, zeros, zeros, zeros, moving, zeros, typical_mm, slow_typical_mm, 0, halves, halves, np.zeros(bins, dtype=int))


def polarity(rise_share: np.ndarray | float, rise_below: np.ndarray | float, windows: np.ndarray | int,
             min_windows: int = POLARITY_MIN_WINDOWS) -> np.ndarray:
    """+1 where a chest trace is right way up (rises on inhale), -1 upside down, 0 unsure (see :data:`POLARITY_MARGIN`)."""

    margin = 0.5 - np.asarray(rise_share, dtype=float)
    below = np.asarray(rise_below, dtype=float)
    enough = np.asarray(windows) >= min_windows
    right = enough & (margin >= POLARITY_MARGIN) & (below >= POLARITY_AGREEMENT)
    upside_down = enough & (margin <= -POLARITY_MARGIN) & (1.0 - below >= POLARITY_AGREEMENT)
    return right.astype(int) - upside_down.astype(int)


def _chest_velocity(displacement: np.ndarray, fs: float, chest_sign: float) -> np.ndarray:
    """Velocity of the chest (rising on inhale) in :data:`POLARITY_BAND_HZ`, per column."""

    high = min(POLARITY_BAND_HZ[1], 0.45 * fs)
    sos = butter(2, [POLARITY_BAND_HZ[0], high], btype="bandpass", fs=fs, output="sos")
    return np.gradient(sosfiltfilt(sos, chest_sign * displacement, axis=0), axis=0)


def _rise_share(velocity: np.ndarray) -> np.ndarray:
    """Per column: the share of the moving samples (above :data:`POLARITY_DEAD_FRACTION` of the rms) that rise."""

    threshold = POLARITY_DEAD_FRACTION * np.sqrt(np.mean(velocity**2, axis=0))
    up, down = (velocity > threshold).sum(axis=0), (velocity < -threshold).sum(axis=0)
    return np.where(up + down > 0, up / np.maximum(up + down, 1), 0.5)


def _typical_breath(reference: np.ndarray, window_score: np.ndarray, size: np.ndarray) -> float:
    """Median rhythm size of the bin that scores best over the windows marked in ``reference`` (windows x bins)."""

    rows = reference.any(axis=1)
    if not rows.any():
        return 0.0
    best = int(np.argmax(np.median(np.where(reference, window_score, 0.0)[rows], axis=0)))
    return float(np.median(size[reference[:, best], best])) if reference[:, best].any() else 0.0


def score_displacement(
    displacement: np.ndarray,
    fs: float,
    *,
    moving: np.ndarray | None = None,
    typical_mm: float | None = None,
    window_s: float = RATE_WINDOW_S,
    step_s: float = SCORE_STEP_S,
    chest_sign: float = A121_CHEST_SIGN,
) -> RangeScores:
    """Breathing scores (see the module docstring) of each column of ``displacement`` in mm, and its polarity.

    ``typical_mm`` -- the breath that a fast rhythm must reach a share of -- is measured here unless given (a live
    caller passes the one it measured while breathing, so the heartbeat of a hold cannot set it, and inf until it
    has seen breathing at a normal pace).  ``chest_sign`` turns ``displacement`` into chest motion rising on inhale
    (the recorded phase falls on inhale); only the polarity depends on it.
    """

    d = np.asarray(displacement, dtype=float)
    d = d.reshape(len(d), -1)
    n, bins = d.shape
    moving = np.ones(bins, dtype=bool) if moving is None else np.asarray(moving, dtype=bool)
    width, step = int(round(window_s * fs)), max(1, int(round(step_s * fs)))
    if n < width or width < 8 or not moving.any():
        return _empty_scores(bins, moving, typical_mm or 0.0)
    starts = list(range(0, n - width + 1, step))
    if starts[-1] != n - width:
        starts.append(n - width)  # the last window ends at the last sample: what a live caller sees now
    freqs = np.fft.rfftfreq(width, 1.0 / fs)[:, None]
    noise_band = (freqs >= SCORE_NOISE_BAND_HZ[0]) & (freqs <= min(SCORE_NOISE_BAND_HZ[1], 0.45 * fs))
    floor = NOISE_FLOOR_MM**2 * float(np.sum(np.hanning(width) ** 2)) * max(1, int(noise_band.sum()))
    peak, share, size, clarity, rise = (np.zeros((len(starts), bins)) for _ in range(5))
    velocity = _chest_velocity(d, fs, chest_sign)
    for index, start in enumerate(starts):
        rise[index] = _rise_share(velocity[start : start + width])
        rhythm = window_rhythm(d[start : start + width], fs)
        peak[index], share[index], size[index] = rhythm.peak_hz, rhythm.share, rhythm.size
        # A faster rhythm widens the breathing band, up to the noise band.
        top = np.maximum(SCORE_BREATH_BAND_HZ[1], np.minimum(rhythm.peak_hz + 0.1, SCORE_NOISE_BAND_HZ[0]))
        breath = np.where((freqs >= SCORE_BREATH_BAND_HZ[0]) & (freqs <= top), rhythm.power, 0.0).sum(axis=0)
        noise = np.where(noise_band, rhythm.power, 0.0).sum(axis=0)
        clarity[index] = np.sqrt(breath / np.maximum(noise, floor))

    dominant = (share >= RATE_DOMINANCE) & moving
    slow = dominant & (peak < FAST_BREATHING_HZ)
    slow_typical = _typical_breath(slow, clarity * share, size)
    if typical_mm is None:
        # A recording breathing fast throughout has no normal breath to compare with: any rhythm is the reference.
        typical_mm = slow_typical if slow.any() else _typical_breath(dominant, clarity * share, size)
    counts = dominant & ((peak < FAST_BREATHING_HZ) | (size >= RATE_MIN_DEPTH * typical_mm))
    window_score = np.where(counts, clarity * share, 0.0)
    breathing = counts.any(axis=1)
    if not breathing.any():
        return _empty_scores(bins, moving, typical_mm, slow_typical)
    rate = np.array([float(np.median(peak[counts[:, b], b])) if counts[:, b].any() else 0.0 for b in range(bins)])
    rise_share = np.array([float(np.median(rise[counts[:, b], b])) if counts[:, b].any() else 0.5 for b in range(bins)])
    rise_below = np.array([float(np.mean(rise[counts[:, b], b] < 0.5)) if counts[:, b].any() else 0.5 for b in range(bins)])
    return RangeScores(
        score=np.median(window_score[breathing], axis=0),
        clarity=np.median(clarity[breathing], axis=0),
        steadiness=np.median(share[breathing], axis=0),
        rate_hz=rate,
        moving=moving,
        latest=window_score[-1],
        typical_mm=typical_mm,
        slow_typical_mm=slow_typical,
        windows=int(breathing.sum()),
        rise_share=rise_share,
        rise_below=rise_below,
        polarity_windows=counts.sum(axis=0),
    )


def score_range_bins(profile: np.ndarray, fs: float, **kwargs: Any) -> RangeScores:
    """:func:`score_displacement` of every bin of a complex range profile (frames x bins)."""

    moving = motion_snr(profile, fs) >= BREATHING_MOTION_SNR_MIN
    return score_displacement(bin_displacement(profile), fs, moving=moving, **kwargs)


def _breathing_band(displacement: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(2, [SCORE_BREATH_BAND_HZ[0], min(SCORE_BREATH_BAND_HZ[1], 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, displacement, axis=0)


def chest_group(scores: RangeScores, displacement: np.ndarray, fs: float) -> np.ndarray:
    """Contiguous bins around the best-scoring one that score nearly as well and move with it; empty if none breathes."""

    score = np.where(scores.moving, scores.score, 0.0)
    best = int(np.argmax(score))
    if score[best] <= 0:
        return np.asarray([], dtype=int)
    lo, hi = max(0, best - GROUP_HALF_WIDTH), min(len(score), best + GROUP_HALF_WIDTH + 1)
    breath = _breathing_band(np.asarray(displacement, dtype=float)[:, lo:hi], fs)
    reference = breath[:, best - lo]
    group = [best]
    for side in (-1, 1):
        index = best + side
        while lo <= index < hi and score[index] >= GROUP_FRACTION * score[best]:
            other = breath[:, index - lo]
            if np.std(other) <= 0 or np.corrcoef(reference, other)[0, 1] < GROUP_MIN_CORRELATION:
                break
            group.append(index)
            index += side
    return np.asarray(sorted(group), dtype=int)


def in_phase(a: np.ndarray, b: np.ndarray, fs: float) -> bool:
    """Do two displacement traces breathe in the same direction (breathing-band correlation not negative)?"""

    band = _breathing_band(np.column_stack([a, b]), fs)
    return bool(np.std(band[:, 0]) > 0 and np.std(band[:, 1]) > 0 and np.corrcoef(band[:, 0], band[:, 1])[0, 1] >= 0)


def clearly_better(
    new_score: float,
    old_score: float,
    new_trace: np.ndarray,
    old_trace: np.ndarray,
    fs: float,
    new_polarity: int = 0,
    old_polarity: int = 0,
) -> bool:
    """The rule for leaving the old choice, offline and live.

    When the two breathe in the same direction: the old one scores below :data:`KEEP_OLD_FRACTION` of the new one,
    and the new one is not upside down (:func:`polarity`).  When they move against each other, one of them is a
    reflection moving the other way, and a score cannot say which (on the 2 m retest the best bins were such a
    reflection three times, and their labels were turned inside out); the shape of the breath can: the new one is
    right way up and the old one is not.  Unsure polarities keep the old choice.
    """

    if in_phase(new_trace, old_trace, fs):
        return old_score < KEEP_OLD_FRACTION * new_score and new_polarity >= 0
    return new_polarity > 0 and old_polarity <= 0


@dataclass(frozen=True)
class RangeChoice:
    gate: np.ndarray
    source: str
    meta: dict[str, Any]
    sign: float = 1.0  # -1 turns the chosen trace over (only with POLARITY_FLIP, when it is upside down)


def _polarity_meta(prefix: str, scores: RangeScores) -> dict[str, Any]:
    return {
        f"{prefix}_polarity": int(scores.polarity()[0]),
        f"{prefix}_rise_share": round(float(scores.rise_share[0]), 3),
        f"{prefix}_rise_below": round(float(scores.rise_below[0]), 2),
        f"{prefix}_polarity_windows": int(scores.polarity_windows[0]),
    }


def choose_chest_range(
    profile: np.ndarray,
    fs: float,
    distances: np.ndarray,
    amplitude: np.ndarray,
    old_gate: np.ndarray,
    old_source: str,
) -> RangeChoice:
    """The chest bins of a whole recording: the best group ("breathing-score") when it is :func:`clearly_better`
    than ``old_gate``, else ``old_gate``.  ``amplitude`` is each bin's median echo, the weight bins are combined
    with."""

    displacement = bin_displacement(profile)
    scores = score_range_bins(profile, fs)
    group = chest_group(scores, displacement, fs)

    def weights(gate: np.ndarray) -> np.ndarray:
        w = np.maximum(np.asarray(amplitude, dtype=float)[gate], 0.0)
        return w / (float(np.sum(w)) + 1e-12)

    meta: dict[str, Any] = {
        "range_scores": [round(float(value), 2) for value in scores.score],
        "range_typical_breath_mm": round(scores.typical_mm, 3),
        "old_gate_source": old_source,
        "old_gate_m": [float(distances[int(old_gate[0])]), float(distances[int(old_gate[-1])])],
    }
    if not len(group):
        return RangeChoice(old_gate, old_source, {**meta, "range_note": "no bin breathes; old gate kept"})
    old_trace = displacement[:, old_gate] @ weights(old_gate)
    group_trace = displacement[:, group] @ weights(group)
    old_scores = score_displacement(old_trace, fs, typical_mm=scores.typical_mm)
    group_scores = score_displacement(group_trace, fs, typical_mm=scores.typical_mm)
    old_score, group_score = float(old_scores.score[0]), float(group_scores.score[0])
    old_polarity, group_polarity = int(old_scores.polarity()[0]), int(group_scores.polarity()[0])
    same_way = in_phase(group_trace, old_trace, fs)
    meta |= {"old_gate_score": round(old_score, 2), "best_group_score": round(group_score, 2),
             "best_group_m": [float(distances[int(group[0])]), float(distances[int(group[-1])])],
             "best_group_in_phase": same_way, **_polarity_meta("old_gate", old_scores), **_polarity_meta("best_group", group_scores)}
    if clearly_better(group_score, old_score, group_trace, old_trace, fs, group_polarity, old_polarity):
        gate, source, chosen_polarity = group, "breathing-score", group_polarity
        if not same_way:
            meta["range_note"] = "best group moves against the old gate and is right way up; old gate upside down"
    else:
        gate, source, chosen_polarity = old_gate, old_source, old_polarity
        if not same_way:
            meta["range_note"] = "best group moves against the old gate and is not right way up; old gate kept"
        elif group_polarity < 0 and old_score < KEEP_OLD_FRACTION * group_score:
            meta["range_note"] = "best group is upside down; old gate kept"
    sign = -1.0 if POLARITY_FLIP and chosen_polarity < 0 else 1.0
    if sign < 0:
        meta["range_note"] = meta.get("range_note", "") + ("; " if meta.get("range_note") else "") + "chosen trace upside down: turned over"
    return RangeChoice(gate, source, meta, sign)


def _motion_mm(trace: np.ndarray, width: int) -> np.ndarray:
    """5th-95th percentile range of ``trace`` in windows of ``width`` samples, every quarter window; the last one
    ends at the last sample."""

    starts = list(range(0, max(1, len(trace) - width + 1), max(1, width // 4)))
    if starts[-1] != max(0, len(trace) - width):
        starts.append(max(0, len(trace) - width))
    return np.array([np.subtract(*np.percentile(trace[start : start + width], [95, 5])) for start in starts])


class LiveRangeSelector:
    """The chest bin of a live stream, rescored every :data:`LIVE_UPDATE_S` over the last :data:`LIVE_WINDOW_S`.

    It starts at the ``prior`` the caller gives (Acconeer's target or the strongest echo) and goes back to it
    whenever the prior changes (a new acquisition).  The per-bin scores are averaged over updates
    (:data:`LIVE_SMOOTH_S`), and the selection moves to the best bin (two or more bins away) only after it has been
    :func:`clearly_better` than the current one at every update for :data:`LIVE_SWITCH_AFTER_S`; to a bin moving
    against the current one only when the polarity, gathered per bin over minutes of breathing
    (:data:`LIVE_POLARITY_S`), says the new bin is right way up and the current one is not.  While no range
    breathes -- a hold, when only the heartbeat or an arm is left to win -- it is frozen: no switch, the count starts
    again, the polarity does not change, and nothing recorded up to the end of the freeze is scored again
    (decisions resume once :data:`~respi_net.phase_baseline.RATE_WINDOW_S` have been recorded after it).  The trace
    is never turned over live.
    """

    def __init__(
        self,
        fs: float,
        bins: int,
        *,
        window_s: float = LIVE_WINDOW_S,
        update_s: float = LIVE_UPDATE_S,
        switch_after_s: float = LIVE_SWITCH_AFTER_S,
        still_fraction: float = LIVE_STILL_FRACTION,
        freeze: bool = True,
    ) -> None:
        self.fs, self.bins = float(fs), int(bins)
        self.update_s, self.switch_after_s = float(update_s), float(switch_after_s)
        self.still_fraction, self.freeze = float(still_fraction), bool(freeze)
        size = max(8, int(round(window_s * self.fs)))
        self.times: deque[float] = deque(maxlen=size)
        self.iq: deque[np.ndarray] = deque(maxlen=size)
        self.phase: deque[np.ndarray] = deque(maxlen=size)
        self.prior: int | None = None
        self.current: int | None = None
        self.source = ""
        self.frozen = False
        self.frozen_until_s = -np.inf  # time of the last frozen update
        self.typical_mm: float | None = None
        self.scores: RangeScores | None = None  # of the last update
        self.smoothed: np.ndarray | None = None  # per-bin scores averaged over the updates while breathing
        self.usual_motion_mm: float | None = None  # of the current bin over LIVE_STILL_S, at the last unfrozen update
        self.rise_share = np.full(self.bins, 0.5)  # per bin, averaged over the updates in which it breathes
        self.rise_below = np.full(self.bins, 0.5)
        self.polarity_s = np.zeros(self.bins)  # breathing seconds behind rise_share
        self._pending: tuple[int, float] | None = None  # candidate bin and when it started to win
        self._next_update_s: float | None = None

    def update(self, time_s: float, iq: np.ndarray, phase_rad: np.ndarray, prior: int) -> int:
        """Add one frame (complex IQ and unwrapped phase of every bin) and return the bin to use."""

        if prior != self.prior:
            self.prior = self.current = int(prior)
            self.source, self._pending, self.usual_motion_mm = "prior", None, None
        self.times.append(float(time_s))
        self.iq.append(np.array(iq, dtype=np.complex128))
        self.phase.append(np.array(phase_rad, dtype=float))
        if self._next_update_s is None:
            self._next_update_s = time_s + RATE_WINDOW_S
        if time_s >= self._next_update_s:
            self._next_update_s = time_s + self.update_s
            self._evaluate(time_s)
        assert self.current is not None
        return self.current

    def _evaluate(self, now_s: float) -> None:
        iq = np.vstack(self.iq)
        displacement = np.vstack(self.phase) * PHASE_TO_MM
        moving = motion_snr(iq, self.fs) >= BREATHING_MOTION_SNR_MIN
        typical = self.typical_mm if self.typical_mm is not None else np.inf
        scores = score_displacement(displacement, self.fs, moving=moving, typical_mm=typical)
        self.scores = scores
        assert self.current is not None
        motion = _motion_mm(displacement[:, self.current], max(2, int(round(LIVE_STILL_S * self.fs))))
        still = self.usual_motion_mm is not None and motion[-1] < self.still_fraction * self.usual_motion_mm
        self.frozen = self.freeze and (bool(still) or not np.any(moving & (scores.latest > 0)))
        if self.frozen:
            self.frozen_until_s = now_s
            self.usual_motion_mm = None if self.usual_motion_mm is None else LIVE_MOTION_DECAY * self.usual_motion_mm
            self._pending = None
            return
        self.usual_motion_mm = float(np.median(motion))
        if scores.slow_typical_mm > 0:  # only breathing at a normal pace sets what a breath is
            slow = scores.slow_typical_mm
            self.typical_mm = slow if self.typical_mm is None else 0.8 * self.typical_mm + 0.2 * slow
        # Decide on what was recorded after the last freeze only: the arm that swung through a hold is still in the window.
        clean = np.asarray(self.times) > self.frozen_until_s
        if not clean.all():
            if clean.sum() < RATE_WINDOW_S * self.fs:
                self._pending = None
                return
            displacement = displacement[clean]
            scores = score_displacement(displacement, self.fs, moving=moving, typical_mm=typical)
        weight = min(1.0, self.update_s / LIVE_SMOOTH_S)
        self.smoothed = scores.score.copy() if self.smoothed is None else (1.0 - weight) * self.smoothed + weight * scores.score
        self._add_polarity(scores, moving)
        score = np.where(moving, self.smoothed, 0.0)
        best = int(np.argmax(score))
        current = float(score[self.current])
        signs = self.polarity()
        if abs(best - self.current) <= 1 or not clearly_better(
            score[best], current, displacement[:, best], displacement[:, self.current], self.fs, int(signs[best]), int(signs[self.current])
        ):
            self._pending = None
            return
        if self._pending is None or abs(self._pending[0] - best) > 1:
            self._pending = (best, now_s)
        elif now_s - self._pending[1] >= self.switch_after_s:
            self.current, self.source, self._pending, self.usual_motion_mm = best, "breathing-score", None, None

    def _add_polarity(self, scores: RangeScores, moving: np.ndarray) -> None:
        breathing = moving & (scores.polarity_windows > 0)
        updates = self.polarity_s / self.update_s + 1.0
        weight = np.where(breathing, np.maximum(self.update_s / LIVE_POLARITY_S, 1.0 / updates), 0.0)
        self.rise_share += weight * (scores.rise_share - self.rise_share)
        self.rise_below += weight * ((scores.rise_share < 0.5) - self.rise_below)
        self.polarity_s += np.where(breathing, self.update_s, 0.0)

    def polarity(self) -> np.ndarray:
        """Per bin: +1 right way up, -1 upside down, 0 unsure or not yet :data:`LIVE_POLARITY_MIN_S` of breathing."""

        return polarity(self.rise_share, self.rise_below, (self.polarity_s >= LIVE_POLARITY_MIN_S).astype(int), min_windows=1)
