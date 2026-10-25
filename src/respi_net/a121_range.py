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

Offline (:func:`choose_chest_range`) the whole recording is scored once and the old choice -- Acconeer's range,
else its target, else the strongest echo -- stays unless the best group of bins is :func:`clearly_better`: the
old trace scores under half of it, and the group breathes in the same direction.  Live
(:class:`LiveRangeSelector`) the last :data:`LIVE_WINDOW_S` are rescored every :data:`LIVE_UPDATE_S`; the
selection moves only to a range that stays clearly better for :data:`LIVE_SWITCH_AFTER_S`, and not at all while
no range breathes (a hold): then only the heartbeat and arm movements are left to win.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .chest_signal import PHASE_TO_MM, RATE_BAND_HZ
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
# The old choice stays unless its trace scores below this share of the best group's -- and always when the best
# group moves against it: one of the two is then a reflection moving the other way, and a score cannot say which
# one moves like the chest (on the 2 m retest the best bins were such a reflection twice, and their labels were
# turned inside out).
KEEP_OLD_FRACTION = 0.5

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


def _empty_scores(bins: int, moving: np.ndarray, typical_mm: float = 0.0, slow_typical_mm: float = 0.0) -> RangeScores:
    zeros = np.zeros(bins)
    return RangeScores(zeros, zeros, zeros, zeros, moving, zeros, typical_mm, slow_typical_mm, 0)


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
) -> RangeScores:
    """Breathing scores (see the module docstring) of each column of ``displacement`` in mm.

    ``typical_mm`` -- the breath that a fast rhythm must reach a share of -- is measured here unless given (a live
    caller passes the one it measured while breathing, so the heartbeat of a hold cannot set it, and inf until it
    has seen breathing at a normal pace).
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
    peak, share, size, clarity = (np.zeros((len(starts), bins)) for _ in range(4))
    for index, start in enumerate(starts):
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


def clearly_better(new_score: float, old_score: float, new_trace: np.ndarray, old_trace: np.ndarray, fs: float) -> bool:
    """The rule for leaving the old choice, offline and live: it scores below :data:`KEEP_OLD_FRACTION` of the
    new one, and the new one breathes in the same direction."""

    return old_score < KEEP_OLD_FRACTION * new_score and in_phase(new_trace, old_trace, fs)


@dataclass(frozen=True)
class RangeChoice:
    gate: np.ndarray
    source: str
    meta: dict[str, Any]


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
    old_score = float(score_displacement(old_trace, fs, typical_mm=scores.typical_mm).score[0])
    group_score = float(score_displacement(group_trace, fs, typical_mm=scores.typical_mm).score[0])
    meta |= {"old_gate_score": round(old_score, 2), "best_group_score": round(group_score, 2),
             "best_group_m": [float(distances[int(group[0])]), float(distances[int(group[-1])])]}
    if not clearly_better(group_score, old_score, group_trace, old_trace, fs):
        if not in_phase(group_trace, old_trace, fs):
            meta["range_note"] = "best group moves against the old gate; old gate kept"
        return RangeChoice(old_gate, old_source, meta)
    return RangeChoice(group, "breathing-score", meta)


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
    :func:`clearly_better` than the current one at every update for :data:`LIVE_SWITCH_AFTER_S`.  While no range
    breathes -- a hold, when only the heartbeat or an arm is left to win -- it is frozen: no switch, the count starts
    again, and nothing recorded up to the end of the freeze is scored again (decisions resume once
    :data:`~respi_net.phase_baseline.RATE_WINDOW_S` have been recorded after it).
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
        score = np.where(moving, self.smoothed, 0.0)
        best = int(np.argmax(score))
        current = float(score[self.current])
        if abs(best - self.current) <= 1 or not clearly_better(score[best], current, displacement[:, best], displacement[:, self.current], self.fs):
            self._pending = None
            return
        if self._pending is None or abs(self._pending[0] - best) > 1:
            self._pending = (best, now_s)
        elif now_s - self._pending[1] >= self.switch_after_s:
            self.current, self.source, self._pending, self.usual_motion_mm = best, "breathing-score", None, None
