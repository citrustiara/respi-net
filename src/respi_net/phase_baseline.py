"""Deterministic breathing-phase detector: the baseline a network has to beat.

It reads the direction of chest motion.  Rising is an inhale, falling an
exhale, and standing still for at least ``min_hold_s`` a hold -- after inhale
if the last breath went in, after exhale otherwise.  Shorter still moments
are turning points.  Offline, the next breath starts at the turn itself: an
inhale at the trough, an exhale at the crest of the trace smoothed to 1 Hz,
within 0.8 s of where its motion begins -- the place a hand label snaps to.
A breath into a hold likewise lasts until its turn: its slow end reads as
still, but the chest is still moving.

A hold after a deep breath often begins with a quick settle the other way:
with the glottis closed, volume shifts between chest and belly without air
moving, so the chest drops a little while the lungs stay full.  A short move
back (``max_settle_s``) followed by a hold that stays near where the breath
before it ended -- within ``max_settle_fraction`` of that breath's depth --
is therefore part of the hold, and the hold is named after that breath.  A
real exhale into a hold takes the chest most of the way down and keeps its
own name, even when its fast part is short and it finishes slowly.

Offline it sees the whole recording: smoothing is zero-phase and a hold is
labelled from its first still sample.  In real-time mode it sees only the
past: the smoothing lags, a reversal must be confirmed, and a hold can only
be declared once the stillness has lasted ``min_hold_s``.  That delay is not
a weakness of this detector but of deciding causally, and the network pays
it too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .breath_phases import EXHALE, HOLD_AFTER_EXHALE, HOLD_AFTER_INHALE, IGNORE, INHALE, label_runs, merge_holds
from .chest_signal import light_filter
from .phase_metrics import enforce_min_duration


def _slow_motion(smooth: np.ndarray, fs: float, fraction: float, window_s: float, depth_window_s: float = 30.0) -> np.ndarray:
    """Direction (+1 / -1 / 0) of a slow but real movement that the speed threshold misses.

    A shallow breath right after deep, fast ones is slower than half the speed set by those deep breaths and would
    pass for stillness.  It still moves the trace by a visible part of the local breath depth within ``window_s``,
    which a hold (noise and a slow settle) does not: net change over the window above ``fraction`` of the depth
    (95th minus 5th percentile over ``depth_window_s``) counts as motion.
    """

    n = len(smooth)
    half_window = max(1, int(round(window_s * fs / 2)))
    ahead = np.minimum(np.arange(n) + half_window, n - 1)
    behind = np.maximum(np.arange(n) - half_window, 0)
    change = smooth[ahead] - smooth[behind]
    depth = np.empty(n)
    step = max(1, int(round(fs)))
    reach = int(round(depth_window_s * fs / 2))
    for start in range(0, n, step):
        lo, hi = max(0, start - reach), min(n, start + reach)
        depth[start : start + step] = np.subtract(*np.percentile(smooth[lo:hi], [95, 5]))
    limit = fraction * np.maximum(depth, 1e-12)
    return np.where(change > limit, 1, np.where(change < -limit, -1, 0)).astype(np.int8)


def _speed_scale(speed: np.ndarray, fs: float, window_s: float, *, causal: bool) -> np.ndarray:
    """Typical breathing speed around each sample (80th percentile of |velocity|).

    Taken over a moving window, updated once a second, so a detector keeps
    working when breaths get shallower through a night.
    """

    n = len(speed)
    step = max(1, int(round(fs)))
    half = max(1, int(round(window_s * fs)))
    scale = np.empty(n)
    for start in range(0, n, step):
        stop = min(n, start + step)
        if causal:
            lo, hi = max(0, stop - half), stop
        else:
            centre = (start + stop) // 2
            lo, hi = max(0, centre - half // 2), min(n, centre + half // 2)
        scale[start:stop] = np.percentile(speed[lo:hi], 80) if hi > lo else np.nan
    fallback = float(np.nanmedian(scale)) if np.isfinite(scale).any() else 1.0
    return np.where(np.isfinite(scale) & (scale > 0), scale, fallback)


HOLD_OF = {INHALE: HOLD_AFTER_INHALE, EXHALE: HOLD_AFTER_EXHALE}
BREATH_OF = {1: INHALE, -1: EXHALE}


def _absorb_settles(
    direction: np.ndarray,
    smooth: np.ndarray,
    fs: float,
    min_hold_s: float,
    max_settle_s: float,
    max_settle_fraction: float,
) -> np.ndarray:
    """Direction with every settle into a hold turned into stillness (see module docstring)."""

    direction = np.array(direction, copy=True)
    runs = label_runs(direction)
    min_hold = int(round(min_hold_s * fs))
    max_settle = int(round(max_settle_s * fs))
    for index, (start, stop, value) in enumerate(runs[:-1]):
        following = runs[index + 1]
        if value == 0 or stop - start > max_settle or following[2] != 0 or following[1] - following[0] < min_hold:
            continue
        # The breath before, past a short pause at its turning point.
        before = index - 1
        while before >= 0 and runs[before][2] == 0 and runs[before][1] - runs[before][0] < min_hold:
            before -= 1
        if before < 0 or runs[before][2] != -value:
            continue
        breath_start, breath_stop, _ = runs[before]
        breath_end = smooth[breath_stop - 1]
        depth = abs(breath_end - smooth[breath_start])
        hold_level = float(np.median(smooth[following[0] : following[1]]))
        if abs(hold_level - breath_end) <= max_settle_fraction * depth:
            direction[start:stop] = 0
    return direction


def _offline(direction: np.ndarray, turn: np.ndarray, fs: float, min_hold_s: float, reach_s: float) -> np.ndarray:
    labels = np.full(len(direction), IGNORE, dtype=np.int8)
    runs = label_runs(direction)
    min_hold = int(round(min_hold_s * fs))
    reach = int(round(reach_s * fs))
    motion_label = {1: INHALE, -1: EXHALE}
    splits = []
    for index, (start, stop, value) in enumerate(runs):
        if value != 0:
            labels[start:stop] = motion_label[value]
            continue
        before = next((runs[j][2] for j in range(index - 1, -1, -1) if runs[j][2] != 0), None)
        after = next((runs[j][2] for j in range(index + 1, len(runs)) if runs[j][2] != 0), None)
        if stop - start >= min_hold:
            hold = HOLD_AFTER_INHALE if before == 1 else HOLD_AFTER_EXHALE
            labels[start:stop] = hold
            # The breath into the hold lasts until its turn: its slow end reads as still but the chest still moves.
            if index > 0 and runs[index - 1][2] != 0:
                lo, hi = max(runs[index - 1][0] + 1, start - reach), min(stop - 1, start + reach + 1)
                at = lo + int(np.argmax(turn[lo:hi]) if runs[index - 1][2] == 1 else np.argmin(turn[lo:hi]))
                if lo < at < hi - 1:
                    splits.append((lo, at, hi, motion_label[runs[index - 1][2]], hold))
            continue
        first = motion_label.get(before if before is not None else after, IGNORE)
        second = motion_label.get(after if after is not None else before, IGNORE)
        middle = (start + stop) // 2
        if before is not None and after is not None and before != after and 0 < index < len(runs) - 1:
            # A short pause is the turning point itself.  The next breath is seen to begin where its motion starts;
            # it starts at the turn within reach of that (the hand-labelling viewer snaps a click the same way).
            # The motion threshold puts that onset a little off the turn (the smoothing, a slow exhale tail).
            lo, hi = max(runs[index - 1][0] + 1, stop - reach), min(runs[index + 1][1] - 1, stop + reach + 1)
            at = lo + int(np.argmin(turn[lo:hi]) if after == 1 else np.argmax(turn[lo:hi]))
            if lo < at < hi - 1:  # an extreme at the edge of the reach is a slope, not a turn
                splits.append((min(start, lo), at, max(stop, hi), first, second))
                continue
        labels[start:middle] = first
        labels[middle:stop] = second
    for lo, at, hi, first, second in splits:  # after every run is labelled, so a neighbour cannot overwrite them
        labels[lo:at] = first
        labels[at:hi] = second
    return labels


def _causal(
    direction: np.ndarray,
    smooth: np.ndarray,
    fs: float,
    min_hold_s: float,
    min_phase_s: float,
    max_settle_s: float,
    max_settle_fraction: float,
) -> np.ndarray:
    labels = np.full(len(direction), IGNORE, dtype=np.int8)
    min_hold = int(round(min_hold_s * fs))
    min_phase = int(round(min_phase_s * fs))
    max_settle = int(round(max_settle_s * fs))
    current = IGNORE
    since_switch = min_phase
    still = 0
    breath_start = 0
    # The breath before this one: its class, depth and where it ended.
    previous: tuple[int, float, float] = (IGNORE, 0.0, 0.0)
    for index, value in enumerate(direction):
        since_switch += 1
        wanted = current
        if value > 0:
            still = 0
            wanted = INHALE
        elif value < 0:
            still = 0
            wanted = EXHALE
        else:
            still += 1
            if still >= min_hold and current in HOLD_OF:
                wanted = HOLD_OF[current]
                # Was this "breath" only the settle after the one before?
                label, depth, end = previous
                short = index - still - breath_start <= max_settle
                if label in HOLD_OF and label != current and short and abs(smooth[index] - end) <= max_settle_fraction * depth:
                    wanted = HOLD_OF[label]
        if wanted != current and (current == IGNORE or since_switch >= min_phase):
            if wanted in HOLD_OF:
                if current in HOLD_OF:
                    previous = (current, abs(smooth[index] - smooth[breath_start]), float(smooth[index]))
                else:
                    previous = (IGNORE, 0.0, 0.0)
                breath_start = index
            current = wanted
            since_switch = 0
        labels[index] = current
    return labels


# Settings tuned on the 19 corrected coach recordings (one subject, 15 s holds)
# by a small grid over low-pass, still threshold and minimum hold.  "Still"
# is deliberately generous -- slower than 40-50 % of the typical breathing
# speed -- because a real hold is not motionless: the chest keeps sinking a
# little and the heartbeat ripples on top.  The minimum hold is kept at 2 s
# offline rather than the grid's 3 s so the setting is not fitted to this
# protocol's long holds.
# Slow-motion check (offline only); None = off.  On by default since 2026-10-02: it helped the coach runs (+1.5 points), was neutral on the belt
# data judged by the 4 s hold rule and on the radar against the phone labels.
DEFAULT_SLOW_MOTION_FRACTION: float | None = 0.25
_DEFAULT: Any = object()

# A hold is a stillness of at least 4 s (natural rests between breaths last 0.8-3.1 s); it was 2 s before 2026-10-02.
OFFLINE_SETTINGS = {"low_pass_hz": 0.5, "still_fraction": 0.5, "min_hold_s": 4.0}
REALTIME_SETTINGS = {"low_pass_hz": 1.0, "still_fraction": 0.4, "min_hold_s": 1.0}
# Offline, a breath starts at the turn of the trace smoothed to 1 Hz within 0.8 s of where its motion begins
# (the hand-labelling viewer snaps a click to the same turn).
TURN_LOW_PASS_HZ = 1.0
TURN_REACH_S = 0.8

# Offline settings for fast breathing.  The ones above are made for 6-30 breaths/min: smoothing to 0.5 Hz, the 1.5 s
# slow-motion window and 0.4 s phases swallow breaths faster than about 45/min.  Where the trace has a clear rhythm
# above a tier's rate (dominant frequency over a sliding window), that stretch takes the tier's lighter settings;
# everywhere else the labels are exactly those of the settings above.
FAST_TIERS: tuple[tuple[float, dict[str, Any]], ...] = (
    (0.6, {"low_pass_hz": 1.2, "slow_motion_window_s": 0.75, "min_phase_s": 0.25, "turn_reach_s": 0.4, "turn_low_pass_hz": 1.5}),
    (0.9, {"low_pass_hz": 2.0, "slow_motion_fraction": None, "min_phase_s": 0.15, "turn_reach_s": 0.25, "turn_low_pass_hz": 2.0}),
)
RATE_WINDOW_S = 20.0
RATE_DOMINANCE = 0.4  # share of the 0.1-2.5 Hz power in the peak +-0.1 Hz: a rhythm, not noise or a hold
# ... and as large as breathing: a hold shows the heartbeat (about 1 Hz, a few % of a breath) as a clear rhythm too.
RATE_MIN_DEPTH = 0.2  # size of the window's rhythm (5th-95th percentile) over the recording's median


def local_rate_hz(chest: np.ndarray, fs: float, window_s: float = RATE_WINDOW_S, step_s: float = 2.0) -> np.ndarray:
    """Dominant breathing frequency around each sample, 0 where no single rhythm dominates (holds, noise, irregular)
    or where that rhythm is much smaller than the recording's breathing (the heartbeat in a hold)."""

    chest = np.asarray(chest, dtype=float)
    n, width, step = len(chest), int(round(window_s * fs)), max(1, int(round(step_s * fs)))
    rate = np.zeros(n)
    if n < width or width < 8:
        return rate
    freqs = np.fft.rfftfreq(width, 1.0 / fs)
    band = (freqs >= 0.1) & (freqs <= min(2.5, 0.45 * fs))
    taper = np.hanning(width)
    starts = list(range(0, n - width + 1, step))
    found: list[tuple[int, float, float]] = []
    for start in starts:
        part = chest[start : start + width]
        part = part - np.polyval(np.polyfit(np.arange(width), part, 1), np.arange(width))
        power = np.abs(np.fft.rfft(part * taper)) ** 2
        total = float(power[band].sum())
        if total <= 0:
            continue
        peak = float(freqs[band][np.argmax(power[band])])
        near = band & (np.abs(freqs - peak) <= 0.1)
        share = float(power[near].sum()) / total
        # Size of the rhythm itself: the window's component within 0.1 Hz of the peak.
        rhythm = np.fft.irfft(np.where(near, np.fft.rfft(part), 0.0), width)
        found.append((start, peak if share >= RATE_DOMINANCE else 0.0, float(np.subtract(*np.percentile(rhythm, [95, 5])))))
    typical = float(np.median([spread for _, _, spread in found])) if found else 0.0
    for start, peak, spread in found:
        centre = slice(start + (width - step) // 2, start + (width + step) // 2)
        rate[centre] = peak if spread >= RATE_MIN_DEPTH * typical else 0.0
    first, last = (width - step) // 2, (n - width) // step * step + (width + step) // 2
    rate[:first], rate[last:] = rate[first], rate[last - 1]
    return rate


def detect_phases(
    chest: np.ndarray,
    fs: float,
    *,
    causal: bool = False,
    low_pass_hz: float | None = None,
    still_fraction: float | None = None,
    min_hold_s: float | None = None,
    min_phase_s: float = 0.4,
    scale_window_s: float = 30.0,
    max_settle_s: float = 0.8,
    max_settle_fraction: float = 0.4,
    slow_motion_fraction: float | None = _DEFAULT,
    slow_motion_window_s: float = 1.5,
    detailed: bool = False,
    adaptive: bool = True,
) -> np.ndarray:
    """Phase labels (exhale, hold, inhale; IGNORE before the first motion) for a chest signal rising on inhale.

    ``detailed=True`` names each hold after the breath before it (the two-hold fallback scheme).

    ``slow_motion_fraction`` (offline only; default :data:`DEFAULT_SLOW_MOTION_FRACTION`) also counts a slow movement as motion when it changes
    the trace by that fraction of the local breath depth within ``slow_motion_window_s``; the phone labeller uses
    0.25 (see :func:`_slow_motion`).

    Settings left as ``None`` come from :data:`REALTIME_SETTINGS` or
    :data:`OFFLINE_SETTINGS`.  Offline and with ``adaptive`` (the default), stretches of clearly fast breathing take
    the lighter settings of :data:`FAST_TIERS`; the rest is unchanged.
    """

    defaults = REALTIME_SETTINGS if causal else OFFLINE_SETTINGS
    if slow_motion_fraction is _DEFAULT:
        slow_motion_fraction = DEFAULT_SLOW_MOTION_FRACTION
    low_pass_hz = defaults["low_pass_hz"] if low_pass_hz is None else low_pass_hz
    still_fraction = defaults["still_fraction"] if still_fraction is None else still_fraction
    min_hold_s = defaults["min_hold_s"] if min_hold_s is None else min_hold_s
    chest = np.asarray(chest, dtype=float)
    settings = dict(low_pass_hz=low_pass_hz, still_fraction=still_fraction, min_hold_s=min_hold_s, min_phase_s=min_phase_s,
                    scale_window_s=scale_window_s, max_settle_s=max_settle_s, max_settle_fraction=max_settle_fraction,
                    slow_motion_fraction=slow_motion_fraction, slow_motion_window_s=slow_motion_window_s)
    if causal:
        return merge_holds(_detect(chest, fs, causal=True, **settings), detailed)
    labels = _detect(chest, fs, **settings)
    if adaptive:
        rate = local_rate_hz(chest, fs)
        for floor, tier in FAST_TIERS:
            fast = rate >= floor
            if not fast.any():
                break
            lighter = {
                **settings,
                "low_pass_hz": max(low_pass_hz, tier["low_pass_hz"]),
                "min_phase_s": min(min_phase_s, tier["min_phase_s"]),
                "slow_motion_window_s": min(slow_motion_window_s, tier.get("slow_motion_window_s", slow_motion_window_s)),
                "slow_motion_fraction": tier.get("slow_motion_fraction", slow_motion_fraction),
            }
            labels = np.where(fast, _detect(chest, fs, turn_reach_s=tier["turn_reach_s"], turn_low_pass_hz=tier["turn_low_pass_hz"], **lighter), labels)
    return merge_holds(labels.astype(np.int8), detailed)


def _detect(
    chest: np.ndarray,
    fs: float,
    *,
    causal: bool = False,
    low_pass_hz: float,
    still_fraction: float,
    min_hold_s: float,
    min_phase_s: float,
    scale_window_s: float,
    max_settle_s: float,
    max_settle_fraction: float,
    slow_motion_fraction: float | None,
    slow_motion_window_s: float,
    turn_reach_s: float = TURN_REACH_S,
    turn_low_pass_hz: float = TURN_LOW_PASS_HZ,
) -> np.ndarray:
    """One pass of the detector with fixed settings; detailed holds."""

    smooth = light_filter(chest, fs, low_pass_hz, causal=causal)
    if causal:
        velocity = np.diff(smooth, prepend=smooth[0]) * fs
    else:
        velocity = np.gradient(smooth) * fs
    scale = _speed_scale(np.abs(velocity), fs, scale_window_s, causal=causal)
    threshold = still_fraction * scale
    direction = np.where(velocity > threshold, 1, np.where(velocity < -threshold, -1, 0)).astype(np.int8)
    if slow_motion_fraction is not None and not causal:
        direction = np.where(direction == 0, _slow_motion(smooth, fs, slow_motion_fraction, slow_motion_window_s), direction).astype(np.int8)
    if causal:
        return _causal(direction, smooth, fs, min_hold_s, min_phase_s, max_settle_s, max_settle_fraction)
    direction = _absorb_settles(direction, smooth, fs, min_hold_s, max_settle_s, max_settle_fraction)
    turn = light_filter(chest, fs, turn_low_pass_hz)
    return enforce_min_duration(_offline(direction, turn, fs, min_hold_s, turn_reach_s), fs, min_phase_s)
