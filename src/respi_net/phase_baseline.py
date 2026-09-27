"""Deterministic breathing-phase detector: the baseline a network has to beat.

It reads the direction of chest motion.  Rising is an inhale, falling an
exhale, and standing still for at least ``min_hold_s`` a hold -- after inhale
if the last breath went in, after exhale otherwise.  Shorter still moments
are turning points and are split between the breaths on either side.

Offline it sees the whole recording: smoothing is zero-phase and a hold is
labelled from its first still sample.  In real-time mode it sees only the
past: the smoothing lags, a reversal must be confirmed, and a hold can only
be declared once the stillness has lasted ``min_hold_s``.  That delay is not
a weakness of this detector but of deciding causally, and the network pays
it too.
"""

from __future__ import annotations

import numpy as np

from .breath_phases import EXHALE, HOLD_AFTER_EXHALE, HOLD_AFTER_INHALE, IGNORE, INHALE, label_runs
from .chest_signal import light_filter
from .phase_metrics import enforce_min_duration


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


def _offline(direction: np.ndarray, fs: float, min_hold_s: float) -> np.ndarray:
    labels = np.full(len(direction), IGNORE, dtype=np.int8)
    runs = label_runs(direction)
    min_hold = int(round(min_hold_s * fs))
    motion_label = {1: INHALE, -1: EXHALE}
    for index, (start, stop, value) in enumerate(runs):
        if value != 0:
            labels[start:stop] = motion_label[value]
            continue
        before = next((runs[j][2] for j in range(index - 1, -1, -1) if runs[j][2] != 0), None)
        after = next((runs[j][2] for j in range(index + 1, len(runs)) if runs[j][2] != 0), None)
        if stop - start >= min_hold:
            labels[start:stop] = HOLD_AFTER_INHALE if before == 1 else HOLD_AFTER_EXHALE
            continue
        # A short pause is the turning point itself: half to each side.
        middle = (start + stop) // 2
        first = motion_label.get(before if before is not None else after, IGNORE)
        second = motion_label.get(after if after is not None else before, IGNORE)
        labels[start:middle] = first
        labels[middle:stop] = second
    return labels


def _causal(direction: np.ndarray, fs: float, min_hold_s: float, min_phase_s: float) -> np.ndarray:
    labels = np.full(len(direction), IGNORE, dtype=np.int8)
    min_hold = int(round(min_hold_s * fs))
    min_phase = int(round(min_phase_s * fs))
    current = IGNORE
    since_switch = min_phase
    still = 0
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
            if still >= min_hold and current in (INHALE, EXHALE):
                wanted = HOLD_AFTER_INHALE if current == INHALE else HOLD_AFTER_EXHALE
        if wanted != current and (current == IGNORE or since_switch >= min_phase):
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
OFFLINE_SETTINGS = {"low_pass_hz": 0.5, "still_fraction": 0.5, "min_hold_s": 2.0}
REALTIME_SETTINGS = {"low_pass_hz": 1.0, "still_fraction": 0.4, "min_hold_s": 1.0}


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
) -> np.ndarray:
    """Phase labels (0-3, IGNORE before the first motion) for a chest signal rising on inhale.

    Settings left as ``None`` come from :data:`REALTIME_SETTINGS` or
    :data:`OFFLINE_SETTINGS`.
    """

    defaults = REALTIME_SETTINGS if causal else OFFLINE_SETTINGS
    low_pass_hz = defaults["low_pass_hz"] if low_pass_hz is None else low_pass_hz
    still_fraction = defaults["still_fraction"] if still_fraction is None else still_fraction
    min_hold_s = defaults["min_hold_s"] if min_hold_s is None else min_hold_s
    chest = np.asarray(chest, dtype=float)
    smooth = light_filter(chest, fs, low_pass_hz, causal=causal)
    if causal:
        velocity = np.diff(smooth, prepend=smooth[0]) * fs
    else:
        velocity = np.gradient(smooth) * fs
    scale = _speed_scale(np.abs(velocity), fs, scale_window_s, causal=causal)
    threshold = still_fraction * scale
    direction = np.where(velocity > threshold, 1, np.where(velocity < -threshold, -1, 0)).astype(np.int8)
    if causal:
        return _causal(direction, fs, min_hold_s, min_phase_s)
    return enforce_min_duration(_offline(direction, fs, min_hold_s), fs, min_phase_s)
