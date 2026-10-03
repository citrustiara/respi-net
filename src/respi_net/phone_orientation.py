"""The phone's orientation as a breathing signal that keeps the lung level.

The usual phone trace is the accelerometer projected on one axis (:func:`respi_net.chest_signal.imu_chest_signal`).  After a
deep inhale into a hold that trace falls back to its baseline within seconds, so a hold after inhale looks like an exhale
followed by a hold at rest.  The phone, however, tilts in two directions; in the second one the offset stays through the hold
and disappears when the real exhale starts.  This module keeps both: the gravity direction is turned into a 2-D tilt
(degrees, in the plane perpendicular to the session's resting gravity), and the signal is the distance of that tilt from
the *rest orientation*, the tilt at the end of the preceding normal exhales.

Rest is estimated from the phone's own breath troughs (any trough detector would do), so no radar and no sign are needed
for the distance itself.  It is an offline signal: the rest orientation looks back, the smoothing is zero-phase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from .chest_signal import light_filter


def gravity_tilt(acc: np.ndarray, fs: float, *, low_pass_hz: float = 1.0) -> np.ndarray:
    """2-D tilt of the gravity direction in degrees, shape ``(n, 2)``, from accelerometer samples ``acc`` of shape ``(n, 3)``.

    The tilt is measured in the plane perpendicular to the median gravity direction of the whole recording; the two basis
    vectors are arbitrary but fixed, so only distances in this plane are meaningful.
    """

    acc = np.asarray(acc, dtype=float)
    smooth = light_filter(acc, fs, low_pass_hz, axis=0)
    unit = smooth / np.linalg.norm(smooth, axis=1, keepdims=True)
    rest = np.median(unit, axis=0)
    rest /= np.linalg.norm(rest)
    reference = np.eye(3)[int(np.argmin(np.abs(rest)))]
    e1 = reference - reference.dot(rest) * rest
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(rest, e1)
    return np.degrees(np.column_stack([unit @ e1, unit @ e2]))


def breath_troughs(trace: np.ndarray, fs: float, *, min_gap_s: float = 2.0, prominence: float = 0.2) -> np.ndarray:
    """Sample indices of the troughs (ends of exhales) of a trace that rises on inhale.

    ``prominence`` is a fraction of the trace's 5-95 percentile spread.
    """

    trace = np.asarray(trace, dtype=float)
    spread = float(np.subtract(*np.percentile(trace, [95, 5])))
    found, _ = find_peaks(-trace, distance=max(1, int(round(min_gap_s * fs))), prominence=max(prominence * spread, 1e-9))
    return found


def orientation_distance(
    tilt: np.ndarray, troughs: np.ndarray, fs: float, *, window_s: float = 30.0, mean_s: float = 0.5
) -> np.ndarray:
    """Distance (degrees) of ``tilt`` from the rest orientation, which is re-estimated at every trough.

    The rest orientation at time *t* is the median tilt at the troughs of the preceding ``window_s`` seconds (at least
    the last three troughs before *t*), each trough value taken as a median over ``mean_s``.  During a hold there are no
    new troughs, so the rest stays the one from the breathing before it: an elevated hold stays elevated.
    """

    n = len(tilt)
    half = max(1, int(round(mean_s * fs / 2)))
    at_trough = np.array([np.median(tilt[max(0, k - half) : k + half + 1], axis=0) for k in troughs]) if len(troughs) else np.zeros((0, 2))
    rest = np.empty_like(tilt)
    last = -1  # index into troughs of the latest trough at or before the current sample
    for i in range(n):
        while last + 1 < len(troughs) and troughs[last + 1] <= i:
            last += 1
        if last < 0:
            rest[i] = at_trough[0] if len(troughs) else np.median(tilt, axis=0)
            continue
        lo = last
        while lo > 0 and (i - troughs[lo - 1]) <= window_s * fs:
            lo -= 1
        lo = min(lo, max(0, last - 2))
        rest[i] = np.median(at_trough[lo : last + 1], axis=0)
    return np.linalg.norm(tilt - rest, axis=1)


def run_orientation_distance(run_prefix: Path | str, grid: np.ndarray, trace: np.ndarray) -> np.ndarray:
    """Orientation distance of one recorder run on ``grid`` (seconds from the run start), rest taken from the troughs of
    ``trace`` (the single-axis phone trace on the same grid, rising on inhale)."""

    import pandas as pd  # local: only the tools need it

    cues = pd.read_csv(f"{run_prefix}_cues.csv")
    origin = float(cues["start_wall_ms"].iloc[0])
    df = pd.read_csv(f"{run_prefix}_iphone.csv")
    t = (df["Time_ms"].to_numpy(float) - origin) / 1000.0
    order = np.argsort(t, kind="stable")
    t = t[order]
    fs = 1.0 / float(np.median(np.diff(t)))
    tilt_raw = gravity_tilt(df[["ax", "ay", "az"]].to_numpy(float)[order], fs)
    tilt = np.column_stack([np.interp(grid, t, tilt_raw[:, k]) for k in range(2)])
    grid_fs = 1.0 / float(np.median(np.diff(grid)))
    return orientation_distance(tilt, breath_troughs(trace, grid_fs), grid_fs)
