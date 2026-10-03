from __future__ import annotations

import numpy as np

from respi_net.phone_orientation import breath_troughs, gravity_tilt, orientation_distance

FS = 20.0


def _tilted(angle_a: np.ndarray, angle_b: np.ndarray) -> np.ndarray:
    """Unit gravity vectors for tilts of ``angle_a`` / ``angle_b`` degrees around two horizontal axes."""

    a, b = np.radians(angle_a), np.radians(angle_b)
    vec = np.stack([np.sin(a), np.sin(b), np.cos(a) * np.cos(b)], axis=1)
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)


def test_gravity_tilt_measures_degrees_in_the_plane_of_rest() -> None:
    t = np.arange(0.0, 20.0, 1 / FS)
    wiggle = 2.0 * np.sin(2 * np.pi * 0.2 * t)
    tilt = gravity_tilt(_tilted(wiggle, np.zeros_like(wiggle)), FS)
    assert abs(float(np.ptp(tilt, axis=0).max()) - 4.0) < 0.2  # +-2 degrees -> 4 degrees peak to peak


def test_a_hold_after_inhale_stays_elevated_in_the_second_tilt_direction() -> None:
    t = np.arange(0.0, 80.0, 1 / FS)
    # breaths 0-40 s: tilt a goes 0 -> 2 -> 0 every 5 s.  At 40 s a deep inhale: a spikes to 4 degrees and returns to 0 within
    # 3 s (the usual single-axis trace), but tilt b rises to 1.5 degrees and stays there until the real exhale at 60 s.
    a = np.where(t < 40, 1 - np.cos(2 * np.pi * t / 5.0), 0.0)
    spike = (t >= 40) & (t < 43)
    a[spike] = 4.0 * np.sin(np.pi * (t[spike] - 40) / 3.0)
    b = np.zeros_like(t)
    b[(t >= 40) & (t < 60)] = 1.5
    b[(t >= 60) & (t < 62)] = 1.5 * (1 - (t[(t >= 60) & (t < 62)] - 60) / 2.0)
    tilt = gravity_tilt(_tilted(a, b), FS)
    trace = a  # the single-axis trace rises on inhale
    troughs = breath_troughs(trace[: int(40 * FS)], FS)
    d = orientation_distance(tilt, troughs, FS)
    hold = (t >= 47) & (t < 57)
    after = (t >= 66) & (t < 70)
    assert d[hold].mean() > 3.0 * d[after].mean() + 0.5  # elevated through the hold, gone after the exhale
    assert trace[hold].max() < 0.2  # while the single-axis trace is back at its baseline
