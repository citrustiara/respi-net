from __future__ import annotations

import numpy as np

from respi_net.chest_signal import regular_frame_times_ms


def _driver_stamps(frames: int, period_ms: float = 50.0, drift: float = 0.0017) -> np.ndarray:
    """Whole-millisecond stamps from a clock that runs ``drift`` fast (50, 50, 51, 50, ...)."""

    return np.floor(1000.0 + np.arange(frames) * period_ms * (1 + drift))


def test_the_driver_drift_is_removed() -> None:
    stamps = _driver_stamps(14721)
    assert stamps[-1] - stamps[0] > 14720 * 50.0 + 1000  # 1.2 s too long
    fixed = regular_frame_times_ms(stamps)
    assert abs((fixed[-1] - fixed[0]) - 14720 * 50.0) < 1.0
    assert fixed[0] == stamps[0]


def test_dropped_frames_keep_their_gap() -> None:
    stamps = _driver_stamps(400)
    stamps[200:] += 100.0  # two frames missing
    fixed = regular_frame_times_ms(stamps)
    steps = np.diff(fixed)
    assert np.isclose(steps[199], 150.0, atol=1.5)
    assert np.allclose(np.delete(steps, 199), 50.0)


def test_short_or_irregular_input_is_left_alone() -> None:
    few = np.array([0.0, 50.0, 100.0])
    assert np.array_equal(regular_frame_times_ms(few), few)
