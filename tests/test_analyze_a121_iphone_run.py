from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import analyze_a121_iphone_run as tool  # noqa: E402

FS = 20.0


def _breaths(time_s: np.ndarray, shift_s: float = 0.0) -> np.ndarray:
    return np.sin(2 * np.pi * (time_s - shift_s) / 5.0) + 0.3 * np.sin(2 * np.pi * (time_s - shift_s) / 13.0)


def test_best_lag_finds_the_delay_of_the_second_signal() -> None:
    t = np.arange(0, 90, 1 / FS)
    reference, other = _breaths(t), _breaths(t, shift_s=0.3)
    lag, corr, _, _ = tool.best_lag(reference, other, FS)
    assert abs(lag - 0.3) < 0.03
    assert corr > 0.99


def test_best_lag_is_negative_when_the_second_signal_leads() -> None:
    t = np.arange(0, 90, 1 / FS)
    lag, _, _, _ = tool.best_lag(_breaths(t), _breaths(t, shift_s=-0.2), FS)
    assert abs(lag + 0.2) < 0.03


def test_shifted_brings_a_late_signal_back() -> None:
    t = np.arange(0, 60, 1 / FS)
    late = _breaths(t, shift_s=0.4)
    back = tool.shifted(t, late, 0.4)
    assert np.max(np.abs(back[50:-50] - _breaths(t)[50:-50])) < 0.05


def test_event_differences_report_phone_minus_radar() -> None:
    t = np.arange(0, 60, 1 / FS)
    radar = np.sin(2 * np.pi * t / 5.0)
    phone = np.sin(2 * np.pi * (t + 0.25) / 5.0)  # the phone is 0.25 s early
    # inhale = the rising half of each cycle (sin from -1 to +1), exhale = the falling half
    cues = pd.DataFrame(
        [
            {"start_s": 3.75 + 5 * k + offset, "end_s": 3.75 + 5 * k + offset + 2.5, "kind": kind}
            for k in range(1, 10)
            for offset, kind in ((0.0, "inhale"), (2.5, "exhale"))
        ]
    )
    differences = tool.event_differences(t, radar, phone, cues, FS)
    pooled = np.concatenate([values[:, 1] for values in differences.values()])
    assert abs(np.median(pooled) + 0.25) < 0.1


def test_chunk_delays_follow_a_drifting_signal() -> None:
    t = np.arange(0, 300, 1 / FS)
    reference = _breaths(t)
    drifting = np.interp(t, t * 1.002, reference)  # the second signal's clock runs 0.2 percent slow
    rows = tool.chunk_delays(t, reference, drifting, 0.0, FS)
    delays = [row["delay_s"] for row in rows if row["correlation"] > 0.9]
    assert len(delays) >= 4
    assert delays[-1] - delays[0] > 0.3  # the second signal falls further behind along the recording


def test_long_holds_lists_only_long_holds() -> None:
    t = np.arange(0, 30, 0.1)
    labels = np.full(len(t), 2, dtype=np.int8)
    labels[50:60] = 1  # 1 s hold
    labels[100:180] = 3  # 8 s hold
    assert [(round(b), round(e), c) for b, e, c in tool.long_holds(labels, t)] == [(10, 18, 3)]


def test_cued_breaths_are_recognised() -> None:
    assert tool.has_cued_breaths(pd.DataFrame({"kind": ["normal", "inhale"]}))
    assert not tool.has_cued_breaths(pd.DataFrame({"kind": ["settle", "normal", "interference"]}))
