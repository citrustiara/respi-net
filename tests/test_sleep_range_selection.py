import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from respi_net.chest_signal import A121_CHEST_SIGN, PHASE_TO_MM


def _window(breathing_bin: int, strong_bin: int, seconds: float = 60.0, fs: float = 20.0, seed: int = 0) -> pd.DataFrame:
    """A121 rows where one bin breathes and another, recorded as Acconeer's target, only echoes strongly."""

    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    distances = 0.15 + 0.02 * np.arange(24)
    # Quick inhale (40% of the breath), slower exhale: right way up by the shape of the breath.
    cycle = (t * 0.25) % 1.0
    chest_mm = np.where(cycle < 0.4, cycle / 0.4, 1.0 - (cycle - 0.4) / 0.6) * 4.0
    profile = 3.0 * (rng.standard_normal((len(t), len(distances))) + 1j * rng.standard_normal((len(t), len(distances))))
    profile[:, breathing_bin] += 200.0 * np.exp(1j * A121_CHEST_SIGN * chest_mm / PHASE_TO_MM)
    profile[:, strong_bin] += 1500.0
    as_text = lambda row: json.dumps([round(float(v), 4) for v in row])
    return pd.DataFrame({
        "Timestamp_ms": 1.0e12 + t * 1000.0,
        "Frame": np.arange(len(t)),
        "Distances_m": as_text(distances),
        "Real": [as_text(row) for row in profile.real],
        "Imag": [as_text(row) for row in profile.imag],
        "AcconeerTargetDistance_m": distances[strong_bin],
    })


def test_breathing_target_leaves_a_strong_still_echo_for_the_breathing_bin() -> None:
    from analyze_a121_sleep_vitals_gated import breathing_target

    work = _window(breathing_bin=14, strong_bin=4)
    target, source, span = breathing_target(work, 20.0)
    assert source == "breathing-score"
    assert target == pytest.approx(0.15 + 0.02 * 14, abs=0.025)
    assert span[0] <= target <= span[1]


def test_breathing_target_keeps_acconeer_when_its_range_breathes() -> None:
    from analyze_a121_sleep_vitals_gated import breathing_target

    work = _window(breathing_bin=10, strong_bin=17)
    work["AcconeerTargetDistance_m"] = 0.15 + 0.02 * 10
    target, source, _ = breathing_target(work, 20.0)
    assert source.startswith("acconeer")
    assert target == pytest.approx(0.15 + 0.02 * 10, abs=0.025)


def test_analyze_refuses_an_unknown_range_selection(tmp_path) -> None:
    from analyze_a121_sleep_vitals_gated import analyze

    with pytest.raises(ValueError, match="range_selection"):
        analyze(tmp_path / "missing.csv", tmp_path, range_selection="strongest")
