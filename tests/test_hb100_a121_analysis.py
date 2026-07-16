from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_hb100_a121_comparison import (  # noqa: E402
    discover_unregistered_range_measurements,
    estimate_hb100_harmonic_rhythm,
)


def test_unregistered_complete_range_retest_is_discovered_without_manifest_mutation(tmp_path: Path) -> None:
    stem = "run_10_range-150-r1_150cm"
    cues = tmp_path / f"{stem}_cues.csv"
    pd.DataFrame(
        {
            "start_s": [0.0],
            "end_s": [90.0],
            "kind": ["normal"],
            "start_wall_ms": [1_000.0],
        }
    ).to_csv(cues, index=False)
    (tmp_path / f"{stem}_hb100.csv").write_text("Timestamp_ms,Voltage_mV\n", encoding="utf-8")
    (tmp_path / f"{stem}_a121.csv").write_text("Timestamp_ms,PeakPhase_rad\n", encoding="utf-8")

    discovered = discover_unregistered_range_measurements(tmp_path, registered=[])

    assert len(discovered) == 1
    assert discovered[0]["status"] == "discovered_unregistered"
    assert discovered[0]["step"]["distance_cm"] == 150.0
    assert discovered[0]["files"]["cues"] == str(cues)


def test_harmonic_rhythm_estimator_recovers_fundamental_hidden_by_harmonics() -> None:
    sample_rate_hz = 100.0
    time_s = np.arange(0.0, 35.0, 1.0 / sample_rate_hz)
    rng = np.random.default_rng(17)
    signal_mv = (
        0.15 * np.sin(2.0 * np.pi * 0.20 * time_s)
        + 1.10 * np.sin(2.0 * np.pi * 0.40 * time_s + 0.3)
        + 0.65 * np.sin(2.0 * np.pi * 0.60 * time_s - 0.2)
        + rng.normal(0.0, 0.03, len(time_s))
    )

    estimate = estimate_hb100_harmonic_rhythm(time_s, signal_mv, 0.0, 34.99)

    assert abs(estimate.frequency_hz - 0.20) <= 0.003
    assert estimate.contributing_harmonics >= 2
    assert estimate.accepted is True


def test_harmonic_rhythm_estimator_rejects_single_component_ambiguity() -> None:
    sample_rate_hz = 100.0
    time_s = np.arange(0.0, 35.0, 1.0 / sample_rate_hz)
    signal_mv = np.sin(2.0 * np.pi * 0.40 * time_s)

    estimate = estimate_hb100_harmonic_rhythm(time_s, signal_mv, 0.0, 34.99)

    assert estimate.contributing_harmonics == 1
    assert estimate.score_vs_competitor < 1.05
    assert estimate.accepted is False
