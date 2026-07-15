from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_hb100_a121_comparison import discover_unregistered_range_measurements  # noqa: E402


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
