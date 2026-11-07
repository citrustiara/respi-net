import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_split_cuts_one_folder_per_belt_piece_with_stamps_evened_over_the_whole_recording(tmp_path):
    import split_a121_belt_alternation as tool

    start = 1.79e12
    k = np.arange(20 * 600)  # 10 min of radar at 20 Hz, stamped 0.17% fast like the driver
    radar = pd.DataFrame({"Timestamp_ms": start + np.floor(k * 50.085), "Frame": k})
    radar_path = tmp_path / "radar.csv"
    radar.to_csv(radar_path, index=False)
    pieces = []
    for name, begin_s, length_s in (("late.csv", 400.0, 120.0), ("short.csv", 100.0, 20.0)):
        t = start + 1000.0 * (begin_s + np.arange(0, length_s, 0.1))
        path = tmp_path / name
        pd.DataFrame({"host_time_ms": t, "raw": -5.3e6}).to_csv(path, index=False)
        pieces.append(path)
    rows = tool.split(radar_path, pieces, "test", out_dir=tmp_path / "out")
    assert [row["status"] for row in rows] == ["written", "skipped: shorter than 60 s"]
    folder = Path(rows[0]["folder"]) if Path(rows[0]["folder"]).is_absolute() else tool.ROOT / rows[0]["folder"]
    cut = pd.read_csv(folder / "a121.csv")
    # Evened over the whole file: frame k sits at start + 50 k ms, not at the drifted start + 50.085 k.
    assert np.allclose(cut["Timestamp_ms"] - start, cut["Frame"] * 50.0)
    assert cut["Timestamp_ms"].iloc[0] <= start + 395_000 + 50 and cut["Timestamp_ms"].iloc[-1] >= start + 524_000
    assert (folder / "belt.csv").exists() and (folder / "source.json").exists()
