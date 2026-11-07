#!/usr/bin/env python3
"""Radar + belt folders from one long radar recording and the belt pieces captured beside it.

When the strain belt and the heart-rate strap take turns on BLE beside one continuous A121 recording
(``record-a121-sleep`` plus ``capture-strain-belt`` / ``capture-heart-strap`` in a loop), every belt piece
becomes one folder for ``tools/build_a121_belt_runs.py``::

    data/raw/nn/a121_belt/<name>_<HH-MM>/a121.csv   radar frames from 5 s before to 5 s after the piece
    data/raw/nn/a121_belt/<name>_<HH-MM>/belt.csv   the belt piece as captured
    data/raw/nn/a121_belt/<name>_<HH-MM>/source.json

The radar stamps are evened over the WHOLE recording first
(:func:`respi_net.chest_signal.regular_frame_times_ms`) and written into the cut ``Timestamp_ms``.  The
driver's stamps run 0.17% fast, so evening each cut on its own would start it from an already drifted
stamp: on 2026-11-07 the radar then trailed the belt by 1.1, 1.9 and 2.7 s at 10, 19 and 28 min, and by
0.0 s with the stamps evened first.

    uv run python tools/split_a121_belt_alternation.py RADAR.csv BELT1.csv BELT2.csv ... --name sitting_130cm_alt
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from respi_net.chest_signal import regular_frame_times_ms

RAW_DIR = ROOT / "data" / "raw" / "nn" / "a121_belt"
MARGIN_MS = 5000.0
MIN_PIECE_S = 60.0  # shorter belt pieces (a capture stopped early) are left out


def split(radar: Path, belts: list[Path], name: str, out_dir: Path = RAW_DIR, min_piece_s: float = MIN_PIECE_S) -> list[dict[str, object]]:
    frame = pd.read_csv(radar)
    raw = pd.to_numeric(frame["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    frame["Timestamp_ms"] = regular_frame_times_ms(raw)
    rows = []
    for belt_path in belts:
        belt = pd.read_csv(belt_path)
        start, end = float(belt["host_time_ms"].iloc[0]), float(belt["host_time_ms"].iloc[-1])
        row: dict[str, object] = {"belt": belt_path.name, "minutes": round((end - start) / 60000.0, 2)}
        cut = frame[(frame["Timestamp_ms"] >= start - MARGIN_MS) & (frame["Timestamp_ms"] <= end + MARGIN_MS)]
        if (end - start) / 1000.0 < min_piece_s:
            row["status"] = f"skipped: shorter than {min_piece_s:g} s"
        elif len(cut) < 0.9 * (end - start) / 50.0:
            row["status"] = "skipped: the radar does not cover the belt piece"
        else:
            folder = out_dir / f"{name}_{datetime.fromtimestamp(start / 1000.0):%Y-%m-%d_%H-%M}"
            folder.mkdir(parents=True, exist_ok=True)
            cut.to_csv(folder / "a121.csv", index=False)
            belt.to_csv(folder / "belt.csv", index=False)
            source = {
                "radar": str(radar.relative_to(ROOT)) if radar.is_relative_to(ROOT) else str(radar),
                "belt": str(belt_path.relative_to(ROOT)) if belt_path.is_relative_to(ROOT) else str(belt_path),
                "radar_stamps": "evened over the whole recording (regular_frame_times_ms) before cutting",
                "margin_s": MARGIN_MS / 1000.0,
            }
            (folder / "source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
            row.update({"status": "written", "folder": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder)})
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("radar", type=Path, help="the continuous record-a121-sleep CSV")
    parser.add_argument("belts", nargs="+", type=Path, help="belt pieces captured during it")
    parser.add_argument("--name", required=True, help="folder prefix, e.g. sitting_130cm_alt")
    parser.add_argument("-o", "--output-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--min-piece-s", type=float, default=MIN_PIECE_S)
    args = parser.parse_args()
    for row in split(args.radar, args.belts, args.name, args.output_dir, args.min_piece_s):
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
