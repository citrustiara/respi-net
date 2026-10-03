#!/usr/bin/env python3
"""Phone labels from the single-axis trace and from the orientation distance, each scored against the radar's labels.

The same deterministic detector (phone settings) runs on both phone signals; the radar's labels come from the detector on the
radar (offline settings).  The radar is not a reference, but at close range it is the only signal that shows the lung level
through a hold, so it is the yardstick for the one question here: does the orientation distance mend the holds after a deep
inhale without spoiling normal breathing?  Scores are unguarded, boundaries within 0.5 s count as found, and they are split
into hold episodes (each radar hold of at least 4 s with 4 s on both sides, so the inhale before and the exhale after are
included) and the rest.

    uv run python tools/score_orientation_vs_radar.py --out outputs/orientation_vs_radar.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from plot_orientation_vs_pca import GRID_HZ, episode_row, load, radar_holds
from respi_net.breath_phases import IGNORE, merge_holds
from respi_net.phase_baseline import detect_phases
from respi_net.phone_annotation import PHONE_DETECTOR, compare_labels

# The stretches labelled by hand (same windows as the annotations), and whether the orientation signal was developed on it.
WINDOWS = {
    "phase1_halfside_02": (10.0, 160.0, False),
    "phase1_lying_01": (20.0, 180.0, False),
    "self_lying_3min_01": (20.0, 158.5, True),
}
MARGIN_S = 4.0


def score(radar_labels: np.ndarray, labels: np.ndarray, grid: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    result = compare_labels(np.where(mask, radar_labels, IGNORE), np.where(mask, labels, IGNORE), grid, tolerance_s=0.5)
    return {"accuracy": result["accuracy"], "seconds": result["samples"] / GRID_HZ, "boundaries_found": result["boundaries_found"]}


def session_scores(session: str, start: float, end: float) -> dict:
    grid, trace, radar, distance = load(session)
    radar_labels = merge_holds(detect_phases(radar, GRID_HZ))
    phone = {
        "single_axis": merge_holds(detect_phases(trace, GRID_HZ, **PHONE_DETECTOR)),
        "orientation": merge_holds(detect_phases(distance, GRID_HZ, **PHONE_DETECTOR)),
    }
    window = (grid >= start) & (grid <= end)
    holds = radar_holds(grid, radar, start, end)
    episodes = np.zeros(len(grid), dtype=bool)
    for hold_start, hold_end, _ in holds:
        episodes |= (grid >= hold_start - MARGIN_S) & (grid <= hold_end + MARGIN_S)
    out: dict = {"window_s": [start, end], "scores": {}, "holds": []}
    for name, labels in phone.items():
        out["scores"][name] = {
            "all": score(radar_labels, labels, grid, window),
            "hold_episodes": score(radar_labels, labels, grid, window & episodes),
            "rest": score(radar_labels, labels, grid, window & ~episodes),
        }
    for hold_start, hold_end, _ in holds:
        old, new, own = (episode_row(grid, signal, None, hold_start, hold_end, None) for signal in (trace, distance, radar))
        out["holds"].append({"hold_s": [round(hold_start, 2), round(hold_end, 2)], "radar_kept": own[0], "kept_single_axis": old[0],
                             "kept_orientation": new[0], "drop_single_axis_s": old[1], "drop_orientation_s": new[1], "drop_radar_s": own[1]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "orientation_vs_radar.json")
    args = parser.parse_args()
    report = {}
    for session, (start, end, development) in WINDOWS.items():
        report[session] = {"developed_on": development, **session_scores(session, start, end)}
        print(f"\n{session}{' (signal developed on this one)' if development else ''}")
        for name, parts in report[session]["scores"].items():
            print(f"  {name:12s} " + "   ".join(f"{part} {value['accuracy'] * 100:5.1f}% ({value['seconds']:.0f} s)" for part, value in parts.items()))
        for hold in report[session]["holds"]:
            print(f"  hold {hold['hold_s'][0]:6.1f}-{hold['hold_s'][1]:6.1f} s  level kept: radar {hold['radar_kept'] * 100:4.0f}%  "
                  f"single axis {hold['kept_single_axis'] * 100:4.0f}%  orientation {hold['kept_orientation'] * 100:4.0f}%  "
                  f"half-way drop after the radar hold ends: radar {hold["drop_radar_s"]:+.1f}, single axis {hold["drop_single_axis_s"]:+.1f}, orientation {hold["drop_orientation_s"]:+.1f} s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=float))
    print(f"\nReport: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
