#!/usr/bin/env python
"""CLI wrapper for A121 sleep phase/scoring plots and optional Garmin FIT overlay.

Reusable implementation lives in src/respi_net/a121_sleep.py.  This script is kept for direct
one-off use from the repository checkout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from respi_net.a121_sleep import _default_outputs, create_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A121 sleep vitals/phases with optional Garmin FIT overlay.")
    parser.add_argument("trend_csv", type=Path, help="*_gated_sleep_vitals.csv produced by analyze_a121_sleep_vitals_gated.py")
    parser.add_argument("garmin", nargs="*", type=Path, help="Optional Garmin .fit file(s) or directory(s)")
    parser.add_argument("--output", type=Path, help="Output PNG. Defaults beside trend_csv.")
    parser.add_argument("--csv-output", type=Path, help="Merged output CSV. Defaults beside trend_csv unless --no-save-data.")
    parser.add_argument("--score-output", type=Path, help="Sleep-score JSON. Defaults beside trend_csv unless disabled.")
    parser.add_argument("--sleep-phases", dest="sleep_phases", action="store_true", default=True)
    parser.add_argument("--no-sleep-phases", dest="sleep_phases", action="store_false")
    parser.add_argument("--sleep-score", dest="sleep_score", action="store_true", default=True)
    parser.add_argument("--no-sleep-score", dest="sleep_score", action="store_false")
    parser.add_argument("--save-data", dest="save_data", action="store_true", default=True)
    parser.add_argument("--no-save-data", dest="save_data", action="store_false")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    garmin_sources = args.garmin or None
    default_png, default_csv, default_score = _default_outputs(args.trend_csv, garmin=bool(garmin_sources))
    output = args.output or default_png
    csv_output = (args.csv_output or default_csv) if args.save_data else None
    score_output = (args.score_output or default_score) if args.save_data and args.sleep_phases and args.sleep_score else None
    plot_path, merged_csv, score_json = create_plot(
        args.trend_csv,
        garmin_sources,
        output,
        csv_output,
        score_output,
        dpi=args.dpi,
        include_sleep=args.sleep_phases,
        include_score=args.sleep_score,
    )
    print(f"Saved plot: {plot_path}")
    if merged_csv is not None:
        print(f"Saved merged CSV: {merged_csv}")
    if score_json is not None:
        print(f"Saved radar sleep score: {score_json}")


if __name__ == "__main__":
    main()
