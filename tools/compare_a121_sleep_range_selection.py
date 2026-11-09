#!/usr/bin/env python
"""Night heart and breathing rate against the Garmin watch, with the range Acconeer picked and the range that breathes.

``tools/analyze_a121_sleep_vitals_gated.py`` is run on every night twice, ``--range-selection acconeer`` and
``--range-selection breathing`` (output suffixes ``_acconeer`` and ``_breathing``).  Both trends are scored on the
same windows: the sleep interval and non-awake windows that the sleep classifier finds on the Acconeer trend, each
matched to the nearest Garmin value within 2 min (as in the thesis tables).  Per night and rate:

* coverage -- the share of those windows with a radar value,
* bias (mean and median radar minus Garmin), mean absolute difference and Pearson r where both have a value,
* the same mean absolute difference and r on 10-minute medians (bins with at least 5 matched windows): the
  trend over the night, which the 1-minute values of the watch hide in their scatter,
* for the breathing selection, the share of windows in which the breathing score replaced Acconeer's range.

The heart rate is the smoothed ``hr_bpm``; the breathing rate is Acconeer's recorded rate (``rr_bpm``).  The range
selection can only change the heart rate: the analyzer takes Acconeer's recorded breathing rate whenever there is
one, so the breathing rate is the same in both runs.

    uv run python tools/compare_a121_sleep_range_selection.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from respi_net.a121_sleep import _merge_garmin_nearest, classify_radar_sleep, load_garmin_reference

TREND_DIR = ROOT / "data" / "processed" / "a121_sleep_range"
GARMIN_DIR = ROOT / "data" / "raw" / "a121"
# thesis name -> (recording stem, Garmin folder)
NIGHTS = {
    "noc 1 (krawędź, plano-convex)": ("a121_sleep_sleep-night-1_1-real_2026-06-08_02-50-10", "garmin_2026-06-08"),
    "noc 2 (krawędź, FZP)": ("a121_sleep_sleep-night-2-real_2026-06-09_02-18-14", "garmin_2026-06-09"),
    "zapis próbny (płaska, 2,5 h)": ("a121_sleep_sleep-night-3-real-flatcover_2026-06-10_02-35-45", "garmin_2026-06-10"),
    "nieudana geometria (1,6 m)": ("a121_sleep_sleep-night-4-real_2026-06-11_02-49-31", "garmin_reference_a121_sleep_sleep-night-4-real_2026-06-11_02-49-31"),
    "noc 3 (płaska, nad łóżkiem)": ("a121_sleep_sleep-night-5-real_2026-06-13_02-58-40", "garmin_a121_sleep_sleep-night-5-real_2026-06-13_02-58-40"),
}
RATES = {"hr": ("hr_bpm", "garmin_hr_bpm"), "rr": ("rr_bpm", "garmin_rr_bpm")}
TREND_BIN_S = 600.0
TREND_MIN_WINDOWS = 5


def _r(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else math.nan


def agreement(radar: pd.Series, garmin: pd.Series) -> dict[str, float]:
    """Agreement of window values indexed by elapsed seconds, and of their 10-minute medians."""

    both = radar.notna() & garmin.notna()
    diff = (radar - garmin)[both]
    pairs = pd.DataFrame({"radar": radar[both], "garmin": garmin[both]})
    group = (pairs.index.to_numpy(dtype=float) // TREND_BIN_S).astype(int)
    sizes = pairs.groupby(group).size()
    trend = pairs.groupby(group).median()[sizes >= TREND_MIN_WINDOWS]
    return {"coverage": float(radar.notna().mean()) if len(radar) else math.nan, "n": int(both.sum()),
            "bias": float(diff.mean()) if len(diff) else math.nan, "bias_median": float(diff.median()) if len(diff) else math.nan,
            "mae": float(diff.abs().mean()) if len(diff) else math.nan, "r": _r(pairs["radar"], pairs["garmin"]),
            "mae_10min": float((trend["radar"] - trend["garmin"]).abs().mean()) if len(trend) else math.nan,
            "r_10min": _r(trend["radar"], trend["garmin"]), "bins_10min": int(len(trend))}


def night_scores(stem: str, garmin_folder: str, trend_dir: Path = TREND_DIR) -> dict[str, Any]:
    trends = {mode: pd.read_csv(trend_dir / f"{stem}_{mode}_gated_sleep_vitals.csv") for mode in ("acconeer", "breathing")}
    classified = classify_radar_sleep(trends["acconeer"])
    asleep = classified.frame.loc[classified.onset_idx : classified.wake_idx]
    asleep = asleep[asleep["radar_sleep_phase"] != "awake"]
    garmin = load_garmin_reference(GARMIN_DIR / garmin_folder)
    out: dict[str, Any] = {"windows": int(len(asleep))}
    for mode, trend in trends.items():
        frame = trend.copy()
        frame["clock_time_dt"] = pd.to_datetime(frame["clock_time"])
        merged = _merge_garmin_nearest(frame, garmin).set_index("elapsed_s")
        rows = merged.loc[merged.index.intersection(asleep["elapsed_s"])]
        out[mode] = {rate: agreement(pd.to_numeric(rows[radar], errors="coerce"), pd.to_numeric(rows[ref], errors="coerce"))
                     for rate, (radar, ref) in RATES.items()}
        if mode == "breathing":
            source = rows["range_source"].astype(str)
            out[mode]["replaced_share"] = float((source == "breathing-score").mean())
    return out


def _f(value: float, digits: int = 1, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = []
    for rate, title in (("hr", "heart rate [bpm]"), ("rr", "breathing rate, Acconeer app [breaths/min]; the same in both runs")):
        lines += [f"### {title}\n", "| night | range | coverage % | bias | median | MAE | r | MAE 10 min | r 10 min | n |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for night, scores in summary.items():
            for mode in ("acconeer", "breathing"):
                row = scores[mode][rate]
                extra = f" ({100 * scores['breathing']['replaced_share']:.0f}% replaced)" if mode == "breathing" else ""
                lines.append(f"| {night} | {mode}{extra} | {_f(100 * row['coverage'], 0)} | {_f(row['bias'], 1, True)} | "
                             f"{_f(row['bias_median'], 1, True)} | {_f(row['mae'])} | {_f(row['r'], 2)} | {_f(row['mae_10min'])} | "
                             f"{_f(row['r_10min'], 2)} | {row['n']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trend-dir", type=Path, default=TREND_DIR)
    args = parser.parse_args()
    summary = {night: night_scores(stem, folder, args.trend_dir) for night, (stem, folder) in NIGHTS.items()
               if (args.trend_dir / f"{stem}_breathing_gated_sleep_vitals.csv").exists()}
    (args.trend_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    table = markdown(summary)
    (args.trend_dir / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
