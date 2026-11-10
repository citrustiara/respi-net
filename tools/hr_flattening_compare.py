#!/usr/bin/env python
"""How flat are the held-out heart rates of each network variant?  From ``predictions<tag>.npz`` of
``tools/train_sleep_heart_rate.py`` (every watch night held out once).

Per test night and variant, on 10-minute medians of the scored windows (as ``tools/hr_peak_tracking.py``): the slope
of the estimate on the watch (1 = the watch's full swing), the error inside the watch's peaks (more than one SD
above its night median) and outside them, MAE on all windows, and the 10-minute r.  Seeds are averaged first.

    uv run python tools/hr_flattening_compare.py --tag _balanced_s0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from hr_peak_tracking import slope, ten_minutes

HR_DIR = ROOT / "runs" / "sleep_heart_rate"


def flattening(estimate: np.ndarray, watch: np.ndarray) -> dict[str, float]:
    est10, watch10 = ten_minutes(estimate), ten_minutes(watch)
    peak = watch10 > np.nanmedian(watch10) + np.nanstd(watch10)
    ok = np.isfinite(est10) & np.isfinite(watch10)
    covered = np.isfinite(estimate)
    return {"mae": float(np.mean(np.abs(estimate[covered] - watch[covered]))), "coverage": float(covered.mean()),
            "slope": slope(est10, watch10), "error_peak": float(np.nanmean(est10[peak] - watch10[peak])),
            "error_rest": float(np.nanmean(est10[~peak] - watch10[~peak])), "r_10min": float(np.corrcoef(est10[ok], watch10[ok])[0, 1]),
            "sd_ratio": float(np.nanstd(est10) / np.nanstd(watch10))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="_balanced_s0")
    args = parser.parse_args()
    data = np.load(HR_DIR / f"predictions{args.tag}.npz")
    keys = list(data.keys())
    nights = sorted({k.split("|")[0] for k in keys if k.endswith("|label")})
    variants = sorted({k.split("|")[1] for k in keys if k.endswith("|bpm")})
    rows = []
    for night in nights:
        watch = data[f"{night}|label"]
        rows.append({"night": night, "estimate": "pipeline", **flattening(data[f"{night}|pipeline"], watch)})
        for variant in variants:
            seeds = [data[k] for k in keys if k.startswith(f"{night}|{variant}|") and k.endswith("|bpm")]
            rows.append({"night": night, "estimate": variant, **flattening(np.mean(seeds, axis=0), watch)})
    table = pd.DataFrame(rows)
    mean = table.groupby("estimate", sort=False).mean(numeric_only=True).reset_index()
    lines = [f"### Flattening of the held-out heart rate ({args.tag.strip('_')})\n",
             "| night | estimate | MAE | coverage % | slope on watch | SD ratio | error in peaks | error elsewhere | r 10 min |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in [*table.itertuples(), *[type("Row", (), {**row, "night": "**mean**"}) for row in mean.to_dict("records")]]:
        lines.append(f"| {r.night} | {r.estimate} | {r.mae:.2f} | {100 * r.coverage:.0f} | {r.slope:.2f} | {r.sd_ratio:.2f} | "
                     f"{r.error_peak:+.2f} | {r.error_rest:+.2f} | {r.r_10min:.2f} |")
    out = HR_DIR / f"flattening{args.tag}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
