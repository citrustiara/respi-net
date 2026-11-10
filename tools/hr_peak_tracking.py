#!/usr/bin/env python
"""Do the radar heart-rate estimates follow the watch's peaks?  Per watch night, on 10-minute medians.

A peak: the watch's 10-minute heart rate more than one standard deviation above its night median.  Reported: how
much of the watch's swing the estimate keeps (slope of the estimate on the watch), the mean error inside and outside
the peaks, the pipeline's coverage there, the share of movement and the watch's stage during the peaks.  Estimates:
the final network (``--variant shared-pre`` or ``shared``; in-sample on the June nights it was trained on, held out
on 10 November) and the gated night pipeline (not trained).

    uv run --extra ml python tools/hr_peak_tracking.py --variant shared-pre
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
import torch

from build_sleep_epochs import NIGHTS, OUT as EPOCHS
from hr_distance_scan import load_models
from train_sleep_heart_rate import NIGHT_DIR, night_data, predict

OUT = ROOT / "runs" / "hr_peak_tracking"


def ten_minutes(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rolling(20, min_periods=5, center=True).median().to_numpy()


def slope(estimate: np.ndarray, watch: np.ndarray) -> float:
    ok = np.isfinite(estimate) & np.isfinite(watch)
    return float(np.polyfit(watch[ok], estimate[ok], 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="shared-pre", choices=("shared", "shared-pre"))
    args = parser.parse_args()
    models = load_models(args.variant)
    epochs = pd.read_csv(EPOCHS)
    rows = []
    for night in NIGHTS:
        if night.garmin is None or not (NIGHT_DIR / f"{night.stem}.npz").exists():
            continue
        windows, trend = night_data(night, None)
        if len(windows.label) < 50:
            continue
        net = np.mean([predict(m, windows.x, torch.device("cpu"))[0] for m in models], 0)
        pipe = pd.to_numeric(trend.set_index("elapsed_s")["hr_bpm"], errors="coerce").reindex(windows.end_s).to_numpy()
        ep = epochs[epochs["night"] == night.name].set_index("elapsed_s")
        stage = ep["garmin_stage"].reindex(windows.end_s).to_numpy()
        moving = ep["movement_fraction"].reindex(windows.end_s).to_numpy()
        watch, net10, pipe10 = ten_minutes(windows.label), ten_minutes(net), ten_minutes(pipe)
        peak = watch > np.nanmedian(watch) + np.nanstd(watch)
        rows.append({"night": night.name, "trained_on": night.name != "2026-11-10", "windows": len(watch),
                     "network_slope": slope(net10, watch), "pipeline_slope": slope(pipe10, watch),
                     "network_error_peak": np.nanmean(net10[peak] - watch[peak]), "network_error_rest": np.nanmean(net10[~peak] - watch[~peak]),
                     "pipeline_error_peak": np.nanmean(pipe10[peak] - watch[peak]),
                     "pipeline_coverage_peak": np.mean(np.isfinite(pipe[peak])), "pipeline_coverage_rest": np.mean(np.isfinite(pipe[~peak])),
                     "movement_peak": np.nanmean(moving[peak]), "movement_rest": np.nanmean(moving[~peak]),
                     "rem_peak": np.mean(stage[peak] == "rem"), "light_peak": np.mean(stage[peak] == "light")})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"peaks_{args.variant}.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
