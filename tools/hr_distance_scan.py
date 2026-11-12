#!/usr/bin/env python
"""Which distance of a night gives the best heart rate?  Every range bin as the centre of the network's gate.

For each still window of the gated night pipeline (with a watch heart rate), the final heart-rate networks
(``runs/sleep_heart_rate/final_shared-pre_s{0,1,2}.pt``, averaged) run on the 7 bins around every bin in turn -- fewer
at the ends of the recorded range -- and their estimate is compared with the watch.  Also per bin: the plain
spectral peak (42-120 bpm) of the centre bin and the share of breathing (0.1-0.6 Hz) in its motion spectrum.
Per-window choices are scored too: the Acconeer target (as the pipeline and the network's training), the bin with
the most breathing, the bin where the network is most confident.

The final networks were trained on the June watch nights: on those nights the scan is in-sample, on the November nights not.

    uv run --extra ml python tools/hr_distance_scan.py --night 2026-11-10
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

from build_sleep_epochs import ALL_NIGHTS
from train_sleep_heart_rate import (A121_CHEST_SIGN, BINS, FS, NIGHT_DIR, OUT_DIR, PAD, PHASE_TO_MM, WINDOW, garmin_labels,
                                    make_model, predict, regular_frame_times_ms, still_mask)

OUT = ROOT / "runs" / "hr_distance_scan"


def load_models(variant: str = "shared-pre") -> list[torch.nn.Module]:
    models = []
    for seed in (0, 1, 2):
        model = make_model("shared")
        model.load_state_dict(torch.load(OUT_DIR / f"final_{variant}_s{seed}.pt", map_location="cpu"))
        model.eval()
        models.append(model)
    return models


def gate(centre: int, n_bins: int) -> slice:
    return slice(max(0, centre - BINS // 2), min(n_bins, centre + BINS // 2 + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--night", default="2026-11-10")
    args = parser.parse_args()
    night = next(n for n in ALL_NIGHTS if n.name == args.night)
    trend = pd.read_csv(NIGHT_DIR / f"{night.stem}_gated_sleep_vitals.csv")
    with np.load(NIGHT_DIR / f"{night.stem}.npz") as raw:
        stamps, dist, iq, target = raw["timestamp_ms"], raw["distances_m"], raw["iq"], raw["acconeer_target_m"]
    t_s = (stamps - stamps[0]) / 1000.0
    regular = regular_frame_times_ms(stamps)
    complex_iq = iq[..., 0].astype(np.float32) + 1j * iq[..., 1].astype(np.float32)
    phase = A121_CHEST_SIGN * np.unwrap(np.angle(complex_iq), axis=0) * PHASE_TO_MM
    los, centres, tgt = [], [], []
    for end in trend["elapsed_s"].to_numpy()[still_mask(trend)]:
        hi = int(np.searchsorted(t_s, end)) + PAD
        lo = hi - WINDOW - 2 * PAD
        if lo < 0 or hi > len(t_s):
            continue
        finite = target[lo:hi][np.isfinite(target[lo:hi]) & (target[lo:hi] > 0)]
        los.append(lo); centres.append(regular[lo + PAD + WINDOW // 2])
        tgt.append(int(np.argmin(np.abs(dist - np.median(finite)))) if len(finite) else len(dist) // 2)
    label = garmin_labels(np.asarray(centres), night.garmin)
    keep = np.isfinite(label)
    los, tgt, label = np.asarray(los)[keep], np.asarray(tgt)[keep], label[keep]
    print(f"{night.name}: {len(los)} still windows with a watch heart rate (median {np.median(label):.1f})", flush=True)
    models, device = load_models(), torch.device("cpu")
    f = np.fft.rfftfreq(WINDOW, 1 / FS)
    cardiac, breath = (f >= 0.7) & (f <= 2.0), (f >= 0.1) & (f <= 0.6)
    pred = np.full((len(dist), len(los)), np.nan); conf = pred.copy(); spec = pred.copy(); bshare = pred.copy()
    for c in range(len(dist)):
        g = gate(c, len(dist))
        x = np.stack([phase[lo : lo + WINDOW + 2 * PAD, g].T for lo in los]).astype(np.float32)
        x -= x.mean(-1, keepdims=True)
        outs = [predict(m, x, device) for m in models]
        pred[c] = np.mean([o[0] for o in outs], 0); conf[c] = np.mean([o[1] for o in outs], 0)
        centre = x[:, c - g.start, PAD : PAD + WINDOW]
        p = np.abs(np.fft.rfft((centre - centre.mean(1, keepdims=True)) * np.hanning(WINDOW), axis=1)) ** 2
        spec[c] = f[cardiac][p[:, cardiac].argmax(1)] * 60.0
        bshare[c] = p[:, breath].sum(1) / p[:, f > 0.05].sum(1)
    trend10 = lambda v: pd.Series(v).rolling(20, min_periods=5).median()
    rows = [{"distance_m": round(float(dist[c]), 2), "network_mae": np.abs(pred[c] - label).mean(),
             "network_r": np.corrcoef(pred[c], label)[0, 1], "r_10min": trend10(pred[c]).corr(trend10(label)),
             "spectral_peak_mae": np.abs(spec[c] - label).mean(), "breathing_share": np.median(bshare[c])} for c in range(len(dist))]
    table = pd.DataFrame(rows)
    idx = np.arange(len(los))
    choices = {"Acconeer target": tgt, "most breathing per window": np.nanargmax(bshare, 0),
               "most confident per window": np.nanargmax(conf, 0)}
    lines = [f"### {night.name}: heart rate by distance ({len(los)} still windows; watch median {np.median(label):.1f})\n",
             "| distance m | network MAE | r | r 10 min | spectral peak MAE | breathing share |", "|---|---|---|---|---|---|"]
    lines += [f"| {r.distance_m:.2f} | {r.network_mae:.2f} | {r.network_r:.2f} | {r.r_10min:.2f} | {r.spectral_peak_mae:.1f} | {r.breathing_share:.2f} |"
              for r in table.itertuples()]
    lines += ["", "| choice | MAE | r 10 min |", "|---|---|---|"]
    for name, sel in choices.items():
        p = pred[sel, idx]
        lines.append(f"| {name} | {np.abs(p - label).mean():.2f} | {trend10(p).corr(trend10(label)):.2f} |")
    lines.append(f"| the night's watch median, constant | {np.abs(np.median(label) - label).mean():.2f} | - |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{night.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
