#!/usr/bin/env python
"""The breathing-phase network on whole nights: does it find breaths in sleep, and per-30 s breathing features from it.

For every epoch of ``tools/build_sleep_epochs.py`` (a 60 s window ending at the epoch's end) the chest signal is
formed as in training -- the 3 bins around Acconeer's target, unwrapped, straight line removed, low-passed at
2 Hz, velocity and echo strength -- over 90 s so the causal 30 s normalisation has its full history, and the last
60 s go through the chosen network (TCN pretrained and fine-tuned, ``runs/breath_phase/compare_v1/pretrain_tcn_s*``,
the three seeds' log-probabilities averaged).  The Viterbi path over the window (its decoding settings) gives the
phases; from its last 40 s (after the 20 s the network uses as context):

* ``net_rate_bpm`` -- inhale onsets per minute; ``net_interval_cv`` -- variation of the intervals between onsets;
* ``net_inhale_fraction`` -- inhale time over inhale + exhale time;
* ``net_hold_share``, ``net_longest_hold_s`` -- pauses; ``net_noise_share`` -- the network's noise class.

Is it breathing?  ``net_rate_bpm`` is compared with Acconeer's breathing rate and with the watch's.

    uv run --extra ml python tools/breath_phases_at_night.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import torch
from scipy.signal import detrend

from build_sleep_epochs import NIGHT_DIR, NIGHTS, OUT as EPOCHS
from respi_net.chest_signal import A121_CHEST_SIGN, PHASE_TO_MM, light_filter
from respi_net.nn_dataset import causal_normalise
from respi_net.phase_metrics import transition_log_matrix
from respi_net.phase_model import load_checkpoint
from train_breath_phase_model import viterbi_windows

MODELS = [ROOT / "runs" / "breath_phase" / "compare_v1" / f"pretrain_tcn_s{seed}" / "model.pt" for seed in (0, 1, 2)]
OUT = ROOT / "data" / "processed" / "sleep_epochs" / "net_breath_features.csv"
FS = 20.0
WINDOW, HISTORY, CONTEXT = int(60 * FS), int(30 * FS), int(20 * FS)
EXHALE, HOLD, INHALE, NOISE = 0, 1, 2, 3


def chest_features(iq: np.ndarray, distances: np.ndarray, target: np.ndarray) -> np.ndarray:
    """[3, n] chest (mm, rising on inhale), velocity (mm/s), echo (dB) of the 3 bins around the median target."""

    z = iq[..., 0].astype(np.float32) + 1j * iq[..., 1].astype(np.float32)
    amplitude = np.median(np.abs(z), axis=0)
    finite = target[np.isfinite(target) & (target > 0)]
    centre = int(np.argmin(np.abs(distances - float(np.median(finite))))) if len(finite) else int(np.argmax(np.where(distances >= 0.2, amplitude, -np.inf)))
    gate = np.arange(max(0, centre - 1), min(len(distances), centre + 2))
    weight = amplitude[gate] / (float(amplitude[gate].sum()) + 1e-12)
    phase = np.unwrap(np.angle(z[:, gate]), axis=0)
    chest = A121_CHEST_SIGN * detrend(phase, axis=0, type="linear") @ weight * PHASE_TO_MM
    chest = light_filter(chest, FS)
    echo = np.abs(z[:, gate]) @ weight
    return np.vstack([chest, np.gradient(chest) * FS, 20.0 * np.log10(np.maximum(echo, 1e-9))])


def phase_features(path: np.ndarray) -> dict[str, float]:
    part = path[CONTEXT:]
    onsets = np.flatnonzero((part[1:] == INHALE) & (part[:-1] != INHALE)) + 1
    seconds = len(part) / FS
    intervals = np.diff(onsets) / FS
    breathing = np.isin(part, (INHALE, EXHALE))
    holds, longest, run = part == HOLD, 0, 0
    for h in holds:
        run = run + 1 if h else 0
        longest = max(longest, run)
    return {
        "net_rate_bpm": len(onsets) * 60.0 / seconds,
        "net_interval_cv": float(np.std(intervals) / np.mean(intervals)) if len(intervals) >= 3 else math.nan,
        "net_inhale_fraction": float(np.mean(part[breathing] == INHALE)) if breathing.any() else math.nan,
        "net_hold_share": float(holds.mean()),
        "net_longest_hold_s": longest / FS,
        "net_noise_share": float(np.mean(part == NOISE)),
    }


def night_features(night, models, transitions: np.ndarray, batch: int = 128) -> pd.DataFrame:
    trend = pd.read_csv(NIGHT_DIR / f"{night.stem}_gated_sleep_vitals.csv")
    with np.load(NIGHT_DIR / f"{night.stem}.npz") as raw:
        stamps, distances, iq, target = raw["timestamp_ms"], raw["distances_m"], raw["iq"], raw["acconeer_target_m"]
    t_s = (stamps - stamps[0]) / 1000.0
    inputs, keep = [], []
    for k, end in enumerate(trend["elapsed_s"]):
        hi = int(np.searchsorted(t_s, end))
        lo = hi - WINDOW - HISTORY
        if lo < 0:
            continue
        feats = chest_features(iq[lo:hi], distances, target[lo:hi])
        inputs.append(causal_normalise(feats, FS)[:, -WINDOW:])
        keep.append(k)
    x = torch.from_numpy(np.stack(inputs)) if inputs else torch.zeros((0, 3, WINDOW))
    log_probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            stacked = torch.stack([torch.log_softmax(m(x[i : i + batch]), dim=1) for m in models])
            log_probs.append(torch.logsumexp(stacked, dim=0).numpy() - math.log(len(models)))
    lp = np.concatenate(log_probs) if log_probs else np.zeros((0, 4, WINDOW))
    paths = viterbi_windows(lp, transitions)
    rows = [{"night": night.name, "elapsed_s": float(trend["elapsed_s"].iloc[k]), **phase_features(p)} for k, p in zip(keep, paths)]
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    loaded = [load_checkpoint(path) for path in MODELS]
    models = [m for m, _ in loaded]
    decoding = loaded[0][1]["decoding"]
    transitions = transition_log_matrix(math.log(decoding["switch_prob"]))
    frames = []
    for night in NIGHTS:
        if (NIGHT_DIR / f"{night.stem}.npz").exists():
            frames.append(night_features(night, models, transitions))
            print(f"{night.name}: {len(frames[-1])} epochs", flush=True)
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    epochs = pd.read_csv(EPOCHS, low_memory=False)[["night", "elapsed_s", "rr_bpm", "garmin_rr_bpm", "movement_fraction"]]
    merged = epochs.merge(out, on=["night", "elapsed_s"])
    still = merged["movement_fraction"] < 0.2
    for ref in ("rr_bpm", "garmin_rr_bpm"):
        ok = still & merged[ref].notna() & merged["net_rate_bpm"].notna()
        d = merged.loc[ok, "net_rate_bpm"] - merged.loc[ok, ref]
        r = np.corrcoef(merged.loc[ok, "net_rate_bpm"], merged.loc[ok, ref])[0, 1] if ok.sum() > 2 else math.nan
        print(f"net breathing rate vs {ref}: {ok.sum()} still epochs, bias {d.mean():+.2f}, MAE {d.abs().mean():.2f}, r {r:.2f}")
    print(merged.loc[still, ["net_inhale_fraction", "net_hold_share", "net_longest_hold_s", "net_noise_share"]].describe().round(3).to_string())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
