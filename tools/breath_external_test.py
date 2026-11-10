#!/usr/bin/env python
"""External test of the breathing-phase network: other people, another radar, references independent of it.

The network was trained on one person (the author) sitting in front of the A121.  Here it runs, unchanged, on two
public 24 GHz continuous-wave radar datasets with their own breathing reference:

* Schellenberger et al. 2020 (30 people lying; resting, Valsalva, tilt up and down, apnea): thoracic impedance
  ``tfm_z0`` -- it rises on inhalation (air conducts poorly);
* Shi et al. 2020 (11 people, radar at the front or back of the chest; free breathing, breath holds after inhaling
  and after exhaling, apnea, after exercise, standing, lying, speech, movement): a temperature airflow sensor at the
  nose -- warm on exhaling, so it is turned over.

Labels come from the reference with the deterministic detector (offline settings), exactly as the belt labels of the
author's data.  The radar's I/Q becomes displacement (circle-fit demodulation, as ``tools/build_radar_ecg_windows.py``)
at 20 Hz; which way is "inhale up" is decided from the radar alone by the shape of the breath (the polarity test of
the range selection: inhaling is quicker than exhaling), and how often that agrees with the reference is reported.
The lag between radar and reference is measured by cross-correlation (within +-1.5 s) and the labels are shifted.

The network (TCN pretrained + fine-tuned, three seeds averaged; echo channel 0, as in its belt pretraining) runs over
each whole recording with its causal normalisation; Viterbi decoding; scoring starts after the first 20 s.  The
deterministic detector on the radar (offline and causal) is scored the same way.

    uv run --extra ml python tools/breath_external_test.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import scipy.io as sio
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from breath_phases_at_night import MODELS
from build_radar_ecg_windows import displacement_mm
from respi_net.a121_range import score_displacement
from respi_net.chest_signal import light_filter
from respi_net.nn_dataset import causal_normalise
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import score_phases, transition_log_matrix
from respi_net.phase_model import load_checkpoint
from train_breath_phase_model import pooled_boundaries, viterbi_windows

SCHELL = ROOT / "data" / "external" / "schellenberger2020"
SHI = ROOT / "data" / "external" / "shi2020" / "datasets"
OUT = ROOT / "runs" / "breath_external"
FS = 20.0
SKIP_S = 20.0
HOLD = 1


def _band(x: np.ndarray, fs: float) -> np.ndarray:
    return sosfiltfilt(butter(2, (0.08, 0.8), "bandpass", fs=fs, output="sos"), x - np.mean(x))


def lag_samples(radar: np.ndarray, reference: np.ndarray, max_s: float = 1.5) -> int:
    """Shift (samples) that best aligns ``reference`` to ``radar`` (positive: the reference lags)."""

    a, b = _band(radar, FS), _band(reference, FS)
    best, best_c = 0, -np.inf
    for k in range(-int(max_s * FS), int(max_s * FS) + 1):
        c = np.dot(a[max(0, -k) : len(a) - max(0, k)], b[max(0, k) : len(b) - max(0, -k)])
        if c > best_c:
            best, best_c = k, c
    return best


def radar_polarity(chest: np.ndarray) -> int:
    scores = score_displacement(chest[:, None], FS)
    return int(scores.polarity()[0])


def recordings() -> list[dict[str, Any]]:
    out = []
    for path in sorted(SCHELL.glob("GDN*/GDN*_*.mat")):
        m = sio.loadmat(path)
        fs_r, fs_z = float(m["fs_radar"].squeeze()), float(m["fs_z0"].squeeze())
        z = m["tfm_z0"].ravel().astype(float)
        if np.mean(np.isfinite(z)) < 0.9:
            continue
        z = np.where(np.isfinite(z), z, np.nanmedian(z))
        disp = displacement_mm(m["radar_i"].ravel(), m["radar_q"].ravel())
        radar = resample_poly(disp - disp.mean(), 1, int(round(fs_r / FS)))
        ref = resample_poly(z - z.mean(), 1, int(round(fs_z / FS)))
        n = min(len(radar), len(ref))
        out.append({"dataset": "schellenberger", "person": path.stem.split("_")[0], "scenario": path.stem.split("_", 2)[-1],
                    "radar": radar[:n], "reference": ref[:n], "reference_up_is_inhale": True})
    for path in sorted(SHI.rglob("*.mat")):
        placement = path.relative_to(SHI).parts[1]
        if "radar_neck" in placement:
            continue
        m = sio.loadmat(path)
        fs = float(m["Fs"].squeeze())
        disp = displacement_mm(m["radar_I"].ravel(), m["radar_Q"].ravel())
        radar = resample_poly(disp - disp.mean(), 1, int(round(fs / FS)))
        resp = m["respiration"].ravel().astype(float)
        ref = resample_poly(resp - resp.mean(), 1, int(round(fs / FS)))
        n = min(len(radar), len(ref))
        if n < int(40 * FS):
            continue
        out.append({"dataset": "shi", "person": path.relative_to(SHI).parts[0],
                    "scenario": path.parent.name, "radar": radar[:n], "reference": ref[:n], "reference_up_is_inhale": False})
    return out


def network_paths(chests: list[np.ndarray], models, transitions) -> list[np.ndarray]:
    paths = []
    with torch.no_grad():
        for chest in chests:
            feats = np.vstack([chest, np.gradient(chest) * FS, np.zeros_like(chest)])
            x = torch.from_numpy(causal_normalise(feats, FS))[None]
            stacked = torch.stack([torch.log_softmax(m(x), dim=1) for m in models])
            lp = (torch.logsumexp(stacked, dim=0) - math.log(len(models))).numpy()
            paths.append(viterbi_windows(lp, transitions)[0])
    return paths


def holds_found(y: np.ndarray, p: np.ndarray, min_s: float = 5.0) -> tuple[int, int]:
    """Reference holds of at least ``min_s``, and how many the prediction calls a hold for half of their length."""

    total = found = 0
    k = 0
    while k < len(y):
        if y[k] == HOLD:
            j = k
            while j < len(y) and y[j] == HOLD:
                j += 1
            if (j - k) / FS >= min_s:
                total += 1
                found += int(np.mean(p[k:j] == HOLD) >= 0.5)
            k = j
        else:
            k += 1
    return total, found


def score(ys: list[np.ndarray], ps: list[np.ndarray]) -> dict[str, Any]:
    skip = int(SKIP_S * FS)
    y = np.concatenate([a[skip:] for a in ys]); p = np.concatenate([b[skip:] for b in ps])
    s = score_phases(y, p)
    width = int(40 * FS)
    yw = np.stack([a[skip : skip + width] for a in ys if len(a) >= skip + width])
    pw = np.stack([b[skip : skip + width] for a, b in zip(ys, ps) if len(a) >= skip + width])
    boundary = pooled_boundaries(yw, pw, np.zeros(len(yw)), FS)
    holds = [holds_found(a[skip:], b[skip:]) for a, b in zip(ys, ps)]
    return {"accuracy": s.accuracy, "macro_f1": s.macro_f1, "f1": {k: v["f1"] for k, v in s.per_class.items() if v["support"]},
            "boundary_error_s": boundary["mean_abs_error_s"], "false_per_min": boundary["false_per_minute"],
            "holds": int(sum(h[0] for h in holds)), "holds_found": int(sum(h[1] for h in holds)), "recordings": len(ys)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    loaded = [load_checkpoint(p) for p in MODELS]
    models = [m for m, _ in loaded]
    transitions = transition_log_matrix(math.log(loaded[0][1]["decoding"]["switch_prob"]))
    recs = recordings()
    print(f"{len(recs)} recordings", flush=True)
    for r in recs:
        ref = light_filter(r["reference"] if r["reference_up_is_inhale"] else -r["reference"], FS)
        chest = light_filter(r["radar"], FS)
        pol = radar_polarity(chest)
        corr = np.corrcoef(_band(chest, FS), _band(ref, FS))[0, 1]
        r["polarity_by_shape"], r["polarity_by_reference"] = pol, int(np.sign(corr))
        chest = chest * (pol if pol != 0 else 1)
        lag = lag_samples(chest, ref)
        ref = np.roll(ref, -lag)
        r["lag_s"] = lag / FS
        r["chest"], r["labels"] = chest, detect_phases(ref, FS)
        r["det_offline"], r["det_causal"] = detect_phases(chest, FS), detect_phases(chest, FS, causal=True)
    paths = network_paths([r["chest"] for r in recs], models, transitions)
    for r, p in zip(recs, paths):
        r["net"] = p
    result: dict[str, Any] = {}
    for dataset in ("schellenberger", "shi"):
        group = [r for r in recs if r["dataset"] == dataset]
        agree = np.mean([r["polarity_by_shape"] == r["polarity_by_reference"] for r in group if r["polarity_by_shape"] != 0])
        result[dataset] = {"recordings": len(group), "people": len({r["person"] for r in group}),
                           "polarity_agreement": float(agree), "polarity_unsure": float(np.mean([r["polarity_by_shape"] == 0 for r in group])),
                           "lag_median_s": float(np.median([r["lag_s"] for r in group])), "scenarios": {}}
        for scenario in ["all", *sorted({r["scenario"] for r in group})]:
            sub = group if scenario == "all" else [r for r in group if r["scenario"] == scenario]
            if len(sub) < 2:
                continue
            result[dataset]["scenarios"][scenario] = {name: score([r["labels"] for r in sub], [r[key] for r in sub])
                                                      for name, key in (("network", "net"), ("detector_offline", "det_offline"), ("detector_causal", "det_causal"))}
        print(dataset, json.dumps({k: v for k, v in result[dataset].items() if k != "scenarios"}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    lines = []
    for dataset, res in result.items():
        lines += [f"### {dataset}: {res['recordings']} recordings, {res['people']} people; radar polarity by shape agrees with the "
                  f"reference in {100 * res['polarity_agreement']:.0f}% (unsure {100 * res['polarity_unsure']:.0f}%); median lag {res['lag_median_s']:+.2f} s\n",
                  "| scenario | n | method | accuracy % | macro-F1 | F1 exhale/hold/inhale | boundary error s | false /min | holds >= 5 s found |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for scenario, methods in res["scenarios"].items():
            for name, row in methods.items():
                f = row["f1"]
                f1 = " / ".join(f"{f.get(c, float('nan')):.2f}" for c in ("exhale", "hold", "inhale"))
                lines.append(f"| {scenario} | {row['recordings']} | {name} | {100 * row['accuracy']:.1f} | {row['macro_f1']:.3f} | {f1} | "
                             f"{row['boundary_error_s']:.2f} | {row['false_per_min']:.1f} | {row['holds_found']} of {row['holds']} |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
