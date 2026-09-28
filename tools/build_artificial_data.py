#!/usr/bin/env python3
"""Training sets with artificial data, and how many minutes of each kind they hold.

Real labelled recordings are short, so the training splits are topped up with

* augmented copies of every real *training* run: played at 0.7-1.4x speed
  (the breathing rate changes and the labels stretch with it), with real A121
  noise -- the chest signal during the coached breath holds -- added at 3-30%
  of the breath size, and a slow drift.  Validation and test runs stay real;
* synthetic A121 runs (:func:`respi_net.artificial_runs.synthetic_run`): 5-35
  breaths/min, 0.2-12 mm deep, holds after inhale and exhale, sighs,
  heartbeat and motion bursts labelled as noise, read through the A121 model.

Two datasets in the nn_dataset format, ready for
``tools/train_breath_phase_model.py``, go next to the coach labels:

* ``dataset_pretrain_v1.npz``: the supervisor's belt recordings (real, split by
  recording as in ``dataset_szymanski_belt_v1.npz``), their augmented copies
  and the synthetic runs (training only);
* ``dataset_finetune_v1.npz``: the own A121 coach runs (real, split as in
  ``dataset_coach_v1.npz``) and their augmented copies (training only).

``artificial_summary.json`` lists the labelled minutes per source, class and
split.  Run ``tools/build_breath_phase_labels.py`` and
``tools/analyze_szymanski_dataset.py --export`` first.

    uv run python tools/build_artificial_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from respi_net.artificial_runs import augment_run, noise_bank_from_holds, synthetic_run
from respi_net.breath_phases import CLASS_NAMES, IGNORE, NUM_CLASSES
from respi_net.nn_dataset import LabelledRun, assign_splits, export_dataset, load_dataset, runs_in
from respi_net.szymanski_dataset import recordings, to_labelled_run

OUT = ROOT / "data" / "processed" / "breath_phases"
STRETCH = (0.7, 0.8, 0.9, 1.1, 1.25, 1.4)  # one augmented copy per factor
NOISE_RATIO = (0.03, 0.3)  # noise std over breath size, log-uniform per copy
FS = 20.0


def reference_splits(path: Path, runs: list[LabelledRun]) -> dict[str, str]:
    """The split each real run got in its own exported dataset, so nothing moves between splits."""

    if path.exists():
        known = {run_id: info["split"] for run_id, info in load_dataset(path)["meta"]["runs"].items()}
        if all(run.run_id in known for run in runs):
            return {run.run_id: known[run.run_id] for run in runs}
    return assign_splits(runs)


def minutes(runs: list[LabelledRun]) -> dict[str, float]:
    labels = np.concatenate([run.labels for run in runs]) if runs else np.zeros(0, np.int8)
    fs = runs[0].fs if runs else FS
    counts = np.bincount(labels[labels != IGNORE], minlength=NUM_CLASSES)
    return {"labelled": round(float(counts.sum()) / fs / 60.0, 1), **{name: round(float(c) / fs / 60.0, 1) for name, c in zip(CLASS_NAMES, counts)}}


def augment(runs: list[LabelledRun], splits: dict[str, str], bank, rng: np.random.Generator) -> list[LabelledRun]:
    low, high = np.log(NOISE_RATIO)
    return [
        augment_run(run, stretch, float(np.exp(rng.uniform(low, high))), bank, rng)
        for run in runs
        if splits[run.run_id] == "train"
        for stretch in STRETCH
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic-minutes", type=float, default=300.0, help="total length of the synthetic runs")
    parser.add_argument("--synthetic-run-minutes", type=float, default=5.0)
    parser.add_argument("--stride-s", type=float, default=10.0, help="window stride of the exported sets")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    coach = list(runs_in(OUT / "runs"))
    belt = [to_labelled_run(recording) for recording in recordings() if recording.sensor == "tensometer"]
    if not coach:
        raise SystemExit("No coach runs: run tools/build_breath_phase_labels.py first.")
    coach_splits = reference_splits(OUT / "dataset_coach_v1.npz", coach)
    belt_splits = reference_splits(OUT / "dataset_szymanski_belt_v1.npz", belt)
    bank = noise_bank_from_holds(coach)

    coach_aug = augment(coach, coach_splits, bank, rng)
    belt_aug = augment(belt, belt_splits, bank, rng)
    count = int(round(args.synthetic_minutes / args.synthetic_run_minutes))
    synthetic = [synthetic_run(index, args.synthetic_run_minutes * 60.0, FS, rng) for index in range(count)]

    train_only = {run.run_id: "train" for run in [*coach_aug, *belt_aug, *synthetic]}
    sets = {
        "dataset_pretrain_v1.npz": ([*belt, *belt_aug, *synthetic], {**belt_splits, **train_only},
                                    "Belt (Szymanski et al., corrected labels) + augmented copies + synthetic A121; local use only"),
        "dataset_finetune_v1.npz": ([*coach, *coach_aug], {**coach_splits, **train_only},
                                    "Own A121 coach runs + augmented copies of the training runs"),
    }
    summary: dict[str, object] = {
        "noise_bank_minutes": round(sum(segment.shape[-1] for segment in bank.segments) / FS / 60.0, 1),
        "stretch_factors": list(STRETCH),
        "noise_ratio": list(NOISE_RATIO),
        "sources": {
            "own_a121_coach": {"people": 1, **minutes(coach), "train": minutes([r for r in coach if coach_splits[r.run_id] == "train"])["labelled"]},
            "szymanski_belt": {"people": 3, **minutes(belt), "train": minutes([r for r in belt if belt_splits[r.run_id] == "train"])["labelled"]},
            "augmented_own_a121": {"copies_per_run": len(STRETCH), **minutes(coach_aug)},
            "augmented_belt": {"copies_per_run": len(STRETCH), **minutes(belt_aug)},
            "synthetic_a121": {"runs": count, **minutes(synthetic)},
        },
        "datasets": {},
    }
    for name, (runs, splits, description) in sets.items():
        meta = export_dataset(runs, OUT / name, window_s=60.0, stride_s=args.stride_s, warmup_s=20.0, splits=splits, description=description)
        summary["datasets"][name] = {part: meta["splits"][part]["windows"] for part in ("train", "val", "test")}
        print(f"{name}: " + ", ".join(f"{part} {windows} windows" for part, windows in summary["datasets"][name].items()))
    (OUT / "artificial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nLabelled minutes (noise bank: {summary['noise_bank_minutes']} min of real A121 holds)")
    header = f"{'source':22} {'labelled':>9} " + " ".join(f"{name[:12]:>12}" for name in CLASS_NAMES)
    print(header)
    for source, values in summary["sources"].items():
        print(f"{source:22} {values['labelled']:9.1f} " + " ".join(f"{values[name]:12.1f}" for name in CLASS_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
