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

With ``--enhanced``, ``enhanced_v2/`` holds addition-only v2 datasets and
``*_v1_plus_v2.npz`` unions with content deduplication. Existing originals
and the identical synthetic runs remain in v1 only. All splits are provisional.

``artificial_summary.json`` lists the labelled minutes per source, class and
split, and ``--thesis-figure`` draws ``docs/thesis/figures/fazy_dane_sztuczne.png``
(a synthetic run, and a real run next to one of its copies); ``--notes-figure
PNG`` draws the same in English for local notes.  Run
``tools/build_breath_phase_labels.py`` and
``tools/analyze_szymanski_dataset.py --export`` first.

    uv run python tools/build_artificial_data.py --thesis-figure
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from respi_net.artificial_runs import augment_run, enhanced_augment_run, noise_bank_from_holds, reduced_power_run, synthetic_run
from respi_net.breath_phases import (
    CLASS_COLOURS,
    CLASS_NAMES,
    CLASS_NAMES_PL,
    EXHALE,
    HOLD_AFTER_EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    NOISE,
    NUM_CLASSES,
    label_runs,
)
from respi_net.nn_dataset import LabelledRun, assign_splits, export_dataset, load_dataset, runs_in, save_run
from respi_net.szymanski_dataset import recordings, to_labelled_run

OUT = ROOT / "data" / "processed" / "breath_phases"
FIGURE = ROOT / "docs" / "thesis" / "figures" / "fazy_dane_sztuczne.png"
TEXT = {
    "pl": {
        "y": "ruch klatki [mm]",
        "x": "czas [s]",
        "a": "a) nagranie syntetyczne po modelu radaru A121 (SNR {snr:.0f} dB)",
        "b": "b) fragment rzeczywistego nagrania A121",
        "c": "c) te same oddechy w kopii: {stretch} raza dłużej, z szumem radaru i dryfem",
        "decimal": ",",
        "names": CLASS_NAMES_PL,
    },
    "en": {
        "y": "chest motion [mm]",
        "x": "time [s]",
        "a": "a) synthetic recording after the A121 radar model (SNR {snr:.0f} dB)",
        "b": "b) part of a real A121 recording",
        "c": "c) the same breaths in a copy: {stretch}x as long, with radar noise and drift",
        "decimal": ".",
        "names": tuple(name.replace("_", " ") for name in CLASS_NAMES),
    },
}
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


def training_noise_bank(runs: list[LabelledRun], splits: dict[str, str]):
    """Real hold noise from the TRAINING runs only: a bank that included validation or test runs would put their signal into training copies."""

    return noise_bank_from_holds([run for run in runs if splits[run.run_id] == "train"])


def augment(runs: list[LabelledRun], splits: dict[str, str], bank, rng: np.random.Generator) -> list[LabelledRun]:
    low, high = np.log(NOISE_RATIO)
    return [
        augment_run(run, stretch, float(np.exp(rng.uniform(low, high))), bank, rng)
        for run in runs
        if splits[run.run_id] == "train"
        for stretch in STRETCH
    ]


def enhanced_copies(runs: list[LabelledRun], splits: dict[str, str], bank, seed: int) -> list[LabelledRun]:
    """Source labels survive corruption; only time stretching changes their grid."""
    rng = np.random.default_rng(seed)
    copies = []
    for index, run in enumerate(runs):
        if splits[run.run_id] != "train":
            continue
        for stretch, drift in zip(STRETCH, (.05, .1, .2, .3, .45, .6)):
            ratio = float(np.exp(rng.uniform(*np.log(NOISE_RATIO))))
            copies.append(enhanced_augment_run(run, stretch, ratio, bank, rng, drift_per_min=drift))
        if run.meta.get("units") == "mm":
            for drop in (0, 6, 12):
                copies.append(reduced_power_run(run, drop, np.random.default_rng(seed + 100 + index)))
    return [replace(run, run_id=run.run_id + "__enhanced_v2") for run in copies]


def merge_window_datasets(base_path: Path, addition_path: Path, output_path: Path) -> dict:
    """Union by window contents (features + labels), rejecting split conflicts.

    Original windows occur once. New run IDs have a v2 suffix, but their group
    remains the source group. Deduplication uses contents, not just names.
    """
    base, addition = load_dataset(base_path), load_dataset(addition_path)
    for key in ("fs", "window_s", "warmup_s", "channels", "normalisation"):
        if base["meta"][key] != addition["meta"][key]:
            raise ValueError(f"Incompatible datasets: {key}")
    group_splits = {}
    for data in (base, addition):
        for info in data["meta"]["runs"].values():
            group, split = info["group"], info["split"]
            if group in group_splits and group_splits[group] != split:
                raise ValueError("A source group appears on different sides of the split")
            group_splits[group] = split
    selected = {"X": [], "y": [], "split": [], "run_id": [], "start_s": []}
    seen = {}
    duplicates = 0
    for data in (base, addition):
        for index in range(len(data["y"])):
            fingerprint = hashlib.sha256(data["X"][index].tobytes() + data["y"][index].tobytes()).digest()
            split = str(data["split"][index])
            if fingerprint in seen:
                if seen[fingerprint] != split:
                    raise ValueError("Identical window appears on different sides of the split")
                duplicates += 1
                continue
            seen[fingerprint] = split
            for key in selected:
                selected[key].append(data[key][index])
    arrays = {key: np.asarray(values, dtype=base[key].dtype) for key, values in selected.items()}
    meta = {**base["meta"], "description": "v1 plus additional v2 augmentations; identical windows stored once; provisional splits",
            "runs": {**base["meta"]["runs"], **addition["meta"]["runs"]},
            "parents": [str(base_path), str(addition_path)], "duplicates_removed": duplicates}
    counts = {}
    for split in ("train", "val", "test"):
        mask = arrays["split"] == split
        labels = arrays["y"][mask]
        counts[split] = {"windows": int(np.sum(mask)), "runs": sorted(set(arrays["run_id"][mask])),
                         "class_samples": np.bincount(labels[labels >= 0], minlength=NUM_CLASSES).tolist()}
    meta["splits"] = counts
    np.savez_compressed(output_path, **arrays, meta=np.array(json.dumps(meta)))
    return {"windows": {split: counts[split]["windows"] for split in counts}, "duplicates_removed": duplicates}


def _shade(ax: plt.Axes, time_s: np.ndarray, chest: np.ndarray, labels: np.ndarray, title: str, ylabel: str) -> None:
    for start, stop, label in label_runs(labels):
        if label != IGNORE:
            ax.axvspan(time_s[start], time_s[min(stop, len(time_s) - 1)], color=CLASS_COLOURS[label], alpha=0.28, lw=0)
    ax.plot(time_s, chest, color="#111827", lw=1.0)
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(alpha=0.25)


def plot_examples(synthetic: list[LabelledRun], real: LabelledRun, copy: LabelledRun, path: Path, language: str = "pl") -> None:
    """a) 60 s of a synthetic run with both kinds of hold; b) 30 s of a real run; c) the same breaths in its copy."""

    width = int(60 * FS)
    best = (-1, synthetic[0], 0)
    for run in synthetic:
        for start in range(0, len(run.labels) - width, int(5 * FS)):
            present = set(np.unique(run.labels[start : start + width]))
            score = 2 * (HOLD_AFTER_INHALE in present) + 2 * (HOLD_AFTER_EXHALE in present) + (NOISE in present)
            if score > best[0]:
                best = (score, run, start)
    _, run, start = best
    stretch = float(copy.meta["stretch"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), gridspec_kw={"hspace": 0.55})
    part = slice(start, start + width)
    text = TEXT[language]
    _shade(axes[0], run.time_s[part] - run.time_s[start], run.features[0][part], run.labels[part],
           text["a"].format(snr=run.meta["snr_db"]), text["y"])
    original = slice(int(10 * FS), int(40 * FS))
    _shade(axes[1], real.time_s[original], real.features[0][original], real.labels[original], text["b"], text["y"])
    copied = slice(int(10 * FS * stretch), int(40 * FS * stretch))
    _shade(axes[2], copy.time_s[copied], copy.features[0][copied], copy.labels[copied],
           text["c"].format(stretch=f"{stretch:g}".replace(".", text["decimal"])), text["y"])
    axes[2].set_xlabel(text["x"])
    shown = (INHALE, EXHALE, HOLD_AFTER_INHALE, HOLD_AFTER_EXHALE, NOISE)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[label], alpha=0.5) for label in shown]
    fig.legend(handles, [text["names"][label] for label in shown], loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic-minutes", type=float, default=300.0, help="total length of the synthetic runs")
    parser.add_argument("--synthetic-run-minutes", type=float, default=5.0)
    parser.add_argument("--stride-s", type=float, default=10.0, help="window stride of the exported sets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thesis-figure", action="store_true")
    parser.add_argument("--notes-figure", type=Path, default=None, metavar="PNG", help="the same examples in English")
    parser.add_argument("--enhanced", action="store_true", help="v2: time-varying noise, wider drift and reduced-power A121 replays")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to a separate enhanced_v2 directory with --enhanced")
    parser.add_argument("--summary-path", type=Path, default=None, help="also save the aggregate summary at this path")
    parser.add_argument("--own-examples-dir", type=Path, default=None, help="save small own-radar examples (never belt files)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    coach = list(runs_in(OUT / "runs"))
    belt = [to_labelled_run(recording) for recording in recordings() if recording.sensor == "tensometer"]
    if not coach:
        raise SystemExit("No coach runs: run tools/build_breath_phase_labels.py first.")
    coach_splits = reference_splits(OUT / "dataset_coach_v1.npz", coach)
    belt_splits = reference_splits(OUT / "dataset_szymanski_belt_v1.npz", belt)
    bank = training_noise_bank(coach, coach_splits)

    coach_aug = augment(coach, coach_splits, bank, rng)
    belt_aug = augment(belt, belt_splits, bank, rng)
    count = int(round(args.synthetic_minutes / args.synthetic_run_minutes))
    synthetic = [synthetic_run(index, args.synthetic_run_minutes * 60.0, FS, rng) for index in range(count)]

    # Consume the legacy RNG draws above even in v2: purely synthetic runs
    # remain the same as v1 for the same settings. Real copies use a new stream.
    if args.enhanced:
        coach_aug = enhanced_copies(coach, coach_splits, bank, args.seed + 20261006)
        belt_aug = enhanced_copies(belt, belt_splits, bank, args.seed + 20261007)
    destination = args.output_dir or (OUT / "enhanced_v2" if args.enhanced else OUT)
    version = "v2" if args.enhanced else "v1"

    train_only = {run.run_id: "train" for run in [*coach_aug, *belt_aug, *synthetic]}
    sets = {
        f"dataset_pretrain_{version}.npz": ([*belt, *belt_aug, *synthetic], {**belt_splits, **train_only},
                                    "Belt (Szymanski et al., corrected labels) + augmented copies + synthetic A121; local use only"),
        f"dataset_finetune_{version}.npz": ([*coach, *coach_aug], {**coach_splits, **train_only},
                                    "Own A121 coach runs + augmented copies of the training runs"),
    }
    if args.enhanced:
        # V2 is an addition, not a second complete dataset. Real originals,
        # old augmentation and synthetic runs already exist in v1.
        sets = {
            "dataset_pretrain_v2.npz": (belt_aug, train_only, "Additional v2 belt augmentations only; local use only"),
            "dataset_finetune_v2.npz": (coach_aug, train_only, "Additional v2 own A121 augmentations only"),
        }
    summary: dict[str, object] = {
        "version": version,
        "relationship_to_v1": "addition_only; real sources and the 60 identical synthetic runs are stored in v1 only" if args.enhanced else "base",
        "seed": args.seed,
        "split_status": "PROVISIONAL: inherited technical grouping; finalize from the complete acquisition plan before model selection or evaluation",
        "label_transfer": "Existing source labels and IGNORE masks are inherited; nearest-neighbour time warp only; no detector or re-annotation on artificial signals",
        "enhancements": {"time_varying_extra_noise": bool(args.enhanced), "drift_per_min": [.05, .1, .2, .3, .45, .6] if args.enhanced else [.3], "radar_power_drop_db": [0, 6, 12] if args.enhanced else [], "reduced_power_belt": False},
        "noise_bank_minutes": round(sum(segment.shape[-1] for segment in bank.segments) / FS / 60.0, 1),
        "stretch_factors": list(STRETCH),
        "noise_ratio": list(NOISE_RATIO),
        "sources": {
            "own_a121_coach": {"people": 1, **minutes(coach), "train": minutes([r for r in coach if coach_splits[r.run_id] == "train"])["labelled"]},
            "szymanski_belt": {"people": 3, **minutes(belt), "train": minutes([r for r in belt if belt_splits[r.run_id] == "train"])["labelled"]},
            "augmented_own_a121": {"copies_per_run": 9 if args.enhanced else len(STRETCH), "runs": len(coach_aug), **minutes(coach_aug)},
            "augmented_belt": {"copies_per_run": len(STRETCH), "runs": len(belt_aug), **minutes(belt_aug)},
            "synthetic_a121": {"runs": 0 if args.enhanced else count, **minutes([] if args.enhanced else synthetic)},
        },
        "datasets": {},
    }
    for name, (runs, splits, description) in sets.items():
        meta = export_dataset(runs, destination / name, window_s=60.0, stride_s=args.stride_s, warmup_s=20.0, splits=splits, description=description + "; splits provisional; source labels inherited")
        summary["datasets"][name] = {part: meta["splits"][part]["windows"] for part in ("train", "val", "test")}
        print(f"{name}: " + ", ".join(f"{part} {windows} windows" for part, windows in summary["datasets"][name].items()))
    if args.enhanced:
        summary["combined"] = {}
        for kind in ("pretrain", "finetune"):
            result = merge_window_datasets(OUT / f"dataset_{kind}_v1.npz", destination / f"dataset_{kind}_v2.npz",
                                          destination / f"dataset_{kind}_v1_plus_v2.npz")
            summary["combined"][kind] = result
            print(f"Combined {kind}: {result}")
    (destination / "artificial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.enhanced:
        # Remove former v2 names produced by this builder before the addition
        # convention, so an old full export cannot be mistaken for extra data.
        for path in (destination / "copies").glob("*.npz"):
            if not path.stem.endswith("__enhanced_v2"):
                path.unlink()
        for run in [*coach_aug, *belt_aug]:
            save_run(run, destination / "copies" / f"{run.run_id}.npz")
    if args.own_examples_dir:
        real = next(run for run in coach if coach_splits[run.run_id] == "train")
        save_run(real, args.own_examples_dir / "source.npz")
        for suffix in ("__x0.7", "__x1.4", "__iq_drop12"):
            copy = next((run for run in coach_aug if run.run_id == real.run_id + suffix + ("__enhanced_v2" if args.enhanced else "")), None)
            if copy is not None:
                save_run(copy, args.own_examples_dir / f"example{suffix}.npz")

    print(f"\nLabelled minutes (noise bank: {summary['noise_bank_minutes']} min of real A121 holds)")
    header = f"{'source':22} {'labelled':>9} " + " ".join(f"{name[:12]:>12}" for name in CLASS_NAMES)
    print(header)
    for source, values in summary["sources"].items():
        print(f"{source:22} {values['labelled']:9.1f} " + " ".join(f"{values[name]:12.1f}" for name in CLASS_NAMES))
    copy = next(run for run in coach_aug if run.meta.get("stretch") == 1.25)
    real = next(run for run in coach if run.run_id == copy.meta["augmented_from"])
    if args.thesis_figure:
        plot_examples(synthetic, real, copy, FIGURE, "pl")
        print(f"\nFigure: {FIGURE}")
    if args.notes_figure:
        plot_examples(synthetic, real, copy, args.notes_figure, "en")
        print(f"Notes figure: {args.notes_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
