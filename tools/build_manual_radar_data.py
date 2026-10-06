#!/usr/bin/env python3
"""Build reviewed near-range radar labels, trace snapshots and artificial copies.

    uv run python tools/build_manual_radar_data.py

Labels are editable JSON intervals, not detector outputs. Sources and generated
samples are small tracked NPZ artifacts; large raw IQ CSVs remain local. When
raw CSVs are present, their SHA256 must match the snapshot used for review.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from respi_net.artificial_runs import enhanced_augment_run, noise_bank_from_holds, reduced_power_run
from respi_net.breath_phases import CLASS_COLOURS, CLASS_NAMES, IGNORE, label_runs
from respi_net.chest_signal import ChestSignal, a121_chest_signal
from respi_net.nn_dataset import LabelledRun, export_dataset, labelled_run, save_run

BASE = ROOT / "annotations" / "radar_close_v1"
FIGURES = ROOT / "docs" / "datasets" / "radar_close_v1"
SEED = 20261006
STRETCH = (0.7, 0.8, 0.9, 1.1, 1.25, 1.4)


def reviewed_labels(time_s: np.ndarray, annotation: dict) -> np.ndarray:
    labels = np.full(len(time_s), IGNORE, dtype=np.int8)
    previous_end = -np.inf
    for start, end, label in annotation["segments"]:
        if start < previous_end or end <= start or label not in range(5):
            raise ValueError("Invalid, overlapping or unordered annotation segments")
        previous_end = end
        labels[(time_s >= start) & (time_s < end)] = label
    # One mask per boundary: take the larger adjacent default unless a
    # reviewed, explicit override narrows/widens this particular edge.
    guards = {}
    for start, end, label in annotation["segments"]:
        guard = annotation["hold_boundary_guard_s"] if label in (1, 3) else annotation["boundary_guard_s"]
        for edge in (start, end):
            guards[edge] = max(guards.get(edge, 0), guard)
    for override in annotation.get("boundary_overrides", []):
        edge, guard = override["time_s"], override["guard_s"]
        if edge not in guards or not np.isfinite(guard) or guard < 0:
            raise ValueError("Boundary override must name an existing edge and a nonnegative finite margin")
        guards[edge] = guard
    for edge, guard in guards.items():
        if guard > 0:
            labels[np.abs(time_s - edge) <= guard] = IGNORE
    return labels


def load_source(annotation: dict) -> ChestSignal:
    path = ROOT / annotation["source"]
    snapshot = BASE / "sources" / f'{annotation["session"]}.npz'
    if snapshot.exists():
        with np.load(snapshot, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != meta["raw_sha256"]:
                raise ValueError(f"Raw recording changed since annotation: {path}")
            return ChestSignal("a121", data["time_s"], float(data["fs"]), data["chest"], "mm", data["raw"], data["echo_db"], meta)
    manifest = json.loads((ROOT / annotation["manifest"]).read_text())
    origin = manifest["measurements"][0]["measurement_start_wall_ms"]
    duration = manifest["trials"][0]["duration_s"]
    signal = a121_chest_signal(path, origin_ms=origin)
    valid = (signal.time_s >= 0) & (signal.time_s < duration)
    meta = {**signal.meta, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "raw_source": annotation["source"]}
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(snapshot, time_s=signal.time_s[valid], fs=signal.fs,
                        chest=signal.chest[valid], raw=signal.raw[valid], echo_db=signal.echo_db[valid], meta=json.dumps(meta))
    return ChestSignal("a121", signal.time_s[valid], signal.fs, signal.chest[valid], "mm", signal.raw[valid], signal.echo_db[valid], meta)


def artificial_copies(real: list[LabelledRun], splits: dict[str, str]) -> list[LabelledRun]:
    # Neither held-out waveforms nor their hold noise enter training.
    training = [run for run in real if splits[run.run_id] == "train"]
    bank = noise_bank_from_holds(training)
    rng = np.random.default_rng(SEED)
    copies = []
    for run in training:
        for index, stretch in enumerate(STRETCH):
            ratio = float(np.exp(rng.uniform(np.log(.03), np.log(.3))))
            drift = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6)[index]
            copy = enhanced_augment_run(run, stretch, ratio, bank, rng, drift_per_min=drift)
            copies.append(copy)
        for drop in (0, 6, 12):
            iq_rng = np.random.default_rng(SEED + 100 + len(copies) // 9)
            copies.append(reduced_power_run(run, drop, iq_rng))
    return copies


def plot_labels(run: LabelledRun, raw: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    edges = np.linspace(10, run.time_s[-1], 4)
    for ax, low, high in zip(axes, edges[:-1], edges[1:]):
        ax.plot(run.time_s, raw, color=".7", lw=.6, label="przed filtrem")
        ax.plot(run.time_s, run.features[0], color="black", lw=1, label="LP 2 Hz")
        for start, stop, label in label_runs(run.labels):
            if label != IGNORE:
                end = run.time_s[stop] if stop < len(run.labels) else run.time_s[-1] + 1 / run.fs
                ax.axvspan(run.time_s[start], end, color=CLASS_COLOURS[label], alpha=.25)
        ax.set_xlim(low, high)
        ax.set_ylim(float(np.min(run.features[0])) - 1, float(np.max(run.features[0])) + 1)
        ax.set_ylabel("mm"); ax.grid(alpha=.25)
    axes[0].set_title(run.run_id + " — wstępne etykiety z wykresu radaru; białe: pominięte")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("czas od początku nagrania [s]")
    fig.savefig(path, dpi=130); plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    real, splits = [], {}
    for path in sorted(BASE.glob("*.json")):
        annotation = json.loads(path.read_text())
        signal = load_source(annotation)
        labels = reviewed_labels(signal.time_s, annotation)
        run = labelled_run(annotation["session"], signal, labels, group=annotation["session"], meta={"annotation": str(path.relative_to(ROOT)), "label_origin": annotation["method"], "independently_validated": False})
        real.append(run); splits[run.run_id] = annotation["split"]
        plot_labels(run, signal.raw, FIGURES / f'{run.run_id}.png')
    copies = artificial_copies(real, splits)
    runs = real + copies
    for run in runs:
        save_run(run, BASE / "generated" / "runs" / f'{run.run_id}.npz')
        if run.run_id not in splits:
            splits[run.run_id] = splits[run.group]
    meta = export_dataset(runs, BASE / "generated" / "dataset_manual_radar_v1.npz", splits=splits, stride_s=10, description="Provisional visually reviewed near radar intervals; augmented training only; no independent physiological reference.")
    summary = {"seed": SEED, "real_runs": len(real), "artificial_runs": len(copies), "real_labelled_seconds": {run.run_id: round(run.labelled_s, 2) for run in real}, "real_class_seconds": {name: round(sum(np.sum(run.labels == i) / run.fs for run in real), 2) for i, name in enumerate(CLASS_NAMES)}, "artificial_labelled_seconds": round(sum(run.labelled_s for run in copies), 2), "splits": meta["splits"], "validation": "Signal-based labels; not independent physiological ground truth. No IMU labels and no training performed."}
    (BASE / "generated" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), constrained_layout=True)
    base = next(run for run in real if run.run_id == "phase1_halfside_02")
    examples = [base, next(run for run in copies if run.run_id == base.run_id + "__x0.7"), next(run for run in copies if run.run_id == base.run_id + "__x1.4"), next(run for run in copies if run.run_id == base.run_id + "__iq_drop12")]
    for ax, run in zip(axes, examples):
        ax.plot(run.time_s, run.features[0], lw=.8, color="black")
        ax.set_title(run.run_id); ax.set_xlim(10, run.time_s[-1]); ax.grid(alpha=.25); ax.set_ylabel("mm")
    fig.savefig(FIGURES / "artificial_examples.png", dpi=130); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
