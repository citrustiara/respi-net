#!/usr/bin/env python
"""Three ways of training the breathing-phase network, on both architectures, against the plain detector.

Every variant is scored on the same validation and test windows of the fine-tuning set (the author's
belt-labelled radar sessions and the held-out coach runs; no copies, no synthetic data):

    real      real radar only -- the fine-tuning set without its time-stretched copies (``--exclude-runs __x``)
    copies    the whole fine-tuning set: real radar + copies
    pretrain  first the pretraining set (Szymanski belt, own belt traces, their copies, synthetic A121),
              then the whole fine-tuning set starting from those weights (``--init``)

for ``tcn`` and ``gru`` and a few seeds each.  Beside them, :func:`respi_net.phase_baseline.detect_phases`
is run on the chest channel of the very same windows, offline and causal, and scored the same way.  The
network's Viterbi path is decoded over the whole window (it looks ahead within it), so it is compared with
the offline detector; its live smoother (``online``) is causal and compared with the realtime detector.

Runs are written to ``runs/breath_phase/compare_v1/<variant>_<model>_s<seed>/`` by
``tools/train_breath_phase_model.py`` and are skipped when their ``model.pt`` exists, so an interrupted
comparison resumes.  TCN trains on Apple's GPU (``mps``), GRU on the CPU, where each is faster.

    uv run --extra ml python tools/compare_breath_phase_training.py
    uv run --extra ml python tools/compare_breath_phase_training.py --seeds 0 --models tcn   # a quick look
    uv run --extra ml python tools/compare_breath_phase_training.py --summary-only
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from respi_net.breath_phases import CLASS_NAMES
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import score_phases
from train_breath_phase_model import _jsonable, load_windows, pooled_boundaries

DATA_DIR = ROOT / "data" / "processed" / "breath_phases"
PRETRAIN = DATA_DIR / "dataset_pretrain_v1.npz"
FINETUNE = DATA_DIR / "dataset_finetune_v1.npz"
OUT_DIR = ROOT / "runs" / "breath_phase" / "compare_v1"
TRAIN_SCRIPT = ROOT / "tools" / "train_breath_phase_model.py"
VARIANTS = ("real", "copies", "pretrain")
MODELS = ("tcn", "gru")
SEEDS = (0, 1, 2)
DEVICE = {"tcn": "mps", "gru": "cpu"}
COMMON = ["--augment"]  # random depth and small shifts on every variant alike
SCORED_SPLITS = ("val", "test")


# ------------------------------------------------------------------- jobs


def run_dir(stage: str, model: str, seed: int) -> Path:
    return OUT_DIR / f"{stage}_{model}_s{seed}"


def train_command(stage: str, model: str, seed: int) -> list[str]:
    dataset = PRETRAIN if stage == "pre" else FINETUNE
    command = [sys.executable, str(TRAIN_SCRIPT), str(dataset), "--model", model, "--seed", str(seed),
               "--device", DEVICE[model], "--out-dir", str(OUT_DIR), "--name", run_dir(stage, model, seed).name,
               "--overwrite", *COMMON]
    if stage == "real":
        command += ["--exclude-runs", "__x"]
    if stage == "pretrain":
        command += ["--init", str(run_dir("pre", model, seed) / "model.pt")]
    return command


def run_job(stage: str, model: str, seed: int, threads: int, lock: threading.Lock) -> Path:
    folder = run_dir(stage, model, seed)
    if (folder / "model.pt").exists():
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    with lock:
        print(f"[{time.strftime('%H:%M:%S')}] start {folder.name}", flush=True)
    log = OUT_DIR / f"{folder.name}.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(train_command(stage, model, seed), cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    with lock:
        status = "done" if result.returncode == 0 else f"FAILED ({result.returncode}), see {log}"
        print(f"[{time.strftime('%H:%M:%S')}] {status} {folder.name} in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    if result.returncode:
        raise RuntimeError(f"{folder.name} failed; see {log}")
    return folder


def run_grid(models: list[str], seeds: list[int], variants: list[str], gru_workers: int, tcn_workers: int) -> None:
    """TCN jobs share the GPU, GRU jobs the CPU cores; a fine-tune waits for its own pretraining."""

    lock = threading.Lock()
    threads = max(1, (os.cpu_count() or 4) // max(1, gru_workers))
    pools = {"tcn": ThreadPoolExecutor(tcn_workers), "gru": ThreadPoolExecutor(gru_workers)}
    futures: list[Future] = []
    for model in models:
        pool = pools[model]
        for seed in seeds:
            if "pretrain" in variants:
                # Pretraining then fine-tuning run back to back in one slot.
                futures.append(pool.submit(lambda m=model, s=seed: (run_job("pre", m, s, threads, lock),
                                                                    run_job("pretrain", m, s, threads, lock))))
            for variant in variants:
                if variant != "pretrain":
                    futures.append(pool.submit(run_job, variant, model, seed, threads, lock))
    failures = 0
    for future in futures:
        try:
            future.result()
        except RuntimeError as error:
            failures += 1
            print(error, flush=True)
    for pool in pools.values():
        pool.shutdown()
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


# --------------------------------------------------------------- baseline


def baseline_scores(dataset: Path = FINETUNE) -> dict[str, dict[str, dict[str, Any]]]:
    """The plain detector on the chest channel of every val/test window, scored like the network."""

    data = load_windows(dataset)
    chest = data.X[:, data.channels.index("chest")]
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, causal in (("offline", False), ("realtime", True)):
        results[mode] = {}
        for split in SCORED_SPLITS:
            index = data.indices(split)
            predicted = np.stack([detect_phases(chest[i].astype(float), data.fs, causal=causal) for i in index])
            y = data.y[index]
            scores = score_phases(y.ravel(), predicted.ravel())
            results[mode][split] = {
                "accuracy": scores.accuracy,
                "macro_f1": scores.macro_f1,
                "per_class": scores.per_class,
                "boundary": pooled_boundaries(y, predicted, data.start_s[index], data.fs),
                # The detector leaves frames before the first motion undecided; those count as errors above.
                "undecided_samples": int(np.sum((y >= 0) & (predicted < 0))),
            }
    return results


# ---------------------------------------------------------------- summary


def _row(split: dict[str, Any], online: bool = False) -> dict[str, float]:
    part = split["online"] if online else split
    f1 = part.get("per_class")  # older runs have none for the live smoother
    row = {
        "accuracy": part["accuracy"],
        "macro_f1": part["macro_f1"],
        "boundary_detection": part["boundary"]["detection_rate"],
        "boundary_error_s": part["boundary"]["mean_abs_error_s"],
        "false_per_min": part["boundary"]["false_per_minute"],
    }
    if f1 is not None:
        row.update({f"f1_{name}": f1[name]["f1"] for name in CLASS_NAMES if f1[name]["support"]})
    return row


def _mean_std(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean))


def summarise(models: list[str], seeds: list[int], variants: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for model in models:
            per_seed = []
            for seed in seeds:
                path = run_dir(variant, model, seed) / "metrics.json"
                if path.exists():
                    per_seed.append(json.loads(path.read_text()))
            if not per_seed:
                continue
            for split in SCORED_SPLITS:
                for decoding in ("viterbi", "online"):
                    values: dict[str, list[float]] = {}
                    for metrics in per_seed:
                        for key, value in _row(metrics["splits"][split], online=decoding == "online").items():
                            values.setdefault(key, []).append(value)
                    row = {"variant": variant, "model": model, "split": split, "decoding": decoding, "seeds": len(per_seed),
                           "train_windows": per_seed[0].get("train_windows"),
                           "best_epochs": [m.get("best_epoch") for m in per_seed]}
                    for key, series in values.items():
                        row[key], row[f"{key}_std"] = _mean_std(series)
                    rows.append(row)
    baseline = baseline_scores()
    for mode, splits in baseline.items():
        for split, scores in splits.items():
            row = {"variant": "detector", "model": "-", "split": split,
                   "decoding": "viterbi" if mode == "offline" else "online", "seeds": 1, "mode": mode,
                   "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"],
                   "boundary_detection": scores["boundary"]["detection_rate"],
                   "boundary_error_s": scores["boundary"]["mean_abs_error_s"],
                   "false_per_min": scores["boundary"]["false_per_minute"]}
            row.update({f"f1_{name}": scores["per_class"][name]["f1"] for name in CLASS_NAMES if scores["per_class"][name]["support"]})
            rows.append(row)
    return {"rows": rows, "baseline": baseline, "seeds": seeds}


def _cell(row: dict[str, Any], key: str, digits: int = 3) -> str:
    value = row.get(key)
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    std = row.get(f"{key}_std")
    if isinstance(std, float) and math.isfinite(std) and row.get("seeds", 1) > 1:
        return f"{value:.{digits}f} ± {std:.{digits}f}"
    return f"{value:.{digits}f}"


def markdown(summary: dict[str, Any], split: str = "test") -> str:
    lines = []
    for decoding, title in (("online", "causal (live smoother vs realtime detector)"),
                            ("viterbi", "whole window (Viterbi vs offline detector)")):
        lines.append(f"### {split}: {title}\n")
        lines.append("| variant | model | accuracy | macro-F1 | F1 inhale | F1 hold | F1 exhale | boundaries found | error s | false /min |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in summary["rows"]:
            if row["split"] != split or row["decoding"] != decoding:
                continue
            lines.append(
                f"| {row['variant']} | {row['model']} | {_cell(row, 'accuracy')} | {_cell(row, 'macro_f1')} | "
                f"{_cell(row, 'f1_inhale', 2)} | {_cell(row, 'f1_hold', 2)} | {_cell(row, 'f1_exhale', 2)} | "
                f"{_cell(row, 'boundary_detection', 2)} | {_cell(row, 'boundary_error_s', 2)} | {_cell(row, 'false_per_min', 1)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--gru-workers", type=int, default=3, help="GRU runs at once on the CPU")
    parser.add_argument("--tcn-workers", type=int, default=2, help="TCN runs at once on the GPU")
    parser.add_argument("--summary-only", action="store_true", help="only collect the runs already there")
    args = parser.parse_args()
    models = [m for m in args.models.split(",") if m]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    unknown = (set(models) - set(MODELS)) | (set(variants) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown {sorted(unknown)}")
    if not args.summary_only:
        run_grid(models, seeds, variants, args.gru_workers, args.tcn_workers)
    summary = summarise(models, seeds, variants)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    table = markdown(summary, "test") + "\n" + markdown(summary, "val")
    (OUT_DIR / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"wrote {OUT_DIR / 'summary.json'} and summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
