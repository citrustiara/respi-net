#!/usr/bin/env python
"""Leave-one-session-out cross-validation of the breathing-phase network on the author's radar + belt sessions.

The fixed split of ``tools/compare_breath_phase_training.py`` tests on two of the author's (S01) four
radar + belt sessions and validates on one, and the variant was picked on that test split.  Here each of
the four sessions is the test set once (:data:`FOLDS`), with another S01 session as the validation set
that picks the best epoch; the father's sessions (S02) and the remaining S01 sessions train.  For every
fold both datasets are rebuilt by ``tools/build_artificial_data.py --belt-split``, so the held-out
session's belt trace leaves pretraining and its copies leave training too.  Coach runs keep their own
split; only the held-out radar + belt session is scored.

Per fold, variant (``real``, ``copies``, ``pretrain``; TCN by default) and seed the saved model is run on
the held-out session's windows and scored as in the comparison (Viterbi over the window and the live
smoother), next to the offline and realtime detector on the same windows.  The variant is picked by the
mean validation macro-F1 of the live smoother over the folds, never by a test score; the test scores of
all four sessions are then pooled (every session's windows once) and also given per session.

    uv run --extra ml python tools/crossval_breath_phase_training.py
    uv run --extra ml python tools/crossval_breath_phase_training.py --summary-only
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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from compare_breath_phase_training import DATA_DIR, VARIANTS, Setup, _hold_events, _mean_std, run_dir, run_job
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import score_phases, transition_log_matrix
from train_breath_phase_model import _jsonable, load_windows, pooled_boundaries

# fold -> (held-out test session, validation session); every other radar + belt session trains.
FOLDS = {
    "side80": ("sitting_side_80cm_free_2026-10-25", "sitting_side_50cm_free_2026-10-26"),
    "lens140": ("sitting_140cm_new_lens_2026-10-27", "sitting_side_50cm_free_2026-10-26"),
    "side50": ("sitting_side_50cm_free_2026-10-26", "sitting_side_80cm_free_2026-10-25"),
    "phone50": ("sitting_50cm_phone_scrolling_2026-10-27", "sitting_side_50cm_free_2026-10-26"),
}
FOLD_NAMES = {"side80": "z boku, 80 cm", "lens140": "140 cm, nowa soczewka", "side50": "z boku, 50 cm", "phone50": "50 cm, przewijanie telefonu"}
DATA_CV = DATA_DIR / "crossval"
CV_DIR = ROOT / "runs" / "breath_phase" / "crossval_v1"
BUILDER = ROOT / "tools" / "build_artificial_data.py"


def setup_for(fold: str) -> Setup:
    return Setup(DATA_CV / fold / "dataset_pretrain_v1.npz", DATA_CV / fold / "dataset_finetune_v1.npz", CV_DIR / fold)


def build_fold(fold: str) -> None:
    setup = setup_for(fold)
    if setup.pretrain.exists() and setup.finetune.exists():
        return
    test, val = FOLDS[fold]
    command = [sys.executable, str(BUILDER), "--output-dir", str(setup.finetune.parent),
               "--belt-split", f"{test}=test", "--belt-split", f"{val}=val"]
    print(f"building {fold}: test {test}, val {val}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def run_folds(folds: list[str], seeds: list[int], variants: list[str], workers: int, model: str) -> None:
    """All folds share one pool of TCN slots; a fine-tune follows its own pretraining in the same slot."""

    lock = threading.Lock()
    pool = ThreadPoolExecutor(workers)
    futures: list[Future] = []
    for seed in seeds:  # long pretraining chains first, so the short runs fill the tail
        for fold in folds:
            setup = setup_for(fold)
            setup.out_dir.mkdir(parents=True, exist_ok=True)
            if "pretrain" in variants:
                futures.append(pool.submit(lambda f=setup, s=seed: (run_job("pre", model, s, 1, lock, f),
                                                                    run_job("pretrain", model, s, 1, lock, f))))
    for seed in seeds:
        for fold in folds:
            for variant in variants:
                if variant != "pretrain":
                    futures.append(pool.submit(run_job, variant, model, seed, 1, lock, setup_for(fold)))
    failures = 0
    for future in futures:
        try:
            future.result()
        except RuntimeError as error:
            failures += 1
            print(error, flush=True)
    pool.shutdown()
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


# ---------------------------------------------------------------- scoring


def held_out(data: Any, fold: str) -> np.ndarray:
    """Test windows of the held-out radar + belt session (not the coach runs that share the test split)."""

    index = data.indices("test")
    keep = np.char.find(data.run_id[index].astype(str), FOLDS[fold][0]) >= 0
    return index[keep]


def model_paths(folder: Path, data: Any, index: np.ndarray) -> dict[str, np.ndarray]:
    import torch

    from respi_net.phase_model import load_checkpoint
    from train_breath_phase_model import SMOOTHER_KEYS, forward_split, online_windows, select_channels, viterbi_windows

    model, info = load_checkpoint(folder / "model.pt")
    log_probs, _ = forward_split(model, select_channels(data, info["channels"])[index], torch.device("cpu"))
    decoding = info["decoding"]
    return {"viterbi": viterbi_windows(log_probs, transition_log_matrix(math.log(decoding["switch_prob"]))),
            "online": online_windows(log_probs, data.fs, {key: decoding[key] for key in SMOOTHER_KEYS})}


def detector_paths(data: Any, index: np.ndarray) -> dict[str, np.ndarray]:
    chest = data.X[index, data.channels.index("chest")].astype(float)
    return {"viterbi": np.stack([detect_phases(window, data.fs) for window in chest]),
            "online": np.stack([detect_phases(window, data.fs, causal=True) for window in chest])}


def score(y: np.ndarray, path: np.ndarray, start_s: np.ndarray, run_id: np.ndarray, fs: float) -> dict[str, float]:
    scores = score_phases(y.ravel(), path.ravel())
    boundary = pooled_boundaries(y, path, start_s, fs)
    holds, found = _hold_events(y, path, run_id, start_s, fs)
    return {"accuracy": scores.accuracy, "macro_f1": scores.macro_f1, "f1_hold": scores.per_class["hold"]["f1"],
            "boundary_error_s": boundary["mean_abs_error_s"], "false_per_min": boundary["false_per_minute"],
            "holds": holds, "holds_found": found, "windows": int(len(y)),
            "scored_minutes": float(np.sum(y >= 0)) / fs / 60.0}  # overlapping windows count a sample more than once


def summarise(folds: list[str], seeds: list[int], variants: list[str], model: str) -> dict[str, Any]:
    """Per fold and pooled over folds, for every variant (mean over seeds) and the detector."""

    per_fold: dict[str, dict[str, Any]] = {}
    pooled: dict[str, dict[str, dict[int, list]]] = {}  # variant -> decoding -> seed -> [(y, path, start, run)]
    validation: dict[str, list[float]] = {variant: [] for variant in variants}
    for fold in folds:
        setup = setup_for(fold)
        data = load_windows(setup.finetune)
        index = held_out(data, fold)
        y, start, run = data.y[index], data.start_s[index], data.run_id[index]
        per_fold[fold] = {}
        sources: dict[str, dict[int, dict[str, np.ndarray]]] = {"detector": {0: detector_paths(data, index)}}
        for variant in variants:
            sources[variant] = {}
            for seed in seeds:
                folder = run_dir(variant, model, seed, setup)
                if (folder / "model.pt").exists():
                    sources[variant][seed] = model_paths(folder, data, index)
                    metrics = json.loads((folder / "metrics.json").read_text())
                    validation[variant].append(metrics["splits"]["val"]["online"]["macro_f1"])
        for name, by_seed in sources.items():
            if not by_seed:
                continue
            per_fold[fold][name] = {}
            for decoding in ("online", "viterbi"):
                rows = [score(y, paths[decoding], start, run, data.fs) for paths in by_seed.values()]
                per_fold[fold][name][decoding] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
                per_fold[fold][name][decoding]["holds_found_per_seed"] = [r["holds_found"] for r in rows]
                for seed, paths in by_seed.items():
                    pooled.setdefault(name, {}).setdefault(decoding, {}).setdefault(seed, []).append((y, paths[decoding], start, run))
    overall: dict[str, dict[str, Any]] = {}
    for name, decodings in pooled.items():
        overall[name] = {}
        for decoding, by_seed in decodings.items():
            complete = {seed: parts for seed, parts in by_seed.items() if len(parts) == len(folds)}
            rows = [score(*(np.concatenate([part[k] for part in parts]) for k in range(4)), data.fs) for parts in complete.values()]
            if not rows:
                continue
            fold_means = [per_fold[fold][name][decoding] for fold in folds]
            overall[name][decoding] = {
                **{key: _mean_std([r[key] for r in rows])[0] for key in rows[0]},
                **{f"{key}_std": _mean_std([r[key] for r in rows])[1] for key in ("accuracy", "macro_f1")},
                # spread between sessions (each session's seed-mean), not between seeds
                **{f"{key}_fold_std": float(np.std([f[key] for f in fold_means])) for key in ("accuracy", "macro_f1")},
                "holds_found_per_seed": [r["holds_found"] for r in rows],
                "seeds": len(rows),
            }
    val_mean = {variant: _mean_std(values)[0] for variant, values in validation.items() if values}
    chosen = max(val_mean, key=val_mean.get) if val_mean else None
    return {"folds": {fold: {"test": FOLDS[fold][0], "val": FOLDS[fold][1]} for fold in folds}, "per_fold": per_fold,
            "overall": overall, "validation_macro_f1": val_mean, "chosen_by_validation": chosen, "seeds": seeds, "model": model}


def _pct(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{100 * value:.1f}"


def markdown(summary: dict[str, Any]) -> str:
    names = [*[v for v in VARIANTS if v in summary["overall"]], "detector"]
    lines = [f"Chosen by mean validation macro-F1 (live): **{summary['chosen_by_validation']}** "
             + "(" + ", ".join(f"{v} {m:.3f}" for v, m in summary["validation_macro_f1"].items()) + ")\n"]
    for decoding, title in (("online", "live (causal smoother vs realtime detector)"), ("viterbi", "whole window (Viterbi vs offline detector)")):
        lines += [f"### all four sessions pooled, {title}\n",
                  "| variant | accuracy % | ± seeds | ± sessions | macro-F1 | F1 hold | error s | false /min | holds found |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in names:
            row = summary["overall"].get(name, {}).get(decoding)
            if not row:
                continue
            found = ", ".join(map(str, row["holds_found_per_seed"]))
            lines.append(f"| {name} | {_pct(row['accuracy'])} | {_pct(row.get('accuracy_std', float('nan')))} | "
                         f"{_pct(row['accuracy_fold_std'])} | {row['macro_f1']:.3f} | {row['f1_hold']:.2f} | "
                         f"{row['boundary_error_s']:.2f} | {row['false_per_min']:.1f} | {found} of {row['holds']:.0f} |")
        lines.append("")
    lines += ["### per held-out session, live accuracy % (mean over seeds)\n",
              "| session | windows | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for fold, rows in summary["per_fold"].items():
        windows = rows["detector"]["online"]["windows"]
        lines.append(f"| {fold} | {windows:.0f} | " + " | ".join(_pct(rows[n]["online"]["accuracy"]) if n in rows else "-" for n in names) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--model", default="tcn", choices=("tcn", "gru"))
    parser.add_argument("--workers", type=int, default=3, help="runs at once")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    folds = [f for f in args.folds.split(",") if f]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    unknown = (set(folds) - set(FOLDS)) | (set(variants) - set(VARIANTS))
    if unknown:
        parser.error(f"unknown {sorted(unknown)}")
    if not args.summary_only:
        for fold in folds:
            build_fold(fold)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        run_folds(folds, seeds, variants, args.workers, args.model)
    summary = summarise(folds, seeds, variants, args.model)
    CV_DIR.mkdir(parents=True, exist_ok=True)
    (CV_DIR / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    table = markdown(summary)
    (CV_DIR / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"wrote {CV_DIR / 'summary.json'} and summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
