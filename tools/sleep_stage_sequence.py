#!/usr/bin/env python
"""Whole-night sleep stages: a bidirectional GRU over the 30 s epochs, pretrained on open data, tested on radar nights.

Input per epoch (``tools/build_open_sleep_epochs.py``): heart rate, breathing (rate, depth, regularity, interval
variation) and movement, each scaled within its night, with their 10-minute medians and spreads, the time since the
last movement and since the recording started, and masks for the missing modalities.  The network reads a
stretch of the night forward and backward -- a sleep stage depends on where in the night and in its cycle the
epoch lies -- and gives the stage of every epoch (awake, light, deep, REM).

Training draws random 2-4 h stretches from the nights (many examples from few nights), weights the classes by
inverse frequency, and drops the heart or the breathing features of a stretch now and then (modality dropout),
so one network learns from watch nights (heart + movement) and radar nights (breathing + movement) and uses
whatever a night has.

Evaluations, 3 seeds each:

* ``sleepbrl``: radar against polysomnography -- 4 folds over the SLEEPBRL nights, each trained on Walch and the
  other SLEEPBRL nights;
* ``own``: the author's radar nights against the Garmin watch, every labelled night held out once --
  ``scratch`` (own nights only), ``zeroshot`` (open data only), ``finetune`` (open data, then the other own nights).

    uv run --extra ml python tools/sleep_stage_sequence.py
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
import pandas as pd
import torch
from torch import nn

from build_open_sleep_epochs import OUT_DIR as DATA_DIR, STAGES, feature_columns
from sleep_stage_crossval import kappa, scores

OUT_DIR = ROOT / "runs" / "sleep_stage_sequence"
FEATURES = feature_columns()
HR_COLS = [i for i, c in enumerate(FEATURES) if c.startswith("z_hr")]
BREATH_COLS = [i for i, c in enumerate(FEATURES) if c.startswith(("z_rr", "z_breath"))]
HAS_HR, HAS_BREATH = FEATURES.index("has_hr"), FEATURES.index("has_breath")
CROP = (240, 480)
DROP_P = 0.3


class Night:
    def __init__(self, frame: pd.DataFrame):
        self.name = f"{frame['source'].iloc[0]}:{frame['night'].iloc[0]}"
        self.x = frame[FEATURES].to_numpy(dtype=np.float32)
        self.y = frame["stage"].map({s: i for i, s in enumerate(STAGES)}).fillna(-1).to_numpy(dtype=np.int64)


def load(source: str) -> list[Night]:
    frame = pd.read_csv(DATA_DIR / f"{source}.csv", low_memory=False)
    return [Night(g.sort_values("hours")) for _, g in frame.groupby("night", sort=True)]


class StageGRU(nn.Module):
    def __init__(self, features: int = len(FEATURES), hidden: int = 32):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(2 * hidden, len(STAGES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.inp(x))
        return self.out(h)


def _batch(nights: list[Night], rng: np.random.Generator, size: int = 16, dropout: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    length = int(rng.integers(*CROP))
    xs, ys = [], []
    for _ in range(size):
        night = nights[int(rng.integers(len(nights)))]
        n = len(night.y)
        start = int(rng.integers(0, max(1, n - length)))
        x, y = night.x[start : start + length].copy(), night.y[start : start + length].copy()
        if len(y) < length:
            x = np.pad(x, ((0, length - len(y)), (0, 0)))
            y = np.pad(y, (0, length - len(y)), constant_values=-1)
        if dropout:
            if x[0, HAS_HR] > 0 and x[0, HAS_BREATH] > 0:
                drop = rng.random()
                if drop < DROP_P / 2:
                    x[:, HR_COLS], x[:, HAS_HR] = 0.0, 0.0
                elif drop < DROP_P:
                    x[:, BREATH_COLS], x[:, HAS_BREATH] = 0.0, 0.0
        xs.append(x)
        ys.append(y)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def class_weights(nights: list[Night]) -> torch.Tensor:
    y = np.concatenate([n.y for n in nights])
    counts = np.bincount(y[y >= 0], minlength=len(STAGES)).astype(float)
    return torch.tensor(counts.sum() / (len(STAGES) * np.maximum(counts, 1)), dtype=torch.float32)


def predict(model: nn.Module, night: Night) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(night.x)[None]).argmax(-1)[0].numpy()


def evaluate(model: nn.Module, nights: list[Night]) -> float:
    ys, ps = [], []
    for night in nights:
        keep = night.y >= 0
        ys.append(night.y[keep]); ps.append(predict(model, night)[keep])
    return kappa(np.concatenate(ys), np.concatenate(ps), len(STAGES)) if ys else math.nan


def train(train_nights: list[Night], val_nights: list[Night], seed: int, model: nn.Module | None = None,
          steps: int = 1500, lr: float = 3e-3, patience: int = 6) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model or StageGRU()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_nights), ignore_index=-1)
    best, best_state, since = -math.inf, None, 0
    for step in range(1, steps + 1):
        model.train()
        x, y = _batch(train_nights, rng)
        loss = loss_fn(model(x).reshape(-1, len(STAGES)), y.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0 and val_nights:
            k = evaluate(model, val_nights)
            if k > best + 1e-3:
                best, since = k, 0
                best_state = {key: v.detach().clone() for key, v in model.state_dict().items()}
            else:
                since += 1
                if since >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _split_val(nights: list[Night], share: float, seed: int) -> tuple[list[Night], list[Night]]:
    order = np.random.default_rng(seed).permutation(len(nights))
    n_val = max(1, int(round(share * len(nights))))
    return [nights[i] for i in order[n_val:]], [nights[i] for i in order[:n_val]]


def pooled(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    y = np.concatenate([p[0] for p in pairs]); p = np.concatenate([p[1] for p in pairs])
    return scores(y, p)


def run_sleepbrl(walch: list[Night], brl: list[Night], seeds: list[int], folds: int = 4) -> dict[str, Any]:
    out: dict[str, Any] = {}
    order = np.random.default_rng(0).permutation(len(brl))
    fold_of = {brl[i].name: k % folds for k, i in enumerate(order)}
    for variant, extra in (("radar_only", []), ("with_walch", walch)):
        per_seed = []
        for seed in seeds:
            pairs = []
            for k in range(folds):
                test = [n for n in brl if fold_of[n.name] == k]
                rest = [n for n in brl if fold_of[n.name] != k]
                train_n, val_n = _split_val(rest, 0.15, seed)
                model = train(train_n + extra, val_n, seed)
                pairs += [(n.y[n.y >= 0], predict(model, n)[n.y >= 0]) for n in test]
            per_seed.append(pooled(pairs))
            print(f"sleepbrl {variant} seed {seed}: kappa {per_seed[-1]['kappa']:.3f} acc {per_seed[-1]['accuracy']:.3f}", flush=True)
        out[variant] = {"kappa": float(np.mean([s["kappa"] for s in per_seed])), "kappa_sd": float(np.std([s["kappa"] for s in per_seed])),
                        "accuracy": float(np.mean([s["accuracy"] for s in per_seed])), "f1": {c: float(np.mean([s["f1"][c] for s in per_seed])) for c in STAGES}}
    return out


def run_own(open_nights: list[Night], own: list[Night], seeds: list[int]) -> dict[str, Any]:
    labelled = [n for n in own if (n.y >= 0).sum() > 100]
    out: dict[str, Any] = {"nights": [n.name for n in labelled]}
    for variant in ("scratch", "zeroshot", "finetune"):
        per_seed, per_night = [], {n.name: [] for n in labelled}
        for seed in seeds:
            pre = None
            if variant != "scratch":
                tr, va = _split_val(open_nights, 0.1, seed)
                pre = train(tr, va, seed)
            pairs = []
            for test in labelled:
                rest = [n for n in labelled if n is not test]
                if variant == "zeroshot":
                    model = pre
                else:
                    tr, va = _split_val(rest, 1 / len(rest), seed)
                    init = None
                    if variant == "finetune":
                        init = StageGRU(); init.load_state_dict(pre.state_dict())
                    model = train(tr, va, seed, model=init, steps=600 if init is not None else 1500, lr=1e-3 if init is not None else 3e-3)
                keep = test.y >= 0
                pred = predict(model, test)[keep]
                pairs.append((test.y[keep], pred))
                per_night[test.name].append(kappa(test.y[keep], pred, len(STAGES)))
            per_seed.append(pooled(pairs))
            print(f"own {variant} seed {seed}: kappa {per_seed[-1]['kappa']:.3f} acc {per_seed[-1]['accuracy']:.3f}", flush=True)
        out[variant] = {"kappa": float(np.mean([s["kappa"] for s in per_seed])), "kappa_sd": float(np.std([s["kappa"] for s in per_seed])),
                        "accuracy": float(np.mean([s["accuracy"] for s in per_seed])), "f1": {c: float(np.mean([s["f1"][c] for s in per_seed])) for c in STAGES},
                        "kappa_per_night": {k: float(np.mean(v)) for k, v in per_night.items()}}
    return out


def markdown(result: dict[str, Any]) -> str:
    lines = ["| evaluation | variant | kappa | ± seeds | accuracy % | F1 awake | F1 light | F1 deep | F1 REM |", "|---|---|---|---|---|---|---|---|---|"]
    for part in ("sleepbrl", "own"):
        for variant, row in result.get(part, {}).items():
            if not isinstance(row, dict) or "kappa" not in row:
                continue
            f = row["f1"]
            lines.append(f"| {part} | {variant} | {row['kappa']:.3f} | {row['kappa_sd']:.3f} | {100 * row['accuracy']:.1f} | "
                         f"{f['awake']:.2f} | {f['light']:.2f} | {f['deep']:.2f} | {f['rem']:.2f} |")
    if "own" in result:
        lines.append("")
        lines.append("kappa per own night: " + "; ".join(
            f"{v}: " + ", ".join(f"{k.split(':')[1]} {x:.2f}" for k, x in result["own"][v]["kappa_per_night"].items())
            for v in ("scratch", "zeroshot", "finetune") if v in result["own"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--open", default="walch,sleepbrl", help="open sources for pretraining")
    parser.add_argument("--parts", default="sleepbrl,own")
    parser.add_argument("--ablation", choices=("walch", "bidsleep", "sleepbrl", "own"), help="signal subsets on one dataset instead")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1, help="ablation: trainings in parallel, one thread each "
                        "(one at a time while the machine is busy with something else); keep below the core count")
    parser.add_argument("--tag", default="", help="suffix of summary.json / summary.md")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.ablation:
        ablation_main(args.ablation, seeds, args.tag, args.workers)
        return 0
    sources = {s: load(s) for s in {*args.open.split(","), "walch", "sleepbrl", "own"} if (DATA_DIR / f"{s}.csv").exists()}
    open_nights = [n for s in args.open.split(",") if s in sources for n in sources[s]]
    result: dict[str, Any] = {"open_sources": args.open, "seeds": seeds}
    if "sleepbrl" in args.parts:
        result["sleepbrl"] = run_sleepbrl(sources["walch"], sources["sleepbrl"], seeds)
    if "own" in args.parts:
        result["own"] = run_own(open_nights, sources["own"], seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"summary{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = markdown(result)
    (OUT_DIR / f"summary{args.tag}.md").write_text(table, encoding="utf-8")
    print(table)
    return 0




# ------------------------------------------------------------- ablation

GROUPS = {
    "heart": [c for c in FEATURES if c.startswith("z_hr")] + ["has_hr"],
    "breath": [c for c in FEATURES if c.startswith(("z_rr", "z_breath"))] + ["has_breath"],
    "net": [c for c in FEATURES if c.startswith("z_net")] + ["has_net"],
    "motion": [c for c in FEATURES if c.startswith("z_motion")] + ["minutes_since_movement_scaled"],
    "time": ["hours_scaled"],
}
ABLATIONS = {
    # which signals each dataset can answer for, against its own reference
    "walch": ("time", "heart", "motion", "heart+motion", "heart+motion+time"),
    "bidsleep": ("time", "heart", "motion", "heart+motion", "heart+motion+time"),
    "sleepbrl": ("time", "breath", "motion", "breath+motion", "breath+motion+time"),
    "own": ("time", "heart", "breath", "net", "motion", "heart+motion", "breath+motion", "heart+breath+motion",
            "heart+breath+net+motion", "heart+breath+net+motion+time"),
}


def masked(nights: list[Night], subset: str) -> list[Night]:
    """Copies of the nights with every feature outside ``subset`` (groups joined by +) set to 0."""

    keep = {c for g in subset.split("+") for c in GROUPS[g]}
    idx = [i for i, c in enumerate(FEATURES) if c not in keep]
    out = []
    for night in nights:
        copy = Night.__new__(Night)
        copy.name, copy.y, copy.x = night.name, night.y, night.x.copy()
        copy.x[:, idx] = 0.0
        out.append(copy)
    return out


def stage_minutes(y: np.ndarray) -> np.ndarray:
    return np.bincount(y[y >= 0], minlength=len(STAGES)) / 2.0



def _init_worker() -> None:
    torch.set_num_threads(1)


def _fit_and_predict(tr: list[Night], va: list[Night], test: list[Night], seed: int, steps: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """One fold of one seed: train, then (truth, prediction) on the labelled epochs of each test night."""
    model = train(tr, va, seed, steps=steps)
    return [(n.y[n.y >= 0], predict(model, n)[n.y >= 0]) for n in test]


def run_tasks(fn, tasks: list[tuple], workers: int) -> list[Any]:
    """``fn(*task)`` for every task, in order of the tasks.  With ``workers`` > 1 in that many single-thread processes,
    dropping to one at a time while the machine is busy with something else (``machine_load.machine_busy``)."""
    if workers <= 1:
        return [fn(*task) for task in tasks]
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from machine_load import machine_busy

    results: dict[int, Any] = {}
    queue = list(enumerate(tasks))
    pending: dict[Any, int] = {}
    throttled = False
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) as pool:
        while queue or pending:
            busy = machine_busy(len(pending))
            if busy != throttled:
                throttled = busy
                print("machine busy: one training at a time" if busy else f"machine free: {workers} trainings at a time", flush=True)
            limit = 1 if busy else workers
            while queue and len(pending) < limit:
                i, task = queue.pop(0)
                pending[pool.submit(fn, *task)] = i
            finished, _ = wait(list(pending), timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                results[pending.pop(future)] = future.result()
            if finished:
                print(f"  {len(results)}/{len(tasks)} trainings done", flush=True)
    return [results[i] for i in range(len(tasks))]


def run_ablation(dataset: str, nights: list[Night], seeds: list[int], folds: int = 4, steps: int = 1000,
                 workers: int = 1) -> dict[str, Any]:
    """Each subset: leave-one-night-out (own) or ``folds`` folds (open sets), pooled kappa, and per-night stage
    minutes against the reference (correlation over nights and mean absolute difference, per stage)."""

    nights = [n for n in nights if (n.y >= 0).sum() > 100]
    if dataset == "own":
        fold_of = {n.name: k for k, n in enumerate(nights)}
    else:
        order = np.random.default_rng(0).permutation(len(nights))
        fold_of = {nights[i].name: k % folds for k, i in enumerate(order)}
    out: dict[str, Any] = {}
    folds_k = sorted(set(fold_of.values()))
    tasks, keys = [], []
    for subset in ABLATIONS[dataset]:
        data = masked(nights, subset)
        for seed in seeds:
            for k in folds_k:
                test = [n for n in data if fold_of[n.name] == k]
                rest = [n for n in data if fold_of[n.name] != k]
                tr, va = _split_val(rest, max(1 / len(rest), 0.15), seed)
                tasks.append((tr, va, test, seed, steps))
                keys.append((subset, seed, k))
    done = dict(zip(keys, run_tasks(_fit_and_predict, tasks, workers)))
    for subset in ABLATIONS[dataset]:
        per_seed, minutes = [], []
        for seed in seeds:
            pairs, mins = [], []
            for k in folds_k:
                for truth, pred in done[(subset, seed, k)]:
                    pairs.append((truth, pred))
                    mins.append((stage_minutes(truth), stage_minutes(pred)))
            per_seed.append(pooled(pairs))
            minutes.append(mins)
        ref = np.array([[m[0] for m in mins] for mins in minutes]).mean(0)  # nights x stages
        pred = np.array([[m[1] for m in mins] for mins in minutes]).mean(0)
        corr = {s: float(np.corrcoef(ref[:, i], pred[:, i])[0, 1]) if ref[:, i].std() > 0 and pred[:, i].std() > 0 else math.nan
                for i, s in enumerate(STAGES)}
        out[subset] = {"kappa": float(np.mean([p["kappa"] for p in per_seed])), "kappa_sd": float(np.std([p["kappa"] for p in per_seed])),
                       "accuracy": float(np.mean([p["accuracy"] for p in per_seed])),
                       "f1": {c: float(np.mean([p["f1"][c] for p in per_seed])) for c in STAGES},
                       "minutes_r": corr, "minutes_mae": {s: float(np.mean(np.abs(ref[:, i] - pred[:, i]))) for i, s in enumerate(STAGES)}}
        print(f"{dataset} {subset}: kappa {out[subset]['kappa']:.3f} +- {out[subset]['kappa_sd']:.3f}", flush=True)
    return out


def ablation_markdown(dataset: str, result: dict[str, Any]) -> str:
    ref = {"own": "Garmin", "bidsleep": "Dreem EEG headband"}.get(dataset, "PSG")
    lines = [f"### {dataset} vs {ref}\n", "| signals | kappa | ± seeds | accuracy % | F1 awake | F1 light | F1 deep | F1 REM | "
             "minutes r (awake/light/deep/REM) | minutes MAE (awake/light/deep/REM) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for subset, row in result.items():
        f, r, m = row["f1"], row["minutes_r"], row["minutes_mae"]
        lines.append(f"| {subset} | {row['kappa']:.3f} | {row['kappa_sd']:.3f} | {100 * row['accuracy']:.1f} | {f['awake']:.2f} | {f['light']:.2f} | "
                     f"{f['deep']:.2f} | {f['rem']:.2f} | " + " / ".join(f"{r[s]:.2f}" for s in STAGES) + " | " + " / ".join(f"{m[s]:.0f}" for s in STAGES) + " |")
    return "\n".join(lines) + "\n"


def ablation_main(dataset: str, seeds: list[int], tag: str = "", workers: int = 1) -> None:
    result = run_ablation(dataset, load(dataset), seeds, workers=workers)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"ablation_{dataset}{tag}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = ablation_markdown(dataset, result)
    (OUT_DIR / f"ablation_{dataset}{tag}.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    raise SystemExit(main())
