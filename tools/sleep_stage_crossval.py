#!/usr/bin/env python
"""Sleep stages from radar breathing, heart rate, movement and all of them, against the Garmin watch, night by night.

The epochs of ``tools/build_sleep_epochs.py`` that lie inside the watch's sleep interval carry its stage (awake,
light, deep, REM).  With so few nights (one person) the model is deliberately small: a multinomial logistic
regression on the scaled features of one set and their 10-minute context, classes weighted by inverse frequency,
followed by a hidden-Markov smoothing of the night (Viterbi, transitions counted on the training nights).  With the
classes weighted the classifier's output already treats the stages as equally likely, so it is the emission as it is;
dividing it by the stage frequencies once more would push every epoch out of light sleep.  Each night is held out once (leave-one-night-out) and
the held-out predictions are pooled.

Compared on the same epochs:

* ``majority`` -- always the most common stage of the training nights (light sleep),
* ``time`` -- the hours since the recording started (and their square) only: deep sleep comes early, REM late,
* ``heuristic`` -- the rule-based classifier of the sleep chapter,
* ``breathing``, ``heart``, ``movement``, ``all`` -- the radar feature sets, and ``all+time``.

The watch is not a reference: it stages sleep with its own undisclosed rules from wrist heart rate and movement.
Agreement with it says how close the radar comes to a device people trust, not how right either is.

    uv run --extra ml python tools/sleep_stage_crossval.py
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

from build_sleep_epochs import OUT as EPOCHS, STAGES, feature_columns

OUT_DIR = ROOT / "runs" / "sleep_stages"
MODELS = ("majority", "time", "heuristic", "breathing", "heart", "movement", "all", "all+time")
WEIGHT_DECAY = 1e-2
STEPS = 400


def columns_for(model: str) -> list[str]:
    if model == "time":
        return ["hours_since_start", "hours_since_start_sq"]
    if model == "all+time":
        return feature_columns("all") + ["hours_since_start", "hours_since_start_sq"]
    return feature_columns(model)


def fit_logistic(x: np.ndarray, y: np.ndarray, classes: int, seed: int = 0) -> torch.nn.Linear:
    torch.manual_seed(seed)
    model = torch.nn.Linear(x.shape[1], classes)
    counts = np.bincount(y, minlength=classes).astype(float)
    weight = torch.tensor(np.where(counts > 0, counts.sum() / (classes * np.maximum(counts, 1)), 0.0), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=WEIGHT_DECAY)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    return model


def transition_log(sequences: list[np.ndarray], classes: int) -> np.ndarray:
    counts = np.ones((classes, classes))
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def viterbi(emission: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    n, k = emission.shape
    score = emission[0].copy()
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        candidates = score[:, None] + transitions
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(k)] + emission[t]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def kappa(y: np.ndarray, p: np.ndarray, classes: int) -> float:
    confusion = np.zeros((classes, classes))
    np.add.at(confusion, (y, p), 1)
    total = confusion.sum()
    observed = np.trace(confusion) / total
    expected = float((confusion.sum(0) * confusion.sum(1)).sum()) / total**2
    return float((observed - expected) / (1 - expected)) if expected < 1 else math.nan


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    k = len(STAGES)
    f1 = {}
    for c, name in enumerate(STAGES):
        tp = np.sum((p == c) & (y == c))
        precision = tp / max(np.sum(p == c), 1)
        recall = tp / max(np.sum(y == c), 1)
        f1[name] = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {"accuracy": float(np.mean(y == p)), "kappa": kappa(y, p, k), "macro_f1": float(np.mean(list(f1.values()))), "f1": f1}


def crossval(epochs: pd.DataFrame) -> dict[str, Any]:
    data = epochs[epochs["garmin_stage"].isin(STAGES)].copy()
    data["hours_since_start"] = data["hours_since_start"] / 10.0  # about 0-1 over a night, like the scaled features
    data["hours_since_start_sq"] = data["hours_since_start"] ** 2
    data["y"] = data["garmin_stage"].map({s: i for i, s in enumerate(STAGES)}).astype(int)
    nights = sorted(data["night"].unique())
    k = len(STAGES)
    predictions: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    truth: list[np.ndarray] = []
    per_night: dict[str, dict[str, Any]] = {}
    for held in nights:
        train, test = data[data["night"] != held], data[data["night"] == held].sort_values("elapsed_s")
        y_test = test["y"].to_numpy()
        truth.append(y_test)
        prior = np.bincount(train["y"], minlength=k) / len(train)
        transitions = transition_log([g.sort_values("elapsed_s")["y"].to_numpy() for _, g in train.groupby("night")], k)
        per_night[held] = {"epochs": int(len(test)), "garmin_minutes": {s: float(np.sum(y_test == i)) / 2 for i, s in enumerate(STAGES)}}
        for model in MODELS:
            if model == "majority":
                pred = np.full(len(test), int(np.argmax(prior)))
            elif model == "heuristic":
                pred = test["heuristic_stage"].map({s: i for i, s in enumerate(STAGES)}).fillna(1).astype(int).to_numpy()
            else:
                cols = columns_for(model)
                x_train = train[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                x_test = test[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                net = fit_logistic(x_train, train["y"].to_numpy(), k)
                with torch.no_grad():
                    log_post = torch.log_softmax(net(torch.tensor(x_test)), dim=1).numpy()
                pred = viterbi(log_post, transitions)
            predictions[model].append(pred)
            per_night[held][model] = {**scores(y_test, pred), "minutes": {s: float(np.sum(pred == i)) / 2 for i, s in enumerate(STAGES)}}
    y_all = np.concatenate(truth)
    pooled = {model: scores(y_all, np.concatenate(preds)) for model, preds in predictions.items()}
    return {"nights": nights, "epochs": int(len(y_all)), "pooled": pooled, "per_night": per_night,
            "garmin_share": {s: float(np.mean(y_all == i)) for i, s in enumerate(STAGES)}}


def markdown(result: dict[str, Any]) -> str:
    lines = [f"{len(result['nights'])} nights, {result['epochs']} epochs inside the watch's sleep interval; Garmin shares "
             + ", ".join(f"{s} {100 * v:.0f}%" for s, v in result["garmin_share"].items()) + "\n",
             "| model | accuracy % | kappa | macro-F1 | F1 awake | F1 light | F1 deep | F1 REM | kappa per night |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model, row in result["pooled"].items():
        per = ", ".join(f"{result['per_night'][n][model]['kappa']:.2f}" for n in result["nights"])
        f1 = row["f1"]
        lines.append(f"| {model} | {100 * row['accuracy']:.1f} | {row['kappa']:.3f} | {row['macro_f1']:.3f} | {f1['awake']:.2f} | "
                     f"{f1['light']:.2f} | {f1['deep']:.2f} | {f1['rem']:.2f} | {per} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=Path, default=EPOCHS)
    args = parser.parse_args()
    result = crossval(pd.read_csv(args.epochs, low_memory=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = markdown(result)
    (OUT_DIR / "summary.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
