#!/usr/bin/env python
"""Train, or only score, the causal breathing-phase network on an nn_dataset .npz.

The dataset is a ``respi_net.nn_dataset`` export: windows ``X`` [N, C, W],
labels ``y`` [N, W] (classes 0-4, -1 = ignore), ``split``, ``run_id``,
``start_s`` and a JSON ``meta``.  Training minimises cross-entropy over the
labelled frames, with inverse-frequency class weights from the train split,
and keeps the epoch with the best validation macro-F1 after Viterbi decoding.
A run is saved to ``runs/breath_phase/<name>/``: ``model.pt``,
``metrics.json`` and ``history.csv``.

Run from the repository root with the ``ml`` extra::

    uv run --extra ml python tools/train_breath_phase_model.py DATASET.npz --model tcn --name tcn_v1
    uv run --extra ml python tools/train_breath_phase_model.py DATASET.npz --model gru --augment --device auto
    uv run --extra ml python tools/train_breath_phase_model.py --evaluate runs/breath_phase/tcn_v1/model.pt DATASET.npz
    uv run --extra ml python tools/train_breath_phase_model.py --smoke

``--smoke`` needs no files: it builds a few cartoon windows in memory, takes
three optimiser steps, saves the checkpoint to a temporary directory, scores
it again from an .npz the way ``--evaluate`` does and exits 0.  It proves the
code runs; it trains nothing worth keeping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Make direct execution (python tools/...) work as well as uv run.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from torch import nn

from respi_net.breath_phases import (
    CLASS_NAMES,
    EXHALE,
    HOLD_AFTER_EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    NOISE,
    NUM_CLASSES,
    label_runs,
    mask_around,
)
from respi_net.phase_metrics import boundary_errors, score_phases, transition_log_matrix, viterbi_decode
from respi_net.phase_model import (
    MODELS,
    OnlinePhaseSmoother,
    build_model,
    class_weights,
    count_parameters,
    load_checkpoint,
    masked_cross_entropy,
    save_checkpoint,
)

DEFAULT_RUNS_DIR = ROOT / "runs" / "breath_phase"
SPLITS = ("train", "val", "test")
# Breath depth lives in these two; the echo level is another quantity and is never scaled.
AMPLITUDE_CHANNELS = ("chest", "velocity")
# switch_prob as in phase_metrics.transition_log_matrix; the rest are OnlinePhaseSmoother's defaults.
DEFAULT_DECODING: dict[str, Any] = {"switch_prob": 0.02, "min_phase_s": 0.3, "confirm_frames": 3, "threshold": 0.5}
SMOOTHER_KEYS = ("min_phase_s", "confirm_frames", "threshold")
HISTORY_FIELDS = ("epoch", "steps", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "seconds", "best")

Log = Callable[[str], None]


def _print(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class PhaseWindows:
    """An nn_dataset export: windows, labels and where each window came from."""

    X: np.ndarray  # float32 [N, C, W]
    y: np.ndarray  # int64 [N, W], IGNORE = -1
    split: np.ndarray  # str [N]
    run_id: np.ndarray  # str [N]
    start_s: np.ndarray  # float [N]
    meta: dict[str, Any]

    @property
    def fs(self) -> float:
        return float(self.meta["fs"])

    @property
    def channels(self) -> list[str]:
        names = self.meta.get("channels") or [f"ch{index}" for index in range(self.X.shape[1])]
        return [str(name) for name in names]

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def load_windows(path: Path) -> PhaseWindows:
    """Read an nn_dataset .npz with numpy alone; pickled objects are refused."""

    with np.load(path, allow_pickle=False) as data:
        missing = {"X", "y", "split", "meta"} - set(data.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}; is it an nn_dataset export?")
        raw_meta = data["meta"][()]
        meta = json.loads(raw_meta.decode() if isinstance(raw_meta, bytes) else str(raw_meta))
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64)
        count = len(X)
        split = data["split"].astype(str)
        run_id = data["run_id"].astype(str) if "run_id" in data.files else np.full(count, "", dtype=str)
        start_s = data["start_s"].astype(float) if "start_s" in data.files else np.zeros(count)
    if X.ndim != 3 or y.shape != (count, X.shape[2]) or split.shape != (count,):
        raise ValueError(f"{path}: expected X [N, C, W], y [N, W], split [N]; got {X.shape}, {y.shape}, {split.shape}")
    if "fs" not in meta:
        raise ValueError(f"{path}: meta has no sample rate 'fs'")
    if np.any((y < IGNORE) | (y >= NUM_CLASSES)):
        raise ValueError(f"{path}: labels outside {IGNORE}..{NUM_CLASSES - 1}")
    return PhaseWindows(X, y, split, run_id, start_s, meta)


def save_windows(path: Path, data: PhaseWindows) -> None:
    """Write ``data`` in the nn_dataset .npz layout (used by --smoke)."""

    np.savez_compressed(
        path,
        X=data.X.astype(np.float32),
        y=data.y.astype(np.int8),
        split=data.split.astype(str),
        run_id=data.run_id.astype(str),
        start_s=data.start_s.astype(np.float32),
        meta=np.array(json.dumps(data.meta)),
    )


def select_channels(data: PhaseWindows, wanted: Sequence[str]) -> np.ndarray:
    """``X`` restricted to (and ordered as) the named channels."""

    names = data.channels
    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(f"the dataset has channels {names}, not {missing}")
    return np.ascontiguousarray(data.X[:, [names.index(name) for name in wanted]])


_TOY_DURATIONS_S = {
    INHALE: (1.0, 2.5),
    HOLD_AFTER_INHALE: (0.6, 3.0),
    EXHALE: (1.2, 3.0),
    HOLD_AFTER_EXHALE: (0.6, 3.0),
    NOISE: (0.5, 1.5),
}


def _toy_window(width: int, fs: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One cartoon window: channels [3, width] and labels [width]."""

    cycle = (INHALE, HOLD_AFTER_INHALE, EXHALE, HOLD_AFTER_EXHALE)
    position = int(rng.integers(len(cycle)))
    lung = 1.0 if cycle[position] in (HOLD_AFTER_INHALE, EXHALE) else 0.0
    levels: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total = 0
    while total < width:
        phase = cycle[position % len(cycle)]
        position += 1
        if phase in (HOLD_AFTER_INHALE, HOLD_AFTER_EXHALE) and rng.random() < 0.4:
            continue  # not every breath has a hold
        if rng.random() < 0.08:
            count = int(rng.uniform(*_TOY_DURATIONS_S[NOISE]) * fs)
            levels.append(lung + rng.normal(0.0, 0.6, count))
            labels.append(np.full(count, NOISE))
            total += count
        count = max(2, int(rng.uniform(*_TOY_DURATIONS_S[phase]) * fs))
        target = {INHALE: 1.0, EXHALE: 0.0}.get(phase, lung)
        levels.append(lung + (target - lung) * (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, count))))
        labels.append(np.full(count, phase))
        lung = target
        total += count
    level = np.concatenate(levels)[:width]
    label = np.concatenate(labels)[:width].astype(np.int8)
    time_s = np.arange(width) / fs
    chest = rng.uniform(0.5, 2.0) * level + rng.normal(0.0, 0.01, width) + rng.normal(0.0, 0.1) * time_s / time_s[-1]
    echo = rng.normal(0.0, 0.05, width) - 3.0 * (label == NOISE)
    # Scaled roughly like nn_dataset.causal_normalise: breath depth for chest and velocity, echo about its median.
    scale = max(float(np.percentile(chest, 95) - np.percentile(chest, 5)), 1e-3)
    channels = np.stack([(chest - np.median(chest)) / scale, np.gradient(chest) * fs / scale, echo - np.median(echo)])
    boundaries = [(start - 0.5) / fs for start, _, _ in label_runs(label)[1:]]
    return channels.astype(np.float32), mask_around(label, time_s, boundaries, 0.1)


def toy_windows(n_windows: int = 18, window_s: float = 8.0, fs: float = 20.0, seed: int = 0) -> PhaseWindows:
    """A few cartoon breathing windows in the nn_dataset layout, for --smoke and tests.

    Half-cosine inhales and exhales, optional holds, the odd burst of noise and
    IGNORE within 0.1 s of every boundary.  It exercises the code; it is not a
    model of radar data.
    """

    if n_windows < 3:
        raise ValueError("need at least three windows, one per split")
    rng = np.random.default_rng(seed)
    width = int(round(window_s * fs))
    pairs = [_toy_window(width, fs, rng) for _ in range(n_windows)]
    held_out = max(1, n_windows // 6)
    split = ["train"] * (n_windows - 2 * held_out) + ["val"] * held_out + ["test"] * held_out
    meta = {
        "fs": float(fs),
        "window_s": float(window_s),
        "stride_s": float(window_s),
        "channels": ["chest", "velocity", "echo_db"],
        "class_names": list(CLASS_NAMES),
        "source": "toy windows (--smoke)",
    }
    return PhaseWindows(
        X=np.stack([x for x, _ in pairs]),
        y=np.stack([y for _, y in pairs]).astype(np.int64),
        split=np.array(split),
        run_id=np.array([f"toy_{index:03d}" for index in range(n_windows)]),
        start_s=np.arange(n_windows) * float(window_s),
        meta=meta,
    )


# ------------------------------------------------------------------ training


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 10
    grad_clip: float = 1.0
    augment: bool = False
    aug_scale: tuple[float, float] = (0.7, 1.4)
    aug_shift_s: float = 1.0
    max_steps: int | None = None  # --smoke stops after a few optimiser steps
    seed: int = 0


def pick_device(name: str) -> torch.device:
    """``auto`` takes CUDA, then Apple MPS, then CPU; CPU is the default because it is reproducible."""

    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    generator: torch.Generator,
    *,
    amplitude_channels: Sequence[int],
    scale_range: tuple[float, float],
    max_shift: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random breath depth and a small time shift, drawn per window.

    Chest and velocity get the same factor (velocity is the chest's
    derivative); the echo level is left alone.  The shift moves signal and
    labels together -- it changes where a window starts relative to the
    breaths, not how labels align with the signal -- and the frames it
    uncovers repeat the edge value with IGNORE labels.
    """

    batch, channels, width = x.shape
    if amplitude_channels:
        low, high = scale_range
        log_scale = torch.empty(batch, 1, 1).uniform_(math.log(low), math.log(high), generator=generator)
        index = list(amplitude_channels)
        x = x.clone()
        x[:, index] = x[:, index] * torch.exp(log_scale).to(x.device)
    if max_shift > 0:
        shifts = torch.randint(-max_shift, max_shift + 1, (batch, 1), generator=generator).to(x.device)
        source = torch.arange(width, device=x.device)[None, :] - shifts
        inside = (source >= 0) & (source < width)
        source = source.clamp(0, width - 1)
        x = torch.gather(x, 2, source[:, None, :].expand(batch, channels, width))
        y = torch.gather(y, 1, source).masked_fill(~inside, IGNORE)
    return x, y


@torch.no_grad()
def forward_split(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    y: np.ndarray | None = None,
    weight: torch.Tensor | None = None,
) -> tuple[np.ndarray, float]:
    """Log-probabilities [N, K, W] (each window from a cold start) and, with ``y``, the weighted loss."""

    model.eval()
    parts: list[np.ndarray] = []
    loss_sum = weight_sum = 0.0
    for start in range(0, len(X), batch_size):
        logits = model(torch.from_numpy(X[start : start + batch_size]).to(device)).float()
        parts.append(torch.log_softmax(logits, dim=1).cpu().numpy())
        if y is None:
            continue
        targets = torch.from_numpy(y[start : start + batch_size]).to(device)
        labelled = targets != IGNORE
        if labelled.any():
            # Summed, then divided once, so the split loss does not depend on batching.
            loss_sum += float(masked_cross_entropy(logits, targets, weight, reduction="sum"))
            weight_sum += float(weight[targets[labelled]].sum()) if weight is not None else float(labelled.sum())
    classes = int(getattr(model, "config", {}).get("num_classes", NUM_CLASSES))
    log_probs = np.concatenate(parts) if parts else np.zeros((0, classes, X.shape[-1]), np.float32)
    return log_probs, (loss_sum / weight_sum if weight_sum > 0 else float("nan"))


def viterbi_windows(log_probs: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    """Viterbi path for every window: [N, K, W] -> [N, W]."""

    if not len(log_probs):
        return np.zeros((0, log_probs.shape[-1]), dtype=np.int64)
    return np.stack([viterbi_decode(window.T, transitions) for window in log_probs])


def online_windows(log_probs: np.ndarray, fs: float, smoother: dict[str, Any]) -> np.ndarray:
    """What the live smoother would have emitted for every window: [N, K, W] -> [N, W]."""

    online = OnlinePhaseSmoother(fs, **smoother)
    paths = []
    for window in log_probs:
        online.reset()
        paths.append(online.update(np.exp(window.T)))
    return np.stack(paths) if paths else np.zeros((0, log_probs.shape[-1]), dtype=np.int64)


def pooled_boundaries(y: np.ndarray, predicted: np.ndarray, start_s: np.ndarray, fs: float) -> dict[str, Any]:
    """:func:`boundary_errors` per window, pooled over the windows.

    Counts, detection rate, mean error and false changes per minute pool
    exactly.  A median does not, so the matched-weighted median of the
    per-window medians is reported under its own name.
    """

    width = y.shape[1]
    offsets = np.arange(width) / fs
    minutes = (width - 1) / fs / 60.0
    reference = matched = false = 0
    error_sum = 0.0
    medians: list[tuple[float, int]] = []
    for labels, path, start in zip(y, predicted, start_s):
        scores = boundary_errors(labels, path, float(start) + offsets)
        reference += scores.reference_boundaries
        matched += scores.matched
        if scores.matched:
            error_sum += scores.mean_abs_error_s * scores.matched
            medians.append((scores.median_abs_error_s, scores.matched))
        if math.isfinite(scores.false_per_minute):
            false += int(round(scores.false_per_minute * minutes))
    median = float("nan")
    if medians:
        medians.sort()
        cumulative = np.cumsum([count for _, count in medians])
        median = medians[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))][0]
    total_minutes = minutes * len(y)
    return {
        "reference_boundaries": reference,
        "matched": matched,
        "detection_rate": matched / reference if reference else float("nan"),
        "mean_abs_error_s": error_sum / matched if matched else float("nan"),
        "median_of_window_medians_s": median,
        "false_boundaries": false,
        "false_per_minute": false / total_minutes if total_minutes > 0 else float("nan"),
    }


def score_split(
    log_probs: np.ndarray,
    y: np.ndarray,
    start_s: np.ndarray,
    fs: float,
    transitions: np.ndarray,
    smoother: dict[str, Any],
) -> dict[str, Any]:
    """Metrics of one split: Viterbi first, then plain argmax and the live smoother for comparison.

    Windows overlap when the dataset stride is shorter than a window, so a
    sample can be scored more than once, each time with the context its
    window gave it.
    """

    viterbi = viterbi_windows(log_probs, transitions)
    scores = score_phases(y.ravel(), viterbi.ravel())
    argmax = score_phases(y.ravel(), log_probs.argmax(axis=1).ravel())
    online_path = online_windows(log_probs, fs, smoother)
    online = score_phases(y.ravel(), online_path.ravel())
    return {
        "windows": int(len(y)),
        "labelled_samples": scores.samples,
        "accuracy": scores.accuracy,
        "macro_f1": scores.macro_f1,
        "per_class": scores.per_class,
        "confusion": scores.confusion.tolist(),
        "boundary": pooled_boundaries(y, viterbi, start_s, fs),
        "argmax": {"accuracy": argmax.accuracy, "macro_f1": argmax.macro_f1},
        "online": {
            "accuracy": online.accuracy,
            "macro_f1": online.macro_f1,
            # Labelled frames before the smoother's first decision are not scored at all.
            "undecided_samples": int(np.sum((y != IGNORE) & (online_path == IGNORE))),
            "boundary": pooled_boundaries(y, online_path, start_s, fs),
        },
    }


def evaluate_splits(
    model: nn.Module,
    X: np.ndarray,
    data: PhaseWindows,
    device: torch.device,
    decoding: dict[str, Any],
    *,
    splits: Sequence[str] = SPLITS,
    batch_size: int = 64,
) -> dict[str, dict[str, Any]]:
    transitions = transition_log_matrix(math.log(decoding["switch_prob"]))
    smoother = {key: decoding[key] for key in SMOOTHER_KEYS}
    results: dict[str, dict[str, Any]] = {}
    for name in splits:
        index = data.indices(name)
        if len(index):
            log_probs, _ = forward_split(model, X[index], device, batch_size)
            results[name] = score_split(log_probs, data.y[index], data.start_s[index], data.fs, transitions, smoother)
    return results


def train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    weight: np.ndarray,
    transitions: np.ndarray,
    device: torch.device,
    settings: TrainSettings,
    amplitude_channels: Sequence[int] = (),
    max_shift: int = 0,
    log: Log = _print,
) -> tuple[list[dict[str, Any]], int]:
    """Train in place and keep the weights of the best validation macro-F1; returns (history, best epoch)."""

    torch.manual_seed(settings.seed)
    order_rng = np.random.default_rng(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    weight_t = torch.as_tensor(weight, dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score, best_epoch, since_best, steps = -math.inf, 0, 0, 0
    try:
        for epoch in range(1, settings.epochs + 1):
            started = time.perf_counter()
            model.train()
            loss_sum, batches = 0.0, 0
            order = order_rng.permutation(len(X_train))
            for start in range(0, len(order), settings.batch_size):
                batch = order[start : start + settings.batch_size]
                x = torch.from_numpy(X_train[batch]).to(device)
                y = torch.from_numpy(y_train[batch]).to(device)
                if settings.augment:
                    x, y = augment_batch(
                        x,
                        y,
                        generator,
                        amplitude_channels=amplitude_channels,
                        scale_range=settings.aug_scale,
                        max_shift=max_shift,
                    )
                if not bool((y != IGNORE).any()):
                    continue
                loss = masked_cross_entropy(model(x), y, weight_t)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if settings.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
                optimiser.step()
                loss_sum += loss.item()
                batches += 1
                steps += 1
                if settings.max_steps and steps >= settings.max_steps:
                    break
            log_probs, val_loss = forward_split(model, X_val, device, settings.batch_size, y_val, weight_t)
            val = score_phases(y_val.ravel(), viterbi_windows(log_probs, transitions).ravel())
            score = val.macro_f1 if math.isfinite(val.macro_f1) else -math.inf
            improved = best_state is None or score > best_score + 1e-4
            if improved:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_score, best_epoch, since_best = score, epoch, 0
            else:
                since_best += 1
            row = {
                "epoch": epoch,
                "steps": steps,
                "train_loss": loss_sum / batches if batches else float("nan"),
                "val_loss": val_loss,
                "val_accuracy": val.accuracy,
                "val_macro_f1": val.macro_f1,
                "seconds": time.perf_counter() - started,
                "best": improved,
            }
            history.append(row)
            log(
                f"epoch {epoch:3d}  train loss {row['train_loss']:.4f}  val loss {val_loss:.4f}  "
                f"val acc {val.accuracy:.3f}  val macro-F1 {val.macro_f1:.3f}  ({row['seconds']:.1f} s){'  *' if improved else ''}"
            )
            if settings.max_steps and steps >= settings.max_steps:
                break
            if since_best >= settings.patience:
                log(f"early stop: no better validation macro-F1 for {settings.patience} epochs")
                break
    except KeyboardInterrupt:
        log("interrupted: keeping the best epoch so far")
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_epoch


# ------------------------------------------------------------------- output


def _jsonable(value: Any) -> Any:
    """Plain JSON (NaN -> null), also safe for torch.load(weights_only=True)."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (str, Path)):
        # str() also strips str subclasses (numpy strings, torch's TorchVersion)
        # that the weights-only unpickler refuses.
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def write_history(path: Path, history: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _fmt(value: Any, width: int = 6) -> str:
    return f"{value:{width}.3f}" if isinstance(value, float) and math.isfinite(value) else f"{'-':>{width}}"


def print_metrics(results: dict[str, dict[str, Any]], log: Log = _print) -> None:
    short = {"inhale": "in", "hold_after_exhale": "hold-ex", "exhale": "ex", "hold_after_inhale": "hold-in", "noise": "noise"}
    log(
        f"{'split':<6}{'windows':>8}{'acc':>7}{'mF1':>7}{'argmax':>7}{'online':>7}  "
        + "".join(f"{short.get(name, name):>8}" for name in CLASS_NAMES)
        + f"{'bnd det':>9}{'bnd err':>9}"
    )
    for name, split in results.items():
        f1 = [split["per_class"][cls]["f1"] for cls in CLASS_NAMES]
        log(
            f"{name:<6}{split['windows']:>8}{_fmt(split['accuracy'], 7)}{_fmt(split['macro_f1'], 7)}"
            f"{_fmt(split['argmax']['macro_f1'], 7)}{_fmt(split['online']['macro_f1'], 7)}  "
            + "".join(_fmt(value, 8) for value in f1)
            + f"{_fmt(split['boundary']['detection_rate'], 9)}{_fmt(split['boundary']['mean_abs_error_s'], 9)}"
        )
    log("(acc, mF1 = macro-F1 after Viterbi; argmax / online = macro-F1 without it / with the live smoother;"
        " bnd = boundary detection rate and mean error in s)")


# ---------------------------------------------------------------------- modes


def _parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.replace(" ", "").split(",") if part]


def model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.model == "tcn":
        return {
            "channels": args.tcn_channels,
            "kernel_size": args.kernel_size,
            "dilations": _parse_ints(args.dilations),
            "dropout": args.dropout,
        }
    return {"hidden": args.gru_hidden, "layers": args.gru_layers, "dropout": args.dropout}


def decoding_from(args: argparse.Namespace, stored: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decoder settings: those given on the command line, else the checkpoint's, else the defaults."""

    decoding = {**DEFAULT_DECODING, **(stored or {})}
    for key in DEFAULT_DECODING:
        if getattr(args, key, None) is not None:
            decoding[key] = getattr(args, key)
    return decoding


def settings_from(args: argparse.Namespace) -> TrainSettings:
    return TrainSettings(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        augment=args.augment,
        aug_scale=(float(args.aug_scale[0]), float(args.aug_scale[1])),
        aug_shift_s=args.aug_shift_s,
        seed=args.seed,
    )


def fit_and_save(
    data: PhaseWindows,
    args: argparse.Namespace,
    out_dir: Path,
    *,
    source: str,
    settings: TrainSettings,
    log: Log = _print,
) -> dict[str, Any]:
    """Train on ``data``, score every split with the best epoch and write the run to ``out_dir``."""

    inputs = [name for name in (args.inputs or "").split(",") if name] or data.channels
    X = select_channels(data, inputs)
    train_index, val_index = data.indices("train"), data.indices("val")
    if not len(train_index):
        raise SystemExit("The dataset has no 'train' windows.")
    if not len(val_index):
        log("warning: no 'val' windows, so the train split picks the epoch (expect overfitting)")
        val_index = train_index
    device = pick_device(args.device)
    torch.manual_seed(settings.seed)  # the initial weights too, not just the batches
    model = build_model(args.model, len(inputs), **model_kwargs(args)).to(device)
    decoding = decoding_from(args)
    transitions = transition_log_matrix(math.log(decoding["switch_prob"]))
    weight = class_weights(data.y[train_index], power=args.class_weight_power)
    width = data.X.shape[-1]

    log(f"dataset {source}: {len(data.X)} windows of {width / data.fs:.1f} s at {data.fs:g} Hz, inputs {inputs}")
    log("windows per split: " + ", ".join(f"{name} {len(data.indices(name))}" for name in SPLITS))
    log(f"model {args.model}: {count_parameters(model):,} parameters on {device}")
    if hasattr(model, "receptive_field_samples"):
        field = model.receptive_field_samples()
        log(f"receptive field {field} samples = {field / data.fs:.1f} s")
        if field > width:
            log("warning: the receptive field is longer than a window, so in training its oldest part only ever sees padding")
    log("class weights: " + ", ".join(f"{name} {value:.2f}" for name, value in zip(CLASS_NAMES, weight)))

    amplitude = [inputs.index(name) for name in AMPLITUDE_CHANNELS if name in inputs]
    history, best_epoch = train(
        model,
        X[train_index],
        data.y[train_index],
        X[val_index],
        data.y[val_index],
        weight=weight,
        transitions=transitions,
        device=device,
        settings=settings,
        amplitude_channels=amplitude,
        max_shift=int(round(settings.aug_shift_s * data.fs)),
        log=log,
    )
    results = evaluate_splits(model, X, data, device, decoding, batch_size=settings.batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "dataset": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "window_s": width / data.fs,
        "class_weights": dict(zip(CLASS_NAMES, weight.tolist())),
        "train_settings": asdict(settings),
        "args": vars(args),
    }
    save_checkpoint(
        out_dir / "model.pt",
        model,
        channels=inputs,
        fs=data.fs,
        decoding=_jsonable(decoding),
        run=_jsonable(run_info),
    )
    metrics = {
        "model_name": args.model,
        "model_config": dict(model.config),
        "parameters": count_parameters(model),
        "receptive_field_samples": getattr(model, "receptive_field_samples", lambda: None)(),
        "channels": inputs,
        "fs": data.fs,
        "decoding": decoding,
        **run_info,
        "splits": results,
    }
    write_json(out_dir / "metrics.json", metrics)
    write_history(out_dir / "history.csv", history)
    log(f"best epoch {best_epoch}; saved {out_dir}")
    print_metrics(results, log)
    return metrics


def evaluate_checkpoint(
    model_path: Path,
    dataset_path: Path,
    *,
    args: argparse.Namespace,
    out: Path | None = None,
    log: Log = _print,
) -> dict[str, Any]:
    """Score a saved model on every split of a dataset and write the metrics JSON."""

    device = pick_device(args.device)
    model, info = load_checkpoint(model_path, map_location=device)
    data = load_windows(dataset_path)
    if not math.isclose(data.fs, float(info["fs"]), rel_tol=1e-6):
        raise SystemExit(f"The model was trained at {info['fs']:g} Hz but the dataset is at {data.fs:g} Hz.")
    X = select_channels(data, info["channels"])
    decoding = decoding_from(args, info.get("decoding"))
    splits = [name for name in (args.splits or ",".join(SPLITS)).split(",") if name]
    results = evaluate_splits(model, X, data, device, decoding, splits=splits, batch_size=args.batch_size)
    report = {
        "model": str(model_path),
        "dataset": str(dataset_path),
        "model_name": info["model_name"],
        "model_config": info["model_config"],
        "channels": info["channels"],
        "fs": data.fs,
        "decoding": decoding,
        "splits": results,
    }
    out = out or model_path.parent / f"eval_{dataset_path.stem}.json"
    write_json(out, report)
    log(f"{info['model_name']} from {model_path} on {dataset_path}:")
    print_metrics(results, log)
    log(f"wrote {out}")
    return report


def run_smoke(args: argparse.Namespace) -> int:
    data = toy_windows(seed=args.seed)
    # Small batches so a dozen windows still give a few optimiser steps.
    settings = replace(settings_from(args), epochs=args.smoke_steps, batch_size=4, max_steps=args.smoke_steps)
    with tempfile.TemporaryDirectory(prefix="breath_phase_smoke_") as tmp:
        out_dir = (args.out_dir or Path(tmp)) / (args.name or f"smoke_{args.model}")
        fit_and_save(data, args, out_dir, source="toy windows (--smoke)", settings=settings)
        dataset_path = Path(tmp) / "toy_windows.npz"
        save_windows(dataset_path, data)
        evaluate_checkpoint(out_dir / "model.pt", dataset_path, args=args)
    _print("smoke run OK")
    return 0


def run_training(args: argparse.Namespace) -> int:
    data = load_windows(args.dataset)
    name = args.name or f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = (args.out_dir or DEFAULT_RUNS_DIR) / name
    if (out_dir / "model.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already holds a model; pick another --name or pass --overwrite.")
    fit_and_save(data, args, out_dir, source=str(args.dataset), settings=settings_from(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, help="nn_dataset .npz to train on")
    parser.add_argument("--evaluate", nargs=2, type=Path, metavar=("MODEL_PT", "DATASET_NPZ"), help="only score a saved model")
    parser.add_argument("--smoke", action="store_true", help="tiny in-memory run that proves the code works, no files needed")
    parser.add_argument("--smoke-steps", type=int, default=3, help="optimiser steps in --smoke (default 3)")

    model = parser.add_argument_group("model")
    model.add_argument("--model", choices=sorted(MODELS), default="tcn")
    model.add_argument("--inputs", default=None, help="comma-separated dataset channels to use (default: all)")
    model.add_argument("--tcn-channels", type=int, default=32, help="TCN width")
    model.add_argument("--kernel-size", type=int, default=5)
    model.add_argument("--dilations", default="1,2,4,8,16,32,64", help="TCN dilations, one causal conv each")
    model.add_argument("--gru-hidden", type=int, default=64)
    model.add_argument("--gru-layers", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.1, help="TCN dropout / dropout between GRU layers")

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=60)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=1e-2)
    training.add_argument("--patience", type=int, default=10, help="epochs without a better val macro-F1 before stopping")
    training.add_argument("--grad-clip", type=float, default=1.0, help="max gradient norm, 0 = off")
    training.add_argument("--class-weight-power", type=float, default=1.0,
                          help="1 = inverse class frequency (default), 0.5 = its square root, 0 = unweighted")
    training.add_argument("--augment", action="store_true", help="random chest/velocity scaling and small time shifts")
    training.add_argument("--aug-scale", type=float, nargs=2, default=(0.7, 1.4), metavar=("LOW", "HIGH"))
    training.add_argument("--aug-shift-s", type=float, default=1.0, help="largest time shift in seconds")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", choices=("cpu", "auto", "mps", "cuda"), default="cpu",
                          help="cpu (default, reproducible) or auto = cuda > mps > cpu")

    decoding = parser.add_argument_group("decoding (default: the checkpoint's settings, else these)")
    decoding.add_argument("--switch-prob", type=float, default=None, help="Viterbi switch probability (0.02)")
    decoding.add_argument("--min-phase-s", type=float, default=None, help="live smoother: shortest phase (0.3)")
    decoding.add_argument("--confirm-frames", type=int, default=None, help="live smoother: frames to confirm (3)")
    decoding.add_argument("--threshold", type=float, default=None, help="live smoother: confidence (0.5)")

    output = parser.add_argument_group("output")
    output.add_argument("--name", default=None, help="run name (default: <model>_<timestamp>)")
    output.add_argument("--out-dir", type=Path, default=None, help=f"parent of the run folder (default {DEFAULT_RUNS_DIR})")
    output.add_argument("--overwrite", action="store_true", help="replace an existing run of the same name")
    output.add_argument("--out", type=Path, default=None, help="--evaluate: metrics JSON (default next to the model)")
    output.add_argument("--splits", default=None, help="--evaluate: comma-separated splits (default train,val,test)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if [args.dataset is not None, args.evaluate is not None, args.smoke].count(True) != 1:
        parser.error("give exactly one of DATASET, --evaluate MODEL_PT DATASET_NPZ or --smoke")
    if args.smoke:
        return run_smoke(args)
    if args.evaluate:
        evaluate_checkpoint(args.evaluate[0], args.evaluate[1], args=args, out=args.out)
        return 0
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
