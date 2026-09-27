"""Training windows for the breathing-phase network, in one file format.

A labelled run keeps its physical signals -- chest in mm, velocity in mm/s,
echo strength in dB -- and one label per sample.  Windows are cut and
normalised only when a dataset is exported, so the same runs can feed any
window length later.

Chest and velocity are divided by the breathing scale of the *preceding*
30 s, so deep coached breaths and sub-millimetre sleep breaths look alike to
the network -- phases are about the shape of a breath, not its depth -- and a
sample is scaled exactly as it would be live, from the past only.  The first
seconds of every window are context, not training targets: a causal model
sees a full receptive field there only in a live stream, so those samples
are labelled IGNORE.  Splits are by run or a coarser group: windows cut from
one recording never land on both sides of a split.

Exported file (``.npz``): ``X`` float32 [N, C, W], ``y`` int8 [N, W] (0-4,
-1 = ignore), ``split`` [N] ("train"/"val"/"test"), ``run_id`` [N],
``start_s`` float32 [N], ``meta`` JSON (fs, window_s, stride_s, warmup_s,
channels, class_names, runs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .breath_phases import CLASS_NAMES, IGNORE, NUM_CLASSES
from .chest_signal import ChestSignal

CHANNELS = ("chest", "velocity", "echo_db")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class LabelledRun:
    run_id: str
    group: str
    subject: str
    fs: float
    time_s: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(len(self.labels) / self.fs)

    @property
    def labelled_s(self) -> float:
        return float(np.count_nonzero(self.labels != IGNORE) / self.fs)


def labelled_run(
    run_id: str,
    signal: ChestSignal,
    labels: np.ndarray,
    *,
    group: str | None = None,
    subject: str = "S01",
    meta: Mapping[str, Any] | None = None,
) -> LabelledRun:
    """A run in :data:`CHANNELS` order from a chest signal and its per-sample labels."""

    labels = np.asarray(labels, dtype=np.int8)
    if len(labels) != len(signal.time_s):
        raise ValueError("labels and signal differ in length")
    features = np.vstack([signal.chest, signal.velocity(), signal.echo_db]).astype(np.float32)
    info = {"source": signal.source, "units": signal.units, **signal.meta, **dict(meta or {})}
    return LabelledRun(run_id, group or run_id, subject, float(signal.fs), np.asarray(signal.time_s, dtype=float), features, labels, info)


def save_run(run: LabelledRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run.run_id, "group": run.group, "subject": run.subject, "fs": run.fs, "channels": list(CHANNELS), **run.meta}
    np.savez_compressed(path, time_s=run.time_s, features=run.features, labels=run.labels, meta=np.array(json.dumps(meta, default=str)))


def load_run(path: Path) -> LabelledRun:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return LabelledRun(
            run_id=meta.pop("run_id"),
            group=meta.pop("group"),
            subject=meta.pop("subject"),
            fs=float(meta.pop("fs")),
            time_s=data["time_s"],
            features=data["features"],
            labels=data["labels"],
            meta=meta,
        )


def causal_normalise(
    features: np.ndarray,
    fs: float,
    *,
    history_s: float = 30.0,
    step_s: float = 0.5,
    min_scale: float = 1e-3,
) -> np.ndarray:
    """Scale a run the way a live system can: from the past ``history_s`` only.

    Every ``step_s`` the chest median and breathing scale (5th-95th
    percentile spread, a robust breath depth) and the echo median are taken
    over the samples up to that moment; chest is centred and divided by the
    scale, velocity divided by it, echo centred.  ``min_scale`` keeps a still
    stretch from being blown up into noise.  Until ``history_s`` has passed
    the history is simply shorter.
    """

    features = np.asarray(features, dtype=float)
    out = np.empty_like(features)
    history = max(1, int(round(history_s * fs)))
    step = max(1, int(round(step_s * fs)))
    for start in range(0, features.shape[1], step):
        stop = min(features.shape[1], start + step)
        past = features[:, max(0, start - history) : start + 1]
        chest = past[0]
        scale = max(float(np.percentile(chest, 95) - np.percentile(chest, 5)), min_scale)
        out[0, start:stop] = (features[0, start:stop] - np.median(chest)) / scale
        out[1, start:stop] = features[1, start:stop] / scale
        out[2:, start:stop] = features[2:, start:stop] - np.median(past[2:], axis=-1, keepdims=True)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def make_windows(
    runs: Sequence[LabelledRun],
    *,
    window_s: float = 60.0,
    stride_s: float = 5.0,
    warmup_s: float = 20.0,
    min_labelled_fraction: float = 0.25,
) -> dict[str, np.ndarray]:
    """Causally normalised windows inside each run; the first ``warmup_s`` of each is context only."""

    xs, ys, run_ids, starts = [], [], [], []
    for run in runs:
        width = int(round(window_s * run.fs))
        stride = max(1, int(round(stride_s * run.fs)))
        warmup = min(width, int(round(warmup_s * run.fs)))
        normalised = causal_normalise(run.features, run.fs)
        for start in range(0, len(run.labels) - width + 1, stride):
            labels = run.labels[start : start + width].astype(np.int8)
            labels[:warmup] = IGNORE
            if np.mean(labels[warmup:] != IGNORE) < min_labelled_fraction:
                continue
            xs.append(normalised[:, start : start + width])
            ys.append(labels)
            run_ids.append(run.run_id)
            starts.append(float(run.time_s[start]))
    if not xs:
        channels = len(CHANNELS)
        return {
            "X": np.zeros((0, channels, 0), np.float32),
            "y": np.zeros((0, 0), np.int8),
            "run_id": np.array([], dtype=str),
            "start_s": np.zeros(0, np.float32),
        }
    return {
        "X": np.stack(xs),
        "y": np.stack(ys),
        "run_id": np.array(run_ids),
        "start_s": np.array(starts, dtype=np.float32),
    }


def assign_splits(
    runs: Sequence[LabelledRun],
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    *,
    seed: int = 0,
) -> dict[str, str]:
    """``run_id -> split``, whole groups at a time, filled by labelled duration.

    Groups are shuffled with ``seed`` and handed to whichever split is
    furthest below its share; with at least three groups every split gets one.
    """

    groups: dict[str, list[LabelledRun]] = {}
    for run in runs:
        groups.setdefault(run.group, []).append(run)
    order = list(groups)
    np.random.default_rng(seed).shuffle(order)
    target = np.asarray(fractions, dtype=float) / float(np.sum(fractions))
    filled = np.zeros(len(SPLITS))
    total = sum(run.labelled_s for run in runs) or 1.0
    assignment: dict[str, str] = {}
    for index, group in enumerate(order):
        remaining = len(order) - index
        empty = [k for k in range(len(SPLITS)) if filled[k] == 0 and target[k] > 0]
        if empty and remaining <= len(empty):
            choice = empty[0]
        else:
            choice = int(np.argmax(target - filled / total))
        duration = sum(run.labelled_s for run in groups[group])
        filled[choice] += duration
        for run in groups[group]:
            assignment[run.run_id] = SPLITS[choice]
    return assignment


def export_dataset(
    runs: Sequence[LabelledRun],
    path: Path,
    *,
    window_s: float = 60.0,
    stride_s: float = 5.0,
    warmup_s: float = 20.0,
    splits: Mapping[str, str] | None = None,
    min_labelled_fraction: float = 0.25,
    description: str = "",
) -> dict[str, Any]:
    """Write the windows of ``runs`` to one ``.npz`` in the format described above."""

    fs_values = {round(run.fs, 3) for run in runs}
    if len(fs_values) != 1:
        raise ValueError(f"Runs must share one sample rate, got {sorted(fs_values)}")
    splits = dict(splits or assign_splits(runs))
    windows = make_windows(
        runs, window_s=window_s, stride_s=stride_s, warmup_s=warmup_s, min_labelled_fraction=min_labelled_fraction
    )
    split = np.array([splits[run_id] for run_id in windows["run_id"]])
    counts = {
        name: {
            "windows": int(np.sum(split == name)),
            "runs": sorted({run.run_id for run in runs if splits[run.run_id] == name}),
            "class_samples": np.bincount(
                windows["y"][split == name][windows["y"][split == name] >= 0].ravel(), minlength=NUM_CLASSES
            ).tolist()
            if np.any(split == name)
            else [0] * NUM_CLASSES,
        }
        for name in SPLITS
    }
    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "fs": float(next(iter(fs_values))),
        "window_s": float(window_s),
        "stride_s": float(stride_s),
        "warmup_s": float(warmup_s),
        "channels": list(CHANNELS),
        "class_names": list(CLASS_NAMES),
        "normalisation": "causal, trailing 30 s: chest minus median and / (p95-p5 of chest); velocity / same; echo_db minus median",
        "runs": {run.run_id: {"group": run.group, "subject": run.subject, "split": splits[run.run_id], **run.meta} for run in runs},
        "splits": counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=windows["X"],
        y=windows["y"],
        split=split,
        run_id=windows["run_id"],
        start_s=windows["start_s"],
        meta=np.array(json.dumps(meta, default=str)),
    )
    return meta


def load_dataset(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        loaded = {key: data[key] for key in data.files if key != "meta"}
        loaded["meta"] = json.loads(str(data["meta"]))
    return loaded


def runs_in(directory: Path) -> Iterable[LabelledRun]:
    for path in sorted(directory.glob("*.npz")):
        yield load_run(path)
