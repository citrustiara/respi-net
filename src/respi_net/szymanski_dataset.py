"""Loader for the respiratory-phase dataset of Szymański et al. (Sci Data 2025).

"Parallel Datasets for Classification of Respiratory Rhythm Phases",
doi:10.1038/s41597-025-04625-5, data doi:10.34808/ray2-4g53 (CC BY-NC-ND 4.0:
use it, do not redistribute modified copies -- it stays in the git-ignored
``data/external/szymanski2025/``).  Three people breathing in set ways
(normal, slow, shallow, hyperventilation, coughing, holds after inhale and
after exhale), recorded by a strain-gauge belt (HX711, ~10 Hz) and two
accelerometers, labelled by hand per sample.

Layout after unzipping ``Breath_dataset_and_apps.zip`` into ``unpacked/``::

    data/tens/synchronous_with_{inode,wit_motion}_acc/{raw,labelled}/tens_<name>.*
    data/{inode_acc,wit_motion_acc}/{raw,labelled}/acc_<name>.*

``raw/*.csv`` hold ``seconds,data`` (two files have no header row);
``labelled/*.txt`` hold ``normalised value, label, seconds`` per sample.  The
labelled rows are raw samples picked by time, but not always from the first
one or up to the last (the iNode-session "normal" files start a sample late
and stop 36 s early), so samples are paired by time, not by row.

The labels in the files are the paper's codes minus one -- their training
code adds the one back -- and their labelling tool names them: -1 "BREATH
OUT", 0 "OUT NO BREATH", 1 "BREATH IN", 2 "IN NO BREATH", 999 "NO RELEVANT".
So -1 is exhale, 0 hold after exhale, 1 inhale, 2 hold after inhale and 999
noise; with that reading every common transition in the files is a natural
one (exhale, hold, inhale, hold, exhale).  Here they become this project's
classes, which are the paper's codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .breath_phases import EXHALE, HOLD_AFTER_EXHALE, HOLD_AFTER_INHALE, IGNORE, INHALE, NOISE
from .chest_signal import ChestSignal, light_filter
from .nn_dataset import LabelledRun, labelled_run
from .paths import DATA_DIR

DEFAULT_ROOT = DATA_DIR / "external" / "szymanski2025" / "unpacked"
FILE_CODE_TO_CLASS = {-1: EXHALE, 0: HOLD_AFTER_EXHALE, 1: INHALE, 2: HOLD_AFTER_INHALE, 999: NOISE}


@dataclass(frozen=True)
class SzymanskiRecording:
    sensor: str
    group: str
    name: str
    subject: str
    time_s: np.ndarray
    raw: np.ndarray
    labels: np.ndarray

    @property
    def fs(self) -> float:
        return float(1.0 / np.median(np.diff(self.time_s)))

    @property
    def run_id(self) -> str:
        return f"szymanski_{self.group.replace('/', '_')}_{self.name}"


def nearest_index(times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of the sample in sorted ``times`` closest to each of ``targets``."""

    right = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    left = right - 1
    return np.where(targets - times[left] <= times[right] - targets, left, right)


def read_raw(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(seconds, value)`` from a raw file, with or without the header row."""

    first = path.read_text(encoding="utf-8").split("\n", 1)[0]
    has_header = not first.split(",")[0].replace(".", "", 1).lstrip("-").isdigit()
    frame = pd.read_csv(path, header=0 if has_header else None, names=["seconds", "data"])
    return frame["seconds"].to_numpy(dtype=float), frame["data"].to_numpy(dtype=float)


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(seconds, class)`` from a labelled file, in this project's classes."""

    frame = pd.read_csv(path, header=None, names=["normalised", "code", "seconds"])
    codes = frame["code"].round().astype(int)
    unknown = sorted(set(codes) - set(FILE_CODE_TO_CLASS))
    if unknown:
        raise ValueError(f"{path.name}: unexpected label codes {unknown}")
    return frame["seconds"].to_numpy(dtype=float), codes.map(FILE_CODE_TO_CLASS).to_numpy(dtype=np.int8)


def recordings(root: Path = DEFAULT_ROOT) -> list[SzymanskiRecording]:
    """Every labelled recording, with its raw signal cut to the labelled samples."""

    data_dir = root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Unzip Breath_dataset_and_apps.zip into {root} first.")
    found: list[SzymanskiRecording] = []
    for labelled in sorted(data_dir.rglob("labelled/*.txt")):
        raw_path = labelled.parent.parent / "raw" / f"{labelled.stem}.csv"
        if not raw_path.exists():
            continue
        group = "/".join(labelled.relative_to(data_dir).parts[:-2])
        sensor = "tensometer" if group.startswith("tens") else group
        raw_time, raw = read_raw(raw_path)
        label_time, labels = read_labels(labelled)
        name = labelled.stem.split("_", 1)[1]
        subject = {"second_subject": "S2", "third_subject": "S3"}.get(name, "S1")
        paired = raw[nearest_index(raw_time, label_time)]
        found.append(SzymanskiRecording(sensor, group, name, subject, label_time, paired, labels))
    return found


def orientation(recording: SzymanskiRecording) -> float:
    """+1 if the raw signal rises during labelled inhales, -1 if it falls."""

    velocity = np.gradient(recording.raw)
    inhale = recording.labels == INHALE
    exhale = recording.labels == EXHALE
    if not inhale.any() or not exhale.any():
        return 1.0
    return 1.0 if np.median(velocity[inhale]) >= np.median(velocity[exhale]) else -1.0


def to_labelled_run(
    recording: SzymanskiRecording,
    *,
    fs: float = 20.0,
    low_pass_hz: float = 2.0,
    sign: float | None = None,
) -> LabelledRun:
    """The recording as a :class:`LabelledRun` on this project's grid and conventions.

    Resampled to ``fs`` (linear for the signal, nearest sample for labels),
    oriented to rise on inhale and scaled to unit spread, since belt counts
    and accelerometer units mean nothing to the radar model.  There is no
    radar echo, so the echo channel is zero and flagged in ``meta``.
    """

    sign = orientation(recording) if sign is None else sign
    time_s = np.arange(recording.time_s[0], recording.time_s[-1], 1.0 / fs)
    values = np.interp(time_s, recording.time_s, sign * recording.raw)
    values = light_filter(values - np.median(values), fs, low_pass_hz)
    spread = float(np.percentile(values, 95) - np.percentile(values, 5)) or 1.0
    labels = recording.labels[nearest_index(recording.time_s, time_s)].astype(np.int8)
    signal = ChestSignal(
        source=recording.sensor,
        time_s=time_s - time_s[0],
        fs=fs,
        chest=values / spread,
        units="spread",
        raw=values,
        echo_db=np.zeros(len(time_s)),
        meta={"echo_available": False, "orientation": sign},
    )
    return labelled_run(
        recording.run_id,
        signal,
        labels,
        group=f"szymanski_{recording.subject}_{recording.name}",
        subject=f"szymanski_{recording.subject}",
        meta={"dataset": "szymanski2025", "sensor": recording.sensor, "recording": recording.name},
    )


def unlabelled_share(labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(labels) == IGNORE))
