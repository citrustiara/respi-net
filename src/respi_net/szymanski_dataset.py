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

The published labels are early, for two reasons found in the dataset's own
code (``code/brp-ml-model-main/scripts``).  The labelling tool plots the
first column of the labelled files, a smoothed and normalised copy of the
signal, and not the raw signal:

* the smoothing is ``np.convolve(x, ones(w) / w, mode="valid")`` written
  against the untrimmed time stamps, so every value is the mean of the *next*
  ``w`` samples and the copy runs (w-1)/2 samples ahead of the raw signal --
  2 samples (0.2 s) on the belt, 5 (0.2 s) on the iNode accelerometer.
  Rebuilt from the raw files with their code, the column matches exactly for
  the whole iNode session, and the labelled turns sit 0.2 s before the turns
  of the raw signal;
* the normalisation is a running min-max over the last 15 s, so a breath
  deeper than anything in that window is pinned at exactly +1 or -1 while it
  is still moving.  The end of a deep inhale looks flat there and was
  labelled as the start of the hold.

:func:`corrected_labels` undoes both: it delays the labels by the smoothing
lead and lets a breath run into a labelled hold for as long as the chest
still moves that breath's way.  A quick settle the other way at the start of
a hold (a little air let out after a deep inhale) is neither breath nor hold
and is left unlabelled.  The WitMotion labels were copied from the belt by
time stamp (``transfer_labels.py``) and sit 0.2-0.8 s early depending on the
file, so no fixed rule corrects them; they are left as published.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .breath_phases import EXHALE, HOLD_AFTER_EXHALE, HOLD_AFTER_INHALE, IGNORE, INHALE, NOISE, label_runs
from .chest_signal import ChestSignal, light_filter
from .nn_dataset import LabelledRun, labelled_run
from .paths import DATA_DIR

DEFAULT_ROOT = DATA_DIR / "external" / "szymanski2025" / "unpacked"
FILE_CODE_TO_CLASS = {-1: EXHALE, 0: HOLD_AFTER_EXHALE, 1: INHALE, 2: HOLD_AFTER_INHALE, 999: NOISE}
# Moving-average width in their preprocessing (categorise_automatically.py).
SMOOTHING_WINDOW = {"tensometer": 5, "inode_acc": 11, "wit_motion_acc": 11}
UNRELIABLE_TIMING = frozenset({"wit_motion_acc"})
HOLD_OF = {INHALE: HOLD_AFTER_INHALE, EXHALE: HOLD_AFTER_EXHALE}


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


def recordings(root: Path = DEFAULT_ROOT, *, corrected: bool = True) -> list[SzymanskiRecording]:
    """Every labelled recording, with its raw signal cut to the labelled samples.

    With ``corrected`` (the default) the labels go through
    :func:`corrected_labels`; without it they are as published.
    """

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
        recording = SzymanskiRecording(sensor, group, name, subject, label_time, paired, labels)
        if corrected:
            recording = replace(recording, labels=corrected_labels(recording))
        found.append(recording)
    return found


def delay_labels(labels: np.ndarray, samples: int) -> np.ndarray:
    """``labels`` moved ``samples`` later; the samples left uncovered get IGNORE."""

    delayed = np.full_like(labels, IGNORE)
    delayed[samples:] = labels[: len(labels) - samples]
    return delayed


def fix_hold_starts(
    labels: np.ndarray,
    chest: np.ndarray,
    fs: float,
    *,
    motion_fraction: float = 0.2,
    max_extend_s: float = 1.5,
    max_settle_s: float = 2.0,
    settle_gap_s: float = 0.3,
    min_stop_s: float = 0.2,
) -> np.ndarray:
    """Start each hold where the chest stops moving the preceding breath's way.

    ``chest`` rises on inhale.  Moving means faster than ``motion_fraction``
    of the recording's typical breathing speed (80th percentile of |velocity|),
    as in :func:`respi_net.label_alignment.align_cues`; the breath has stopped
    once it stays slower than that for ``min_stop_s``, so a brief slowdown in
    a two-step inhale does not end it.  Boundaries only move later and never
    past the end of the hold; a settle the other way right after the breath
    becomes IGNORE.
    """

    labels = np.array(labels, copy=True)
    velocity = np.gradient(light_filter(chest - np.median(chest), fs, 2.0)) * fs
    threshold = motion_fraction * float(np.percentile(np.abs(velocity), 80))
    pause = max(1, int(round(min_stop_s * fs)))
    runs = label_runs(labels)
    for (_, _, before), (start, stop, after) in zip(runs[:-1], runs[1:]):
        if before not in HOLD_OF or after != HOLD_OF[before]:
            continue
        direction = 1.0 if before == INHALE else -1.0
        limit = min(stop, start + int(round(max_extend_s * fs)))
        end = start
        while end < limit and np.any(direction * velocity[end : min(limit, end + pause)] > threshold):
            end += 1
        labels[start:end] = before
        # The settle starts once the chest has turned, which takes a moment.
        limit = min(stop, end + int(round(max_settle_s * fs)))
        turn = min(limit, end + int(round(settle_gap_s * fs)))
        settled = next((index for index in range(end, turn) if -direction * velocity[index] > threshold), None)
        if settled is None:
            continue
        while settled < limit and -direction * velocity[settled] > threshold:
            settled += 1
        labels[end:settled] = IGNORE
    return labels


def corrected_labels(recording: SzymanskiRecording) -> np.ndarray:
    """The recording's labels with the timing faults described above undone."""

    if recording.sensor in UNRELIABLE_TIMING:
        return recording.labels
    lead = (SMOOTHING_WINDOW[recording.sensor] - 1) // 2
    labels = delay_labels(recording.labels, lead)
    return fix_hold_starts(labels, orientation(recording) * recording.raw, recording.fs)


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
