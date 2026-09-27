from pathlib import Path

import numpy as np

from respi_net.breath_phases import EXHALE, HOLD_AFTER_EXHALE, HOLD_AFTER_INHALE, IGNORE, INHALE, NOISE
from respi_net.szymanski_dataset import (
    delay_labels,
    fix_hold_starts,
    nearest_index,
    orientation,
    recordings,
    to_labelled_run,
)

FS = 10.0
# File codes, one second each: exhale, hold after exhale, inhale, hold after inhale, noise.
PATTERN = [-1] * 10 + [0] * 10 + [1] * 10 + [2] * 10 + [999] * 10
LEAD = 2  # the belt labels run two samples ahead of the raw signal


def _write_recording(root: Path, group: str, name: str, *, header: bool, rising_on_inhale: bool, skip_first: int) -> np.ndarray:
    """Belt-like counts for the PATTERN, labelled like the published files.

    The labels lead the signal by LEAD samples, and the labelled file starts
    ``skip_first`` samples late.  Returns the counts.
    """
    codes = np.tile(PATTERN, 3)
    time_s = np.arange(len(codes)) / FS
    step = np.select([codes == 1, codes == -1], [1.0, -1.0], 0.0)
    counts = 600000.0 + 5000.0 * np.cumsum(step) * (1 if rising_on_inhale else -1)
    published = np.r_[codes[LEAD:], [999] * LEAD]
    raw_dir, labelled_dir = root / "data" / group / "raw", root / "data" / group / "labelled"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labelled_dir.mkdir(parents=True, exist_ok=True)
    rows = [f"{t:.3f},{value:.1f}" for t, value in zip(time_s, counts)]
    (raw_dir / f"{name}.csv").write_text("\n".join((["seconds,data"] if header else []) + rows) + "\n")
    labelled = [f"0.0,{code:.1f},{t:.3f}" for t, code in zip(time_s[skip_first:], published[skip_first:])]
    (labelled_dir / f"{name}.txt").write_text("\n".join(labelled) + "\n")
    return counts


def test_nearest_index_picks_the_closest_sample() -> None:
    times = np.array([0.0, 0.1, 0.2, 0.3])
    assert nearest_index(times, np.array([-1.0, 0.04, 0.06, 0.26, 9.0])).tolist() == [0, 0, 1, 3, 3]


def test_recordings_map_codes_pair_by_time_and_undo_the_lead(tmp_path: Path) -> None:
    belt = _write_recording(tmp_path, "tens/synchronous_with_inode_acc", "tens_normal", header=True, rising_on_inhale=False, skip_first=1)
    _write_recording(tmp_path, "inode_acc", "acc_second_subject", header=False, rising_on_inhale=True, skip_first=0)
    found = {recording.sensor: recording for recording in recordings(tmp_path)}
    assert set(found) == {"tensometer", "inode_acc"}

    tens = found["tensometer"]
    assert tens.name == "normal" and tens.subject == "S1" and found["inode_acc"].subject == "S2"
    assert abs(tens.fs - FS) < 1e-6
    # The labelled file starts one sample late; values stay paired with their own times.
    assert abs(tens.time_s[0] - 0.1) < 1e-9
    assert np.array_equal(tens.raw, belt[1:])

    # As published the labels are LEAD samples early; corrected they are back on the signal.
    published = {r.sensor: r for r in recordings(tmp_path, corrected=False)}["tensometer"]
    assert published.labels[:10].tolist() == [EXHALE] * 7 + [HOLD_AFTER_EXHALE] * 3
    assert tens.labels[:LEAD].tolist() == [IGNORE] * LEAD
    assert tens.labels[LEAD:30].tolist() == [EXHALE] * 7 + [HOLD_AFTER_EXHALE] * 10 + [INHALE] * 10 + [HOLD_AFTER_INHALE]
    assert NOISE in tens.labels

    # The belt counts fall on inhale here; the run is flipped to rise on inhale.
    assert orientation(tens) == -1.0 and orientation(found["inode_acc"]) == 1.0
    run = to_labelled_run(tens, fs=20.0)
    assert run.fs == 20.0 and run.meta["dataset"] == "szymanski2025"
    inhale = run.labels == INHALE
    assert np.median(np.gradient(run.features[0])[inhale]) > 0
    assert np.all(run.features[2] == 0.0)
    assert set(np.unique(run.labels)) == {IGNORE, EXHALE, HOLD_AFTER_EXHALE, INHALE, HOLD_AFTER_INHALE, NOISE}


def test_delay_labels_moves_labels_later() -> None:
    assert delay_labels(np.array([0, 0, 2, 2, 3], dtype=np.int8), 2).tolist() == [IGNORE, IGNORE, 0, 0, 2]


def test_holds_start_where_the_breath_stops() -> None:
    fs = 20.0
    time_s = np.arange(0.0, 30.0, 1.0 / fs)
    chest = np.sin(2 * np.pi * time_s / 4.0)
    labels = np.where(np.cos(2 * np.pi * time_s / 4.0) > 0, INHALE, EXHALE).astype(np.int8)
    # From 20 s: an inhale that goes on 0.5 s past where the hold is labelled,
    # then a quick settle of 0.3 s, then stillness.
    tail = time_s >= 20.0
    t = time_s[tail] - 20.0
    chest[tail] = np.select([t < 2.5, t < 2.8], [-1.0 + 0.9 * t, 1.25 - 0.5 * (t - 2.5) / 0.3], 0.75)
    labels[tail] = np.where(t < 2.0, INHALE, HOLD_AFTER_INHALE)

    fixed = fix_hold_starts(labels, chest, fs)
    hold_start = int(np.flatnonzero((fixed == HOLD_AFTER_INHALE) & tail)[0])
    assert 22.4 <= time_s[hold_start] <= 23.2
    assert np.all(fixed[(time_s >= 20.2) & (time_s < 22.3)] == INHALE)
    assert np.any(fixed[(time_s >= 22.4) & (time_s < 23.0)] == IGNORE)
    assert np.all(fixed[time_s >= 23.3] == HOLD_AFTER_INHALE)
    # Nothing else changes.
    assert np.array_equal(fixed[time_s < 19.5], labels[time_s < 19.5])
