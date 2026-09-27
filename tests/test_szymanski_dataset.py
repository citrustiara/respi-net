from pathlib import Path

import numpy as np

from respi_net.breath_phases import EXHALE, HOLD_AFTER_EXHALE, HOLD_AFTER_INHALE, INHALE, NOISE
from respi_net.szymanski_dataset import nearest_index, orientation, recordings, to_labelled_run

FS = 10.0
# File codes, one second each: exhale, hold after exhale, inhale, hold after inhale, noise.
PATTERN = [-1] * 10 + [0] * 10 + [1] * 10 + [2] * 10 + [999] * 10


def _write_recording(root: Path, group: str, name: str, *, header: bool, rising_on_inhale: bool, skip_first: int) -> np.ndarray:
    """Belt-like counts for the PATTERN; the labelled file starts ``skip_first`` samples late."""
    codes = np.tile(PATTERN, 3)
    time_s = np.arange(len(codes)) / FS
    step = np.select([codes == 1, codes == -1], [1.0, -1.0], 0.0)
    counts = 600000.0 + 5000.0 * np.cumsum(step) * (1 if rising_on_inhale else -1)
    raw_dir, labelled_dir = root / "data" / group / "raw", root / "data" / group / "labelled"
    raw_dir.mkdir(parents=True, exist_ok=True)
    labelled_dir.mkdir(parents=True, exist_ok=True)
    rows = [f"{t:.3f},{value:.1f}" for t, value in zip(time_s, counts)]
    (raw_dir / f"{name}.csv").write_text("\n".join((["seconds,data"] if header else []) + rows) + "\n")
    labelled = [f"0.0,{code:.1f},{t:.3f}" for t, code in zip(time_s[skip_first:], codes[skip_first:])]
    (labelled_dir / f"{name}.txt").write_text("\n".join(labelled) + "\n")
    return counts


def test_nearest_index_picks_the_closest_sample() -> None:
    times = np.array([0.0, 0.1, 0.2, 0.3])
    assert nearest_index(times, np.array([-1.0, 0.04, 0.06, 0.26, 9.0])).tolist() == [0, 0, 1, 3, 3]


def test_recordings_map_codes_and_pair_samples_by_time(tmp_path: Path) -> None:
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
    assert tens.labels[:30].tolist() == [EXHALE] * 9 + [HOLD_AFTER_EXHALE] * 10 + [INHALE] * 10 + [HOLD_AFTER_INHALE]
    assert NOISE in tens.labels

    # The belt counts fall on inhale here; the run is flipped to rise on inhale.
    assert orientation(tens) == -1.0 and orientation(found["inode_acc"]) == 1.0
    run = to_labelled_run(tens, fs=20.0)
    assert run.fs == 20.0 and run.meta["dataset"] == "szymanski2025"
    chest = run.features[0]
    inhale = run.labels == INHALE
    assert np.median(np.gradient(chest)[inhale]) > 0
    assert np.all(run.features[2] == 0.0)
    assert set(np.unique(run.labels)) == {EXHALE, HOLD_AFTER_EXHALE, INHALE, HOLD_AFTER_INHALE, NOISE}
