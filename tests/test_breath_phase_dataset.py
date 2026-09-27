from pathlib import Path

import numpy as np
import pytest

from respi_net.breath_phases import EXHALE, HOLD_AFTER_EXHALE, IGNORE, INHALE
from respi_net.chest_signal import ChestSignal
from respi_net.nn_dataset import (
    CHANNELS,
    assign_splits,
    export_dataset,
    labelled_run,
    load_dataset,
    load_run,
    causal_normalise,
    save_run,
)
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import score_phases

FS = 20.0


def _paced_with_hold(duration_s: float = 90.0, depth_mm: float = 8.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2 s inhale / 3 s exhale breathing with a 15 s hold after exhale at 40 s."""
    time_s = np.arange(0.0, duration_s, 1.0 / FS)
    chest = np.zeros(len(time_s))
    labels = np.full(len(time_s), IGNORE, dtype=np.int8)
    for start in np.arange(0.0, duration_s, 5.0):
        if 40.0 <= start < 55.0:
            continue
        inhale = (time_s >= start) & (time_s < start + 2.0)
        exhale = (time_s >= start + 2.0) & (time_s < start + 5.0)
        chest[inhale] = 0.5 - 0.5 * np.cos(np.pi * (time_s[inhale] - start) / 2.0)
        chest[exhale] = 0.5 + 0.5 * np.cos(np.pi * (time_s[exhale] - start - 2.0) / 3.0)
        labels[inhale], labels[exhale] = INHALE, EXHALE
    labels[(time_s >= 40.0) & (time_s < 55.0)] = HOLD_AFTER_EXHALE
    return time_s, depth_mm * chest, labels


def _signal(time_s: np.ndarray, chest: np.ndarray) -> ChestSignal:
    return ChestSignal("a121", time_s, FS, chest, "mm", chest, np.full(len(time_s), 40.0), {})


def test_baseline_finds_breaths_and_the_hold() -> None:
    rng = np.random.default_rng(1)
    time_s, chest, truth = _paced_with_hold()
    noisy = chest + 0.05 * rng.standard_normal(len(chest))
    offline = score_phases(truth, detect_phases(noisy, FS))
    realtime = score_phases(truth, detect_phases(noisy, FS, causal=True))
    assert offline.accuracy > 0.9
    assert offline.per_class["hold_after_exhale"]["f1"] > 0.9
    assert realtime.accuracy > 0.75
    assert realtime.accuracy < offline.accuracy


def test_causal_normalisation_ignores_depth_and_the_future() -> None:
    time_s, chest, _ = _paced_with_hold()
    deep = np.vstack([chest, np.gradient(chest) * FS, np.full(len(chest), 50.0)])
    shallow = np.vstack([0.05 * chest, np.gradient(0.05 * chest) * FS, np.full(len(chest), 20.0)])
    # After the first seconds of history (warm-up) depth no longer matters.
    assert np.allclose(causal_normalise(deep, FS)[:, 100:], causal_normalise(shallow, FS)[:, 100:], atol=1e-4)
    changed_future = deep.copy()
    changed_future[:, 1200:] *= 7.0
    assert np.allclose(causal_normalise(deep, FS)[:, :1200], causal_normalise(changed_future, FS)[:, :1200])


def test_runs_round_trip_and_export_splits_by_run(tmp_path: Path) -> None:
    runs = []
    for index in range(6):
        time_s, chest, labels = _paced_with_hold()
        run = labelled_run(f"run{index}", _signal(time_s, chest), labels, meta={"distance_cm": 50.0 + index})
        save_run(run, tmp_path / "runs" / f"{run.run_id}.npz")
        runs.append(load_run(tmp_path / "runs" / f"{run.run_id}.npz"))
    assert runs[0].features.shape == (len(CHANNELS), len(runs[0].labels))
    assert runs[0].meta["distance_cm"] == 50.0

    splits = assign_splits(runs, seed=3)
    assert set(splits.values()) == {"train", "val", "test"}
    meta = export_dataset(runs, tmp_path / "data.npz", window_s=60.0, stride_s=10.0, warmup_s=20.0, splits=splits)
    data = load_dataset(tmp_path / "data.npz")
    assert data["X"].shape[1:] == (len(CHANNELS), 1200)
    assert data["y"].shape == (data["X"].shape[0], 1200)
    assert np.all(data["y"][:, :400] == -1)
    for run_id in np.unique(data["run_id"]):
        assert len(set(data["split"][data["run_id"] == run_id])) == 1
    assert meta["fs"] == pytest.approx(FS)
    assert data["meta"]["channels"] == list(CHANNELS)
