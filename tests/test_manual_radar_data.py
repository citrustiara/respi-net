"""Checks for annotation uncertainty and leakage in the reviewed radar bundle."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from respi_net.nn_dataset import load_run

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manual_radar", ROOT / "tools" / "build_manual_radar_data.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_reviewed_labels_mask_hold_edges_and_unreviewed_regions():
    t = np.arange(0, 15, .05)
    annotation = {"segments": [[2, 5, 2], [5, 10, 3]], "boundary_guard_s": .3, "hold_boundary_guard_s": .6}
    y = builder.reviewed_labels(t, annotation)
    assert np.all(y[t < 1.7] == -1)
    assert np.all(y[(t > 2.4) & (t < 4.3)] == 2)
    assert np.all(y[np.abs(t - 5) < .6] == -1)
    assert np.all(y[(t > 5.7) & (t < 9.3)] == 3)
    assert np.all(y[t > 10] == -1)
    with pytest.raises(ValueError):
        builder.reviewed_labels(t, {**annotation, "segments": [[2, 6, 2], [5, 10, 3]]})


def test_bundle_preserves_source_groups_and_reduced_power_model():
    path = ROOT / "annotations" / "radar_close_v1" / "generated"
    with np.load(path / "dataset_manual_radar_v1.npz", allow_pickle=False) as data:
        assert np.all(np.isfinite(data["X"]))
        assert data["X"].shape[0] == len(data["split"])
        assert set(np.unique(data["y"])) <= {-1, 0, 1, 2, 3, 4}
        for split in ("val", "test"):
            assert not any("__" in name for name in data["run_id"][data["split"] == split])
        assert set(data["run_id"][data["split"] == "val"]) == {"self_lying_3min_01"}
        assert set(data["run_id"][data["split"] == "test"]) == {"sit_nophone_01"}
        # Both retention classes occur in training; noise still needs new data.
        assert {0, 1, 2, 3} <= set(np.unique(data["y"][data["split"] == "train"]))
    for source in ("phase1_halfside_02", "pilot_coach_1m"):
        original = load_run(path / "runs" / f"{source}.npz")
        full = load_run(path / "runs" / f"{source}__iq_drop0.npz")
        weak = load_run(path / "runs" / f"{source}__iq_drop12.npz")
        assert full.group == weak.group == original.group
        np.testing.assert_array_equal(original.labels, weak.labels)
        assert 10 < np.median(full.features[2] - weak.features[2]) < 14
        for stretch in builder.STRETCH:
            copy = load_run(path / "runs" / f"{source}__x{stretch:g}.npz")
            assert copy.group == original.group
            np.testing.assert_allclose(copy.features[1], np.gradient(copy.features[0]) * copy.fs, rtol=.001, atol=.001)
