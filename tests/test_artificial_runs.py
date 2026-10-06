import numpy as np

from respi_net.artificial_runs import augment_run, breath_size, noise_bank_from_holds, synthetic_run
from respi_net.breath_phases import EXHALE, HOLD_AFTER_EXHALE, IGNORE, INHALE, NOISE
from respi_net.chest_signal import ChestSignal
from respi_net.nn_dataset import labelled_run

FS = 20.0


def _run():
    """90 s of 2 s inhale / 3 s exhale breathing, 8 mm deep, with a 15 s hold after exhale at 40 s."""
    rng = np.random.default_rng(0)
    time_s = np.arange(0.0, 90.0, 1.0 / FS)
    chest = np.zeros(len(time_s))
    labels = np.full(len(time_s), IGNORE, dtype=np.int8)
    for start in np.arange(0.0, 90.0, 5.0):
        if 40.0 <= start < 55.0:
            continue
        inhale = (time_s >= start) & (time_s < start + 2.0)
        exhale = (time_s >= start + 2.0) & (time_s < start + 5.0)
        chest[inhale] = 0.5 - 0.5 * np.cos(np.pi * (time_s[inhale] - start) / 2.0)
        chest[exhale] = 0.5 + 0.5 * np.cos(np.pi * (time_s[exhale] - start - 2.0) / 3.0)
        labels[inhale], labels[exhale] = INHALE, EXHALE
    labels[(time_s >= 40.0) & (time_s < 55.0)] = HOLD_AFTER_EXHALE
    chest = 8.0 * chest + 0.1 * rng.standard_normal(len(time_s))
    signal = ChestSignal("a121", time_s, FS, chest, "mm", chest, np.full(len(time_s), 40.0), {})
    return labelled_run("run0", signal, labels, group="session0", subject="S01")


def test_augmented_copy_keeps_labels_in_step_and_its_split_group() -> None:
    run = _run()
    bank = noise_bank_from_holds([run])
    rng = np.random.default_rng(1)
    for stretch in (0.7, 1.4):
        copy = augment_run(run, stretch, 0.1, bank, rng)
        assert abs(len(copy.labels) - stretch * len(run.labels)) <= 1
        assert copy.group == "session0" and copy.subject == "S01" and copy.run_id == f"run0__x{stretch:g}"
        assert copy.meta["augmented_from"] == "run0" and copy.meta["stretch"] == stretch
        # Every phase lasts `stretch` times as long.
        for label in (INHALE, EXHALE, HOLD_AFTER_EXHALE):
            assert abs(np.sum(copy.labels == label) - stretch * np.sum(run.labels == label)) <= 3
        # The chest still rises on inhale and the hold is still there.
        velocity = copy.features[1]
        assert np.median(velocity[copy.labels == INHALE]) > 0 > np.median(velocity[copy.labels == EXHALE])
        assert np.all(np.isfinite(copy.features))


def test_noise_is_sized_against_the_breathing() -> None:
    run = _run()
    bank = noise_bank_from_holds([run])
    quiet = augment_run(run, 1.0, 0.02, bank, np.random.default_rng(2), drift_per_min=0.0)
    loud = augment_run(run, 1.0, 0.3, bank, np.random.default_rng(2), drift_per_min=0.0)
    hold = run.labels == HOLD_AFTER_EXHALE
    assert np.std(loud.features[0][hold]) > 3 * np.std(quiet.features[0][hold])
    assert np.std(loud.features[0][hold]) < 0.6 * breath_size(run.features[0])


def test_synthetic_runs_are_labelled_a121_like_runs() -> None:
    run = synthetic_run(3, 120.0, FS, np.random.default_rng(4))
    assert run.run_id == "synthetic_0003" and run.fs == FS and run.features.shape == (3, int(120 * FS))
    assert set(np.unique(run.labels)) <= {EXHALE, HOLD_AFTER_EXHALE, INHALE, 3, NOISE}
    assert np.all(np.isfinite(run.features))
    breathing = run.labels != NOISE
    velocity = run.features[1]
    assert np.median(velocity[breathing & (run.labels == INHALE)]) > 0


def test_the_noise_bank_comes_from_training_runs_only() -> None:
    import sys
    from pathlib import Path

    tools = str(Path(__file__).resolve().parents[1] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from build_artificial_data import training_noise_bank

    train = _run()
    other = _run()
    object.__setattr__(other, "run_id", "run1")  # the same hold noise, but from a held-out run
    bank = training_noise_bank([train, other], {"run0": "train", "run1": "test"})
    assert len(bank.segments) == len(noise_bank_from_holds([train]).segments) > 0


def test_enhanced_copies_inherit_labels_instead_of_detecting_corrupted_signal() -> None:
    from respi_net.artificial_runs import enhanced_augment_run, reduced_power_run
    from respi_net.breath_synth import time_warp

    run = _run()
    before_x, before_y = run.features.copy(), run.labels.copy()
    bank = noise_bank_from_holds([run])
    for stretch in (.7, 1.4):
        copy = enhanced_augment_run(run, stretch, .3, bank, np.random.default_rng(8), drift_per_min=.6)
        _, expected = time_warp(run.features[0], run.labels, stretch)
        np.testing.assert_array_equal(copy.labels, expected)
        assert copy.group == run.group
        np.testing.assert_allclose(copy.features[1], np.gradient(copy.features[0]) * copy.fs, rtol=.001, atol=.001)
    for drop in (0, 6, 12):
        copy = reduced_power_run(run, drop, np.random.default_rng(8))
        np.testing.assert_array_equal(copy.labels, run.labels)
        np.testing.assert_array_equal(copy.time_s, run.time_s)
        assert copy.group == run.group and copy.meta["snr_db"] == 35 - drop
    np.testing.assert_array_equal(run.features, before_x)
    np.testing.assert_array_equal(run.labels, before_y)


def test_reduced_power_model_rejects_nonphysical_belt_units() -> None:
    import pytest
    from dataclasses import replace
    from respi_net.artificial_runs import reduced_power_run

    run = _run()
    belt = replace(run, meta={**run.meta, "units": "spread"})
    with pytest.raises(ValueError, match="mm displacement"):
        reduced_power_run(belt, 12, np.random.default_rng(0))


def test_combined_export_deduplicates_contents_and_rejects_group_leakage(tmp_path) -> None:
    import json
    import sys
    import pytest
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from build_artificial_data import merge_window_datasets

    source = Path(__file__).resolve().parents[1] / "annotations/radar_close_v1/generated/dataset_manual_radar_v1.npz"
    result = merge_window_datasets(source, source, tmp_path / "combined.npz")
    with np.load(source, allow_pickle=False) as original, np.load(tmp_path / "combined.npz", allow_pickle=False) as combined:
        assert result["duplicates_removed"] >= len(original["y"])
        assert len(combined["y"]) <= len(original["y"])
        # Change a group's split without changing its contents: fail instead
        # of silently moving the same source between train and test.
        meta = json.loads(str(original["meta"]))
        first = next(iter(meta["runs"].values()))
        first["split"] = "test" if first["split"] != "test" else "train"
        fields = {key: original[key] for key in original.files if key != "meta"}
        np.savez_compressed(tmp_path / "bad.npz", **fields, meta=np.array(json.dumps(meta)))
    with pytest.raises(ValueError, match="source group"):
        merge_window_datasets(source, tmp_path / "bad.npz", tmp_path / "bad_combined.npz")
