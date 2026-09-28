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
