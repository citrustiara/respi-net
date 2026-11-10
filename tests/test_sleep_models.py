import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

torch = pytest.importorskip("torch")


def test_viterbi_keeps_a_stage_through_a_one_epoch_blip() -> None:
    from sleep_stage_crossval import transition_log, viterbi

    transitions = transition_log([np.array([1] * 50 + [2] * 50)], 4)
    emission = np.log(np.full((9, 4), 0.1))
    emission[:, 1] = np.log(0.7)
    emission[4] = np.log([0.1, 0.35, 0.45, 0.1])  # one epoch leans to deep sleep, the rest are light
    assert list(viterbi(emission, transitions)) == [1] * 9


def test_kappa_is_one_for_perfect_and_zero_for_constant_agreement() -> None:
    from sleep_stage_crossval import kappa

    y = np.array([0, 1, 1, 2, 3, 1, 2, 0])
    assert kappa(y, y, 4) == pytest.approx(1.0)
    assert kappa(y, np.ones_like(y), 4) == pytest.approx(0.0)


def test_spectral_input_puts_a_heartbeat_at_its_bpm() -> None:
    from train_sleep_heart_rate import BINS, FS, SPEC_MIN, WINDOW, spectral_input

    t = np.arange(WINDOW) / FS
    x = torch.tensor(np.tile(0.05 * np.sin(2 * np.pi * (72 / 60) * t), (1, BINS, 1)), dtype=torch.float32)
    spec = spectral_input(x)
    assert int(spec[0].mean(0).argmax()) + SPEC_MIN == 72


def test_time_stretch_scales_the_rhythm_and_the_label_together() -> None:
    from train_sleep_heart_rate import BINS, FS, PAD, SPEC_MIN, STRETCH, WINDOW, crop_and_stretch, spectral_input

    t = np.arange(WINDOW + 2 * PAD) / FS
    x = torch.tensor(np.tile(np.sin(2 * np.pi * (60 / 60) * t), (4, BINS, 1)), dtype=torch.float32)
    out, label = crop_and_stretch(x, torch.full((4,), 60.0), stretch=True)
    assert out.shape == (4, BINS, WINDOW)
    assert torch.all((label >= 60 * STRETCH[0]) & (label <= 60 * STRETCH[1]))
    peaks = spectral_input(out).mean(1).argmax(-1) + SPEC_MIN
    assert torch.all(torch.abs(peaks.float() - label) <= 1.0)


def test_balanced_stretch_spreads_the_heart_rate_and_keeps_the_rhythm_true() -> None:
    from train_sleep_heart_rate import BALANCED_BPM, BINS, FS, SPEC_MIN, WINDOW, crop_and_stretch, spectral_input

    torch.manual_seed(0)
    pad = int(12 * FS)
    t = np.arange(WINDOW + 2 * pad) / FS
    x = torch.tensor(np.tile(np.sin(2 * np.pi * (54 / 60) * t), (256, BINS, 1)), dtype=torch.float32)
    out, label = crop_and_stretch(x, torch.full((256,), 54.0), stretch=True, balanced=True)
    assert label.min() >= BALANCED_BPM[0] - 1e-3 and label.max() <= 54 * 1.4 + 1e-3
    assert (label > 66).float().mean() > 0.2  # well above the 0.9-1.1x range
    peaks = spectral_input(out).mean(1).argmax(-1) + SPEC_MIN
    assert torch.all(torch.abs(peaks.float() - label) <= 1.0)


def test_coverage_at_mae_takes_the_most_confident_windows_first() -> None:
    from train_sleep_heart_rate import coverage_at_mae

    label = np.full(4, 60.0)
    bpm = np.array([60.0, 61.0, 70.0, 80.0])
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    assert coverage_at_mae(bpm, conf, label, 1.0) == pytest.approx(0.5)
    assert coverage_at_mae(bpm, conf, label, 0.1) == pytest.approx(0.25)


def test_validation_is_the_end_of_every_training_night_with_a_gap() -> None:
    from train_sleep_heart_rate import BINS, Windows, split_train_val

    nights = [Windows(np.zeros((100, BINS, 4), np.float32), np.arange(100, dtype=float) + 1000 * k, np.zeros(100), np.zeros(100), str(k)) for k in range(2)]
    train, val = split_train_val(nights)
    assert len(val.label) == 40 and len(train.label) == 2 * (80 - 6)
    assert train.label[train.label < 1000].max() == 73 and val.label[val.label < 1000].min() == 80


def test_heart_variability_recovers_beat_to_beat_jitter() -> None:
    from build_sleep_epochs import heart_variability

    fs, rng = 20.0, np.random.default_rng(0)
    ibi = 1.0 + rng.normal(0, 0.04, 70)  # 60 bpm with 40 ms beat-to-beat scatter
    beats = np.cumsum(ibi)
    t = np.arange(int(beats[-1] * fs)) / fs
    chest = sum(0.05 * np.exp(-0.5 * ((t - b) / 0.12) ** 2) for b in beats)
    out = heart_variability(chest, fs, 60.0)
    assert out["radar_beats"] >= 50
    assert 30 < out["radar_sdnn_ms"] < 60
    assert 40 < out["radar_rmssd_ms"] < 80  # RMSSD of white jitter is about sqrt(2) x its SD
