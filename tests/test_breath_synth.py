from dataclasses import replace

import numpy as np
import pytest

from respi_net.breath_phases import (
    EXHALE,
    HOLD_AFTER_EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    NOISE,
    label_runs,
    labels_from_segments,
    transitions_allowed,
)
from respi_net.breath_synth import (
    A121_WAVELENGTH_MM,
    NoiseBank,
    SynthConfig,
    a121_forward,
    add_drift,
    add_noise,
    displacement_from_iq,
    random_crop,
    scale_depth,
    simulate_breathing,
    time_warp,
)

FS = 20.0

# Breathing alone: no holds, sighs, bursts or noise to blur counts and levels.
QUIET = SynthConfig(
    hold_after_inhale_prob=0.0,
    hold_after_exhale_prob=0.0,
    sigh_prob=0.0,
    heart_mm=(0.0, 0.0),
    drift_mm_per_min=0.0,
    posture_steps_per_min=0.0,
    white_noise_mm=(0.0, 0.0),
    lf_noise_mm=(0.0, 0.0),
    motion_bursts_per_min=0.0,
)
# Breaths deep enough that the radar phase wraps many times over.
DEEP = replace(QUIET, depth_mm=(6.0, 8.0), rate_bpm=(10.0, 14.0), heart_mm=(0.05, 0.3))


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _runs(labels: np.ndarray, label: int) -> list[tuple[int, int]]:
    return [(start, stop) for start, stop, run_label in label_runs(labels) if run_label == label]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


@pytest.mark.parametrize(("seed", "fs"), [(0, 20.0), (1, 20.0), (2, 5.0), (3, 1.0)])
def test_every_sample_gets_a_phase_in_an_allowed_order(seed: int, fs: float) -> None:
    config = SynthConfig(hold_after_inhale_prob=0.3, hold_after_exhale_prob=0.3, motion_bursts_per_min=2.0)
    sim = simulate_breathing(300, fs, _rng(seed), config)

    assert sim.labels.dtype == np.int8
    assert len(sim.labels) == len(sim.chest_mm) == len(sim.time_s) == round(300 * fs)
    assert set(np.unique(sim.labels)) == {INHALE, HOLD_AFTER_EXHALE, EXHALE, HOLD_AFTER_INHALE, NOISE}
    assert transitions_allowed(sim.labels)
    # The events tile the recording and rebuild the labels exactly.
    assert sim.events[0][0] == 0.0 and sim.events[-1][1] == pytest.approx(300)
    assert all(before[1] == after[0] for before, after in zip(sim.events, sim.events[1:]))
    assert np.array_equal(labels_from_segments(sim.time_s, sim.events), sim.labels)


def test_motion_bursts_are_large_and_the_only_noise() -> None:
    calm = simulate_breathing(300, FS, _rng(4), replace(QUIET, depth_mm=(0.5, 1.0)))
    moving = simulate_breathing(
        300, FS, _rng(4), replace(QUIET, depth_mm=(0.5, 1.0), motion_bursts_per_min=3.0, motion_burst_mm=(10.0, 20.0))
    )

    assert NOISE not in calm.labels
    bursts = [(a, b) for a, b in _runs(moving.labels, NOISE) if a > 0 and b < len(moving.labels)]
    assert len(bursts) >= 3
    assert all(np.ptp(moving.chest_mm[a:b]) > 5.0 for a, b in bursts)


def test_inhale_onsets_follow_the_configured_rate() -> None:
    steady = replace(QUIET, rate_bpm=(15.0, 15.0), rate_walk=0.0, period_jitter=0.0)
    assert abs(len(_runs(simulate_breathing(600, FS, _rng(5), steady).labels, INHALE)) - 150) <= 1

    for seed in range(4):
        sim = simulate_breathing(1200, FS, _rng(seed), replace(QUIET, rate_bpm=(8.0, 20.0)))
        per_minute = len(_runs(sim.labels, INHALE)) / 20
        assert 8.0 * 0.95 <= per_minute <= 20.0 * 1.05


def test_holds_keep_the_chest_where_the_last_breath_left_it() -> None:
    noisy = replace(
        QUIET,
        depth_mm=(2.0, 4.0),
        hold_after_inhale_prob=0.5,
        hold_after_exhale_prob=0.5,
        heart_mm=(0.05, 0.4),
        lf_noise_mm=(0.02, 0.1),
    )
    sim = simulate_breathing(600, FS, _rng(6), noisy)
    full = [np.median(sim.chest_mm[a:b]) for a, b in _runs(sim.labels, HOLD_AFTER_INHALE)]
    empty = [np.median(sim.chest_mm[a:b]) for a, b in _runs(sim.labels, HOLD_AFTER_EXHALE)]

    assert len(full) > 5 and len(empty) > 5
    assert min(full) > max(empty) + 0.5

    clean = simulate_breathing(600, FS, _rng(7), replace(QUIET, hold_after_inhale_prob=0.5, hold_after_exhale_prob=0.5))
    for start, stop in _runs(clean.labels, HOLD_AFTER_INHALE):
        plateau = clean.chest_mm[start:stop]
        assert start == 0 or plateau[0] >= clean.chest_mm[start - 1]  # starts at the top of the inhale
        assert plateau.min() >= 0.9 * plateau[0]  # and sags by no more than hold_sag
    for start, stop in _runs(clean.labels, HOLD_AFTER_EXHALE):
        assert np.all(clean.chest_mm[start:stop] == 0.0)  # the end-expiratory level


def test_scale_depth_scales_breaths_and_leaves_the_input_alone() -> None:
    chest = simulate_breathing(60, FS, _rng(8), QUIET).chest_mm
    original = chest.copy()

    shallow = scale_depth(chest, 0.1)

    assert np.array_equal(chest, original)
    assert np.ptp(shallow) == pytest.approx(0.1 * np.ptp(original))
    stacked = scale_depth(np.stack([original, -original]), 0.5)
    assert stacked.shape == (2, len(original))
    assert np.allclose(stacked[1], -0.5 * original)


@pytest.mark.parametrize("factor", [0.5, 0.8, 1.5, 2.0])
def test_time_warp_stretches_phases_by_the_factor_and_keeps_labels_on_their_samples(factor: float) -> None:
    labels = np.repeat(
        np.array([INHALE, HOLD_AFTER_INHALE, EXHALE, IGNORE, INHALE], dtype=np.int8), [60, 40, 70, 10, 20]
    )
    position = np.arange(len(labels), dtype=float)  # each sample's signal is its own position
    signal = np.stack([position, -position])

    warped, warped_labels = time_warp(signal, labels, factor)

    assert warped.shape == (2, round(len(labels) * factor))
    assert warped_labels.dtype == np.int8
    assert [b - a for a, b in _runs(warped_labels, HOLD_AFTER_INHALE)] == [pytest.approx(40 * factor, abs=1)]
    # Every output label is the label of the input sample its signal was read from;
    # samples read exactly half-way between two inputs may take either neighbour.
    source = warped[0]
    exact = np.abs(source - np.floor(source) - 0.5) > 1e-6
    assert np.array_equal(warped_labels[exact], labels[np.rint(source[exact]).astype(int)])
    assert np.array_equal(warped[1], -warped[0])
    assert np.array_equal(signal[0], position)


def test_round_trip_is_exact_without_clutter_or_noise_even_when_the_phase_wraps() -> None:
    chest = simulate_breathing(120, FS, _rng(9), DEEP).chest_mm
    ideal = a121_forward(chest, FS, _rng(10), snr_db=300.0, static_clutter=0.0, fading=0.0)

    assert np.ptp(chest) > A121_WAVELENGTH_MM  # well past a quarter wavelength
    assert np.any(np.abs(np.diff(np.angle(ideal[:, 2]))) > np.pi)  # so the raw phase wraps
    # Up to an offset, with the sign kept: chest_mm rises on inhale either way.
    assert np.allclose(displacement_from_iq(ideal), chest - chest.mean(), atol=1e-6)


@pytest.mark.parametrize("depth_mm", [(0.5, 1.0), (6.0, 8.0)])
def test_round_trip_recovers_the_chest_at_high_snr(depth_mm: tuple[float, float]) -> None:
    chest = simulate_breathing(120, FS, _rng(11), replace(DEEP, depth_mm=depth_mm)).chest_mm
    iq = a121_forward(chest, FS, _rng(12), snr_db=30.0)

    assert iq.shape == (len(chest), 5) and np.iscomplexobj(iq)
    assert _correlation(displacement_from_iq(iq), chest) > 0.99


def test_low_snr_breaks_the_round_trip_with_phase_slips() -> None:
    chest = simulate_breathing(120, FS, _rng(13), DEEP).chest_mm
    recovered = {snr: displacement_from_iq(a121_forward(chest, FS, _rng(14), snr_db=snr)) for snr in (30.0, 0.0)}

    assert _correlation(recovered[30.0], chest) > 0.99
    assert _correlation(recovered[0.0], chest) < 0.6
    # Unwrapping slips add whole turns, steps of λ/2, that a clean read never shows.
    error = recovered[0.0] - (chest - chest.mean())
    assert np.ptp(error) > A121_WAVELENGTH_MM / 2


def test_same_seed_same_output() -> None:
    first = simulate_breathing(60, FS, _rng(15))
    again = simulate_breathing(60, FS, _rng(15))
    other = simulate_breathing(60, FS, _rng(16))

    assert np.array_equal(first.chest_mm, again.chest_mm)
    assert np.array_equal(first.labels, again.labels) and first.events == again.events
    assert not np.array_equal(first.chest_mm, other.chest_mm)
    iq = [a121_forward(first.chest_mm, FS, _rng(17), snr_db=10.0) for _ in range(2)]
    assert np.array_equal(iq[0], iq[1])
    drifted = [add_drift(first.chest_mm, FS, _rng(18), 1.0) for _ in range(2)]
    assert np.array_equal(drifted[0], drifted[1])
    crops = [random_crop(first.chest_mm, first.labels, 100, _rng(19)) for _ in range(2)]
    assert np.array_equal(crops[0][0], crops[1][0]) and np.array_equal(crops[0][1], crops[1][1])


def test_noise_bank_crops_long_segments_and_tiles_short_ones() -> None:
    recorded = _rng(20).standard_normal(500)
    bank = NoiseBank([recorded])

    crop = bank.sample(50, _rng(21))
    windows = np.lib.stride_tricks.sliding_window_view(recorded - recorded.mean(), 50)
    assert np.any(np.all(windows == crop, axis=1))  # one contiguous stretch, mean removed
    crop[:] = 1e9
    assert not np.any(bank.segments[0] == 1e9)

    tiled = NoiseBank([np.arange(10.0)]).sample(35, _rng(22))
    assert tiled.shape == (35,)
    assert set(tiled + 4.5) <= set(range(10))
    assert np.ptp(tiled) == 9.0  # it runs through the whole segment
    assert np.all(np.abs(np.diff(tiled)) <= 1.0)  # forwards then backwards: no jump where copies meet


def test_add_noise_shares_one_trace_across_channels() -> None:
    signal = np.zeros((2, 100))

    noisy = add_noise(signal, np.ones(100), scale=0.5)

    assert noisy.shape == (2, 100) and np.all(noisy == 0.5) and np.all(signal == 0.0)
    with pytest.raises(ValueError, match="NoiseBank.sample"):
        add_noise(signal, np.ones(99))


def test_add_drift_never_climbs_faster_than_allowed() -> None:
    drift = add_drift(np.zeros(int(FS * 300)), FS, _rng(23), max_mm_per_min=2.0)
    slope_mm_per_min = np.abs(np.gradient(drift)) * FS * 60.0

    assert drift[0] == 0.0
    assert 0.0 < slope_mm_per_min.max() <= 2.0 + 1e-9


def test_random_crop_keeps_signal_and_labels_together_and_pads_with_ignore() -> None:
    position = np.arange(100.0)
    labels = np.repeat(np.array([INHALE, EXHALE] * 5, dtype=np.int8), 10)

    crop, crop_labels = random_crop(np.stack([position, position]), labels, 30, _rng(24))
    start = int(crop[0, 0])
    assert crop.shape == (2, 30)
    assert np.array_equal(crop[0], position[start : start + 30])
    assert np.array_equal(crop_labels, labels[start : start + 30])

    padded, padded_labels = random_crop(position[:10], labels[:10], 16, _rng(25))
    assert padded.shape == padded_labels.shape == (16,)
    assert np.sum(padded_labels == IGNORE) == 6
    assert np.array_equal(padded[padded_labels != IGNORE], position[:10])


@pytest.mark.parametrize(
    "changes",
    [
        {"rate_bpm": (20.0, 10.0)},
        {"depth_mm": (0.0, 1.0)},
        {"inhale_fraction": (0.3, 1.0)},
        {"sigh_prob": 1.5},
        {"heart_mm": (-0.1, 0.2)},
    ],
)
def test_config_rejects_impossible_ranges(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="SynthConfig"):
        SynthConfig(**changes)
