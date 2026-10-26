"""Choosing the A121 chest bins by how well they breathe (respi_net.a121_range)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from respi_net import a121_range
from respi_net.a121_range import LiveRangeSelector, motion_snr, polarity, score_displacement, score_range_bins
from respi_net.chest_signal import PHASE_TO_MM, a121_chest_signal

FS = 20.0
WATCH, CHEST = (2, 3, 4), (8, 9, 10)
WATCH_FIXTURE = Path(__file__).parent / "data" / "a121_watch_side_90s.npz"


def _band_noise(rng: np.random.Generator, n: int, rms: float, band: tuple[float, float] = (1.5, 4.0)) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    x = sosfiltfilt(butter(2, band, btype="bandpass", fs=FS, output="sos"), rng.standard_normal(n))
    return rms * x / np.std(x)


def _scene(duration_s: float = 90.0, *, seed: int = 0, watch_sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A side view: a watch (strong echo, small breath, the pulse, wrist tremor), the chest (weaker echo, the
    whole breath, body micro-motion) and empty bins of receiver noise.  Displacements in mm, chest rising on inhale."""

    rng = np.random.default_rng(seed)
    time_s = np.arange(0.0, duration_s, 1.0 / FS)
    n = len(time_s)
    breath = np.sin(2 * np.pi * 0.21 * time_s)
    pulse = np.sin(2 * np.pi * 1.15 * time_s) + 0.4 * np.sin(2 * np.pi * 2.3 * time_s)
    amplitude = np.zeros(12)
    displacement = np.zeros((n, 12))
    amplitude[list(WATCH)] = [600.0, 1000.0, 600.0]
    displacement[:, list(WATCH)] = (watch_sign * 0.15 * breath + 0.04 * pulse + _band_noise(rng, n, 0.05))[:, None]
    amplitude[list(CHEST)] = [200.0, 250.0, 200.0]
    displacement[:, list(CHEST)] = (0.8 * breath + 0.01 * pulse + _band_noise(rng, n, 0.06))[:, None]
    noise = 3.0 * (rng.standard_normal((n, 12)) + 1j * rng.standard_normal((n, 12)))
    # The recorded phase falls on inhale: exp(-j * displacement / PHASE_TO_MM).
    profile = amplitude * np.exp(-1j * displacement / PHASE_TO_MM) + noise
    return time_s, profile


def _frame(time_s: np.ndarray, profile: np.ndarray, distances: np.ndarray, gate: tuple[int, int] | None) -> pd.DataFrame:
    distances_json = json.dumps([round(float(value), 6) for value in distances])
    frame = pd.DataFrame(
        {
            "Timestamp_ms": 1_000_000.0 + 1000.0 * time_s,
            "Distances_m": distances_json,
            "Real": [json.dumps(row.real.round(4).tolist()) for row in profile],
            "Imag": [json.dumps(row.imag.round(4).tolist()) for row in profile],
        }
    )
    if gate is not None:
        frame["AcconeerRangeStartIndex"], frame["AcconeerRangeEndIndex"] = gate
    return frame


def test_a_strong_watch_loses_to_the_weaker_breathing_chest() -> None:
    time_s, profile = _scene()
    distances = 0.40 + 0.03 * np.arange(profile.shape[1])
    scores = score_range_bins(profile, FS)
    assert int(np.argmax(scores.score)) in CHEST
    assert scores.score[list(CHEST)].min() > 2.0 * scores.score[list(WATCH)].max()
    # Empty bins: the unwrapped phase of noise wanders like slow breathing, but the raw IQ does not move.
    assert not scores.moving[[0, 6, 11]].any() and np.all(scores.score[[0, 6, 11]] == 0)

    # Acconeer chose the watch (its range covers the strongest echo); the breathing score moves to the chest.
    frame = _frame(time_s, profile, distances, gate=(WATCH[0], WATCH[-1] + 1))
    signal = a121_chest_signal(frame)
    assert signal.meta["gate_source"] == "breathing-score"
    assert signal.meta["old_gate_source"] == "acconeer-range"
    assert distances[CHEST[0]] <= signal.meta["gate_start_m"] <= signal.meta["gate_end_m"] <= distances[CHEST[-1]]
    assert signal.meta["old_gate_score"] < 0.5 * signal.meta["best_group_score"]
    assert np.corrcoef(signal.chest, np.sin(2 * np.pi * 0.21 * time_s))[0, 1] > 0.95  # rises on inhale
    old = a121_chest_signal(frame, range_selection="acconeer")
    assert old.meta["gate_source"] == "acconeer-range" and "old_gate_score" not in old.meta
    assert old.meta["gate_start_m"] == pytest.approx(distances[WATCH[0]], abs=1e-6)


def test_a_group_breathing_against_the_old_gate_does_not_replace_it() -> None:
    """Which of two reflections moving in antiphase shows the chest's direction, a score cannot say."""

    time_s, profile = _scene(watch_sign=-1.0)
    distances = 0.40 + 0.03 * np.arange(profile.shape[1])
    signal = a121_chest_signal(_frame(time_s, profile, distances, gate=(WATCH[0], WATCH[-1] + 1)))
    assert signal.meta["gate_source"] == "acconeer-range"
    assert "against" in signal.meta["range_note"]


def test_the_heartbeat_is_not_breathing() -> None:
    """A clean pulse (about 1 Hz, a tenth of a breath) beats a noisy breathing chest on clarity, but it is a fast rhythm
    far smaller than the typical breath -- the rule of the detector's fast-breathing test -- so it does not count."""

    rng = np.random.default_rng(3)
    time_s = np.arange(0.0, 90.0, 1.0 / FS)
    pulse = 0.08 * np.sin(2 * np.pi * 1.1 * time_s) + 0.002 * rng.standard_normal(len(time_s))
    chest = 1.0 * np.sin(2 * np.pi * 0.22 * time_s) + _band_noise(rng, len(time_s), 0.15)
    displacement = np.column_stack([pulse, chest])
    scores = score_displacement(displacement, FS)
    assert scores.score[0] == 0.0 and scores.score[1] > 3.0
    assert scores.rate_hz[1] == pytest.approx(0.22, abs=0.05)
    # Without the size rule every fast rhythm would count, and the pulse would win.
    unguarded = score_displacement(displacement, FS, typical_mm=0.0)
    assert unguarded.score[0] > unguarded.score[1]
    # Fast breathing of a breath's size still counts.
    fast = score_displacement(np.column_stack([pulse, 0.8 * np.sin(2 * np.pi * 1.0 * time_s), chest]), FS)
    assert fast.score[1] > 0 and fast.rate_hz[1] == pytest.approx(1.0, abs=0.05)


def test_motion_snr_is_amplitude_free_and_low_for_noise() -> None:
    _, profile = _scene(seed=5)
    snr = motion_snr(profile, FS)
    assert snr[list(CHEST)].min() > 100 and snr[[0, 6, 11]].max() < 3
    assert np.allclose(motion_snr(10 * profile, FS), snr)


def _feed(selector: LiveRangeSelector, time_s: np.ndarray, profile: np.ndarray, prior: int) -> tuple[list[int], list[bool]]:
    phase = np.unwrap(np.angle(profile), axis=0)
    chosen, frozen = [], []
    for index, now in enumerate(time_s):
        chosen.append(selector.update(float(now), profile[index], phase[index], prior))
        frozen.append(selector.frozen)
    return chosen, frozen


def test_live_selection_moves_from_the_watch_only_after_it_is_clearly_worse_for_a_while() -> None:
    time_s, profile = _scene(duration_s=60.0, seed=1)
    chosen, _ = _feed(LiveRangeSelector(FS, profile.shape[1]), time_s, profile, prior=WATCH[1])
    chosen = np.asarray(chosen)
    # First scored after 20 s, then clearly better at every update for 10 s.
    assert np.all(chosen[time_s < 30.0] == WATCH[1])
    assert chosen[-1] in CHEST
    switch_s = time_s[np.flatnonzero(chosen != WATCH[1])[0]]
    assert 30.0 <= switch_s <= 40.0
    assert np.all(chosen[time_s >= switch_s] == chosen[-1])  # and it stays


def _hold_scene(seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Breathing, a 40 s hold (the chest still but for the heart), breathing.  In the hold an arm swings
    rhythmically in bin 5, and a strong pulse sits in bin 2 all along: both look like breathing."""

    rng = np.random.default_rng(seed)
    time_s = np.arange(0.0, 150.0, 1.0 / FS)
    n = len(time_s)
    hold = (time_s >= 60.0) & (time_s < 100.0)
    breath = np.where(hold, 0.0, np.sin(2 * np.pi * 0.2 * time_s))
    amplitude, displacement = np.zeros(12), np.zeros((n, 12))
    amplitude[list(CHEST)] = [200.0, 250.0, 200.0]
    displacement[:, list(CHEST)] = (1.0 * breath + 0.03 * np.sin(2 * np.pi * 1.1 * time_s) + _band_noise(rng, n, 0.05))[:, None]
    amplitude[2] = 2000.0
    displacement[:, 2] = 0.08 * np.sin(2 * np.pi * 1.1 * time_s)
    amplitude[5] = 800.0
    displacement[:, 5] = np.where(hold, 2.0 * np.sin(2 * np.pi * 0.3 * time_s), 0.0)
    noise = 3.0 * (rng.standard_normal((n, 12)) + 1j * rng.standard_normal((n, 12)))
    return time_s, amplitude * np.exp(-1j * displacement / PHASE_TO_MM) + noise, hold


def test_a_hold_freezes_the_live_selection() -> None:
    time_s, profile, hold = _hold_scene()
    chosen, frozen = _feed(LiveRangeSelector(FS, profile.shape[1]), time_s, profile, prior=CHEST[1])
    chosen, frozen = np.asarray(chosen), np.asarray(frozen)
    assert np.all(chosen == CHEST[1])  # neither the arm nor the pulse takes over
    settled = hold & (time_s >= 66.0)  # the freeze needs the last 4 s still and an update
    assert frozen[settled].mean() > 0.9
    assert not frozen[(time_s >= 110.0)].any()  # breathing again
    # The freeze is what keeps the chest: without it the swinging arm wins during the hold.
    unfrozen, _ = _feed(LiveRangeSelector(FS, profile.shape[1], freeze=False), time_s, profile, prior=CHEST[1])
    assert 5 in unfrozen


@pytest.mark.skipif(not WATCH_FIXTURE.exists(), reason="watch recording fixture missing")
def test_real_recording_with_a_watch_in_the_beam() -> None:
    """90 s sitting, radar from the side at about 0.8 m: Acconeer's range is the wristwatch at 0.49 m."""

    with np.load(WATCH_FIXTURE) as data:
        recording = {key: data[key] for key in data.files}
    profile = recording["real"].astype(float) + 1j * recording["imag"].astype(float)
    frame = _frame(recording["timestamp_ms"] / 1000.0, profile, recording["distances_m"].astype(float), gate=None)
    frame["AcconeerRangeStartIndex"] = recording["range_start_index"]
    frame["AcconeerRangeEndIndex"] = recording["range_end_index"]
    old = a121_chest_signal(frame, range_selection="acconeer")
    new = a121_chest_signal(frame)
    assert (old.meta["gate_start_m"], old.meta["gate_end_m"]) == pytest.approx((0.485, 0.515), abs=0.002)
    assert new.meta["gate_source"] == "breathing-score"
    assert 0.63 <= new.meta["gate_start_m"] <= new.meta["gate_end_m"] <= 0.74
    assert new.meta["old_gate_score"] < 0.3 * new.meta["best_group_score"]


# Polarity: which way up a trace breathes, from the shape of the breath (no other sensor).

BELT_FIXTURE = Path(__file__).parent / "data" / "a121_belt_side_50cm_90s.npz"


def _breath(time_s: np.ndarray, rate_hz: float = 0.22, inhale_share: float = 0.38) -> np.ndarray:
    """A breath rising on inhale: a quick active inhale (``inhale_share`` of the cycle) and a passive exhale that
    decays, 0 (empty) to 1 (full)."""

    phase = (time_s * rate_hz) % 1.0
    exhale = (phase - inhale_share) / (1.0 - inhale_share)
    return np.where(phase < inhale_share, 0.5 - 0.5 * np.cos(np.pi * phase / inhale_share), (np.exp(-4.0 * exhale) - np.exp(-4.0)) / (1.0 - np.exp(-4.0)))


def test_polarity_comes_from_the_shape_of_the_breath() -> None:
    rng = np.random.default_rng(7)
    time_s = np.arange(0.0, 120.0, 1.0 / FS)
    breath = _breath(time_s)
    noise = _band_noise(rng, len(time_s), 0.05) + 0.02 * rng.standard_normal(len(time_s))
    symmetric = 0.5 - 0.5 * np.cos(2 * np.pi * 0.22 * time_s)
    chest = np.column_stack([breath + noise, -breath + noise, symmetric + noise])  # rising on inhale: chest_sign +1
    scores = score_displacement(chest, FS, chest_sign=1.0)
    assert list(scores.polarity()) == [1, -1, 0]
    # Rising for 38% of the cycle; the slow end of the exhale is mostly under the dead band, so the share is nearer 0.5.
    assert scores.rise_share[0] < 0.46 and scores.rise_below[0] == 1.0
    assert scores.rise_share[1] > 0.54 and scores.rise_below[1] == 0.0
    # The recorded phase falls on inhale, which is what the default chest sign expects.
    assert list(score_displacement(-chest, FS).polarity()) == [1, -1, 0]
    # Too little breathing to tell.
    assert list(score_displacement(chest[: int(30 * FS)], FS, chest_sign=1.0).polarity()) == [0, 0, 0]
    assert list(polarity(np.array([0.46, 0.46, 0.54]), np.array([0.8, 0.6, 0.1]), np.array([10, 10, 3]))) == [1, 0, 0]


def _reflector_scene(*, reflector_sign: float, reflector_breath: float, reflector_noise: float, chest_noise: float,
                     seed: int = 4, duration_s: float = 120.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A strong reflector (bins 2-4) moving ``reflector_sign`` with the breath and the chest (bins 8-10) moving with it,
    the breath asymmetric as people breathe; the recorded phase falls on inhale."""

    rng = np.random.default_rng(seed)
    time_s = np.arange(0.0, duration_s, 1.0 / FS)
    n = len(time_s)
    breath = _breath(time_s)
    amplitude, displacement = np.zeros(12), np.zeros((n, 12))
    amplitude[list(WATCH)] = [600.0, 1000.0, 600.0]
    displacement[:, list(WATCH)] = (reflector_sign * reflector_breath * breath + _band_noise(rng, n, reflector_noise))[:, None]
    amplitude[list(CHEST)] = [200.0, 250.0, 200.0]
    displacement[:, list(CHEST)] = (0.9 * breath + _band_noise(rng, n, chest_noise))[:, None]
    noise = 3.0 * (rng.standard_normal((n, 12)) + 1j * rng.standard_normal((n, 12)))
    return time_s, amplitude * np.exp(-1j * displacement / PHASE_TO_MM) + noise, breath


def test_an_upside_down_strong_reflector_gives_way_to_the_chest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 50 cm side view: Acconeer's range is a strong echo moving against the chest; the chest group moves the
    other way and its breath is right way up, so the old gate cannot be kept."""

    time_s, profile, breath = _reflector_scene(reflector_sign=-1.0, reflector_breath=0.3, reflector_noise=0.08, chest_noise=0.05)
    distances = 0.40 + 0.03 * np.arange(profile.shape[1])
    frame = _frame(time_s, profile, distances, gate=(WATCH[0], WATCH[-1] + 1))
    signal = a121_chest_signal(frame)
    assert signal.meta["gate_source"] == "breathing-score"
    assert distances[CHEST[0]] <= signal.meta["gate_start_m"] <= signal.meta["gate_end_m"] <= distances[CHEST[-1]]
    assert signal.meta["best_group_in_phase"] is False
    assert signal.meta["best_group_polarity"] == 1 and signal.meta["old_gate_polarity"] <= 0
    assert "right way up" in signal.meta["range_note"]
    assert np.corrcoef(signal.chest, breath)[0, 1] > 0.95
    old = a121_chest_signal(frame, range_selection="acconeer")
    assert np.corrcoef(old.chest, breath)[0, 1] < -0.5  # upside down
    # Before the polarity, a group moving against the old gate never replaced it.
    monkeypatch.setattr(a121_range, "POLARITY_MARGIN", np.inf)
    assert a121_chest_signal(frame).meta["gate_source"] == "acconeer-range"


def test_an_upside_down_reflection_that_breathes_best_does_not_replace_the_chest() -> None:
    """The 2 m retest: the best-scoring bins are a reflection moving against the chest that Acconeer chose."""

    time_s, profile, breath = _reflector_scene(reflector_sign=-1.0, reflector_breath=1.2, reflector_noise=0.02, chest_noise=0.25)
    distances = 1.60 + 0.02 * np.arange(profile.shape[1])
    signal = a121_chest_signal(_frame(time_s, profile, distances, gate=(CHEST[0], CHEST[-1] + 1)))
    assert signal.meta["best_group_m"][0] <= distances[WATCH[-1]]  # the reflection breathes best ...
    assert signal.meta["best_group_score"] > signal.meta["old_gate_score"]
    assert signal.meta["best_group_polarity"] == -1  # ... but upside down
    assert signal.meta["gate_source"] == "acconeer-range"
    assert "not right way up" in signal.meta["range_note"]
    assert np.corrcoef(signal.chest, breath)[0, 1] > 0.9


def test_a_group_upside_down_is_not_chosen_even_moving_with_the_old_gate() -> None:
    time_s, profile, _ = _reflector_scene(reflector_sign=-1.0, reflector_breath=0.15, reflector_noise=0.08, chest_noise=0.05)
    # The chest bins turned upside down too (recorded phase rising on inhale): everything breathes the wrong way.
    profile[:, list(CHEST)] = np.conj(profile[:, list(CHEST)])
    distances = 0.40 + 0.03 * np.arange(profile.shape[1])
    signal = a121_chest_signal(_frame(time_s, profile, distances, gate=(WATCH[0], WATCH[-1] + 1)))
    assert signal.meta["best_group_in_phase"] is True and signal.meta["best_group_polarity"] == -1
    assert signal.meta["old_gate_score"] < 0.5 * signal.meta["best_group_score"]
    assert signal.meta["gate_source"] == "acconeer-range"
    assert signal.meta["range_note"] == "best group is upside down; old gate kept"


def test_live_selection_leaves_an_upside_down_bin_once_the_polarity_is_known(monkeypatch: pytest.MonkeyPatch) -> None:
    time_s, profile, _ = _reflector_scene(reflector_sign=-1.0, reflector_breath=0.3, reflector_noise=0.08, chest_noise=0.05, duration_s=150.0)
    selector = LiveRangeSelector(FS, profile.shape[1])
    chosen = np.asarray(_feed(selector, time_s, profile, prior=WATCH[1])[0])
    assert chosen[-1] in CHEST
    # It waits for LIVE_POLARITY_MIN_S of breathing (scored from 20 s on), then LIVE_SWITCH_AFTER_S.
    switch_s = time_s[np.flatnonzero(chosen != WATCH[1])[0]]
    assert 80.0 <= switch_s <= 100.0
    assert np.all(chosen[time_s >= switch_s] == chosen[-1])
    assert selector.polarity()[chosen[-1]] == 1 and selector.polarity()[WATCH[1]] <= 0
    monkeypatch.setattr(a121_range, "POLARITY_MARGIN", np.inf)
    assert np.all(np.asarray(_feed(LiveRangeSelector(FS, profile.shape[1]), time_s, profile, prior=WATCH[1])[0]) == WATCH[1])


def test_live_polarity_does_not_change_in_a_hold() -> None:
    """Breathing for 100 s, a 40 s hold in which an arm swings in bin 5 (rising quickly, falling slowly: upside down
    by shape), breathing again.  The chest's polarity is settled before the hold and does not move in it."""

    rng = np.random.default_rng(11)
    time_s = np.arange(0.0, 200.0, 1.0 / FS)
    n = len(time_s)
    hold = (time_s >= 100.0) & (time_s < 140.0)
    breath = np.where(hold, 1.0, _breath(time_s))
    amplitude, displacement = np.zeros(12), np.zeros((n, 12))
    amplitude[list(CHEST)] = [200.0, 250.0, 200.0]
    displacement[:, list(CHEST)] = (2.0 * breath + 0.03 * np.sin(2 * np.pi * 1.1 * time_s) + _band_noise(rng, n, 0.05))[:, None]
    amplitude[5] = 800.0
    displacement[:, 5] = np.where(hold, -2.0 * _breath(time_s, rate_hz=0.3), 0.0)
    noise = 3.0 * (rng.standard_normal((n, 12)) + 1j * rng.standard_normal((n, 12)))
    profile = amplitude * np.exp(-1j * displacement / PHASE_TO_MM) + noise
    selector = LiveRangeSelector(FS, profile.shape[1])
    phase = np.unwrap(np.angle(profile), axis=0)
    signs, seconds, frozen, chosen = [], [], [], []
    for index, now in enumerate(time_s):
        chosen.append(selector.update(float(now), profile[index], phase[index], CHEST[1]))
        signs.append(int(selector.polarity()[CHEST[1]]))
        seconds.append(float(selector.polarity_s[CHEST[1]]))
        frozen.append(selector.frozen)
    signs, seconds, frozen = np.asarray(signs), np.asarray(seconds), np.asarray(frozen)
    assert np.all(np.asarray(chosen) == CHEST[1])
    decided = np.flatnonzero(signs != 0)[0]
    assert time_s[decided] < 100.0 and np.all(signs[decided:] == 1)  # settled before the hold, never turned
    settled = hold & (time_s >= 106.0)
    assert frozen[settled].mean() > 0.9
    assert np.ptp(seconds[settled & frozen]) == 0.0  # nothing gathered while frozen


@pytest.mark.skipif(not BELT_FIXTURE.exists(), reason="50 cm belt recording fixture missing")
def test_real_recording_from_the_side_at_50_cm_comes_out_right_way_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """90 s sitting, radar about 50 cm away very much from the side, with the strain belt: Acconeer's range
    (0.395-0.425 m) moves against the belt; the bins at 0.23-0.34 m breathe with it."""

    with np.load(BELT_FIXTURE) as data:
        recording = {key: data[key] for key in data.files}
    profile = (recording["real_x8"].astype(float) + 1j * recording["imag_x8"].astype(float)) / 8.0
    frame = _frame(recording["timestamp_ms"] / 1000.0, profile, recording["distances_m"].astype(float), gate=None)
    frame["AcconeerRangeStartIndex"] = recording["range_start_index"]
    frame["AcconeerRangeEndIndex"] = recording["range_end_index"]

    def belt_correlation(signal) -> float:
        from scipy.signal import butter, sosfiltfilt

        belt = np.interp(signal.time_s, recording["belt_time_s"], recording["belt_chest"])
        sos = butter(2, (0.05, 0.7), "bandpass", fs=signal.fs, output="sos")
        keep = signal.time_s >= 3.0
        return float(np.corrcoef(sosfiltfilt(sos, signal.chest)[keep], sosfiltfilt(sos, belt)[keep])[0, 1])

    old = a121_chest_signal(frame, range_selection="acconeer")
    new = a121_chest_signal(frame)
    assert (old.meta["gate_start_m"], old.meta["gate_end_m"]) == pytest.approx((0.395, 0.425), abs=0.002)
    assert belt_correlation(old) < -0.3
    assert new.meta["gate_source"] == "breathing-score" and new.meta["best_group_in_phase"] is False
    assert 0.23 <= new.meta["gate_start_m"] <= new.meta["gate_end_m"] <= 0.34
    assert new.meta["best_group_polarity"] == 1 and new.meta["best_group_rise_share"] < 0.45
    assert new.meta["old_gate_polarity"] <= 0
    assert belt_correlation(new) > 0.9
    monkeypatch.setattr(a121_range, "POLARITY_MARGIN", np.inf)
    assert a121_chest_signal(frame).meta["gate_source"] == "acconeer-range"  # the rule before the polarity kept it
