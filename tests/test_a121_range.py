"""Choosing the A121 chest bins by how well they breathe (respi_net.a121_range)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from respi_net.a121_range import LiveRangeSelector, motion_snr, score_displacement, score_range_bins
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
