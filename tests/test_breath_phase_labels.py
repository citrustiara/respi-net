import json

import numpy as np
import pandas as pd
import pytest
from scipy.signal import detrend

from respi_net.breath_phases import (
    EXHALE,
    HOLD_AFTER_EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    NOISE,
    classes_for_cue_kinds,
    export_code,
    export_codes,
    from_export_code,
    mask_around,
    transitions_allowed,
)
from respi_net.chest_signal import (
    A121_CHEST_SIGN,
    PHASE_TO_MM,
    a121_chest_signal,
    light_filter,
    rate_band_filter,
)
from respi_net.label_alignment import align_cues
from respi_net.phase_metrics import (
    boundary_errors,
    breath_rate_bpm,
    enforce_min_duration,
    score_phases,
    viterbi_decode,
)

FS = 20.0


def _coach_cues() -> list[dict[str, object]]:
    """The 90 s coach protocol: 10 s free, 7 x (2 s in, 3 s out), 15 s hold, 6 x (2 s, 3 s)."""
    cues: list[dict[str, object]] = []
    cursor = 0.0

    def add(kind: str, seconds: float) -> None:
        nonlocal cursor
        cues.append({"start_s": cursor, "end_s": cursor + seconds, "kind": kind})
        cursor += seconds

    add("normal", 10.0)
    for _ in range(7):
        add("inhale", 2.0)
        add("exhale", 3.0)
    add("hold", 15.0)
    for _ in range(6):
        add("inhale", 2.0)
        add("exhale", 3.0)
    return cues


def _breathing_from_boundaries(time_s: np.ndarray, cues: list[dict[str, object]], true_times: list[float]) -> np.ndarray:
    """A chest trace that really turns at ``true_times`` (one per cue after the first)."""
    chest = np.zeros(len(time_s))
    starts = [0.0, *true_times]
    ends = [*true_times, time_s[-1] + 1.0]
    for cue, start, end in zip(cues, starts, ends):
        inside = (time_s >= start) & (time_s < end)
        u = np.clip((time_s[inside] - start) / (end - start), 0.0, 1.0)
        smooth = 0.5 - 0.5 * np.cos(np.pi * u)
        if cue["kind"] == "inhale":
            chest[inside] = smooth
        elif cue["kind"] == "exhale":
            chest[inside] = 1.0 - smooth
        elif cue["kind"] == "normal":
            chest[inside] = 0.3 + 0.3 * np.sin(2 * np.pi * time_s[inside] / 4.0)
    return chest


def test_classes_follow_the_supervisors_codes() -> None:
    assert [export_code(label) for label in (EXHALE, HOLD_AFTER_EXHALE, INHALE, HOLD_AFTER_INHALE, NOISE)] == [0, 1, 2, 3, 999]
    assert all(from_export_code(export_code(label)) == label for label in range(5))
    assert export_codes(np.array([NOISE, IGNORE, INHALE])).tolist() == [999, -1, 2]


def test_holds_take_the_class_of_the_breath_before_them() -> None:
    labels = classes_for_cue_kinds(["normal", "inhale", "hold", "exhale", "hold", "settle", "hold"])
    assert labels == [IGNORE, INHALE, HOLD_AFTER_INHALE, EXHALE, HOLD_AFTER_EXHALE, IGNORE, HOLD_AFTER_EXHALE]


def test_transition_rules_and_masking() -> None:
    good = np.array([INHALE] * 3 + [HOLD_AFTER_INHALE] * 3 + [EXHALE] * 3 + [IGNORE] * 2 + [INHALE] * 2)
    bad = np.array([EXHALE] * 3 + [HOLD_AFTER_INHALE] * 3)
    assert transitions_allowed(good)
    assert not transitions_allowed(bad)
    time_s = np.arange(10) / 10.0
    masked = mask_around(np.zeros(10, dtype=np.int8), time_s, [0.5], 0.1)
    assert masked.tolist() == [0, 0, 0, 0, -1, -1, -1, 0, 0, 0]


def test_scores_boundaries_rate_and_decoding() -> None:
    time_s = np.arange(0, 60, 1 / FS)
    reference = np.where((time_s % 5) < 2, INHALE, EXHALE).astype(np.int8)
    late = np.roll(reference, 4)
    scores = score_phases(reference, reference)
    assert scores.accuracy == 1.0 and scores.macro_f1 == 1.0
    boundaries = boundary_errors(reference, late, time_s)
    assert boundaries.detection_rate == 1.0
    assert boundaries.median_abs_error_s == pytest.approx(0.2)
    assert boundaries.median_delay_s == pytest.approx(0.2)
    assert boundary_errors(late, reference, time_s).median_delay_s == pytest.approx(-0.2)
    assert breath_rate_bpm(reference, time_s) == pytest.approx(12.0)

    flicker = reference.copy()
    flicker[::9] = EXHALE
    log_probs = np.full((len(time_s), 5), np.log(0.02))
    log_probs[np.arange(len(time_s)), flicker] = np.log(0.9)
    assert score_phases(reference, viterbi_decode(log_probs)).accuracy > 0.99
    assert score_phases(reference, enforce_min_duration(flicker, FS, 0.3)).accuracy > 0.95


def test_light_filter_keeps_a_hold_flat_where_the_rate_band_pass_eats_it() -> None:
    time_s = np.arange(0, 90, 1 / FS)
    # Breathing, then a 15 s hold at the empty-lungs level, then breathing again.
    chest = np.where((time_s >= 40) & (time_s < 55), -1.0, np.sin(2 * np.pi * 0.2 * time_s))
    hold = (time_s >= 41) & (time_s < 54.5)
    light = light_filter(chest, FS)
    band = rate_band_filter(chest, FS)
    # Low-pass only: the hold stays flat and at the level it was held at.
    assert np.ptp(light[hold]) < 0.02
    assert np.mean(light[hold]) == pytest.approx(-1.0, abs=0.02)
    # The rate band-pass moves the hold to the middle of the breath and adds a
    # slow dip before it ends, so it looks like breathing instead of a hold.
    assert abs(np.mean(band[hold]) - (-1.0)) > 0.7
    assert np.ptp(band[hold]) > 0.4
    causal = light_filter(chest, FS, causal=True)
    assert abs(causal[0] - chest[0]) < 1e-6


def test_a121_chest_signal_rises_on_inhale() -> None:
    time_s = np.arange(0, 20, 1 / FS)
    chest_mm = 3.0 * np.sin(2 * np.pi * 0.25 * time_s)
    phase = A121_CHEST_SIGN * chest_mm / PHASE_TO_MM + 1.0
    rows = []
    for index, value in enumerate(phase):
        profile = np.array([0.1, 1.0, 5.0, 1.0, 0.1]) * np.exp(1j * value)
        rows.append(
            {
                "Timestamp_ms": 1_000_000.0 + 1000.0 * time_s[index],
                "Distances_m": json.dumps([0.4, 0.45, 0.5, 0.55, 0.6]),
                "Real": json.dumps(profile.real.tolist()),
                "Imag": json.dumps(profile.imag.tolist()),
                "AcconeerRangeStartIndex": 1,
                "AcconeerRangeEndIndex": 4,
            }
        )
    signal = a121_chest_signal(pd.DataFrame(rows))
    assert signal.fs == pytest.approx(FS)
    assert signal.meta["gate_source"] == "acconeer-range"
    expected = detrend(chest_mm, type="linear")
    assert np.corrcoef(signal.chest, expected)[0, 1] > 0.999
    assert np.ptp(signal.chest) == pytest.approx(np.ptp(expected), rel=0.02)


def test_alignment_recovers_late_and_uneven_breaths() -> None:
    rng = np.random.default_rng(4)
    cues = _coach_cues()
    time_s = np.arange(0, 92, 1 / FS)
    cue_boundaries = [float(cue["start_s"]) for cue in cues[1:]]
    true_times = [moment + 0.4 + rng.normal(0.0, 0.12) for moment in cue_boundaries]
    chest = 8.0 * _breathing_from_boundaries(time_s, cues, true_times)
    chest += 0.03 * rng.standard_normal(len(time_s)) + 0.02 * time_s
    result = align_cues(time_s, light_filter(chest, FS), cues, fs=FS)

    assert result.lag_s == pytest.approx(0.4, abs=0.1)
    assert result.sign == 1.0
    assert transitions_allowed(result.labels)
    snapped = {round(b.cue_s, 3): b for b in result.boundaries if b.method in ("turn", "stop", "start")}
    errors = [abs(snapped[round(cue, 3)].time_s - true) for cue, true in zip(cue_boundaries, true_times) if round(cue, 3) in snapped]
    assert len(errors) >= 0.85 * len(cue_boundaries)
    assert np.median(errors) < 0.08
    kinds = {b.method for b in result.boundaries}
    assert {"turn", "stop", "start"} <= kinds
    hold = (time_s > true_times[14] + 1.0) & (time_s < true_times[15] - 1.0)
    assert np.all(result.labels[hold] == HOLD_AFTER_EXHALE)
    assert np.all(result.labels[time_s < 9.0] == IGNORE)


def test_alignment_flips_an_upside_down_reference() -> None:
    cues = _coach_cues()
    time_s = np.arange(0, 92, 1 / FS)
    true_times = [float(cue["start_s"]) + 0.3 for cue in cues[1:]]
    chest = -5.0 * _breathing_from_boundaries(time_s, cues, true_times)
    result = align_cues(time_s, chest, cues, fs=FS)
    assert result.sign == -1.0
    assert result.lag_s == pytest.approx(0.3, abs=0.1)


def test_a_breath_finished_after_the_hold_is_labelled_as_that_breath() -> None:
    # Exhale stopped halfway, 15 s hold, the rest of the exhale, then inhale:
    # what people often do when a hold cue catches them mid-breath.
    cues = _coach_cues()
    time_s = np.arange(0, 92, 1 / FS)
    lag = 0.4
    chest = 8.0 * _breathing_from_boundaries(time_s, cues, [float(cue["start_s"]) + lag for cue in cues[1:]])
    hold_start, hold_end, turn = 45.0 + lag, 60.0 + lag, 61.0 + lag
    last_exhale = (time_s >= 42.0 + lag) & (time_s < hold_start)
    u = (time_s[last_exhale] - (42.0 + lag)) / 3.0
    chest[last_exhale] = 8.0 * (1.0 - 0.6 * (0.5 - 0.5 * np.cos(np.pi * u)))
    chest[(time_s >= hold_start) & (time_s < hold_end)] = 8.0 * 0.4
    rest = (time_s >= hold_end) & (time_s < turn)
    chest[rest] = 8.0 * 0.4 * (0.5 + 0.5 * np.cos(np.pi * (time_s[rest] - hold_end) / (turn - hold_end)))
    inhale = (time_s >= turn) & (time_s < 62.0 + lag)
    chest[inhale] = 8.0 * (0.5 - 0.5 * np.cos(np.pi * (time_s[inhale] - turn) / (62.0 + lag - turn)))

    result = align_cues(time_s, chest, cues, fs=FS)

    kinds = [(b.before, b.after, b.method) for b in result.boundaries]
    assert (HOLD_AFTER_EXHALE, EXHALE, "start") in kinds
    assert transitions_allowed(result.labels)
    assert result.labels[np.argmin(np.abs(time_s - (hold_end + 0.3)))] == EXHALE
    assert result.labels[np.argmin(np.abs(time_s - (turn + 0.5)))] == INHALE
    assert result.labels[np.argmin(np.abs(time_s - 52.0))] == HOLD_AFTER_EXHALE
