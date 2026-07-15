from pathlib import Path

import pytest

from respi_net.imu_guided_protocol import (
    build_trials,
    counter_delta,
    cue_at,
    delete_trial_outputs,
    output_paths,
    sample_timing_summary,
)


def test_guided_imu_protocol_contains_two_60s_and_two_90s_trials() -> None:
    trials = build_trials()

    assert [trial.duration_s for trial in trials] == [60.0, 60.0, 90.0, 90.0]
    assert trials[0].cues[-1].end_s == 60.0
    assert trials[2].cues[-1].end_s == 90.0
    assert cue_at(trials[0].cues, 0)[0].kind == "settle"
    assert cue_at(trials[0].cues, 10)[0].kind == "normal"
    assert cue_at(trials[0].cues, 45)[0].kind == "hold"
    assert cue_at(trials[2].cues, 10)[0].kind == "inhale"
    assert cue_at(trials[2].cues, 45)[0].kind == "hold"


def test_phone_timing_summary_estimates_gap_from_device_time_axis() -> None:
    summary = sample_timing_summary(
        [[0.0, 0, 0, 1, 0, 0, 0], [10.0, 0, 0, 1, 0, 0, 0], [20.0, 0, 0, 1, 0, 0, 0], [40.0, 0, 0, 1, 0, 0, 0]]
    )

    assert summary["sample_rate_hz"] == pytest.approx(100.0)
    assert summary["estimated_missing_samples"] == 1


def test_discard_deletes_all_current_trial_outputs(tmp_path: Path) -> None:
    trial = build_trials()[0]
    paths = output_paths(tmp_path, trial)
    for path in paths.values():
        path.write_text("partial", encoding="utf-8")
    delete_trial_outputs(paths)

    assert not any(path.exists() for path in paths.values())


def test_ble_counter_delta_is_relative_to_measurement_start() -> None:
    delta = counter_delta(
        {"received_batches": 12, "missing_batches": 3, "invalid_batches": 1},
        {"received_batches": 5, "missing_batches": 2, "invalid_batches": 1},
    )

    assert delta == {"received_batches": 7, "missing_batches": 1, "invalid_batches": 0}
