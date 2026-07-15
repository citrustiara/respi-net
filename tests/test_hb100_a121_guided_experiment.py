from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from hb100_a121_guided_experiment import (  # noqa: E402
    HB100Transport,
    breathing_cue_at,
    build_breathing_cues,
    build_initial_steps,
    make_extension_step,
    parse_hb100_payload,
    summarize_hb100_rows,
)
import hb100_a121_guided_experiment as guided  # noqa: E402


def test_short_protocol_has_two_runs_at_each_core_distance() -> None:
    steps = build_initial_steps(include_interference=True, hb100_only=False)

    assert [step.step_id for step in steps[:3]] == ["INT-A", "INT-B", "INT-C"]
    range_steps = [step for step in steps if step.kind == "range"]
    assert [step.distance_cm for step in range_steps] == [30, 30, 60, 60, 100, 100]
    assert [step.repeat_number for step in range_steps] == [1, 2, 1, 2, 1, 2]
    assert all(step.sensor_mode == "both" for step in range_steps)


def test_hb100_only_skips_interference_and_a121() -> None:
    steps = build_initial_steps(include_interference=True, hb100_only=True)

    assert len(steps) == 6
    assert all(step.sensor_mode == "hb100" for step in steps)


def test_breathing_protocol_boundaries_fill_90_seconds() -> None:
    cues = build_breathing_cues()

    assert cues[0].start_s == 0
    assert cues[-1].end_s == 90
    assert breathing_cue_at(cues, 0)[0].kind == "normal"
    assert breathing_cue_at(cues, 10)[0].kind == "inhale"
    assert breathing_cue_at(cues, 12)[0].kind == "exhale"
    assert breathing_cue_at(cues, 45)[0].kind == "hold"
    assert breathing_cue_at(cues, 60)[0].kind == "inhale"


def test_ch9102_compatibility_mapping_recovers_only_valid_csv_rows() -> None:
    # Bit 6 set: p-y map to 0-9 and l maps to comma.
    corrupted = b"q23tlqypulqwrv\r\ninvalid\r\n"

    strict, _ = parse_hb100_payload(corrupted, repair_ch9102_bit6=False)
    repaired, repaired_bytes = parse_hb100_payload(corrupted, repair_ch9102_bit6=True)

    assert strict == []
    assert repaired == [(1234, 1905, 1726)]
    assert repaired_bytes > 0


def test_probe_stops_after_preferred_strict_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    payload = b"".join(f"{1000 + index * 3},2048,1650\n".encode() for index in range(20))

    def fake_read(port: str, baud: int, *, warmup_s: float, duration_s: float) -> bytes:
        _ = port, warmup_s, duration_s
        calls.append(baud)
        return payload

    monkeypatch.setattr(guided, "_read_probe_payload", fake_read)
    transport, attempts = guided.probe_hb100_transport("TEST")

    assert transport.baud == 230400
    assert transport.repair_ch9102_bit6 is False
    assert calls == [230400]
    assert [attempt.baud for attempt in attempts] == [230400]


def test_synthetic_paced_breathing_meets_working_hb100_criterion() -> None:
    fs = 50.0
    t = np.arange(0.0, 90.0, 1.0 / fs)
    rng = np.random.default_rng(7)
    signal = rng.normal(0.0, 0.05, len(t))
    paced = ((t >= 10.0) & (t < 45.0)) | ((t >= 60.0) & (t < 90.0))
    signal[paced] += 12.0 * np.sin(2 * np.pi * 0.2 * t[paced])
    signal[t < 10.0] += 4.0 * np.sin(2 * np.pi * 0.25 * t[t < 10.0])
    voltage = 1650.0 + signal
    start_wall_ms = 1_000_000.0
    rows = [
        [index * 20.0, start_wall_ms + value * 1000.0, 2048.0, float(millivolts)]
        for index, (value, millivolts) in enumerate(zip(t, voltage))
    ]
    transport = HB100Transport("TEST", 230400, False, 100, "test")

    summary = summarize_hb100_rows(
        rows,
        start_wall_ms=start_wall_ms,
        duration_s=90.0,
        range_protocol=True,
        transport=transport,
        diagnostics={"total_bytes": 0, "repaired_bytes": 0, "malformed_lines": 0},
    )

    assert abs(summary["paced_1_peak_hz"] - 0.2) < 0.01
    assert abs(summary["paced_2_peak_hz"] - 0.2) < 0.01
    assert summary["paced_1_snr_db"] > 6
    assert summary["paced_2_snr_db"] > 6
    assert summary["hold_drop_db"] < -3
    assert summary["auto_usable"] is True


def test_failed_extension_retry_keeps_same_distance() -> None:
    steps = build_initial_steps(include_interference=False, hb100_only=False)

    first = make_extension_step(steps, 150, hb100_only=False)
    steps.append(first)
    retry = make_extension_step(steps, 150, hb100_only=False, retry_after_failure=True)

    assert first.distance_cm == retry.distance_cm == 150
    assert first.repeat_number == 1
    assert retry.repeat_number == 2
    assert retry.retry_after_failure is True
