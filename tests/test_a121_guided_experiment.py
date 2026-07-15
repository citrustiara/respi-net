from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from a121_guided_experiment import (  # noqa: E402
    MeasurementWorker,
    breathing_cue_at,
    load_config,
    make_a121_config,
    parse_breathing_protocol,
    parse_steps,
)


FOIL_2M_CONFIG = ROOT / "configs" / "a121_foil_2m_retest.json"
ORIGINAL_GUIDED_CONFIG = ROOT / "configs" / "a121_foil_lens_experiment.json"


def test_foil_2m_sequence_uses_one_foil_change_and_three_repeats() -> None:
    steps = parse_steps(load_config(FOIL_2M_CONFIG))

    assert [step.condition_id for step in steps] == [
        "NF",
        "PF",
        "PF",
        "NF",
        "NF",
        "PF",
        "P0",
        "N0",
        "N0",
        "P0",
        "P0",
        "N0",
    ]
    assert all(step.patch == "foil" for step in steps[:6])
    assert all(step.patch == "unglued" for step in steps[6:])
    assert [step.repeat_number for step in steps if step.condition_id == "N0"] == [1, 2, 3]
    assert all(step.repeat_total == 3 for step in steps)
    assert all(step.distance_cm == 200 for step in steps)
    assert all(step.lens == "hyperbolic" for step in steps)


def test_original_guided_config_still_expands_to_36_runs() -> None:
    steps = parse_steps(load_config(ORIGINAL_GUIDED_CONFIG))

    assert len(steps) == 36
    assert steps[0].condition_id == ""
    assert steps[0].repeat_number == 1
    assert steps[2].repeat_number == 3


def test_foil_2m_breathing_protocol_boundaries() -> None:
    cues = parse_breathing_protocol(load_config(FOIL_2M_CONFIG))

    assert len(cues) == 28
    assert cues[0].start_s == 0
    assert cues[-1].end_s == 90
    assert breathing_cue_at(cues, 0)[0].kind == "normal"
    assert breathing_cue_at(cues, 10)[0].kind == "inhale"
    assert breathing_cue_at(cues, 12)[0].kind == "exhale"
    assert breathing_cue_at(cues, 44.99)[0].kind == "exhale"
    assert breathing_cue_at(cues, 45)[0].kind == "hold"
    assert breathing_cue_at(cues, 59.99)[0].kind == "hold"
    assert breathing_cue_at(cues, 60)[0].kind == "inhale"
    assert breathing_cue_at(cues, 89.99)[0].kind == "exhale"


def test_breathing_protocol_must_fill_the_measurement() -> None:
    with pytest.raises(ValueError, match="lasts 5 s, but measurement_seconds is 10 s"):
        parse_breathing_protocol(
            {
                "measurement_seconds": 10,
                "breathing_protocol": [
                    {"kind": "normal", "duration_seconds": 5},
                ],
            }
        )


def test_abort_deletes_any_current_output(tmp_path: Path) -> None:
    config = load_config(FOIL_2M_CONFIG)
    output_path = tmp_path / "partial.csv"
    output_path.write_text("partial recording", encoding="utf-8")
    worker = MeasurementWorker(
        step=parse_steps(config)[0],
        config=config,
        a121_config=make_a121_config(config),
        output_path=output_path,
        port="TEST",
    )

    worker._abort()

    assert not output_path.exists()
