from pathlib import Path
import json

from click.testing import CliRunner
import pytest

from respi_net.breathing import (
    BreathPhase,
    PatternError,
    cue_at,
    discover_patterns,
    format_clock,
    format_seconds,
    load_pattern,
    pattern_from_dict,
    phases_from_cues,
)
from respi_net.cli import cli


def _paced(cycles: int, **extra: object) -> dict[str, object]:
    return {"kind": "paced", "cycles": cycles, "inhale_seconds": 2, "exhale_seconds": 3, **extra}


def test_every_shipped_pattern_loads() -> None:
    patterns, problems = discover_patterns()

    assert problems == []
    assert {pattern.pattern_id for pattern in patterns} >= {
        "paced_12_hold",
        "natural_hold",
        "box_breathing",
        "slow_6_per_min",
    }
    assert all(pattern.phases and pattern.name for pattern in patterns)


def test_research_pattern_sections_are_named_from_the_blocks() -> None:
    pattern = load_pattern("paced_12_hold")

    assert pattern.duration_s == 90
    assert len(pattern.phases) == 28
    assert [(s.start_s, s.end_s, s.label) for s in pattern.sections] == [
        (0, 10, "Swobodnie 10 s"),
        (10, 45, "12/min × 7"),
        (45, 60, "Wstrzymanie 15 s"),
        (60, 90, "12/min × 6"),
    ]
    assert [s.short_label for s in pattern.sections] == ["Swobodnie", "12/min", "Wstrzymanie", "12/min"]


def test_paced_block_with_holds_is_box_breathing() -> None:
    pattern = load_pattern("box_breathing")
    cycle = pattern.phases[1:5]

    assert [(phase.kind, phase.duration_s) for phase in cycle] == [
        ("inhale", 4),
        ("hold", 4),
        ("exhale", 4),
        ("hold", 4),
    ]
    assert cycle[1].detail != cycle[3].detail
    assert pattern.sections[1].label == "4-4-4-4 s × 8"
    assert pattern.duration_s == 10 + 8 * 16


def test_repeat_block_repeats_its_sections() -> None:
    pattern = pattern_from_dict(
        {
            "blocks": [
                {
                    "kind": "repeat",
                    "times": 2,
                    "blocks": [_paced(3), {"kind": "hold", "duration_seconds": 10, "label": "Pauza"}],
                }
            ]
        }
    )

    assert pattern.duration_s == 2 * (3 * 5 + 10)
    assert [section.label for section in pattern.sections] == ["12/min × 3", "Pauza", "12/min × 3", "Pauza"]
    assert [phase.section for phase in pattern.phases[:8]] == [0] * 6 + [1, 2]


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        ([_paced(3, hold_after_inhale_second=2)], r"blocks\[0\]: unknown key 'hold_after_inhale_second'"),
        ([{"kind": "breathe", "duration_seconds": 5}], r"blocks\[0\]\.kind 'breathe' is not one of"),
        ([{"kind": "normal", "duration_seconds": 0}], r"blocks\[0\]\.duration_seconds must be positive"),
        ([{"kind": "normal"}], r"blocks\[0\]\.duration_seconds is required"),
        ([{"kind": "normal", "duration_seconds": True}], r"must be a number of seconds"),
        ([_paced(2.5)], r"blocks\[0\]\.cycles must be a whole number"),
        ([_paced(2, hold_after_exhale_seconds=-1)], r"hold_after_exhale_seconds cannot be negative"),
        ([{"kind": "repeat", "times": 0, "blocks": [_paced(1)]}], r"blocks\[0\]\.times must be a whole number"),
        ([{"kind": "repeat", "times": 2, "blocks": []}], r"blocks\[0\]\.blocks must be a non-empty list"),
        ([{"duration_seconds": 5}], r"blocks\[0\] needs a 'kind'"),
        ([], r"blocks must be a non-empty list"),
    ],
)
def test_invalid_patterns_name_the_offending_key(blocks: list[object], message: str) -> None:
    with pytest.raises(PatternError, match=message):
        pattern_from_dict({"name": "broken", "blocks": blocks})


def test_unknown_top_level_key_is_rejected() -> None:
    with pytest.raises(PatternError, match="unknown key 'phases'"):
        pattern_from_dict({"phases": []})


def test_runaway_repeat_is_refused() -> None:
    with pytest.raises(PatternError, match="more than 20000 phases"):
        pattern_from_dict({"blocks": [{"kind": "repeat", "times": 10**9, "blocks": [_paced(1)]}]})


def test_load_pattern_by_name_or_path_and_reports_what_exists(tmp_path: Path) -> None:
    by_name = load_pattern("natural_hold")
    by_path = load_pattern(by_name.source)

    assert by_path == by_name
    assert load_pattern("natural_hold.json") == by_name
    with pytest.raises(PatternError, match="Available: .*paced_12_hold"):
        load_pattern("no_such_pattern")

    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"blocks": [{"kind": "normal", "duration_seconds": -1}]}), encoding="utf-8")
    with pytest.raises(PatternError, match=r"broken\.json: blocks\[0\]\.duration_seconds must be positive"):
        load_pattern(broken)


def test_cue_at_hands_a_boundary_to_the_cue_starting_there() -> None:
    phases = load_pattern("paced_12_hold").phases

    assert cue_at(phases, -1.0)[0].kind == "normal"
    assert cue_at(phases, 10.0)[0].kind == "inhale"
    assert cue_at(phases, 44.99)[0].kind == "exhale"
    assert cue_at(phases, 45.0)[0].kind == "hold"
    last, remaining = cue_at(phases, 120.0)
    assert last is phases[-1] and remaining == 0.0
    assert cue_at([], 3.0) is None


def test_position_names_the_next_phase_and_the_next_stretch() -> None:
    pattern = load_pattern("paced_12_hold")

    inhale = pattern.position(21.3)
    assert inhale.phase.kind == "inhale"
    assert inhale.remaining_s == pytest.approx(0.7)
    assert inhale.fraction == pytest.approx(0.65)
    assert inhale.next_phase.kind == "exhale"
    assert inhale.next_section.label == "Wstrzymanie 15 s"
    assert inhale.until_next_section_s == pytest.approx(23.7)

    hold = pattern.position(57.6)
    assert hold.phase.kind == "hold"
    assert hold.next_section.first_phase == hold.index + 1

    end = pattern.position(95.0)
    assert end.finished and end.index == len(pattern.phases) - 1
    assert end.remaining_s == 0.0 and end.next_phase is None and end.next_section is None


def test_recording_tool_cues_regroup_into_the_same_sections() -> None:
    pattern = load_pattern("paced_12_hold")
    recorded = [
        {"start_s": p.start_s, "end_s": p.end_s, "kind": p.kind, "title": p.cue, "detail": p.detail}
        for p in pattern.phases
    ]

    class Record:
        def __init__(self, values: dict[str, object]) -> None:
            self.__dict__.update(values)

    phases = phases_from_cues(Record(values) for values in recorded)

    assert [phase.cue for phase in phases] == [phase.cue for phase in pattern.phases]
    rebuilt = pattern.from_phases(phases).sections
    assert [(s.start_s, s.end_s, s.label) for s in rebuilt] == [
        (s.start_s, s.end_s, s.label) for s in pattern.sections
    ]


def test_formatting_helpers() -> None:
    assert format_seconds(2.0) == "2"
    assert format_seconds(2.5) == "2.5"
    assert format_clock(90) == "1:30"
    assert format_clock(12.7) == "0:12"
    assert format_clock(12.2, round_up=True) == "0:13"
    assert BreathPhase(1.0, 3.5, "inhale", "WDECH", "").duration_s == 2.5


def test_coach_cli_lists_prints_and_rejects_patterns() -> None:
    runner = CliRunner()

    listed = runner.invoke(cli, ["coach", "--list"])
    assert listed.exit_code == 0
    assert "paced_12_hold" in listed.output and "1:30" in listed.output

    printed = runner.invoke(cli, ["coach", "--print", "paced_12_hold"])
    assert printed.exit_code == 0
    assert "12/min × 7" in printed.output and "WSTRZYMAJ ODDECH" in printed.output

    missing = runner.invoke(cli, ["coach", "--print", "no_such_pattern"])
    assert missing.exit_code != 0
    assert "Available" in missing.output
