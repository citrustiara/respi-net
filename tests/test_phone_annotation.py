from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from respi_net.breath_phases import EXHALE, HOLD, IGNORE, INHALE, NOISE
from respi_net.phone_annotation import (
    Segment,
    compare_labels,
    guard_mask,
    load_segments,
    marks_to_segments,
    next_label,
    save_segments,
    segments_to_labels,
)

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from annotate_phone_phases import AnnotationSession  # noqa: E402


def test_breaths_alternate_and_a_hold_is_left_by_the_other_breath() -> None:
    assert next_label([]) == INHALE
    assert next_label([INHALE]) == EXHALE
    assert next_label([INHALE, EXHALE]) == INHALE
    assert next_label([INHALE, HOLD]) == EXHALE  # held after an inhale -> exhale next
    assert next_label([EXHALE, HOLD]) == INHALE
    assert next_label([INHALE, NOISE]) == EXHALE


def test_marks_become_segments_and_labels() -> None:
    segments = marks_to_segments([(2.0, INHALE), (4.0, EXHALE), (6.0, HOLD)], end_s=10.0)
    assert segments == [Segment(2.0, 4.0, INHALE), Segment(4.0, 6.0, EXHALE), Segment(6.0, 10.0, HOLD)]
    grid = np.arange(0.0, 12.0, 0.5)
    labels = segments_to_labels(segments, grid)
    assert labels[0] == IGNORE and labels[4] == INHALE and labels[8] == EXHALE and labels[12] == HOLD and labels[-1] == IGNORE


def test_segments_survive_a_csv_round_trip(tmp_path: Path) -> None:
    segments = [Segment(1.0, 2.5, INHALE), Segment(2.5, 5.0, NOISE)]
    save_segments(tmp_path / "x.csv", segments)
    assert load_segments(tmp_path / "x.csv") == segments


def test_guard_mask_hides_the_samples_around_a_change() -> None:
    grid = np.arange(0.0, 10.0, 0.1)
    labels = np.where(grid < 5.0, INHALE, EXHALE).astype(np.int8)
    masked = guard_mask(labels, grid, 0.3)
    assert masked[49] == IGNORE and masked[50] == IGNORE and masked[40] == INHALE and masked[60] == EXHALE


def test_two_passes_with_a_shifted_boundary_agree_away_from_it() -> None:
    grid = np.arange(0.0, 20.0, 0.05)
    first = segments_to_labels([Segment(0, 5, INHALE), Segment(5, 10, EXHALE), Segment(10, 20, HOLD)], grid)
    second = segments_to_labels([Segment(0, 5.2, INHALE), Segment(5.2, 10.1, EXHALE), Segment(10.1, 20, HOLD)], grid)
    scores = compare_labels(first, second, grid)
    assert scores["accuracy"] > 0.97
    assert scores["boundaries_found"] == 1.0
    assert 0.0 < scores["boundary_median_abs_error_s"] < 0.25


def test_session_clicks_use_keys_then_the_usual_order() -> None:
    session = AnnotationSession(0.0, 60.0)
    session.click(5.0)  # first click -> inhale by default
    session.click(7.5)  # exhale
    session.key("h")
    session.click(10.0)  # hold
    session.click(15.0)  # after a hold that followed an exhale -> inhale
    session.key(".")
    session.click(20.0)
    assert [segment.label for segment in session.segments()] == [INHALE, EXHALE, HOLD, INHALE]
    assert session.segments()[-1].end_s == 20.0
    session.key("backspace")  # removes the end mark first
    assert session.segments()[-1].end_s == 60.0
    session.key("backspace")
    assert [segment.label for segment in session.segments()] == [INHALE, EXHALE, HOLD]


def test_undo_keys_and_right_click_remove_marks() -> None:
    session = AnnotationSession(0.0, 60.0)
    for time_s in (5.0, 10.0, 15.0, 20.0):
        session.click(time_s)
    session.key("ctrl+z")
    session.key("cmd+z")
    assert [mark[0] for mark in session.marks] == [5.0, 10.0]
    session.remove_near(9.0)  # a wrong click anywhere: the nearest mark goes
    assert [mark[0] for mark in session.marks] == [5.0]
    session.remove_near(40.0)  # nothing within reach
    assert [mark[0] for mark in session.marks] == [5.0]
    session.key("u")
    assert session.marks == []
