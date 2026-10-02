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
    snap_labels,
    snap_segments,
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


def test_the_edges_of_a_hold_get_a_wider_guard() -> None:
    grid = np.arange(0.0, 20.0, 0.1)
    labels = np.where(grid < 10.0, EXHALE, HOLD).astype(np.int8)
    masked = guard_mask(labels, grid, 0.2, 1.5)
    assert masked[90] == IGNORE and masked[110] == IGNORE  # within 1.5 s of the start of the hold
    assert masked[80] == EXHALE and masked[120] == HOLD
    leaving = guard_mask(np.where(grid < 10.0, HOLD, INHALE).astype(np.int8), grid, 0.2, 1.5)
    assert leaving[85] == IGNORE and leaving[70] == HOLD  # the end of a hold is as fuzzy as its start


def test_boundaries_snap_to_the_turning_points_of_the_trace() -> None:
    grid = np.arange(0.0, 30.0, 0.05)
    trace = -np.cos(2 * np.pi * grid / 6.0)  # troughs at 0, 6, 12 ...; crests at 3, 9, 15 ...
    late = [Segment(6.2, 9.3, INHALE), Segment(9.3, 12.2, EXHALE), Segment(12.2, 15.3, INHALE)]  # clicks 0.2-0.3 s after the turns
    snapped = snap_segments(late, grid, trace)
    assert abs(snapped[0].end_s - 9.0) < 0.15 and snapped[0].end_s == snapped[1].start_s
    assert abs(snapped[1].end_s - 12.0) < 0.15
    assert snapped[0].start_s == 6.2 and snapped[2].end_s == 15.3  # outer edges are untouched


def test_snapping_leaves_holds_alone_and_works_on_label_arrays() -> None:
    grid = np.arange(0.0, 30.0, 0.05)
    trace = -np.cos(2 * np.pi * grid / 6.0)
    segments = [Segment(6.2, 9.3, INHALE), Segment(9.3, 14.0, HOLD)]
    assert snap_segments(segments, grid, trace) == segments
    labels = segments_to_labels([Segment(6.2, 9.3, INHALE), Segment(9.3, 12.2, EXHALE)], grid)
    snapped = snap_labels(labels, grid, trace)
    assert abs(grid[np.flatnonzero(snapped == EXHALE)[0]] - 9.0) < 0.15


def test_a_steady_click_lag_is_measured_and_taken_out() -> None:
    from respi_net.phone_annotation import boundary_offsets, shift_boundaries

    grid = np.arange(0.0, 60.0, 0.05)
    trace = -np.cos(2 * np.pi * grid / 6.0)  # troughs at 0, 6, 12 ...; crests at 3, 9, 15 ...
    lag = 0.25
    marks = [(6.0 + lag, INHALE), (9.0 + lag, EXHALE), (12.0 + lag, INHALE), (15.0 + lag, EXHALE), (18.0 + lag, INHALE)]
    segments = marks_to_segments(marks, 21.25)
    offsets = boundary_offsets(segments, grid, trace)
    assert abs(offsets["crest"]["median_s"] - lag) < 0.1 and abs(offsets["trough"]["median_s"] - lag) < 0.1
    corrected = shift_boundaries(segments, offsets)
    assert abs(corrected[0].end_s - 9.0) < 0.12 and abs(corrected[1].end_s - 12.0) < 0.12


def _trough_snapper(grid: np.ndarray, trace: np.ndarray):
    from respi_net.chest_signal import light_filter
    from respi_net.phone_annotation import turning_point

    smooth = light_filter(trace, 1.0 / float(np.median(np.diff(grid))), 1.0)
    return lambda time_s, label: turning_point(grid, smooth, time_s, label)


def test_clicks_snap_to_the_turning_point_and_keep_the_raw_time() -> None:
    grid = np.arange(0.0, 60.0, 0.05)
    trace = -np.cos(2 * np.pi * grid / 6.0)  # troughs at 0, 6, 12 ...; crests at 3, 9, 15 ...
    session = AnnotationSession(0.0, 60.0, _trough_snapper(grid, trace))
    session.click(6.3)  # an inhale starts at a trough: lands on 6.0
    session.click(9.25)  # exhale starts at a crest: 9.0
    session.key("h")
    session.click(12.4)  # a hold is placed exactly where it is clicked
    assert [round(t, 1) for t, _ in session.marks] == [6.0, 9.0, 12.4]
    assert [round(c, 2) for c in session.clicks] == [6.3, 9.25, 12.4]
    assert [round(x, 2) for x in session.clicked()] == [6.3, 9.25, 12.4]
    session.key("ctrl+z")
    assert len(session.marks) == 2 and len(session.clicks) == 2


def test_the_raw_click_time_is_saved_as_an_extra_column(tmp_path: Path) -> None:
    save_segments(tmp_path / "x.csv", [Segment(1.0, 2.0, INHALE), Segment(2.0, 3.0, EXHALE)], [1.2, None])
    text = (tmp_path / "x.csv").read_text(encoding="utf-8").splitlines()
    assert text[0] == "start_s,end_s,class,clicked_start_s" and text[1].endswith(",1.200") and text[2].endswith(",")
    assert load_segments(tmp_path / "x.csv") == [Segment(1.0, 2.0, INHALE), Segment(2.0, 3.0, EXHALE)]
