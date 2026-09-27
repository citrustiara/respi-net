import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from respi_net.breathing import discover_patterns, load_pattern  # noqa: E402
from respi_net.breathing_coach import LUNGS_EMPTY, LUNGS_FULL, BreathingCoach, phase_shapes  # noqa: E402
from respi_net.coach_app import CoachWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_coach_follows_a_whole_trial(qapp: QApplication) -> None:
    pattern = load_pattern("paced_12_hold")
    coach = BreathingCoach()
    coach.resize(900, 360)
    coach.set_pattern(pattern)
    coach.show_idle("PO STARCIE: ODDYCHAJ SWOBODNIE")
    assert coach.next_tag.text() == "NA START"
    assert coach.next_cue.text() == "ODDYCHAJ SWOBODNIE"

    coach.show_countdown(2.0, 8.0)
    assert coach.next_tag.text() == "ZA 2 s"

    new_phases = sum(coach.set_elapsed(step * 0.05) for step in range(0, 1801))
    assert new_phases == len(pattern.phases)

    coach.set_elapsed(21.3)
    assert coach.cue_label.text() == "WDECH"
    assert coach.next_cue.text() == "WYDECH"
    assert coach.later_label.text() == "za 24 s: Wstrzymanie 15 s"
    assert coach.clock_label.text() == "0:21 / 1:30"

    coach.set_elapsed(57.6)
    assert coach.next_tag.text() == "ZA 3 s"
    assert coach.next_cue.text() == "WDECH"
    assert coach.later_label.isHidden()

    coach.set_elapsed(88.5)
    assert coach.next_cue.text() == "KONIEC"
    assert not coach.grab().isNull()

    coach.show_message("KONIEC", "Oddychaj swobodnie.", finished=True)
    assert coach.countdown_row.isHidden() and coach.next_frame.isHidden()
    assert coach.clock_label.text() == "1:30 / 1:30"
    assert coach.set_elapsed(0.0) is True  # the next run starts on a fresh phase


def test_coach_without_a_schedule_only_shows_the_message(qapp: QApplication) -> None:
    coach = BreathingCoach()
    coach.set_phases(())
    coach.show_message("DECYZJA O ZASIĘGU")

    assert coach.timeline.isHidden()
    assert coach.set_elapsed(10.0) is False
    assert coach.cue_label.text() == "DECYZJA O ZASIĘGU"


def test_timeline_draws_what_the_lungs_do() -> None:
    inhale, hold_full, exhale, hold_empty = phase_shapes(load_pattern("box_breathing"))[1:5]
    assert inhale == (LUNGS_EMPTY, LUNGS_FULL)
    assert hold_full == (LUNGS_FULL, LUNGS_FULL)
    assert exhale == (LUNGS_FULL, LUNGS_EMPTY)
    assert hold_empty == (LUNGS_EMPTY, LUNGS_EMPTY)

    paced = load_pattern("paced_12_hold")
    shapes = phase_shapes(paced)
    assert shapes[paced.sections[2].first_phase] == (LUNGS_EMPTY, LUNGS_EMPTY)

    settle, free, hold = phase_shapes(load_pattern("natural_hold"))
    assert settle[0] == settle[1] and free[0] == free[1]
    assert hold == (LUNGS_EMPTY, LUNGS_EMPTY)


def test_coach_window_counts_in_pauses_and_finishes(qapp: QApplication) -> None:
    now = [1000.0]
    patterns, _ = discover_patterns()
    window = CoachWindow(
        patterns=patterns,
        initial=load_pattern("natural_hold"),
        lead_in_s=2,
        sound=False,
        clock=lambda: now[0],
    )
    assert window.state == "idle"
    assert window.pattern.pattern_id == "natural_hold"
    assert window.pattern_combo.count() == len(patterns)

    window.toggle()
    assert window.state == "lead_in"
    now[0] += 2.5
    window._tick()
    assert window.state == "running"
    assert window.elapsed() == pytest.approx(0.5)

    window.toggle()
    assert window.state == "paused"
    now[0] += 30.0
    assert window.elapsed() == pytest.approx(0.5)
    window.toggle()
    now[0] += 12.0
    window._tick()
    assert window.elapsed() == pytest.approx(12.5)
    assert window.coach.cue_label.text() == "ODDYCHAJ SWOBODNIE"

    now[0] += 60.0
    window._tick()
    assert window.state == "finished"
    assert window.start_button.isEnabled()

    window.toggle()
    window.stop()
    assert window.state == "idle"
    window.close()
