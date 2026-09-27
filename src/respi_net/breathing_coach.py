"""The breathing coach panel: what to do now, how long is left, what comes next.

It is read from a bed or a chair a few metres from the screen, so it is dark,
large, and says three things:

* along the top, the whole session as one shape per phase -- width is time,
  colour is what the phase asks for, and the outline is what the lungs do:
  an inhale climbs, an exhale falls, a hold stays high after an inhale and low
  after an exhale -- with each section named underneath and a cursor where
  the session has got to;
* the cue, with a bar draining as the phase runs out and the seconds left;
* the phase after this one and, inside a run of cycles, when the next stretch
  of the session starts.

The panel only draws.  Whoever owns the clock calls
:meth:`BreathingCoach.set_elapsed` and decides whether a new phase deserves a
beep.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .breathing import BreathPhase, BreathingPattern, CoachPosition, format_clock, format_seconds


@dataclass(frozen=True)
class KindStyle:
    colour: str
    ink: str


KIND_STYLES = {
    "inhale": KindStyle("#3b82f6", "#bfdbfe"),
    "exhale": KindStyle("#22c55e", "#bbf7d0"),
    "hold": KindStyle("#ef4444", "#fecaca"),
    "normal": KindStyle("#94a3b8", "#e2e8f0"),
    "settle": KindStyle("#64748b", "#cbd5e1"),
    "interference": KindStyle("#eab308", "#fef08a"),
}

# The timeline is a picture of the lungs, in shares of the plot's height.
# Empty stays well clear of the floor so a hold after an exhale is still a
# visible strip rather than a line.
LUNGS_EMPTY = 0.18
LUNGS_FULL = 1.0
# Phases that prescribe no breathing are flat, each at a height of its own.
FLAT_LEVELS = {"normal": 0.44, "settle": 0.32, "interference": 0.44}

PANEL = "#0b1220"
PANEL_EDGE = "#1e293b"
TEXT = "#f1f5f9"
TEXT_DIM = "#cbd5e1"
TEXT_MUTE = "#94a3b8"
TEXT_FAINT = "#475569"


def kind_style(kind: str) -> KindStyle:
    return KIND_STYLES.get(kind, KIND_STYLES["normal"])


def _css_rgba(colour: str, alpha: float) -> str:
    rgb = QColor(colour)
    return f"rgba({rgb.red()}, {rgb.green()}, {rgb.blue()}, {round(max(0.0, min(1.0, alpha)) * 255)})"


def seconds_left(remaining_s: float) -> int:
    """What a countdown shows: 2 until one second is left, then 1, then 0 at the end."""

    return max(0, math.ceil(remaining_s - 1e-6))


def phase_shapes(pattern: BreathingPattern) -> list[tuple[float, float]]:
    """``(start, end)`` height of every phase of ``pattern`` on the timeline.

    An inhale climbs from empty to full and an exhale falls back, so a run of
    cycles draws as the breaths themselves.  A hold stays where the last
    breath left the lungs: high after an inhale, low after an exhale -- and
    low after free breathing too, because the protocols hold after breathing
    out.  Box breathing comes out as the box it is named after.
    """

    shapes: list[tuple[float, float]] = []
    lungs = LUNGS_EMPTY
    for phase in pattern.phases:
        if phase.kind == "inhale":
            shapes.append((LUNGS_EMPTY, LUNGS_FULL))
            lungs = LUNGS_FULL
        elif phase.kind == "exhale":
            shapes.append((LUNGS_FULL, LUNGS_EMPTY))
            lungs = LUNGS_EMPTY
        elif phase.kind == "hold":
            shapes.append((lungs, lungs))
        else:
            level = FLAT_LEVELS.get(phase.kind, FLAT_LEVELS["normal"])
            shapes.append((level, level))
            lungs = LUNGS_EMPTY
    return shapes


def _keep_space_when_hidden(widget: QWidget) -> None:
    # The panel keeps one height in every state; a row that appeared only
    # while counting made everything below it jump at each start and stop.
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)


def _phase_outline(left: float, right: float, bottom: float, top_left: float, top_right: float) -> QPainterPath:
    path = QPainterPath()
    path.moveTo(left, bottom)
    path.lineTo(left, top_left)
    path.lineTo(right, top_right)
    path.lineTo(right, bottom)
    path.closeSubpath()
    return path


class CoachTimeline(QWidget):
    """The whole session as shapes; the running one fills and a cursor marks now."""

    PLOT_HEIGHT = 54
    LABEL_HEIGHT = 19

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pattern: BreathingPattern | None = None
        self._shapes: list[tuple[float, float]] = []
        self._elapsed: float | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.PLOT_HEIGHT + self.LABEL_HEIGHT)
        self.setMinimumWidth(240)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(640, self.PLOT_HEIGHT + self.LABEL_HEIGHT)

    def set_pattern(self, pattern: BreathingPattern | None) -> None:
        self._pattern = pattern
        self._shapes = phase_shapes(pattern) if pattern is not None else []
        self._elapsed = None
        self.update()

    def set_elapsed(self, elapsed_s: float | None) -> None:
        """``None`` draws the session not yet started: no cursor, nothing done."""

        self._elapsed = elapsed_s
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = float(self.width())
        plot = QRectF(0.0, 0.0, width, float(self.PLOT_HEIGHT))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 9))
        painter.drawRoundedRect(plot, 6.0, 6.0)
        pattern = self._pattern
        if pattern is None or pattern.duration_s <= 0:
            return
        scale = width / pattern.duration_s
        position = pattern.position(self._elapsed) if self._elapsed is not None else None
        bottom = plot.bottom()
        usable = plot.height() - 8.0
        for index, phase in enumerate(pattern.phases):
            span = phase.duration_s * scale
            gap = 1.5 if span > 6.0 else 0.5 if span > 2.0 else 0.0
            left = phase.start_s * scale
            right = left + max(1.0, span - gap)
            start_level, end_level = self._shapes[index]
            outline = _phase_outline(left, right, bottom, bottom - start_level * usable, bottom - end_level * usable)
            active = position is not None and not position.finished and index == position.index
            past = position is not None and (position.finished or phase.end_s <= position.elapsed_s)
            colour = QColor(kind_style(phase.kind).colour)
            colour.setAlphaF(1.0 if active else 0.28 if past else 0.78)
            painter.fillPath(outline, colour)
            if active:
                painter.save()
                painter.setClipPath(outline)
                painter.fillRect(QRectF(left, 0.0, (right - left) * position.fraction, bottom), QColor(255, 255, 255, 84))
                painter.restore()
                painter.strokePath(outline, QPen(QColor(255, 255, 255, 210), 1.0))
        if position is not None:
            x = min(width - 1.0, max(1.0, position.elapsed_s * scale))
            painter.fillRect(QRectF(x - 1.0, 0.0, 2.0, plot.height()), QColor("#ffffff"))
        self._paint_sections(painter, pattern, position, scale, plot.bottom())

    def _paint_sections(
        self,
        painter: QPainter,
        pattern: BreathingPattern,
        position: CoachPosition | None,
        scale: float,
        top: float,
    ) -> None:
        font = QFont(self.font())
        font.setPixelSize(12)
        bold = QFont(font)
        bold.setWeight(QFont.Weight.DemiBold)
        current = position.section.index if position is not None and not position.finished else None
        for section in pattern.sections:
            left = section.start_s * scale
            right = section.end_s * scale
            if section.index > 0:
                painter.fillRect(QRectF(left - 0.5, top + 3.0, 1.0, 9.0), QColor(255, 255, 255, 46))
            is_current = section.index == current
            painter.setFont(bold if is_current else font)
            metrics = painter.fontMetrics()
            room = right - left - 8.0
            text = next(
                (label for label in (section.label, section.short_label) if metrics.horizontalAdvance(label) <= room),
                "",
            )
            if not text:
                continue
            if is_current:
                colour = TEXT
            elif position is not None and (position.finished or section.end_s <= position.elapsed_s):
                colour = TEXT_FAINT
            else:
                colour = TEXT_MUTE
            painter.setPen(QColor(colour))
            painter.drawText(
                QRectF(left, top + 3.0, right - left, self.LABEL_HEIGHT - 3.0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                text,
            )


class PhaseCountdown(QWidget):
    """The time left in a phase as a bar draining to the left, notched every second."""

    TRACK_HEIGHT = 12.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._left = 0.0
        self._colour = QColor(KIND_STYLES["normal"].colour)
        self._seconds = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(480, 24)

    def set_state(self, fraction_left: float, colour: str, seconds: float) -> None:
        self._left = max(0.0, min(1.0, fraction_left))
        self._colour = QColor(colour)
        self._seconds = max(0.0, seconds)
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = float(self.width())
        top = (self.height() - self.TRACK_HEIGHT) / 2.0
        radius = self.TRACK_HEIGHT / 2.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 22))
        painter.drawRoundedRect(QRectF(0.0, top, width, self.TRACK_HEIGHT), radius, radius)
        filled = width * self._left
        if filled > 0.5:
            painter.setBrush(self._colour)
            fill_radius = min(radius, filled / 2.0)
            painter.drawRoundedRect(QRectF(0.0, top, filled, self.TRACK_HEIGHT), fill_radius, fill_radius)
        if self._seconds > 1.0 and width / self._seconds >= 16.0:
            painter.setBrush(QColor(PANEL))
            for second in range(1, math.ceil(self._seconds - 1e-6)):
                x = width * second / self._seconds
                painter.drawRect(QRectF(x - 1.0, top, 2.0, self.TRACK_HEIGHT))


class BreathingCoach(QFrame):
    """Session timeline, cue, time left and what comes next, on one dark panel."""

    # The last seconds of a long phase are when the next one matters: coming
    # out of a fifteen second hold, "inhale" should be in view before it is due.
    READY_SECONDS = 3.0
    READY_MIN_PHASE_S = 6.0

    def __init__(self, title: str = "Przebieg próby", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("BreathingCoach")
        self._pattern: BreathingPattern | None = None
        self._index: int | None = None
        self._css: dict[QWidget, str] = {}
        self._cue_kind = "normal"
        self._countdown: tuple[float, float, float] | None = None
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(33)
        self._countdown_timer.timeout.connect(self._advance_countdown)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        self._mono = mono
        self.setStyleSheet(
            f"QFrame#BreathingCoach {{ background: {PANEL}; border: 1px solid {PANEL_EDGE}; border-radius: 14px; }}"
            f"QFrame#BreathingCoach QLabel {{ background: transparent; border: 0; color: {TEXT}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 13, 18, 15)
        layout.setSpacing(7)

        header = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setStyleSheet(f"color: {TEXT_MUTE}; font-size: 12px; font-weight: 700;")
        self.clock_label = QLabel()
        self.clock_label.setStyleSheet(f"color: {TEXT_DIM}; font-family: '{mono}'; font-size: 13px;")
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.clock_label)
        layout.addLayout(header)

        self.timeline = CoachTimeline(self)
        layout.addWidget(self.timeline)
        layout.addSpacing(4)

        self.cue_label = QLabel()
        self.cue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cue_label.setWordWrap(True)
        self.cue_label.setMinimumHeight(68)
        layout.addWidget(self.cue_label, 1)

        self.detail_label = QLabel()
        self.detail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.detail_label.setMinimumHeight(22)
        self.detail_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 15px;")
        layout.addWidget(self.detail_label)

        self.countdown_row = QWidget(self)
        countdown = QHBoxLayout(self.countdown_row)
        countdown.setContentsMargins(0, 2, 0, 0)
        countdown.setSpacing(14)
        self.countdown_bar = PhaseCountdown(self.countdown_row)
        self.countdown_value = QLabel()
        self.countdown_value.setMinimumSize(92, 42)
        self.countdown_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        countdown.addWidget(self.countdown_bar, 1)
        countdown.addWidget(self.countdown_value)
        layout.addWidget(self.countdown_row)

        self.next_frame = QFrame(self)
        self.next_frame.setObjectName("CoachNext")
        upcoming = QHBoxLayout(self.next_frame)
        upcoming.setContentsMargins(14, 9, 14, 9)
        upcoming.setSpacing(9)
        self.next_tag = QLabel()
        self.next_tag.setStyleSheet(f"color: {TEXT_MUTE}; font-size: 11px; font-weight: 700;")
        self.next_dot = QLabel("●")
        self.next_cue = QLabel()
        self.next_cue.setStyleSheet(f"color: {TEXT}; font-size: 17px; font-weight: 700;")
        self.next_length = QLabel()
        self.next_length.setStyleSheet(f"color: {TEXT_MUTE}; font-family: '{mono}'; font-size: 14px;")
        self.later_label = QLabel()
        self.later_label.setStyleSheet(f"color: {TEXT_MUTE}; font-size: 14px;")
        upcoming.addWidget(self.next_tag)
        upcoming.addWidget(self.next_dot)
        upcoming.addWidget(self.next_cue)
        upcoming.addWidget(self.next_length)
        upcoming.addStretch(1)
        upcoming.addWidget(self.later_label)
        layout.addWidget(self.next_frame)

        for widget in (self.clock_label, self.timeline, self.countdown_row, self.next_frame, self.later_label):
            _keep_space_when_hidden(widget)
        self.set_title(title)
        self.set_pattern(None)
        self.show_message("")

    # -- schedule -----------------------------------------------------------------

    @property
    def pattern(self) -> BreathingPattern | None:
        return self._pattern

    def set_title(self, title: str) -> None:
        self.title_label.setText(title.upper())

    def set_pattern(self, pattern: BreathingPattern | None) -> None:
        """What the timeline draws; ``None`` (or no phases) hides it."""

        self._pattern = pattern if pattern is not None and pattern.phases else None
        self._index = None
        self._stop_countdown()
        self.timeline.set_pattern(self._pattern)
        self.timeline.setVisible(self._pattern is not None)
        self.clock_label.setVisible(self._pattern is not None)

    def set_phases(self, phases: Sequence[BreathPhase]) -> None:
        self.set_pattern(BreathingPattern.from_phases(phases) if phases else None)

    # -- states ---------------------------------------------------------------------

    def show_idle(self, cue: str, detail: str = "", *, kind: str = "normal") -> None:
        """Before the start: the whole session ahead, and what it opens with."""

        self._reset_position()
        self._show_cue(cue, detail, kind)
        self.countdown_row.hide()
        first = self._pattern.phases[0] if self._pattern is not None else None
        self._show_next(first, "NA START")
        self._show_later("")
        self._show_clock(0.0)

    def show_countdown(
        self,
        remaining_s: float,
        total_s: float | None = None,
        *,
        cue: str = "PRZYGOTUJ SIĘ",
        detail: str = "",
    ) -> None:
        """The run-up to the start, draining smoothly between once-a-second updates."""

        remaining = max(0.0, float(remaining_s))
        total = max(remaining, float(total_s or 0.0), 1e-6)
        self._reset_position()
        self._show_cue(cue, detail, "normal")
        self._show_later("")
        self._show_clock(0.0)
        self._countdown = (remaining, total, time.monotonic())
        self._render_countdown(remaining)
        if remaining > 0 and not self._countdown_timer.isActive():
            self._countdown_timer.start()

    def set_elapsed(self, elapsed_s: float) -> bool:
        """Show the moment ``elapsed_s`` into the session; ``True`` when a new phase began."""

        self._stop_countdown()
        pattern = self._pattern
        if pattern is None:
            return False
        position = pattern.position(elapsed_s)
        assert position is not None
        changed = position.index != self._index
        self._index = position.index
        phase = position.phase
        style = kind_style(phase.kind)
        self.timeline.set_elapsed(position.elapsed_s)
        self._show_cue(phase.cue, phase.detail, phase.kind)
        self.countdown_row.show()
        self.countdown_bar.set_state(1.0 - position.fraction, style.colour, phase.duration_s)
        self._show_seconds(position.remaining_s, style.ink)
        ready = phase.duration_s >= self.READY_MIN_PHASE_S and position.remaining_s <= self.READY_SECONDS
        tag = f"ZA {seconds_left(position.remaining_s)} s" if ready else "NASTĘPNIE"
        self._show_next(position.next_phase, tag, highlight=ready, ending=True)
        self._show_later(self._later_text(position))
        self._show_clock(position.elapsed_s)
        return changed

    def show_message(self, cue: str, detail: str = "", *, kind: str = "normal", finished: bool = False) -> None:
        """A cue with nothing counting: the end, an abort, a decision between trials.

        ``finished`` also draws the whole session as done.
        """

        self._reset_position(keep_timeline=not finished)
        if finished and self._pattern is not None:
            self.timeline.set_elapsed(self._pattern.duration_s)
            self._show_clock(self._pattern.duration_s)
        self._show_cue(cue, detail, kind)
        self.countdown_row.hide()
        self.next_frame.hide()

    # -- internals ------------------------------------------------------------------

    def _reset_position(self, *, keep_timeline: bool = False) -> None:
        self._stop_countdown()
        self._index = None
        if not keep_timeline:
            self.timeline.set_elapsed(None)

    def _stop_countdown(self) -> None:
        self._countdown = None
        self._countdown_timer.stop()

    def _advance_countdown(self) -> None:
        if self._countdown is None:
            self._countdown_timer.stop()
            return
        remaining, _, anchor = self._countdown
        # Never run more than a second ahead of the last update: if the next
        # one is late the bar waits for it rather than inventing a start.
        shown = max(0.0, remaining - 1.0, remaining - (time.monotonic() - anchor))
        self._render_countdown(shown)
        if shown <= 0.0:
            self._countdown_timer.stop()

    def _render_countdown(self, remaining: float) -> None:
        assert self._countdown is not None
        total = self._countdown[1]
        self.countdown_row.show()
        self.countdown_bar.set_state(remaining / total, KIND_STYLES["normal"].colour, total)
        self._show_seconds(remaining, TEXT)
        first = self._pattern.phases[0] if self._pattern is not None else None
        ready = remaining <= self.READY_SECONDS
        tag = f"ZA {seconds_left(remaining)} s" if ready else "NA START"
        self._show_next(first, tag, highlight=ready and first is not None)

    def _later_text(self, position: CoachPosition) -> str:
        upcoming = position.next_section
        if upcoming is None or upcoming.first_phase == position.index + 1:
            return ""
        until = position.until_next_section_s or 0.0
        when = format_clock(until, round_up=True) if until >= 60.0 else f"{seconds_left(until)} s"
        return f"za {when}: {upcoming.label}"

    def _restyle(self, widget: QWidget, css: str) -> None:
        if self._css.get(widget) != css:
            self._css[widget] = css
            widget.setStyleSheet(css)

    @staticmethod
    def _set_text(label: QLabel, text: str) -> None:
        if label.text() != text:
            label.setText(text)

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._style_cue()

    def _type_scale(self) -> float:
        # 1.0 at the width of the recording tools' windows; a full-screen coach
        # on a wide display sets the cue up to twice as large.
        return max(1.0, min(2.0, self.width() / 900.0))

    def _style_cue(self) -> None:
        style = kind_style(self._cue_kind)
        # A one- or two-word command is set as large as the panel allows; a
        # sentence between trials is set smaller so it stays on one line.
        size = round((40 if len(self.cue_label.text()) <= 18 else 30) * self._type_scale())
        self._restyle(
            self.cue_label,
            f"background: {_css_rgba(style.colour, 0.2)}; color: {style.ink}; font-size: {size}px; "
            "font-weight: 800; padding: 10px 14px; border-radius: 10px;",
        )

    def _show_cue(self, cue: str, detail: str, kind: str) -> None:
        self._cue_kind = kind
        self._set_text(self.cue_label, cue)
        self._style_cue()
        self._set_text(self.detail_label, detail)

    def _show_seconds(self, remaining: float, ink: str) -> None:
        self._restyle(
            self.countdown_value,
            f"color: {ink}; font-family: '{self._mono}'; font-size: {round(30 * self._type_scale())}px; font-weight: 700;",
        )
        self._set_text(
            self.countdown_value,
            f"{seconds_left(remaining)}<span style='font-size: 16px; color: {TEXT_MUTE};'> s</span>",
        )

    def _show_next(
        self,
        phase: BreathPhase | None,
        tag: str,
        *,
        highlight: bool = False,
        ending: bool = False,
    ) -> None:
        """The phase after this one; with ``ending``, no phase means the session ends."""

        if phase is None and not ending:
            self.next_frame.hide()
            return
        self.next_frame.show()
        colour = kind_style(phase.kind).colour if phase is not None else TEXT_MUTE
        background = _css_rgba(colour, 0.24) if highlight else "rgba(255, 255, 255, 12)"
        border = colour if highlight else "transparent"
        self._restyle(
            self.next_frame,
            f"QFrame#CoachNext {{ background: {background}; border: 1px solid {border}; border-radius: 10px; }}",
        )
        self._restyle(self.next_dot, f"color: {colour}; font-size: 16px;")
        self._set_text(self.next_tag, tag)
        self._set_text(self.next_cue, phase.cue if phase is not None else "KONIEC")
        self._set_text(self.next_length, f"{format_seconds(phase.duration_s)} s" if phase is not None else "")

    def _show_later(self, text: str) -> None:
        self._set_text(self.later_label, text)
        self.later_label.setVisible(bool(text))

    def _show_clock(self, elapsed_s: float) -> None:
        if self._pattern is not None:
            self._set_text(
                self.clock_label,
                f"{format_clock(elapsed_s)} / {format_clock(self._pattern.duration_s)}",
            )
