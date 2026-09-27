"""Standalone breathing coach: pick a pattern file and follow it.

The recording tools drive the same :class:`~respi_net.breathing_coach.BreathingCoach`
from their measurement clock; here the clock is the window's own, so any
pattern file can be rehearsed, shown to a subject before a session, or used
on its own.  Edit a pattern in any text editor and press "Wczytaj ponownie"
to see the change without restarting.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Callable

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .breathing import BreathingPattern, PatternError, discover_patterns, format_clock, load_pattern
from .breathing_coach import BreathingCoach
from .paths import BREATHING_PATTERNS_DIR, PROJECT_ROOT


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class CoachWindow(QWidget):
    TICK_MS = 40

    def __init__(
        self,
        *,
        patterns: list[BreathingPattern],
        initial: BreathingPattern | None = None,
        lead_in_s: int = 3,
        sound: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.setWindowTitle("RespiNet — trener oddechu")
        self.resize(980, 600)
        self._clock = clock
        self.state = "idle"
        self.pattern: BreathingPattern | None = None
        self._lead_in_end = 0.0
        self._lead_in_total = 0.0
        self._run_start = 0.0
        self._paused_at = 0.0
        self._paused_total = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

        layout = QVBoxLayout(self)
        picker = QHBoxLayout()
        picker.addWidget(QLabel("Wzorzec"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.setMinimumWidth(320)
        self.open_button = QPushButton("Otwórz plik…")
        self.reload_button = QPushButton("Wczytaj ponownie")
        picker.addWidget(self.pattern_combo, 1)
        picker.addWidget(self.open_button)
        picker.addWidget(self.reload_button)
        layout.addLayout(picker)

        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.description)

        # The panel takes whatever room the window has: made full screen, the
        # cue grows with it and reads from across the room.
        self.coach = BreathingCoach()
        self.coach.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.coach, 1)

        controls = QHBoxLayout()
        self.start_button = QPushButton("Start (spacja)")
        self.stop_button = QPushButton("Stop (Esc)")
        self.lead_in_spin = QSpinBox()
        self.lead_in_spin.setRange(0, 60)
        self.lead_in_spin.setSuffix(" s")
        self.lead_in_spin.setValue(max(0, int(lead_in_s)))
        self.sound_check = QCheckBox("Sygnał przy zmianie fazy")
        self.sound_check.setChecked(sound)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        controls.addWidget(QLabel("Odliczanie przed startem"))
        controls.addWidget(self.lead_in_spin)
        controls.addWidget(self.sound_check)
        layout.addLayout(controls)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # Space belongs to the window, not to whichever button was clicked last.
        for button in (self.start_button, self.stop_button, self.open_button, self.reload_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.start_button.clicked.connect(self.toggle)
        self.stop_button.clicked.connect(self.stop)
        self.open_button.clicked.connect(self.open_file)
        self.reload_button.clicked.connect(self.reload_pattern)
        self.pattern_combo.currentIndexChanged.connect(self._on_pattern_chosen)
        QShortcut(QKeySequence("Space"), self, activated=self.toggle)
        QShortcut(QKeySequence("Esc"), self, activated=self.stop)

        for pattern in patterns:
            self._add_pattern(pattern)
        if initial is not None:
            self._select(self._add_pattern(initial))
        elif patterns:
            self._select(0)
        self._show_idle()

    # -- patterns ---------------------------------------------------------------------

    def _add_pattern(self, pattern: BreathingPattern) -> int:
        """Add ``pattern`` to the picker, replacing an entry loaded from the same file."""

        text = f"{pattern.name} · {format_clock(pattern.duration_s)}"
        for index in range(self.pattern_combo.count()):
            existing = self.pattern_combo.itemData(index)
            if existing is not None and existing.source is not None and existing.source == pattern.source:
                self.pattern_combo.setItemText(index, text)
                self.pattern_combo.setItemData(index, pattern)
                return index
        self.pattern_combo.addItem(text, pattern)
        return self.pattern_combo.count() - 1

    def _select(self, index: int) -> None:
        if self.pattern_combo.currentIndex() == index:
            self._on_pattern_chosen(index)
        else:
            self.pattern_combo.setCurrentIndex(index)

    @Slot(int)
    def _on_pattern_chosen(self, index: int) -> None:
        pattern = self.pattern_combo.itemData(index)
        if pattern is None:
            return
        self.pattern = pattern
        self.coach.set_title(pattern.name)
        self.coach.set_pattern(pattern)
        source = _display_path(pattern.source)
        text = pattern.description
        if source:
            text = f"{text}\nPlik: {source}" if text else f"Plik: {source}"
        self.description.setText(text)
        self._show_idle()

    @Slot()
    def open_file(self) -> None:
        if self.state not in ("idle", "finished"):
            return
        start_dir = BREATHING_PATTERNS_DIR if BREATHING_PATTERNS_DIR.is_dir() else PROJECT_ROOT
        path, _ = QFileDialog.getOpenFileName(self, "Otwórz wzorzec oddechu", str(start_dir), "Wzorzec oddechu (*.json)")
        if not path:
            return
        try:
            pattern = load_pattern(Path(path))
        except PatternError as exc:
            QMessageBox.warning(self, "Nieprawidłowy wzorzec", str(exc))
            return
        self._select(self._add_pattern(pattern))

    @Slot()
    def reload_pattern(self) -> None:
        if self.state not in ("idle", "finished") or self.pattern is None or self.pattern.source is None:
            return
        try:
            pattern = load_pattern(self.pattern.source)
        except PatternError as exc:
            QMessageBox.warning(self, "Nieprawidłowy wzorzec", str(exc))
            return
        self._select(self._add_pattern(pattern))
        self.status.setText("Wczytano ponownie z pliku.")

    # -- session ----------------------------------------------------------------------

    def _show_idle(self) -> None:
        self.state = "idle"
        self._timer.stop()
        if self.pattern is not None:
            self.coach.show_idle("GOTOWY", "Spacja rozpoczyna, Esc przerywa.")
            self.status.setText(f"{len(self.pattern.phases)} faz, {format_clock(self.pattern.duration_s)}.")
        else:
            self.coach.show_message("BRAK WZORCA", "Otwórz plik wzorca oddechu.")
            self.status.setText(f"Nie znaleziono wzorców w {_display_path(BREATHING_PATTERNS_DIR)}.")
        self._update_controls()

    def _update_controls(self) -> None:
        editable = self.state in ("idle", "finished")
        self.pattern_combo.setEnabled(editable)
        self.open_button.setEnabled(editable)
        self.reload_button.setEnabled(editable and self.pattern is not None and self.pattern.source is not None)
        self.lead_in_spin.setEnabled(editable)
        self.start_button.setEnabled(self.pattern is not None)
        self.stop_button.setEnabled(not editable)
        labels = {
            "idle": "Start (spacja)",
            "finished": "Jeszcze raz (spacja)",
            "lead_in": "Anuluj (spacja)",
            "running": "Pauza (spacja)",
            "paused": "Wznów (spacja)",
        }
        self.start_button.setText(labels[self.state])

    @Slot()
    def toggle(self) -> None:
        if self.pattern is None:
            return
        if self.state in ("idle", "finished"):
            self.start()
        elif self.state == "lead_in":
            self.stop()
        elif self.state == "running":
            self.pause()
        elif self.state == "paused":
            self.resume()

    def start(self) -> None:
        if self.pattern is None:
            return
        now = self._clock()
        lead_in = float(self.lead_in_spin.value())
        self._paused_total = 0.0
        if lead_in > 0:
            self.state = "lead_in"
            self._lead_in_total = lead_in
            self._lead_in_end = now + lead_in
            self.status.setText("Przygotuj się.")
        else:
            self._begin(now)
        self._update_controls()
        self._timer.start()
        self._tick()

    def _begin(self, at: float) -> None:
        self.state = "running"
        self._run_start = at
        self.status.setText("Podążaj za komendą. Spacja wstrzymuje.")
        self._update_controls()

    def pause(self) -> None:
        if self.state != "running":
            return
        self._tick()
        self.state = "paused"
        self._paused_at = self._clock()
        self._timer.stop()
        self.status.setText("Pauza. Spacja wznawia, Esc kończy.")
        self._update_controls()

    def resume(self) -> None:
        if self.state != "paused":
            return
        self._paused_total += self._clock() - self._paused_at
        self.state = "running"
        self.status.setText("Podążaj za komendą. Spacja wstrzymuje.")
        self._update_controls()
        self._timer.start()
        self._tick()

    @Slot()
    def stop(self) -> None:
        if self.state in ("idle", "finished"):
            return
        self._show_idle()
        self.status.setText("Przerwano.")

    def elapsed(self) -> float:
        now = self._paused_at if self.state == "paused" else self._clock()
        return max(0.0, now - self._run_start - self._paused_total)

    @Slot()
    def _tick(self) -> None:
        if self.pattern is None:
            return
        if self.state == "lead_in":
            remaining = self._lead_in_end - self._clock()
            if remaining > 0:
                self.coach.show_countdown(remaining, self._lead_in_total, detail="Ustabilizuj pozycję i rozluźnij ramiona.")
                return
            self._begin(self._lead_in_end)
        if self.state != "running":
            return
        elapsed = self.elapsed()
        if elapsed >= self.pattern.duration_s:
            self._finish()
            return
        if self.coach.set_elapsed(elapsed) and self.sound_check.isChecked():
            QApplication.beep()

    def _finish(self) -> None:
        self.state = "finished"
        self._timer.stop()
        self.coach.show_message("KONIEC", "Oddychaj swobodnie.", finished=True)
        self.status.setText(f"Ukończono: {self.pattern.name if self.pattern else ''}.")
        if self.sound_check.isChecked():
            QApplication.beep()
        self._update_controls()


def launch_coach(pattern: str | Path | None = None, *, lead_in_s: int = 3, sound: bool = True) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("RespiNet Breathing Coach")
    patterns, problems = discover_patterns()
    initial = load_pattern(pattern) if pattern is not None else None
    window = CoachWindow(patterns=patterns, initial=initial, lead_in_s=lead_in_s, sound=sound)
    if problems:
        window.status.setText(
            "Pominięto nieprawidłowe pliki: " + "; ".join(message for _, message in problems)
        )
    window.show()
    return int(app.exec())
