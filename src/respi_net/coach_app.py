"""Standalone breathing coach: pick a pattern file and follow it.

The recording tools drive the same :class:`~respi_net.breathing_coach.BreathingCoach`
from their measurement clock; here the clock is the window's own, so any
pattern file can be rehearsed, shown to a subject before a session, or used
on its own.  With a strain belt attached (``respi coach --belt``) the window also
plots the belt live and saves every completed run as a CSV with the coached
phase on each sample.  Edit a pattern in any text editor and press "Wczytaj ponownie"
to see the change without restarting.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Callable

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

if TYPE_CHECKING:
    from .strain_belt import StrainBeltCapture


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class CoachWindow(QWidget):
    TICK_MS = 40
    BELT_PLOT_MS = 100
    BELT_WINDOW_S = 30.0

    def __init__(
        self,
        *,
        patterns: list[BreathingPattern],
        initial: BreathingPattern | None = None,
        lead_in_s: int = 3,
        sound: bool = True,
        clock: Callable[[], float] = time.monotonic,
        belt: StrainBeltCapture | None = None,
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
        self.belt = belt
        self.last_belt_path: Path | None = None
        # Belt samples carry wall-clock stamps; the coach runs on ``clock``.
        self._wall_offset = time.time() - clock()
        self._pauses_wall: list[tuple[float, float]] = []
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
        if belt is not None:
            self._build_belt_plot(layout)

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
        self._pauses_wall = []
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
        now = self._clock()
        self._paused_total += now - self._paused_at
        self._pauses_wall.append((self._paused_at + self._wall_offset, now + self._wall_offset))
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
        if self.belt is not None:
            self._save_belt_run()
        if self.sound_check.isChecked():
            QApplication.beep()
        self._update_controls()

    # -- strain belt ------------------------------------------------------------------

    def _build_belt_plot(self, layout: QVBoxLayout) -> None:
        import pyqtgraph as pg

        self.belt_plot = pg.PlotWidget()
        self.belt_plot.setMinimumHeight(170)
        self.belt_plot.setMaximumHeight(240)
        self.belt_plot.setLabel("bottom", "czas [s]")
        self.belt_plot.setLabel("left", "pas (HX711, bez trendu)")
        self.belt_plot.setXRange(-self.BELT_WINDOW_S, 0, padding=0)
        self.belt_plot.showGrid(x=True, y=True, alpha=0.2)
        self.belt_curve = self.belt_plot.plot(pen=pg.mkPen("#2f80ed", width=2))
        layout.addWidget(self.belt_plot)
        self.belt_status = QLabel()
        layout.addWidget(self.belt_status)
        self._belt_timer = QTimer(self)
        self._belt_timer.setInterval(self.BELT_PLOT_MS)
        self._belt_timer.timeout.connect(self._update_belt_plot)
        self._belt_timer.start()

    @Slot()
    def _update_belt_plot(self) -> None:
        belt = self.belt
        if belt is None:
            return
        now = time.time()
        rows = [row for row in belt.snapshot_rows(max(0, belt.data_count() - 600)) if row[0] / 1000.0 >= now - self.BELT_WINDOW_S]
        if len(rows) >= 2:
            import numpy as np

            times = np.array([row[0] / 1000.0 - now for row in rows])
            values = np.array([float(row[3]) for row in rows])
            # A straight line over the visible window takes the belt's slow creep out of the view.
            trend = np.polyval(np.polyfit(times, values, 1), times)
            self.belt_curve.setData(times, values - trend)
        else:
            self.belt_curve.setData([], [])
        rate = len([row for row in rows if row[0] / 1000.0 >= now - 5.0]) / 5.0
        link = "połączony" if belt.connected else f"ROZŁĄCZONY ({belt.status})"
        saved = f" · ostatni zapis: {_display_path(self.last_belt_path)}" if self.last_belt_path else ""
        self.belt_status.setText(f"Pas: {link} · {rate:.1f} Hz · próbek {belt.data_count()}{saved}")

    def _save_belt_run(self) -> None:
        from .strain_belt import label_coach_rows, save_coach_recording

        if self.belt is None or self.pattern is None:
            return
        start_wall = self._run_start + self._wall_offset
        end_wall = self._clock() + self._wall_offset
        rows = label_coach_rows(self.belt.snapshot_rows(), self.pattern, start_wall, end_wall, self._pauses_wall)
        if len(rows) < 10:
            self.status.setText(f"{self.status.text()} Za mało próbek z pasa ({len(rows)}) — nic nie zapisano.")
            return
        self.last_belt_path = save_coach_recording(rows, self.pattern.pattern_id or "pattern", phases=self.pattern.phases)
        self.status.setText(f"{self.status.text()} Zapisano pas: {_display_path(self.last_belt_path)} ({len(rows)} próbek).")

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        if self.belt is not None:
            self.belt.stop()
        super().closeEvent(event)


def launch_coach(
    pattern: str | Path | None = None,
    *,
    lead_in_s: int = 3,
    sound: bool = True,
    belt: StrainBeltCapture | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("RespiNet Breathing Coach")
    patterns, problems = discover_patterns()
    initial = load_pattern(pattern) if pattern is not None else None
    window = CoachWindow(patterns=patterns, initial=initial, lead_in_s=lead_in_s, sound=sound, belt=belt)
    if problems:
        window.status.setText(
            "Pominięto nieprawidłowe pliki: " + "; ".join(message for _, message in problems)
        )
    window.show()
    return int(app.exec())
