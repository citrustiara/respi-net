#!/usr/bin/env python3
"""Guided, simultaneous LSM6DS3 UART and iPhone BLE chest IMU recordings.

The program is intentionally separate from the radar runners.  It guides four
supine trials with both devices fixed beside one another on the upper chest:

* two 60 s recordings: 10 s settling, 35 s natural breathing, 15 s hold;
* two 90 s recordings: the paced 2 s inhale / 3 s exhale protocol.

Each accepted trial contains two sensor CSV files, a common cue sidecar and a
manifest.  Press Escape or use ``Przerwij i odrzuć`` to discard a running
trial; after a completed recording use ``Odrzuć i powtórz`` before accepting
it.  The LSM6DS3 firmware must use the timestamped eight-field UART format.

Run from the repository root::

    uv run python tools/imu_lsm6ds3_iphone_guided_experiment.py

When automatic UART selection is ambiguous, pass the ESP32 port explicitly::

    uv run python tools/imu_lsm6ds3_iphone_guided_experiment.py \\
        --lsm-port /dev/cu.usbserial-XXXX
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import serial.tools.list_ports
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from respi_net.imu import (
    IMU_COLUMNS,
    LSM6DS3_CAPTURE_COLUMNS,
    BreathCapture,
    summarize_lsm6ds3_capture_rows,
)
from respi_net.imu_guided_protocol import (
    Cue,
    Trial,
    build_trials,
    counter_delta,
    cue_at,
    delete_trial_outputs,
    output_paths,
    sample_timing_summary,
)
from respi_net.iphone_imu import IPhoneIMUBluetoothCapture


DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "imu" / "lsm6ds3_iphone_guided"


def _lsm_port_candidates(preferred: str | None) -> list[str]:
    if preferred:
        return [preferred]
    ports = list(serial.tools.list_ports.comports())

    def score(port: Any) -> tuple[int, str]:
        text = " ".join(
            str(getattr(port, attr, "") or "")
            for attr in ("device", "description", "manufacturer", "product", "hwid")
        ).lower()
        value = 0
        if any(token in text for token in ("esp32", "ch910", "ch340", "usb single serial", "wch")):
            value += 20
        if any(token in text for token in ("a121", "interface a", "interface b", "bluetooth")):
            value -= 100
        return (-value, str(port.device))

    return [str(port.device) for port in sorted(ports, key=score)]


class MeasurementWorker(QObject):
    status_changed = Signal(str)
    prep_changed = Signal(int)
    progress_changed = Signal(float, float)
    finished = Signal(object)
    failed = Signal(str)
    aborted = Signal()

    def __init__(
        self,
        *,
        trial: Trial,
        session_dir: Path,
        lsm_port: str | None,
        lsm_baud: int,
        iphone_device: str | None,
        iphone_scan_timeout_s: float,
        prep_seconds: int,
    ) -> None:
        super().__init__()
        self.trial = trial
        self.session_dir = session_dir
        self.lsm_port = lsm_port
        self.lsm_baud = lsm_baud
        self.iphone_device = iphone_device
        self.iphone_scan_timeout_s = iphone_scan_timeout_s
        self.prep_seconds = prep_seconds
        self.paths = output_paths(session_dir, trial)
        self.abort_event = threading.Event()
        self.lsm: BreathCapture | None = None
        self.iphone: IPhoneIMUBluetoothCapture | None = None
        self.resolved_lsm_port: str | None = None

    def request_abort(self) -> None:
        self.abort_event.set()

    def _delete_outputs(self) -> None:
        delete_trial_outputs(self.paths)

    def _stop(self) -> None:
        if self.lsm is not None:
            self.lsm.stop()
        if self.iphone is not None:
            self.iphone.stop()

    def _wait_for_rows(self, capture: Any, minimum_rows: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.abort_event.is_set():
                return False
            if capture.data_count() >= minimum_rows:
                return True
            time.sleep(0.05)
        return capture.data_count() >= minimum_rows

    def _connect_lsm(self) -> None:
        failures: list[str] = []
        for candidate in _lsm_port_candidates(self.lsm_port):
            if self.abort_event.is_set():
                return
            self.status_changed.emit(f"Łączenie z LSM6DS3/ESP32: {candidate} ({self.lsm_baud} baud)…")
            capture = BreathCapture(baud=self.lsm_baud, output_dir=self.session_dir)
            if not capture.connect(candidate, exact_port=True):
                failures.append(f"{candidate}: nie można otworzyć")
                continue
            if self._wait_for_rows(capture, 12, 3.0):
                diagnostics = capture.diagnostics()
                if diagnostics["format"] == "timestamped":
                    self.lsm = capture
                    self.resolved_lsm_port = candidate
                    return
                failures.append(f"{candidate}: odebrano tylko stary format UART ({diagnostics['format']})")
            else:
                failures.append(f"{candidate}: brak poprawnych próbek")
            capture.stop()
        hint = " Podaj --lsm-port, jeśli do komputera podłączono więcej niż jeden ESP32." if self.lsm_port is None else ""
        raise RuntimeError("Nie znaleziono LSM6DS3 z nowym formatem czasu/licznika. " + "; ".join(failures) + hint)

    def _connect_iphone(self) -> None:
        self.status_changed.emit("Łączenie z aplikacją iPhone przez BLE…")
        self.iphone = IPhoneIMUBluetoothCapture(
            output_dir=self.session_dir,
            device=self.iphone_device,
            scan_timeout_s=self.iphone_scan_timeout_s,
        )
        if not self.iphone.connect(self.iphone_device):
            raise RuntimeError("Nie połączono z aplikacją RespiPhoneIMU przez BLE.")
        if not self._wait_for_rows(self.iphone, 12, 3.0):
            raise RuntimeError("iPhone połączył się, ale nie wysłał poprawnych próbek IMU.")

    def _abort(self) -> None:
        self.status_changed.emit("Przerwano próbę; pliki bieżącego zapisu są usuwane…")
        self._stop()
        self._delete_outputs()
        self.aborted.emit()

    @Slot()
    def run(self) -> None:
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._delete_outputs()
            self._connect_lsm()
            if self.abort_event.is_set():
                self._abort()
                return
            self._connect_iphone()
            if self.abort_event.is_set():
                self._abort()
                return
            self.status_changed.emit("Oba tory działają. Ustabilizuj ułożenie telefon + LSM6DS3.")
            for remaining in range(self.prep_seconds, 0, -1):
                if self.abort_event.wait(1.0):
                    self._abort()
                    return
                self.prep_changed.emit(remaining - 1)

            assert self.lsm is not None and self.iphone is not None
            lsm_start = self.lsm.data_count()
            iphone_start = self.iphone.data_count()
            iphone_counters_before = self.iphone.counter_snapshot()
            start_wall_ms = time.time() * 1000.0
            started = time.monotonic()
            while True:
                if self.abort_event.is_set():
                    self._abort()
                    return
                if not self.lsm.running:
                    raise RuntimeError("Transmisja LSM6DS3 zatrzymała się przed końcem próby.")
                if not self.iphone.running:
                    raise RuntimeError("Transmisja iPhone BLE zatrzymała się przed końcem próby.")
                elapsed_s = time.monotonic() - started
                self.progress_changed.emit(min(elapsed_s, self.trial.duration_s), self.trial.duration_s)
                if elapsed_s >= self.trial.duration_s:
                    break
                if self.abort_event.wait(0.05):
                    self._abort()
                    return
            lsm_end = self.lsm.data_count()
            iphone_end = self.iphone.data_count()
            iphone_counters_after = self.iphone.counter_snapshot()
            end_wall_ms = time.time() * 1000.0
            self.status_changed.emit("Zatrzymywanie czujników i zapisywanie danych…")
            self._stop()

            lsm_rows = self.lsm.snapshot_capture_since(lsm_start)[: lsm_end - lsm_start]
            iphone_rows = self.iphone.snapshot_data_since(iphone_start)[: iphone_end - iphone_start]
            if len(lsm_rows) < 10 or len(iphone_rows) < 10:
                raise RuntimeError(f"Za mało próbek: LSM6DS3={len(lsm_rows)}, iPhone={len(iphone_rows)}.")

            pd.DataFrame(lsm_rows, columns=LSM6DS3_CAPTURE_COLUMNS).to_csv(self.paths["lsm6ds3"], index=False)
            pd.DataFrame(iphone_rows, columns=IMU_COLUMNS).to_csv(self.paths["iphone"], index=False)
            cue_rows = [
                {
                    **cue.as_dict(),
                    "start_wall_ms": start_wall_ms + cue.start_s * 1000.0,
                    "end_wall_ms": start_wall_ms + cue.end_s * 1000.0,
                }
                for cue in self.trial.cues
            ]
            pd.DataFrame(cue_rows).to_csv(self.paths["cues"], index=False)

            lsm_summary = summarize_lsm6ds3_capture_rows(lsm_rows)
            lsm_summary["malformed_lines"] = int(self.lsm.malformed_lines)
            iphone_summary = sample_timing_summary(iphone_rows)
            iphone_summary["ble_batches"] = counter_delta(iphone_counters_after, iphone_counters_before)
            summary = {
                "trial": self.trial.as_dict(),
                "measurement_start_utc": datetime.fromtimestamp(start_wall_ms / 1000.0, tz=timezone.utc).isoformat(),
                "measurement_start_wall_ms": start_wall_ms,
                "measurement_end_wall_ms": end_wall_ms,
                "actual_seconds": (end_wall_ms - start_wall_ms) / 1000.0,
                "time_alignment": "Time_ms in both files is host-epoch time; LSM6DS3 DeviceTime_us is preserved separately.",
                "files": {key: str(value) for key, value in self.paths.items()},
                "lsm6ds3": {"port": self.resolved_lsm_port, "baud": self.lsm_baud, **lsm_summary},
                "iphone": iphone_summary,
            }
            self.finished.emit(summary)
        except Exception as exc:
            self._stop()
            self._delete_outputs()
            if self.abort_event.is_set():
                self.aborted.emit()
            else:
                self.failed.emit(str(exc))


def format_summary(summary: dict[str, Any]) -> str:
    lsm = summary.get("lsm6ds3", {})
    iphone = summary.get("iphone", {})
    batches = iphone.get("ble_batches", {}) if isinstance(iphone, dict) else {}
    return "\n".join(
        [
            f"LSM6DS3: {lsm.get('rows', 0)} próbek, fs urządzenia {lsm.get('device_sample_rate_hz') or 0:.1f} Hz, "
            f"luki z licznika {lsm.get('missing_samples', 0)} ({lsm.get('missing_percent', 0):.3f}%).",
            f"iPhone: {iphone.get('rows', 0)} próbek, fs {iphone.get('sample_rate_hz') or 0:.1f} Hz, "
            f"szacowane luki czasowe {iphone.get('estimated_missing_samples', 0)} "
            f"({iphone.get('estimated_missing_percent', 0):.3f}%).",
            f"BLE: brakujące pakiety={batches.get('missing_batches', 0)}, niepoprawne={batches.get('invalid_batches', 0)}, "
            f"duplikaty={batches.get('duplicate_batches', 0)}, resety sekwencji={batches.get('sequence_resets', 0)}.",
            *[f"{name}: {path}" for name, path in summary.get("files", {}).items()],
        ]
    )


class GuidedImuWindow(QWidget):
    def __init__(
        self,
        *,
        trials: list[Trial],
        session_dir: Path,
        lsm_port: str | None,
        lsm_baud: int,
        iphone_device: str | None,
        iphone_scan_timeout_s: float,
        prep_seconds: int,
        start_trial: int,
    ) -> None:
        super().__init__()
        self.trials = trials
        self.session_dir = session_dir
        self.lsm_port = lsm_port
        self.lsm_baud = lsm_baud
        self.iphone_device = iphone_device
        self.iphone_scan_timeout_s = iphone_scan_timeout_s
        self.prep_seconds = prep_seconds
        self.current_index = start_trial - 1
        self.active_worker: MeasurementWorker | None = None
        self.active_thread: QThread | None = None
        self.pending_summary: dict[str, Any] | None = None
        self.current_cue_window: tuple[float, float] | None = None
        self.manifest_path = session_dir / "manifest.json"
        self.manifest = self._load_manifest()

        self.setWindowTitle("LSM6DS3 + iPhone — prowadzony pomiar na plecach")
        self.resize(860, 700)
        self._build_ui()
        self.show_trial()
        self.abort_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.abort_shortcut.activated.connect(self.abort_current)

    def _load_manifest(self) -> dict[str, Any]:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        manifest = {
            "experiment": "Simultaneous LSM6DS3 UART and iPhone BLE chest IMU comparison",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "position": "supine; iPhone and LSM6DS3 adjacent on the upper chest, same orientation",
            "trials": [trial.as_dict() for trial in self.trials],
            "measurements": [],
        }
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: dict[str, Any] | None = None) -> None:
        payload = manifest if manifest is not None else self.manifest
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.trial_label = QLabel()
        self.trial_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self.trial_label)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.instructions.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.instructions)

        cue_box = QGroupBox("Wspólne komendy oddechowe")
        cue_layout = QVBoxLayout(cue_box)
        self.cue_label = QLabel()
        self.cue_label.setAlignment(Qt.AlignCenter)
        self.cue_detail = QLabel()
        self.cue_detail.setAlignment(Qt.AlignCenter)
        self.cue_detail.setWordWrap(True)
        self.cue_timer = QLabel()
        self.cue_timer.setAlignment(Qt.AlignCenter)
        cue_layout.addWidget(self.cue_label)
        cue_layout.addWidget(self.cue_detail)
        cue_layout.addWidget(self.cue_timer)
        layout.addWidget(cue_box)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.progress)
        self.status = QLabel("Gotowy.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QGridLayout()
        self.start_button = QPushButton("Start próby")
        self.abort_button = QPushButton("Przerwij i odrzuć (Esc)")
        self.keep_button = QPushButton("Zachowaj próbę")
        self.discard_button = QPushButton("Odrzuć i powtórz")
        self.abort_button.setStyleSheet("background: #9b1c1c; color: white; font-weight: bold;")
        self.discard_button.setStyleSheet("background: #9b1c1c; color: white; font-weight: bold;")
        buttons.addWidget(self.start_button, 0, 0)
        buttons.addWidget(self.abort_button, 0, 1)
        buttons.addWidget(self.keep_button, 1, 0)
        buttons.addWidget(self.discard_button, 1, 1)
        layout.addLayout(buttons)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary, stretch=1)

        self.start_button.clicked.connect(self.start_measurement)
        self.abort_button.clicked.connect(self.abort_current)
        self.keep_button.clicked.connect(self.accept_pending)
        self.discard_button.clicked.connect(self.discard_pending)

    def current_trial(self) -> Trial | None:
        return self.trials[self.current_index] if 0 <= self.current_index < len(self.trials) else None

    def _show_cue(self, title: str, detail: str, timer: str, kind: str) -> None:
        colours = {
            "settle": ("#eeeeee", "#222222"),
            "normal": ("#eeeeee", "#222222"),
            "inhale": ("#cfe8ff", "#063b66"),
            "exhale": ("#dcf5df", "#174d20"),
            "hold": ("#ffe0e0", "#6d1111"),
        }
        background, foreground = colours.get(kind, colours["normal"])
        self.cue_label.setText(title)
        self.cue_label.setStyleSheet(
            "font-size: 30px; font-weight: bold; padding: 12px; "
            f"background: {background}; color: {foreground}; border-radius: 6px;"
        )
        self.cue_detail.setText(detail)
        self.cue_timer.setText(timer)

    def show_trial(self, message: str | None = None) -> None:
        trial = self.current_trial()
        self.pending_summary = None
        self.current_cue_window = None
        self.progress.setValue(0)
        self.summary.clear()
        self.abort_button.setEnabled(False)
        self.keep_button.setEnabled(False)
        self.discard_button.setEnabled(False)
        if trial is None:
            self.trial_label.setText("Wszystkie cztery próby są zakończone")
            self.instructions.setText(f"Zaakceptowane pliki i manifest znajdują się w:\n{self.session_dir}")
            self.start_button.setEnabled(False)
            self.status.setText("Gotowe.")
            self._show_cue("KONIEC BADANIA", "", "", "normal")
            return
        self.trial_label.setText(f"Próba {trial.number}/4 — {trial.label}")
        self.instructions.setText(
            "Pozycja: leżenie na plecach. Telefon i LSM6DS3 umieść obok siebie na górnej części klatki "
            "piersiowej, w tej samej orientacji; nie zmieniaj położenia w trakcie próby.\n\n"
            f"Czas rejestracji: {trial.duration_s:.0f} s. Oba tory otrzymają te same komendy zapisane potem w pliku cues.csv."
        )
        first = trial.cues[0]
        self._show_cue(first.title, first.detail, "", first.kind)
        self.status.setText(message or "Sprawdź ułożenie czujników, uruchom aplikację RespiPhoneIMU i wybierz Start próby.")
        self.start_button.setEnabled(True)

    @Slot()
    def start_measurement(self) -> None:
        trial = self.current_trial()
        if trial is None or self.active_thread is not None:
            return
        self.start_button.setEnabled(False)
        self.abort_button.setEnabled(True)
        worker = MeasurementWorker(
            trial=trial,
            session_dir=self.session_dir,
            lsm_port=self.lsm_port,
            lsm_baud=self.lsm_baud,
            iphone_device=self.iphone_device,
            iphone_scan_timeout_s=self.iphone_scan_timeout_s,
            prep_seconds=self.prep_seconds,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status_changed.connect(self.status.setText)
        worker.prep_changed.connect(self.on_preparation)
        worker.progress_changed.connect(self.on_progress)
        worker.finished.connect(self.on_finished)
        worker.failed.connect(self.on_failed)
        worker.aborted.connect(self.on_aborted)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.aborted.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self.on_thread_finished)
        self.active_worker = worker
        self.active_thread = thread
        thread.start()

    @Slot(int)
    def on_preparation(self, remaining: int) -> None:
        self._show_cue("PRZYGOTUJ SIĘ", "Ustabilizuj ułożenie czujników i ciała.", f"Start za {remaining} s", "settle")

    @Slot(float, float)
    def on_progress(self, elapsed_s: float, total_s: float) -> None:
        trial = self.current_trial()
        if trial is None:
            return
        state = cue_at(trial.cues, elapsed_s)
        if state is not None:
            cue, remaining = state
            key = (cue.start_s, cue.end_s)
            if key != self.current_cue_window:
                self.current_cue_window = key
                QApplication.beep()
            self._show_cue(cue.title, cue.detail, f"Jeszcze {remaining:.1f} s • pomiar {elapsed_s:.1f}/{total_s:.0f} s", cue.kind)
        self.progress.setValue(int(round(1000 * elapsed_s / max(total_s, 1e-9))))

    @Slot(object)
    def on_finished(self, result: object) -> None:
        self.pending_summary = dict(result)  # type: ignore[arg-type]
        self.progress.setValue(1000)
        self.abort_button.setEnabled(False)
        self.keep_button.setEnabled(True)
        self.discard_button.setEnabled(True)
        self.summary.setPlainText(format_summary(self.pending_summary))
        self.status.setText("Próba zapisana tymczasowo. Zachowaj ją albo odrzuć i powtórz.")
        self._show_cue("KONIEC", "Oddychaj swobodnie.", "", "normal")

    @Slot(str)
    def on_failed(self, message: str) -> None:
        self.abort_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.status.setText(f"Próba nieudana: {message}")
        self.summary.setPlainText("Bieżące pliki usunięto. Popraw połączenie lub ustawienie i uruchom próbę ponownie.")

    @Slot()
    def on_aborted(self) -> None:
        self.abort_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.status.setText("Bieżąca próba została odrzucona, a jej pliki usunięte.")
        self.summary.setPlainText("Przerwany zapis został usunięty.")

    @Slot()
    def on_thread_finished(self) -> None:
        thread = self.active_thread
        self.active_worker = None
        self.active_thread = None
        if thread is not None:
            thread.deleteLater()

    @Slot()
    def abort_current(self) -> None:
        if self.active_worker is None:
            return
        self.abort_button.setEnabled(False)
        self.status.setText("Przerywanie i usuwanie bieżącej próby…")
        self.active_worker.request_abort()

    @Slot()
    def discard_pending(self) -> None:
        if self.pending_summary is None:
            return
        delete_trial_outputs({key: Path(path) for key, path in self.pending_summary.get("files", {}).items()})
        self.show_trial("Ukończoną próbę odrzucono. Ustaw stanowisko i uruchom ją ponownie.")

    @Slot()
    def accept_pending(self) -> None:
        if self.pending_summary is None:
            return
        entry = dict(self.pending_summary)
        entry["status"] = "accepted"
        entry["accepted_at"] = datetime.now(timezone.utc).isoformat()
        number = int(entry["trial"]["number"])
        entries = [item for item in self.manifest.get("measurements", []) if int(item.get("trial", {}).get("number", -1)) != number]
        entries.append(entry)
        entries.sort(key=lambda item: int(item["trial"]["number"]))
        self.manifest["measurements"] = entries
        self._save_manifest()
        self.current_index += 1
        self.show_trial("Próba została zachowana w manifeście.")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if self.active_worker is not None:
            self.active_worker.request_abort()
        if self.active_thread is not None:
            self.active_thread.wait(6000)
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lsm-port", help="Port szeregowy ESP32 z LSM6DS3; bez niego program próbuje wykryć go automatycznie.")
    parser.add_argument("--lsm-baud", type=int, default=921600, help="Baudrate firmware LSM6DS3 (domyślnie 921600).")
    parser.add_argument("--iphone-device", help="Opcjonalna nazwa/adres BLE iPhone'a; domyślnie RespiPhoneIMU.")
    parser.add_argument("--iphone-scan-timeout", type=float, default=10.0, help="Czas skanowania BLE [s].")
    parser.add_argument("--prep-seconds", type=int, default=5, help="Czas ustabilizowania po połączeniu czujników [s].")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Katalog sesji IMU.")
    parser.add_argument("--session-name", help="Nazwa katalogu sesji; domyślnie czas uruchomienia.")
    parser.add_argument("--start-trial", type=int, choices=range(1, 5), default=1, help="Numer próby, od której rozpocząć.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    session_name = args.session_name or f"imu_chest_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    session_dir = Path(args.output_dir) / _safe_name(session_name)
    app = QApplication.instance() or QApplication(sys.argv)
    window = GuidedImuWindow(
        trials=build_trials(),
        session_dir=session_dir,
        lsm_port=args.lsm_port,
        lsm_baud=args.lsm_baud,
        iphone_device=args.iphone_device,
        iphone_scan_timeout_s=args.iphone_scan_timeout,
        prep_seconds=max(0, args.prep_seconds),
        start_trial=args.start_trial,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
