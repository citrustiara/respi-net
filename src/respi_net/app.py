from __future__ import annotations

import csv
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.signal import welch

from .a121 import A121_COLUMNS, A121Config, A121Capture, parse_json_array
from .a121_vitals import (
    A121_RATE_WINDOW_S,
    HEART_BAND_HZ,
    RESP_BAND_HZ,
    RESP_RATE_CONFIDENCE_MIN,
    A121LiveTraceProcessor,
    HeartRateKalmanTracker,
    analyze_a121_vitals,
    bandpass_filter,
    clean_signal,
)
from .imu import IMU_COLUMNS, BreathCapture
from .paths import DATA_DIR, RAW_A121_DIR, RAW_IMU_DIR, RAW_RADAR_DIR
from .radar import DOPPLER_HZ_PER_MPS, RADAR_COLUMNS, RadarCapture
from .serial_utils import list_serial_ports

DB_PATH = DATA_DIR / "respi_recordings.sqlite3"


class RecordingStore:
    """Small SQLite store for live sessions and their raw samples."""

    def __init__(self, path: str | Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_schema(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sensor TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    stopped_at TEXT,
                    csv_path TEXT,
                    sample_count INTEGER DEFAULT 0,
                    stats_json TEXT DEFAULT '{}'
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS radar_samples (
                    session_id INTEGER NOT NULL,
                    timestamp_ms REAL NOT NULL,
                    raw_adc REAL NOT NULL,
                    voltage_mv REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS imu_samples (
                    session_id INTEGER NOT NULL,
                    time_ms REAL NOT NULL,
                    ax REAL NOT NULL,
                    ay REAL NOT NULL,
                    az REAL NOT NULL,
                    gx REAL NOT NULL,
                    gy REAL NOT NULL,
                    gz REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS a121_samples (
                    session_id INTEGER NOT NULL,
                    timestamp_ms REAL NOT NULL,
                    frame INTEGER NOT NULL,
                    peak_distance_m REAL NOT NULL,
                    peak_amplitude REAL NOT NULL,
                    peak_phase_rad REAL NOT NULL,
                    mean_amplitude REAL NOT NULL,
                    distances_m TEXT NOT NULL,
                    amplitude TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    real TEXT NOT NULL,
                    imag TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_radar_session ON radar_samples(session_id)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_imu_session ON imu_samples(session_id)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_a121_session ON a121_samples(session_id)")

    def create_session(self, sensor: str, csv_path: Path | None) -> int:
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO sessions(sensor, started_at, csv_path) VALUES (?, ?, ?)",
                (sensor, datetime.now().isoformat(timespec="seconds"), str(csv_path) if csv_path else None),
            )
            return int(cur.lastrowid)

    def append_samples(self, sensor: str, session_id: int, rows: list[list[float]]) -> None:
        if not rows:
            return
        with self._connect() as db:
            if sensor == "radar":
                db.executemany(
                    "INSERT INTO radar_samples(session_id, timestamp_ms, raw_adc, voltage_mv) VALUES (?, ?, ?, ?)",
                    [(session_id, row[0], row[1], row[2]) for row in rows],
                )
            elif sensor == "a121":
                db.executemany(
                    """
                    INSERT INTO a121_samples(
                        session_id, timestamp_ms, frame, peak_distance_m, peak_amplitude,
                        peak_phase_rad, mean_amplitude, distances_m, amplitude, phase, real, imag
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(session_id, *row) for row in rows],
                )
            else:
                db.executemany(
                    """
                    INSERT INTO imu_samples(session_id, time_ms, ax, ay, az, gx, gy, gz)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(session_id, *row) for row in rows],
                )

    def finish_session(self, session_id: int, sample_count: int, stats: dict[str, float | str]) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE sessions SET stopped_at = ?, sample_count = ?, stats_json = ? WHERE id = ?",
                (datetime.now().isoformat(timespec="seconds"), sample_count, json.dumps(stats), session_id),
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT id, sensor, started_at, stopped_at, csv_path, sample_count, stats_json "
                "FROM sessions ORDER BY id DESC LIMIT 500"
            ).fetchall()
        return [dict(row) for row in rows]

    def load_session(self, session_id: int, sensor: str) -> pd.DataFrame:
        with self._connect() as db:
            if sensor == "radar":
                return pd.read_sql_query(
                    "SELECT timestamp_ms AS Timestamp_ms, raw_adc AS RawADC, voltage_mv AS Voltage_mV "
                    "FROM radar_samples WHERE session_id = ? ORDER BY rowid",
                    db,
                    params=(session_id,),
                )
            if sensor == "a121":
                return pd.read_sql_query(
                    """
                    SELECT timestamp_ms AS Timestamp_ms, frame AS Frame,
                           peak_distance_m AS PeakDistance_m, peak_amplitude AS PeakAmplitude,
                           peak_phase_rad AS PeakPhase_rad, mean_amplitude AS MeanAmplitude,
                           distances_m AS Distances_m, amplitude AS Amplitude,
                           phase AS Phase, real AS Real, imag AS Imag
                    FROM a121_samples WHERE session_id = ? ORDER BY rowid
                    """,
                    db,
                    params=(session_id,),
                )
            return pd.read_sql_query(
                "SELECT time_ms AS Time_ms, ax, ay, az, gx, gy, gz FROM imu_samples "
                "WHERE session_id = ? ORDER BY rowid",
                db,
                params=(session_id,),
            )


def _sample_rate(times_ms: np.ndarray, default: float) -> float:
    if len(times_ms) < 2:
        return default
    diffs = np.diff(times_ms) / 1000.0
    diffs = diffs[diffs > 0]
    return float(1.0 / np.mean(diffs)) if len(diffs) else default


def _radar_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"sample_rate_hz": 0.0, "peak_hz": 0.0, "resp_bpm": 0.0, "speed_mps": 0.0}
    fs = _sample_rate(df["Timestamp_ms"].to_numpy(dtype=float), 500.0)
    voltage = df["Voltage_mV"].to_numpy(dtype=float)
    peak_hz = 0.0
    resp_bpm = 0.0
    if len(voltage) > 32:
        window = voltage[-4096:] if len(voltage) > 4096 else voltage
        window = (window - np.mean(window)) * np.hanning(len(window))
        spectrum = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(len(window), d=1.0 / max(fs, 1e-9))
        valid = freqs >= 0.05
        if np.any(valid):
            idx = int(np.argmax(spectrum[valid]))
            peak_hz = float(freqs[valid][idx])
        resp_valid = (freqs >= 0.1) & (freqs <= 0.6)
        if np.any(resp_valid):
            idx = int(np.argmax(spectrum[resp_valid]))
            resp_bpm = float(freqs[resp_valid][idx] * 60.0)
    return {"sample_rate_hz": fs, "peak_hz": peak_hz, "resp_bpm": resp_bpm, "speed_mps": peak_hz / DOPPLER_HZ_PER_MPS}


def _a121_stats(df: pd.DataFrame) -> dict[str, float]:
    analysis = analyze_a121_vitals(df, max_frames=1200)
    return {
        "sample_rate_hz": analysis.sample_rate_hz,
        "peak_distance_m": analysis.peak_distance_m,
        "peak_amplitude": analysis.peak_amplitude,
        "mean_amplitude": analysis.mean_amplitude,
        "target_distance_m": analysis.target_distance_m,
        "gate_min_m": analysis.gate_min_m,
        "gate_max_m": analysis.gate_max_m,
        "presence_score": analysis.presence_score,
        "present": 1.0 if analysis.present else 0.0,
        "resp_bpm": analysis.resp_bpm,
        "heart_bpm": analysis.heart_bpm,
    }


def _imu_stats(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"sample_rate_hz": 0.0, "accel_rms_g": 0.0, "gyro_rms_dps": 0.0, "resp_bpm": 0.0, "heart_bpm": 0.0}
    fs = _sample_rate(df["Time_ms"].to_numpy(dtype=float), 100.0)
    accel = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    gyro = df[["gx", "gy", "gz"]].to_numpy(dtype=float)
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    resp_bpm = 0.0
    heart_bpm = 0.0
    if len(accel_mag) > 32:
        signal = (accel_mag - np.mean(accel_mag)) * np.hanning(len(accel_mag))
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / max(fs, 1e-9))
        resp_valid = (freqs >= 0.1) & (freqs <= 0.6)
        heart_valid = (freqs >= 0.65) & (freqs <= 4.0)
        if np.any(resp_valid):
            resp_bpm = float(freqs[resp_valid][int(np.argmax(spectrum[resp_valid]))] * 60.0)
        if np.any(heart_valid):
            heart_bpm = float(freqs[heart_valid][int(np.argmax(spectrum[heart_valid]))] * 60.0)
    return {
        "sample_rate_hz": fs,
        "accel_rms_g": float(np.sqrt(np.mean(np.square(accel_mag)))) if len(accel_mag) else 0.0,
        "gyro_rms_dps": float(np.sqrt(np.mean(np.square(gyro_mag)))) if len(gyro_mag) else 0.0,
        "resp_bpm": resp_bpm,
        "heart_bpm": heart_bpm,
    }


def _detect_sensor(df: pd.DataFrame) -> str:
    if all(col in df.columns for col in RADAR_COLUMNS):
        return "radar"
    if all(col in df.columns for col in A121_COLUMNS):
        return "a121"
    if all(col in df.columns for col in IMU_COLUMNS):
        return "imu"
    raise ValueError("CSV does not contain recognized HB100 radar, A121 radar, or IMU columns.")


class A121AnalysisThread(QtCore.QThread):
    analysis_completed = QtCore.Signal(int, object)

    def __init__(
        self,
        request_id: int,
        rows: list[list[Any]],
        cutoff_ms: float,
        gate_half_width: float,
        target_distance_m: float | None,
        heart_prior_hz: float | None,
        heart_prior_std_hz: float | None,
        use_gating: bool,
        auto_gate: bool,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.rows = rows
        self.cutoff_ms = cutoff_ms
        self.gate_half_width = gate_half_width
        self.target_distance_m = target_distance_m
        self.heart_prior_hz = heart_prior_hz
        self.heart_prior_std_hz = heart_prior_std_hz
        self.use_gating = use_gating
        self.auto_gate = auto_gate

    def run(self) -> None:
        try:
            analysis_rows = [row for row in self.rows if float(row[0]) >= self.cutoff_ms]
            if not analysis_rows and self.rows:
                analysis_rows = [self.rows[-1]]
            df = pd.DataFrame(analysis_rows, columns=A121_COLUMNS)
            analysis = analyze_a121_vitals(
                df,
                auto_gate=self.auto_gate,
                gate_half_width_m=self.gate_half_width,
                max_frames=len(df),
                target_distance_m=self.target_distance_m,
                heart_prior_hz=self.heart_prior_hz,
                heart_prior_std_hz=self.heart_prior_std_hz,
                use_gating=self.use_gating,
            )
            self.analysis_completed.emit(self.request_id, analysis)
        except Exception as exc:
            print(f"A121 analysis failed: {exc}", file=sys.stderr)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, default_sensor: str = "radar", default_port: str | None = None, default_baud: int = 921600) -> None:
        super().__init__()
        pg.setConfigOptions(antialias=False, background="#111827", foreground="#e5e7eb")
        self.setWindowTitle("RespiNet Sensor Studio")
        self.resize(1600, 1000)
        self.setMinimumSize(1350, 860)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1220; color: #e5e7eb; }
            QFrame { background: #111827; border: 1px solid #263244; border-radius: 10px; }
            QLabel { color: #e5e7eb; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget {
                background: #0f172a; color: #e5e7eb; border: 1px solid #334155; border-radius: 6px; padding: 4px;
            }
            QPushButton { background: #1d4ed8; color: white; border: 0; border-radius: 7px; padding: 8px; font-weight: 600; }
            QPushButton:hover { background: #2563eb; }
            QPushButton:disabled { background: #374151; color: #9ca3af; }
            QTabWidget::pane { border: 1px solid #263244; border-radius: 8px; }
            QTabBar::tab { background: #111827; color: #cbd5e1; padding: 8px 14px; border-top-left-radius: 7px; border-top-right-radius: 7px; }
            QTabBar::tab:selected { background: #1e293b; color: #ffffff; }
            """
        )
        self.default_sensor = default_sensor.lower()
        self.default_port = default_port
        self.default_baud = default_baud

        self.store = RecordingStore()
        self.capture: RadarCapture | A121Capture | BreathCapture | None = None
        self.active_sensor = "radar"
        self.session_id: int | None = None
        self.persisted_index = 0
        self.csv_file: Any | None = None
        self.csv_writer: csv.writer | None = None
        self.csv_path: Path | None = None
        self.last_persist_monotonic = 0.0
        self.persist_interval_s = 0.75
        self.last_a121_analysis_monotonic = 0.0
        self.cached_a121_analysis: Any | None = None
        self.a121_gate_center_m: float | None = None
        self.last_a121_gate_update_monotonic = 0.0
        self.a121_heart_tracker = HeartRateKalmanTracker()
        self.a121_live_trace = A121LiveTraceProcessor()
        self.last_a121_tracker_update_monotonic = 0.0
        self.a121_analysis_thread = None
        self.a121_analysis_request_id = 0
        self.a121_analysis_interval_s = 1.0
        self.a121_resp_display_hz = 0.0
        self.a121_resp_pending_hz = 0.0
        self.a121_resp_pending_count = 0
        self.a121_resp_missed_count = 0
        self.a121_lock_bad_count = 0

        self._build_ui()
        self._build_menu()
        self._refresh_ports()
        self._apply_startup_defaults()
        self._refresh_recordings()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)

        side = QtWidgets.QFrame()
        side.setFixedWidth(520)
        side.setMinimumWidth(520)
        side.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        side_layout = QtWidgets.QVBoxLayout(side)

        title = QtWidgets.QLabel("RespiNet Studio")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        side_layout.addWidget(title)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
        form.setHorizontalSpacing(14)
        self.sensor_combo = QtWidgets.QComboBox()
        self.sensor_combo.addItems(["HB100 Radar", "A121 Radar", "IMU"])
        self.sensor_combo.currentTextChanged.connect(self._sensor_changed)
        form.addRow("Sensor", self.sensor_combo)

        port_row = QtWidgets.QHBoxLayout()
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_ports_btn = QtWidgets.QPushButton("↻")
        self.refresh_ports_btn.setFixedWidth(34)
        self.refresh_ports_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(self.refresh_ports_btn)
        form.addRow("Port", port_row)

        self.baud_spin = QtWidgets.QSpinBox()
        self.baud_spin.setRange(9600, 2_000_000)
        self.baud_spin.setValue(self.default_baud)
        self.baud_spin.setSingleStep(115200)
        self.baud_spin.setMinimumWidth(220)
        form.addRow("Baud", self.baud_spin)

        self.storage_combo = QtWidgets.QComboBox()
        self.storage_combo.addItems(["CSV + SQLite", "CSV only", "SQLite only"])
        form.addRow("Record to", self.storage_combo)

        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(2, 120)
        self.window_spin.setValue(10)
        self.window_spin.setSuffix(" s")
        self.window_spin.setMinimumWidth(220)
        self.window_spin.valueChanged.connect(lambda _value: self._invalidate_a121_analysis())
        form.addRow("Live window", self.window_spin)

        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["Vitals (filtered)", "Rate FFT", "Raw signal"])
        self.view_combo.currentTextChanged.connect(self._configure_live_plots)
        form.addRow("View", self.view_combo)

        self.a121_use_gating_check = QtWidgets.QCheckBox("enable range gating")
        self.a121_use_gating_check.setChecked(False)
        self.a121_use_gating_check.toggled.connect(self._configure_gating_ui)
        form.addRow("A121 range gating", self.a121_use_gating_check)

        self.a121_show_gate_check = QtWidgets.QCheckBox("show gate/target on graph")
        self.a121_show_gate_check.setChecked(False)
        form.addRow("A121 gate display", self.a121_show_gate_check)

        self.a121_auto_gate_check = QtWidgets.QCheckBox("reacquire if peak leaves gate (≥10 s)")
        self.a121_auto_gate_check.setChecked(False)
        self.a121_auto_gate_check.toggled.connect(lambda _checked: self._invalidate_a121_analysis(reset_tracker=True))
        form.addRow("A121 gate update", self.a121_auto_gate_check)

        self.a121_gate_spin = QtWidgets.QDoubleSpinBox()
        self.a121_gate_spin.setRange(0.02, 1.00)
        self.a121_gate_spin.setDecimals(2)
        self.a121_gate_spin.setSingleStep(0.01)
        self.a121_gate_spin.setValue(0.05)
        self.a121_gate_spin.setSuffix(" m")
        self.a121_gate_spin.setMinimumWidth(220)
        self.a121_gate_spin.valueChanged.connect(lambda _value: self._invalidate_a121_analysis(reset_tracker=True))
        form.addRow("A121 gate", self.a121_gate_spin)

        self.a121_start_spin = QtWidgets.QDoubleSpinBox()
        self.a121_start_spin.setRange(0.03, 10.0)
        self.a121_start_spin.setDecimals(2)
        self.a121_start_spin.setSingleStep(0.05)
        self.a121_start_spin.setValue(0.20)
        self.a121_start_spin.setSuffix(" m")
        self.a121_start_spin.setMinimumWidth(220)
        form.addRow("A121 start", self.a121_start_spin)

        self.a121_end_spin = QtWidgets.QDoubleSpinBox()
        self.a121_end_spin.setRange(0.05, 10.0)
        self.a121_end_spin.setDecimals(2)
        self.a121_end_spin.setSingleStep(0.05)
        self.a121_end_spin.setValue(1.00)
        self.a121_end_spin.setSuffix(" m")
        self.a121_end_spin.setMinimumWidth(220)
        form.addRow("A121 end", self.a121_end_spin)

        self.a121_profile_combo = QtWidgets.QComboBox()
        self.a121_profile_combo.addItems(["1", "2", "3", "4", "5"])
        self.a121_profile_combo.setCurrentText("3")
        form.addRow("A121 profile", self.a121_profile_combo)

        self.a121_hwaas_spin = QtWidgets.QSpinBox()
        self.a121_hwaas_spin.setRange(1, 511)
        self.a121_hwaas_spin.setValue(32)
        self.a121_hwaas_spin.setMinimumWidth(220)
        form.addRow("A121 HWAAS", self.a121_hwaas_spin)

        self.a121_sweeps_spin = QtWidgets.QSpinBox()
        self.a121_sweeps_spin.setRange(1, 128)
        self.a121_sweeps_spin.setValue(16)
        self.a121_sweeps_spin.setMinimumWidth(220)
        form.addRow("A121 sweeps", self.a121_sweeps_spin)

        self.a121_frame_rate_spin = QtWidgets.QDoubleSpinBox()
        self.a121_frame_rate_spin.setRange(5.0, 100.0)
        self.a121_frame_rate_spin.setDecimals(1)
        self.a121_frame_rate_spin.setSingleStep(5.0)
        self.a121_frame_rate_spin.setValue(20.0)
        self.a121_frame_rate_spin.setSuffix(" Hz")
        self.a121_frame_rate_spin.setMinimumWidth(220)
        form.addRow("A121 fps", self.a121_frame_rate_spin)
        side_layout.addLayout(form)

        self.start_btn = QtWidgets.QPushButton("Start recording")
        self.start_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay))
        self.start_btn.clicked.connect(self._start_recording)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaStop))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_recording)
        side_layout.addWidget(self.start_btn)
        side_layout.addWidget(self.stop_btn)

        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setWordWrap(True)
        side_layout.addWidget(self.status_label)

        self.stats_box = QtWidgets.QTextEdit()
        self.stats_box.setReadOnly(True)
        self.stats_box.setMinimumHeight(160)
        self.stats_box.setText("Live stats will appear here.")
        side_layout.addWidget(self.stats_box)
        side_layout.addStretch(1)

        hint = QtWidgets.QLabel("Graph controls:\n• drag to pan\n• mouse wheel to zoom\n• right-click for plot menu")
        hint.setStyleSheet("color: #9ca3af;")
        side_layout.addWidget(hint)

        self.tabs = QtWidgets.QTabWidget()
        self.live_tab = QtWidgets.QWidget()
        live_layout = QtWidgets.QVBoxLayout(self.live_tab)
        self.live_graph = pg.GraphicsLayoutWidget()
        self.live_plot_a = self.live_graph.addPlot(row=0, col=0)
        self.live_plot_b = self.live_graph.addPlot(row=1, col=0)
        self.live_plot_c = self.live_graph.addPlot(row=2, col=0)
        live_layout.addWidget(self.live_graph)
        self.tabs.addTab(self.live_tab, "Live")

        self.history_tab = QtWidgets.QWidget()
        hist_layout = QtWidgets.QHBoxLayout(self.history_tab)
        left_hist = QtWidgets.QVBoxLayout()
        self.recording_table = QtWidgets.QTableWidget(0, 5)
        self.recording_table.setHorizontalHeaderLabels(["Source", "Sensor", "Started/File", "Samples", "Path/ID"])
        self.recording_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.recording_table.horizontalHeader().setStretchLastSection(True)
        self.recording_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.recording_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.recording_table.doubleClicked.connect(self._open_selected_recording)
        left_hist.addWidget(self.recording_table)
        btn_row = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open")
        self.open_btn.clicked.connect(self._open_selected_recording)
        self.refresh_recordings_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_recordings_btn.clicked.connect(self._refresh_recordings)
        self.open_file_btn = QtWidgets.QPushButton("Open CSV…")
        self.open_file_btn.clicked.connect(self._open_csv_dialog)
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.refresh_recordings_btn)
        btn_row.addWidget(self.open_file_btn)
        left_hist.addLayout(btn_row)
        hist_layout.addLayout(left_hist, 1)

        right_hist = QtWidgets.QVBoxLayout()
        self.history_graph = pg.GraphicsLayoutWidget()
        self.history_plot_a = self.history_graph.addPlot(row=0, col=0)
        self.history_plot_b = self.history_graph.addPlot(row=1, col=0)
        self.history_plot_c = self.history_graph.addPlot(row=2, col=0)
        right_hist.addWidget(self.history_graph, 4)
        self.history_stats = QtWidgets.QTextEdit()
        self.history_stats.setReadOnly(True)
        self.history_stats.setMaximumHeight(150)
        right_hist.addWidget(self.history_stats)
        hist_layout.addLayout(right_hist, 2)
        self.tabs.addTab(self.history_tab, "Recordings")

        root_layout.addWidget(side)
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self._configure_live_plots()
        self._configure_gating_ui()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_csv = QtGui.QAction("Open CSV…", self)
        open_csv.triggered.connect(self._open_csv_dialog)
        file_menu.addAction(open_csv)
        file_menu.addSeparator()
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        capture_menu = self.menuBar().addMenu("Capture")
        start_action = QtGui.QAction("Start", self)
        start_action.triggered.connect(self._start_recording)
        capture_menu.addAction(start_action)
        stop_action = QtGui.QAction("Stop", self)
        stop_action.triggered.connect(self._stop_recording)
        capture_menu.addAction(stop_action)

        view_menu = self.menuBar().addMenu("View")
        refresh_ports = QtGui.QAction("Refresh ports", self)
        refresh_ports.triggered.connect(self._refresh_ports)
        view_menu.addAction(refresh_ports)
        refresh_recordings = QtGui.QAction("Refresh recordings", self)
        refresh_recordings.triggered.connect(self._refresh_recordings)
        view_menu.addAction(refresh_recordings)

    def _sensor_changed(self, text: str) -> None:
        label = text.lower()
        if "a121" in label:
            self.active_sensor = "a121"
        elif "imu" in label:
            self.active_sensor = "imu"
        else:
            self.active_sensor = "radar"
        self._configure_live_plots()

    def _reset_a121_rate_display(self) -> None:
        self.a121_resp_display_hz = 0.0
        self.a121_resp_pending_hz = 0.0
        self.a121_resp_pending_count = 0
        self.a121_resp_missed_count = 0

    def _update_a121_resp_display(self, analysis: Any) -> float:
        measured_hz = float(getattr(analysis, "resp_hz", 0.0) or 0.0)
        confidence = float(getattr(analysis, "resp_confidence", 0.0) or 0.0)
        quality = float(getattr(analysis, "signal_quality", 0.0) or 0.0)
        present = bool(getattr(analysis, "present", False))
        valid = bool(
            present
            and RESP_BAND_HZ[0] <= measured_hz <= RESP_BAND_HZ[1]
            and confidence >= RESP_RATE_CONFIDENCE_MIN
            and quality >= 0.25
        )
        if not present:
            self._reset_a121_rate_display()
            return 0.0
        if not valid:
            self.a121_resp_missed_count += 1
            if self.a121_resp_missed_count > 40:
                self.a121_resp_display_hz = 0.0
                self.a121_resp_pending_count = 0
            return self.a121_resp_display_hz

        self.a121_resp_missed_count = 0
        if self.a121_resp_display_hz <= 0:
            if abs(measured_hz - self.a121_resp_pending_hz) <= 0.06:
                self.a121_resp_pending_count += 1
            else:
                self.a121_resp_pending_hz = measured_hz
                self.a121_resp_pending_count = 1
            if self.a121_resp_pending_count >= 2 or confidence >= 8.0:
                self.a121_resp_display_hz = measured_hz
                self.a121_resp_pending_count = 0
            return self.a121_resp_display_hz

        if abs(measured_hz - self.a121_resp_display_hz) <= 0.10:
            alpha = 0.20 if confidence >= 8.0 else 0.10
            self.a121_resp_display_hz = (1.0 - alpha) * self.a121_resp_display_hz + alpha * measured_hz
            self.a121_resp_pending_count = 0
            return self.a121_resp_display_hz

        if abs(measured_hz - self.a121_resp_pending_hz) <= 0.06:
            self.a121_resp_pending_count += 1
        else:
            self.a121_resp_pending_hz = measured_hz
            self.a121_resp_pending_count = 1
        if self.a121_resp_pending_count >= 4:
            self.a121_resp_display_hz = measured_hz
            self.a121_resp_pending_count = 0
        return self.a121_resp_display_hz

    def _invalidate_a121_analysis(self, *, reset_tracker: bool = False) -> None:
        self.cached_a121_analysis = None
        self.a121_analysis_request_id += 1
        if reset_tracker:
            self.a121_heart_tracker.reset()
            self.a121_live_trace.reset()
            self._reset_a121_rate_display()
            self.a121_lock_bad_count = 0
            self.last_a121_tracker_update_monotonic = 0.0

    def _configure_gating_ui(self) -> None:
        enabled = self.a121_use_gating_check.isChecked()
        self.a121_gate_spin.setEnabled(enabled)
        self.a121_auto_gate_check.setEnabled(enabled)
        self.a121_show_gate_check.setEnabled(enabled)
        self._invalidate_a121_analysis(reset_tracker=True)

    def _configure_plot(self, plot: pg.PlotItem, title: str, left: str, bottom: str = "Time [s]") -> None:
        plot.clear()
        plot.setTitle(title)
        plot.setLabel("left", left)
        plot.setLabel("bottom", bottom)
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setMouseEnabled(x=True, y=True)
        plot.setDownsampling(auto=True, mode="peak")
        plot.setClipToView(True)
        plot.vb.setDefaultPadding(0.04)
        if plot.legend is None:
            plot.addLegend(offset=(10, 10))

    def _configure_live_plots(self, *_: Any) -> None:
        view_text = self.view_combo.currentText().lower() if hasattr(self, "view_combo") else ""
        raw_view = view_text.startswith("raw")
        fft_view = view_text.startswith("rate")
        if self.active_sensor == "radar":
            self._configure_plot(self.live_plot_a, "Live HB100 radar voltage", "Voltage [mV]")
            self._configure_plot(self.live_plot_b, "Live HB100 radar FFT", "Magnitude", "Frequency [Hz]")
            self._configure_plot(self.live_plot_c, "HB100 filtered vital bands", "Filtered voltage [mV]")
            self.live_curves = {
                "voltage": self.live_plot_a.plot(pen=pg.mkPen("#22d3ee", width=1.5), name="Voltage"),
                "fft": self.live_plot_b.plot(pen=pg.mkPen("#f472b6", width=1.5), name="FFT"),
                "resp": self.live_plot_c.plot(pen=pg.mkPen("#34d399", width=1.5), name="Resp band"),
                "heart": self.live_plot_c.plot(pen=pg.mkPen("#fb7185", width=1.2), name="Heart band"),
            }
        elif self.active_sensor == "a121":
            self._configure_plot(self.live_plot_a, "A121 range profile", "Amplitude", "Distance [m]")
            if raw_view:
                self._configure_plot(self.live_plot_b, "A121 raw selected phase (IQ)", "Phase [rad]")
                self._configure_plot(self.live_plot_c, "A121 raw selected IQ", "I / Q")
                self.live_curves = {
                    "amplitude": self.live_plot_a.plot(pen=pg.mkPen("#22d3ee", width=1.5), name="Amplitude"),
                    "target": self.live_plot_a.plot(pen=None, symbol="o", symbolBrush="#facc15", symbolSize=10, name="Target"),
                    "gate": self.live_plot_a.plot(pen=pg.mkPen("#facc15", width=2), name="Gate"),
                    "raw_phase": self.live_plot_b.plot(pen=pg.mkPen("#f472b6", width=1.4), name="Raw phase"),
                    "raw_i": self.live_plot_c.plot(pen=pg.mkPen("#38bdf8", width=1.1), name="I"),
                    "raw_q": self.live_plot_c.plot(pen=pg.mkPen("#fb7185", width=1.1), name="Q"),
                }
            elif fft_view:
                self._configure_plot(self.live_plot_b, "Respiration rate spectrum (Welch FFT)", "Power", "Frequency [Hz]")
                self._configure_plot(self.live_plot_c, "Heart rate spectrum (Welch FFT)", "Power", "Frequency [Hz]")
                self.live_curves = {
                    "amplitude": self.live_plot_a.plot(pen=pg.mkPen("#22d3ee", width=1.5), name="Amplitude"),
                    "target": self.live_plot_a.plot(pen=None, symbol="o", symbolBrush="#facc15", symbolSize=10, name="Target"),
                    "gate": self.live_plot_a.plot(pen=pg.mkPen("#facc15", width=2), name="Gate"),
                    "resp_fft": self.live_plot_b.plot(pen=pg.mkPen("#34d399", width=1.5), name="Resp PSD"),
                    "resp_peak": self.live_plot_b.plot(pen=None, symbol="o", symbolBrush="#facc15", symbolSize=9, name="Resp peak"),
                    "heart_fft": self.live_plot_c.plot(pen=pg.mkPen("#fb7185", width=1.4), name="Heart PSD"),
                    "heart_peak": self.live_plot_c.plot(pen=None, symbol="o", symbolBrush="#facc15", symbolSize=9, name="Heart peak"),
                }
            else:
                self._configure_plot(self.live_plot_b, "Respiration from A121 phase (0.10-0.70 Hz)", "Phase displacement [rad]")
                self._configure_plot(self.live_plot_c, "Heart motion from A121 phase (0.65-3.00 Hz)", "Phase displacement [rad]")
                self.live_curves = {
                    "amplitude": self.live_plot_a.plot(pen=pg.mkPen("#22d3ee", width=1.5), name="Amplitude"),
                    "target": self.live_plot_a.plot(pen=None, symbol="o", symbolBrush="#facc15", symbolSize=10, name="Target"),
                    "gate": self.live_plot_a.plot(pen=pg.mkPen("#facc15", width=2), name="Gate"),
                    "resp": self.live_plot_b.plot(pen=pg.mkPen("#34d399", width=1.6), name="Respiration"),
                    "heart": self.live_plot_c.plot(pen=pg.mkPen("#fb7185", width=1.3), name="Heart"),
                }
        else:
            self._configure_plot(self.live_plot_a, "Live IMU accelerometer", "g")
            self._configure_plot(self.live_plot_b, "Live IMU gyroscope", "deg/s")
            self._configure_plot(self.live_plot_c, "IMU filtered vital bands", "Filtered accel magnitude [g]")
            self.live_curves = {
                "ax": self.live_plot_a.plot(pen="#38bdf8", name="ax"),
                "ay": self.live_plot_a.plot(pen="#34d399", name="ay"),
                "az": self.live_plot_a.plot(pen="#fbbf24", name="az"),
                "gx": self.live_plot_b.plot(pen="#fb7185", name="gx"),
                "gy": self.live_plot_b.plot(pen="#a78bfa", name="gy"),
                "gz": self.live_plot_b.plot(pen="#f97316", name="gz"),
                "resp": self.live_plot_c.plot(pen=pg.mkPen("#34d399", width=1.4), name="Resp band"),
                "heart": self.live_plot_c.plot(pen=pg.mkPen("#fb7185", width=1.2), name="Heart band"),
            }

    def _refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = list_serial_ports()
        self.port_combo.addItem("Auto")
        self.port_combo.addItems(ports)
        if current:
            idx = self.port_combo.findText(current)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)

    def _apply_startup_defaults(self) -> None:
        if self.default_sensor == "imu":
            self.sensor_combo.setCurrentText("IMU")
        elif self.default_sensor == "a121":
            self.sensor_combo.setCurrentText("A121 Radar")
        else:
            self.sensor_combo.setCurrentText("HB100 Radar")
        self.baud_spin.setValue(self.default_baud)
        if self.default_port:
            idx = self.port_combo.findText(self.default_port)
            if idx < 0:
                self.port_combo.addItem(self.default_port)
                idx = self.port_combo.findText(self.default_port)
            self.port_combo.setCurrentIndex(idx)

    def _start_recording(self) -> None:
        if self.capture is not None:
            return
        sensor = self.active_sensor
        port = self.port_combo.currentText()
        port_name = None if port == "Auto" else port
        output_dir = RAW_RADAR_DIR if sensor == "radar" else RAW_A121_DIR if sensor == "a121" else RAW_IMU_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        storage = self.storage_combo.currentText()
        if "CSV" in storage:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if sensor == "radar":
                name = f"radar_raw_{stamp}.csv"
            elif sensor == "a121":
                name = f"a121_sparse_iq_{stamp}.csv"
            else:
                name = f"respiratory_6axis_raw_{stamp}.csv"
            self.csv_path = output_dir / name

        if sensor == "radar":
            self.capture = RadarCapture(baud=self.baud_spin.value(), output_dir=output_dir)
        elif sensor == "a121":
            self.capture = A121Capture(
                output_dir=output_dir,
                config=A121Config(
                    start_m=float(self.a121_start_spin.value()),
                    end_m=float(self.a121_end_spin.value()),
                    profile=int(self.a121_profile_combo.currentText()),
                    hwaas=int(self.a121_hwaas_spin.value()),
                    sweeps_per_frame=int(self.a121_sweeps_spin.value()),
                    frame_rate_hz=float(self.a121_frame_rate_spin.value()),
                ),
            )
        else:
            self.capture = BreathCapture(baud=self.baud_spin.value(), output_dir=output_dir)
        if not self.capture.connect(port_name):
            self.capture = None
            QtWidgets.QMessageBox.warning(self, "Connection failed", f"Could not connect to {sensor} serial port.")
            return

        if self.csv_path is not None:
            self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(RADAR_COLUMNS if sensor == "radar" else A121_COLUMNS if sensor == "a121" else IMU_COLUMNS)
        self.session_id = self.store.create_session(sensor, self.csv_path) if "SQLite" in storage else None
        self.persisted_index = 0
        self.last_persist_monotonic = 0.0
        self.last_a121_analysis_monotonic = 0.0
        self.cached_a121_analysis = None
        self.a121_gate_center_m = None
        self.last_a121_gate_update_monotonic = 0.0
        self.a121_heart_tracker.reset()
        self.a121_live_trace.reset()
        self._reset_a121_rate_display()
        self.a121_lock_bad_count = 0
        self.last_a121_tracker_update_monotonic = 0.0
        self.a121_analysis_request_id += 1
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.sensor_combo.setEnabled(False)
        self.status_label.setText(f"Recording {sensor}…")
        self.tabs.setCurrentWidget(self.live_tab)
        self.timer.start()

    def _stop_recording(self) -> None:
        if self.capture is None:
            return
        self.a121_analysis_request_id += 1
        self.timer.stop()
        self.capture.stop()
        if self.a121_analysis_thread is not None and self.a121_analysis_thread.isRunning():
            self.a121_analysis_thread.wait()
        self._persist_new_samples()
        rows = self.capture.snapshot_data_storage() if isinstance(self.capture, A121Capture) else list(self.capture.data_storage)
        if self.active_sensor == "radar":
            stats = _radar_stats(pd.DataFrame(rows, columns=RADAR_COLUMNS))
        elif self.active_sensor == "a121":
            stats = _a121_stats(pd.DataFrame(rows, columns=A121_COLUMNS))
        else:
            stats = _imu_stats(pd.DataFrame(rows, columns=IMU_COLUMNS))
        if self.session_id is not None:
            self.store.finish_session(self.session_id, len(rows), stats)
        self._close_csv()
        self.capture = None
        self.session_id = None
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.sensor_combo.setEnabled(True)
        saved_to = []
        if self.csv_path:
            saved_to.append(str(self.csv_path))
        if self.storage_combo.currentText() != "CSV only":
            saved_to.append(str(self.store.path))
        self.status_label.setText("Stopped. Saved to:\n" + "\n".join(saved_to))
        self._refresh_recordings()

    def _close_csv(self) -> None:
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None

    def _persist_new_samples(self) -> None:
        if self.capture is None:
            return
        rows = self.capture.snapshot_data_since(self.persisted_index) if isinstance(self.capture, A121Capture) else self.capture.data_storage[self.persisted_index :]
        if not rows:
            return
        rows_copy = [list(row) for row in rows]
        if self.csv_writer is not None:
            self.csv_writer.writerows(rows_copy)
            if self.csv_file is not None:
                self.csv_file.flush()
        if self.session_id is not None:
            self.store.append_samples(self.active_sensor, self.session_id, rows_copy)
        self.persisted_index += len(rows_copy)

    def _tick(self) -> None:
        if self.capture is None:
            return
        now = time.monotonic()
        if now - self.last_persist_monotonic >= self.persist_interval_s:
            self._persist_new_samples()
            self.last_persist_monotonic = now
        if self.active_sensor == "a121" and isinstance(self.capture, A121Capture):
            live_rows = self.capture.snapshot_live_buffer()
            storage_count = self.capture.data_count()
            if not live_rows and storage_count == 0:
                self.stats_box.setText("Waiting for samples…")
                return
            self._update_live_a121(storage_count, live_rows)
            return

        rows = self.capture.data_storage
        if not rows:
            self.stats_box.setText("Waiting for samples…")
            return
        if self.active_sensor == "radar":
            self._update_live_radar(rows)
        else:
            self._update_live_imu(rows)

    def _time_window_df(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        seconds = self.window_spin.value()
        max_ms = float(df[time_col].iloc[-1])
        return df[df[time_col] >= max_ms - seconds * 1000.0]

    def _set_time_plot_range(self, plot: pg.PlotItem, times_s: np.ndarray, values: np.ndarray, *, min_y_span: float = 1e-3) -> None:
        if len(times_s):
            plot.setXRange(float(times_s[0]), max(float(times_s[-1]), float(times_s[0]) + 0.5), padding=0.0)
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            return
        
        # Trim the first and last 5% of samples (min 1, max 50) to ignore boundary transients of bandpass filters
        trim = int(np.clip(len(finite) // 20, 1, 50)) if len(finite) > 20 else 0
        scale_data = finite[trim:-trim] if trim > 0 else finite
        if len(scale_data) == 0:
            scale_data = finite

        low = float(np.min(scale_data))
        high = float(np.max(scale_data))
        center = (low + high) * 0.5
        span = max(high - low, min_y_span)
        plot.setYRange(center - span * 0.65, center + span * 0.65, padding=0.0)

    def _band_spectrum(self, values: np.ndarray, fs: float, band_hz: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        x = clean_signal(np.asarray(values, dtype=float))
        if len(x) < 16 or fs <= 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        duration_s = len(x) / fs
        low = max(float(band_hz[0]), 1.0 / max(duration_s, 1e-9))
        high = min(float(band_hz[1]), fs * 0.46)
        if low >= high:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        nperseg = min(len(x), max(64, int(round(fs * min(24.0, duration_s)))))
        freqs, power = welch(x, fs=fs, nperseg=nperseg, scaling="spectrum")
        valid = (freqs >= low) & (freqs <= high)
        return freqs[valid], power[valid]

    def _a121_array_from_row(self, row: list[Any], column_index: int) -> np.ndarray:
        return parse_json_array(row[column_index])

    def _a121_plot_latest_profile(
        self,
        live_rows: list[list[Any]],
        analysis: Any | None,
        target_distance_m: float | None,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        if live_rows:
            latest_distances = self._a121_array_from_row(live_rows[-1], 6)
            latest_amplitude = self._a121_array_from_row(live_rows[-1], 7)
        else:
            latest_distances = analysis.distances_m if analysis is not None else np.asarray([], dtype=float)
            latest_amplitude = analysis.latest_amplitude if analysis is not None else np.asarray([], dtype=float)
        if len(latest_distances) != len(latest_amplitude) or len(latest_amplitude) == 0:
            if analysis is not None:
                latest_distances = analysis.distances_m
                latest_amplitude = analysis.latest_amplitude
        selected_idx = 0
        target_m = float(target_distance_m) if target_distance_m is not None and np.isfinite(target_distance_m) else 0.0
        if len(latest_distances) == len(latest_amplitude) and len(latest_amplitude):
            if target_m <= 0:
                search = latest_distances >= 0.28
                if not np.any(search):
                    search = np.ones_like(latest_distances, dtype=bool)
                selected_idx = int(np.argmax(np.where(search, latest_amplitude, -1e18)))
                target_m = float(latest_distances[selected_idx])
            else:
                selected_idx = int(np.argmin(np.abs(latest_distances - target_m)))
            target_amp = float(latest_amplitude[selected_idx])
            self.live_curves["amplitude"].setData(latest_distances, latest_amplitude)
            show_gate = self.a121_show_gate_check.isChecked() and self.a121_use_gating_check.isChecked()
            self.live_curves["target"].setVisible(show_gate)
            self.live_curves["gate"].setVisible(show_gate)
            if show_gate:
                gate_half_width = float(self.a121_gate_spin.value())
                self.live_curves["target"].setData([target_m], [target_amp])
                self.live_curves["gate"].setData(
                    [target_m - gate_half_width, target_m - gate_half_width, target_m + gate_half_width, target_m + gate_half_width],
                    [0, target_amp, target_amp, 0],
                )
            else:
                self.live_curves["target"].setData([], [])
                self.live_curves["gate"].setData([], [])
            self.live_plot_a.setXRange(float(latest_distances[0]), float(latest_distances[-1]), padding=0.0)
            self.live_plot_a.setYRange(0.0, max(float(np.max(latest_amplitude)) * 1.12, target_amp * 1.25, 1.0), padding=0.0)
        return latest_distances, latest_amplitude, target_m, selected_idx

    def _plot_a121_raw_direct(self, live_rows: list[list[Any]], selected_idx: int, window_s: float) -> bool:
        if not live_rows:
            return False
        latest_ms = float(live_rows[-1][0])
        recent_rows = [row for row in live_rows if float(row[0]) >= latest_ms - window_s * 1000.0]
        if len(recent_rows) < 2:
            return False
        times = np.asarray([float(row[0]) for row in recent_rows], dtype=float)
        diffs = np.diff(times)
        if np.count_nonzero(diffs <= 0) > len(diffs) * 0.05:
            fs = max((len(times) - 1) / max((times[-1] - times[0]) / 1000.0, 1e-9), 1.0)
            plot_times = np.arange(len(times), dtype=float) / fs
        else:
            plot_times = (times - times[0]) / 1000.0
        real_values: list[float] = []
        imag_values: list[float] = []
        for row in recent_rows:
            real = self._a121_array_from_row(row, 9)
            imag = self._a121_array_from_row(row, 10)
            if len(real) <= selected_idx or len(imag) <= selected_idx:
                return False
            real_values.append(float(real[selected_idx]))
            imag_values.append(float(imag[selected_idx]))
        raw_i = np.asarray(real_values, dtype=float)
        raw_q = np.asarray(imag_values, dtype=float)
        raw_phase = np.unwrap(np.angle(raw_i + 1j * raw_q))
        raw_phase = raw_phase - float(np.median(raw_phase))
        self.live_curves["raw_phase"].setData(plot_times, raw_phase)
        self.live_curves["raw_i"].setData(plot_times, raw_i)
        self.live_curves["raw_q"].setData(plot_times, raw_q)
        self._set_time_plot_range(self.live_plot_b, plot_times, raw_phase, min_y_span=0.05)
        self._set_time_plot_range(self.live_plot_c, plot_times, np.concatenate([raw_i, raw_q]), min_y_span=1.0)
        return True

    def _update_live_radar(self, rows: list[list[float]]) -> None:
        df = pd.DataFrame(rows[-20000:], columns=RADAR_COLUMNS)
        view = self._time_window_df(df, "Timestamp_ms")
        t = (view["Timestamp_ms"].to_numpy(dtype=float) - float(view["Timestamp_ms"].iloc[0])) / 1000.0
        voltage = view["Voltage_mV"].to_numpy(dtype=float)
        self.live_curves["voltage"].setData(t, voltage)
        stats = _radar_stats(df)
        if len(voltage) > 32:
            fft_source = voltage[-4096:] if len(voltage) > 4096 else voltage
            fft_source = (fft_source - np.mean(fft_source)) * np.hanning(len(fft_source))
            spectrum = np.abs(np.fft.rfft(fft_source)) / max(len(fft_source), 1)
            freqs = np.fft.rfftfreq(len(fft_source), d=1.0 / max(stats["sample_rate_hz"], 1e-9))
            self.live_curves["fft"].setData(freqs, spectrum)
        if len(voltage) > 48:
            self.live_curves["resp"].setData(t, bandpass_filter(voltage, stats["sample_rate_hz"], RESP_BAND_HZ)[-len(t):])
            self.live_curves["heart"].setData(t, bandpass_filter(voltage, stats["sample_rate_hz"], HEART_BAND_HZ)[-len(t):])
        self.stats_box.setText(
            f"Samples: {len(rows)}\n"
            f"Fs: {stats['sample_rate_hz']:.1f} Hz\n"
            f"Resp: {stats['resp_bpm']:.1f} BPM\n"
            f"Peak: {stats['peak_hz']:.2f} Hz\n"
            f"Doppler speed: {stats['speed_mps']:.3f} m/s"
        )

    def _update_live_a121(self, storage_count: int, live_rows: list[list[Any]]) -> None:
        now = time.monotonic()
        window_s = float(self.window_spin.value())
        gate_half_width = float(self.a121_gate_spin.value())
        latest_peak_m = float(live_rows[-1][2]) if live_rows else self.a121_gate_center_m
        previous_gate_center = self.a121_gate_center_m
        user_gate_enabled = self.a121_use_gating_check.isChecked()

        # Do not continuously chase PeakDistance_m. It is just the strongest bin in the
        # current frame and can jump to multipath/clutter, which made both the waveform and
        # rates look random. If the user enables gating before the analyzer has locked a
        # target, seed the gate once; otherwise the analyzer/live trace establish a stable
        # Acconeer-style segment and hold it until explicit reacquisition.
        if user_gate_enabled and latest_peak_m is not None and np.isfinite(latest_peak_m) and latest_peak_m >= 0.28:
            if self.a121_gate_center_m is None:
                self.a121_gate_center_m = latest_peak_m
                self.last_a121_gate_update_monotonic = now
            elif self.a121_auto_gate_check.isChecked():
                peak_outside_gate = abs(latest_peak_m - self.a121_gate_center_m) > gate_half_width
                update_allowed = now - self.last_a121_gate_update_monotonic >= 10.0
                if peak_outside_gate and update_allowed:
                    self.a121_gate_center_m = latest_peak_m
                    self.last_a121_gate_update_monotonic = now

        gate_changed = previous_gate_center != self.a121_gate_center_m
        if gate_changed:
            self._invalidate_a121_analysis(reset_tracker=True)

        # The live trace is append-only and stateful, so historical samples do not change shape
        # as new data arrives. It uses the locked center when available, even if the gate overlay
        # is hidden.
        live_result = self.a121_live_trace.process_rows(
            live_rows,
            target_distance_m=self.a121_gate_center_m,
            use_gating=self.a121_gate_center_m is not None,
            gate_half_width_m=gate_half_width,
        )

        analysis_target_m = self.a121_gate_center_m
        analysis_use_gating = analysis_target_m is not None
        should_analyze = (
            self.cached_a121_analysis is None
            or now - self.last_a121_analysis_monotonic >= self.a121_analysis_interval_s
            or gate_changed
        )
        if should_analyze and (self.a121_analysis_thread is None or self.a121_analysis_thread.isFinished()):
            latest_ms = float(live_rows[-1][0]) if live_rows else 0.0
            analysis_s = max(window_s + A121_RATE_WINDOW_S, 30.0)
            cutoff_analysis_ms = latest_ms - analysis_s * 1000.0
            heart_prior_hz = self.a121_heart_tracker.current_hz
            heart_prior_std_hz = self.a121_heart_tracker.current_std_hz if heart_prior_hz is not None else None
            self.a121_analysis_request_id += 1
            request_id = self.a121_analysis_request_id
            self.a121_analysis_thread = A121AnalysisThread(
                request_id=request_id,
                rows=list(live_rows),
                cutoff_ms=cutoff_analysis_ms,
                gate_half_width=gate_half_width,
                target_distance_m=analysis_target_m,
                heart_prior_hz=heart_prior_hz,
                heart_prior_std_hz=heart_prior_std_hz,
                use_gating=analysis_use_gating,
                auto_gate=self.a121_auto_gate_check.isChecked(),
            )
            self.a121_analysis_thread.analysis_completed.connect(self._on_a121_analysis_completed)
            self.a121_analysis_thread.start()
            self.last_a121_analysis_monotonic = now

        analysis = self.cached_a121_analysis
        view_text = self.view_combo.currentText().lower()
        raw_view = view_text.startswith("raw")
        fft_view = view_text.startswith("rate")
        profile_target = self.a121_gate_center_m or (live_result.target_distance_m if live_result is not None else None) or (analysis.target_distance_m if analysis is not None else None)
        _, _, _direct_target_m, direct_selected_idx = self._a121_plot_latest_profile(live_rows, analysis, profile_target)

        trace_times = live_result.times_s if live_result is not None else np.asarray([], dtype=float)
        trace_mask = trace_times >= (trace_times[-1] - window_s) if len(trace_times) else np.asarray([], dtype=bool)
        trace_plot_times = trace_times[trace_mask]

        if raw_view and live_result is not None and len(trace_plot_times):
            plot_phase = live_result.raw_phase[trace_mask]
            plot_i = live_result.raw_i[trace_mask]
            plot_q = live_result.raw_q[trace_mask]
            self.live_curves["raw_phase"].setData(trace_plot_times, plot_phase)
            self.live_curves["raw_i"].setData(trace_plot_times, plot_i)
            self.live_curves["raw_q"].setData(trace_plot_times, plot_q)
            self._set_time_plot_range(self.live_plot_b, trace_plot_times, plot_phase, min_y_span=0.05)
            self._set_time_plot_range(self.live_plot_c, trace_plot_times, np.concatenate([plot_i, plot_q]), min_y_span=1.0)
        elif raw_view:
            self._plot_a121_raw_direct(live_rows, direct_selected_idx, window_s)
        elif fft_view and analysis is not None:
            resp_freqs, resp_power = self._band_spectrum(analysis.resp_signal, analysis.sample_rate_hz, RESP_BAND_HZ)
            heart_freqs, heart_power = self._band_spectrum(analysis.heart_signal, analysis.sample_rate_hz, HEART_BAND_HZ)
            self.live_curves["resp_fft"].setData(resp_freqs, resp_power)
            self.live_curves["heart_fft"].setData(heart_freqs, heart_power)
            if len(resp_freqs) and analysis.resp_hz > 0:
                idx = int(np.argmin(np.abs(resp_freqs - analysis.resp_hz)))
                self.live_curves["resp_peak"].setData([resp_freqs[idx]], [resp_power[idx]])
            else:
                self.live_curves["resp_peak"].setData([], [])
            if len(heart_freqs) and analysis.heart_hz > 0:
                idx = int(np.argmin(np.abs(heart_freqs - analysis.heart_hz)))
                self.live_curves["heart_peak"].setData([heart_freqs[idx]], [heart_power[idx]])
            else:
                self.live_curves["heart_peak"].setData([], [])
            self.live_plot_b.setXRange(RESP_BAND_HZ[0], RESP_BAND_HZ[1], padding=0.0)
            self.live_plot_c.setXRange(HEART_BAND_HZ[0], HEART_BAND_HZ[1], padding=0.0)
            self.live_plot_b.setYRange(0.0, max(float(np.max(resp_power)) * 1.15 if len(resp_power) else 1.0, 1e-12), padding=0.0)
            self.live_plot_c.setYRange(0.0, max(float(np.max(heart_power)) * 1.15 if len(heart_power) else 1.0, 1e-12), padding=0.0)
        elif not fft_view and live_result is not None and len(trace_plot_times):
            plot_resp = live_result.resp_signal[trace_mask]
            plot_heart = live_result.heart_signal[trace_mask]
            self.live_curves["resp"].setData(trace_plot_times, plot_resp)
            self.live_curves["heart"].setData(trace_plot_times, plot_heart)
            self._set_time_plot_range(self.live_plot_b, trace_plot_times, plot_resp, min_y_span=0.05)
            self._set_time_plot_range(self.live_plot_c, trace_plot_times, plot_heart, min_y_span=0.03)

        if analysis is None:
            target_text = f"{profile_target:.3f} m" if profile_target is not None and np.isfinite(profile_target) else "acquiring"
            self.stats_box.setText(f"Frames: {storage_count}\nA121 live trace: locked target {target_text}\nRates: acquiring {A121_RATE_WINDOW_S:.0f} s analysis window…")
            return

        displayed_resp_hz = self.a121_resp_display_hz
        tracked_heart_hz = self.a121_heart_tracker.current_hz or 0.0
        presence = "YES" if analysis.present else "no"
        gate_mode = "shown/locked" if user_gate_enabled else "auto-locked" if self.a121_gate_center_m is not None else "acquiring"
        resp_text = f"{displayed_resp_hz * 60.0:.1f} BPM ({displayed_resp_hz:.2f} Hz)" if displayed_resp_hz > 0 else "acquiring"
        tracked_text = f"{tracked_heart_hz * 60.0:.1f} BPM ({tracked_heart_hz:.2f} Hz)" if tracked_heart_hz > 0 else "acquiring"
        self.stats_box.setText(
            f"Frames: {storage_count}\n"
            f"Fs: {analysis.sample_rate_hz:.1f} Hz\n"
            f"Presence: {presence} ({analysis.presence_score:.0f}/100)\n"
            f"Target: {gate_mode} center {analysis.target_distance_m:.3f} m  range {analysis.gate_min_m:.2f}-{analysis.gate_max_m:.2f} m\n"
            f"Latest peak: {analysis.peak_distance_m:.3f} m  amp {analysis.peak_amplitude:.1f}\n"
            f"Range bins: {analysis.candidate_bins}  SQI: {analysis.signal_quality:.2f}\n"
            f"Respiration: {resp_text}  conf {analysis.resp_confidence:.1f}\n"
            f"Heart: {tracked_text}  candidate conf {analysis.heart_confidence:.1f}\n"
            f"Live plot: causal append-only filters; Rates: locked range + conservative validation"
        )

    def _on_a121_analysis_completed(self, request_id: int, analysis: Any) -> None:
        if request_id != self.a121_analysis_request_id or self.capture is None or self.active_sensor != "a121":
            return
        self.cached_a121_analysis = analysis
        now = time.monotonic()
        if (
            analysis.present
            and self.a121_gate_center_m is None
            and np.isfinite(analysis.target_distance_m)
            and analysis.target_distance_m >= 0.28
        ):
            self.a121_gate_center_m = float(analysis.target_distance_m)
            self.last_a121_gate_update_monotonic = now
            self.a121_lock_bad_count = 0
            self.a121_live_trace.reset()
        if self.a121_gate_center_m is not None and not self.a121_use_gating_check.isChecked():
            bad_lock = (not analysis.present) or (analysis.signal_quality < 0.30 and analysis.resp_hz <= 0 and analysis.heart_hz <= 0)
            self.a121_lock_bad_count = self.a121_lock_bad_count + 1 if bad_lock else 0
            if self.a121_lock_bad_count >= 5:
                self.a121_gate_center_m = None
                self._invalidate_a121_analysis(reset_tracker=True)
                return
        self._update_a121_resp_display(analysis)
        if analysis.present:
            tracker_dt = now - self.last_a121_tracker_update_monotonic if self.last_a121_tracker_update_monotonic else 0.25
            self.a121_heart_tracker.update(
                analysis.heart_hz,
                tracker_dt,
                confidence=analysis.heart_confidence,
                quality=analysis.signal_quality,
            )
            self.last_a121_tracker_update_monotonic = now
        else:
            self.a121_heart_tracker.reset()

    def _update_live_imu(self, rows: list[list[float]]) -> None:
        df = pd.DataFrame(rows[-20000:], columns=IMU_COLUMNS)
        view = self._time_window_df(df, "Time_ms")
        t = (view["Time_ms"].to_numpy(dtype=float) - float(view["Time_ms"].iloc[0])) / 1000.0
        for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
            self.live_curves[col].setData(t, view[col].to_numpy(dtype=float))
        stats = _imu_stats(df)
        accel_mag = np.linalg.norm(view[["ax", "ay", "az"]].to_numpy(dtype=float), axis=1)
        if len(accel_mag) > 48:
            self.live_curves["resp"].setData(t, bandpass_filter(accel_mag, stats["sample_rate_hz"], RESP_BAND_HZ))
            self.live_curves["heart"].setData(t, bandpass_filter(accel_mag, stats["sample_rate_hz"], HEART_BAND_HZ))
        self.stats_box.setText(
            f"Samples: {len(rows)}\n"
            f"Fs: {stats['sample_rate_hz']:.1f} Hz\n"
            f"Resp: {stats['resp_bpm']:.1f} BPM\n"
            f"Heart band: {stats['heart_bpm']:.1f} BPM\n"
            f"Accel RMS: {stats['accel_rms_g']:.3f} g\n"
            f"Gyro RMS: {stats['gyro_rms_dps']:.2f} deg/s"
        )

    def _refresh_recordings(self) -> None:
        records: list[dict[str, Any]] = []
        for path in sorted(RAW_RADAR_DIR.glob("radar_raw_*.csv"), reverse=True):
            records.append({"source": "CSV", "sensor": "radar", "label": path.name, "samples": "", "path": str(path), "data": path})
        for path in sorted(RAW_A121_DIR.glob("a121_sparse_iq_*.csv"), reverse=True):
            records.append({"source": "CSV", "sensor": "a121", "label": path.name, "samples": "", "path": str(path), "data": path})
        for path in sorted(RAW_IMU_DIR.glob("respiratory_6axis_raw_*.csv"), reverse=True):
            records.append({"source": "CSV", "sensor": "imu", "label": path.name, "samples": "", "path": str(path), "data": path})
        for session in self.store.list_sessions():
            records.append(
                {
                    "source": "SQLite",
                    "sensor": session["sensor"],
                    "label": session["started_at"],
                    "samples": str(session["sample_count"] or ""),
                    "path": f"session #{session['id']}",
                    "data": session,
                }
            )

        self.recording_table.setRowCount(len(records))
        for row, record in enumerate(records):
            first_item = QtWidgets.QTableWidgetItem(record["source"])
            first_item.setData(QtCore.Qt.ItemDataRole.UserRole, record)
            self.recording_table.setItem(row, 0, first_item)
            self.recording_table.setItem(row, 1, QtWidgets.QTableWidgetItem(record["sensor"]))
            self.recording_table.setItem(row, 2, QtWidgets.QTableWidgetItem(record["label"]))
            self.recording_table.setItem(row, 3, QtWidgets.QTableWidgetItem(record["samples"]))
            self.recording_table.setItem(row, 4, QtWidgets.QTableWidgetItem(record["path"]))

    def _selected_record(self) -> dict[str, Any] | None:
        rows = self.recording_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.recording_table.item(rows[0].row(), 0)
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def _open_selected_recording(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        try:
            if record["source"] == "CSV":
                self._open_csv(Path(record["data"]))
            else:
                session = record["data"]
                df = self.store.load_session(int(session["id"]), session["sensor"])
                self._plot_history_df(session["sensor"], df, f"SQLite session #{session['id']}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))

    def _open_csv_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open recording CSV", str(DATA_DIR), "CSV files (*.csv)")
        if path:
            try:
                self._open_csv(Path(path))
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))

    def _open_csv(self, path: Path) -> None:
        df = pd.read_csv(path)
        sensor = _detect_sensor(df)
        self._plot_history_df(sensor, df, path.name)

    def _plot_history_df(self, sensor: str, df: pd.DataFrame, title: str) -> None:
        if df.empty:
            raise ValueError("Recording has no samples.")
        self.history_plot_a.clear()
        self.history_plot_b.clear()
        self.history_plot_c.clear()
        self._configure_plot(self.history_plot_a, title, "", "Time [s]")
        if sensor == "radar":
            df = df.sort_values("Timestamp_ms")
            t = (df["Timestamp_ms"].to_numpy(dtype=float) - float(df["Timestamp_ms"].iloc[0])) / 1000.0
            voltage = df["Voltage_mV"].to_numpy(dtype=float)
            self.history_plot_a.setLabel("left", "Voltage [mV]")
            self.history_plot_a.plot(t, voltage, pen=pg.mkPen("#22d3ee", width=1.2), name="Voltage")
            stats = _radar_stats(df)
            self._configure_plot(self.history_plot_b, "FFT", "Magnitude", "Frequency [Hz]")
            if len(voltage) > 32:
                source = (voltage - np.mean(voltage)) * np.hanning(len(voltage))
                spectrum = np.abs(np.fft.rfft(source)) / max(len(source), 1)
                freqs = np.fft.rfftfreq(len(source), d=1.0 / max(stats["sample_rate_hz"], 1e-9))
                self.history_plot_b.plot(freqs, spectrum, pen=pg.mkPen("#f472b6", width=1.2), name="FFT")
            self._configure_plot(self.history_plot_c, "Filtered vital bands", "Filtered voltage [mV]", "Time [s]")
            if len(voltage) > 48:
                self.history_plot_c.plot(t, bandpass_filter(voltage, stats["sample_rate_hz"], RESP_BAND_HZ), pen=pg.mkPen("#34d399", width=1.2), name="Resp band")
                self.history_plot_c.plot(t, bandpass_filter(voltage, stats["sample_rate_hz"], HEART_BAND_HZ), pen=pg.mkPen("#fb7185", width=1.1), name="Heart band")
            self.history_stats.setText(
                f"{title}\nSamples: {len(df)}\nFs: {stats['sample_rate_hz']:.1f} Hz\n"
                f"Resp: {stats['resp_bpm']:.1f} BPM\nPeak: {stats['peak_hz']:.2f} Hz\n"
                f"Doppler speed: {stats['speed_mps']:.3f} m/s"
            )
        elif sensor == "a121":
            df = df.sort_values("Timestamp_ms")
            analysis = analyze_a121_vitals(
                df,
                auto_gate=self.a121_auto_gate_check.isChecked(),
                gate_half_width_m=float(self.a121_gate_spin.value()),
                max_frames=2400,
            )
            self.history_plot_a.setLabel("left", "Amplitude")
            self.history_plot_a.setLabel("bottom", "Distance [m]")
            if len(analysis.distances_m) == len(analysis.latest_amplitude) and len(analysis.latest_amplitude):
                self.history_plot_a.plot(analysis.distances_m, analysis.latest_amplitude, pen=pg.mkPen("#22d3ee", width=1.2), name="Latest amplitude")
                target_amp = float(analysis.latest_amplitude[min(analysis.selected_index, len(analysis.latest_amplitude) - 1)])
                self.history_plot_a.plot([analysis.target_distance_m], [target_amp], pen=None, symbol="o", symbolBrush="#facc15", symbolSize=10, name="Target")
                self.history_plot_a.plot([analysis.gate_min_m, analysis.gate_min_m, analysis.gate_max_m, analysis.gate_max_m], [0, target_amp, target_amp, 0], pen=pg.mkPen("#facc15", width=2), name="Gate")
            self._configure_plot(self.history_plot_b, "A121 respiration phase band", "Phase displacement [rad]", "Time [s]")
            self._configure_plot(self.history_plot_c, "A121 heart phase band", "Phase displacement [rad]", "Time [s]")
            self.history_plot_b.plot(analysis.times_s, analysis.resp_signal, pen=pg.mkPen("#34d399", width=1.3), name="Respiration")
            self.history_plot_c.plot(analysis.times_s, analysis.heart_signal, pen=pg.mkPen("#fb7185", width=1.2), name="Heart")
            presence = "YES" if analysis.present else "no"
            self.history_stats.setText(
                f"{title}\nFrames: {len(df)}\nFs: {analysis.sample_rate_hz:.1f} Hz\n"
                f"Presence: {presence} ({analysis.presence_score:.0f}/100)\n"
                f"Auto target: {analysis.target_distance_m:.3f} m  gate {analysis.gate_min_m:.2f}-{analysis.gate_max_m:.2f} m\n"
                f"Latest peak: {analysis.peak_distance_m:.3f} m  amp {analysis.peak_amplitude:.1f}\n"
                f"Respiration: {analysis.resp_bpm:.1f} BPM\nHeart: {analysis.heart_bpm:.1f} BPM"
            )
        else:
            df = df.sort_values("Time_ms")
            t = (df["Time_ms"].to_numpy(dtype=float) - float(df["Time_ms"].iloc[0])) / 1000.0
            self.history_plot_a.setLabel("left", "Acceleration [g]")
            self.history_plot_a.plot(t, df["ax"].to_numpy(dtype=float), pen="#38bdf8", name="ax")
            self.history_plot_a.plot(t, df["ay"].to_numpy(dtype=float), pen="#34d399", name="ay")
            self.history_plot_a.plot(t, df["az"].to_numpy(dtype=float), pen="#fbbf24", name="az")
            self._configure_plot(self.history_plot_b, "Gyroscope", "deg/s", "Time [s]")
            self.history_plot_b.plot(t, df["gx"].to_numpy(dtype=float), pen="#fb7185", name="gx")
            self.history_plot_b.plot(t, df["gy"].to_numpy(dtype=float), pen="#a78bfa", name="gy")
            self.history_plot_b.plot(t, df["gz"].to_numpy(dtype=float), pen="#f97316", name="gz")
            stats = _imu_stats(df)
            accel_mag = np.linalg.norm(df[["ax", "ay", "az"]].to_numpy(dtype=float), axis=1)
            self._configure_plot(self.history_plot_c, "Filtered vital bands", "Filtered accel magnitude [g]", "Time [s]")
            if len(accel_mag) > 48:
                self.history_plot_c.plot(t, bandpass_filter(accel_mag, stats["sample_rate_hz"], RESP_BAND_HZ), pen=pg.mkPen("#34d399", width=1.2), name="Resp band")
                self.history_plot_c.plot(t, bandpass_filter(accel_mag, stats["sample_rate_hz"], HEART_BAND_HZ), pen=pg.mkPen("#fb7185", width=1.1), name="Heart band")
            self.history_stats.setText(
                f"{title}\nSamples: {len(df)}\nFs: {stats['sample_rate_hz']:.1f} Hz\n"
                f"Resp: {stats['resp_bpm']:.1f} BPM\nHeart band: {stats['heart_bpm']:.1f} BPM\n"
                f"Accel RMS: {stats['accel_rms_g']:.3f} g\nGyro RMS: {stats['gyro_rms_dps']:.2f} deg/s"
            )
        self.history_plot_a.enableAutoRange()
        self.history_plot_b.enableAutoRange()
        self.history_plot_c.enableAutoRange()
        self.tabs.setCurrentWidget(self.history_tab)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt override name
        if self.capture is not None:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Recording active",
                "Stop recording and exit?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._stop_recording()
        event.accept()


def launch_app(default_sensor: str = "radar", default_port: str | None = None, default_baud: int = 921600) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("RespiNet Sensor Studio")
    window = MainWindow(default_sensor=default_sensor, default_port=default_port, default_baud=default_baud)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(launch_app())
