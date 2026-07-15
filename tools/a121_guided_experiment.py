#!/usr/bin/env python3
"""Guided, repeatable A121 lens/foil experiment runner.

The default experiment is the sequence requested for the A121 three-lens setup:

    glued patch:   50 cm -> Hyperbolic, FZP, Flat Cover
                   100 cm -> Hyperbolic, FZP, Flat Cover
    unglued patch: 50 cm -> Hyperbolic, FZP, Flat Cover
                   100 cm -> Hyperbolic, FZP, Flat Cover

Run from the repository root with::

    uv run python tools/a121_guided_experiment.py

The sequence, timing, A121 settings, and instructions are read from the JSON config
file in configs/a121_foil_lens_experiment.json.  Use --start-step to begin at any
1-based expanded measurement run, and --end-step to stop after a selected run.
The default configuration repeats each of the 12 conditions three times, for 36 runs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make direct execution (python tools/...) work as well as uv run.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from respi_net.a121 import A121_CAPTURE_COLUMNS, A121Config, A121Capture, find_a121_serial_ports


DEFAULT_CONFIG_PATH = ROOT / "configs" / "a121_foil_lens_experiment.json"

LENS_LABELS = {
    "hyperbolic": "Hyperbolic lens",
    "fzp": "FZP lens",
    "flat_cover": "Flat Cover lens",
}

PATCH_LABELS = {
    "glued": "Foil patch glued to chest",
    "unglued": "Foil patch removed",
    "sham": "Sham/control patch",
    "foil": "Aluminium foil patch",
}


@dataclass(frozen=True)
class ExperimentStep:
    number: int
    condition_number: int
    repeat_number: int
    repeat_total: int
    lens: str
    lens_label: str
    patch: str
    patch_label: str
    distance_cm: float
    condition_id: str = ""
    geometry: str = ""
    geometry_label: str = ""
    instruction: str = "Keep still and breathe normally."

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BreathingCue:
    start_s: float
    end_s: float
    cue: str
    detail: str
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config() -> dict[str, Any]:
    lenses = ["hyperbolic", "fzp", "flat_cover"]
    steps: list[dict[str, Any]] = []
    number = 1
    for patch, patch_instruction in (
        ("glued", "Keep the foil patch in the same marked position."),
        ("unglued", "Remove the foil patch and leave the chest otherwise unchanged."),
    ):
        for distance_cm in (50, 100):
            for lens in lenses:
                steps.append(
                    {
                        "lens": lens,
                        "patch": patch,
                        "distance_cm": distance_cm,
                        "instruction": f"{patch_instruction} Keep still and breathe normally.",
                    }
                )
                number += 1

    return {
        "experiment_name": "A121 aluminium foil and lens comparison",
        "prep_seconds": 10,
        "measurement_seconds": 60,
        "reconnect_delay_seconds": 5,
        "connect_attempts": 2,
        "repeats_per_condition": 3,
        "output_dir": "data/raw/a121/guided",
        "a121": {
            "start_m": 0.2,
            "end_m": 1.5,
            "profile": 3,
            "hwaas": 32,
            "sweeps_per_frame": 8,
            "frame_rate_hz": 20.0,
            "step_length": 1,
        },
        "steps": steps,
    }


def load_config(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = default_config()

    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")

    base = default_config()
    merged = {**base, **payload}
    merged["a121"] = {**base["a121"], **payload.get("a121", {})}
    if not isinstance(merged.get("steps"), list) or not merged["steps"]:
        raise ValueError("Config must contain a non-empty 'steps' list.")
    return merged


def _make_experiment_step(
    raw: dict[str, Any],
    *,
    run_number: int,
    condition_number: int,
    repeat_number: int,
    repeat_total: int,
    condition_id: str = "",
) -> ExperimentStep:
    lens = str(raw.get("lens", "")).strip()
    patch = str(raw.get("patch", "")).strip()
    if not lens or not patch:
        raise ValueError(f"Condition {condition_number} needs both 'lens' and 'patch'.")
    try:
        distance_cm = float(raw["distance_cm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Condition {condition_number} needs numeric 'distance_cm'.") from exc
    if distance_cm <= 0:
        raise ValueError(f"Condition {condition_number} distance must be positive.")

    geometry = str(raw.get("geometry", "")).strip()
    geometry_label = str(raw.get("geometry_label") or geometry.replace("_", " ").title()).strip()
    return ExperimentStep(
        number=run_number,
        condition_number=condition_number,
        repeat_number=repeat_number,
        repeat_total=repeat_total,
        lens=lens,
        lens_label=str(raw.get("lens_label") or LENS_LABELS.get(lens, lens)),
        patch=patch,
        patch_label=str(raw.get("patch_label") or PATCH_LABELS.get(patch, patch)),
        distance_cm=distance_cm,
        condition_id=condition_id,
        geometry=geometry,
        geometry_label=geometry_label,
        instruction=str(raw.get("instruction") or "Keep still and breathe normally."),
    )


def parse_steps(payload: dict[str, Any]) -> list[ExperimentStep]:
    raw_conditions = payload["steps"]
    for condition_number, raw in enumerate(raw_conditions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Condition {condition_number} must be a JSON object.")

    run_sequence = payload.get("run_sequence")
    if run_sequence is not None:
        if not isinstance(run_sequence, list) or not run_sequence:
            raise ValueError("run_sequence must be a non-empty list of condition IDs.")
        conditions_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
        for condition_number, raw in enumerate(raw_conditions, start=1):
            condition_id = str(raw.get("condition_id", "")).strip()
            if not condition_id:
                raise ValueError(
                    f"Condition {condition_number} needs 'condition_id' when run_sequence is used."
                )
            if condition_id in conditions_by_id:
                raise ValueError(f"Duplicate condition_id in config: {condition_id}")
            conditions_by_id[condition_id] = (condition_number, raw)

        sequence = [str(item).strip() for item in run_sequence]
        unknown = sorted({condition_id for condition_id in sequence if condition_id not in conditions_by_id})
        if unknown:
            raise ValueError(f"run_sequence contains unknown condition IDs: {', '.join(unknown)}")
        repeat_totals = Counter(sequence)
        repeats_seen: Counter[str] = Counter()
        steps: list[ExperimentStep] = []
        for run_number, condition_id in enumerate(sequence, start=1):
            repeats_seen[condition_id] += 1
            condition_number, raw = conditions_by_id[condition_id]
            steps.append(
                _make_experiment_step(
                    raw,
                    run_number=run_number,
                    condition_number=condition_number,
                    repeat_number=repeats_seen[condition_id],
                    repeat_total=repeat_totals[condition_id],
                    condition_id=condition_id,
                )
            )
        return steps

    steps = []
    run_number = 1
    default_repeats = int(payload.get("repeats_per_condition", 1))
    if default_repeats < 1:
        raise ValueError("repeats_per_condition must be at least 1.")
    for condition_number, raw in enumerate(raw_conditions, start=1):
        repeat_total = int(raw.get("repeats", default_repeats))
        if repeat_total < 1:
            raise ValueError(f"Condition {condition_number} repeats must be at least 1.")
        for repeat_number in range(1, repeat_total + 1):
            steps.append(
                _make_experiment_step(
                    raw,
                    run_number=run_number,
                    condition_number=condition_number,
                    repeat_number=repeat_number,
                    repeat_total=repeat_total,
                    condition_id=str(raw.get("condition_id", "")).strip(),
                )
            )
            run_number += 1
    return steps


def _positive_float(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be numeric.") from exc
    if result <= 0:
        raise ValueError(f"{description} must be positive.")
    return result


def parse_breathing_protocol(payload: dict[str, Any]) -> list[BreathingCue]:
    blocks = payload.get("breathing_protocol")
    if blocks in (None, []):
        return []
    if not isinstance(blocks, list):
        raise ValueError("breathing_protocol must be a list.")

    cues: list[BreathingCue] = []
    cursor = 0.0

    def append_cue(kind: str, duration_s: float, cue: str, detail: str) -> None:
        nonlocal cursor
        cues.append(
            BreathingCue(
                start_s=cursor,
                end_s=cursor + duration_s,
                cue=cue,
                detail=detail,
                kind=kind,
            )
        )
        cursor += duration_s

    for block_number, raw in enumerate(blocks, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Breathing protocol block {block_number} must be a JSON object.")
        kind = str(raw.get("kind", "")).strip().lower()
        if kind == "paced":
            try:
                cycles = int(raw.get("cycles", 0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Breathing protocol block {block_number} cycles must be an integer.") from exc
            if cycles < 1:
                raise ValueError(f"Breathing protocol block {block_number} needs at least one cycle.")
            inhale_s = _positive_float(raw.get("inhale_seconds"), f"Block {block_number} inhale_seconds")
            exhale_s = _positive_float(raw.get("exhale_seconds"), f"Block {block_number} exhale_seconds")
            inhale_cue = str(raw.get("inhale_cue") or "WDECH")
            exhale_cue = str(raw.get("exhale_cue") or "WYDECH")
            inhale_detail = str(raw.get("inhale_detail") or "Spokojny, równy wdech.")
            exhale_detail = str(raw.get("exhale_detail") or "Spokojny, równy wydech.")
            for _ in range(cycles):
                append_cue("inhale", inhale_s, inhale_cue, inhale_detail)
                append_cue("exhale", exhale_s, exhale_cue, exhale_detail)
            continue

        if not kind:
            raise ValueError(f"Breathing protocol block {block_number} needs 'kind'.")
        duration_s = _positive_float(raw.get("duration_seconds"), f"Block {block_number} duration_seconds")
        default_cues = {
            "normal": "ODDYCHAJ SWOBODNIE",
            "hold": "WSTRZYMAJ ODDECH",
        }
        cue = str(raw.get("cue") or default_cues.get(kind, kind.upper()))
        detail = str(raw.get("detail") or "")
        append_cue(kind, duration_s, cue, detail)

    measurement_seconds = _positive_float(payload.get("measurement_seconds", 60), "measurement_seconds")
    if abs(cursor - measurement_seconds) > 0.05:
        raise ValueError(
            "Expanded breathing protocol lasts "
            f"{cursor:g} s, but measurement_seconds is {measurement_seconds:g} s."
        )
    return cues


def breathing_cue_at(cues: list[BreathingCue], elapsed_s: float) -> tuple[BreathingCue, float] | None:
    if not cues:
        return None
    elapsed_s = max(0.0, float(elapsed_s))
    for index, cue in enumerate(cues):
        if elapsed_s < cue.end_s or index == len(cues) - 1:
            return cue, max(0.0, cue.end_s - elapsed_s)
    return None


def make_a121_config(payload: dict[str, Any]) -> A121Config:
    valid_names = {field.name for field in fields(A121Config)}
    values = {key: value for key, value in payload.get("a121", {}).items() if key in valid_names}
    return A121Config(**values)


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return safe or "measurement"


def discover_a121_port() -> str | None:
    """Return the same preferred A121 port used by the main application."""
    candidates = find_a121_serial_ports()
    return candidates[0] if candidates else None


def write_rows_csv(rows: list[list[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=A121_CAPTURE_COLUMNS).to_csv(path, index=False)


def median_column(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def summarize_measurement(
    path: Path,
    step: ExperimentStep,
    port: str,
    requested_seconds: float,
) -> dict[str, Any]:
    df = pd.read_csv(path)
    timestamp = pd.to_numeric(df.get("Timestamp_ms"), errors="coerce").to_numpy(dtype=float)
    timestamp = timestamp[np.isfinite(timestamp)]
    if len(timestamp) > 1:
        differences = np.diff(timestamp) / 1000.0
        differences = differences[np.isfinite(differences) & (differences > 0)]
        sample_rate_hz = float(1.0 / np.median(differences)) if len(differences) else 0.0
        captured_seconds = float((timestamp[-1] - timestamp[0]) / 1000.0)
    else:
        sample_rate_hz = 0.0
        captured_seconds = 0.0

    summary: dict[str, Any] = {
        "step": step.number,
        "condition_number": step.condition_number,
        "condition_id": step.condition_id,
        "repeat_number": step.repeat_number,
        "repeat_total": step.repeat_total,
        "lens": step.lens,
        "lens_label": step.lens_label,
        "patch": step.patch,
        "patch_label": step.patch_label,
        "geometry": step.geometry,
        "geometry_label": step.geometry_label,
        "distance_cm": step.distance_cm,
        "port": port,
        "csv_path": str(path),
        "frames": int(len(df)),
        "requested_seconds": requested_seconds,
        "captured_seconds": captured_seconds,
        "sample_rate_hz": sample_rate_hz,
        "median_peak_distance_m": median_column(df, "PeakDistance_m"),
        "median_peak_amplitude": median_column(df, "PeakAmplitude"),
        "median_mean_amplitude": median_column(df, "MeanAmplitude"),
        "median_target_distance_m": median_column(df, "AcconeerTargetDistance_m"),
        "median_breathing_rate_bpm": median_column(df, "AcconeerBreathingRate_BPM"),
        "presence_percent": None,
        "app_state_last": str(df["AcconeerAppState"].iloc[-1]) if "AcconeerAppState" in df and len(df) else "",
    }
    if "AcconeerPresenceDetected" in df:
        presence = pd.to_numeric(df["AcconeerPresenceDetected"], errors="coerce")
        summary["presence_percent"] = float(presence.mean() * 100.0)

    # Run the repository's batch analyzer for a useful post-capture quality readout.
    try:
        from respi_net.a121_vitals import analyze_a121_vitals

        analysis = analyze_a121_vitals(df, max_frames=900)
        summary.update(
            {
                "analysis_resp_bpm": float(analysis.resp_bpm),
                "analysis_resp_confidence": float(analysis.resp_confidence),
                "analysis_signal_quality": float(analysis.signal_quality),
                "analysis_present": bool(analysis.present),
            }
        )
    except Exception as exc:  # The raw recording remains valid if optional analysis fails.
        summary["analysis_error"] = str(exc)
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    def value(key: str, digits: int = 3, suffix: str = "") -> str:
        raw = summary.get(key)
        if raw is None:
            return "n/a"
        try:
            return f"{float(raw):.{digits}f}{suffix}"
        except (TypeError, ValueError):
            return str(raw)

    lines = [
        f"Measurement finished — repeat {summary.get('repeat_number', '?')}/{summary.get('repeat_total', '?')} for this condition",
        f"Frames: {summary.get('frames', 0)}  |  sample rate: {value('sample_rate_hz', 2, ' Hz')}",
        f"Peak distance: {value('median_peak_distance_m', 3, ' m')}  |  peak amplitude: {value('median_peak_amplitude', 1)}",
        f"Target distance: {value('median_target_distance_m', 3, ' m')}  |  presence: {value('presence_percent', 1, '%')}",
        f"Acconeer state: {summary.get('app_state_last', 'n/a')}",
        f"Respiration: {value('analysis_resp_bpm', 1, ' BPM')}  |  confidence: {value('analysis_resp_confidence', 1)}",
        f"Signal quality: {value('analysis_signal_quality', 3)}",
        f"Saved: {summary.get('csv_path', '')}",
    ]
    condition_id = str(summary.get("condition_id") or "")
    geometry_label = str(summary.get("geometry_label") or "")
    if condition_id or geometry_label:
        lines.insert(1, f"Condition: {condition_id or 'n/a'}  |  geometry: {geometry_label or 'n/a'}")
    if summary.get("analysis_error"):
        lines.append(f"Analysis warning: {summary['analysis_error']}")
    return "\n".join(lines)


class MeasurementWorker(QObject):
    status_changed = Signal(str)
    prep_changed = Signal(int)
    measurement_changed = Signal(float, float)
    finished = Signal(object)
    failed = Signal(str)
    aborted = Signal()

    def __init__(
        self,
        *,
        step: ExperimentStep,
        config: dict[str, Any],
        a121_config: A121Config,
        output_path: Path,
        port: str | None,
    ) -> None:
        super().__init__()
        self.step = step
        self.config = config
        self.a121_config = a121_config
        self.output_path = output_path
        self.requested_port = port
        self.abort_event = threading.Event()
        self.capture: A121Capture | None = None
        self.resolved_port = port or ""

    def request_abort(self) -> None:
        self.abort_event.set()

    def _stop_capture(self) -> None:
        if self.capture is not None:
            self.capture.stop()

    def _abort(self) -> None:
        self.status_changed.emit("Measurement interrupted; discarding current capture…")
        self._stop_capture()
        self.output_path.unlink(missing_ok=True)
        self.aborted.emit()

    @Slot()
    def run(self) -> None:
        try:
            self.resolved_port = self.requested_port or discover_a121_port() or ""
            if not self.resolved_port:
                raise RuntimeError("No A121 serial port found. Connect the WCH USB Dual_Serial device or pass --port.")

            reconnect_delay = max(0.0, float(self.config.get("reconnect_delay_seconds", 5)))
            connect_attempts = max(1, int(self.config.get("connect_attempts", 2)))
            # The WCH USB Dual_Serial interface can need a few seconds after stop_session()
            # before it accepts the next A121 setup request. This matters from step 2 onward,
            # when the operator has just changed lenses or patch state.
            if self.step.number > 1 and reconnect_delay > 0:
                self.status_changed.emit(
                    f"Allowing the A121 interface to settle for {reconnect_delay:.0f} seconds…"
                )
                if self.abort_event.wait(reconnect_delay):
                    self._abort()
                    return

            connected = False
            last_connection_error = ""
            for attempt in range(1, connect_attempts + 1):
                if self.abort_event.is_set():
                    self._abort()
                    return
                self.status_changed.emit(
                    f"Connecting to A121 on {self.resolved_port} (attempt {attempt}/{connect_attempts})…"
                )
                self.capture = A121Capture(
                    output_dir=self.output_path.parent,
                    config=self.a121_config,
                )
                if self.capture.connect(self.resolved_port):
                    connected = True
                    break
                last_connection_error = "A121 connection/setup failed."
                if attempt < connect_attempts:
                    self.status_changed.emit(
                        f"A121 did not respond; retrying in {reconnect_delay:.0f} seconds…"
                    )
                    if self.abort_event.wait(reconnect_delay):
                        self._abort()
                        return
            if not connected:
                raise RuntimeError(
                    f"{last_connection_error} Check that Interface A is selected and the sensor is powered."
                )

            prep_seconds = max(0, int(round(float(self.config.get("prep_seconds", 10)))))
            self.status_changed.emit("Connected. Hold position and prepare for the measurement.")
            for remaining in range(prep_seconds, 0, -1):
                if self.abort_event.is_set():
                    self._abort()
                    return
                self.prep_changed.emit(remaining)
                if self.abort_event.wait(1.0):
                    self._abort()
                    return

            if self.abort_event.is_set():
                self._abort()
                return

            measurement_seconds = max(1.0, float(self.config.get("measurement_seconds", 60)))
            start_index = self.capture.data_count()
            started = time.monotonic()
            if self.config.get("breathing_protocol"):
                self.status_changed.emit("Measuring now. Follow the breathing cue shown below.")
            else:
                self.status_changed.emit("Measuring now. Stay still and breathe normally.")
            while True:
                if self.abort_event.is_set():
                    self._abort()
                    return
                if not self.capture.running:
                    raise RuntimeError("A121 stopped running before the measurement finished.")
                elapsed = time.monotonic() - started
                self.measurement_changed.emit(min(elapsed, measurement_seconds), measurement_seconds)
                if elapsed >= measurement_seconds:
                    break
                if self.abort_event.wait(0.1):
                    self._abort()
                    return

            if self.abort_event.is_set():
                self._abort()
                return
            self.status_changed.emit("Stopping capture and preparing the review summary…")
            self.capture.stop()
            if self.abort_event.is_set():
                self._abort()
                return
            rows = self.capture.snapshot_data_since(start_index)
            if len(rows) < 10:
                raise RuntimeError(f"Only {len(rows)} usable frames were captured; measurement was not saved.")
            write_rows_csv(rows, self.output_path)
            if self.abort_event.is_set():
                self._abort()
                return
            summary = summarize_measurement(
                self.output_path,
                self.step,
                self.resolved_port,
                measurement_seconds,
            )
            if self.abort_event.is_set():
                self._abort()
                return
            self.finished.emit(summary)
        except Exception as exc:
            if self.abort_event.is_set():
                self._abort()
            else:
                self.failed.emit(str(exc))
        finally:
            self._stop_capture()


class GuidedExperimentWindow(QWidget):
    def __init__(
        self,
        *,
        config: dict[str, Any],
        steps: list[ExperimentStep],
        breathing_cues: list[BreathingCue],
        a121_config: A121Config,
        start_step: int,
        end_step: int,
        port: str | None,
        session_dir: Path,
    ) -> None:
        super().__init__()
        self.config = config
        self.steps = steps
        self.breathing_cues = breathing_cues
        self.a121_config = a121_config
        self.current_index = start_step - 1
        self.end_index = end_step - 1
        self.condition_count = max(step.condition_number for step in steps)
        self.port = port
        self.session_dir = session_dir
        self.active_thread: QThread | None = None
        self.active_worker: MeasurementWorker | None = None
        self.pending_summary: dict[str, Any] | None = None
        self.current_cue_key: tuple[float, float] | None = None
        self.manifest_path = self.session_dir / "manifest.json"
        self.manifest = self._load_or_create_manifest()

        self.setWindowTitle(str(config.get("experiment_name", "A121 guided experiment")))
        self.resize(820, 760)
        self._build_ui()
        self.show_step()

        self.abort_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.abort_shortcut.activated.connect(self.abort_current_measurement)

    def _load_or_create_manifest(self) -> dict[str, Any]:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except (OSError, json.JSONDecodeError):
                pass
        manifest = {
            "experiment_name": self.config.get("experiment_name", "A121 guided experiment"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "steps": [step.as_dict() for step in self.steps],
            "breathing_cues": [cue.as_dict() for cue in self.breathing_cues],
            "measurements": [],
        }
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: dict[str, Any] | None = None) -> None:
        payload = manifest if manifest is not None else self.manifest
        temporary = self.manifest_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        temporary.replace(self.manifest_path)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        title = QLabel(str(self.config.get("experiment_name", "A121 guided experiment")))
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)

        self.step_label = QLabel()
        self.step_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.step_label)

        instructions_box = QGroupBox("Current setup")
        instructions_layout = QVBoxLayout(instructions_box)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.instructions.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        instructions_layout.addWidget(self.instructions)
        layout.addWidget(instructions_box)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.cue_box = QGroupBox("Polecenie oddechowe")
        cue_layout = QVBoxLayout(self.cue_box)
        self.cue_label = QLabel("PRZYGOTUJ SIĘ")
        self.cue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cue_label.setStyleSheet("font-size: 32px; font-weight: bold; padding: 12px;")
        self.cue_detail = QLabel()
        self.cue_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cue_detail.setWordWrap(True)
        self.cue_countdown = QLabel()
        self.cue_countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cue_countdown.setStyleSheet("font-size: 18px; font-weight: bold;")
        cue_layout.addWidget(self.cue_label)
        cue_layout.addWidget(self.cue_detail)
        cue_layout.addWidget(self.cue_countdown)
        self.cue_box.setVisible(bool(self.breathing_cues))
        layout.addWidget(self.cue_box)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        summary_box = QGroupBox("Latest measurement")
        summary_layout = QVBoxLayout(summary_box)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(210)
        summary_layout.addWidget(self.summary)
        layout.addWidget(summary_box, stretch=1)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start measurement")
        self.abort_button = QPushButton("Przerwij i odrzuć (Esc)")
        self.accept_button = QPushButton("Good — accept and continue")
        self.redo_button = QPushButton("Redo measurement")
        self.quit_button = QPushButton("Quit")
        self.start_button.clicked.connect(self.start_measurement)
        self.abort_button.clicked.connect(self.abort_current_measurement)
        self.accept_button.clicked.connect(self.accept_measurement)
        self.redo_button.clicked.connect(self.redo_measurement)
        self.quit_button.clicked.connect(self.close)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.abort_button)
        buttons.addWidget(self.accept_button)
        buttons.addWidget(self.redo_button)
        buttons.addStretch(1)
        buttons.addWidget(self.quit_button)
        layout.addLayout(buttons)

        self.abort_button.setEnabled(False)
        self.abort_button.setStyleSheet(
            "QPushButton { color: #991b1b; font-weight: bold; }"
            "QPushButton:enabled { background: #fee2e2; border: 1px solid #dc2626; padding: 5px; }"
        )

        keyboard = QLabel(
            "Esc lub przycisk „Przerwij i odrzuć” zatrzymuje aktywny pomiar i nie zachowuje nagrania."
        )
        keyboard.setStyleSheet("color: #666;")
        layout.addWidget(keyboard)

    def _set_cue_display(self, cue: str, detail: str, countdown: str, kind: str) -> None:
        colors = {
            "inhale": ("#dbeafe", "#1d4ed8"),
            "exhale": ("#dcfce7", "#15803d"),
            "hold": ("#fee2e2", "#b91c1c"),
            "normal": ("#f3f4f6", "#374151"),
        }
        background, foreground = colors.get(kind, colors["normal"])
        self.cue_label.setText(cue)
        self.cue_label.setStyleSheet(
            "font-size: 32px; font-weight: bold; padding: 12px; "
            f"background: {background}; color: {foreground}; border-radius: 6px;"
        )
        self.cue_detail.setText(detail)
        self.cue_countdown.setText(countdown)

    def current_step(self) -> ExperimentStep | None:
        if 0 <= self.current_index < len(self.steps) and self.current_index <= self.end_index:
            return self.steps[self.current_index]
        return None

    def show_step(self, status_text: str | None = None) -> None:
        step = self.current_step()
        self.pending_summary = None
        self.current_cue_key = None
        self.accept_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.abort_button.setEnabled(False)
        self.progress.setValue(0)
        self.summary.clear()
        if self.breathing_cues:
            self._set_cue_display(
                "PO STARCIE: ODDYCHAJ SWOBODNIE",
                "Program następnie poprowadzi wdech, wydech i wstrzymanie oddechu.",
                "",
                "normal",
            )
        if step is None:
            self.step_label.setText("Experiment complete")
            self.instructions.setText(
                f"All selected measurements are complete. Accepted files and manifest are in:\n{self.session_dir}"
            )
            self.status.setText("Finished.")
            self.start_button.setEnabled(False)
            return

        condition_id = f" [{step.condition_id}]" if step.condition_id else ""
        self.step_label.setText(
            f"Run {step.number}/{len(self.steps)} — condition {step.condition_number}/{self.condition_count}"
            f"{condition_id}, repeat {step.repeat_number}/{step.repeat_total}"
        )
        geometry_line = f"Geometry: {step.geometry_label}\n" if step.geometry_label else ""
        self.instructions.setText(
            f"Lens: {step.lens_label}\n"
            f"Patch condition: {step.patch_label}\n"
            f"{geometry_line}"
            f"Distance: {step.distance_cm:g} cm\n\n"
            f"This is repeat {step.repeat_number} of {step.repeat_total} for this condition.\n"
            f"{step.instruction}\n\n"
            "Check the setup, then click Start measurement."
        )
        self.status.setText(status_text or "Ready.")
        self.start_button.setEnabled(True)

    def output_path_for_step(self, step: ExperimentStep) -> Path:
        condition_id = f"_{safe_filename(step.condition_id).lower()}" if step.condition_id else ""
        geometry = f"_{safe_filename(step.geometry)}" if step.geometry else ""
        filename = (
            f"condition_{step.condition_number:02d}{condition_id}_repeat_{step.repeat_number:02d}_"
            f"run_{step.number:02d}_{safe_filename(step.patch)}{geometry}_"
            f"{safe_filename(str(step.distance_cm).replace('.', 'p'))}cm_"
            f"{safe_filename(step.lens)}.csv"
        )
        return self.session_dir / filename

    @Slot()
    def start_measurement(self) -> None:
        step = self.current_step()
        if step is None or self.active_thread is not None:
            return
        old_path = self.output_path_for_step(step)
        if old_path.exists():
            old_path.unlink()

        self.start_button.setEnabled(False)
        self.abort_button.setEnabled(True)
        self.accept_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.summary.clear()
        self.status.setText("Starting…")
        self.current_cue_key = None
        self.progress.setValue(0)

        worker = MeasurementWorker(
            step=step,
            config=self.config,
            a121_config=self.a121_config,
            output_path=old_path,
            port=self.port,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status_changed.connect(self.status.setText)
        worker.prep_changed.connect(self.on_prep_changed)
        worker.measurement_changed.connect(self.on_measurement_changed)
        worker.finished.connect(self.on_measurement_finished)
        worker.failed.connect(self.on_measurement_failed)
        worker.aborted.connect(self.on_measurement_aborted)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.aborted.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self.on_thread_finished)
        self.active_worker = worker
        self.active_thread = thread
        thread.start()

    @Slot(int)
    def on_prep_changed(self, remaining: int) -> None:
        self.status.setText(f"Preparation: {remaining} seconds remaining. Hold still.")
        if self.breathing_cues:
            self._set_cue_display(
                "PRZYGOTUJ SIĘ",
                "Usiądź nieruchomo. Pierwsze 10 sekund nagrania to swobodny oddech.",
                f"Start za {remaining} s",
                "normal",
            )
        self.progress.setValue(0)

    @Slot(float, float)
    def on_measurement_changed(self, elapsed: float, total: float) -> None:
        cue_state = breathing_cue_at(self.breathing_cues, elapsed)
        if cue_state is not None:
            cue, remaining = cue_state
            cue_key = (cue.start_s, cue.end_s)
            if cue_key != self.current_cue_key:
                self.current_cue_key = cue_key
                if bool(self.config.get("audible_cues", True)):
                    QApplication.beep()
            self._set_cue_display(
                cue.cue,
                cue.detail,
                f"Jeszcze {remaining:.1f} s  •  pomiar {elapsed:.1f}/{total:.0f} s",
                cue.kind,
            )
            self.status.setText(f"Measuring: {elapsed:.1f}/{total:.0f} seconds. Follow the cue.")
        else:
            self.status.setText(f"Measuring: {elapsed:.1f}/{total:.0f} seconds. Stay still.")
        self.progress.setValue(int(round(100.0 * elapsed / max(total, 1e-9))))

    @Slot(object)
    def on_measurement_finished(self, summary: object) -> None:
        self.pending_summary = dict(summary)  # type: ignore[arg-type]
        self.abort_button.setEnabled(False)
        self.progress.setValue(100)
        self.summary.setPlainText(format_summary(self.pending_summary))
        self.status.setText("Review the result. Accept it or redo it before continuing.")
        if self.breathing_cues:
            self._set_cue_display("KONIEC", "Oddychaj swobodnie.", "Nagranie gotowe do oceny.", "normal")
        self.accept_button.setEnabled(True)
        self.redo_button.setEnabled(True)

    @Slot(str)
    def on_measurement_failed(self, message: str) -> None:
        self.abort_button.setEnabled(False)
        self.status.setText(f"Measurement failed: {message}")
        self.summary.setPlainText("No accepted recording was created. Fix the setup and click Start again.")
        self.start_button.setEnabled(True)

    @Slot()
    def on_measurement_aborted(self) -> None:
        self.abort_button.setEnabled(False)
        self.status.setText("Current measurement discarded. Click Start measurement to redo it.")
        self.summary.setPlainText("Interrupted capture was not saved.")
        if self.breathing_cues:
            self._set_cue_display("ODRZUCONO", "Oddychaj swobodnie.", "Uruchom ponownie, gdy będziesz gotowy.", "hold")
        self.start_button.setEnabled(True)

    @Slot()
    def on_thread_finished(self) -> None:
        thread = self.active_thread
        self.active_worker = None
        self.active_thread = None
        self.abort_button.setEnabled(False)
        if thread is not None:
            thread.deleteLater()

    @Slot()
    def abort_current_measurement(self) -> None:
        if self.active_worker is None:
            return
        self.abort_button.setEnabled(False)
        self.status.setText("Interrupt requested; stopping and discarding the current measurement…")
        self.active_worker.request_abort()

    @Slot()
    def accept_measurement(self) -> None:
        if self.pending_summary is None:
            return
        entry = dict(self.pending_summary)
        entry["status"] = "accepted"
        entry["accepted_at"] = datetime.now(timezone.utc).isoformat()
        measurements = [
            item for item in self.manifest.get("measurements", []) if item.get("step") != entry.get("step")
        ]
        measurements.append(entry)
        measurements.sort(key=lambda item: int(item.get("step", 0)))
        self.manifest["measurements"] = measurements
        self._save_manifest()
        self.pending_summary = None
        self.current_index += 1
        next_step = self.current_step()
        if next_step is not None and next_step.condition_number == entry.get("condition_number"):
            next_status = (
                f"Accepted. Repeat {next_step.repeat_number}/{next_step.repeat_total} is next for this same condition."
            )
        else:
            next_status = "Accepted. Set up the next condition, then click Start measurement."
        self.show_step(next_status)

    @Slot()
    def redo_measurement(self) -> None:
        step = self.current_step()
        if step is None:
            return
        path = self.output_path_for_step(step)
        if path.exists():
            path.unlink()
        self.pending_summary = None
        self.show_step("Previous capture deleted. Re-establish the setup and click Start measurement.")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API name.
        if self.active_worker is not None:
            self.active_worker.request_abort()
        if self.active_thread is not None:
            self.active_thread.quit()
            self.active_thread.wait(4000)
        event.accept()


def build_parser(
    default_config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    description: str | None = None,
    enable_default_config_write: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=default_config_path, help="JSON experiment configuration.")
    parser.add_argument("--port", help="A121 Interface A serial port. Omit to auto-detect the WCH device.")
    parser.add_argument("--start-step", type=int, default=1, help="1-based expanded measurement run to start at.")
    parser.add_argument("--end-step", type=int, help="1-based expanded measurement run to stop after, inclusive.")
    parser.add_argument("--session-dir", type=Path, help="Existing/new output session directory.")
    parser.add_argument("--duration", type=float, help="Override measurement_seconds from the config.")
    parser.add_argument("--prep-seconds", type=int, help="Override prep_seconds from the config.")
    parser.add_argument("--output-dir", type=Path, help="Override output_dir from the config.")
    if enable_default_config_write:
        parser.add_argument(
            "--write-default-config",
            action="store_true",
            help="Write the built-in default JSON config and exit.",
        )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the sequence without opening the GUI or sensor.")
    return parser


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    if args.duration is not None:
        result["measurement_seconds"] = args.duration
    if args.prep_seconds is not None:
        result["prep_seconds"] = args.prep_seconds
    if args.output_dir is not None:
        result["output_dir"] = str(args.output_dir)
    return result


def main(
    *,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
    description: str | None = None,
    enable_default_config_write: bool = True,
) -> int:
    parser = build_parser(
        default_config_path,
        description=description,
        enable_default_config_write=enable_default_config_write,
    )
    args = parser.parse_args()
    if getattr(args, "write_default_config", False):
        args.config.parent.mkdir(parents=True, exist_ok=True)
        with args.config.open("w", encoding="utf-8") as handle:
            json.dump(default_config(), handle, indent=2, ensure_ascii=False)
        print(f"Wrote default config: {args.config}")
        return 0

    try:
        config = apply_overrides(load_config(args.config), args)
        steps = parse_steps(config)
        breathing_cues = parse_breathing_protocol(config)
        a121_config = make_a121_config(config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if not 1 <= args.start_step <= len(steps):
        parser.error(f"--start-step must be between 1 and {len(steps)}.")
    end_step = args.end_step or len(steps)
    if not args.start_step <= end_step <= len(steps):
        parser.error(f"--end-step must be between --start-step and {len(steps)}.")

    if args.dry_run:
        print(f"Experiment: {config.get('experiment_name', '')}")
        print(f"Preparation: {config.get('prep_seconds')} s | measurement: {config.get('measurement_seconds')} s")
        print(f"A121 config: {a121_config}")
        print(f"Port: {args.port or discover_a121_port() or 'auto-detect failed'}")
        for step in steps[args.start_step - 1 : end_step]:
            condition_id = f" [{step.condition_id}]" if step.condition_id else ""
            geometry = f" | {step.geometry_label}" if step.geometry_label else ""
            print(
                f"{step.number:02d}.{condition_id} {step.patch_label}{geometry} | {step.distance_cm:g} cm | "
                f"{step.lens_label} | repeat {step.repeat_number}/{step.repeat_total} | {step.instruction}"
            )
        if breathing_cues:
            print("Breathing protocol:")
            for cue in breathing_cues:
                print(f"  {cue.start_s:5.1f}–{cue.end_s:5.1f} s  {cue.cue}")
        return 0

    output_dir = Path(config.get("output_dir", "data/raw/a121/guided"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    session_dir = args.session_dir
    if session_dir is None:
        session_prefix = safe_filename(str(config.get("session_prefix", "guided")))
        session_dir = output_dir / f"{session_prefix}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    elif not session_dir.is_absolute():
        session_dir = ROOT / session_dir

    app = QApplication.instance() or QApplication(sys.argv)
    window = GuidedExperimentWindow(
        config=config,
        steps=steps,
        breathing_cues=breathing_cues,
        a121_config=a121_config,
        start_step=args.start_step,
        end_step=end_step,
        port=args.port,
        session_dir=session_dir,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
