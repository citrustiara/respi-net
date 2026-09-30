#!/usr/bin/env python3
"""Guided A121 radar + iPhone IMU recordings for the breathing-phase network.

The Acconeer A121 is the network's input; the iPhone running RespiPhoneIMU lies
on the body and records chest/abdomen motion over BLE as the phase reference
(labels are derived from it later).  Each trial is one breathing-pattern file
from ``configs/breathing_patterns`` and lasts exactly as long as the pattern;
the shared breathing coach shows it as a timeline, the cue, the time left and
what comes next, and beeps at every new phase.

The default session plays the ``nn_*`` patterns in a fixed order: free
breathing first (before any paced exercise can alter it), then varied rates,
holds after inhaling and after exhaling, shallow/deep breathing and an
irregular rhythm.

Every accepted trial leaves three files beside the session's ``manifest.json``::

    run_XX_<pattern-id>_a121.csv    A121 Sparse IQ frames (A121_CAPTURE_COLUMNS)
    run_XX_<pattern-id>_iphone.csv  iPhone IMU samples (IMU_COLUMNS)
    run_XX_<pattern-id>_cues.csv    one row per pattern phase, with host wall times

A trial is rejected automatically when the iPhone stream is incomplete (the
same coverage/gap gate as the LSM6DS3 + iPhone study) or the A121 delivered too
few frames.  Esc, or ``Przerwij i odrzuć``, discards the running trial and
deletes its files; after a completed recording choose ``Zachowaj próbę`` or
``Odrzuć i powtórz``.

Run from the repository root::

    uv run python tools/a121_iphone_guided_recording.py --distance-cm 80 --posture supine

Print the plan without touching hardware or opening the window::

    uv run python tools/a121_iphone_guided_recording.py --dry-run

Choose patterns (ids or paths) and resume an interrupted session::

    uv run python tools/a121_iphone_guided_recording.py --patterns nn_holds,nn_irregular \\
        --session-name subject01 --start-trial 2
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Sequence

LOCK_NAME = ".recorder.lock"
AUTOSTART_DELAY_MS = 3000
AUTOSTART_RETRY_MS = 8000
AUTOSTART_NEXT_TRIAL_MS = 15000
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for _import_dir in (SRC, TOOLS):
    if str(_import_dir) not in sys.path:
        sys.path.insert(0, str(_import_dir))

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from respi_net.a121 import A121_CAPTURE_COLUMNS, A121Capture, A121Config, find_a121_serial_ports
from respi_net.breathing import BreathingPattern, PatternError, format_clock, load_pattern
from respi_net.breathing_coach import BreathingCoach
from respi_net.imu import IMU_COLUMNS
from respi_net.imu_guided_protocol import (
    counter_delta,
    delete_trial_outputs,
    sample_timing_summary,
    stream_coverage_summary,
)
from respi_net.iphone_imu import IPHONE_IMU_DEVICE_NAME, IPhoneIMUBluetoothCapture

# The iPhone completeness gate is the one of the LSM6DS3 + iPhone study, reused
# as is so both datasets reject incomplete phone streams by the same rule.
from imu_lsm6ds3_iphone_guided_experiment import (  # noqa: E402
    IPHONE_EXPECTED_SAMPLE_RATE_HZ,
    IPHONE_MAX_GAP_S,
    IPHONE_MIN_COVERAGE_PERCENT,
    MeasurementWorker as _PhoneStudyWorker,
)


TOOL_NAME = "tools/a121_iphone_guided_recording.py"
TOOL_VERSION = "1.0"
TOOL_DESCRIPTION = (
    "Guided A121 radar + iPhone IMU (RespiPhoneIMU over BLE) recordings for the breathing-phase network: "
    "one breathing-pattern file per trial, the iPhone IMU is the phase reference."
)
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "nn" / "a121_iphone"
DEFAULT_PATTERN_IDS = ("nn_free", "nn_varied_rates", "nn_holds", "nn_shallow_deep", "nn_irregular")

# Radar settings of configs/a121_foil_2m_retest.json, i.e. breathing at up to ~2 m.
A121_BASE_SETTINGS: dict[str, Any] = {
    "profile": 3,
    "hwaas": 32,
    "sweeps_per_frame": 8,
    "frame_rate_hz": 20.0,
    "step_length": 1,
}
DEFAULT_A121_START_M = 0.2
DEFAULT_A121_END_M = 2.5
# With --distance-cm the range is the chest distance ± this, as in the 2 m retest (1.5-2.5 m).
A121_HALF_WINDOW_M = 0.5
A121_MIN_START_M = 0.1
# The WCH Dual_Serial interface may refuse a new session for a few seconds after
# the previous one stopped (see tools/a121_guided_experiment.py).
A121_CONNECT_ATTEMPTS = 2
A121_RECONNECT_DELAY_S = 5.0
A121_FIRST_FRAMES = 10
A121_FIRST_FRAMES_TIMEOUT_S = 5.0
A121_MIN_FRAME_COVERAGE_PERCENT = 90.0
A121_MAX_GAP_S = 0.5

IPHONE_FIRST_SAMPLES = 12
IPHONE_FIRST_SAMPLES_TIMEOUT_S = 3.0
# Both streams keep running this long after the pattern ends, so samples still
# in flight (BLE batching, A121 processing queue) at the end are not lost.
POST_ROLL_S = 1.0

CUE_COLUMNS = ["start_s", "end_s", "kind", "cue", "detail", "section", "start_wall_ms", "end_wall_ms"]

PHONE_PLACEMENT_NOTE = (
    "iPhone running RespiPhoneIMU lies flat on the upper abdomen / lower sternum, screen up, top edge toward "
    "the head, strapped in place. The A121 is aimed at the upper chest and the phone is kept at least 10 cm "
    "from that spot, because the phone is a strong reflector that would dominate the radar echo. "
    "The subject keeps still."
)
PHONE_PLACEMENT_PL = (
    "Telefon połóż płasko, ekranem do góry i górną krawędzią w stronę głowy, na górnej części brzucha / "
    "dolnej części mostka, i przymocuj go pasem. Radar A121 skieruj na górną część klatki piersiowej. "
    "Telefonu NIE kładź w miejscu, na które celuje radar — to silny reflektor, który zdominowałby echo; "
    "trzymaj go co najmniej 10 cm od tego miejsca."
)
TIME_ALIGNMENT_NOTE = (
    "All times are host epoch milliseconds (time.time()). A121 Timestamp_ms follows the sensor tick clock, "
    "anchored to the host clock when the A121 session started. iPhone Time_ms follows the phone's sample clock, "
    "anchored once to the host clock at the arrival of the first BLE packet, so it runs late by about one packet "
    "span plus BLE latency; iphone.ble_arrival.stamp_lateness_estimate_ms estimates that lag (subtract it from "
    "Time_ms; the minimum BLE latency remains). Cue start_wall_ms/end_wall_ms are measurement_start_wall_ms plus "
    "the pattern times. Sensor files hold the rows that arrived from the measurement start until the sensors were "
    "stopped post_roll_s after the pattern ended (the A121 file may open with a few frames still queued from just "
    "before the start); crop them with the cue wall times."
)

POSTURE_LABELS_PL = {
    "supine": "leżenie na plecach",
    "sitting": "siedzenie",
    "side": "leżenie na boku",
    "side_left": "leżenie na lewym boku",
    "side_right": "leżenie na prawym boku",
    "prone": "leżenie na brzuchu",
    "standing": "stanie",
}


# -- plan -------------------------------------------------------------------------------


def safe_name(value: str, fallback: str = "session") -> str:
    """A filesystem-safe name: letters, digits, ``_``, ``.`` and ``-`` only."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or fallback


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class Trial:
    """One recording: one breathing pattern, as long as the pattern."""

    number: int
    pattern: BreathingPattern

    @property
    def pattern_id(self) -> str:
        return safe_name(self.pattern.pattern_id or self.pattern.name, "pattern")

    @property
    def duration_s(self) -> float:
        return float(self.pattern.duration_s)

    @property
    def stem(self) -> str:
        return f"run_{self.number:02d}_{self.pattern_id}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern.name,
            "pattern_description": self.pattern.description,
            "pattern_source": _display_path(self.pattern.source),
            "duration_s": round(self.duration_s, 6),
            "phases": len(self.pattern.phases),
            "sections": [
                {
                    "index": section.index,
                    "start_s": round(section.start_s, 6),
                    "end_s": round(section.end_s, 6),
                    "kind": section.kind,
                    "label": section.label,
                }
                for section in self.pattern.sections
            ],
        }


def parse_pattern_list(value: str | Sequence[str]) -> list[str]:
    """Pattern ids or paths from ``--patterns``: a comma-separated list."""

    items = value.split(",") if isinstance(value, str) else list(value)
    references = [str(item).strip() for item in items if str(item).strip()]
    if not references:
        raise ValueError("--patterns: podaj co najmniej jeden wzorzec (id z configs/breathing_patterns albo ścieżkę).")
    return references


def build_trials(pattern_refs: Sequence[str | Path] = DEFAULT_PATTERN_IDS) -> list[Trial]:
    """One trial per pattern, numbered from 1 in the given order.

    Raises :class:`~respi_net.breathing.PatternError` naming the available
    patterns when one cannot be found or does not load.
    """

    trials = [Trial(number, load_pattern(reference)) for number, reference in enumerate(pattern_refs, start=1)]
    if not trials:
        raise ValueError("Plan sesji nie zawiera żadnej próby.")
    return trials


def output_paths(session_dir: Path, trial: Trial) -> dict[str, Path]:
    return {
        "a121": session_dir / f"{trial.stem}_a121.csv",
        "iphone": session_dir / f"{trial.stem}_iphone.csv",
        "cues": session_dir / f"{trial.stem}_cues.csv",
    }


def cue_rows(pattern: BreathingPattern, start_wall_ms: float) -> list[dict[str, Any]]:
    """The cue sidecar: every phase with its pattern time and host wall time."""

    return [
        {
            "start_s": phase.start_s,
            "end_s": phase.end_s,
            "kind": phase.kind,
            "cue": phase.cue,
            "detail": phase.detail,
            "section": phase.section,
            "start_wall_ms": start_wall_ms + phase.start_s * 1000.0,
            "end_wall_ms": start_wall_ms + phase.end_s * 1000.0,
        }
        for phase in pattern.phases
    ]


def a121_config_for(
    *,
    distance_cm: float | None = None,
    start_m: float | None = None,
    end_m: float | None = None,
) -> A121Config:
    """Radar settings for breathing at up to ~2 m.

    The range is ``--a121-start-m``/``--a121-end-m`` when given, otherwise the
    chest distance ± 0.5 m, otherwise 0.2-2.5 m.
    """

    if distance_cm is not None:
        centre_m = float(distance_cm) / 100.0
        auto_start, auto_end = max(A121_MIN_START_M, centre_m - A121_HALF_WINDOW_M), centre_m + A121_HALF_WINDOW_M
    else:
        auto_start, auto_end = DEFAULT_A121_START_M, DEFAULT_A121_END_M
    start = auto_start if start_m is None else float(start_m)
    end = auto_end if end_m is None else float(end_m)
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0.0:
        raise ValueError("Zakres A121 musi być skończony i nieujemny.")
    if end - start < 0.05:
        raise ValueError(f"Zakres A121 {start:g}-{end:g} m jest pusty: koniec musi leżeć co najmniej 5 cm za początkiem.")
    return A121Config(start_m=round(start, 3), end_m=round(end, 3), **A121_BASE_SETTINGS)


def normalize_posture(value: str) -> str:
    return re.sub(r"\s+", "_", str(value).strip().lower()) or "unspecified"


def posture_label(posture: str) -> str:
    return POSTURE_LABELS_PL.get(posture, posture)


@dataclass(frozen=True)
class SessionSettings:
    """Everything about a session that is not a trial."""

    session_dir: Path
    posture: str = "supine"
    distance_cm: float | None = None
    a121_port: str | None = None
    a121_config: A121Config = field(default_factory=a121_config_for)
    iphone_device: str | None = None
    iphone_scan_timeout_s: float = 10.0
    iphone_autostart: bool = True
    iphone_expected_sample_rate_hz: float = IPHONE_EXPECTED_SAMPLE_RATE_HZ
    prep_seconds: float = 5.0
    post_roll_s: float = POST_ROLL_S

    def setup_dict(self) -> dict[str, Any]:
        return {
            "posture": self.posture,
            "distance_cm": self.distance_cm,
            "phone_placement": PHONE_PLACEMENT_NOTE,
            "prep_seconds": self.prep_seconds,
            "post_roll_s": self.post_roll_s,
        }


def describe_a121_config(config: A121Config) -> str:
    return (
        f"zakres {config.start_m:.2f}–{config.end_m:.2f} m, profil {config.profile}, HWAAS {config.hwaas}, "
        f"{config.sweeps_per_frame} przemiatań/ramkę, {config.frame_rate_hz:g} Hz"
    )


def plan_lines(trials: Sequence[Trial], settings: SessionSettings, *, start_trial: int = 1) -> list[str]:
    """The session plan, as ``--dry-run`` prints it."""

    selected = [trial for trial in trials if trial.number >= start_trial]
    recording_s = sum(trial.duration_s for trial in selected)
    with_prep_s = recording_s + settings.prep_seconds * len(selected)
    width = max(len(trial.pattern_id) for trial in trials)
    lines = ["Plan nagrań A121 + iPhone (sieć faz oddechu):"]
    for trial in trials:
        skipped = "  (pominięta: --start-trial)" if trial.number < start_trial else ""
        lines.append(
            f"  {trial.number:02d}. {trial.pattern_id:<{width}}  {format_clock(trial.duration_s):>5}  "
            f"{trial.pattern.name}{skipped}"
        )
    lines += [
        f"Łącznie {len(selected)} prób: {format_clock(recording_s)} nagrań ({recording_s:.1f} s); "
        f"z przygotowaniem {settings.prep_seconds:g} s przed każdą: {format_clock(with_prep_s)} "
        "(bez łączenia z czujnikami i oceny prób).",
        f"Pozycja: {settings.posture} ({posture_label(settings.posture)}); odległość radaru od klatki: "
        + (f"{settings.distance_cm:g} cm." if settings.distance_cm is not None else "nie podano."),
        f"A121: {describe_a121_config(settings.a121_config)}; port: "
        + (settings.a121_port or "auto-detekcja przy starcie próby") + ".",
        f"iPhone: {settings.iphone_device or IPHONE_IMU_DEVICE_NAME} przez BLE, oczekiwane "
        f"{settings.iphone_expected_sample_rate_hz:g} Hz, "
        + ("START/STOP wysyłane przez BLE." if settings.iphone_autostart else "strumieniem steruje aplikacja."),
        f"Katalog sesji: {settings.session_dir}",
        "Pliki próby: run_XX_<wzorzec>_a121.csv, run_XX_<wzorzec>_iphone.csv, run_XX_<wzorzec>_cues.csv; "
        "plus manifest.json.",
    ]
    return lines


# -- quality gates and summaries -----------------------------------------------------


def iphone_quality_error(coverage: dict[str, Any], *, duration_s: float) -> str | None:
    """The LSM6DS3 + iPhone study's gate: coverage, largest gap and both ends of the window."""

    return _PhoneStudyWorker._iphone_quality_error(coverage, duration_s=duration_s)


def iphone_time_axis_error(rows: Sequence[Sequence[float]], ble_batches: dict[str, int]) -> str | None:
    """A restarted phone stream: the coverage gate sorts time stamps, so it cannot see one.

    The capture anchors the phone clock to the host clock only once, so after
    the app restarts its stream the time stamps jump back and overlap.
    """

    resets = int(ble_batches.get("sequence_resets", 0) or 0)
    if resets:
        return (
            f"aplikacja iPhone zrestartowała strumień w trakcie próby ({resets}× reset numeracji pakietów), "
            "więc oś czasu próbek jest niespójna."
        )
    times = np.asarray([row[0] for row in rows], dtype=float)
    backwards = int(np.sum(np.diff(times) < 0.0)) if len(times) > 1 else 0
    if backwards:
        return f"znaczniki czasu iPhone cofają się ({backwards}×) — strumień był restartowany."
    return None


def _median_column(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def summarize_a121_rows(
    rows: Sequence[Sequence[Any]],
    *,
    window_start_ms: float,
    duration_s: float,
    frame_rate_hz: float,
) -> dict[str, Any]:
    """Frame counts, rate and gaps of an A121 capture inside the trial window."""

    expected = int(round(max(0.0, float(duration_s)) * max(0.0, float(frame_rate_hz))))
    summary: dict[str, Any] = {
        "frames": len(rows),
        "expected_frames": expected,
        "frames_in_window": 0,
        "frame_coverage_percent": 0.0,
        "sample_rate_hz": None,
        "largest_gap_s": None,
        "first_frame_offset_s": None,
        "last_frame_offset_s": None,
        "median_peak_distance_m": None,
        "median_target_distance_m": None,
        "presence_percent": None,
    }
    if not rows:
        return summary
    frame = pd.DataFrame([list(row) for row in rows], columns=A121_CAPTURE_COLUMNS)
    stamps = pd.to_numeric(frame["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    stamps = np.sort(stamps[np.isfinite(stamps)])
    end_ms = window_start_ms + float(duration_s) * 1000.0
    inside = stamps[(stamps >= window_start_ms) & (stamps < end_ms)]
    steps = np.diff(stamps)
    steps = steps[steps > 0]
    summary.update(
        {
            "frames_in_window": int(len(inside)),
            "frame_coverage_percent": float(100.0 * len(inside) / expected) if expected else 100.0,
            "sample_rate_hz": float(1000.0 / np.median(steps)) if len(steps) else None,
            "median_peak_distance_m": _median_column(frame, "PeakDistance_m"),
            "median_target_distance_m": _median_column(frame, "AcconeerTargetDistance_m"),
        }
    )
    if len(inside):
        summary["first_frame_offset_s"] = float((inside[0] - window_start_ms) / 1000.0)
        summary["last_frame_offset_s"] = float((inside[-1] - window_start_ms) / 1000.0)
        summary["largest_gap_s"] = float(np.max(np.diff(inside)) / 1000.0) if len(inside) > 1 else 0.0
    if "AcconeerPresenceDetected" in frame:
        presence = pd.to_numeric(frame["AcconeerPresenceDetected"], errors="coerce").dropna()
        summary["presence_percent"] = float(presence.mean() * 100.0) if len(presence) else None
    return summary


def a121_quality_error(summary: dict[str, Any], *, duration_s: float) -> str | None:
    """Too few frames in the trial window, a hole in the stream, or a missing start or end."""

    expected = int(summary.get("expected_frames") or 0)
    frames = int(summary.get("frames_in_window") or 0)
    minimum = max(10, math.ceil(expected * A121_MIN_FRAME_COVERAGE_PERCENT / 100.0))
    if frames < minimum:
        return (
            f"A121 dostarczył tylko {frames} ramek w oknie próby (oczekiwano około {expected}, "
            f"wymagane co najmniej {minimum})."
        )
    largest_gap_s = float(summary.get("largest_gap_s") or 0.0)
    if largest_gap_s > A121_MAX_GAP_S:
        return f"A121 ma lukę {largest_gap_s:.2f} s między ramkami (dopuszczalne {A121_MAX_GAP_S:.2f} s)."
    first_offset_s = summary.get("first_frame_offset_s")
    last_offset_s = summary.get("last_frame_offset_s")
    edge_tolerance_s = max(0.5, 0.02 * float(duration_s))
    if (
        first_offset_s is None
        or last_offset_s is None
        or float(first_offset_s) > edge_tolerance_s
        or float(last_offset_s) < float(duration_s) - edge_tolerance_s
    ):
        return "ramki A121 nie obejmują początku albo końca próby."
    return None


def arrival_lag_summary(lags_ms: Sequence[float]) -> dict[str, Any] | None:
    """How much later than their time stamps the iPhone's BLE packets reached the host."""

    if not lags_ms:
        return None
    values = np.asarray(lags_ms, dtype=float)
    return {
        "packets": int(len(values)),
        "arrival_minus_last_stamp_ms": {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        },
        "stamp_lateness_estimate_ms": float(max(0.0, -np.min(values))),
    }


def _value(raw: Any, digits: int = 1, suffix: str = "") -> str:
    if raw is None:
        return "b.d."
    try:
        return f"{float(raw):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(raw)


def format_summary(summary: dict[str, Any]) -> str:
    a121 = summary.get("a121", {}) or {}
    iphone = summary.get("iphone", {}) or {}
    coverage = iphone.get("window_coverage", {}) or {}
    batches = iphone.get("ble_batches", {}) or {}
    arrival = iphone.get("ble_arrival") or {}
    lines = [
        f"A121: {a121.get('frames', 0)} ramek; w oknie próby {a121.get('frames_in_window', 0)}/"
        f"{a121.get('expected_frames', 0)} ({_value(a121.get('frame_coverage_percent'))}%), "
        f"{_value(a121.get('sample_rate_hz'), 1, ' Hz')}, utracone wyniki {a121.get('dropped_results', 0)}, "
        f"największa luka {_value(a121.get('largest_gap_s'), 3, ' s')}.",
        f"A121: mediana odległości maksimum echa {_value(a121.get('median_peak_distance_m'), 2, ' m')}, "
        f"cel Acconeer {_value(a121.get('median_target_distance_m'), 2, ' m')}, "
        f"obecność {_value(a121.get('presence_percent'), 0, '%')}.",
        f"iPhone: {iphone.get('rows', 0)} próbek, fs {_value(iphone.get('sample_rate_hz'), 1, ' Hz')}, "
        f"szacowane luki czasowe {iphone.get('estimated_missing_samples', 0)} "
        f"({_value(iphone.get('estimated_missing_percent'), 3, '%')}).",
        f"Pokrycie okna iPhone: {_value(coverage.get('sample_coverage_percent'))}% próbek, "
        f"{_value(coverage.get('time_coverage_percent'))}% czasu, największa luka "
        f"{_value(coverage.get('largest_gap_s'), 3, ' s')}.",
        f"BLE: brakujące pakiety={batches.get('missing_batches', 0)}, niepoprawne={batches.get('invalid_batches', 0)}, "
        f"duplikaty={batches.get('duplicate_batches', 0)}, resety sekwencji={batches.get('sequence_resets', 0)}.",
    ]
    if arrival:
        lines.append(
            f"Znaczniki czasu iPhone spóźniają się o ok. {_value(arrival.get('stamp_lateness_estimate_ms'), 0, ' ms')} "
            f"względem przyjścia pakietów ({arrival.get('packets', 0)} pakietów; zapisane w manifeście)."
        )
    lines.extend(f"{name}: {path}" for name, path in summary.get("files", {}).items())
    return "\n".join(lines)


# -- manifest -----------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write through a temporary file so a crash never leaves half a manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def new_manifest(trials: Sequence[Trial], settings: SessionSettings) -> dict[str, Any]:
    return {
        "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION,
        "description": TOOL_DESCRIPTION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(settings.session_dir),
        "posture": settings.posture,
        "distance_cm": settings.distance_cm,
        "phone_placement": PHONE_PLACEMENT_NOTE,
        "a121_port": settings.a121_port,
        "a121_config": asdict(settings.a121_config),
        "iphone": {
            "device": settings.iphone_device or IPHONE_IMU_DEVICE_NAME,
            "expected_sample_rate_hz": settings.iphone_expected_sample_rate_hz,
            "autostart": settings.iphone_autostart,
            "min_coverage_percent": IPHONE_MIN_COVERAGE_PERCENT,
            "max_gap_s": IPHONE_MAX_GAP_S,
        },
        "prep_seconds": settings.prep_seconds,
        "post_roll_s": settings.post_roll_s,
        "time_alignment": TIME_ALIGNMENT_NOTE,
        "trials": [trial.as_dict() for trial in trials],
        "measurements": [],
    }


def _planned(trials_payload: Sequence[dict[str, Any]]) -> list[tuple[Any, Any]]:
    return [(item.get("number"), item.get("pattern_id")) for item in trials_payload]


def load_or_create_manifest(path: Path, fresh: dict[str, Any]) -> dict[str, Any]:
    """The session's manifest: the existing one when resuming, else ``fresh``.

    A session directory belongs to one plan: resuming it with other patterns
    would mix trial numbers, so that raises :class:`ValueError`.  An unreadable
    manifest is kept aside as ``manifest.json.bak`` rather than overwritten.
    """

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
            path.replace(path.with_name(path.name + ".bak"))
        if isinstance(existing, dict):
            recorded = _planned(existing.get("trials", []))
            planned = _planned(fresh.get("trials", []))
            if recorded and recorded != planned:
                raise ValueError(
                    f"Sesja {path.parent} ma inny plan prób ({', '.join(str(item[1]) for item in recorded)}). "
                    "Użyj innej --session-name albo tych samych --patterns."
                )
            return existing
    save_manifest(path, fresh)
    return fresh


def claim_session(session_dir: Path) -> Path:
    """Mark ``session_dir`` as in use by this recorder; :class:`ValueError` if another one is running there.

    Two recorders on one session share the radar port and the phone and overwrite each other's
    trials (recording a trial again replaces its files), so the second one is refused.  A lock
    left by a crashed recorder is taken over.
    """

    session_dir.mkdir(parents=True, exist_ok=True)
    lock = session_dir / LOCK_NAME
    if lock.exists():
        try:
            other = int(lock.read_text(encoding="utf-8").strip())
            os.kill(other, 0)
        except (ValueError, OSError):
            pass  # unreadable lock or no such process: stale
        else:
            if other != os.getpid():
                raise ValueError(
                    f"Sesja {session_dir} jest już używana przez inny rejestrator (PID {other}). "
                    "Zamknij go albo użyj innej --session-name."
                )
    lock.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: lock.unlink(missing_ok=True))
    return lock


# -- capture ------------------------------------------------------------------------


class ArrivalLoggingIPhoneCapture(IPhoneIMUBluetoothCapture):
    """The shared iPhone BLE capture, also noting when each packet reached the host.

    The shared capture stamps samples with the phone's clock, anchored once to
    the host clock when the first packet arrives, and keeps no arrival times.
    For every stored packet this keeps ``host arrival - time stamp of its last
    sample``; the minimum over a trial estimates how late the stamps run.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._arrival_lock = threading.Lock()
        self._arrival_lags_ms: list[float] = []

    def connect(self, port_name: str | None = None) -> bool:
        with self._arrival_lock:
            self._arrival_lags_ms = []
        return super().connect(port_name)

    def _on_notification(self, sender: Any, payload: bytearray) -> None:
        arrival_ms = time.time() * 1000.0
        before = self.data_count()
        super()._on_notification(sender, payload)
        stored = self.snapshot_data_since(before)
        if stored:
            with self._arrival_lock:
                self._arrival_lags_ms.append(arrival_ms - float(stored[-1][0]))

    def arrival_count(self) -> int:
        with self._arrival_lock:
            return len(self._arrival_lags_ms)

    def snapshot_arrival_lags_since(self, index: int) -> list[float]:
        with self._arrival_lock:
            return list(self._arrival_lags_ms[index:])


def discover_a121_port() -> str | None:
    candidates = find_a121_serial_ports()
    return candidates[0] if candidates else None


class MeasurementWorker(QObject):
    """Connects both sensors, counts in, records one pattern, checks and writes it."""

    status_changed = Signal(str)
    prep_changed = Signal(int)
    progress_changed = Signal(float, float)
    finished = Signal(object)
    failed = Signal(str)
    aborted = Signal()

    def __init__(self, *, trial: Trial, settings: SessionSettings, a121_settle_s: float = 0.0) -> None:
        super().__init__()
        self.trial = trial
        self.settings = settings
        self.a121_settle_s = max(0.0, float(a121_settle_s))
        self.session_dir = settings.session_dir
        self.paths = output_paths(settings.session_dir, trial)
        self.abort_event = threading.Event()
        self.a121: Any | None = None
        self.iphone: Any | None = None
        self.resolved_a121_port: str | None = None
        self.a121_geometry: dict[str, Any] = {}

    def request_abort(self) -> None:
        self.abort_event.set()

    def _delete_outputs(self) -> None:
        delete_trial_outputs(self.paths)

    def _stop(self) -> None:
        for capture in (self.a121, self.iphone):
            if capture is not None:
                try:
                    capture.stop()
                except Exception:
                    pass

    def _abort(self) -> None:
        self.status_changed.emit("Przerwano próbę; pliki bieżącego zapisu są usuwane…")
        self._stop()
        self._delete_outputs()
        self.aborted.emit()

    def _wait_for_rows(self, capture: Any, minimum_rows: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.abort_event.is_set():
                return False
            if capture.data_count() >= minimum_rows:
                return True
            time.sleep(0.02)
        return capture.data_count() >= minimum_rows

    def _check_streams(self) -> None:
        if self.a121 is not None and not self.a121.running:
            raise RuntimeError("Transmisja A121 zatrzymała się przed końcem próby.")
        if self.iphone is not None and not self.iphone.running:
            raise RuntimeError("Transmisja iPhone BLE zatrzymała się przed końcem próby.")

    def _wait(self, seconds: float) -> bool:
        """Wait while watching both streams; ``True`` when the trial was aborted."""

        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return self.abort_event.is_set()
            if self.abort_event.wait(min(0.05, remaining)):
                return True
            self._check_streams()

    def _connect_a121(self) -> None:
        port = self.settings.a121_port or discover_a121_port()
        if not port:
            raise RuntimeError(
                "Nie znaleziono portu A121 (WCH USB Dual_Serial, Interface A). Podłącz radar albo podaj --a121-port."
            )
        self.resolved_a121_port = port
        if self.a121_settle_s > 0.0:
            self.status_changed.emit(f"Czekam {self.a121_settle_s:.0f} s, aż interfejs A121 będzie gotowy na nową sesję…")
            if self.abort_event.wait(self.a121_settle_s):
                return
        for attempt in range(1, A121_CONNECT_ATTEMPTS + 1):
            if self.abort_event.is_set():
                return
            self.status_changed.emit(f"Łączenie z A121: {port} (próba {attempt}/{A121_CONNECT_ATTEMPTS})…")
            capture = A121Capture(output_dir=self.session_dir, config=self.settings.a121_config)
            if capture.connect(port):
                self.a121 = capture
                break
            if attempt < A121_CONNECT_ATTEMPTS:
                self.status_changed.emit(f"A121 nie odpowiedział; ponowna próba za {A121_RECONNECT_DELAY_S:.0f} s…")
                if self.abort_event.wait(A121_RECONNECT_DELAY_S):
                    return
        else:
            raise RuntimeError("A121 nie odpowiedział. Sprawdź, czy wybrano Interface A i czy radar jest zasilany.")
        self.a121_geometry = self._a121_geometry(self.a121)
        if not self._wait_for_rows(self.a121, A121_FIRST_FRAMES, A121_FIRST_FRAMES_TIMEOUT_S):
            if self.abort_event.is_set():
                return
            raise RuntimeError("A121 połączył się, ale nie dostarcza ramek.")

    @staticmethod
    def _a121_geometry(capture: Any) -> dict[str, Any]:
        """The sensor settings actually used (the capture adapts step and sweeps to the UART)."""

        sensor_config = getattr(capture, "sensor_config", None)
        distances = np.asarray(getattr(capture, "distances_m", []), dtype=float)

        def integer(name: str) -> int | None:
            value = getattr(sensor_config, name, None)
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        return {
            "start_point": integer("start_point"),
            "num_points": integer("num_points"),
            "step_length": integer("step_length"),
            "sweeps_per_frame": integer("sweeps_per_frame"),
            "first_distance_m": float(distances[0]) if len(distances) else None,
            "last_distance_m": float(distances[-1]) if len(distances) else None,
        }

    def _connect_iphone(self) -> None:
        self.status_changed.emit("Łączenie z aplikacją RespiPhoneIMU przez BLE…")
        self.iphone = ArrivalLoggingIPhoneCapture(
            output_dir=self.session_dir,
            device=self.settings.iphone_device,
            scan_timeout_s=self.settings.iphone_scan_timeout_s,
            autostart=self.settings.iphone_autostart,
        )
        if not self.iphone.connect(self.settings.iphone_device):
            detail = str(getattr(self.iphone, "status", "") or "")
            raise RuntimeError("Nie połączono z aplikacją RespiPhoneIMU przez BLE." + (f" ({detail})" if detail else ""))
        if not self._wait_for_rows(self.iphone, IPHONE_FIRST_SAMPLES, IPHONE_FIRST_SAMPLES_TIMEOUT_S):
            if self.abort_event.is_set():
                return
            raise RuntimeError(
                "iPhone połączył się, ale nie wysłał poprawnych próbek IMU. "
                "Czy aplikacja jest na pierwszym planie, a strumień uruchomiony?"
            )

    def _arrival_count(self) -> int:
        counter = getattr(self.iphone, "arrival_count", None)
        return int(counter()) if callable(counter) else 0

    def _arrival_lags_since(self, index: int) -> list[float]:
        snapshot = getattr(self.iphone, "snapshot_arrival_lags_since", None)
        return list(snapshot(index)) if callable(snapshot) else []

    @Slot()
    def run(self) -> None:
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._delete_outputs()
            self._connect_a121()
            if self.abort_event.is_set():
                self._abort()
                return
            self._connect_iphone()
            if self.abort_event.is_set():
                self._abort()
                return
            assert self.a121 is not None and self.iphone is not None
            settings = self.settings
            rate_hz = settings.iphone_expected_sample_rate_hz
            self.status_changed.emit("Radar i telefon przesyłają dane. Pozostań nieruchomo i przygotuj się.")

            iphone_preflight_start = self.iphone.data_count()
            preflight_start_wall_ms = time.time() * 1000.0
            remaining = float(settings.prep_seconds)
            while remaining > 1e-9:
                self.prep_changed.emit(max(1, math.ceil(remaining - 1e-9)))
                step = min(1.0, remaining)
                if self._wait(step):
                    self._abort()
                    return
                remaining -= step
            if settings.prep_seconds >= 3:
                preflight_coverage = stream_coverage_summary(
                    self.iphone.snapshot_data_since(iphone_preflight_start),
                    window_start_ms=preflight_start_wall_ms,
                    expected_duration_s=float(settings.prep_seconds),
                    expected_sample_rate_hz=rate_hz,
                )
                preflight_error = iphone_quality_error(preflight_coverage, duration_s=float(settings.prep_seconds))
                if preflight_error is not None:
                    raise RuntimeError(
                        "Nieudana kontrola transmisji iPhone przed próbą: " + preflight_error + " "
                        f"Pozostaw aplikację na pierwszym planie, odblokowany ekran i tryb {rate_hz:g} Hz; "
                        "uruchom próbę ponownie."
                    )

            duration_s = self.trial.duration_s
            a121_start = self.a121.data_count()
            a121_dropped_before = int(getattr(self.a121, "dropped_results", 0))
            iphone_start = self.iphone.data_count()
            iphone_counters_before = self.iphone.counter_snapshot()
            arrivals_start = self._arrival_count()
            start_wall_ms = time.time() * 1000.0
            started = time.monotonic()
            self.status_changed.emit("Nagrywanie — wykonuj komendy trenera oddechu.")
            while True:
                if self.abort_event.is_set():
                    self._abort()
                    return
                self._check_streams()
                elapsed_s = time.monotonic() - started
                self.progress_changed.emit(min(elapsed_s, duration_s), duration_s)
                if elapsed_s >= duration_s:
                    break
                if self.abort_event.wait(0.05):
                    self._abort()
                    return
            actual_seconds = time.monotonic() - started
            if settings.post_roll_s > 0.0:
                self.status_changed.emit("Koniec wzorca — jeszcze chwila zapisu, pozostań nieruchomo…")
                if self._wait(settings.post_roll_s):
                    self._abort()
                    return
            a121_dropped_after = int(getattr(self.a121, "dropped_results", 0))
            iphone_counters_after = self.iphone.counter_snapshot()
            stop_wall_ms = time.time() * 1000.0
            self.status_changed.emit("Zatrzymywanie czujników i sprawdzanie zapisu…")
            self._stop()
            if self.abort_event.is_set():
                self._abort()
                return

            end_wall_ms = start_wall_ms + duration_s * 1000.0
            a121_rows = self.a121.snapshot_data_since(a121_start)
            iphone_rows = self.iphone.snapshot_data_since(iphone_start)
            a121_summary = summarize_a121_rows(
                a121_rows,
                window_start_ms=start_wall_ms,
                duration_s=duration_s,
                frame_rate_hz=float(settings.a121_config.frame_rate_hz),
            )
            a121_error = a121_quality_error(a121_summary, duration_s=duration_s)
            if a121_error is not None:
                raise RuntimeError("Nie zaakceptowano zapisu A121: " + a121_error + " Bieżące pliki zostaną usunięte.")

            # The gate judges the trial window; samples of the post-roll only pad the file.
            window_rows = [row for row in iphone_rows if row[0] <= end_wall_ms]
            iphone_coverage = stream_coverage_summary(
                window_rows,
                window_start_ms=start_wall_ms,
                expected_duration_s=duration_s,
                expected_sample_rate_hz=rate_hz,
            )
            ble_batches = counter_delta(iphone_counters_after, iphone_counters_before)
            iphone_error = iphone_quality_error(iphone_coverage, duration_s=duration_s) or iphone_time_axis_error(
                iphone_rows, ble_batches
            )
            if iphone_error is not None:
                raise RuntimeError(
                    "Nie zaakceptowano niepełnego zapisu iPhone: " + iphone_error + " Bieżące pliki zostaną usunięte."
                )

            pd.DataFrame([list(row) for row in a121_rows], columns=A121_CAPTURE_COLUMNS).to_csv(
                self.paths["a121"], index=False
            )
            pd.DataFrame(iphone_rows, columns=IMU_COLUMNS).to_csv(self.paths["iphone"], index=False)
            pd.DataFrame(cue_rows(self.trial.pattern, start_wall_ms), columns=CUE_COLUMNS).to_csv(
                self.paths["cues"], index=False
            )

            iphone_summary: dict[str, Any] = dict(sample_timing_summary(iphone_rows))
            iphone_summary.update(
                {
                    "device": settings.iphone_device or IPHONE_IMU_DEVICE_NAME,
                    "expected_sample_rate_hz": rate_hz,
                    "window_coverage": iphone_coverage,
                    "ble_batches": ble_batches,
                    "ble_arrival": arrival_lag_summary(self._arrival_lags_since(arrivals_start)),
                }
            )
            summary = {
                "trial": self.trial.as_dict(),
                "setup": settings.setup_dict(),
                "measurement_start_utc": datetime.fromtimestamp(start_wall_ms / 1000.0, tz=timezone.utc).isoformat(),
                "measurement_start_wall_ms": start_wall_ms,
                "measurement_end_wall_ms": end_wall_ms,
                "capture_stop_wall_ms": stop_wall_ms,
                "actual_seconds": actual_seconds,
                "time_alignment": TIME_ALIGNMENT_NOTE,
                "files": {key: str(path) for key, path in self.paths.items()},
                "a121": {
                    "port": self.resolved_a121_port,
                    "config": asdict(settings.a121_config),
                    "effective": self.a121_geometry,
                    **a121_summary,
                    "dropped_results": max(0, a121_dropped_after - a121_dropped_before),
                    "dropped_results_since_connect": a121_dropped_after,
                },
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


# -- window -------------------------------------------------------------------------


class GuidedRecordingWindow(QWidget):
    def __init__(
        self, *, trials: Sequence[Trial], settings: SessionSettings, start_trial: int = 1, autostart_attempts: int = 0
    ) -> None:
        super().__init__()
        self.autostart_attempts = autostart_attempts
        self.autostart_left = autostart_attempts
        self.trials = list(trials)
        self.settings = settings
        self.session_dir = settings.session_dir
        self.current_index = start_trial - 1
        self.active_worker: MeasurementWorker | None = None
        self.active_thread: QThread | None = None
        self.pending_summary: dict[str, Any] | None = None
        self._a121_released_at: float | None = None
        self.manifest_path = self.session_dir / "manifest.json"
        self.manifest = load_or_create_manifest(self.manifest_path, new_manifest(self.trials, settings))

        self.setWindowTitle("A121 + iPhone — nagrania do sieci faz oddechu")
        self.resize(960, 900)
        self._build_ui()
        self.show_trial()
        self.abort_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.abort_shortcut.activated.connect(self.abort_current)
        if autostart_attempts > 0:
            QTimer.singleShot(AUTOSTART_DELAY_MS, self._autostart)

    @Slot()
    def _autostart(self) -> None:
        """Start the current trial by itself (``--autostart``): for recordings where the mouse is out of reach."""

        if self.autostart_attempts <= 0 or self.active_thread is not None or self.current_trial() is None:
            return
        if self.autostart_left <= 0:
            self.status.setText("Automatyczny start: wyczerpano próby połączenia.")
            return
        self.autostart_left -= 1
        self.start_measurement()

    @Slot()
    def _quit_app(self) -> None:
        QApplication.quit()

    def _save_manifest(self) -> None:
        save_manifest(self.manifest_path, self.manifest)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.trial_label = QLabel()
        self.trial_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        self.trial_label.setWordWrap(True)
        layout.addWidget(self.trial_label)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.instructions.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.instructions)

        self.coach = BreathingCoach("Wzorzec oddechu")
        layout.addWidget(self.coach, stretch=2)
        self.status = QLabel("Gotowy.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        buttons = QGridLayout()
        self.start_button = QPushButton("Start próby")
        self.abort_button = QPushButton("Przerwij i odrzuć (Esc)")
        self.keep_button = QPushButton("Zachowaj próbę")
        self.discard_button = QPushButton("Odrzuć i powtórz")
        # Red only while clickable, so a disabled discard button does not look armed.
        danger = "QPushButton:enabled { background: #9b1c1c; color: white; font-weight: bold; }"
        self.abort_button.setStyleSheet(danger)
        self.discard_button.setStyleSheet(danger)
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

    def _instructions_text(self, trial: Trial) -> str:
        settings = self.settings
        config = settings.a121_config
        distance = f"{settings.distance_cm:g} cm" if settings.distance_cm is not None else "nie podano"
        description = f" {trial.pattern.description}" if trial.pattern.description else ""
        return (
            f"Wzorzec {trial.pattern_id}: {format_clock(trial.duration_s)} ({trial.duration_s:.0f} s).{description}\n\n"
            f"Pozycja: {posture_label(settings.posture)}; odległość radaru od klatki: {distance}; "
            f"zakres A121 {config.start_m:.2f}–{config.end_m:.2f} m.\n"
            f"{PHONE_PLACEMENT_PL}\n\n"
            "W trakcie próby pozostań nieruchomo i wykonuj komendy trenera oddechu. "
            f"iPhone: {settings.iphone_expected_sample_rate_hz:g} Hz, aplikacja na pierwszym planie, ekran odblokowany. "
            "Niepełny strumień iPhone albo zbyt mało ramek A121 powoduje automatyczne odrzucenie próby."
        )

    def show_trial(self, message: str | None = None) -> None:
        trial = self.current_trial()
        self.pending_summary = None
        self.summary.clear()
        self.abort_button.setEnabled(False)
        self.keep_button.setEnabled(False)
        self.discard_button.setEnabled(False)
        if trial is None:
            self.trial_label.setText("Wszystkie próby są zakończone")
            self.instructions.setText(f"Zaakceptowane pliki i manifest znajdują się w:\n{self.session_dir}")
            self.start_button.setEnabled(False)
            self.status.setText(message or "Gotowe.")
            self.coach.set_pattern(None)
            self.coach.show_message("KONIEC SESJI")
            return
        self.trial_label.setText(f"Próba {trial.number}/{len(self.trials)} — {trial.pattern.name}")
        self.instructions.setText(self._instructions_text(trial))
        self.coach.set_pattern(trial.pattern)
        first = trial.pattern.phases[0]
        self.coach.show_idle(f"PO STARCIE: {first.cue}", first.detail, kind=first.kind)
        if message is None and self._accepted_entry(trial) is not None:
            message = "Ta próba jest już zachowana w manifeście; Start próby usunie ją i nagra od nowa."
        self.status.setText(
            message or "Sprawdź ułożenie telefonu i radaru, uruchom aplikację RespiPhoneIMU i wybierz Start próby."
        )
        self.start_button.setEnabled(True)

    def _a121_settle_s(self) -> float:
        if self._a121_released_at is None:
            return 0.0
        return max(0.0, A121_RECONNECT_DELAY_S - (time.monotonic() - self._a121_released_at))

    def _accepted_entry(self, trial: Trial) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.manifest.get("measurements", [])
                if int(item.get("trial", {}).get("number", -1)) == trial.number
            ),
            None,
        )

    def _forget_accepted(self, trial: Trial) -> None:
        """A new recording of an accepted trial starts by deleting its files, so drop its entry first."""

        entry = self._accepted_entry(trial)
        if entry is not None:
            self.manifest["measurements"] = [item for item in self.manifest["measurements"] if item is not entry]
            self._save_manifest()

    @Slot()
    def start_measurement(self) -> None:
        trial = self.current_trial()
        if trial is None or self.active_thread is not None:
            return
        self._forget_accepted(trial)
        self.start_button.setEnabled(False)
        self.abort_button.setEnabled(True)
        self.summary.clear()
        worker = MeasurementWorker(trial=trial, settings=self.settings, a121_settle_s=self._a121_settle_s())
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
        self.coach.show_countdown(
            remaining,
            self.settings.prep_seconds,
            detail="Nie ruszaj się; telefon i radar pozostają na swoich miejscach.",
        )
        self.status.setText(f"Przygotowanie: {remaining} s.")

    @Slot(float, float)
    def on_progress(self, elapsed_s: float, total_s: float) -> None:
        if self.current_trial() is None:
            return
        if self.coach.set_elapsed(elapsed_s):
            QApplication.beep()
        text = f"Nagrywanie: {format_clock(elapsed_s)} / {format_clock(total_s)} — wykonuj komendy trenera."
        if self.status.text() != text:
            self.status.setText(text)

    @Slot(object)
    def on_finished(self, result: object) -> None:
        self.pending_summary = dict(result)  # type: ignore[arg-type]
        self.abort_button.setEnabled(False)
        self.keep_button.setEnabled(True)
        self.discard_button.setEnabled(True)
        self.summary.setPlainText(format_summary(self.pending_summary))
        self.status.setText("Próba zapisana tymczasowo. Zachowaj ją albo odrzuć i powtórz.")
        self.coach.show_message("KONIEC", "Oddychaj swobodnie.", finished=True)
        if self.autostart_attempts > 0:
            # The worker only finishes when the stream checks passed, so the trial is kept as is.
            QTimer.singleShot(500, self.accept_pending)

    @Slot(str)
    def on_failed(self, message: str) -> None:
        self.abort_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.coach.show_message("PRÓBA ODRZUCONA", "Oddychaj swobodnie.", kind="hold")
        self.status.setText(f"Próba nieudana: {message}")
        self.summary.setPlainText(
            "Bieżące pliki usunięto. Popraw połączenie albo ułożenie czujników i uruchom próbę ponownie."
        )
        if self.autostart_attempts > 0:
            QTimer.singleShot(AUTOSTART_RETRY_MS, self._autostart)

    @Slot()
    def on_aborted(self) -> None:
        self.abort_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.coach.show_message("ODRZUCONO", "Oddychaj swobodnie.", kind="hold")
        self.status.setText("Bieżąca próba została odrzucona, a jej pliki usunięte.")
        self.summary.setPlainText("Przerwany zapis został usunięty.")

    @Slot()
    def on_thread_finished(self) -> None:
        thread = self.active_thread
        self.active_worker = None
        self.active_thread = None
        self._a121_released_at = time.monotonic()
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
        self.show_trial("Ukończoną próbę odrzucono. Przygotuj stanowisko i uruchom ją ponownie.")

    @Slot()
    def accept_pending(self) -> None:
        if self.pending_summary is None:
            return
        entry = dict(self.pending_summary)
        entry["status"] = "accepted"
        entry["accepted_at"] = datetime.now(timezone.utc).isoformat()
        number = int(entry["trial"]["number"])
        entries = [
            item for item in self.manifest.get("measurements", []) if int(item.get("trial", {}).get("number", -1)) != number
        ]
        entries.append(entry)
        entries.sort(key=lambda item: int(item["trial"]["number"]))
        self.manifest["measurements"] = entries
        self._save_manifest()
        self.current_index += 1
        self.show_trial("Próba została zachowana w manifeście.")
        if self.autostart_attempts > 0:
            self.autostart_left = self.autostart_attempts
            if self.current_trial() is None:
                QTimer.singleShot(3000, self._quit_app)
            else:
                QTimer.singleShot(AUTOSTART_NEXT_TRIAL_MS, self._autostart)

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if self.active_worker is not None:
            self.active_worker.request_abort()
        if self.active_thread is not None:
            self.active_thread.wait(10_000)
        event.accept()


# -- command line -------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--patterns",
        default=",".join(DEFAULT_PATTERN_IDS),
        help="Wzorce oddechu po przecinku: id z configs/breathing_patterns albo ścieżki; jedna próba na wzorzec "
        f"(domyślnie {','.join(DEFAULT_PATTERN_IDS)}).",
    )
    parser.add_argument("--a121-port", help="Port A121 Interface A; bez niego wykrywany automatycznie.")
    parser.add_argument("--iphone-device", help="Nazwa lub adres BLE iPhone'a; domyślnie RespiPhoneIMU.")
    parser.add_argument("--iphone-scan-timeout", type=float, default=10.0, help="Czas skanowania BLE [s].")
    parser.add_argument(
        "--iphone-no-autostart",
        action="store_true",
        help="Nie wysyłaj START/STOP przez BLE; użyj, gdy strumieniem steruje aplikacja iPhone.",
    )
    parser.add_argument(
        "--iphone-expected-sample-rate-hz",
        type=float,
        choices=(25.0, 50.0, 100.0),
        default=IPHONE_EXPECTED_SAMPLE_RATE_HZ,
        help="Częstotliwość wybrana w aplikacji iPhone; służy do kontroli kompletności zapisu.",
    )
    parser.add_argument("--prep-seconds", type=float, default=5.0, help="Odliczanie przed każdą próbą [s] (domyślnie 5).")
    parser.add_argument(
        "--distance-cm",
        type=float,
        help="Odległość radaru od klatki piersiowej [cm]: zapisywana w manifeście i centrująca zakres A121 (±50 cm).",
    )
    parser.add_argument(
        "--posture",
        default="supine",
        help="Pozycja badanego zapisywana w manifeście, np. supine, sitting, side (domyślnie supine).",
    )
    parser.add_argument("--a121-start-m", type=float, help="Początek zakresu A121 [m] (nadpisuje zakres domyślny).")
    parser.add_argument("--a121-end-m", type=float, help="Koniec zakresu A121 [m] (nadpisuje zakres domyślny).")
    parser.add_argument("--session-name", help="Nazwa katalogu sesji; domyślnie a121_iphone_<data_godzina>.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Katalog, w którym powstaje katalog sesji (domyślnie data/raw/nn/a121_iphone).",
    )
    parser.add_argument(
        "--autostart",
        type=int,
        default=0,
        metavar="N",
        help="Start each trial by itself, retry a failed start up to N times and keep a trial that passed the "
        "stream checks, then close the window after the last one (for when the mouse is out of reach).",
    )
    parser.add_argument("--start-trial", type=int, default=1, help="Numer próby, od której zacząć (1 = pierwsza).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Wypisz plan (próby, czasy, łączny czas) bez łączenia z czujnikami i bez okna.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        trials = build_trials(parse_pattern_list(args.patterns))
    except (PatternError, ValueError) as exc:
        parser.error(str(exc))
    if not 1 <= args.start_trial <= len(trials):
        parser.error(f"--start-trial musi należeć do zakresu 1–{len(trials)}.")
    if args.prep_seconds < 0:
        parser.error("--prep-seconds nie może być ujemne.")
    if args.iphone_scan_timeout <= 0:
        parser.error("--iphone-scan-timeout musi być dodatnie.")
    if args.distance_cm is not None and args.distance_cm <= 0:
        parser.error("--distance-cm musi być dodatnie.")
    try:
        a121_config = a121_config_for(distance_cm=args.distance_cm, start_m=args.a121_start_m, end_m=args.a121_end_m)
    except ValueError as exc:
        parser.error(str(exc))

    session_name = safe_name(args.session_name or f"a121_iphone_{datetime.now():%Y-%m-%d_%H-%M-%S}", "a121_iphone")
    settings = SessionSettings(
        session_dir=(Path(args.output_dir).expanduser() / session_name).resolve(),
        posture=normalize_posture(args.posture),
        distance_cm=args.distance_cm,
        a121_port=args.a121_port,
        a121_config=a121_config,
        iphone_device=args.iphone_device,
        iphone_scan_timeout_s=float(args.iphone_scan_timeout),
        iphone_autostart=not args.iphone_no_autostart,
        iphone_expected_sample_rate_hz=float(args.iphone_expected_sample_rate_hz),
        prep_seconds=float(args.prep_seconds),
    )
    if args.dry_run:
        print("\n".join(plan_lines(trials, settings, start_trial=args.start_trial)))
        return 0

    try:
        claim_session(settings.session_dir)
    except ValueError as exc:
        parser.error(str(exc))

    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        window = GuidedRecordingWindow(
            trials=trials, settings=settings, start_trial=args.start_trial, autostart_attempts=max(0, args.autostart)
        )
    except ValueError as exc:
        parser.error(str(exc))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
