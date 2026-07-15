#!/usr/bin/env python3
"""Guided HB100 range experiment with an optional simultaneous A121 recording.

The program is intentionally separate from the earlier A121 lens/foil runner.  It
implements the short HB100 protocol used in the thesis:

* optional 30 s empty-scene interference screen (A121 only, both, HB100 only),
* two 90 s recordings at 30, 60, and 100 cm,
* optional extension from 150 cm in 50 cm increments,
* one repositioned retry after the first failed extension distance,
* visible/audible natural-breathing, 12/min, and breath-hold cues,
* Esc/button abort that discards the current recording.

Run from the repository root::

    uv run python tools/hb100_a121_guided_experiment.py

Probe only the currently connected HB100/ESP32 acquisition path::

    uv run python tools/hb100_a121_guided_experiment.py --probe-hb100

Use ``--hb100-only`` while the A121 is not connected.  The full experiment
records both sensors into separate CSV files with host timestamps for alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import serial
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
from scipy.signal import butter, detrend, periodogram, sosfiltfilt

from respi_net.a121 import A121_CAPTURE_COLUMNS, A121Config, A121Capture, find_a121_serial_ports


HB100_COLUMNS = ["Timestamp_ms", "HostTimestamp_ms", "RawADC", "Voltage_mV"]
HB100_LINE_RE = re.compile(rb"^(\d+),(\d+),(\d+)$")
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "hb100_a121" / "guided"

# The current HB100 firmware uses 230400 baud and should produce strict ASCII.
# Keep the audited workaround below only as a fallback for the older 921600-baud
# image: on the tested macOS + WCH driver it was recoverable through the driver's
# 1,000,000-baud divisor, with bit 6 spuriously set in part of the numeric CSV.
# The repair remains gated by CSV/range/time validation and its byte counts are
# persisted in the manifest, so use of a legacy transport is never hidden.
_CH9102_SOURCE = bytes(list(range(0x70, 0x7A)) + [0x6C, 0x4D, 0x4A])
_CH9102_TARGET = bytes(list(range(0x30, 0x3A)) + [0x2C, 0x0D, 0x0A])
CH9102_REPAIR_TABLE = bytes.maketrans(_CH9102_SOURCE, _CH9102_TARGET)


@dataclass(frozen=True)
class BreathingCue:
    start_s: float
    end_s: float
    cue: str
    detail: str
    kind: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentStep:
    number: int
    step_id: str
    label: str
    kind: str
    sensor_mode: str
    duration_s: float
    instruction: str
    distance_cm: float | None = None
    repeat_number: int = 1
    repeat_total: int = 1
    extension: bool = False
    retry_after_failure: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HB100Transport:
    port: str
    baud: int
    repair_ch9102_bit6: bool
    probe_valid_rows: int
    description: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HB100ProbeAttempt:
    baud: int
    strict_rows: int
    repaired_rows: int
    repaired_bytes: int
    total_bytes: int


def build_breathing_cues() -> list[BreathingCue]:
    """Return the fixed 90 s protocol: 10 s natural, 7 cycles, hold, 6 cycles."""
    cues: list[BreathingCue] = []
    cursor = 0.0

    def add(kind: str, duration: float, cue: str, detail: str) -> None:
        nonlocal cursor
        cues.append(BreathingCue(cursor, cursor + duration, cue, detail, kind))
        cursor += duration

    add(
        "normal",
        10.0,
        "ODDYCHAJ SWOBODNIE",
        "Ustabilizuj pozycję i nie wykonuj dodatkowych ruchów.",
    )
    for _ in range(7):
        add("inhale", 2.0, "WDECH", "Spokojny wdech przez 2 sekundy.")
        add("exhale", 3.0, "WYDECH", "Spokojny wydech przez 3 sekundy.")
    add(
        "hold",
        15.0,
        "WSTRZYMAJ ODDECH",
        "Zatrzymaj oddech po wydechu; przy dyskomforcie przerwij próbę.",
    )
    for _ in range(6):
        add("inhale", 2.0, "WDECH", "Spokojny wdech przez 2 sekundy.")
        add("exhale", 3.0, "WYDECH", "Spokojny wydech przez 3 sekundy.")
    if not math.isclose(cursor, 90.0):
        raise AssertionError(f"Breathing protocol should last 90 s, got {cursor:g} s")
    return cues


def breathing_cue_at(cues: list[BreathingCue], elapsed_s: float) -> tuple[BreathingCue, float] | None:
    if not cues:
        return None
    elapsed_s = max(0.0, float(elapsed_s))
    for index, cue in enumerate(cues):
        if elapsed_s < cue.end_s or index == len(cues) - 1:
            return cue, max(0.0, cue.end_s - elapsed_s)
    return None


def build_initial_steps(*, include_interference: bool, hb100_only: bool) -> list[ExperimentStep]:
    steps: list[ExperimentStep] = []
    if include_interference and not hb100_only:
        steps.extend(
            [
                ExperimentStep(
                    1,
                    "INT-A",
                    "Pusta scena — tylko A121",
                    "interference",
                    "a121",
                    30.0,
                    "Odłącz zasilanie HB100. Pozostaw pustą, nieruchomą scenę; nagrywa tylko A121.",
                ),
                ExperimentStep(
                    2,
                    "INT-B",
                    "Pusta scena — oba radary",
                    "interference",
                    "both",
                    30.0,
                    "Podłącz HB100. Nie poruszaj niczym w scenie; oba radary mają pracować równocześnie.",
                ),
                ExperimentStep(
                    3,
                    "INT-C",
                    "Pusta scena — tylko HB100",
                    "interference",
                    "hb100",
                    30.0,
                    "Zatrzymaj/odłącz A121, pozostaw HB100. Scena ma pozostać pusta i nieruchoma.",
                ),
            ]
        )

    sensor_mode = "hb100" if hb100_only else "both"
    for distance_cm in (30.0, 60.0, 100.0):
        for repeat in (1, 2):
            steps.append(
                ExperimentStep(
                    number=len(steps) + 1,
                    step_id=f"RANGE-{int(distance_cm)}-R{repeat}",
                    label=f"Oddech — {distance_cm:g} cm, powtórzenie {repeat}/2",
                    kind="range",
                    sensor_mode=sensor_mode,
                    duration_s=90.0,
                    distance_cm=distance_cm,
                    repeat_number=repeat,
                    repeat_total=2,
                    instruction=(
                        "Ustaw środki apertur HB100 i A121 5–10 cm od siebie, na wysokości mostka, "
                        "z równoległymi osiami. A121 ma płaską pokrywę. Zachowaj nieruchomą pozycję "
                        "i wykonuj komunikaty oddechowe programu."
                        if not hb100_only
                        else "Ustaw HB100 na wysokości mostka i wykonuj komunikaty oddechowe programu."
                    ),
                )
            )
    return steps


def make_extension_step(
    steps: list[ExperimentStep],
    distance_cm: float,
    *,
    hb100_only: bool,
    retry_after_failure: bool = False,
) -> ExperimentStep:
    repeat = 2 if retry_after_failure else 1
    retry_text = (
        " To powtórzenie pierwszej porażki: ponownie zajmij pozycję albo przesuń się radialnie o 1–2 cm."
        if retry_after_failure
        else ""
    )
    return ExperimentStep(
        number=len(steps) + 1,
        step_id=f"RANGE-{int(distance_cm)}-R{repeat}",
        label=f"Zasięg — {distance_cm:g} cm" + (", powtórzenie po przesunięciu" if retry_after_failure else ""),
        kind="range",
        sensor_mode="hb100" if hb100_only else "both",
        duration_s=90.0,
        distance_cm=distance_cm,
        repeat_number=repeat,
        repeat_total=2 if retry_after_failure else 1,
        extension=True,
        retry_after_failure=retry_after_failure,
        instruction=(
            "Nie zmieniaj wzmocnienia ani orientacji radarów. Ustaw odległość od mostka i wykonuj komunikaty."
            + retry_text
        ),
    )


def find_hb100_serial_ports() -> list[str]:
    """Prefer the WCH USB Single Serial used by the HB100 ESP32 board."""
    ports = list(serial.tools.list_ports.comports())

    def score(port: Any) -> tuple[int, str]:
        text = " ".join(
            str(getattr(port, attr, "") or "")
            for attr in ("device", "description", "manufacturer", "product", "interface", "hwid")
        ).lower()
        value = 0
        if getattr(port, "vid", None) == 0x1A86 and getattr(port, "pid", None) == 0x55D4:
            value += 100
        if "usb single serial" in text or "ch910" in text or "ch340" in text:
            value += 30
        if "bluetooth" in text or "debug-console" in text:
            value -= 100
        return (-value, str(port.device))

    likely = [port for port in ports if score(port)[0] < 0]
    return [str(port.device) for port in sorted(likely, key=score)]


def repair_ch9102_payload(payload: bytes) -> tuple[bytes, int]:
    repaired = payload.translate(CH9102_REPAIR_TABLE)
    changed = sum(a != b for a, b in zip(payload, repaired))
    return repaired, changed


def parse_hb100_payload(payload: bytes, *, repair_ch9102_bit6: bool) -> tuple[list[tuple[int, int, int]], int]:
    repaired_bytes = 0
    if repair_ch9102_bit6:
        payload, repaired_bytes = repair_ch9102_payload(payload)
    rows: list[tuple[int, int, int]] = []
    for line in payload.splitlines():
        match = HB100_LINE_RE.fullmatch(line.strip())
        if match is None:
            continue
        timestamp, raw_adc, voltage_mv = (int(group) for group in match.groups())
        if 0 <= raw_adc <= 4095 and 0 <= voltage_mv <= 3500:
            rows.append((timestamp, raw_adc, voltage_mv))
    return rows, repaired_bytes


def _monotonic_fraction(rows: Iterable[tuple[int, int, int]]) -> float:
    timestamps = [row[0] for row in rows]
    if len(timestamps) < 2:
        return 0.0
    return float(np.mean(np.diff(np.asarray(timestamps, dtype=float)) > 0))


def _drain_serial_input(handle: serial.Serial, duration_s: float) -> None:
    """Actively consume startup output; reset_input_buffer alone misses WCH USB backlog."""
    deadline = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < deadline:
        handle.read(handle.in_waiting or 1)
    handle.reset_input_buffer()


def _open_hb100_serial(port: str, baud: int, *, timeout: float) -> serial.Serial:
    """Open with DTR/RTS already released to avoid resetting the ESP32 on connect."""
    handle = serial.Serial(
        port=None,
        baudrate=baud,
        timeout=timeout,
        dsrdtr=False,
        rtscts=False,
    )
    handle.dtr = False
    handle.rts = False
    handle.port = port
    handle.open()
    return handle


def _read_probe_payload(port: str, baud: int, *, warmup_s: float, duration_s: float) -> bytes:
    handle = _open_hb100_serial(port, baud, timeout=0.03)
    try:
        _drain_serial_input(handle, warmup_s)
        payload = bytearray()
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            payload.extend(handle.read(handle.in_waiting or 1))
        return bytes(payload)
    finally:
        handle.close()


def probe_hb100_transport(
    port: str | None = None,
    *,
    candidate_bauds: tuple[int, ...] = (230_400, 921_600, 1_000_000),
    warmup_s: float = 0.8,
    duration_s: float = 0.8,
) -> tuple[HB100Transport, list[HB100ProbeAttempt]]:
    resolved_port = port or (find_hb100_serial_ports()[0] if find_hb100_serial_ports() else "")
    if not resolved_port:
        raise RuntimeError("Nie znaleziono portu HB100/ESP32. Podłącz moduł albo podaj --hb100-port.")

    attempts: list[HB100ProbeAttempt] = []
    repaired_candidates: list[tuple[int, int, int]] = []
    for baud in dict.fromkeys(int(value) for value in candidate_bauds):
        payload = _read_probe_payload(resolved_port, baud, warmup_s=warmup_s, duration_s=duration_s)
        strict_rows, _ = parse_hb100_payload(payload, repair_ch9102_bit6=False)
        repaired_rows, repaired_count = parse_hb100_payload(payload, repair_ch9102_bit6=True)
        attempts.append(
            HB100ProbeAttempt(
                baud=baud,
                strict_rows=len(strict_rows),
                repaired_rows=len(repaired_rows),
                repaired_bytes=repaired_count,
                total_bytes=len(payload),
            )
        )
        if len(strict_rows) >= 10 and _monotonic_fraction(strict_rows) >= 0.98:
            return (
                HB100Transport(
                    resolved_port,
                    baud,
                    False,
                    len(strict_rows),
                    "surowy ASCII CSV bez korekcji transportu",
                ),
                attempts,
            )
        if len(repaired_rows) >= 10 and _monotonic_fraction(repaired_rows) >= 0.98:
            repaired_candidates.append((len(repaired_rows), -repaired_count, baud))

    if repaired_candidates:
        valid_rows, _negative_repairs, baud = max(repaired_candidates)
        return (
            HB100Transport(
                resolved_port,
                baud,
                True,
                valid_rows,
                "tryb zgodności macOS/WCH: jawna korekcja bitu 6 w numerycznym ASCII CSV",
            ),
            attempts,
        )
    details = "; ".join(
        f"{attempt.baud}: strict={attempt.strict_rows}, compat={attempt.repaired_rows}, bytes={attempt.total_bytes}"
        for attempt in attempts
    )
    raise RuntimeError(f"Port {resolved_port} nie zwrócił poprawnych ramek HB100 ({details}).")


class HB100SerialCapture:
    """Threaded HB100 CSV capture with an explicit, audited transport mode."""

    def __init__(self, transport: HB100Transport):
        self.transport = transport
        self.handle: serial.Serial | None = None
        self.running = False
        self.rows: list[list[float]] = []
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.total_bytes = 0
        self.repaired_bytes = 0
        self.malformed_lines = 0
        self.malformed_examples: list[str] = []
        self._buffer = b""

    def connect(self, warmup_s: float = 0.8) -> None:
        handle = _open_hb100_serial(self.transport.port, self.transport.baud, timeout=0.05)
        _drain_serial_input(handle, warmup_s)
        self.handle = handle
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        assert self.handle is not None
        while self.running:
            try:
                chunk = self.handle.read(self.handle.in_waiting or 1)
                if not chunk:
                    continue
                self.total_bytes += len(chunk)
                if self.transport.repair_ch9102_bit6:
                    chunk, repaired = repair_ch9102_payload(chunk)
                    self.repaired_bytes += repaired
                self._buffer += chunk
                while b"\n" in self._buffer:
                    line, self._buffer = self._buffer.split(b"\n", 1)
                    match = HB100_LINE_RE.fullmatch(line.strip())
                    if match is None:
                        self.malformed_lines += 1
                        if len(self.malformed_examples) < 5:
                            self.malformed_examples.append(line[:160].decode("ascii", errors="backslashreplace"))
                        continue
                    timestamp, raw_adc, voltage_mv = (int(group) for group in match.groups())
                    if not (0 <= raw_adc <= 4095 and 0 <= voltage_mv <= 3500):
                        self.malformed_lines += 1
                        if len(self.malformed_examples) < 5:
                            self.malformed_examples.append(line[:160].decode("ascii", errors="backslashreplace"))
                        continue
                    row = [float(timestamp), time.time() * 1000.0, float(raw_adc), float(voltage_mv)]
                    with self.lock:
                        self.rows.append(row)
            except Exception:
                self.running = False
                break

    def data_count(self) -> int:
        with self.lock:
            return len(self.rows)

    def snapshot_since(self, index: int) -> list[list[float]]:
        with self.lock:
            return [list(row) for row in self.rows[index:]]

    def stop(self) -> None:
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        handle, self.handle = self.handle, None
        if handle is not None and handle.is_open:
            handle.close()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "repaired_bytes": self.repaired_bytes,
            "malformed_lines": self.malformed_lines,
            "malformed_examples": list(self.malformed_examples),
            "repair_fraction": self.repaired_bytes / self.total_bytes if self.total_bytes else 0.0,
        }


def _uniform_signal(df: pd.DataFrame, start_wall_ms: float, duration_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    t = (pd.to_numeric(df["HostTimestamp_ms"], errors="coerce").to_numpy(dtype=float) - start_wall_ms) / 1000.0
    y = pd.to_numeric(df["Voltage_mV"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & (t >= 0) & (t <= duration_s + 0.5)
    t, y = t[mask], y[mask]
    if len(t) < 20:
        raise ValueError("Za mało poprawnych próbek HB100 do analizy.")
    order = np.argsort(t)
    t, y = t[order], y[order]
    unique = np.concatenate(([True], np.diff(t) > 1e-6))
    t, y = t[unique], y[unique]
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    observed_fs = 1.0 / np.median(dt) if len(dt) else 100.0
    fs = float(np.clip(observed_fs, 20.0, 100.0))
    grid = np.arange(0.0, min(duration_s, t[-1]), 1.0 / fs)
    return grid, np.interp(grid, t, y), fs


def _window_spectral_metrics(t: np.ndarray, y: np.ndarray, fs: float, start_s: float, end_s: float) -> dict[str, float]:
    mask = (t >= start_s) & (t < end_s)
    segment = y[mask]
    if len(segment) < max(32, int(8 * fs)):
        return {"peak_hz": float("nan"), "target_snr_db": float("nan")}
    segment = detrend(segment, type="linear")
    nfft = max(4096, 2 ** int(math.ceil(math.log2(len(segment) * 4))))
    frequencies, power = periodogram(segment, fs=fs, window="hann", nfft=nfft, scaling="density")
    respiration = (frequencies >= 0.10) & (frequencies <= 0.60)
    if not np.any(respiration):
        return {"peak_hz": float("nan"), "target_snr_db": float("nan")}
    resp_indices = np.where(respiration)[0]
    peak_idx = int(resp_indices[np.argmax(power[respiration])])
    target = (frequencies >= 0.18) & (frequencies <= 0.22)
    noise = respiration & ~((frequencies >= 0.16) & (frequencies <= 0.24))
    target_power = float(np.max(power[target])) if np.any(target) else float("nan")
    noise_power = float(np.median(power[noise])) if np.any(noise) else float("nan")
    snr = 10.0 * math.log10(max(target_power, 1e-18) / max(noise_power, 1e-18))
    return {"peak_hz": float(frequencies[peak_idx]), "target_snr_db": snr}


def summarize_hb100_rows(
    rows: list[list[float]],
    *,
    start_wall_ms: float,
    duration_s: float,
    range_protocol: bool,
    transport: HB100Transport,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    df = pd.DataFrame(rows, columns=HB100_COLUMNS)
    device_t = pd.to_numeric(df["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    device_dt = np.diff(device_t)
    device_dt = device_dt[np.isfinite(device_dt) & (device_dt > 0)]
    sample_rate = 1000.0 / np.median(device_dt) if len(device_dt) else 0.0
    voltage = pd.to_numeric(df["Voltage_mV"], errors="coerce").to_numpy(dtype=float)
    voltage = voltage[np.isfinite(voltage)]
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "sample_rate_hz": float(sample_rate),
        "voltage_mean_mv": float(np.mean(voltage)),
        "voltage_min_mv": float(np.min(voltage)),
        "voltage_max_mv": float(np.max(voltage)),
        "voltage_p2p_mv": float(np.ptp(voltage)),
        "voltage_std_mv": float(np.std(voltage)),
        "saturation_percent": float(np.mean((voltage <= 50.0) | (voltage >= 3050.0)) * 100.0),
        "transport": transport.as_dict(),
        "transport_diagnostics": diagnostics,
    }
    if not range_protocol:
        summary["background_rms_mv"] = float(np.std(detrend(voltage, type="linear")))
        return summary

    t, uniform_voltage, uniform_fs = _uniform_signal(df, start_wall_ms, duration_s)
    first = _window_spectral_metrics(t, uniform_voltage, uniform_fs, 10.0, 45.0)
    second = _window_spectral_metrics(t, uniform_voltage, uniform_fs, 60.0, 90.0)
    sos = butter(3, [0.10, 0.60], btype="bandpass", fs=uniform_fs, output="sos")
    filtered = sosfiltfilt(sos, uniform_voltage - np.mean(uniform_voltage))

    def rms_between(start_s: float, end_s: float) -> float:
        values = filtered[(t >= start_s) & (t < end_s)]
        return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")

    paced_rms = float(np.nanmean([rms_between(10.0, 45.0), rms_between(60.0, 90.0)]))
    hold_rms = rms_between(47.0, 58.0)  # trim filter/cue transitions inside the 15 s hold
    hold_drop_db = 20.0 * math.log10(max(hold_rms, 1e-12) / max(paced_rms, 1e-12))
    usable = (
        0.18 <= first["peak_hz"] <= 0.22
        and 0.18 <= second["peak_hz"] <= 0.22
        and first["target_snr_db"] >= 6.0
        and second["target_snr_db"] >= 6.0
        and hold_drop_db <= -3.0
        and summary["saturation_percent"] < 0.1
    )
    summary.update(
        {
            "paced_1_peak_hz": first["peak_hz"],
            "paced_1_snr_db": first["target_snr_db"],
            "paced_2_peak_hz": second["peak_hz"],
            "paced_2_snr_db": second["target_snr_db"],
            "paced_rms_mv": paced_rms,
            "hold_rms_mv": hold_rms,
            "hold_drop_db": hold_drop_db,
            "auto_usable": bool(usable),
            "auto_criterion": "oba piki 0.18–0.22 Hz, oba SNR >= 6 dB, hold <= -3 dB, bez nasycenia",
        }
    )
    return summary


def summarize_a121_rows(rows: list[list[Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows, columns=A121_CAPTURE_COLUMNS)
    timestamp = pd.to_numeric(df["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    differences = np.diff(timestamp)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    sample_rate = 1000.0 / np.median(differences) if len(differences) else 0.0
    target = pd.to_numeric(df.get("AcconeerTargetDistance_m"), errors="coerce").to_numpy(dtype=float)
    target = target[np.isfinite(target)]
    return {
        "rows": int(len(df)),
        "sample_rate_hz": float(sample_rate),
        "median_target_distance_m": float(np.median(target)) if len(target) else None,
        "dropped_results": None,
    }


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return safe or "measurement"


def a121_config_for_step(step: ExperimentStep) -> A121Config:
    if step.distance_cm is None:
        start_m, end_m = 0.2, 1.5
    else:
        distance_m = step.distance_cm / 100.0
        start_m, end_m = max(0.05, distance_m - 0.50), distance_m + 0.50
    return A121Config(
        start_m=start_m,
        end_m=end_m,
        profile=3,
        hwaas=32,
        sweeps_per_frame=8,
        frame_rate_hz=20.0,
        step_length=1,
    )


def output_paths(session_dir: Path, step: ExperimentStep) -> dict[str, Path]:
    distance = f"_{step.distance_cm:g}cm" if step.distance_cm is not None else ""
    stem = f"run_{step.number:02d}_{safe_filename(step.step_id).lower()}{distance}"
    return {
        "hb100": session_dir / f"{stem}_hb100.csv",
        "a121": session_dir / f"{stem}_a121.csv",
        "cues": session_dir / f"{stem}_cues.csv",
    }


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
        session_dir: Path,
        cues: list[BreathingCue],
        hb100_port: str | None,
        a121_port: str | None,
        transport: HB100Transport | None,
        prep_seconds: int,
    ) -> None:
        super().__init__()
        self.step = step
        self.session_dir = session_dir
        self.cues = cues
        self.hb100_port = hb100_port
        self.a121_port = a121_port
        self.transport = transport
        self.prep_seconds = prep_seconds
        self.abort_event = threading.Event()
        self.hb100: HB100SerialCapture | None = None
        self.a121: A121Capture | None = None
        self.paths = output_paths(session_dir, step)

    def request_abort(self) -> None:
        self.abort_event.set()

    def _stop(self) -> None:
        if self.hb100 is not None:
            self.hb100.stop()
        if self.a121 is not None:
            self.a121.stop()

    def _delete_outputs(self) -> None:
        for path in self.paths.values():
            path.unlink(missing_ok=True)

    def _abort(self) -> None:
        self.status_changed.emit("Przerwano pomiar; bieżący zapis jest odrzucany…")
        self._stop()
        self._delete_outputs()
        self.aborted.emit()

    def _connect_hb100(self) -> HB100Transport:
        transport = self.transport
        if transport is None:
            self.status_changed.emit("Sprawdzanie formatu transmisji HB100/ESP32…")
            transport, _attempts = probe_hb100_transport(self.hb100_port)
        self.status_changed.emit(
            f"Łączenie z HB100: {transport.port}, {transport.baud} baud"
            + (" (tryb zgodności WCH)" if transport.repair_ch9102_bit6 else "")
        )
        self.hb100 = HB100SerialCapture(transport)
        self.hb100.connect()
        return transport

    @Slot()
    def run(self) -> None:
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._delete_outputs()
            transport: HB100Transport | None = None
            if self.step.sensor_mode in {"hb100", "both"}:
                transport = self._connect_hb100()
            if self.abort_event.is_set():
                self._abort()
                return
            if self.step.sensor_mode in {"a121", "both"}:
                resolved_a121 = self.a121_port or (find_a121_serial_ports()[0] if find_a121_serial_ports() else "")
                if not resolved_a121:
                    raise RuntimeError("Nie znaleziono A121 Interface A. Podłącz radar albo uruchom z --hb100-only.")
                self.status_changed.emit(f"Łączenie z A121: {resolved_a121}…")
                self.a121 = A121Capture(output_dir=self.session_dir, config=a121_config_for_step(self.step))
                if not self.a121.connect(resolved_a121):
                    raise RuntimeError("A121 nie odpowiedział. Sprawdź Interface A i zasilanie.")

            self.status_changed.emit("Czujniki działają. Ustabilizuj pozycję.")
            for remaining in range(self.prep_seconds, 0, -1):
                if self.abort_event.is_set():
                    self._abort()
                    return
                self.prep_changed.emit(remaining)
                if self.abort_event.wait(1.0):
                    self._abort()
                    return

            hb_start = self.hb100.data_count() if self.hb100 is not None else 0
            a121_start = self.a121.data_count() if self.a121 is not None else 0
            start_wall_ms = time.time() * 1000.0
            started = time.monotonic()
            while True:
                if self.abort_event.is_set():
                    self._abort()
                    return
                if self.hb100 is not None and not self.hb100.running:
                    raise RuntimeError("Transmisja HB100 zatrzymała się przed końcem pomiaru.")
                if self.a121 is not None and not self.a121.running:
                    raise RuntimeError("Transmisja A121 zatrzymała się przed końcem pomiaru.")
                elapsed = time.monotonic() - started
                self.measurement_changed.emit(min(elapsed, self.step.duration_s), self.step.duration_s)
                if elapsed >= self.step.duration_s:
                    break
                if self.abort_event.wait(0.05):
                    self._abort()
                    return

            self.status_changed.emit("Zatrzymywanie czujników i obliczanie podsumowania…")
            self._stop()
            hb_rows = self.hb100.snapshot_since(hb_start) if self.hb100 is not None else []
            a121_rows = self.a121.snapshot_data_since(a121_start) if self.a121 is not None else []
            if self.hb100 is not None and len(hb_rows) < 50:
                raise RuntimeError(f"HB100: zapisano tylko {len(hb_rows)} poprawnych próbek.")
            if self.a121 is not None and len(a121_rows) < 10:
                raise RuntimeError(f"A121: zapisano tylko {len(a121_rows)} poprawnych ramek.")

            summary: dict[str, Any] = {
                "step": self.step.as_dict(),
                "measurement_start_utc": datetime.fromtimestamp(start_wall_ms / 1000.0, tz=timezone.utc).isoformat(),
                "measurement_start_wall_ms": start_wall_ms,
                "requested_seconds": self.step.duration_s,
                "files": {},
            }
            if hb_rows and self.hb100 is not None and transport is not None:
                pd.DataFrame(hb_rows, columns=HB100_COLUMNS).to_csv(self.paths["hb100"], index=False)
                summary["files"]["hb100"] = str(self.paths["hb100"])
                summary["hb100"] = summarize_hb100_rows(
                    hb_rows,
                    start_wall_ms=start_wall_ms,
                    duration_s=self.step.duration_s,
                    range_protocol=self.step.kind == "range",
                    transport=transport,
                    diagnostics=self.hb100.diagnostics(),
                )
                summary["hb100_transport"] = transport.as_dict()
            if a121_rows and self.a121 is not None:
                pd.DataFrame(a121_rows, columns=A121_CAPTURE_COLUMNS).to_csv(self.paths["a121"], index=False)
                summary["files"]["a121"] = str(self.paths["a121"])
                a121_summary = summarize_a121_rows(a121_rows)
                a121_summary["dropped_results"] = int(self.a121.dropped_results)
                summary["a121"] = a121_summary
            if self.step.kind == "range":
                cue_rows = [
                    {
                        **cue.as_dict(),
                        "start_wall_ms": start_wall_ms + cue.start_s * 1000.0,
                        "end_wall_ms": start_wall_ms + cue.end_s * 1000.0,
                    }
                    for cue in self.cues
                ]
                pd.DataFrame(cue_rows).to_csv(self.paths["cues"], index=False)
                summary["files"]["cues"] = str(self.paths["cues"])
            self.finished.emit(summary)
        except Exception as exc:
            self._delete_outputs()
            if self.abort_event.is_set():
                self._abort()
            else:
                self.failed.emit(str(exc))
        finally:
            self._stop()


def format_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    hb = summary.get("hb100")
    if isinstance(hb, dict):
        lines.extend(
            [
                f"HB100: {hb.get('rows', 0)} próbek, {hb.get('sample_rate_hz', 0):.1f} Hz",
                f"Napięcie: {hb.get('voltage_min_mv', 0):.0f}–{hb.get('voltage_max_mv', 0):.0f} mV "
                f"(p-p {hb.get('voltage_p2p_mv', 0):.1f} mV), nasycenie {hb.get('saturation_percent', 0):.3f}%",
            ]
        )
        if "paced_1_peak_hz" in hb:
            lines.extend(
                [
                    f"Blok 1: pik {hb['paced_1_peak_hz']:.3f} Hz, SNR {hb['paced_1_snr_db']:.1f} dB",
                    f"Blok 2: pik {hb['paced_2_peak_hz']:.3f} Hz, SNR {hb['paced_2_snr_db']:.1f} dB",
                    f"Wstrzymanie: zmiana RMS {hb['hold_drop_db']:.1f} dB",
                    "Ocena automatyczna: " + ("UŻYTECZNY" if hb.get("auto_usable") else "NIE SPEŁNIA CAŁEGO KRYTERIUM"),
                ]
            )
        else:
            lines.append(f"Tło HB100 RMS: {hb.get('background_rms_mv', 0):.3f} mV")
        transport = hb.get("transport", {})
        diagnostics = hb.get("transport_diagnostics", {})
        if transport.get("repair_ch9102_bit6"):
            lines.append(
                "UWAGA transport: tryb zgodności WCH; skorygowano "
                f"{diagnostics.get('repaired_bytes', 0)}/{diagnostics.get('total_bytes', 0)} bajtów, "
                f"odrzucone linie: {diagnostics.get('malformed_lines', 0)}."
            )
        elif transport:
            lines.append(
                f"Transport HB100: {transport.get('baud', 0)} baud, surowy ASCII bez korekcji; "
                f"odrzucone linie: {diagnostics.get('malformed_lines', 0)}."
            )
    a121 = summary.get("a121")
    if isinstance(a121, dict):
        target = a121.get("median_target_distance_m")
        target_text = f"{target:.3f} m" if isinstance(target, (int, float)) else "n/a"
        lines.append(
            f"A121: {a121.get('rows', 0)} ramek, {a121.get('sample_rate_hz', 0):.1f} Hz, "
            f"mediana celu {target_text}, utracone {a121.get('dropped_results', 0)}"
        )
    for sensor, path in summary.get("files", {}).items():
        lines.append(f"{sensor}: {path}")
    return "\n".join(lines)


class GuidedExperimentWindow(QWidget):
    def __init__(
        self,
        *,
        steps: list[ExperimentStep],
        cues: list[BreathingCue],
        session_dir: Path,
        hb100_port: str | None,
        a121_port: str | None,
        hb100_only: bool,
        prep_seconds: int,
        start_step: int,
    ) -> None:
        super().__init__()
        self.steps = steps
        self.cues = cues
        self.session_dir = session_dir
        self.hb100_port = hb100_port
        self.a121_port = a121_port
        self.hb100_only = hb100_only
        self.prep_seconds = prep_seconds
        self.current_index = start_step - 1
        self.active_worker: MeasurementWorker | None = None
        self.active_thread: QThread | None = None
        self.pending_summary: dict[str, Any] | None = None
        self.cached_transport: HB100Transport | None = None
        self.current_cue_key: tuple[float, float] | None = None
        self.manifest_path = session_dir / "manifest.json"
        self.manifest = self._load_manifest()

        self.setWindowTitle("HB100 + A121 — prowadzony test zasięgu oddechu")
        self.resize(900, 800)
        self._build_ui()
        self.show_step()
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
            "experiment": "HB100 practical respiratory range with simultaneous A121 comparison",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hb100_only": self.hb100_only,
            "steps": [step.as_dict() for step in self.steps],
            "breathing_cues": [cue.as_dict() for cue in self.cues],
            "measurements": [],
        }
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: dict[str, Any] | None = None) -> None:
        payload = manifest if manifest is not None else self.manifest
        self.manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.step_label = QLabel()
        self.step_label.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(self.step_label)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        self.instructions.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.instructions)

        cue_box = QGroupBox("Przebieg próby")
        cue_layout = QVBoxLayout(cue_box)
        self.cue_label = QLabel()
        self.cue_label.setAlignment(Qt.AlignCenter)
        self.cue_detail = QLabel()
        self.cue_detail.setAlignment(Qt.AlignCenter)
        self.cue_detail.setWordWrap(True)
        self.cue_countdown = QLabel()
        self.cue_countdown.setAlignment(Qt.AlignCenter)
        cue_layout.addWidget(self.cue_label)
        cue_layout.addWidget(self.cue_detail)
        cue_layout.addWidget(self.cue_countdown)
        layout.addWidget(cue_box)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.progress)
        self.status = QLabel("Gotowy.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        button_grid = QGridLayout()
        self.start_button = QPushButton("Start pomiaru")
        self.abort_button = QPushButton("Przerwij i odrzuć (Esc)")
        self.abort_button.setStyleSheet("background: #9b1c1c; color: white; font-weight: bold;")
        self.accept_button = QPushButton("Zachowaj zapis")
        self.pass_button = QPushButton("Zachowaj — sygnał użyteczny")
        self.fail_button = QPushButton("Zachowaj — sygnał nieużyteczny")
        self.redo_button = QPushButton("Odrzuć i powtórz")
        self.extend_button = QPushButton("Dodaj kolejny dystans")
        self.finish_button = QPushButton("Zakończ serię")
        button_grid.addWidget(self.start_button, 0, 0, 1, 2)
        button_grid.addWidget(self.abort_button, 0, 2, 1, 2)
        button_grid.addWidget(self.accept_button, 1, 0)
        button_grid.addWidget(self.pass_button, 1, 0)
        button_grid.addWidget(self.fail_button, 1, 1)
        button_grid.addWidget(self.redo_button, 1, 2)
        button_grid.addWidget(self.extend_button, 2, 0, 1, 2)
        button_grid.addWidget(self.finish_button, 2, 2, 1, 2)
        layout.addLayout(button_grid)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        layout.addWidget(self.summary, stretch=1)

        self.start_button.clicked.connect(self.start_measurement)
        self.abort_button.clicked.connect(self.abort_current)
        self.accept_button.clicked.connect(lambda: self.accept_measurement(None))
        self.pass_button.clicked.connect(lambda: self.accept_measurement(True))
        self.fail_button.clicked.connect(lambda: self.accept_measurement(False))
        self.redo_button.clicked.connect(self.redo_measurement)
        self.extend_button.clicked.connect(self.extend_range)
        self.finish_button.clicked.connect(self.finish_series)

    def current_step(self) -> ExperimentStep | None:
        return self.steps[self.current_index] if 0 <= self.current_index < len(self.steps) else None

    def _set_cue(self, cue: str, detail: str, countdown: str, kind: str) -> None:
        colors = {
            "inhale": ("#cfe8ff", "#063b66"),
            "exhale": ("#dcf5df", "#174d20"),
            "hold": ("#ffe0e0", "#6d1111"),
            "normal": ("#eeeeee", "#222222"),
            "interference": ("#fff2bf", "#5b4300"),
        }
        background, foreground = colors.get(kind, colors["normal"])
        self.cue_label.setText(cue)
        self.cue_label.setStyleSheet(
            "font-size: 32px; font-weight: bold; padding: 12px; "
            f"background: {background}; color: {foreground}; border-radius: 6px;"
        )
        self.cue_detail.setText(detail)
        self.cue_countdown.setText(countdown)

    def _set_review_buttons(self, step: ExperimentStep, enabled: bool) -> None:
        self.accept_button.setVisible(step.kind != "range")
        self.pass_button.setVisible(step.kind == "range")
        self.fail_button.setVisible(step.kind == "range")
        self.accept_button.setEnabled(enabled)
        self.pass_button.setEnabled(enabled)
        self.fail_button.setEnabled(enabled)
        self.redo_button.setEnabled(enabled)

    def show_step(self, message: str | None = None) -> None:
        step = self.current_step()
        self.pending_summary = None
        self.current_cue_key = None
        self.start_button.setEnabled(False)
        self.abort_button.setEnabled(False)
        self.extend_button.setVisible(False)
        self.finish_button.setVisible(False)
        self.progress.setValue(0)
        self.summary.clear()
        if step is None:
            self._show_extension_decision(message)
            return
        self.step_label.setText(f"Próba {step.number}/{len(self.steps)} — {step.label}")
        distance_line = f"Odległość: {step.distance_cm:g} cm\n" if step.distance_cm is not None else ""
        sensor_label = {"both": "HB100 + A121", "hb100": "HB100", "a121": "A121"}[step.sensor_mode]
        self.instructions.setText(
            f"Czujniki: {sensor_label}\n{distance_line}Czas: {step.duration_s:g} s\n\n{step.instruction}"
        )
        if step.kind == "range":
            self._set_cue(
                "PO STARCIE: ODDYCHAJ SWOBODNIE",
                "Następnie program poprowadzi 12 oddechów/min, wstrzymanie i drugi blok 12/min.",
                "",
                "normal",
            )
        else:
            self._set_cue(
                "PUSTA I NIERUCHOMA SCENA",
                "Podczas 30 sekund nie poruszaj się w polu widzenia radarów.",
                "",
                "interference",
            )
        self.status.setText(message or "Sprawdź ustawienie i uruchom pomiar.")
        self.start_button.setEnabled(True)
        self._set_review_buttons(step, False)

    def _show_extension_decision(self, message: str | None) -> None:
        measurements = self.manifest.get("measurements", [])
        last = measurements[-1] if measurements else {}
        last_step = last.get("step", {})
        if not last_step or last_step.get("kind") != "range":
            self.finish_series()
            return
        distance = float(last_step.get("distance_cm") or 100.0)
        usable = last.get("operator_usable")
        retry = bool(last_step.get("retry_after_failure"))
        if bool(last_step.get("extension")) and usable is False and not retry:
            self.steps.append(
                make_extension_step(self.steps, distance, hb100_only=self.hb100_only, retry_after_failure=True)
            )
            self.manifest["steps"] = [step.as_dict() for step in self.steps]
            self._save_manifest()
            self.show_step("Pierwsza porażka jest zachowana. Powtórz po ponownym zajęciu pozycji/przesunięciu 1–2 cm.")
            return

        self.step_label.setText("Zaplanowane próby zakończone")
        if retry and usable is False:
            self.instructions.setText(
                f"Pierwsza porażka przy {distance:g} cm powtórzyła się po przesunięciu. "
                "To jest praktyczna granica serii; zakończ badanie."
            )
            self.extend_button.setVisible(False)
        elif usable is True:
            next_distance = 150.0 if distance <= 100.0 else distance + 50.0
            self.instructions.setText(
                f"Ostatni dystans oznaczono jako użyteczny. Możesz dodać {next_distance:g} cm "
                "albo zakończyć serię."
            )
            self.extend_button.setText(f"Dodaj {next_distance:g} cm")
            self.extend_button.setVisible(True)
        else:
            self.instructions.setText(
                "Dwie podstawowe próby przy 100 cm nie dały jednoznacznej podstawy do zwiększania dystansu. "
                "Możesz zakończyć serię."
            )
        self.finish_button.setVisible(True)
        self.finish_button.setEnabled(True)
        self.status.setText(message or "Wybierz dalszy krok.")
        self._set_cue("DECYZJA O ZASIĘGU", "", "", "normal")

    @Slot()
    def start_measurement(self) -> None:
        step = self.current_step()
        if step is None or self.active_thread is not None:
            return
        self.start_button.setEnabled(False)
        self.abort_button.setEnabled(True)
        self._set_review_buttons(step, False)
        self.summary.clear()
        worker = MeasurementWorker(
            step=step,
            session_dir=self.session_dir,
            cues=self.cues,
            hb100_port=self.hb100_port,
            a121_port=self.a121_port,
            transport=self.cached_transport,
            prep_seconds=self.prep_seconds,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status_changed.connect(self.status.setText)
        worker.prep_changed.connect(self.on_prep)
        worker.measurement_changed.connect(self.on_progress)
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
    def on_prep(self, remaining: int) -> None:
        self._set_cue("PRZYGOTUJ SIĘ", "Ustabilizuj pozycję i nie zmieniaj ustawienia czujników.", f"Start za {remaining} s", "normal")
        self.status.setText(f"Przygotowanie: {remaining} s.")

    @Slot(float, float)
    def on_progress(self, elapsed: float, total: float) -> None:
        step = self.current_step()
        if step is not None and step.kind == "range":
            state = breathing_cue_at(self.cues, elapsed)
            if state is not None:
                cue, remaining = state
                cue_key = (cue.start_s, cue.end_s)
                if cue_key != self.current_cue_key:
                    self.current_cue_key = cue_key
                    QApplication.beep()
                self._set_cue(
                    cue.cue,
                    cue.detail,
                    f"Jeszcze {remaining:.1f} s • pomiar {elapsed:.1f}/{total:.0f} s",
                    cue.kind,
                )
        else:
            self._set_cue(
                "PUSTA I NIERUCHOMA SCENA",
                "Nie poruszaj niczym w polu widzenia radarów.",
                f"Jeszcze {max(0.0, total - elapsed):.1f} s",
                "interference",
            )
        self.progress.setValue(int(round(1000 * elapsed / max(total, 1e-9))))

    @Slot(object)
    def on_finished(self, result: object) -> None:
        self.pending_summary = dict(result)  # type: ignore[arg-type]
        transport_dict = self.pending_summary.get("hb100_transport")
        if isinstance(transport_dict, dict):
            self.cached_transport = HB100Transport(**transport_dict)
        self.abort_button.setEnabled(False)
        self.progress.setValue(1000)
        self.summary.setPlainText(format_summary(self.pending_summary))
        self.status.setText("Zapis gotowy. Zachowaj go z oceną użyteczności albo odrzuć i powtórz.")
        self._set_cue("KONIEC", "Oddychaj swobodnie.", "", "normal")
        step = self.current_step()
        if step is not None:
            self._set_review_buttons(step, True)

    @Slot(str)
    def on_failed(self, message: str) -> None:
        self.abort_button.setEnabled(False)
        self.status.setText(f"Pomiar nieudany: {message}")
        self.summary.setPlainText("Nie zapisano bieżącej próby. Popraw połączenie/ustawienie i uruchom ją ponownie.")
        self.start_button.setEnabled(True)

    @Slot()
    def on_aborted(self) -> None:
        self.abort_button.setEnabled(False)
        self.status.setText("Bieżąca próba została odrzucona i nie zapisana.")
        self.summary.setPlainText("Przerwany zapis usunięto.")
        self.start_button.setEnabled(True)

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
        self.status.setText("Przerywanie i odrzucanie bieżącej próby…")
        self.active_worker.request_abort()

    def _remove_pending_files(self) -> None:
        if self.pending_summary is None:
            return
        for value in self.pending_summary.get("files", {}).values():
            Path(value).unlink(missing_ok=True)

    @Slot()
    def redo_measurement(self) -> None:
        self._remove_pending_files()
        self.pending_summary = None
        self.show_step("Poprzedni zapis usunięto. Ustaw stanowisko i uruchom próbę ponownie.")

    def accept_measurement(self, operator_usable: bool | None) -> None:
        if self.pending_summary is None:
            return
        entry = dict(self.pending_summary)
        entry["status"] = "accepted"
        entry["operator_usable"] = operator_usable
        entry["accepted_at"] = datetime.now(timezone.utc).isoformat()
        step_number = int(entry["step"]["number"])
        measurements = [
            item
            for item in self.manifest.get("measurements", [])
            if int(item.get("step", {}).get("number", -1)) != step_number
        ]
        measurements.append(entry)
        measurements.sort(key=lambda item: int(item.get("step", {}).get("number", 0)))
        self.manifest["measurements"] = measurements
        self._save_manifest()
        self.pending_summary = None
        self.current_index += 1
        self.show_step("Zapis zachowano.")

    @Slot()
    def extend_range(self) -> None:
        measurements = self.manifest.get("measurements", [])
        if not measurements:
            return
        distance = float(measurements[-1]["step"].get("distance_cm") or 100.0)
        next_distance = 150.0 if distance <= 100.0 else distance + 50.0
        self.steps.append(make_extension_step(self.steps, next_distance, hb100_only=self.hb100_only))
        self.manifest["steps"] = [step.as_dict() for step in self.steps]
        self._save_manifest()
        self.show_step()

    @Slot()
    def finish_series(self) -> None:
        self.step_label.setText("Seria zakończona")
        self.instructions.setText(f"Wszystkie zaakceptowane pliki i manifest znajdują się w:\n{self.session_dir}")
        self.status.setText("Gotowe.")
        self.start_button.setEnabled(False)
        self.abort_button.setEnabled(False)
        self.extend_button.setVisible(False)
        self.finish_button.setVisible(False)
        if self.steps:
            self._set_review_buttons(self.steps[-1], False)
        self._set_cue("KONIEC BADANIA", "", "", "normal")

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if self.active_worker is not None:
            self.active_worker.request_abort()
        if self.active_thread is not None:
            self.active_thread.quit()
            self.active_thread.wait(4000)
        event.accept()


def print_probe_result(transport: HB100Transport, attempts: list[HB100ProbeAttempt], seconds: float) -> int:
    print(f"HB100 port: {transport.port}")
    for attempt in attempts:
        print(
            f"  {attempt.baud} baud: strict={attempt.strict_rows}, compat={attempt.repaired_rows}, "
            f"repaired_bytes={attempt.repaired_bytes}/{attempt.total_bytes}"
        )
    print(f"Wybrano: {transport.baud} baud — {transport.description}")
    capture = HB100SerialCapture(transport)
    capture.connect()
    try:
        start = capture.data_count()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and capture.running:
            time.sleep(0.05)
    finally:
        capture.stop()
    rows = capture.snapshot_since(start)
    if len(rows) < 20:
        print(f"BŁĄD: tylko {len(rows)} poprawnych próbek.")
        return 2
    summary = summarize_hb100_rows(
        rows,
        start_wall_ms=rows[0][1],
        duration_s=seconds,
        range_protocol=False,
        transport=transport,
        diagnostics=capture.diagnostics(),
    )
    print(format_summary({"hb100": summary, "files": {}}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hb100-port", help="Port HB100/ESP32; domyślnie auto-detekcja USB Single Serial.")
    parser.add_argument("--a121-port", help="Port A121 Interface A; domyślnie auto-detekcja.")
    parser.add_argument("--hb100-only", action="store_true", help="Nagrywaj tylko HB100; pomiń A121 i test zakłóceń.")
    parser.add_argument("--skip-interference", action="store_true", help="Pomiń trzy 30-sekundowe próby pustej sceny.")
    parser.add_argument("--prep-seconds", type=int, default=8, help="Odliczanie przed każdą próbą (domyślnie 8 s).")
    parser.add_argument("--start-step", type=int, default=1, help="Zacznij od wskazanej próby podstawowej (1-based).")
    parser.add_argument("--session-dir", type=Path, help="Katalog sesji; domyślnie nowy katalog w data/raw.")
    parser.add_argument("--dry-run", action="store_true", help="Pokaż plan bez otwierania portów i GUI.")
    parser.add_argument("--probe-hb100", action="store_true", help="Tylko sprawdź i krótko odczytaj podłączony tor HB100.")
    parser.add_argument("--probe-seconds", type=float, default=8.0, help="Czas końcowego odczytu w trybie --probe-hb100.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.prep_seconds < 0:
        raise SystemExit("--prep-seconds nie może być ujemne.")
    if args.probe_seconds <= 0:
        raise SystemExit("--probe-seconds musi być dodatnie.")
    if args.probe_hb100:
        transport, attempts = probe_hb100_transport(args.hb100_port)
        return print_probe_result(transport, attempts, args.probe_seconds)

    steps = build_initial_steps(
        include_interference=not args.skip_interference,
        hb100_only=args.hb100_only,
    )
    if not 1 <= args.start_step <= len(steps):
        raise SystemExit(f"--start-step musi należeć do zakresu 1–{len(steps)}.")
    if args.dry_run:
        print("Plan eksperymentu:")
        for step in steps:
            print(
                f"{step.number:02d}. {step.step_id}: {step.label} | {step.sensor_mode} | "
                f"{step.duration_s:g} s"
            )
        print("Po zaliczeniu 100 cm program pozwoli dodać 150 cm i kolejne dystanse co 50 cm.")
        print("Pierwszą porażkę w części rozszerzonej powtórzy po przesunięciu o 1–2 cm.")
        return 0

    session_dir = args.session_dir
    if session_dir is None:
        session_dir = DEFAULT_OUTPUT_DIR / f"hb100_a121_range_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    elif not session_dir.is_absolute():
        session_dir = ROOT / session_dir
    app = QApplication.instance() or QApplication(sys.argv)
    window = GuidedExperimentWindow(
        steps=steps,
        cues=build_breathing_cues(),
        session_dir=session_dir,
        hb100_port=args.hb100_port,
        a121_port=args.a121_port,
        hb100_only=args.hb100_only,
        prep_seconds=args.prep_seconds,
        start_step=args.start_step,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
