"""BLE receiver for a chest heart-rate strap (Suunto Smart Sensor / Suunto HR belt, or any standard one).

Heart-rate straps use the Bluetooth Heart Rate Service (0x180D): every notification of the Heart Rate
Measurement characteristic (0x2A37) carries the current heart rate and, when the strap measures them,
the R-R intervals of the beats since the last notification in 1/1024 s.  The RR intervals are the
part that matters here: they time each beat, which is what the radar's heart estimate is checked
against.  A notification arrives about once a second, so a row holds the beats of that second.
"""

from __future__ import annotations

import asyncio
import csv
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .imu import click_safe_echo
from .paths import DATA_DIR

HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
BATTERY_LEVEL_UUID = "00002a19-0000-1000-8000-00805f9b34fb"
# Name fragments for when the service UUID is not in the advertisement (macOS sometimes omits it).
HEART_STRAP_NAME_KEYWORDS = ("suunto", "movesense", "smart sensor", "hrm", "polar h", "heart", "hr ")
RAW_HEART_STRAP_DIR = DATA_DIR / "raw" / "heart_strap"
HEART_STRAP_COLUMNS = ["host_time_ms", "heart_rate_bpm", "rr_ms", "sensor_contact", "energy_kj", "payload_hex"]
HEART_STRAP_QUALITY_COLUMNS = ["host_time_ms", "heart_rate_bpm", "rr_beats", "rr_artifact_share", "rr_irregularity", "hr_ok", "rr_ok"]
# An RR interval is an artefact when it is outside 30-200 /min or more than this far from the rate the strap reports
# in the same notification: a clean strap stays within ~5% (2026-11-06 test), a loose or flat one (2026-11-07,
# battery 10%) sends extra short beats.
RR_RANGE_MS = (300.0, 2000.0)
RR_MAX_DEVIATION = 0.30
HEART_QUALITY_WINDOW_S = 60.0
# Share of artefact RR intervals in the window above which the RR intervals are not used (reported rate still is) ...
RR_MAX_ARTIFACT_SHARE = 0.20
# ... and above which the reported rate goes too: on 2026-11-07 it agreed with the radar (54-58 vs 56-59 /min)
# while 37-65% of the intervals were artefacts and climbed from 56 to 85 /min, the radar staying at ~55, once
# 78-100% were.
HR_MAX_ARTIFACT_SHARE = 0.70
# A strap can also lock onto the spurious beats completely: on 2026-11-07 it reported 104-126 /min at rest (radar
# ~55) with RR intervals that agreed with that rate.  Beats at rest are regular, spurious ones are not: the median
# change between successive intervals over the median interval was 0.02-0.04 on clean minutes, 0.11-0.12 where
# some beats were spurious but the rate still matched the radar, and 0.19-0.59 on every bad minute.
RR_MAX_IRREGULARITY = 0.06
HR_MAX_IRREGULARITY = 0.15


@dataclass(frozen=True)
class HeartRateMeasurement:
    heart_rate_bpm: int
    rr_ms: tuple[float, ...] = field(default_factory=tuple)
    sensor_contact: bool | None = None  # None when the strap does not report contact
    energy_kj: int | None = None


def parse_heart_rate_measurement(payload: bytes | bytearray) -> HeartRateMeasurement:
    """Decode one Heart Rate Measurement notification (Bluetooth GATT spec, characteristic 0x2A37)."""

    data = bytes(payload)
    if len(data) < 2:
        raise ValueError(f"Heart-rate packet too short: {data.hex()}")
    flags = data[0]
    index = 1
    if flags & 0x01:  # heart rate as uint16
        if len(data) < 3:
            raise ValueError(f"Heart-rate packet too short for a 16-bit rate: {data.hex()}")
        heart_rate = int.from_bytes(data[1:3], "little")
        index = 3
    else:
        heart_rate = data[1]
        index = 2
    contact_supported = bool(flags & 0x04)
    contact = bool(flags & 0x02) if contact_supported else None
    energy = None
    if flags & 0x08:
        if len(data) < index + 2:
            raise ValueError(f"Heart-rate packet too short for energy expended: {data.hex()}")
        energy = int.from_bytes(data[index : index + 2], "little")
        index += 2
    rr: list[float] = []
    if flags & 0x10:
        while index + 1 < len(data):
            rr.append(int.from_bytes(data[index : index + 2], "little") * 1000.0 / 1024.0)
            index += 2
    return HeartRateMeasurement(int(heart_rate), tuple(round(value, 1) for value in rr), contact, energy)


def _load_bleak() -> tuple[Any, Any]:
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError as exc:
        raise RuntimeError("Missing BLE dependency. Run `uv sync` to install `bleak`.") from exc
    return BleakScanner, BleakClient


def _looks_like_strap(name: str, services: list[str]) -> bool:
    lowered = f"{name.lower()} "
    return HEART_RATE_SERVICE_UUID in services or any(word in lowered for word in HEART_STRAP_NAME_KEYWORDS)


async def discover_heart_straps(timeout_s: float = 8.0, include_all: bool = False) -> list[dict[str, str]]:
    """Advertising heart-rate straps (by the Heart Rate service or a known name); every device with ``include_all``."""

    BleakScanner, _BleakClient = _load_bleak()
    found: dict[str, dict[str, str]] = {}

    def on_detect(device: Any, advertisement: Any) -> None:
        name = device.name or advertisement.local_name or ""
        services = [uuid.lower() for uuid in (advertisement.service_uuids or [])]
        matches = _looks_like_strap(name, services)
        if include_all or matches:
            rssi = getattr(advertisement, "rssi", None)
            found[device.address] = {
                "name": name or "(unnamed)",
                "address": device.address,
                "rssi": "" if rssi is None else str(rssi),
                "matches": "service" if HEART_RATE_SERVICE_UUID in services else ("name" if matches else "no"),
            }

    scanner = BleakScanner(detection_callback=on_detect)
    await scanner.start()
    await asyncio.sleep(timeout_s)
    await scanner.stop()
    return sorted(found.values(), key=lambda item: -int(item["rssi"] or -999))


async def _find_strap(BleakScanner: Any, selector: str | None, timeout_s: float) -> Any | None:
    wanted = (selector or "").lower()

    def matches(device: Any, advertisement: Any) -> bool:
        name = device.name or advertisement.local_name or ""
        if wanted:
            return wanted in name.lower() or wanted == str(device.address).lower()
        return _looks_like_strap(name, [uuid.lower() for uuid in (advertisement.service_uuids or [])])

    return await BleakScanner.find_device_by_filter(matches, timeout=timeout_s)


class HeartStrapCapture:
    """Background BLE capture of heart rate and RR intervals; the connect/stop/save shape of the other captures."""

    def __init__(self, output_dir: str | Path = RAW_HEART_STRAP_DIR, device: str | None = None, scan_timeout_s: float = 10.0) -> None:
        self.output_dir = Path(output_dir)
        self.device = device
        self.scan_timeout_s = float(scan_timeout_s)
        self.running = False
        self.connected = False
        self.status = "idle"
        self.device_name: str | None = None
        self.battery_percent: int | None = None
        self.rows: list[list[Any]] = []
        self.invalid = 0
        self.read_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._failed_message: str | None = None

    def connect(self, selector: str | None = None) -> bool:
        self.device = selector or self.device
        self.running = True
        self.status = "scanning"
        self._stop_event.clear()
        self._connected_event.clear()
        self.read_thread = threading.Thread(target=self._thread_main, daemon=True)
        self.read_thread.start()
        if not self._connected_event.wait(max(5.0, self.scan_timeout_s + 8.0)):
            self.status = self._failed_message or "Timed out while connecting to the heart-rate strap."
            self.stop()
        if not self.connected:
            click_safe_echo(self.status)
        return self.connected

    def stop(self) -> None:
        self.running = False
        self._stop_event.set()
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=5.0)
        self.connected = False

    def data_count(self) -> int:
        with self._lock:
            return len(self.rows)

    def snapshot_rows(self, start: int = 0) -> list[list[Any]]:
        with self._lock:
            return [list(row) for row in self.rows[start:]]

    def latest_heart_rate(self) -> int | None:
        with self._lock:
            return int(self.rows[-1][1]) if self.rows else None

    def save(self, path: str | Path | None = None) -> Path:
        with self._lock:
            rows = [list(row) for row in self.rows]
        if len(rows) < 3:
            raise ValueError("Not enough heart-rate notifications to save; at least 3 are required.")
        if path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"heart_strap_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        write_heart_strap_csv(Path(path), rows)
        return Path(path)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_ble())
        except Exception as exc:
            self._failed_message = str(exc)
            self.status = self._failed_message
            click_safe_echo(f"Heart-rate strap BLE error: {exc}")
        finally:
            self.running = False
            self.connected = False
            self._connected_event.set()

    async def _run_ble(self) -> None:
        BleakScanner, BleakClient = _load_bleak()
        device = await _find_strap(BleakScanner, self.device, self.scan_timeout_s)
        if device is None:
            raise RuntimeError(f"Heart-rate strap not found ({self.device or 'Heart Rate service 0x180D'}). Wet the strap's electrodes and put it on: most straps only advertise while worn.")
        self.device_name = getattr(device, "name", None) or str(device.address)
        self.status = f"connecting to {self.device_name}"
        async with BleakClient(device) as client:
            self.connected = bool(client.is_connected)
            if not self.connected:
                raise RuntimeError("BLE connection failed.")
            try:
                self.battery_percent = int((await client.read_gatt_char(BATTERY_LEVEL_UUID))[0])
            except Exception:
                self.battery_percent = None
            await client.start_notify(HEART_RATE_MEASUREMENT_UUID, self._on_notification)
            self.status = "connected"
            self._connected_event.set()
            try:
                while not self._stop_event.is_set() and client.is_connected:
                    await asyncio.sleep(0.10)
            finally:
                try:
                    await client.stop_notify(HEART_RATE_MEASUREMENT_UUID)
                except Exception:
                    pass

    def _on_notification(self, _sender: Any, payload: bytearray) -> None:
        host_ms = time.time() * 1000.0
        try:
            measurement = parse_heart_rate_measurement(payload)
        except ValueError:
            with self._lock:
                self.invalid += 1
            return
        with self._lock:
            self.rows.append(heart_strap_row(host_ms, measurement, bytes(payload)))


def heart_strap_row(host_ms: float, measurement: HeartRateMeasurement, payload: bytes = b"") -> list[Any]:
    return [
        round(host_ms, 1),
        measurement.heart_rate_bpm,
        ";".join(f"{value:.1f}" for value in measurement.rr_ms),
        "" if measurement.sensor_contact is None else int(measurement.sensor_contact),
        "" if measurement.energy_kj is None else measurement.energy_kj,
        payload.hex(),
    ]


def write_heart_strap_csv(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEART_STRAP_COLUMNS)
        writer.writerows(rows)


def _rr_values(cell: Any) -> list[float]:
    text = "" if cell is None else str(cell)
    if text in ("", "-", "nan"):
        return []
    return [float(part) for part in text.split(";") if part]


def heart_strap_quality(source: Any, window_s: float = HEART_QUALITY_WINDOW_S) -> "pd.DataFrame":
    """Per notification: whether the strap's rate (``hr_ok``) and RR intervals (``rr_ok``) can be trusted.

    ``source`` is a heart-strap CSV, a DataFrame with :data:`HEART_STRAP_COLUMNS`, or rows as written to it.  Each
    RR interval is checked against the rate reported with it (:data:`RR_RANGE_MS`, :data:`RR_MAX_DEVIATION`); the
    share of artefacts within ``window_s`` around a notification decides (:data:`RR_MAX_ARTIFACT_SHARE`,
    :data:`HR_MAX_ARTIFACT_SHARE`), and so does how irregular the beats are (:data:`RR_MAX_IRREGULARITY`,
    :data:`HR_MAX_IRREGULARITY`), which catches a strap locked onto spurious beats.  Lost skin contact or fewer
    than 4 beats in the window makes both untrusted.
    """

    import numpy as np
    import pandas as pd

    if isinstance(source, (str, Path)):
        frame = pd.read_csv(source)
    elif isinstance(source, pd.DataFrame):
        frame = source
    else:
        frame = pd.DataFrame([list(row) for row in source], columns=HEART_STRAP_COLUMNS)
    times = pd.to_numeric(frame["host_time_ms"], errors="coerce").to_numpy(dtype=float)
    rates = pd.to_numeric(frame["heart_rate_bpm"], errors="coerce").to_numpy(dtype=float)
    contact = pd.to_numeric(frame.get("sensor_contact", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
    beat_times, beat_bad, beat_rr = [], [], []
    beats_per_row = []
    for at, rate, cell in zip(times, rates, frame["rr_ms"]):
        values = _rr_values(cell)
        beats_per_row.append(len(values))
        for rr in values:
            implied = 60000.0 / rr if rr > 0 else np.inf
            bad = not (RR_RANGE_MS[0] <= rr <= RR_RANGE_MS[1]) or not (rate > 0) or abs(implied - rate) / rate > RR_MAX_DEVIATION
            beat_times.append(at)
            beat_bad.append(bad)
            beat_rr.append(rr)
    beat_times_arr = np.asarray(beat_times, dtype=float)
    bad_cum = np.concatenate([[0], np.cumsum(np.asarray(beat_bad, dtype=float))])
    half = window_s * 1000.0 / 2.0
    lo = np.searchsorted(beat_times_arr, times - half, side="left")
    hi = np.searchsorted(beat_times_arr, times + half, side="right")
    count = hi - lo
    share = np.where(count > 0, (bad_cum[hi] - bad_cum[lo]) / np.maximum(count, 1), np.nan)
    rr_arr = np.asarray(beat_rr, dtype=float)
    irregularity = np.full(len(times), np.nan)
    for row, (a, b) in enumerate(zip(lo, hi)):
        window = rr_arr[a:b]
        if len(window) >= 4:
            irregularity[row] = float(np.median(np.abs(np.diff(window))) / np.median(window))
    regular = np.isfinite(irregularity)
    no_contact = contact == 0
    hr_ok = (count > 0) & (share <= HR_MAX_ARTIFACT_SHARE) & ~no_contact & (rates > 0)
    hr_ok &= regular & (np.nan_to_num(irregularity, nan=1.0) <= HR_MAX_IRREGULARITY)
    rr_ok = hr_ok & (share <= RR_MAX_ARTIFACT_SHARE) & (np.nan_to_num(irregularity, nan=1.0) <= RR_MAX_IRREGULARITY)
    return pd.DataFrame(
        {
            "host_time_ms": times,
            "heart_rate_bpm": rates,
            "rr_beats": beats_per_row,
            "rr_artifact_share": np.round(share, 3),
            "rr_irregularity": np.round(irregularity, 3),
            "hr_ok": hr_ok,
            "rr_ok": rr_ok,
        },
        columns=HEART_STRAP_QUALITY_COLUMNS,
    )


def heart_strap_quality_path(recording: str | Path) -> Path:
    recording = Path(recording)
    return recording.with_name(f"{recording.stem}_quality.csv")


def write_heart_strap_quality(recording: str | Path) -> tuple[Path, "pd.DataFrame"]:
    """Flag a saved heart-strap CSV and write the flags beside it as ``<name>_quality.csv``."""

    quality = heart_strap_quality(recording)
    path = heart_strap_quality_path(recording)
    quality.to_csv(path, index=False)
    return path, quality


def heart_strap_summary(rows: list[list[Any]], duration_s: float | None = None) -> dict[str, Any]:
    """Notifications, beats, the median rate and how much of the run the strap covered (rows as written to CSV)."""

    rates = [float(row[1]) for row in rows if float(row[1]) > 0]
    beats = sum(len([part for part in str(row[2]).split(";") if part]) for row in rows)
    contact = [row[3] for row in rows if row[3] != ""]
    span_s = (float(rows[-1][0]) - float(rows[0][0])) / 1000.0 if len(rows) > 1 else 0.0
    gaps = [(float(b[0]) - float(a[0])) / 1000.0 for a, b in zip(rows, rows[1:])]
    rates.sort()
    return {
        "notifications": len(rows),
        "rr_beats": beats,
        "median_heart_rate_bpm": rates[len(rates) // 2] if rates else None,
        "span_s": round(span_s, 1),
        "coverage_percent": round(100.0 * span_s / duration_s, 1) if duration_s else None,
        "largest_gap_s": round(max(gaps), 2) if gaps else None,
        "no_contact_share": round(sum(1 for value in contact if int(value) == 0) / len(contact), 3) if contact else None,
        **_quality_shares(rows),
    }


def _quality_shares(rows: list[list[Any]]) -> dict[str, Any]:
    if not rows:
        return {"hr_ok_share": None, "rr_ok_share": None}
    quality = heart_strap_quality(rows)
    return {"hr_ok_share": round(float(quality["hr_ok"].mean()), 3), "rr_ok_share": round(float(quality["rr_ok"].mean()), 3)}
