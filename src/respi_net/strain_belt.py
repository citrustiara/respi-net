"""BLE receiver for the ESP32 + HX711 strain-gauge chest belt (Kabat / parallel thesis).

The belt advertises as ``Pas_Tensometryczny`` and notifies ~10 Hz samples on one characteristic.
The unit tested on 2026-10-11 sends the bare HX711 count as text (e.g. ``b"-5261520"``) and
exposes no other characteristics. Per the thesis the full protocol sends UTF-8 JSON
``{"seq": int, "raw": float, "ts": int, "checksum": str}`` per notification, where ``checksum = md5(f"{seq}:{raw:.3f}:{ts}").hexdigest()[:8]``. Both formats are parsed.
The thesis also mentions a control (write) and a calibration (read/notify) characteristic,
but their UUIDs are not published; ``probe`` lists every characteristic so they can be found.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .imu import click_safe_echo
from .paths import DATA_DIR

STRAIN_BELT_DEVICE_NAME = "Pas_Tensometryczny"
STRAIN_BELT_SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
STRAIN_BELT_DATA_CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
# Name fragments used when the service UUID is not in the advertisement (macOS often omits it).
STRAIN_BELT_NAME_KEYWORDS = ("esp32", "breath", "oddech", "strain", "tenso", "belt", "pas", "respi", "hx711")
# The firmware sends -1 when an HX711 read fails; the belt itself sits near -5.3e6 counts.
HX711_ERROR_VALUE = -1.0
RAW_STRAIN_BELT_DIR = DATA_DIR / "raw" / "strain_belt"
STRAIN_BELT_COLUMNS = ["host_time_ms", "seq", "device_ts", "raw", "checksum_ok", "payload"]
# The belt streams ~10 Hz and BLE delivers it in bursts with arrival gaps of up to ~0.35 s.  A longer silence
# is a dropout: the link fell (a weak signal, about -95 dBm, does it) or the belt stalled.
STRAIN_BELT_MAX_GAP_S = 1.0
# Connected but silent this long counts as a dropout too: the capture drops the link and connects again.
STRAIN_BELT_STALL_S = 5.0
STRAIN_BELT_RECONNECT_PAUSE_S = 1.0
# The first connection fails now and then at a weak signal ("disconnected" while connecting); try it this often.
STRAIN_BELT_CONNECT_ATTEMPTS = 3


def strain_belt_gaps(host_time_ms: Any, max_gap_s: float = STRAIN_BELT_MAX_GAP_S) -> list[tuple[float, float]]:
    """Dropouts of a belt capture: ``(last sample before, first sample after)`` in host ms, longer than ``max_gap_s``."""

    import numpy as np

    stamps = np.asarray(host_time_ms, dtype=float)
    stamps = stamps[np.isfinite(stamps)]
    if len(stamps) < 2:
        return []
    jumps = np.flatnonzero(np.diff(stamps) > max_gap_s * 1000.0)
    return [(float(stamps[i]), float(stamps[i + 1])) for i in jumps]


@dataclass(frozen=True)
class StrainBeltPacket:
    raw: float
    seq: int | None = None
    ts: float | None = None
    checksum_ok: bool | None = None  # None when the packet carries no checksum
    text: str = ""


def strain_belt_checksum(seq: int, raw: float, ts: Any) -> str:
    """Truncated MD5 used by the belt firmware/app (ble_protocol.dart, listing 3.1)."""
    ts_text = str(int(ts)) if isinstance(ts, float) and ts.is_integer() else str(ts)
    return hashlib.md5(f"{seq}:{raw:.3f}:{ts_text}".encode()).hexdigest()[:8]


def parse_strain_belt_packet(payload: bytes | bytearray | str) -> StrainBeltPacket:
    text = payload if isinstance(payload, str) else bytes(payload).decode("utf-8", errors="replace")
    text = text.strip().strip("\x00").strip()
    if not text:
        raise ValueError("Empty strain-belt packet.")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return StrainBeltPacket(raw=float(obj), text=text)
    if isinstance(obj, dict):
        raw_value = obj.get("raw", obj.get("value"))
        if raw_value is None:
            raise ValueError(f"Strain-belt JSON has no 'raw' field: {text!r}")
        raw = float(raw_value)
        seq = int(obj["seq"]) if obj.get("seq") is not None else None
        ts = obj.get("ts")
        checksum = obj.get("checksum")
        checksum_ok = None
        if checksum is not None and seq is not None and ts is not None:
            checksum_ok = str(checksum).lower() == strain_belt_checksum(seq, raw, ts)
        return StrainBeltPacket(
            raw=raw,
            seq=seq,
            ts=float(ts) if ts is not None else None,
            checksum_ok=checksum_ok,
            text=text,
        )
    # Fallback for "seq,raw,ts" style or "raw: 123.4" style text.
    for token in reversed(text.replace(";", ",").replace(":", ",").split(",")):
        try:
            return StrainBeltPacket(raw=float(token), text=text)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised strain-belt payload: {text!r}")


def _load_bleak() -> tuple[Any, Any]:
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError as exc:
        raise RuntimeError("Missing BLE dependency. Run `uv sync` to install `bleak`.") from exc
    return BleakScanner, BleakClient


def _looks_like_belt(name: str, services: list[str]) -> bool:
    lowered = name.lower()
    return STRAIN_BELT_SERVICE_UUID in services or any(word in lowered for word in STRAIN_BELT_NAME_KEYWORDS)


async def discover_strain_belt_devices(timeout_s: float = 6.0, include_all: bool = False) -> list[dict[str, str]]:
    BleakScanner, _BleakClient = _load_bleak()
    found: dict[str, dict[str, str]] = {}

    def on_detect(device: Any, advertisement: Any) -> None:
        name = device.name or advertisement.local_name or ""
        services = [uuid.lower() for uuid in (advertisement.service_uuids or [])]
        matches = _looks_like_belt(name, services)
        if include_all or matches:
            rssi = getattr(advertisement, "rssi", None)
            found[device.address] = {
                "name": name or "(unnamed)",
                "address": device.address,
                "rssi": "" if rssi is None else str(rssi),
                "matches": "service" if STRAIN_BELT_SERVICE_UUID in services else ("name" if matches else "no"),
            }

    scanner = BleakScanner(on_detect)
    await scanner.start()
    try:
        await asyncio.sleep(timeout_s)
    finally:
        await scanner.stop()
    return sorted(found.values(), key=lambda d: -int(d["rssi"] or -999))


async def _find_belt(BleakScanner: Any, selector: str | None, timeout_s: float) -> Any | None:
    selector_norm = selector.lower().strip() if selector else ""

    def matches(device: Any, advertisement: Any) -> bool:
        name = device.name or getattr(advertisement, "local_name", "") or ""
        services = [uuid.lower() for uuid in (getattr(advertisement, "service_uuids", None) or [])]
        if selector_norm:
            return selector_norm == device.address.lower() or selector_norm in name.lower()
        return STRAIN_BELT_SERVICE_UUID in services or name == STRAIN_BELT_DEVICE_NAME

    return await BleakScanner.find_device_by_filter(matches, timeout=timeout_s)


async def probe_strain_belt(
    selector: str | None = None,
    scan_timeout_s: float = 8.0,
    listen_s: float = 5.0,
    echo=click_safe_echo,
) -> int:
    """Connect, print the GATT table, subscribe to every notifiable characteristic and echo payloads.

    Returns the number of data-characteristic notifications that parsed into a sample.
    """
    BleakScanner, BleakClient = _load_bleak()
    device = await _find_belt(BleakScanner, selector, scan_timeout_s)
    if device is None:
        raise RuntimeError(
            f"Strain belt not found ({selector or 'service ' + STRAIN_BELT_SERVICE_UUID}). "
            "Power it on, or pass --device <name or address> from `respi strain-belt-devices --all`."
        )
    echo(f"Connecting to {getattr(device, 'name', None) or '(unnamed)'}  {device.address}")
    parsed = 0
    async with BleakClient(device) as client:
        notifiable: list[str] = []
        for service in client.services:
            echo(f"service {service.uuid}")
            for char in service.characteristics:
                value = ""
                if "read" in char.properties:
                    try:
                        value = f"  = {bytes(await client.read_gatt_char(char.uuid))!r}"
                    except Exception as exc:
                        value = f"  (read failed: {exc})"
                echo(f"  char {char.uuid}  [{','.join(char.properties)}]{value}")
                if {"notify", "indicate"} & set(char.properties):
                    notifiable.append(char.uuid)

        def make_handler(uuid: str):
            def handler(_sender: Any, payload: bytearray) -> None:
                nonlocal parsed
                tag = "data" if uuid.lower() == STRAIN_BELT_DATA_CHARACTERISTIC_UUID else uuid[:8]
                try:
                    packet = parse_strain_belt_packet(payload)
                    if tag == "data":
                        parsed += 1
                    echo(f"[{tag}] raw={packet.raw:.3f} seq={packet.seq} ts={packet.ts} checksum_ok={packet.checksum_ok}  {packet.text!r}")
                except ValueError:
                    echo(f"[{tag}] {bytes(payload)!r}")
            return handler

        for uuid in notifiable:
            await client.start_notify(uuid, make_handler(uuid))
        echo(f"Listening {listen_s:.0f}s on {len(notifiable)} notifiable characteristic(s)...")
        await asyncio.sleep(listen_s)
        for uuid in notifiable:
            try:
                await client.stop_notify(uuid)
            except Exception:
                pass
    return parsed


def strain_belt_gap_summary(capture: "StrainBeltCapture", max_gap_s: float = STRAIN_BELT_MAX_GAP_S) -> str:
    """One line on the dropouts of a finished capture: how many, how long, and the share of time covered."""

    with capture._lock:
        stamps = [row[0] for row in capture.rows]
    gaps = strain_belt_gaps(stamps, max_gap_s)
    if len(stamps) < 2:
        return "Dropouts: no samples."
    span_s = (stamps[-1] - stamps[0]) / 1000.0
    lost_s = sum(end - start for start, end in gaps) / 1000.0
    if not gaps:
        return f"Dropouts: none (no silence over {max_gap_s:g} s in {span_s:.0f} s); disconnects {capture.disconnects}."
    longest = max(end - start for start, end in gaps) / 1000.0
    covered = 100.0 * (1.0 - lost_s / span_s) if span_s > 0 else 0.0
    return (f"Dropouts: {len(gaps)} gaps over {max_gap_s:g} s, {lost_s:.1f} s lost (longest {longest:.1f} s), "
            f"{covered:.1f}% of {span_s:.0f} s covered; disconnects {capture.disconnects}. The analysis leaves the gaps out.")


class StrainBeltCapture:
    """Background BLE capture of belt samples; mirrors the connect/stop/save shape of the IMU captures."""

    def __init__(
        self,
        output_dir: str | Path = RAW_STRAIN_BELT_DIR,
        device: str | None = None,
        scan_timeout_s: float = 10.0,
        reconnect: bool = True,
        stall_s: float = STRAIN_BELT_STALL_S,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.device = device
        self.scan_timeout_s = float(scan_timeout_s)
        self.reconnect = bool(reconnect)
        self.stall_s = float(stall_s)
        self.disconnects = 0
        self.last_packet_ms: float | None = None
        self.running = False
        self.connected = False
        self.status = "idle"
        self.rows: list[list[Any]] = []
        self.received = 0
        self.invalid = 0
        self.bad_checksum = 0
        self.glitches = 0
        self.missing = 0
        self.last_seq: int | None = None
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
        if not self._connected_event.wait(max(5.0, (self.scan_timeout_s + 8.0) * STRAIN_BELT_CONNECT_ATTEMPTS)):
            self.status = self._failed_message or "Timed out while connecting to the strain belt."
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

    def latest_raw(self) -> float | None:
        with self._lock:
            return self.rows[-1][3] if self.rows else None

    def gaps(self, max_gap_s: float = STRAIN_BELT_MAX_GAP_S) -> list[tuple[float, float]]:
        """Dropouts in the samples so far (see :func:`strain_belt_gaps`)."""

        with self._lock:
            stamps = [row[0] for row in self.rows]
        return strain_belt_gaps(stamps, max_gap_s)

    def save(self) -> Path:
        with self._lock:
            rows = [list(row) for row in self.rows]
        if len(rows) < 10:
            raise ValueError("Not enough samples to save; at least 10 are required.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"strain_belt_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(STRAIN_BELT_COLUMNS)
            writer.writerows(rows)
        return path

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_ble())
        except Exception as exc:
            self._failed_message = str(exc)
            self.status = self._failed_message
            click_safe_echo(f"Strain belt BLE error: {exc}")
        finally:
            self.running = False
            self.connected = False
            self._connected_event.set()

    async def _run_ble(self) -> None:
        """Connect, stream, and -- with ``reconnect`` -- connect again after every dropout until stopped.

        Only the first connection may fail the capture, after :data:`STRAIN_BELT_CONNECT_ATTEMPTS` tries; later ones are retried for as long as it runs, and the
        silence between shows up as a gap in the samples (:meth:`gaps`) for the analysis to leave out.
        """

        BleakScanner, BleakClient = _load_bleak()
        first = True
        attempts = 0
        while not self._stop_event.is_set():
            try:
                await self._stream_once(BleakScanner, BleakClient, first)
            except Exception as exc:
                if first and not self.connected:
                    attempts += 1
                    if attempts >= STRAIN_BELT_CONNECT_ATTEMPTS:
                        raise
                    click_safe_echo(f"Strain belt: connection failed ({exc}); retrying ({attempts + 1}/{STRAIN_BELT_CONNECT_ATTEMPTS})...")
                    await asyncio.sleep(STRAIN_BELT_RECONNECT_PAUSE_S)
                    continue
                self.status = f"reconnecting ({exc})"
            if self._stop_event.is_set() or not self.reconnect:
                return
            if self.connected:
                self.disconnects += 1
                lost_at = datetime.fromtimestamp((self.last_packet_ms or time.time() * 1000.0) / 1000.0)
                click_safe_echo(f"Strain belt: link lost after the sample at {lost_at:%H:%M:%S}; reconnecting...")
            first = False
            self.connected = False
            self.status = "reconnecting"
            await asyncio.sleep(STRAIN_BELT_RECONNECT_PAUSE_S)

    async def _stream_once(self, BleakScanner: Any, BleakClient: Any, first: bool) -> None:
        device = await _find_belt(BleakScanner, self.device, self.scan_timeout_s)
        if device is None:
            raise RuntimeError(f"Strain belt not found ({self.device or 'service ' + STRAIN_BELT_SERVICE_UUID}).")
        self.status = f"connecting to {getattr(device, 'name', None) or device.address}"
        async with BleakClient(device) as client:
            if not client.is_connected:
                raise RuntimeError("BLE connection failed.")
            await client.start_notify(STRAIN_BELT_DATA_CHARACTERISTIC_UUID, self._on_notification)
            connected_ms = time.time() * 1000.0
            self.connected = True
            self.status = "connected"
            if not first:
                down_s = (connected_ms - (self.last_packet_ms or connected_ms)) / 1000.0
                click_safe_echo(f"Strain belt: reconnected, {down_s:.1f} s without samples.")
            self._connected_event.set()
            try:
                while not self._stop_event.is_set() and client.is_connected:
                    await asyncio.sleep(0.10)
                    last = max(self.last_packet_ms or 0.0, connected_ms)
                    if time.time() * 1000.0 - last > self.stall_s * 1000.0:
                        self.status = "stalled"
                        break
            finally:
                try:
                    await client.stop_notify(STRAIN_BELT_DATA_CHARACTERISTIC_UUID)
                except Exception:
                    pass

    def _on_notification(self, _sender: Any, payload: bytearray) -> None:
        host_ms = time.time() * 1000.0
        try:
            packet = parse_strain_belt_packet(payload)
        except ValueError:
            with self._lock:
                self.invalid += 1
            return
        with self._lock:
            if packet.checksum_ok is False:
                self.bad_checksum += 1
                return
            if packet.raw == HX711_ERROR_VALUE:
                self.glitches += 1
                return
            if packet.seq is not None:
                if self.last_seq is not None and packet.seq > self.last_seq + 1:
                    self.missing += packet.seq - self.last_seq - 1
                self.last_seq = packet.seq
            self.received += 1
            self.last_packet_ms = host_ms
            self.rows.append([
                round(host_ms, 1),
                "" if packet.seq is None else packet.seq,
                "" if packet.ts is None else packet.ts,
                packet.raw,
                "" if packet.checksum_ok is None else int(packet.checksum_ok),
                packet.text,
            ])


COACH_COLUMNS = STRAIN_BELT_COLUMNS + ["elapsed_s", "phase_index", "phase", "cue"]


def phase_label(phases: list[Any], index: int) -> str:
    """Phase kind, with holds split into ``hold_in`` (full lungs) and ``hold_out`` (empty lungs)."""
    kind = phases[index].kind
    if kind == "hold" and index > 0:
        previous = phases[index - 1].kind
        return {"inhale": "hold_in", "exhale": "hold_out"}.get(previous, kind)
    return kind


def label_coach_rows(
    rows: list[list[Any]],
    pattern: Any,
    run_start_s: float,
    end_s: float,
    pauses: list[tuple[float, float]] = (),
) -> list[list[Any]]:
    """Attach the coached phase to every belt sample taken during a run.

    All times are wall-clock seconds (``time.time()``), like ``host_time_ms``; samples
    received while the coach was paused are dropped.
    """
    phases = pattern.phases
    labelled: list[list[Any]] = []
    for row in rows:
        t = float(row[0]) / 1000.0
        if t < run_start_s or t > end_s or any(start <= t < stop for start, stop in pauses):
            continue
        paused = sum(min(stop, t) - start for start, stop in pauses if start < t)
        elapsed = t - run_start_s - paused
        index = pattern.phase_index_at(elapsed) if elapsed < pattern.duration_s else None
        if index is None:
            continue
        labelled.append(list(row) + [round(elapsed, 3), index, phase_label(phases, index), phases[index].cue])
    return labelled


CUE_COLUMNS = ["start_s", "end_s", "kind", "cue", "detail"]


def cue_sidecar_path(recording: str | Path) -> Path:
    """``<stem>_cues.csv`` beside a coach recording; ``run_XX_<pattern>_cues.csv`` beside a session's ``..._belt.csv``."""
    recording = Path(recording)
    stem = recording.stem.removesuffix("_belt") if recording.stem.startswith("run_") else recording.stem
    return recording.with_name(f"{stem}_cues.csv")


def save_coach_recording(
    rows: list[list[Any]],
    pattern_id: str,
    output_dir: str | Path = RAW_STRAIN_BELT_DIR,
    phases: list[Any] | None = None,
) -> Path:
    """Write a coached run, plus the cue schedule beside it so the run never depends on the pattern file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"strain_belt_coach_{pattern_id}_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COACH_COLUMNS)
        writer.writerows(rows)
    if phases:
        with cue_sidecar_path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CUE_COLUMNS)
            writer.writerows([phase.start_s, phase.end_s, phase.kind, phase.cue, phase.detail] for phase in phases)
    return path


def load_coach_cues(recording: str | Path) -> list[dict[str, Any]]:
    """Cue schedule of a coached belt run: the sidecar, or else the pattern named in the file name."""
    sidecar = cue_sidecar_path(recording)
    if sidecar.exists():
        with sidecar.open(newline="", encoding="utf-8") as handle:
            return [
                {**row, "start_s": float(row["start_s"]), "end_s": float(row["end_s"])}
                for row in csv.DictReader(handle)
            ]
    from .breathing import load_pattern

    stem = Path(recording).stem
    if not stem.startswith("strain_belt_coach_"):
        raise ValueError(f"{recording} is not a coached belt run (no cue sidecar).")
    pattern_id = stem.removeprefix("strain_belt_coach_").rsplit("_", 2)[0]
    pattern = load_pattern(pattern_id)
    return [
        {"start_s": p.start_s, "end_s": p.end_s, "kind": p.kind, "cue": p.cue, "detail": p.detail}
        for p in pattern.phases
    ]


def cue_origin_ms(cues: list[dict[str, Any]]) -> float | None:
    """Host wall time of the pattern's start, when the cues carry wall times (guided radar sessions)."""
    first = cues[0] if cues else {}
    if first.get("start_wall_ms") in (None, ""):
        return None
    return float(first["start_wall_ms"]) - float(first["start_s"]) * 1000.0
