from __future__ import annotations

import asyncio
import struct
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .imu import IMU_COLUMNS, click_safe_echo
from .paths import RAW_IMU_DIR

IPHONE_IMU_DEVICE_NAME = "RespiPhoneIMU"
IPHONE_IMU_SERVICE_UUID = "7b61b4e2-f5b4-4c90-8c7f-a7b2f1e8f4d0"
IPHONE_IMU_DATA_CHARACTERISTIC_UUID = "7b61b4e3-f5b4-4c90-8c7f-a7b2f1e8f4d0"
IPHONE_IMU_CONTROL_CHARACTERISTIC_UUID = "7b61b4e4-f5b4-4c90-8c7f-a7b2f1e8f4d0"
IPHONE_IMU_PAYLOAD_VERSION = 1

_HEADER = struct.Struct("<BBH")
_SAMPLE = struct.Struct("<Ihhhhhh")


@dataclass(frozen=True)
class IPhoneImuSample:
    """One decoded iPhone IMU sample in the same units as the ESP32 IMU CSV."""

    time_ms: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass(frozen=True)
class IPhoneImuPacket:
    sequence: int
    samples: list[IPhoneImuSample]


@dataclass(frozen=True)
class IPhoneImuProbeResult:
    name: str
    address: str
    connected: bool
    batches: int
    samples: int
    dropped_batches: int
    first_sample: IPhoneImuSample | None
    last_sequence: int | None


def parse_iphone_imu_packet(payload: bytes | bytearray | memoryview) -> IPhoneImuPacket:
    """Decode a compact BLE IMU notification from the iPhone app.

    Packet format, little-endian:
    - uint8 version, currently 1
    - uint8 sample_count
    - uint16 sequence
    - repeated samples:
      - uint32 time_ms since phone stream start
      - int16 ax/ay/az in milli-g
      - int16 gx/gy/gz in centi-degrees-per-second
    """

    data = bytes(payload)
    if len(data) < _HEADER.size:
        raise ValueError("iPhone IMU packet is shorter than the header.")
    version, sample_count, sequence = _HEADER.unpack_from(data)
    if version != IPHONE_IMU_PAYLOAD_VERSION:
        raise ValueError(f"Unsupported iPhone IMU packet version: {version}")
    expected_len = _HEADER.size + sample_count * _SAMPLE.size
    if len(data) != expected_len:
        raise ValueError(f"iPhone IMU packet length {len(data)} does not match expected {expected_len}.")

    samples: list[IPhoneImuSample] = []
    offset = _HEADER.size
    for _ in range(sample_count):
        time_ms, ax_mg, ay_mg, az_mg, gx_cdps, gy_cdps, gz_cdps = _SAMPLE.unpack_from(data, offset)
        offset += _SAMPLE.size
        samples.append(
            IPhoneImuSample(
                time_ms=float(time_ms),
                ax=float(ax_mg) / 1000.0,
                ay=float(ay_mg) / 1000.0,
                az=float(az_mg) / 1000.0,
                gx=float(gx_cdps) / 100.0,
                gy=float(gy_cdps) / 100.0,
                gz=float(gz_cdps) / 100.0,
            )
        )
    return IPhoneImuPacket(sequence=int(sequence), samples=samples)


async def discover_iphone_imu_devices(timeout_s: float = 6.0, include_all: bool = False) -> list[dict[str, str]]:
    """Return visible BLE peripherals, preferring devices that look like the iPhone IMU app."""

    BleakScanner, _BleakClient = _load_bleak()
    service_uuid = IPHONE_IMU_SERVICE_UUID.lower()
    found: dict[str, dict[str, str]] = {}

    def on_detect(device: Any, advertisement: Any) -> None:
        name = device.name or advertisement.local_name or ""
        services = [uuid.lower() for uuid in (advertisement.service_uuids or [])]
        matches_iphone_imu = name == IPHONE_IMU_DEVICE_NAME or service_uuid in services
        if include_all or matches_iphone_imu:
            rssi = getattr(advertisement, "rssi", None)
            found[device.address] = {
                "name": name or "(unnamed)",
                "address": device.address,
                "rssi": "" if rssi is None else str(rssi),
                "matches": "yes" if matches_iphone_imu else "no",
            }

    scanner = BleakScanner(on_detect)
    await scanner.start()
    try:
        await asyncio.sleep(timeout_s)
    finally:
        await scanner.stop()
    return list(found.values())


async def probe_iphone_imu_device(
    selector: str | None = None,
    scan_timeout_s: float = 8.0,
    sample_seconds: float = 3.0,
    autostart: bool = True,
) -> IPhoneImuProbeResult:
    """Find, connect to, and read a few notifications from the iPhone IMU app."""

    BleakScanner, BleakClient = _load_bleak()
    device = await _find_iphone_device(BleakScanner, selector, scan_timeout_s)
    if device is None:
        target = selector or IPHONE_IMU_DEVICE_NAME
        raise RuntimeError(f"Could not find BLE device '{target}'. Open the iPhone app and start advertising.")

    batches = 0
    samples = 0
    dropped_batches = 0
    first_sample: IPhoneImuSample | None = None
    last_sequence: int | None = None
    first_packet = asyncio.Event()

    def on_notification(_sender: Any, payload: bytearray) -> None:
        nonlocal batches, samples, dropped_batches, first_sample, last_sequence
        try:
            packet = parse_iphone_imu_packet(payload)
        except ValueError:
            dropped_batches += 1
            return
        batches += 1
        samples += len(packet.samples)
        last_sequence = packet.sequence
        if first_sample is None and packet.samples:
            first_sample = packet.samples[0]
        first_packet.set()

    async with BleakClient(device) as client:
        connected = bool(client.is_connected)
        if not connected:
            raise RuntimeError("BLE connection failed.")

        await client.start_notify(IPHONE_IMU_DATA_CHARACTERISTIC_UUID, on_notification)
        try:
            if autostart:
                await _write_control_if_available(client, b"START")
            deadline = asyncio.get_running_loop().time() + max(0.0, sample_seconds)
            try:
                await asyncio.wait_for(first_packet.wait(), timeout=max(0.0, sample_seconds))
            except asyncio.TimeoutError:
                pass
            remaining_s = max(0.0, deadline - asyncio.get_running_loop().time())
            if remaining_s:
                await asyncio.sleep(remaining_s)
        finally:
            if autostart and client.is_connected:
                await _write_control_if_available(client, b"STOP")
            try:
                await client.stop_notify(IPHONE_IMU_DATA_CHARACTERISTIC_UUID)
            except Exception:
                pass

    return IPhoneImuProbeResult(
        name=getattr(device, "name", None) or "(unnamed)",
        address=device.address,
        connected=connected,
        batches=batches,
        samples=samples,
        dropped_batches=dropped_batches,
        first_sample=first_sample,
        last_sequence=last_sequence,
    )


class IPhoneIMUBluetoothCapture:
    """BLE receiver for the companion iPhone IMU app.

    The public surface intentionally mirrors BreathCapture enough for the Qt app and CLI:
    `connect`, `stop`, `save`, `data_storage`, and `running`.
    """

    def __init__(
        self,
        output_dir: str | Path = RAW_IMU_DIR,
        device: str | None = None,
        scan_timeout_s: float = 10.0,
        autostart: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.device = device
        self.scan_timeout_s = float(scan_timeout_s)
        self.autostart = autostart
        self.running = False
        self.connected = False
        self.status = "idle"
        self.data_storage: list[list[float]] = []
        self.read_thread: threading.Thread | None = None
        self.received_batches = 0
        self.dropped_batches = 0
        self.invalid_batches = 0
        self.missing_batches = 0
        self.duplicate_batches = 0
        self.sequence_resets = 0
        self.last_sequence: int | None = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._failed_message: str | None = None
        self._host_time_offset_ms: float | None = None

    def connect(self, port_name: str | None = None) -> bool:
        selector = port_name or self.device
        self.device = selector
        self.running = True
        self.connected = False
        self.status = "scanning"
        self.data_storage = []
        self.received_batches = 0
        self.dropped_batches = 0
        self.invalid_batches = 0
        self.missing_batches = 0
        self.duplicate_batches = 0
        self.sequence_resets = 0
        self.last_sequence = None
        self._host_time_offset_ms = None
        self._failed_message = None
        self._stop_event.clear()
        self._connected_event.clear()
        self.read_thread = threading.Thread(target=self._thread_main, daemon=True)
        self.read_thread.start()
        wait_s = max(5.0, self.scan_timeout_s + 8.0)
        if not self._connected_event.wait(wait_s):
            self.status = self._failed_message or "Timed out while connecting to the iPhone IMU app."
            self.stop()
            click_safe_echo(self.status)
            return False
        return self.connected

    def stop(self) -> None:
        self.running = False
        self._stop_event.set()
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=5.0)
        self.connected = False

    def save(self) -> Path:
        with self._lock:
            rows = [list(row) for row in self.data_storage]
        if len(rows) < 10:
            raise ValueError("Not enough samples to save; at least 10 are required.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"respiratory_6axis_raw_iphone_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        pd.DataFrame(rows, columns=IMU_COLUMNS).to_csv(path, index=False)
        return path

    def data_count(self) -> int:
        with self._lock:
            return len(self.data_storage)

    def snapshot_data_storage(self) -> list[list[float]]:
        with self._lock:
            return [list(row) for row in self.data_storage]

    def snapshot_data_since(self, start_index: int) -> list[list[float]]:
        with self._lock:
            return [list(row) for row in self.data_storage[start_index:]]

    def counter_snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "received_batches": int(self.received_batches),
                "dropped_batches": int(self.dropped_batches),
                "invalid_batches": int(self.invalid_batches),
                "missing_batches": int(self.missing_batches),
                "duplicate_batches": int(self.duplicate_batches),
                "sequence_resets": int(self.sequence_resets),
            }

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run_ble())
        except Exception as exc:
            self._failed_message = str(exc)
            self.status = self._failed_message
            click_safe_echo(f"iPhone IMU BLE error: {exc}")
        finally:
            self.running = False
            self.connected = False
            self._connected_event.set()

    async def _run_ble(self) -> None:
        BleakScanner, BleakClient = _load_bleak()
        device = await _find_iphone_device(BleakScanner, self.device, self.scan_timeout_s)
        if device is None:
            target = self.device or IPHONE_IMU_DEVICE_NAME
            raise RuntimeError(f"Could not find BLE device '{target}'. Open the iPhone app and start advertising.")

        self.status = f"connecting to {getattr(device, 'name', None) or device.address}"
        async with BleakClient(device) as client:
            self.connected = bool(client.is_connected)
            if not self.connected:
                raise RuntimeError("BLE connection failed.")
            self.status = "connected"
            self._connected_event.set()

            if self.autostart:
                await _write_control_if_available(client, b"START")
            await client.start_notify(IPHONE_IMU_DATA_CHARACTERISTIC_UUID, self._on_notification)
            try:
                while not self._stop_event.is_set() and client.is_connected:
                    await asyncio.sleep(0.10)
            finally:
                if self.autostart and client.is_connected:
                    await _write_control_if_available(client, b"STOP")
                try:
                    await client.stop_notify(IPHONE_IMU_DATA_CHARACTERISTIC_UUID)
                except Exception:
                    pass

    def _on_notification(self, _sender: Any, payload: bytearray) -> None:
        try:
            packet = parse_iphone_imu_packet(payload)
        except ValueError:
            with self._lock:
                self.invalid_batches += 1
                self.dropped_batches += 1
            return
        if not packet.samples:
            return

        with self._lock:
            previous_sequence = self.last_sequence
            if previous_sequence is not None:
                delta = (packet.sequence - previous_sequence) & 0xFFFF
                if delta == 0:
                    self.duplicate_batches += 1
                    return
                if delta < 0x8000:
                    missing = max(0, delta - 1)
                    self.missing_batches += missing
                    self.dropped_batches += missing
                else:
                    self.sequence_resets += 1

        host_now_ms = time.time() * 1000.0
        if self._host_time_offset_ms is None:
            self._host_time_offset_ms = host_now_ms - packet.samples[0].time_ms

        rows = [
            [
                float(self._host_time_offset_ms + sample.time_ms),
                sample.ax,
                sample.ay,
                sample.az,
                sample.gx,
                sample.gy,
                sample.gz,
            ]
            for sample in packet.samples
        ]
        with self._lock:
            self.data_storage.extend(rows)
            self.received_batches += 1
            self.last_sequence = packet.sequence


def _load_bleak() -> tuple[Any, Any]:
    try:
        from bleak import BleakClient, BleakScanner
    except ImportError as exc:
        raise RuntimeError("Missing BLE dependency. Run `uv sync` to install `bleak`.") from exc
    return BleakScanner, BleakClient


async def _find_iphone_device(BleakScanner: Any, selector: str | None, timeout_s: float) -> Any | None:
    service_uuid = IPHONE_IMU_SERVICE_UUID.lower()
    selector_norm = selector.lower().strip() if selector else ""

    def matches(device: Any, advertisement: Any) -> bool:
        name = device.name or getattr(advertisement, "local_name", "") or ""
        services = [uuid.lower() for uuid in (getattr(advertisement, "service_uuids", None) or [])]
        if selector_norm:
            return selector_norm in {device.address.lower(), name.lower()} or selector_norm in name.lower()
        return name == IPHONE_IMU_DEVICE_NAME or service_uuid in services

    return await BleakScanner.find_device_by_filter(matches, timeout=timeout_s)


async def _write_control_if_available(client: Any, payload: bytes) -> None:
    try:
        await client.write_gatt_char(IPHONE_IMU_CONTROL_CHARACTERISTIC_UUID, payload, response=False)
    except Exception as exc:
        command = payload.decode("ascii", errors="replace")
        click_safe_echo(f"iPhone IMU control command {command!r} failed: {exc}")
