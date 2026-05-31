from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import serial.tools.list_ports

from .paths import RAW_A121_DIR

A121_COLUMNS = [
    "Timestamp_ms",
    "Frame",
    "PeakDistance_m",
    "PeakAmplitude",
    "PeakPhase_rad",
    "MeanAmplitude",
    "Distances_m",
    "Amplitude",
    "Phase",
    "Real",
    "Imag",
]

APPROX_BASE_STEP_LENGTH_M = 2.5e-3


@dataclass(frozen=True)
class A121Config:
    start_m: float = 0.2
    end_m: float = 1.5
    # Defaults follow Acconeer's breathing reference app; 20 Hz is still enough for the
    # configured heart band while keeping UART/GUI latency manageable.
    profile: int = 3
    hwaas: int = 32
    # A121 session buffer is limited; for long ranges this is auto-clamped in connect().
    sweeps_per_frame: int = 16
    frame_rate_hz: float = 20.0
    step_length: int = 1


def find_a121_serial_ports() -> list[str]:
    """Return likely Waveshare/Acconeer A121 UART ports, preferring Interface A."""
    ports = list(serial.tools.list_ports.comports())

    def score(port: Any) -> tuple[int, str]:
        text = " ".join(
            str(getattr(port, attr, "") or "")
            for attr in ("device", "description", "manufacturer", "product", "interface", "hwid")
        ).lower()
        value = 0
        if "ch342" in text or "wch" in text or "usb-enhanced" in text:
            value += 20
        if "acconeer" in text or "a121" in text or "waveshare" in text:
            value += 20
        if "interface a" in text or "serial-a" in text or "-a" in text:
            value += 10
        if "interface b" in text or "serial-b" in text or "-b" in text:
            value -= 10
        return (-value, str(port.device))

    likely = [port for port in ports if score(port)[0] < 0]
    return [port.device for port in sorted(likely, key=score)]


def _json_array(values: np.ndarray) -> str:
    return json.dumps(np.asarray(values, dtype=float).round(8).tolist(), separators=(",", ":"))


def parse_json_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if not isinstance(value, str) or not value:
        return np.asarray([], dtype=float)
    return np.asarray(json.loads(value), dtype=float)


class A121Capture:
    """Live Sparse IQ capture for an Acconeer A121 over UART."""

    def __init__(
        self,
        output_dir: str | Path = RAW_A121_DIR,
        config: A121Config | None = None,
        override_baudrate: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config or A121Config()
        self.override_baudrate = override_baudrate
        self.client: Any | None = None
        self.metadata: Any | None = None
        self.sensor_config: Any | None = None
        self.distances_m: np.ndarray = np.asarray([], dtype=float)
        self.acconeer_breathing_processor: Any | None = None
        self.acconeer_app_state: str = "INIT_STATE"
        self.acconeer_presence_detected: bool = False
        self.acconeer_presence_distance_m: float | None = None
        self.acconeer_selected_distance_m: float | None = None
        self.acconeer_range_indices: tuple[int, int] | None = None
        self.acconeer_breathing_rate_bpm: float | None = None
        self.running = False
        self.data_storage: list[list[Any]] = []
        self.data_lock = threading.Lock()
        # Live rows keep NumPy arrays instead of JSON strings so the UI can plot/analyze
        # without repeatedly decoding large arrays. data_storage remains CSV/SQLite-ready.
        # Enough for a long live window even when users raise FPS. These rows hold NumPy
        # arrays for fast live plotting, so avoid making this unbounded.
        self.live_buffer: deque[list[Any]] = deque(maxlen=6000)
        self.read_thread: threading.Thread | None = None
        self.frame_index = 0
        self.session_start_wall_ms: float = 0.0
        self.session_start_tick_s: float | None = None

    def connect(self, port_name: str | None = None) -> bool:
        try:
            from acconeer.exptool import a121
            from acconeer.exptool.a121.algo._utils import get_distances_m
        except Exception as exc:
            safe_echo(f"acconeer-exptool is not installed or failed to import: {exc}")
            return False

        candidates = find_a121_serial_ports()
        port = port_name or (candidates[0] if candidates else None)
        if port is None:
            safe_echo("No A121 serial port found. Select the CH342 Interface A port manually.")
            return False

        try:
            safe_echo(f"Connecting to Acconeer A121 on {port}...")
            open_kwargs: dict[str, Any] = {"serial_port": port}
            if self.override_baudrate:
                open_kwargs["override_baudrate"] = self.override_baudrate
            self.client = a121.Client.open(**open_kwargs)

            profile = getattr(a121.Profile, f"PROFILE_{self.config.profile}")
            start_point = max(0, int(round(self.config.start_m / APPROX_BASE_STEP_LENGTH_M)))
            span_m = max(self.config.end_m - self.config.start_m, APPROX_BASE_STEP_LENGTH_M)

            # Prevent serial bandwidth saturation by calculating maximum total samples per frame
            # that can be sustained at the requested frame rate.
            baudrate = self.override_baudrate or 921600
            requested_fps = max(1.0, float(self.config.frame_rate_hz))
            requested_sweeps = max(1, int(self.config.sweeps_per_frame))

            # Max total samples/frame = baudrate / (48 * frame_rate) to leave safety margin.
            # Enforce the hardware sensor limit of 4095 samples as well.
            max_total_samples = min(int(baudrate / (48 * requested_fps)), 4095)

            # Choose step_length to fit the points and sweeps within max_total_samples
            target_max_points = max(2, max_total_samples // requested_sweeps)
            step_length = max(1, int(self.config.step_length))
            required_step = int(math.ceil(span_m / (APPROX_BASE_STEP_LENGTH_M * (target_max_points - 1))))
            if required_step > step_length:
                step_length = required_step

            # Acconeer A121 requires step_length to be a divisor or multiple of 24.
            # Valid step lengths are: 1, 2, 3, 4, 6, 8, 12, 24, and any multiple of 24.
            # We round up to the next valid step length.
            valid_divisors = [1, 2, 3, 4, 6, 8, 12, 24]
            if step_length <= 24:
                step_length = next(v for v in valid_divisors if v >= step_length)
            else:
                step_length = int(math.ceil(step_length / 24.0) * 24)

            if step_length != int(self.config.step_length):
                safe_echo(f"A121 step_length adjusted to {step_length} to prevent serial bandwidth saturation and satisfy RSS constraints.")

            num_points = max(2, int(math.ceil(span_m / (APPROX_BASE_STEP_LENGTH_M * step_length))) + 1)
            max_sweeps = max(1, max_total_samples // num_points)
            sweeps_per_frame = max(1, min(requested_sweeps, max_sweeps))
            if sweeps_per_frame != requested_sweeps:
                safe_echo(
                    f"A121 sweeps/frame reduced from {requested_sweeps} to {sweeps_per_frame} "
                    f"for {num_points} points to fit serial/sensor buffer constraints."
                )

            self.sensor_config = a121.SensorConfig(
                start_point=start_point,
                num_points=num_points,
                step_length=step_length,
                profile=profile,
                hwaas=self.config.hwaas,
                sweeps_per_frame=sweeps_per_frame,
                frame_rate=self.config.frame_rate_hz,
            )
            self.metadata = self.client.setup_session(self.sensor_config)
            self.distances_m = get_distances_m(self.sensor_config, self.metadata)
            try:
                from acconeer.exptool.a121.algo.breathing._processor import (
                    BreathingProcessorConfig as AcconeerBreathingConfig,
                    Processor as AcconeerBreathingProcessor,
                    ProcessorConfig as AcconeerBreathingProcessorConfig,
                    get_presence_config as get_acconeer_presence_config,
                )

                # This Exploration Tool release uses a mutable processor config whose fields are
                # filled by RefApp after construction, so mirror that instead of passing kwargs.
                processor_config = AcconeerBreathingProcessorConfig()
                processor_config.use_presence_processor = True
                processor_config.num_distances_to_analyze = 3
                processor_config.distance_determination_duration = 5.0
                processor_config.presence_config = get_acconeer_presence_config()
                processor_config.breathing_config = AcconeerBreathingConfig()
                self.acconeer_breathing_processor = AcconeerBreathingProcessor(
                    sensor_config=self.sensor_config,
                    metadata=self.metadata,
                    processor_config=processor_config,
                )
            except Exception as exc:
                safe_echo(f"Acconeer breathing reference state machine unavailable: {exc}")
                self.acconeer_breathing_processor = None
            self.client.start_session()
            self.running = True
            with self.data_lock:
                self.data_storage = []
                self.live_buffer.clear()
                self.frame_index = 0
                self.session_start_tick_s = None
                self._reset_acconeer_selection_unlocked()
            self.session_start_wall_ms = time.time() * 1000.0
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            safe_echo(
                f"A121 session started: {self.distances_m[0]:.3f}-{self.distances_m[-1]:.3f} m, "
                f"{len(self.distances_m)} points, profile {self.config.profile}, HWAAS {self.config.hwaas}."
            )
            return True
        except Exception as exc:
            safe_echo(f"A121 connection/setup failed: {exc}")
            self.stop()
            return False

    def _read_loop(self) -> None:
        while self.running and self.client is not None:
            try:
                result = self.client.get_next()
                self._process_result(result)
            except Exception as exc:
                if self.running:
                    safe_echo(f"A121 read failed: {exc}")
                self.running = False

    def _process_result(self, result: Any) -> None:
        self._update_acconeer_selection(result)
        frame = result.subframes[0] if hasattr(result, "subframes") and result.subframes else result.frame
        frame_np = np.asarray(frame)
        if frame_np.ndim == 0:
            return
        profile = np.mean(frame_np.reshape(-1, frame_np.shape[-1]), axis=0)
        amplitude = np.abs(profile)
        phase = np.angle(profile)
        real = np.real(profile)
        imag = np.imag(profile)
        if len(amplitude) == 0:
            return
        peak_idx = int(np.argmax(amplitude))
        distances = self.distances_m
        if len(distances) != len(amplitude):
            distances = np.arange(len(amplitude), dtype=float)
        with self.data_lock:
            frame_index = self.frame_index
            self.frame_index += 1
        tick_time_s = getattr(result, "tick_time", None)
        if tick_time_s is not None and np.isfinite(float(tick_time_s)):
            tick_time_s = float(tick_time_s)
            with self.data_lock:
                if self.session_start_tick_s is None:
                    self.session_start_tick_s = tick_time_s
                start_tick_s = self.session_start_tick_s
            timestamp_ms = self.session_start_wall_ms + (tick_time_s - start_tick_s) * 1000.0
        else:
            frame_rate = max(float(self.config.frame_rate_hz), 1.0)
            timestamp_ms = self.session_start_wall_ms + frame_index * 1000.0 / frame_rate
        peak_distance = float(distances[peak_idx])
        peak_amplitude = float(amplitude[peak_idx])
        peak_phase = float(phase[peak_idx])
        mean_amplitude = float(np.mean(amplitude))
        live_row: list[Any] = [
            timestamp_ms,
            frame_index,
            peak_distance,
            peak_amplitude,
            peak_phase,
            mean_amplitude,
            distances.copy(),
            amplitude.copy(),
            phase.copy(),
            real.copy(),
            imag.copy(),
        ]
        storage_row: list[Any] = [
            timestamp_ms,
            frame_index,
            peak_distance,
            peak_amplitude,
            peak_phase,
            mean_amplitude,
            _json_array(distances),
            _json_array(amplitude),
            _json_array(phase),
            _json_array(real),
            _json_array(imag),
        ]
        with self.data_lock:
            self.data_storage.append(storage_row)
            self.live_buffer.append(live_row)

    def _reset_acconeer_selection_unlocked(self) -> None:
        self.acconeer_app_state = "INIT_STATE"
        self.acconeer_presence_detected = False
        self.acconeer_presence_distance_m = None
        self.acconeer_selected_distance_m = None
        self.acconeer_range_indices = None
        self.acconeer_breathing_rate_bpm = None

    def _update_acconeer_selection(self, result: Any) -> None:
        """Feed Acconeer's breathing reference state machine with the raw A121 frame.

        The CSV/live rows intentionally store only the mean complex sweep. Presence-distance
        selection in Acconeer's reference app depends on the full sweeps inside each frame, so it
        must run here while the original ``a121.Result`` is still available.
        """
        processor = self.acconeer_breathing_processor
        if processor is None or len(self.distances_m) == 0:
            return
        try:
            ref_result = processor.process(result)
            app_state = getattr(
                getattr(ref_result, "app_state", None),
                "name",
                str(getattr(ref_result, "app_state", "")),
            )
            presence_result = getattr(ref_result, "presence_result", None)
            presence_detected = bool(getattr(presence_result, "presence_detected", False))
            presence_distance = getattr(presence_result, "presence_distance", None)
            if presence_distance is not None and np.isfinite(float(presence_distance)) and float(presence_distance) > 0:
                presence_distance_m: float | None = float(presence_distance)
            else:
                presence_distance_m = None

            selected_distance_m = None
            range_indices = getattr(ref_result, "distances_being_analyzed", None)
            if range_indices is not None:
                start_idx, end_idx = int(range_indices[0]), int(range_indices[1])
                start_idx = int(np.clip(start_idx, 0, max(len(self.distances_m) - 1, 0)))
                end_idx = int(np.clip(end_idx, start_idx + 1, len(self.distances_m)))
                center_idx = int(np.clip((start_idx + end_idx - 1) // 2, 0, len(self.distances_m) - 1))
                selected_distance_m = float(self.distances_m[center_idx])
                stored_range_indices: tuple[int, int] | None = (start_idx, end_idx)
            else:
                stored_range_indices = None
                if presence_distance_m is not None:
                    center_idx = int(np.argmin(np.abs(self.distances_m - presence_distance_m)))
                    selected_distance_m = float(self.distances_m[center_idx])

            breathing_result = getattr(ref_result, "breathing_result", None)
            breathing_rate = getattr(breathing_result, "breathing_rate", None) if breathing_result is not None else None
            breathing_rate_bpm = (
                float(breathing_rate)
                if breathing_rate is not None and np.isfinite(float(breathing_rate))
                else None
            )

            with self.data_lock:
                self.acconeer_app_state = app_state
                self.acconeer_presence_detected = presence_detected
                self.acconeer_presence_distance_m = presence_distance_m
                self.acconeer_selected_distance_m = selected_distance_m
                self.acconeer_range_indices = stored_range_indices
                self.acconeer_breathing_rate_bpm = breathing_rate_bpm
        except Exception:
            # Do not let optional reference-app state tracking stop raw capture.
            return

    def snapshot_acconeer_selection(self) -> dict[str, Any]:
        with self.data_lock:
            range_indices = self.acconeer_range_indices
            range_m = None
            if range_indices is not None and len(self.distances_m):
                start_idx, end_idx = range_indices
                start_idx = int(np.clip(start_idx, 0, len(self.distances_m) - 1))
                last_idx = int(np.clip(end_idx - 1, start_idx, len(self.distances_m) - 1))
                range_m = (float(self.distances_m[start_idx]), float(self.distances_m[last_idx]))
            return {
                "app_state": self.acconeer_app_state,
                "presence_detected": self.acconeer_presence_detected,
                "presence_distance_m": self.acconeer_presence_distance_m,
                "target_distance_m": self.acconeer_selected_distance_m,
                "range_indices": range_indices,
                "range_m": range_m,
                "breathing_rate_bpm": self.acconeer_breathing_rate_bpm,
            }

    def snapshot_data_storage(self) -> list[list[Any]]:
        with self.data_lock:
            return list(self.data_storage)

    def snapshot_data_since(self, index: int) -> list[list[Any]]:
        with self.data_lock:
            return list(self.data_storage[index:])

    def snapshot_live_buffer(self) -> list[list[Any]]:
        with self.data_lock:
            return list(self.live_buffer)

    def data_count(self) -> int:
        with self.data_lock:
            return len(self.data_storage)

    def stop(self) -> None:
        self.running = False
        if self.client is not None:
            try:
                if getattr(self.client, "session_is_started", False):
                    self.client.stop_session()
            except Exception:
                pass
            try:
                self.client.close()
            except Exception:
                pass
        if (
            self.read_thread is not None
            and self.read_thread.is_alive()
            and threading.current_thread() is not self.read_thread
        ):
            self.read_thread.join(timeout=1.0)
        self.client = None
        self.acconeer_breathing_processor = None
        with self.data_lock:
            self._reset_acconeer_selection_unlocked()

    def save(self) -> Path:
        rows = self.snapshot_data_storage()
        if len(rows) < 1:
            raise ValueError("No A121 data to save.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"a121_sparse_iq_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        pd.DataFrame(rows, columns=A121_COLUMNS).to_csv(path, index=False)
        return path


def safe_echo(message: str) -> None:
    try:
        import click

        click.echo(message)
    except Exception:
        print(message)
