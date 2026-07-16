from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

from .paths import IMU_PLOTS_DIR, RAW_IMU_DIR
from .serial_utils import list_serial_ports, ordered_ports

IMU_COLUMNS = ["Time_ms", "ax", "ay", "az", "gx", "gy", "gz"]
DEFAULT_IMU_BAUD = 115_200
LSM6DS3_CAPTURE_COLUMNS = [
    "Time_ms",
    "HostTime_ms",
    "DeviceTime_us",
    "SampleIndex",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
]


@dataclass(frozen=True)
class ImuAnalysisResult:
    csv_path: Path
    plot_path: Path | None
    sample_rate_hz: float
    respiratory_bpm: float
    displacement_bpm: float
    heart_bpm: float


def _measurement_time_ms(df: pd.DataFrame) -> pd.Series:
    """Prefer the device clock when a complete, monotonic LSM6DS3 axis is present."""

    if "DeviceTime_us" in df:
        device_us = pd.to_numeric(df["DeviceTime_us"], errors="coerce")
        if len(device_us) >= 2 and device_us.notna().all() and (device_us.diff().dropna() > 0).all():
            return device_us / 1000.0
    return pd.to_numeric(df["Time_ms"], errors="coerce")


def _sampling_rate(df: pd.DataFrame) -> float:
    if "Time_s" in df:
        time_s = df["Time_s"]
    else:
        time_ms = _measurement_time_ms(df)
        time_s = (time_ms - time_ms.iloc[0]) / 1000.0

    dt = time_s.diff().dropna()
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.empty:
        return 100.0
    return float(1.0 / dt.median())


def _pca_project(data: np.ndarray) -> tuple[np.ndarray, float]:
    cov = np.cov(data, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = int(np.argmax(evals))
    variance = float(evals[idx] / np.sum(evals)) if np.sum(evals) else 0.0
    return data.dot(evecs[:, idx]), variance


def analyze_imu_csv(
    csv_path: str | Path,
    output_dir: str | Path = IMU_PLOTS_DIR,
    save_plot: bool = True,
    show_plot: bool = False,
) -> ImuAnalysisResult:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    df = pd.read_csv(csv_path)
    missing = [column for column in IMU_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing IMU columns in {csv_path}: {', '.join(missing)}")
    if len(df) < 10:
        raise ValueError(f"Not enough samples in {csv_path}")

    measurement_time_ms = _measurement_time_ms(df)
    df["Time_s"] = (measurement_time_ms - measurement_time_ms.iloc[0]) / 1000.0
    fs = _sampling_rate(df)
    t = df["Time_s"].to_numpy()

    ax = df["ax"].to_numpy()
    ay = df["ay"].to_numpy()
    az = df["az"].to_numpy()
    gx = df["gx"].to_numpy()
    gy = df["gy"].to_numpy()
    gz = df["gz"].to_numpy()

    sos_resp = butter(4, [0.1, 0.6], btype="band", fs=fs, output="sos")
    a_filt = np.column_stack([sosfiltfilt(sos_resp, values) for values in (ax, ay, az)])
    resp_g, _ = _pca_project(a_filt)

    g_cf = np.zeros((len(df), 3))
    g_init = np.array([ax[0], ay[0], az[0]], dtype=float)
    norm = np.linalg.norm(g_init)
    g_cf[0] = g_init / norm if norm else np.array([0.0, 0.0, 1.0])
    dt = df["Time_s"].diff().fillna(df["Time_s"].diff().mean()).to_numpy()
    alpha = 0.98
    for i in range(1, len(df)):
        w = np.radians([gx[i], gy[i], gz[i]])
        g_old = g_cf[i - 1]
        cross = np.array(
            [
                w[1] * g_old[2] - w[2] * g_old[1],
                w[2] * g_old[0] - w[0] * g_old[2],
                w[0] * g_old[1] - w[1] * g_old[0],
            ]
        )
        g_pred = g_old - cross * dt[i]
        pred_norm = np.linalg.norm(g_pred)
        if pred_norm:
            g_pred /= pred_norm

        a_meas = np.array([ax[i], ay[i], az[i]], dtype=float)
        a_norm = np.linalg.norm(a_meas)
        a_meas = a_meas / a_norm if a_norm else g_pred
        g_cf_i = alpha * g_pred + (1 - alpha) * a_meas
        g_cf_i /= np.linalg.norm(g_cf_i)
        g_cf[i] = g_cf_i

    g_cf_filt = np.column_stack([sosfiltfilt(sos_resp, g_cf[:, idx]) for idx in range(3)])
    resp_angle_deg = _pca_project(g_cf_filt)[0] * (180.0 / np.pi)

    sos_heart = butter(4, [0.65, 4.0], btype="band", fs=fs, output="sos")
    heart_filt = np.column_stack([sosfiltfilt(sos_heart, values) for values in (ax, ay, az)])
    heart_g, _ = _pca_project(heart_filt)
    heart_fft = np.abs(np.fft.rfft(heart_g - np.mean(heart_g)))
    heart_freqs = np.fft.rfftfreq(len(df), d=1.0 / fs)

    valid_heart = (heart_freqs >= 0.65) & (heart_freqs <= 4.0)
    heart_bpm = float(heart_freqs[valid_heart][np.argmax(heart_fft[valid_heart])] * 60.0) if np.any(valid_heart) else 0.0

    vel = cumulative_trapezoid(resp_g, t, initial=0)
    disp = detrend(cumulative_trapezoid(vel, t, initial=0))
    min_peak_distance = max(1, int(1.5 * fs))
    peaks_angle, _ = find_peaks(resp_angle_deg, distance=min_peak_distance, prominence=0.0005)
    peaks_disp, _ = find_peaks(disp, distance=min_peak_distance)
    duration_minutes = max(float(t[-1] / 60.0), 1e-9)
    respiratory_bpm = float(len(peaks_angle) / duration_minutes)
    displacement_bpm = float(len(peaks_disp) / duration_minutes)

    plot_path: Path | None = None
    fig, axes = plt.subplots(5, 2, figsize=(15, 18))
    axes[0, 0].plot(t, ax, color="b")
    axes[0, 0].set_title("Accelerometer X (ax)")
    axes[0, 1].plot(t, gx, color="r")
    axes[0, 1].set_title("Gyroscope X (gx)")
    axes[1, 0].plot(t, ay, color="b")
    axes[1, 0].set_title("Accelerometer Y (ay)")
    axes[1, 1].plot(t, gy, color="r")
    axes[1, 1].set_title("Gyroscope Y (gy)")
    axes[2, 0].plot(t, az, color="b")
    axes[2, 0].set_title("Accelerometer Z (az)")
    axes[2, 1].plot(t, gz, color="r")
    axes[2, 1].set_title("Gyroscope Z (gz)")
    axes[3, 0].plot(t, disp, color="g")
    if len(peaks_disp):
        axes[3, 0].plot(t[peaks_disp], disp[peaks_disp], "x", color="black")
    axes[3, 0].set_title(f"Resp Displacement (PCA G) | {displacement_bpm:.1f} BPM")
    axes[3, 1].plot(t, resp_angle_deg, color="purple")
    if len(peaks_angle):
        axes[3, 1].plot(t[peaks_angle], resp_angle_deg[peaks_angle], "x", color="black")
    axes[3, 1].set_title(f"Resp Angle (CF PCA) | {respiratory_bpm:.1f} BPM")
    axes[4, 0].plot(t, heart_g, color="crimson")
    axes[4, 0].set_title("Cardiac Signal (PCA G)")
    axes[4, 1].plot(heart_freqs, heart_fft, color="orange")
    axes[4, 1].set_xlim(0, 5)
    axes[4, 1].set_title(f"Cardiac Spectrum (FFT) | {heart_bpm:.1f} BPM")

    for ax_sub in axes.flat:
        ax_sub.grid(True, alpha=0.3)

    fig.suptitle(f"IMU 6-Axis Analysis - {csv_path.name}", fontsize=16)
    fig.tight_layout()

    if save_plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{csv_path.stem}.png"
        fig.savefig(plot_path)
    if show_plot:
        plt.show()
    plt.close(fig)

    return ImuAnalysisResult(csv_path, plot_path, fs, respiratory_bpm, displacement_bpm, heart_bpm)


class BreathCapture:
    def __init__(self, baud: int = DEFAULT_IMU_BAUD, output_dir: str | Path = RAW_IMU_DIR):
        self.baud = baud
        self.output_dir = Path(output_dir)
        self.serial_port: serial.Serial | None = None
        self.running = False
        self.data_storage: list[list[float]] = []
        self.capture_storage: list[list[float]] = []
        self.read_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._device_to_host_offset_ms: float | None = None
        self._last_device_time_us: int | None = None
        self._stream_synced = False
        self.malformed_lines = 0

    def connect(self, port_name: str | None = None, *, exact_port: bool = False) -> bool:
        ports = list_serial_ports()
        if not ports:
            click_safe_echo("No serial ports found.")
            return False

        click_safe_echo(f"Available ports: {ports}")
        if exact_port and port_name:
            candidates = [port_name] if port_name in ports else []
        else:
            candidates = ordered_ports(ports, port_name)
        for port in candidates:
            try:
                click_safe_echo(f"Trying {port} (baud: {self.baud})...")
                self.serial_port = serial.Serial(port, self.baud, timeout=0.1)
                self.serial_port.reset_input_buffer()
                self.running = True
                with self._lock:
                    self.data_storage = []
                    self.capture_storage = []
                    self._device_to_host_offset_ms = None
                    self._last_device_time_us = None
                    self._stream_synced = False
                    self.malformed_lines = 0
                self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
                self.read_thread.start()
                click_safe_echo(f"Connected to {port}. Collecting data...")
                return True
            except (serial.SerialException, PermissionError) as exc:
                click_safe_echo(f"Could not connect to {port}: {exc}")
        return False

    def _read_loop(self) -> None:
        buffer = ""
        while self.running and self.serial_port:
            try:
                if self.serial_port.in_waiting > 0:
                    buffer += self.serial_port.read(self.serial_port.in_waiting).decode("utf-8", errors="ignore")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        self._process_line(line.strip())
                else:
                    time.sleep(0.0001)
            except Exception as exc:
                if self.running:
                    click_safe_echo(f"Serial read error: {exc}")
                self.running = False

    def _process_line(self, line: str, *, host_time_ms: float | None = None) -> None:
        """Parse the timestamped firmware format and the legacy six-value format."""

        values = line.split(",")
        arrival_ms = float(host_time_ms if host_time_ms is not None else time.time() * 1000.0)
        try:
            if len(values) == 8:
                sample_index = int(values[0])
                device_time_us = int(values[1])
                axes = [float(value) for value in values[2:]]
                if not (0 <= sample_index <= 0xFFFFFFFF and device_time_us >= 0):
                    raise ValueError
                if not np.all(np.isfinite(axes)):
                    raise ValueError
                with self._lock:
                    if self._device_to_host_offset_ms is None or (
                        self._last_device_time_us is not None and device_time_us <= self._last_device_time_us
                    ):
                        self._device_to_host_offset_ms = arrival_ms - device_time_us / 1000.0
                    aligned_time_ms = self._device_to_host_offset_ms + device_time_us / 1000.0
                    self._last_device_time_us = device_time_us
                    self._stream_synced = True
                    self.data_storage.append([aligned_time_ms, *axes])
                    self.capture_storage.append(
                        [aligned_time_ms, arrival_ms, float(device_time_us), float(sample_index), *axes]
                    )
                return
            if len(values) == 6:
                axes = [float(value) for value in values]
                if not np.all(np.isfinite(axes)):
                    raise ValueError
                with self._lock:
                    self._stream_synced = True
                    self.data_storage.append([arrival_ms, *axes])
                    self.capture_storage.append([arrival_ms, arrival_ms, float("nan"), float("nan"), *axes])
                return
        except ValueError:
            pass
        with self._lock:
            # Opening the USB-UART port can reset ESP32.  The bootloader may
            # emit a short line at its own baud rate before the application
            # switches to the configured console rate.  Do not report those
            # preamble bytes as an IMU transport error; after the first valid
            # record every malformed line remains visible in diagnostics.
            if self._stream_synced:
                self.malformed_lines += 1

    def stop(self) -> None:
        self.running = False
        if self.read_thread and self.read_thread.is_alive() and self.read_thread is not threading.current_thread():
            self.read_thread.join(timeout=1.0)
        serial_port, self.serial_port = self.serial_port, None
        if serial_port and serial_port.is_open:
            serial_port.close()

    def save(self) -> Path:
        rows = self.snapshot_capture_storage()
        if len(rows) < 10:
            raise ValueError("Not enough samples to save; at least 10 are required.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"respiratory_6axis_raw_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        pd.DataFrame(rows, columns=LSM6DS3_CAPTURE_COLUMNS).to_csv(path, index=False)
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

    def snapshot_capture_storage(self) -> list[list[float]]:
        with self._lock:
            return [list(row) for row in self.capture_storage]

    def snapshot_capture_since(self, start_index: int) -> list[list[float]]:
        with self._lock:
            return [list(row) for row in self.capture_storage[start_index:]]

    def diagnostics(self, start_index: int = 0) -> dict[str, float | int | str | None]:
        diagnostics = summarize_lsm6ds3_capture_rows(self.snapshot_capture_since(start_index))
        diagnostics["malformed_lines"] = int(self.malformed_lines)
        return diagnostics


def summarize_lsm6ds3_capture_rows(rows: list[list[float]]) -> dict[str, float | int | str | None]:
    """Summarize timing and exact UART losses from firmware sample counters."""

    if not rows:
        return {
            "rows": 0,
            "format": "empty",
            "new_format_rows": 0,
            "legacy_rows": 0,
            "device_sample_rate_hz": None,
            "host_arrival_sample_rate_hz": None,
            "missing_samples": 0,
            "missing_percent": 0.0,
            "duplicate_samples": 0,
            "counter_resets": 0,
            "device_time_regressions": 0,
        }

    df = pd.DataFrame(rows, columns=LSM6DS3_CAPTURE_COLUMNS)
    sample_index = pd.to_numeric(df["SampleIndex"], errors="coerce").to_numpy(dtype=float)
    device_time_us = pd.to_numeric(df["DeviceTime_us"], errors="coerce").to_numpy(dtype=float)
    host_time_ms = pd.to_numeric(df["HostTime_ms"], errors="coerce").to_numpy(dtype=float)
    new_mask = np.isfinite(sample_index) & np.isfinite(device_time_us)
    new_indices = sample_index[new_mask].astype(np.uint64)
    new_device_time = device_time_us[new_mask]

    missing_samples = 0
    duplicate_samples = 0
    counter_resets = 0
    for previous, current in zip(new_indices[:-1], new_indices[1:]):
        delta = (int(current) - int(previous)) & 0xFFFFFFFF
        if delta == 0:
            duplicate_samples += 1
        elif delta < 0x80000000:
            missing_samples += max(0, delta - 1)
        else:
            counter_resets += 1

    device_dt = np.diff(new_device_time)
    device_time_regressions = int(np.sum(device_dt <= 0))
    valid_device_dt = device_dt[np.isfinite(device_dt) & (device_dt > 0)]
    host_dt = np.diff(host_time_ms)
    valid_host_dt = host_dt[np.isfinite(host_dt) & (host_dt > 0)]
    new_rows = int(np.sum(new_mask))
    legacy_rows = int(len(df) - new_rows)
    expected_rows = new_rows + missing_samples

    if new_rows and not legacy_rows:
        stream_format = "timestamped"
    elif legacy_rows and not new_rows:
        stream_format = "legacy"
    else:
        stream_format = "mixed"
    return {
        "rows": int(len(df)),
        "format": stream_format,
        "new_format_rows": new_rows,
        "legacy_rows": legacy_rows,
        "device_sample_rate_hz": float(1_000_000.0 / np.median(valid_device_dt)) if len(valid_device_dt) else None,
        "host_arrival_sample_rate_hz": float(1000.0 / np.median(valid_host_dt)) if len(valid_host_dt) else None,
        "missing_samples": int(missing_samples),
        "missing_percent": float(100.0 * missing_samples / expected_rows) if expected_rows else 0.0,
        "duplicate_samples": int(duplicate_samples),
        "counter_resets": int(counter_resets),
        "device_time_regressions": device_time_regressions,
    }


def click_safe_echo(message: str) -> None:
    try:
        import click

        click.echo(message)
    except Exception:
        print(message)
