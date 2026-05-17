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


@dataclass(frozen=True)
class ImuAnalysisResult:
    csv_path: Path
    plot_path: Path | None
    sample_rate_hz: float
    respiratory_bpm: float
    displacement_bpm: float
    heart_bpm: float


def _sampling_rate(df: pd.DataFrame) -> float:
    if "Time_s" not in df:
        df["Time_s"] = (df["Time_ms"] - df["Time_ms"].iloc[0]) / 1000.0

    dt = df["Time_s"].diff().dropna()
    if dt.empty or dt.mean() <= 0:
        return 100.0
    return float(1.0 / dt.mean())


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

    df["Time_s"] = (df["Time_ms"] - df["Time_ms"].iloc[0]) / 1000.0
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
    def __init__(self, baud: int = 921600, output_dir: str | Path = RAW_IMU_DIR):
        self.baud = baud
        self.output_dir = Path(output_dir)
        self.serial_port: serial.Serial | None = None
        self.running = False
        self.data_storage: list[list[float]] = []
        self.read_thread: threading.Thread | None = None

    def connect(self, port_name: str | None = None) -> bool:
        ports = list_serial_ports()
        if not ports:
            click_safe_echo("Nie znaleziono portow szeregowych.")
            return False

        click_safe_echo(f"Dostepne porty: {ports}")
        for port in ordered_ports(ports, port_name):
            try:
                click_safe_echo(f"Proba polaczenia z {port} (baud: {self.baud})...")
                self.serial_port = serial.Serial(port, self.baud, timeout=0.1)
                self.running = True
                self.data_storage = []
                self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
                self.read_thread.start()
                click_safe_echo(f"Polaczono z {port}. Zbieranie danych...")
                return True
            except (serial.SerialException, PermissionError) as exc:
                click_safe_echo(f"Blad polaczenia z {port}: {exc}")
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
                click_safe_echo(f"Blad w petli odczytu: {exc}")
                self.running = False

    def _process_line(self, line: str) -> None:
        try:
            parts = [float(value) for value in line.split(",")]
        except ValueError:
            return
        if len(parts) == 6:
            self.data_storage.append([time.time() * 1000.0] + parts)

    def stop(self) -> None:
        self.running = False
        if self.serial_port:
            self.serial_port.close()

    def save(self) -> Path:
        if len(self.data_storage) < 10:
            raise ValueError("Za malo danych do zapisu.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"respiratory_6axis_raw_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        pd.DataFrame(self.data_storage, columns=IMU_COLUMNS).to_csv(path, index=False)
        return path


def click_safe_echo(message: str) -> None:
    try:
        import click

        click.echo(message)
    except Exception:
        print(message)

