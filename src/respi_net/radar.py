from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from scipy.signal import welch

from .paths import RADAR_PLOTS_DIR, RAW_RADAR_DIR
from .serial_utils import list_serial_ports, ordered_ports

RADAR_COLUMNS = ["Timestamp_ms", "RawADC", "Voltage_mV"]
DOPPLER_HZ_PER_MPS = 70.16


@dataclass(frozen=True)
class RadarAnalysisResult:
    csv_path: Path
    plot_path: Path | None
    sample_rate_hz: float
    peak_frequency_hz: float
    peak_speed_mps: float


def analyze_radar_csv(
    csv_path: str | Path,
    output_dir: str | Path = RADAR_PLOTS_DIR,
    save_plot: bool = True,
    show_plot: bool = False,
) -> RadarAnalysisResult:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    df = pd.read_csv(csv_path)
    missing = [column for column in RADAR_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing radar columns in {csv_path}: {', '.join(missing)}")
    if len(df) < 10:
        raise ValueError(f"Not enough samples in {csv_path}")

    df = df.sort_values("Timestamp_ms").reset_index(drop=True)
    df["Time_s"] = (df["Timestamp_ms"] - df["Timestamp_ms"].min()) / 1000.0
    time_diffs = df["Time_s"].diff().dropna()
    fs = float(1.0 / time_diffs.mean()) if not time_diffs.empty and time_diffs.mean() > 0 else 500.0

    voltage = df["Voltage_mV"].to_numpy()
    voltage_detrended = voltage - np.mean(voltage)
    freqs, psd = welch(voltage_detrended, fs, nperseg=min(len(voltage), 4096))

    peak_frequency = 0.0
    if len(psd) > 1:
        peak_idx = int(np.argmax(psd[1:]) + 1)
        peak_frequency = float(freqs[peak_idx])

    plot_path: Path | None = None
    fig, (ax_time, ax_psd) = plt.subplots(2, 1, figsize=(15, 10))
    ax_time.plot(df["Time_s"], df["Voltage_mV"], color="#1f77b4", linewidth=0.5)
    ax_time.set_title(f"HB100 Radar - Time Domain Signal ({csv_path.name})", fontsize=14)
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Voltage [mV]")
    ax_time.grid(True, alpha=0.3)

    ax_psd.semilogy(freqs, psd, color="#ff7f0e", linewidth=1)
    ax_psd.set_title("HB100 Radar - Power Spectral Density (Welch)", fontsize=14)
    ax_psd.set_xlabel("Frequency [Hz]")
    ax_psd.set_ylabel("Power/Frequency [V^2/Hz]")
    ax_psd.grid(True, alpha=0.3, which="both")

    ax_speed = ax_psd.twiny()
    ax_speed.set_xlim(ax_psd.get_xlim())
    xticks = ax_psd.get_xticks()
    ax_speed.set_xticks(xticks)
    ax_speed.set_xticklabels([f"{tick / DOPPLER_HZ_PER_MPS:.1f}" for tick in xticks])
    ax_speed.set_xlabel("Estimated Speed [m/s] (Doppler shift)")
    fig.tight_layout()

    if save_plot:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{csv_path.stem}.png"
        fig.savefig(plot_path)
    if show_plot:
        plt.show()
    plt.close(fig)

    return RadarAnalysisResult(
        csv_path=csv_path,
        plot_path=plot_path,
        sample_rate_hz=fs,
        peak_frequency_hz=peak_frequency,
        peak_speed_mps=peak_frequency / DOPPLER_HZ_PER_MPS,
    )


class RadarCapture:
    def __init__(self, baud: int = 921600, output_dir: str | Path = RAW_RADAR_DIR):
        self.baud = baud
        self.output_dir = Path(output_dir)
        self.serial_port: serial.Serial | None = None
        self.running = False
        self.data_storage: list[list[float]] = []
        self.live_buffer: deque[list[float]] = deque(maxlen=2048)
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
                self.live_buffer.clear()
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
        if len(parts) == 3:
            self.data_storage.append(parts)
            self.live_buffer.append(parts)

    def stop(self) -> None:
        self.running = False
        if self.serial_port:
            self.serial_port.close()

    def save(self) -> Path:
        if len(self.data_storage) < 10:
            raise ValueError("Za malo danych do zapisu.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"radar_raw_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        pd.DataFrame(self.data_storage, columns=RADAR_COLUMNS).to_csv(path, index=False)
        return path


def click_safe_echo(message: str) -> None:
    try:
        import click

        click.echo(message)
    except Exception:
        print(message)

