from __future__ import annotations

from pathlib import Path
import time

import click
import matplotlib.pyplot as plt
import numpy as np

from .imu import BreathCapture, analyze_imu_csv
from .paths import IMU_PLOTS_DIR, RADAR_PLOTS_DIR, RAW_IMU_DIR, RAW_RADAR_DIR
from .radar import RadarCapture, analyze_radar_csv
from .serial_utils import list_serial_ports


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Capture and analyze respiratory IMU/radar recordings."""


@cli.command("ports")
def ports() -> None:
    """List available serial ports."""
    available_ports = list_serial_ports()
    if not available_ports:
        click.echo("No serial ports found.")
        return
    for port in available_ports:
        click.echo(port)


@cli.command("plot-imu")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=IMU_PLOTS_DIR, show_default=True)
@click.option("--show", is_flag=True, help="Open the plot window after generating it.")
def plot_imu(csv_path: Path, output_dir: Path, show: bool) -> None:
    """Generate an IMU analysis chart from a CSV file."""
    result = analyze_imu_csv(csv_path, output_dir=output_dir, show_plot=show)
    click.echo(f"Saved: {result.plot_path}")
    click.echo(
        f"fs={result.sample_rate_hz:.1f} Hz, "
        f"resp={result.respiratory_bpm:.1f} BPM, "
        f"disp={result.displacement_bpm:.1f} BPM, "
        f"heart={result.heart_bpm:.1f} BPM"
    )


@cli.command("plot-radar")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=RADAR_PLOTS_DIR, show_default=True)
@click.option("--show", is_flag=True, help="Open the plot window after generating it.")
def plot_radar(csv_path: Path, output_dir: Path, show: bool) -> None:
    """Generate a radar analysis chart from a CSV file."""
    result = analyze_radar_csv(csv_path, output_dir=output_dir, show_plot=show)
    click.echo(f"Saved: {result.plot_path}")
    click.echo(
        f"fs={result.sample_rate_hz:.1f} Hz, "
        f"peak={result.peak_frequency_hz:.2f} Hz, "
        f"speed={result.peak_speed_mps:.3f} m/s"
    )


@cli.command("batch-imu")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=RAW_IMU_DIR)
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=IMU_PLOTS_DIR, show_default=True)
def batch_imu(input_dir: Path, output_dir: Path) -> None:
    """Generate IMU charts for every respiratory CSV in a directory."""
    for csv_path in sorted(input_dir.glob("respiratory_*_raw_*.csv")):
        try:
            result = analyze_imu_csv(csv_path, output_dir=output_dir)
            click.echo(f"{csv_path.name} -> {result.plot_path}")
        except Exception as exc:
            click.echo(f"{csv_path.name}: {exc}", err=True)


@cli.command("batch-radar")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=RAW_RADAR_DIR)
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=RADAR_PLOTS_DIR, show_default=True)
def batch_radar(input_dir: Path, output_dir: Path) -> None:
    """Generate radar charts for every radar CSV in a directory."""
    for csv_path in sorted(input_dir.glob("radar_raw_*.csv")):
        try:
            result = analyze_radar_csv(csv_path, output_dir=output_dir)
            click.echo(f"{csv_path.name} -> {result.plot_path}")
        except Exception as exc:
            click.echo(f"{csv_path.name}: {exc}", err=True)


@cli.command("capture-imu")
@click.option("-p", "--port", help="Preferred serial port, for example COM6.")
@click.option("-b", "--baud", default=921600, show_default=True)
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=RAW_IMU_DIR, show_default=True)
@click.option("--plot/--no-plot", default=True, show_default=True)
def capture_imu(port: str | None, baud: int, output_dir: Path, plot: bool) -> None:
    """Capture IMU samples until Ctrl+C, save CSV, and optionally chart it."""
    capture = BreathCapture(baud=baud, output_dir=output_dir)
    if not capture.connect(port):
        raise click.ClickException("Could not connect to an IMU serial port.")
    click.echo("Capturing IMU data. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
    csv_path = capture.save()
    click.echo(f"Saved: {csv_path}")
    if plot:
        result = analyze_imu_csv(csv_path)
        click.echo(f"Plot: {result.plot_path}")


@cli.command("capture-radar")
@click.option("-p", "--port", help="Preferred serial port, for example COM6.")
@click.option("-b", "--baud", default=921600, show_default=True)
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=RAW_RADAR_DIR, show_default=True)
@click.option("--plot/--no-plot", default=True, show_default=True)
def capture_radar(port: str | None, baud: int, output_dir: Path, plot: bool) -> None:
    """Capture radar samples until Ctrl+C, save CSV, and optionally chart it."""
    capture = RadarCapture(baud=baud, output_dir=output_dir)
    if not capture.connect(port):
        raise click.ClickException("Could not connect to a radar serial port.")
    click.echo("Capturing radar data. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
    csv_path = capture.save()
    click.echo(f"Saved: {csv_path}")
    if plot:
        result = analyze_radar_csv(csv_path)
        click.echo(f"Plot: {result.plot_path}")


@cli.command("live-radar")
@click.option("-p", "--port", default="COM6", show_default=True, help="Preferred serial port.")
@click.option("-b", "--baud", default=921600, show_default=True)
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=RAW_RADAR_DIR, show_default=True)
def live_radar(port: str, baud: int, output_dir: Path) -> None:
    """Open a live radar plot, then save and chart the recording when closed."""
    capture = RadarCapture(baud=baud, output_dir=output_dir)
    if not capture.connect(port):
        raise click.ClickException("Could not connect to a radar serial port.")

    plt.style.use("dark_background")
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(12, 8))
    fig.canvas.manager.set_window_title("Radar Viewer")
    line_time, = ax_time.plot([], [], color="#00ffcc", linewidth=1, label="Raw ADC")
    line_freq, = ax_freq.plot([], [], color="#ff3399", linewidth=1.5, label="Spectrum")
    info_text = ax_time.text(0.02, 0.95, "", transform=ax_time.transAxes, va="top", color="white")
    ax_time.set_title("Live Radar Signal (Voltage)")
    ax_time.set_ylabel("mV")
    ax_time.grid(True, color="#333333", alpha=0.5)
    ax_freq.set_title("Live Frequency Spectrum (FFT)")
    ax_freq.set_xlabel("Frequency [Hz]")
    ax_freq.set_ylabel("Magnitude")
    ax_freq.grid(True, color="#333333", alpha=0.5)

    running = [True]

    def stop(_event: object) -> None:
        running[0] = False
        capture.running = False

    fig.canvas.mpl_connect("close_event", stop)
    fig.canvas.mpl_connect("key_press_event", stop)
    last_fs = 500.0
    try:
        while running[0] and capture.running:
            if len(capture.live_buffer) > 64:
                data = list(capture.live_buffer)
                voltages = np.array([row[2] for row in data])
                timestamps = np.array([row[0] for row in data])
                dt_avg = np.mean(np.diff(timestamps)) / 1000.0 if len(timestamps) > 1 else 0
                if dt_avg > 0:
                    last_fs = 1.0 / dt_avg

                display_len = min(len(voltages), 1000)
                v_disp = voltages[-display_len:]
                line_time.set_data(range(display_len), v_disp)
                ax_time.set_xlim(0, display_len)
                margin = max((np.max(v_disp) - np.min(v_disp)) * 0.1, 50)
                ax_time.set_ylim(np.min(v_disp) - margin, np.max(v_disp) + margin)

                v_fft = voltages[-1024:] if len(voltages) >= 1024 else voltages
                v_win = (v_fft - np.mean(v_fft)) * np.hanning(len(v_fft))
                fft_vals = np.abs(np.fft.rfft(v_win)) / (len(v_fft) / 2)
                fft_freqs = np.fft.rfftfreq(len(v_fft), d=1.0 / last_fs)
                line_freq.set_data(fft_freqs, fft_vals)
                ax_freq.set_xlim(0, min(last_fs / 2, 250))
                ax_freq.set_ylim(0, np.max(fft_vals[1:]) * 1.2 + 0.1)
                if len(fft_vals) > 5:
                    peak_idx = int(np.argmax(fft_vals[5:]) + 5)
                    peak_freq = fft_freqs[peak_idx]
                    info_text.set_text(f"Peak: {peak_freq:.1f} Hz | Speed: {peak_freq / 70.16:.2f} m/s")

            plt.pause(0.01)
            if not plt.fignum_exists(fig.number):
                running[0] = False
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        plt.close("all")

    csv_path = capture.save()
    click.echo(f"Saved: {csv_path}")
    result = analyze_radar_csv(csv_path)
    click.echo(f"Plot: {result.plot_path}")


if __name__ == "__main__":
    cli()
