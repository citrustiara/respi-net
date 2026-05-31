from __future__ import annotations

from pathlib import Path
import time

import click

from .imu import BreathCapture, analyze_imu_csv
from .paths import IMU_PLOTS_DIR, RADAR_PLOTS_DIR, RAW_A121_DIR, RAW_IMU_DIR, RAW_RADAR_DIR
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


@cli.command("app")
@click.option("--sensor", type=click.Choice(["radar", "a121", "imu"], case_sensitive=False), default="radar", show_default=True)
@click.option("-p", "--port", help="Preferred serial port, for example COM6.")
@click.option("-b", "--baud", default=921600, show_default=True)
def app(sensor: str, port: str | None, baud: int) -> None:
    """Open the unified radar/IMU desktop app."""
    from .app import launch_app

    raise SystemExit(launch_app(default_sensor=sensor.lower(), default_port=port, default_baud=baud))


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


@cli.command("test-a121")
@click.option("-p", "--port", help="A121 CH342 Interface A serial port, for example COM5.")
@click.option("--start-m", default=0.2, show_default=True, type=float)
@click.option("--end-m", default=1.5, show_default=True, type=float)
@click.option("--profile", default=2, show_default=True, type=click.IntRange(1, 5))
@click.option("--hwaas", default=64, show_default=True, type=int)
@click.option("--sweeps-per-frame", default=12, show_default=True, type=int)
@click.option("--frame-rate-hz", default=50.0, show_default=True, type=float)
@click.option("--frames", default=20, show_default=True, type=int)
def test_a121(
    port: str | None,
    start_m: float,
    end_m: float,
    profile: int,
    hwaas: int,
    sweeps_per_frame: int,
    frame_rate_hz: float,
    frames: int,
) -> None:
    """Connect to an Acconeer A121 and print Sparse IQ peak data."""
    from .a121 import A121Config, A121Capture, find_a121_serial_ports

    if port is None:
        candidates = find_a121_serial_ports()
        if candidates:
            port = candidates[0]
            click.echo(f"Auto-selected likely A121 Interface A port: {port}")
    capture = A121Capture(
        output_dir=RAW_A121_DIR,
        config=A121Config(
            start_m=start_m,
            end_m=end_m,
            profile=profile,
            hwaas=hwaas,
            sweeps_per_frame=sweeps_per_frame,
            frame_rate_hz=frame_rate_hz,
        ),
    )
    if not capture.connect(port):
        raise click.ClickException("Could not connect to the A121. Select the CH342 Interface A port with --port.")
    try:
        while len(capture.data_storage) < frames and capture.running:
            time.sleep(0.05)
        for row in capture.data_storage[:frames]:
            click.echo(
                f"Frame {int(row[1]):03d} | peak={row[2]:.3f} m | "
                f"amp={row[3]:.1f} | phase={row[4]:.2f} rad"
            )
    finally:
        capture.stop()


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
    """Deprecated alias for the unified desktop app in Radar mode."""
    _ = output_dir
    click.echo("live-radar now opens the unified RespiNet app. Use `respi app --sensor radar` directly.")
    from .app import launch_app

    raise SystemExit(launch_app(default_sensor="radar", default_port=port, default_baud=baud))


if __name__ == "__main__":
    cli()
