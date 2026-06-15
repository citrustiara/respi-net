from __future__ import annotations

import asyncio
from pathlib import Path
import csv
import importlib.util
from dataclasses import dataclass
from datetime import datetime
import re
import sys
import threading
import time

import click

from .imu import BreathCapture, analyze_imu_csv
from .iphone_imu import IPhoneIMUBluetoothCapture, discover_iphone_imu_devices
from .paths import DATA_DIR, IMU_PLOTS_DIR, RADAR_PLOTS_DIR, RAW_A121_DIR, RAW_IMU_DIR, RAW_RADAR_DIR
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
@click.option(
    "--sensor",
    type=click.Choice(["radar", "a121", "imu", "iphone-imu", "iphone_imu"], case_sensitive=False),
    default="radar",
    show_default=True,
)
@click.option("-p", "--port", help="Preferred serial port, for example COM6.")
@click.option("-b", "--baud", default=921600, show_default=True)
def app(sensor: str, port: str | None, baud: int) -> None:
    """Open the unified radar/IMU desktop app."""
    from .app import launch_app

    raise SystemExit(launch_app(default_sensor=sensor.lower().replace("-", "_"), default_port=port, default_baud=baud))


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


@cli.command("iphone-imu-devices")
@click.option("--timeout", "timeout_s", default=6.0, show_default=True, type=float, help="BLE scan duration in seconds.")
def iphone_imu_devices(timeout_s: float) -> None:
    """Scan for nearby RespiPhoneIMU BLE peripherals."""
    try:
        devices = asyncio.run(discover_iphone_imu_devices(timeout_s=timeout_s))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    if not devices:
        click.echo("No RespiPhoneIMU devices found. Open the iPhone app and start advertising.")
        return
    for device in devices:
        click.echo(f"{device['name']}  {device['address']}")


@cli.command("capture-iphone-imu")
@click.option("-d", "--device", help="Optional BLE device name/address. Defaults to auto-discovering RespiPhoneIMU.")
@click.option(
    "-s",
    "--seconds",
    type=click.IntRange(1),
    help="Optional capture length. Omit to capture until Ctrl+C.",
)
@click.option("--scan-timeout", default=10.0, show_default=True, type=float, help="BLE discovery timeout in seconds.")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=RAW_IMU_DIR, show_default=True)
@click.option("--plot/--no-plot", default=True, show_default=True)
def capture_iphone_imu(device: str | None, seconds: int | None, scan_timeout: float, output_dir: Path, plot: bool) -> None:
    """Capture batched iPhone IMU samples over BLE, save CSV, and optionally chart it."""
    capture = IPhoneIMUBluetoothCapture(output_dir=output_dir, scan_timeout_s=scan_timeout)
    if not capture.connect(device):
        raise click.ClickException("Could not connect to the iPhone IMU BLE app.")
    if seconds is None:
        click.echo("Capturing iPhone IMU data over BLE. Press Ctrl+C to stop.")
    else:
        click.echo(f"Capturing iPhone IMU data over BLE for {seconds}s. Press Ctrl+C to stop early.")
    started = time.monotonic()
    last_report = -1
    try:
        while capture.running:
            elapsed = time.monotonic() - started
            if seconds is not None and elapsed >= seconds:
                break
            whole_second = int(elapsed)
            if whole_second != last_report:
                last_report = whole_second
                click.echo(f"  {whole_second:>4}s | samples={capture.data_count()}")
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
    csv_path = capture.save()
    click.echo(f"Saved: {csv_path}")
    click.echo(f"Batches: {capture.received_batches} received, {capture.dropped_batches} dropped")
    if plot:
        result = analyze_imu_csv(csv_path)
        click.echo(f"Plot: {result.plot_path}")


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-._")
    return safe or "recording"


def _unique_path(output_dir: Path, stem: str, suffix: str = ".csv") -> Path:
    path = output_dir / f"{stem}{suffix}"
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = output_dir / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise click.ClickException(f"Could not find an unused output filename for {path}")


SLEEP_PLOTS_DIR = DATA_DIR / "plots"


def _default_sleep_outputs(trend_csv: Path, *, garmin: bool = False) -> tuple[Path, Path, Path]:
    stem = trend_csv.stem
    suffix = "_gated_sleep_vitals"
    base = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    tag = "garmin_overlay_sleep_phases" if garmin else "sleep_phases"
    return (
        trend_csv.with_name(f"{base}_{tag}.png"),
        trend_csv.with_name(f"{base}_{tag}.csv"),
        trend_csv.with_name(f"{base}_radar_sleep_score.json"),
    )


def _load_gated_sleep_analyzer():
    tool_path = DATA_DIR.parent / "tools" / "analyze_a121_sleep_vitals_gated.py"
    if not tool_path.exists():
        raise click.ClickException(f"Missing gated sleep analyzer script: {tool_path}")
    spec = importlib.util.spec_from_file_location("respi_net_tools_analyze_a121_sleep_vitals_gated", tool_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load gated sleep analyzer script: {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.analyze


def _run_gated_sleep_analysis(
    csv_path: Path,
    output_dir: Path,
    *,
    window_s: float = 60.0,
    step_s: float = 30.0,
    short_movement_s: float = 30.0,
    min_still_fraction: float = 0.65,
    max_no_presence_fraction: float = 0.05,
    min_hr_confidence: float = 5.0,
    hr_min_bpm: float = 42.0,
    hr_max_bpm: float = 120.0,
    dpi: int = 140,
) -> tuple[Path, Path]:
    analyze = _load_gated_sleep_analyzer()
    return analyze(
        csv_path,
        output_dir,
        window_s=window_s,
        step_s=step_s,
        short_movement_s=short_movement_s,
        min_still_fraction=min_still_fraction,
        max_no_presence_fraction=max_no_presence_fraction,
        min_hr_confidence=min_hr_confidence,
        hr_min_bpm=hr_min_bpm,
        hr_max_bpm=hr_max_bpm,
        dpi=dpi,
    )


@dataclass
class BreathAnnotation:
    timestamp_ms: float
    elapsed_s: float
    frame: int
    event: str
    key: str


class ConsoleBreathMarker:
    """Background key listener for sparse breath phase annotations."""

    def __init__(self, capture: object, started_s: float, inhale_key: str = "i", exhale_key: str = "o"):
        self.capture = capture
        self.started_s = started_s
        self.inhale_key = (inhale_key or "i")[:1].lower()
        self.exhale_key = (exhale_key or "o")[:1].lower()
        self._next_event = {"inhale": "inhale_start", "exhale": "exhale_start"}
        self.annotations: list[BreathAnnotation] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = sys.stdin.isatty()

    def start(self) -> None:
        if not self.available:
            click.echo("Breath marking disabled: stdin is not an interactive terminal.", err=True)
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def snapshot(self) -> list[BreathAnnotation]:
        with self._lock:
            return list(self.annotations)

    def _mark(self, phase: str, key: str) -> None:
        now = time.monotonic()
        elapsed_s = now - self.started_s
        frame = 0
        try:
            frame = int(getattr(self.capture, "total_data_count")())
        except Exception:
            try:
                frame = int(getattr(self.capture, "data_count")())
            except Exception:
                frame = 0
        timestamp_ms = time.time() * 1000.0
        event = self._next_event[phase]
        self._next_event[phase] = f"{phase}_end" if event.endswith("_start") else f"{phase}_start"
        annotation = BreathAnnotation(timestamp_ms=timestamp_ms, elapsed_s=elapsed_s, frame=frame, event=event, key=key)
        with self._lock:
            self.annotations.append(annotation)
        click.echo(f"  marker {event:12s} at {elapsed_s:6.2f}s frame={frame}")

    def _handle_key(self, key: str) -> None:
        key = key.lower()
        if key == self.inhale_key:
            self._mark("inhale", key)
        elif key == self.exhale_key:
            self._mark("exhale", key)

    def _run(self) -> None:
        if sys.platform.startswith("win"):
            try:
                import msvcrt
            except Exception:
                self.available = False
                return
            while not self._stop.is_set():
                if msvcrt.kbhit():
                    raw = msvcrt.getwch()
                    if raw in ("\x00", "\xe0") and msvcrt.kbhit():
                        msvcrt.getwch()
                        continue
                    self._handle_key(raw)
                time.sleep(0.03)
            return

        try:
            import select
            import termios
            import tty
        except Exception:
            self.available = False
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    self._handle_key(sys.stdin.read(1))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _write_breath_annotations(path: Path, annotations: list[BreathAnnotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp_ms", "Elapsed_s", "Frame", "Event", "Key"])
        for ann in annotations:
            writer.writerow([f"{ann.timestamp_ms:.3f}", f"{ann.elapsed_s:.6f}", ann.frame, ann.event, ann.key])


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


@cli.command("record-a121-test")
@click.option("-p", "--port", help="A121 CH342 Interface A serial port, for example COM5.")
@click.option(
    "-s",
    "--seconds",
    type=click.IntRange(20),
    help="Optional recording length for the real-data test. Omit to record until Ctrl+C.",
)
@click.option("--label", default="", help="Optional filename label, e.g. foil-chest or no-foil.")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=RAW_A121_DIR,
    show_default=True,
)
@click.option("--start-m", default=0.2, show_default=True, type=click.FloatRange(0.03, 10.0), help="Requested A121 start range in meters.")
@click.option("--end-m", default=1.5, show_default=True, type=click.FloatRange(0.05, 10.0), help="Requested A121 end range in meters.")
@click.option("--profile", default=3, show_default=True, type=click.IntRange(1, 5), help="A121 profile/radar mode, 1-5.")
@click.option("--hwaas", default=32, show_default=True, type=click.IntRange(1, 511), help="A121 hardware-accelerated averages per sample.")
@click.option("--sweeps-per-frame", default=16, show_default=True, type=click.IntRange(1, 128), help="A121 sweeps per frame.")
@click.option("--frame-rate-hz", default=20.0, show_default=True, type=click.FloatRange(1.0, 100.0), help="Requested A121 frame rate in Hz.")
@click.option("--mark-breaths/--no-mark-breaths", default=False, show_default=True, help="Enable console breath annotations while recording.")
@click.option("--auto-breaths/--no-auto-breaths", default=False, show_default=True, help="After recording, automatically classify inhale/exhale phases and save a breath annotation sidecar.")
@click.option("--inhale-key", default="i", show_default=True, help="Key that toggles inhale_start/inhale_end markers.")
@click.option("--exhale-key", default="o", show_default=True, help="Key that toggles exhale_start/exhale_end markers.")
def record_a121_test(
    port: str | None,
    seconds: int | None,
    label: str,
    output_dir: Path,
    start_m: float,
    end_m: float,
    profile: int,
    hwaas: int,
    sweeps_per_frame: int,
    frame_rate_hz: float,
    mark_breaths: bool,
    auto_breaths: bool,
    inhale_key: str,
    exhale_key: str,
) -> None:
    """Record an A121 Sparse IQ CSV for real-data vitals tests."""
    from .a121 import A121Config, A121Capture, find_a121_serial_ports

    if end_m <= start_m:
        raise click.ClickException("--end-m must be greater than --start-m.")

    if port is None:
        candidates = find_a121_serial_ports()
        if candidates:
            port = candidates[0]
            click.echo(f"Auto-selected likely A121 Interface A port: {port}")
    click.echo(
        "Requested A121 config: "
        f"range {start_m:.3f}-{end_m:.3f} m, "
        f"profile {profile}, HWAAS {hwaas}, "
        f"sweeps/frame {sweeps_per_frame}, frame rate {frame_rate_hz:.1f} Hz"
    )
    capture = A121Capture(
        output_dir=output_dir,
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

    if seconds is None:
        click.echo("Recording A121 real-data test until Ctrl+C. Press Ctrl+C to stop and save data.")
    else:
        click.echo(f"Recording A121 real-data test for {seconds}s. Press Ctrl+C to stop early and save partial data.")
    marker: ConsoleBreathMarker | None = None
    started = time.monotonic()
    if mark_breaths:
        marker = ConsoleBreathMarker(capture, started, inhale_key=inhale_key, exhale_key=exhale_key)
        click.echo(
            "Breath markers enabled: "
            f"press '{marker.inhale_key}' to toggle inhale_start/inhale_end, "
            f"'{marker.exhale_key}' to toggle exhale_start/exhale_end."
        )
        marker.start()
    last_reported_second = -1
    interrupted = False
    final_selection: dict[str, object] = {}
    try:
        while capture.running:
            elapsed_s = time.monotonic() - started
            if seconds is not None and elapsed_s >= seconds:
                break
            whole_second = int(elapsed_s)
            if whole_second != last_reported_second:
                last_reported_second = whole_second
                if seconds is None:
                    click.echo(f"  {whole_second:>5}s | frames={capture.data_count()}")
                else:
                    click.echo(f"  {min(whole_second, seconds):>2}/{seconds}s | frames={capture.data_count()}")
            time.sleep(0.10)
    except KeyboardInterrupt:
        interrupted = True
        click.echo("Interrupted; stopping and saving captured samples...")
    finally:
        if marker is not None:
            marker.stop()
        final_selection = capture.snapshot_acconeer_selection()
        capture.stop()

    rows = capture.data_count()
    if rows < 1:
        raise click.ClickException("No A121 samples were captured; nothing was saved.")
    elapsed_s = time.monotonic() - started
    duration_label = f"{max(1, int(round(elapsed_s)))}s" if seconds is None else f"{seconds}s"
    prefix_parts = ["a121_test", duration_label]
    label = label.strip()
    if label:
        prefix_parts.append(label)
    csv_path = capture.save(prefix="_".join(prefix_parts))
    partial = seconds is not None and (interrupted or elapsed_s + 0.5 < seconds)
    suffix = " partial" if partial else ""
    click.echo(f"Saved{suffix} A121 test: {csv_path}")
    manual_annotation_count = 0
    if marker is not None:
        annotations = marker.snapshot()
        manual_annotation_count = len(annotations)
        if annotations:
            annotation_path = csv_path.with_name(f"{csv_path.stem}_breath_annotations.csv")
            _write_breath_annotations(annotation_path, annotations)
            click.echo(f"Breath annotations: {annotation_path} ({len(annotations)} markers)")
        else:
            click.echo("Breath annotations: no markers were recorded.")
    if auto_breaths:
        from .a121_breaths import generate_a121_breath_annotations

        try:
            auto_path = (
                csv_path.with_name(f"{csv_path.stem}_auto_breath_annotations.csv")
                if manual_annotation_count
                else csv_path.with_name(f"{csv_path.stem}_breath_annotations.csv")
            )
            auto_result = generate_a121_breath_annotations(csv_path, output_path=auto_path, overwrite=False)
            click.echo(
                f"Auto breath annotations: {auto_result.annotations_path} "
                f"({len(auto_result.annotations)} markers, RR {auto_result.resp_bpm:.1f} BPM)"
            )
        except Exception as exc:
            click.echo(f"Auto breath annotation failed: {exc}", err=True)
    click.echo(f"Rows: {rows} | elapsed: {elapsed_s:.1f}s | dropped queued frames: {capture.dropped_results}")
    if final_selection:
        target = final_selection.get("target_distance_m") or final_selection.get("presence_distance_m")
        range_m = final_selection.get("range_m")
        rr = final_selection.get("breathing_rate_bpm")
        target_text = f"{float(target):.3f} m" if target is not None else "none"
        range_text = f"{float(range_m[0]):.3f}-{float(range_m[1]):.3f} m" if range_m else "none"
        rr_text = f"{float(rr):.1f} BPM" if rr is not None else "none"
        click.echo(
            "Final Acconeer selection: "
            f"state {final_selection.get('app_state')}, target {target_text}, range {range_text}, RR {rr_text}"
        )



@cli.command("annotate-a121-breaths")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Output annotation CSV. Defaults to <recording>_breath_annotations.csv.")
@click.option("--overwrite", is_flag=True, help="Overwrite the output annotation CSV if it already exists.")
@click.option(
    "--orientation",
    type=click.Choice(["peak-inhale", "trough-inhale"], case_sensitive=False),
    default="peak-inhale",
    show_default=True,
    help="Mapping from respiration waveform extrema to inhale/exhale phases.",
)
@click.option("--prominence", default=0.30, show_default=True, type=click.FloatRange(0.05, 3.0), help="Extrema prominence threshold on normalized respiration signal.")
def annotate_a121_breaths(csv_path: Path, output: Path | None, overwrite: bool, orientation: str, prominence: float) -> None:
    """Automatically classify inhale/exhale phases for an existing A121 CSV."""
    from .a121_breaths import default_breath_annotations_path, generate_a121_breath_annotations

    output_path = output or default_breath_annotations_path(csv_path)
    try:
        result = generate_a121_breath_annotations(
            csv_path,
            output_path=output_path,
            overwrite=overwrite,
            orientation=orientation.lower(),  # type: ignore[arg-type]
            prominence=prominence,
        )
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"Could not classify A121 breaths: {exc}") from exc
    click.echo(f"Saved auto breath annotations: {result.annotations_path}")
    click.echo(
        f"Markers: {len(result.annotations)} | rows: {result.rows} | duration: {result.duration_s:.1f}s | "
        f"fs: {result.sample_rate_hz:.3f} Hz | target: {result.target_distance_m:.3f} m | RR: {result.resp_bpm:.1f} BPM"
    )


@cli.command("decode-garmin-fit")
@click.argument("sources", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=SLEEP_PLOTS_DIR, show_default=True)
@click.option("--stem", default="garmin_fit", show_default=True, help="Output filename stem for decoded CSV tables.")
def decode_garmin_fit(sources: tuple[Path, ...], output_dir: Path, stem: str) -> None:
    """Decode Garmin FIT file(s)/directory(s) into HR, RR, and sleep-stage CSVs."""
    from .garmin_fit import extract_garmin_reference, write_garmin_reference_csvs

    try:
        data = extract_garmin_reference(list(sources))
        paths = write_garmin_reference_csvs(data, output_dir, stem=stem)
    except Exception as exc:
        raise click.ClickException(f"Could not decode Garmin FIT data: {exc}") from exc
    click.echo(f"Decoded Garmin FIT sources: {', '.join(str(path) for path in sources)}")
    click.echo(f"HR samples: {len(data.heart_rate)} -> {paths['heart_rate']}")
    click.echo(f"RR samples: {len(data.respiration_rate)} -> {paths['respiration_rate']}")
    click.echo(f"Sleep-level changes: {len(data.sleep_levels)} -> {paths['sleep_levels']}")
    click.echo(f"Sleep/event rows: {len(data.sleep_events)} -> {paths['sleep_events']}")


@cli.command("plot-a121-sleep")
@click.argument("trend_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--garmin-fit", "garmin_sources", multiple=True, type=click.Path(exists=True, path_type=Path), help="Garmin .fit file or directory. Repeat for WELLNESS + SLEEP_DATA files.")
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path), help="Output plot PNG. Defaults beside trend CSV.")
@click.option("--csv-output", type=click.Path(dir_okay=False, path_type=Path), help="Optional merged output CSV. Defaults beside trend CSV when --save-data is enabled.")
@click.option("--score-output", type=click.Path(dir_okay=False, path_type=Path), help="Optional sleep-score JSON. Defaults beside trend CSV when score/data are enabled.")
@click.option("--sleep-phases/--no-sleep-phases", default=True, show_default=True, help="Add radar sleep phases to the plot/CSV.")
@click.option("--sleep-score/--no-sleep-score", default=True, show_default=True, help="Compute/write the radar sleep score when phases are enabled.")
@click.option("--save-data/--no-save-data", default=True, show_default=True, help="Write merged CSV/score JSON sidecars.")
@click.option("--dpi", default=150, show_default=True, type=int)
def plot_a121_sleep(
    trend_csv: Path,
    garmin_sources: tuple[Path, ...],
    output: Path | None,
    csv_output: Path | None,
    score_output: Path | None,
    sleep_phases: bool,
    sleep_score: bool,
    save_data: bool,
    dpi: int,
) -> None:
    """Create a sleep vitals/phase plot from an existing gated A121 trend CSV.

    Add --garmin-fit FILE.fit or --garmin-fit DIR to overlay Garmin HR/RR/stages.  If no Garmin
    source is supplied, the plot is radar-only.  Use --no-sleep-phases for a vitals-only plot.
    """
    from .a121_sleep import create_plot

    garmin = list(garmin_sources) or None
    default_png, default_csv, default_score = _default_sleep_outputs(trend_csv, garmin=bool(garmin))
    output_path = output or default_png
    csv_path = (csv_output or default_csv) if save_data else None
    score_path = (score_output or default_score) if save_data and sleep_phases and sleep_score else None
    try:
        plot_path, merged_csv, score_json = create_plot(
            trend_csv,
            garmin,
            output_path,
            csv_path,
            score_path,
            dpi=dpi,
            include_sleep=sleep_phases,
            include_score=sleep_score,
        )
    except Exception as exc:
        raise click.ClickException(f"Could not create A121 sleep plot: {exc}") from exc
    click.echo(f"Plot: {plot_path}")
    if merged_csv is not None:
        click.echo(f"Merged data: {merged_csv}")
    if score_json is not None:
        click.echo(f"Sleep score: {score_json}")


@cli.command("analyze-a121-sleep")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), default=SLEEP_PLOTS_DIR, show_default=True)
@click.option("--garmin-fit", "garmin_sources", multiple=True, type=click.Path(exists=True, path_type=Path), help="Optional Garmin .fit file or directory for overlay/alignment.")
@click.option("--sleep-plot/--no-sleep-plot", default=True, show_default=True, help="Create the sleep phase/comparison plot after vitals extraction.")
@click.option("--sleep-data/--no-sleep-data", default=True, show_default=True, help="Write sleep phase CSV and score JSON sidecars.")
@click.option("--sleep-phases/--no-sleep-phases", default=True, show_default=True, help="Classify radar sleep phases. Disable for vitals-only comparison plot.")
@click.option("--sleep-score/--no-sleep-score", default=True, show_default=True, help="Compute/write the radar sleep score when phases are enabled.")
@click.option("--window-s", default=60.0, show_default=True, type=float)
@click.option("--step-s", default=30.0, show_default=True, type=float)
@click.option("--short-movement-s", default=30.0, show_default=True, type=float)
@click.option("--min-still-fraction", default=0.65, show_default=True, type=float)
@click.option("--max-no-presence-fraction", default=0.05, show_default=True, type=float)
@click.option("--min-hr-confidence", default=5.0, show_default=True, type=float)
@click.option("--hr-min-bpm", default=42.0, show_default=True, type=float, help="Minimum HR accepted/analyzed, in BPM.")
@click.option("--hr-max-bpm", default=120.0, show_default=True, type=float, help="Maximum HR accepted/analyzed, in BPM.")
@click.option("--dpi", default=140, show_default=True, type=int)
def analyze_a121_sleep(
    csv_path: Path,
    output_dir: Path,
    garmin_sources: tuple[Path, ...],
    sleep_plot: bool,
    sleep_data: bool,
    sleep_phases: bool,
    sleep_score: bool,
    window_s: float,
    step_s: float,
    short_movement_s: float,
    min_still_fraction: float,
    max_no_presence_fraction: float,
    min_hr_confidence: float,
    hr_min_bpm: float,
    hr_max_bpm: float,
    dpi: int,
) -> None:
    """Run full offline sleep analysis from a raw A121 sleep CSV."""
    from .a121_sleep import create_plot, write_sleep_outputs

    try:
        vitals_png, trend_csv = _run_gated_sleep_analysis(
            csv_path,
            output_dir,
            window_s=window_s,
            step_s=step_s,
            short_movement_s=short_movement_s,
            min_still_fraction=min_still_fraction,
            max_no_presence_fraction=max_no_presence_fraction,
            min_hr_confidence=min_hr_confidence,
            hr_min_bpm=hr_min_bpm,
            hr_max_bpm=hr_max_bpm,
            dpi=dpi,
        )
    except Exception as exc:
        raise click.ClickException(f"Could not analyze A121 sleep vitals: {exc}") from exc
    click.echo(f"Vitals plot: {vitals_png}")
    click.echo(f"Trend CSV: {trend_csv}")

    garmin = list(garmin_sources) or None
    default_png, default_csv, default_score = _default_sleep_outputs(trend_csv, garmin=bool(garmin))
    if sleep_plot:
        try:
            plot_path, merged_csv, score_json = create_plot(
                trend_csv,
                garmin,
                default_png,
                default_csv if sleep_data else None,
                default_score if sleep_data and sleep_phases and sleep_score else None,
                dpi=dpi,
                include_sleep=sleep_phases,
                include_score=sleep_score,
            )
        except Exception as exc:
            raise click.ClickException(f"Could not create sleep comparison plot: {exc}") from exc
        click.echo(f"Sleep/comparison plot: {plot_path}")
        if merged_csv is not None:
            click.echo(f"Sleep/merged data: {merged_csv}")
        if score_json is not None:
            click.echo(f"Sleep score: {score_json}")
    elif sleep_data and sleep_phases:
        try:
            merged_csv, score_json, score = write_sleep_outputs(
                trend_csv,
                garmin,
                csv_output=default_csv,
                score_output=default_score if sleep_score else None,
                include_score=sleep_score,
            )
        except Exception as exc:
            raise click.ClickException(f"Could not write sleep phase data: {exc}") from exc
        if merged_csv is not None:
            click.echo(f"Sleep/merged data: {merged_csv}")
        if score_json is not None:
            click.echo(f"Sleep score: {score_json}")
        if score is not None:
            click.echo(f"Radar sleep score: {score.score} ({score.quality})")


@cli.command("record-a121-sleep")
@click.option("-p", "--port", help="A121 CH342 Interface A serial port, for example COM5.")
@click.option("--hours", type=click.FloatRange(0.01, 24.0), help="Optional maximum recording length. Omit to record until Ctrl+C.")
@click.option("--label", default="", help="Optional filename label, e.g. sleep-night-1.")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=RAW_A121_DIR,
    show_default=True,
)
@click.option("--start-m", default=0.2, show_default=True, type=click.FloatRange(0.03, 10.0), help="Requested A121 start range in meters.")
@click.option("--end-m", default=1.5, show_default=True, type=click.FloatRange(0.05, 10.0), help="Requested A121 end range in meters.")
@click.option("--profile", default=3, show_default=True, type=click.IntRange(1, 5), help="A121 profile/radar mode, 1-5.")
@click.option("--hwaas", default=32, show_default=True, type=click.IntRange(1, 511), help="A121 hardware-accelerated averages per sample.")
@click.option("--sweeps-per-frame", default=16, show_default=True, type=click.IntRange(1, 128), help="A121 sweeps per frame.")
@click.option("--frame-rate-hz", default=20.0, show_default=True, type=click.FloatRange(1.0, 100.0), help="Requested A121 frame rate in Hz.")
@click.option("--summary-interval-s", default=30.0, show_default=True, type=click.FloatRange(5.0, 600.0), help="Progress print/flush interval.")
@click.option("--auto-analyze/--no-auto-analyze", default=True, show_default=True, help="After stopping, extract gated vitals and optionally create sleep outputs.")
@click.option("--auto-plot/--no-auto-plot", default=True, show_default=True, help="After auto-analysis, create the sleep/comparison plot.")
@click.option("--sleep-data/--no-sleep-data", default=True, show_default=True, help="After auto-analysis, write sleep phase CSV and score JSON.")
@click.option("--sleep-phases/--no-sleep-phases", default=True, show_default=True, help="Classify radar sleep phases during auto-analysis.")
@click.option("--sleep-score/--no-sleep-score", default=True, show_default=True, help="Compute/write radar sleep score during auto-analysis.")
@click.option("--garmin-fit", "garmin_sources", multiple=True, type=click.Path(exists=True, path_type=Path), help="Optional Garmin .fit file or directory to overlay after recording.")
@click.option("--analysis-output-dir", type=click.Path(file_okay=False, path_type=Path), default=SLEEP_PLOTS_DIR, show_default=True, help="Output directory for auto-generated plots/data.")
@click.option("--analysis-dpi", default=140, show_default=True, type=int, help="DPI for auto-generated analysis plots.")
def record_a121_sleep(
    port: str | None,
    hours: float | None,
    label: str,
    output_dir: Path,
    start_m: float,
    end_m: float,
    profile: int,
    hwaas: int,
    sweeps_per_frame: int,
    frame_rate_hz: float,
    summary_interval_s: float,
    auto_analyze: bool,
    auto_plot: bool,
    sleep_data: bool,
    sleep_phases: bool,
    sleep_score: bool,
    garmin_sources: tuple[Path, ...],
    analysis_output_dir: Path,
    analysis_dpi: int,
) -> None:
    """Stream A121 overnight; by default auto-generate vitals, sleep phases, score, and plot when stopped."""
    from .a121 import A121Config, A121Capture, A121_CAPTURE_COLUMNS, find_a121_serial_ports

    if end_m <= start_m:
        raise click.ClickException("--end-m must be greater than --start-m.")

    if port is None:
        candidates = find_a121_serial_ports()
        if candidates:
            port = candidates[0]
            click.echo(f"Auto-selected likely A121 Interface A port: {port}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_parts = ["a121_sleep"]
    label = label.strip()
    if label:
        prefix_parts.append(label)
    stem = f"{_safe_filename_part('_'.join(prefix_parts))}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    csv_path = _unique_path(output_dir, stem)

    write_lock = threading.Lock()
    rows_written = 0
    f = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(A121_CAPTURE_COLUMNS)

    def write_row(row: list[object]) -> None:
        nonlocal rows_written
        with write_lock:
            writer.writerow(row)
            rows_written += 1

    click.echo(
        "Requested A121 sleep config: "
        f"range {start_m:.3f}-{end_m:.3f} m, profile {profile}, HWAAS {hwaas}, "
        f"sweeps/frame {sweeps_per_frame}, frame rate {frame_rate_hz:.1f} Hz"
    )
    capture = A121Capture(
        output_dir=output_dir,
        config=A121Config(
            start_m=start_m,
            end_m=end_m,
            profile=profile,
            hwaas=hwaas,
            sweeps_per_frame=sweeps_per_frame,
            frame_rate_hz=frame_rate_hz,
        ),
        storage_max_rows=0,
        on_storage_row=write_row,
    )
    connected = False
    interrupted = False
    completed_time_limit = False
    unexpected_stop = False
    try:
        if not capture.connect(port):
            with write_lock:
                f.close()
            try:
                csv_path.unlink()
            except Exception:
                pass
            raise click.ClickException("Could not connect to the A121. Select the CH342 Interface A port with --port.")
        connected = True
        if hours is None:
            click.echo("Recording A121 sleep CSV until Ctrl+C. Press Ctrl+C when you wake up to stop and keep the file.")
        else:
            click.echo(f"Recording A121 sleep CSV for up to {hours:.2f} h. Press Ctrl+C to stop early and keep the file.")
        click.echo(f"Streaming to: {csv_path}")
        started = time.monotonic()
        next_report = 0.0
        max_seconds = float(hours) * 3600.0 if hours is not None else None
        while capture.running:
            elapsed_s = time.monotonic() - started
            if max_seconds is not None and elapsed_s >= max_seconds:
                completed_time_limit = True
                break
            if elapsed_s >= next_report:
                with write_lock:
                    f.flush()
                selection = capture.snapshot_acconeer_selection()
                rr = selection.get("breathing_rate_bpm")
                target = selection.get("target_distance_m") or selection.get("presence_distance_m")
                rr_text = f", RR={float(rr):.1f} BPM" if rr is not None else ""
                target_text = f", target={float(target):.3f} m" if target is not None else ""
                click.echo(f"  elapsed {elapsed_s / 3600.0:5.2f} h | rows={rows_written}{target_text}{rr_text}")
                next_report = elapsed_s + float(summary_interval_s)
            time.sleep(0.25)
        if not capture.running and not completed_time_limit:
            unexpected_stop = True
            click.echo("A121 capture stopped unexpectedly; keeping captured samples...", err=True)
    except KeyboardInterrupt:
        interrupted = True
        click.echo("Interrupted; stopping and keeping captured samples...")
    finally:
        final_selection = capture.snapshot_acconeer_selection() if connected else {}
        capture.stop()
        with write_lock:
            if not f.closed:
                f.flush()
                f.close()

    if rows_written < 1:
        try:
            csv_path.unlink()
        except Exception:
            pass
        raise click.ClickException("No A121 samples were captured; removed empty sleep CSV.")
    suffix = " partial" if interrupted or unexpected_stop else ""
    click.echo(f"Saved{suffix} A121 sleep recording: {csv_path}")
    click.echo(f"Rows: {rows_written} | dropped queued frames: {capture.dropped_results}")
    if final_selection:
        target = final_selection.get("target_distance_m") or final_selection.get("presence_distance_m")
        range_m = final_selection.get("range_m")
        rr = final_selection.get("breathing_rate_bpm")
        target_text = f"{float(target):.3f} m" if target is not None else "none"
        range_text = f"{float(range_m[0]):.3f}-{float(range_m[1]):.3f} m" if range_m else "none"
        rr_text = f"{float(rr):.1f} BPM" if rr is not None else "none"
        click.echo(
            "Final Acconeer selection: "
            f"state {final_selection.get('app_state')}, target {target_text}, range {range_text}, RR {rr_text}"
        )

    if auto_analyze:
        from .a121_sleep import create_plot, write_sleep_outputs

        try:
            vitals_png, trend_csv = _run_gated_sleep_analysis(csv_path, analysis_output_dir, dpi=analysis_dpi)
        except Exception as exc:
            click.echo(f"Auto sleep vitals analysis failed: {exc}", err=True)
            return
        click.echo(f"Auto vitals plot: {vitals_png}")
        click.echo(f"Auto trend CSV: {trend_csv}")

        garmin = list(garmin_sources) or None
        default_png, default_csv, default_score = _default_sleep_outputs(trend_csv, garmin=bool(garmin))
        if auto_plot:
            try:
                plot_path, merged_csv, score_json = create_plot(
                    trend_csv,
                    garmin,
                    default_png,
                    default_csv if sleep_data else None,
                    default_score if sleep_data and sleep_phases and sleep_score else None,
                    dpi=analysis_dpi,
                    include_sleep=sleep_phases,
                    include_score=sleep_score,
                )
            except Exception as exc:
                click.echo(f"Auto sleep/comparison plot failed: {exc}", err=True)
                return
            click.echo(f"Auto sleep/comparison plot: {plot_path}")
            if merged_csv is not None:
                click.echo(f"Auto sleep/merged data: {merged_csv}")
            if score_json is not None:
                click.echo(f"Auto sleep score: {score_json}")
        elif sleep_data and sleep_phases:
            try:
                merged_csv, score_json, score = write_sleep_outputs(
                    trend_csv,
                    garmin,
                    csv_output=default_csv,
                    score_output=default_score if sleep_score else None,
                    include_score=sleep_score,
                )
            except Exception as exc:
                click.echo(f"Auto sleep data export failed: {exc}", err=True)
                return
            if merged_csv is not None:
                click.echo(f"Auto sleep/merged data: {merged_csv}")
            if score_json is not None:
                click.echo(f"Auto sleep score: {score_json}")
            if score is not None:
                click.echo(f"Auto radar sleep score: {score.score} ({score.quality})")


cli.add_command(record_a121_test, "capture-a121")


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
