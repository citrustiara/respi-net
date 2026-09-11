from pathlib import Path
import asyncio
import importlib
import struct
import time

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from respi_net.a121 import A121_COLUMNS
from respi_net.a121_vitals import A121LiveTraceProcessor, HeartRateKalmanTracker, analyze_a121_vitals, sample_rate_from_ms
from respi_net.app import _a121_stats, _detect_sensor
from respi_net.cli import cli
from respi_net.imu import BreathCapture, _sampling_rate, analyze_imu_csv
from respi_net.iphone_imu import parse_iphone_imu_packet, probe_iphone_imu_device
from respi_net.radar import RadarCapture, analyze_radar_csv


def test_analyze_radar_csv(tmp_path: Path) -> None:
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    csv_path = tmp_path / "radar_raw_sample.csv"
    pd.DataFrame(
        {
            "Timestamp_ms": t * 1000,
            "RawADC": 2048 + 20 * np.sin(2 * np.pi * 2.0 * t),
            "Voltage_mV": 1650 + 50 * np.sin(2 * np.pi * 2.0 * t),
        }
    ).to_csv(csv_path, index=False)

    result = analyze_radar_csv(csv_path, output_dir=tmp_path)

    assert result.plot_path and result.plot_path.exists()
    assert 95 <= result.sample_rate_hz <= 105
    assert 1.8 <= result.peak_frequency_hz <= 2.2


def test_analyze_imu_csv(tmp_path: Path) -> None:
    fs = 100.0
    t = np.arange(0, 30, 1 / fs)
    breathing = 0.03 * np.sin(2 * np.pi * 0.25 * t)
    heart = 0.005 * np.sin(2 * np.pi * 1.2 * t)
    csv_path = tmp_path / "respiratory_6axis_raw_sample.csv"
    pd.DataFrame(
        {
            "Time_ms": t * 1000,
            "ax": breathing + heart,
            "ay": 0.5 * breathing,
            "az": 1.0 + 0.2 * breathing,
            "gx": 0.01 * np.cos(2 * np.pi * 0.25 * t),
            "gy": 0.01 * np.sin(2 * np.pi * 0.25 * t),
            "gz": 0.01 * np.cos(2 * np.pi * 0.1 * t),
        }
    ).to_csv(csv_path, index=False)

    result = analyze_imu_csv(csv_path, output_dir=tmp_path)

    assert result.plot_path and result.plot_path.exists()
    assert 95 <= result.sample_rate_hz <= 105
    assert result.heart_bpm >= 40


def test_imu_sampling_rate_uses_median_without_mutating_dataframe() -> None:
    df = pd.DataFrame({"Time_ms": [0.0, 10.0, 20.0, 1030.0, 1040.0]})

    assert _sampling_rate(df) == pytest.approx(100.0)
    assert list(df.columns) == ["Time_ms"]


def test_radar_sampling_rate_ignores_timestamp_outlier(tmp_path: Path) -> None:
    timestamps_ms = np.concatenate([np.arange(100) * 10.0, 2000.0 + np.arange(100) * 10.0])
    csv_path = tmp_path / "radar_raw_outlier.csv"
    pd.DataFrame(
        {
            "Timestamp_ms": timestamps_ms,
            "RawADC": np.arange(len(timestamps_ms)),
            "Voltage_mV": np.sin(np.arange(len(timestamps_ms))),
        }
    ).to_csv(csv_path, index=False)

    result = analyze_radar_csv(csv_path, output_dir=tmp_path, save_plot=False)

    assert result.sample_rate_hz == pytest.approx(100.0)


def test_parse_iphone_imu_packet() -> None:
    payload = struct.pack(
        "<BBH"
        "Ihhhhhh"
        "Ihhhhhh",
        1,
        2,
        513,
        10,
        1000,
        -250,
        42,
        1234,
        -567,
        0,
        20,
        -1000,
        500,
        100,
        -1234,
        567,
        25,
    )

    packet = parse_iphone_imu_packet(payload)

    assert packet.sequence == 513
    assert len(packet.samples) == 2
    assert packet.samples[0].time_ms == 10
    assert packet.samples[0].ax == 1.0
    assert packet.samples[0].ay == -0.25
    assert packet.samples[0].az == 0.042
    assert packet.samples[0].gx == 12.34
    assert packet.samples[0].gy == -5.67
    assert packet.samples[1].gx == -12.34


def test_parse_iphone_imu_packet_rejects_bad_length() -> None:
    payload = struct.pack("<BBH", 1, 1, 0)

    with pytest.raises(ValueError):
        parse_iphone_imu_packet(payload)


def test_iphone_probe_timeout_respects_total_sample_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    iphone_imu = importlib.import_module("respi_net.iphone_imu")

    class Device:
        name = "RespiPhoneIMU"
        address = "test-device"

    class FakeScanner:
        @staticmethod
        async def find_device_by_filter(_filter, timeout):
            return Device()

    class FakeClient:
        commands: list[bytes] = []

        def __init__(self, _device):
            self.is_connected = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def start_notify(self, _uuid, _callback):
            return None

        async def stop_notify(self, _uuid):
            return None

        async def write_gatt_char(self, _uuid, payload, response):
            self.commands.append(payload)

    monkeypatch.setattr(iphone_imu, "_load_bleak", lambda: (FakeScanner, FakeClient))
    started = time.monotonic()

    result = asyncio.run(probe_iphone_imu_device(sample_seconds=0.3))

    elapsed = time.monotonic() - started
    assert result.samples == 0
    assert 0.25 <= elapsed < 0.45
    assert FakeClient.commands == [b"START", b"STOP"]


@pytest.mark.parametrize("capture_class", [BreathCapture, RadarCapture])
def test_serial_capture_stops_reader_before_closing_port(capture_class) -> None:
    events: list[str] = []

    class FakeThread:
        def is_alive(self):
            return True

        def join(self, timeout):
            events.append("joined")

    class FakePort:
        is_open = True

        def close(self):
            events.append("closed")

    capture = capture_class()
    capture.running = True
    capture.read_thread = FakeThread()
    capture.serial_port = FakePort()

    capture.stop()

    assert events == ["joined", "closed"]


def test_capture_cli_reports_short_capture_as_click_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cli_module = importlib.import_module("respi_net.cli")

    class FakeCapture:
        def __init__(self, **_kwargs):
            pass

        def connect(self, _port):
            return True

        def stop(self):
            return None

        def save(self):
            raise ValueError("Not enough samples to save; at least 10 are required.")

    monkeypatch.setattr(cli_module, "BreathCapture", FakeCapture)
    monkeypatch.setattr(cli_module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()))

    result = CliRunner().invoke(cli, ["capture-imu", "--no-plot"])

    assert result.exit_code == 1
    assert "Error: Not enough samples to save" in result.output


def test_a121_csv_detection_and_stats() -> None:
    fs = 20.0
    t = np.arange(0, 5, 1 / fs)
    distances = "[0.2,0.3,0.4]"
    df = pd.DataFrame(
        {
            "Timestamp_ms": t * 1000,
            "Frame": np.arange(len(t)),
            "PeakDistance_m": 0.3 + 0.005 * np.sin(2 * np.pi * 0.25 * t),
            "PeakAmplitude": 100 + 5 * np.sin(2 * np.pi * 0.25 * t),
            "PeakPhase_rad": np.sin(2 * np.pi * 0.25 * t),
            "MeanAmplitude": 20.0,
            "Distances_m": distances,
            "Amplitude": "[10,100,20]",
            "Phase": "[0.1,0.2,0.3]",
            "Real": "[1,2,3]",
            "Imag": "[0,1,0]",
        }
    )

    assert list(df.columns) == A121_COLUMNS
    assert _detect_sensor(df) == "a121"
    stats = _a121_stats(df)
    assert 18 <= stats["sample_rate_hz"] <= 22
    assert 0.29 <= stats["peak_distance_m"] <= 0.31


def test_a121_fixed_target_gate_overrides_latest_peak() -> None:
    fs = 20.0
    t = np.arange(0, 10, 1 / fs)
    distances = np.linspace(0.2, 1.2, 81)
    locked_idx = int(np.argmin(np.abs(distances - 0.62)))
    new_peak_idx = int(np.argmin(np.abs(distances - 0.95)))
    rows = []
    for frame, ts in enumerate(t):
        phase = 0.2 * np.sin(2 * np.pi * 0.25 * ts) * np.ones(len(distances))
        amp = 25 * np.ones(len(distances))
        amp += 60 * np.exp(-0.5 * ((distances - distances[locked_idx]) / 0.025) ** 2)
        amp += 180 * np.exp(-0.5 * ((distances - distances[new_peak_idx]) / 0.025) ** 2)
        iq = amp * np.exp(1j * phase)
        rows.append(
            {
                "Timestamp_ms": ts * 1000,
                "Frame": frame,
                "PeakDistance_m": float(distances[int(np.argmax(amp))]),
                "PeakAmplitude": float(np.max(amp)),
                "PeakPhase_rad": float(np.angle(iq[int(np.argmax(amp))])),
                "MeanAmplitude": float(np.mean(amp)),
                "Distances_m": "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "Amplitude": "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "Phase": "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "Real": "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "Imag": "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            }
        )
    df = pd.DataFrame(rows, columns=A121_COLUMNS)

    analysis = analyze_a121_vitals(df, auto_gate=False, gate_half_width_m=0.08, target_distance_m=float(distances[locked_idx]))

    assert abs(analysis.peak_distance_m - distances[new_peak_idx]) <= 0.02
    assert abs(analysis.target_distance_m - distances[locked_idx]) <= 0.02
    assert analysis.gate_min_m <= distances[locked_idx] <= analysis.gate_max_m


def test_a121_auto_gate_stays_near_latest_peak() -> None:
    fs = 20.0
    t = np.arange(0, 20, 1 / fs)
    distances = np.linspace(0.2, 1.2, 81)
    peak_idx = int(np.argmin(np.abs(distances - 0.62)))
    moving_clutter_idx = int(np.argmin(np.abs(distances - 0.95)))
    rng = np.random.default_rng(7)
    rows = []
    for frame, ts in enumerate(t):
        phase = 0.03 * rng.normal(size=len(distances))
        phase[peak_idx] = 0.20 * np.sin(2 * np.pi * 0.25 * ts)
        phase[moving_clutter_idx] = 1.20 * np.sin(2 * np.pi * 1.0 * ts)
        amp = 25 + rng.normal(size=len(distances))
        amp += 180 * np.exp(-0.5 * ((distances - distances[peak_idx]) / 0.025) ** 2)
        amp[moving_clutter_idx] += 35
        iq = amp * np.exp(1j * phase)
        rows.append(
            {
                "Timestamp_ms": ts * 1000,
                "Frame": frame,
                "PeakDistance_m": float(distances[int(np.argmax(amp))]),
                "PeakAmplitude": float(np.max(amp)),
                "PeakPhase_rad": float(np.angle(iq[int(np.argmax(amp))])),
                "MeanAmplitude": float(np.mean(amp)),
                "Distances_m": "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "Amplitude": "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "Phase": "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "Real": "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "Imag": "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            }
        )
    df = pd.DataFrame(rows, columns=A121_COLUMNS)

    analysis = analyze_a121_vitals(df, gate_half_width_m=0.08)

    assert abs(analysis.target_distance_m - distances[peak_idx]) <= 0.02
    assert analysis.gate_min_m <= distances[peak_idx] <= analysis.gate_max_m
    assert abs(analysis.target_distance_m - distances[moving_clutter_idx]) > 0.20


def test_a121_iq_phase_vitals_auto_gate_presence() -> None:
    fs = 20.0
    t = np.arange(0, 60, 1 / fs)
    distances = np.linspace(0.2, 1.2, 81)
    target_idx = int(np.argmin(np.abs(distances - 0.62)))
    rng = np.random.default_rng(42)
    rows = []
    for frame, ts in enumerate(t):
        phase_motion = 0.32 * np.sin(2 * np.pi * 0.25 * ts) + 0.08 * np.sin(2 * np.pi * 1.20 * ts)
        amp = 35 + 2 * rng.normal(size=len(distances))
        amp += 180 * np.exp(-0.5 * ((distances - distances[target_idx]) / 0.035) ** 2)
        phase = 0.5 * rng.normal(size=len(distances))
        phase[target_idx - 1 : target_idx + 2] = phase_motion + 0.03 * rng.normal(size=3)
        iq = amp * np.exp(1j * phase)
        rows.append(
            {
                "Timestamp_ms": ts * 1000,
                "Frame": frame,
                "PeakDistance_m": float(distances[int(np.argmax(amp))]),
                "PeakAmplitude": float(np.max(amp)),
                "PeakPhase_rad": float(np.angle(iq[int(np.argmax(amp))])),
                "MeanAmplitude": float(np.mean(amp)),
                "Distances_m": "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "Amplitude": "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "Phase": "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "Real": "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "Imag": "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            }
        )
    df = pd.DataFrame(rows, columns=A121_COLUMNS)

    analysis = analyze_a121_vitals(df)

    assert analysis.present
    assert abs(analysis.target_distance_m - distances[target_idx]) <= 0.03
    assert 14.0 <= analysis.resp_bpm <= 16.0
    assert 68.0 <= analysis.heart_bpm <= 76.0
    assert len(analysis.resp_signal) == len(analysis.times_s)
    assert len(analysis.heart_signal) == len(analysis.times_s)


def test_a121_iq_phase_vitals_with_heavy_clutter() -> None:
    fs = 20.0
    t = np.arange(0, 60, 1 / fs)
    distances = np.linspace(0.2, 1.2, 81)
    target_idx = int(np.argmin(np.abs(distances - 0.62)))
    rng = np.random.default_rng(42)
    clutter = 100.0 + 100.0j
    rows = []
    for frame, ts in enumerate(t):
        phase_motion = 0.32 * np.sin(2 * np.pi * 0.25 * ts) + 0.08 * np.sin(2 * np.pi * 1.20 * ts)
        amp = 35 + 2 * rng.normal(size=len(distances))
        amp += 180 * np.exp(-0.5 * ((distances - distances[target_idx]) / 0.035) ** 2)
        phase = 0.5 * rng.normal(size=len(distances))
        phase[target_idx - 1 : target_idx + 2] = phase_motion + 0.03 * rng.normal(size=3)
        iq = amp * np.exp(1j * phase) + clutter
        rows.append(
            {
                "Timestamp_ms": ts * 1000,
                "Frame": frame,
                "PeakDistance_m": float(distances[int(np.argmax(np.abs(iq)))]),
                "PeakAmplitude": float(np.max(np.abs(iq))),
                "PeakPhase_rad": float(np.angle(iq[int(np.argmax(np.abs(iq)))])),
                "MeanAmplitude": float(np.mean(np.abs(iq))),
                "Distances_m": "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "Amplitude": "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "Phase": "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "Real": "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "Imag": "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            }
        )
    df = pd.DataFrame(rows, columns=A121_COLUMNS)

    analysis = analyze_a121_vitals(df)

    assert analysis.present
    assert abs(analysis.target_distance_m - distances[target_idx]) <= 0.03
    assert 14.0 <= analysis.resp_bpm <= 16.0
    assert 68.0 <= analysis.heart_bpm <= 76.0
    assert len(analysis.resp_signal) == len(analysis.times_s)
    assert len(analysis.heart_signal) == len(analysis.times_s)


def test_a121_bursty_host_timestamps_use_total_frame_rate() -> None:
    times = [0.0]
    for _ in range(100):
        times.extend([times[-1], times[-1] + 0.5, times[-1] + 50.0])
    fs = sample_rate_from_ms(np.asarray(times), default=20.0)
    assert 55.0 <= fs <= 65.0


def test_a121_gating_disabled_ignores_stale_target() -> None:
    fs = 20.0
    t = np.arange(0, 30, 1 / fs)
    distances = np.linspace(0.2, 1.2, 81)
    target_idx = int(np.argmin(np.abs(distances - 0.80)))
    stale_idx = int(np.argmin(np.abs(distances - 0.40)))
    rows = []
    for frame, ts in enumerate(t):
        phase_motion = 0.25 * np.sin(2 * np.pi * 0.25 * ts) + 0.05 * np.sin(2 * np.pi * 1.2 * ts)
        amp = 25 * np.ones(len(distances))
        amp += 170 * np.exp(-0.5 * ((distances - distances[target_idx]) / 0.030) ** 2)
        amp[stale_idx] += 30
        phase = 0.2 * np.random.default_rng(frame).normal(size=len(distances))
        phase[target_idx - 1 : target_idx + 2] = phase_motion
        iq = amp * np.exp(1j * phase)
        rows.append(
            {
                "Timestamp_ms": ts * 1000,
                "Frame": frame,
                "PeakDistance_m": float(distances[int(np.argmax(amp))]),
                "PeakAmplitude": float(np.max(amp)),
                "PeakPhase_rad": float(np.angle(iq[int(np.argmax(amp))])),
                "MeanAmplitude": float(np.mean(amp)),
                "Distances_m": "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "Amplitude": "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "Phase": "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "Real": "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "Imag": "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            }
        )
    df = pd.DataFrame(rows, columns=A121_COLUMNS)

    analysis = analyze_a121_vitals(
        df,
        target_distance_m=float(distances[stale_idx]),
        use_gating=False,
    )

    assert abs(analysis.target_distance_m - distances[target_idx]) <= 0.03


def test_a121_short_recording_does_not_report_random_rates() -> None:
    fs = 20.0
    t = np.arange(0, 8, 1 / fs)
    distances = np.linspace(0.2, 1.2, 81)
    target_idx = int(np.argmin(np.abs(distances - 0.62)))
    rows = []
    for frame, ts in enumerate(t):
        phase_motion = 0.3 * np.sin(2 * np.pi * 0.25 * ts) + 0.08 * np.sin(2 * np.pi * 1.2 * ts)
        amp = 30 * np.ones(len(distances))
        amp += 180 * np.exp(-0.5 * ((distances - distances[target_idx]) / 0.035) ** 2)
        phase = np.zeros(len(distances))
        phase[target_idx - 1 : target_idx + 2] = phase_motion
        iq = amp * np.exp(1j * phase)
        rows.append(
            {
                "Timestamp_ms": ts * 1000,
                "Frame": frame,
                "PeakDistance_m": float(distances[target_idx]),
                "PeakAmplitude": float(np.max(amp)),
                "PeakPhase_rad": float(np.angle(iq[target_idx])),
                "MeanAmplitude": float(np.mean(amp)),
                "Distances_m": "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "Amplitude": "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "Phase": "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "Real": "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "Imag": "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            }
        )
    analysis = analyze_a121_vitals(pd.DataFrame(rows, columns=A121_COLUMNS))

    assert analysis.resp_bpm == 0.0
    assert analysis.resp_confidence == 0.0 or analysis.resp_hz == 0.0


def test_a121_static_clutter_does_not_report_vital_rates() -> None:
    fs = 20.0
    t = np.arange(0, 60, 1 / fs)
    distances = np.linspace(0.2, 1.2, 81)
    clutter_idx = int(np.argmin(np.abs(distances - 0.58)))
    rng = np.random.default_rng(124)
    rows = []
    for frame, ts in enumerate(t):
        amp = 30 * np.ones(len(distances))
        amp += 150 * np.exp(-0.5 * ((distances - distances[clutter_idx]) / 0.03) ** 2)
        phase = 0.02 * rng.normal(size=len(distances))
        iq = amp * np.exp(1j * phase)
        rows.append(
            {
                "Timestamp_ms": ts * 1000,
                "Frame": frame,
                "PeakDistance_m": float(distances[clutter_idx]),
                "PeakAmplitude": float(np.max(amp)),
                "PeakPhase_rad": float(np.angle(iq[clutter_idx])),
                "MeanAmplitude": float(np.mean(amp)),
                "Distances_m": "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "Amplitude": "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "Phase": "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "Real": "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "Imag": "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            }
        )

    analysis = analyze_a121_vitals(pd.DataFrame(rows, columns=A121_COLUMNS), use_gating=False)

    assert analysis.resp_bpm == 0.0
    assert analysis.heart_bpm == 0.0


def test_a121_live_trace_wide_gate_keeps_compact_phase_segment() -> None:
    fs = 20.0
    t = np.arange(0, 30, 1 / fs)
    distances = np.linspace(0.3, 0.7, 81)
    target_idx = int(np.argmin(np.abs(distances - 0.50)))
    rows = []
    for frame, ts in enumerate(t):
        phase_motion = 0.30 * np.sin(2 * np.pi * 0.25 * ts)
        amp = 90 * np.ones(len(distances))
        amp[target_idx - 1 : target_idx + 2] = [110, 130, 110]
        phase = np.zeros(len(distances))
        phase[target_idx - 1 : target_idx + 2] = phase_motion
        iq = amp * np.exp(1j * phase)
        rows.append(
            [
                ts * 1000,
                frame,
                float(distances[target_idx]),
                float(np.max(amp)),
                float(np.angle(iq[target_idx])),
                float(np.mean(amp)),
                "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            ]
        )

    processor = A121LiveTraceProcessor()
    result = processor.process_rows(rows, target_distance_m=float(distances[target_idx]), use_gating=True, gate_half_width_m=0.10)

    assert result is not None
    tail = result.resp_signal[int(20 * fs) :]
    assert np.std(tail) > 0.03


def test_a121_live_trace_is_append_only_for_existing_samples() -> None:
    fs = 20.0
    t = np.arange(0, 12, 1 / fs)
    distances = np.linspace(0.3, 0.7, 9)
    target_idx = 4
    rows = []
    for frame, ts in enumerate(t):
        phase_motion = 0.30 * np.sin(2 * np.pi * 0.25 * ts)
        amp = 20 * np.ones(len(distances))
        amp[target_idx] = 120
        phase = np.zeros(len(distances))
        phase[target_idx - 1 : target_idx + 2] = phase_motion
        iq = amp * np.exp(1j * phase)
        rows.append(
            [
                ts * 1000,
                frame,
                float(distances[target_idx]),
                float(np.max(amp)),
                float(np.angle(iq[target_idx])),
                float(np.mean(amp)),
                "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            ]
        )

    processor = A121LiveTraceProcessor()
    first = processor.process_rows(rows[:160], target_distance_m=float(distances[target_idx]), use_gating=True)
    first_resp = first.resp_signal.copy()
    second = processor.process_rows(rows[:220], target_distance_m=float(distances[target_idx]), use_gating=True)

    assert len(second.resp_signal) > len(first_resp)
    assert np.allclose(second.resp_signal[: len(first_resp)], first_resp)


def test_a121_real_recording_smoke_uses_bursty_timestamp_fix() -> None:
    path = Path("data/raw/a121/a121_sparse_iq_2026-05-31_21-18-57.csv")
    if not path.exists():
        pytest.skip("local A121 real recording fixture is not available")

    df = pd.read_csv(path).tail(2400)
    analysis = analyze_a121_vitals(df, max_frames=2400, use_gating=False)

    assert 35.0 <= analysis.sample_rate_hz <= 45.0
    assert analysis.present
    assert 45.0 <= analysis.heart_bpm <= 95.0
    assert np.isfinite(analysis.resp_signal).all()
    assert np.isfinite(analysis.heart_signal).all()


def test_a121_heart_confidence_can_lock_tracker() -> None:
    fs = 20.0
    t = np.arange(0, 60, 1 / fs)
    distances = np.linspace(0.2, 1.2, 81)
    target_idx = int(np.argmin(np.abs(distances - 0.62)))
    rows = []
    for frame, ts in enumerate(t):
        phase_motion = 0.25 * np.sin(2 * np.pi * 0.25 * ts) + 0.10 * np.sin(2 * np.pi * 1.2 * ts)
        amp = 30 * np.ones(len(distances))
        amp += 180 * np.exp(-0.5 * ((distances - distances[target_idx]) / 0.035) ** 2)
        phase = np.zeros(len(distances))
        phase[target_idx - 1 : target_idx + 2] = phase_motion
        iq = amp * np.exp(1j * phase)
        rows.append(
            {
                "Timestamp_ms": ts * 1000,
                "Frame": frame,
                "PeakDistance_m": float(distances[target_idx]),
                "PeakAmplitude": float(np.max(amp)),
                "PeakPhase_rad": float(np.angle(iq[target_idx])),
                "MeanAmplitude": float(np.mean(amp)),
                "Distances_m": "[" + ",".join(f"{x:.6f}" for x in distances) + "]",
                "Amplitude": "[" + ",".join(f"{x:.6f}" for x in np.abs(iq)) + "]",
                "Phase": "[" + ",".join(f"{x:.6f}" for x in np.angle(iq)) + "]",
                "Real": "[" + ",".join(f"{x:.6f}" for x in np.real(iq)) + "]",
                "Imag": "[" + ",".join(f"{x:.6f}" for x in np.imag(iq)) + "]",
            }
        )
    analysis = analyze_a121_vitals(pd.DataFrame(rows, columns=A121_COLUMNS))
    tracker = HeartRateKalmanTracker()
    tracked = tracker.update(analysis.heart_hz, 0.25, confidence=analysis.heart_confidence, quality=analysis.signal_quality)

    assert 68.0 <= analysis.heart_bpm <= 76.0
    assert analysis.heart_confidence >= 1.3
    assert 68.0 <= tracked * 60.0 <= 76.0


# Synthetic sparse-IQ recordings for the heart-rate tests below. One helper builds the rows that
# the A121 tests above write out by hand.
A121_TEST_DISTANCES = np.linspace(0.25, 1.05, 81)


def _json_list(values: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in values) + "]"


def make_a121_recording(
    *,
    seed: int = 1,
    kind: str = "person",
    duration_s: float = 60.0,
    fs: float = 20.0,
    resp_bpm: float = 14.0,
    heart_bpm: float = 63.0,
    target_m: float = 0.60,
    heart_offset_m: float = 0.10,
    heart_amp: float = 0.12,
    heart_in_target: bool = False,
    harmonic: int = 5,
    harmonic_amp: float = 0.20,
    phase_noise: float = 0.02,
    static_clutter: float = 0.0,
    reflector_offset_m: float | None = None,
    timestamps: str = "regular",
    scale: float = 1.0,
) -> pd.DataFrame:
    """Build a synthetic sparse-IQ recording in the CSV row format.

    kind="person" has a chest return at target_m that moves with breathing, a breathing harmonic
    in the central chest bins and heart motion in its own bins heart_offset_m away (or in the
    central bins when heart_in_target is set). kind="static" keeps the same returns without any
    motion and kind="noise" is receiver noise only. scale multiplies every IQ value.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    d = A121_TEST_DISTANCES
    target_idx = int(np.argmin(np.abs(d - target_m)))
    heart_idx = int(np.argmin(np.abs(d - (target_m + heart_offset_m))))
    ts_ms = t * 1000.0
    if timestamps == "jitter":
        steps = np.maximum(5.0, 1000.0 / fs + rng.normal(0.0, 6.0, n))
        ts_ms = np.cumsum(steps) - steps[0]
    elif timestamps == "bursty":
        frames = np.arange(n)
        ts_ms = np.floor(frames / 3) * 3000.0 / fs + (frames % 3) * 0.15

    base_phase = rng.uniform(-np.pi, np.pi, len(d))
    amp = 18.0 + 2.0 * rng.random(len(d))
    chest = np.exp(-0.5 * ((d - d[target_idx]) / 0.035) ** 2)
    heart_shape = np.exp(-0.5 * ((d - d[heart_idx]) / 0.018) ** 2)
    if kind != "noise":
        amp += 180.0 * chest + 65.0 * heart_shape
    static = np.zeros(len(d), dtype=complex)
    if static_clutter:
        static += static_clutter * np.exp(1j * rng.uniform(-np.pi, np.pi, len(d)))
    if reflector_offset_m is not None:
        reflector = np.exp(-0.5 * ((d - (target_m + reflector_offset_m)) / 0.018) ** 2)
        static += 400.0 * reflector * np.exp(1j * rng.uniform(-np.pi, np.pi))
    torso = np.exp(-0.5 * ((d - d[target_idx]) / 0.045) ** 2)
    center = np.exp(-0.5 * ((d - d[target_idx]) / 0.020) ** 2)
    heart_bins = center if heart_in_target else heart_shape
    resp_hz = resp_bpm / 60.0
    heart_hz = heart_bpm / 60.0

    rows = []
    for i in range(n):
        if kind == "noise":
            iq = 18.0 * (rng.normal(size=len(d)) + 1j * rng.normal(size=len(d))) / np.sqrt(2.0)
        else:
            phase = base_phase + phase_noise * rng.normal(size=len(d))
            if kind == "person":
                phase = phase + 0.30 * torso * np.sin(2 * np.pi * resp_hz * t[i])
                phase = phase + harmonic_amp * center * np.sin(2 * np.pi * harmonic * resp_hz * t[i] + 0.7)
                phase = phase + heart_amp * heart_bins * np.sin(2 * np.pi * heart_hz * t[i] + 1.1)
            iq = amp * (1.0 + 0.005 * rng.normal(size=len(d))) * np.exp(1j * phase) + static
        iq = iq * scale
        mag = np.abs(iq)
        peak = int(np.argmax(mag))
        rows.append(
            {
                "Timestamp_ms": float(ts_ms[i]),
                "Frame": i,
                "PeakDistance_m": float(d[peak]),
                "PeakAmplitude": float(mag[peak]),
                "PeakPhase_rad": float(np.angle(iq[peak])),
                "MeanAmplitude": float(np.mean(mag)),
                "Distances_m": _json_list(d),
                "Amplitude": _json_list(mag),
                "Phase": _json_list(np.angle(iq)),
                "Real": _json_list(np.real(iq)),
                "Imag": _json_list(np.imag(iq)),
            }
        )
    return pd.DataFrame(rows, columns=A121_COLUMNS)


@pytest.mark.parametrize("heart_offset_m", [0.10, -0.10])
def test_a121_heart_10cm_from_target_ignores_stronger_breathing_harmonic(heart_offset_m: float) -> None:
    # Breathing at 14/min puts its 5th harmonic at 70/min, and it is stronger than the heartbeat.
    analysis = analyze_a121_vitals(make_a121_recording(heart_offset_m=heart_offset_m))

    assert analysis.present
    assert abs(analysis.resp_bpm - 14.0) <= 1.0
    assert abs(analysis.heart_bpm - 63.0) <= 2.0
    assert abs(analysis.target_distance_m - 0.60) <= 0.03
    assert analysis.heart_confidence > 0.0


def test_a121_heart_close_to_breathing_harmonic_is_kept() -> None:
    # 67/min is only 3/min from the 70/min harmonic. A notch or blanking rule would remove it.
    analysis = analyze_a121_vitals(make_a121_recording(heart_bpm=67.0))

    assert abs(analysis.heart_bpm - 67.0) <= 1.5


def test_a121_heart_with_stronger_harmonic_in_same_bins_and_matching_heart_signal() -> None:
    # The heartbeat shares the chest bins with a stronger 5th harmonic at 66/min.
    analysis = analyze_a121_vitals(
        make_a121_recording(resp_bpm=13.2, heart_bpm=73.2, heart_in_target=True, harmonic_amp=0.18)
    )

    assert abs(analysis.heart_bpm - 73.2) <= 2.0
    assert len(analysis.heart_signal) == len(analysis.times_s)
    signal = analysis.heart_signal - np.mean(analysis.heart_signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / analysis.sample_rate_hz)
    band = (freqs >= 0.7) & (freqs <= 2.0)
    dominant_bpm = float(freqs[band][int(np.argmax(spectrum[band]))] * 60.0)
    assert abs(dominant_bpm - analysis.heart_bpm) <= 4.0


def test_a121_breathing_without_heartbeat_reports_no_heart_rate() -> None:
    analysis = analyze_a121_vitals(make_a121_recording(heart_amp=0.0, harmonic_amp=0.30))

    assert abs(analysis.resp_bpm - 14.0) <= 1.0
    assert analysis.heart_bpm == 0.0
    assert analysis.heart_confidence == 0.0


def test_a121_heart_search_reaches_bins_outside_the_gate() -> None:
    analysis = analyze_a121_vitals(make_a121_recording(heart_bpm=72.0, heart_offset_m=0.11))

    assert abs(analysis.heart_bpm - 72.0) <= 1.0
    assert abs(analysis.target_distance_m - 0.60) <= 0.03
    assert analysis.gate_min_m <= 0.60 <= analysis.gate_max_m


def test_a121_heart_with_strong_static_clutter_and_reflector() -> None:
    analysis = analyze_a121_vitals(make_a121_recording(static_clutter=250.0, reflector_offset_m=0.18))

    assert abs(analysis.resp_bpm - 14.0) <= 1.0
    assert abs(analysis.heart_bpm - 63.0) <= 2.0
    assert abs(analysis.target_distance_m - 0.60) <= 0.03


@pytest.mark.parametrize(
    ("timestamps", "resp_bpm", "heart_bpm", "harmonic"),
    [("jitter", 14.0, 63.0, 5), ("bursty", 15.0, 67.2, 4)],
)
def test_a121_heart_with_uneven_timestamps(timestamps: str, resp_bpm: float, heart_bpm: float, harmonic: int) -> None:
    analysis = analyze_a121_vitals(
        make_a121_recording(timestamps=timestamps, resp_bpm=resp_bpm, heart_bpm=heart_bpm, harmonic=harmonic)
    )

    assert abs(analysis.resp_bpm - resp_bpm) <= 1.0
    assert abs(analysis.heart_bpm - heart_bpm) <= 2.0


def test_a121_rates_do_not_depend_on_iq_amplitude_scale() -> None:
    reference = analyze_a121_vitals(make_a121_recording())

    for scale in (0.02, 20.0):
        scaled = analyze_a121_vitals(make_a121_recording(scale=scale))
        assert scaled.resp_bpm == pytest.approx(reference.resp_bpm, abs=0.1)
        assert scaled.heart_bpm == pytest.approx(reference.heart_bpm, abs=0.1)
        assert scaled.heart_bpm > 0.0


def test_a121_heart_prior_arguments_still_work() -> None:
    analysis = analyze_a121_vitals(make_a121_recording(), heart_prior_hz=63.0 / 60.0, heart_prior_std_hz=0.05)

    assert abs(analysis.heart_bpm - 63.0) <= 2.0


@pytest.mark.parametrize(
    ("kind", "seed", "extra"),
    [
        ("static", 1, {"static_clutter": 250.0, "reflector_offset_m": 0.18}),
        ("static", 10, {}),
        ("static", 10, {"scale": 0.02}),
        ("noise", 7, {}),
        ("noise", 9, {}),
    ],
)
def test_a121_static_scene_and_noise_report_no_rates(kind: str, seed: int, extra: dict) -> None:
    analysis = analyze_a121_vitals(make_a121_recording(kind=kind, seed=seed, **extra))

    assert analysis.resp_bpm == 0.0
    assert analysis.heart_bpm == 0.0
    assert analysis.resp_confidence == 0.0
    assert analysis.heart_confidence == 0.0


def test_a121_short_recording_reports_no_rates() -> None:
    analysis = analyze_a121_vitals(make_a121_recording(duration_s=8.0))

    assert analysis.resp_bpm == 0.0
    assert analysis.heart_bpm == 0.0
    assert analysis.heart_confidence == 0.0
