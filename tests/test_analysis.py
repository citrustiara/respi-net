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
