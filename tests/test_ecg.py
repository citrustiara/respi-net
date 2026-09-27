"""Tests for the AD8232 ECG heart-rate pipeline (synthetic signals, no hardware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, sosfiltfilt

from respi_net.ecg import (
    detect_r_peaks,
    heart_rate_series,
    hr_labels_for_windows,
    load_ecg_csv,
    mechanical_beat_times,
    simulate_ecg,
)

NOISY = {"noise_mv": 0.05, "baseline_wander_mv": 0.5, "mains_mv": 0.1}


def _score(detected_s: np.ndarray, true_s: np.ndarray, tolerance_s: float = 0.075) -> tuple[float, np.ndarray]:
    """F1 against the true beats and signed timing errors of the matched beats.

    True beats are >= 0.3 s apart, so nearest-neighbour matching within 75 ms is one-to-one.
    """
    if len(detected_s) == 0:
        return 0.0, np.array([])
    offsets = detected_s[None, :] - true_s[:, None]
    nearest = offsets[np.arange(len(true_s)), np.argmin(np.abs(offsets), axis=1)]
    matched = np.abs(nearest) <= tolerance_s
    hits = int(matched.sum())
    precision, recall = hits / len(detected_s), hits / len(true_s)
    return (2 * precision * recall / (precision + recall) if hits else 0.0), nearest[matched]


@pytest.mark.parametrize(("hr_bpm", "fs"), [(50, 250.0), (80, 250.0), (120, 250.0), (80, 500.0)])
def test_detects_r_peaks_with_noise_and_baseline_wander(hr_bpm: int, fs: float) -> None:
    sim = simulate_ecg(60.0, fs, hr_bpm, np.random.default_rng(hr_bpm), **NOISY)
    assert 60.0 / np.mean(np.diff(sim.r_peak_times_s)) == pytest.approx(hr_bpm, rel=0.05)

    peaks = detect_r_peaks(sim.ecg_mv, sim.fs)
    f1, errors = _score(peaks / sim.fs, sim.r_peak_times_s)

    assert f1 >= 0.98
    assert np.median(np.abs(errors)) < 0.010


def test_detection_tolerates_heavier_noise_and_mains_hum() -> None:
    sim = simulate_ecg(60.0, 250.0, 70, np.random.default_rng(1), noise_mv=0.15, baseline_wander_mv=1.0, mains_mv=0.3)
    f1, errors = _score(detect_r_peaks(sim.ecg_mv, sim.fs) / sim.fs, sim.r_peak_times_s)

    assert f1 >= 0.98
    assert np.median(np.abs(errors)) < 0.010


def test_inverted_polarity_still_times_the_r_apex() -> None:
    upright = simulate_ecg(60.0, 250.0, 70, np.random.default_rng(3), **NOISY)
    inverted = simulate_ecg(60.0, 250.0, 70, np.random.default_rng(3), inverted=True, **NOISY)
    np.testing.assert_allclose(inverted.ecg_mv, -upright.ecg_mv)

    peaks = detect_r_peaks(inverted.ecg_mv, inverted.fs)
    f1, errors = _score(peaks / inverted.fs, inverted.r_peak_times_s)

    assert f1 >= 0.98
    # The S wave sits 30 ms after R: small errors mean the negative R apex was taken, not S.
    assert np.percentile(np.abs(errors), 95) < 0.010
    np.testing.assert_array_equal(peaks, detect_r_peaks(upright.ecg_mv, upright.fs))


def test_detection_recovers_after_a_motion_artifact() -> None:
    rng = np.random.default_rng(11)
    sim = simulate_ecg(60.0, 250.0, 75, rng, **NOISY)
    ecg = sim.ecg_mv.copy()
    burst = (sim.time_s >= 20.0) & (sim.time_s < 23.0)
    motion_band = butter(2, [1.0, 30.0], btype="bandpass", fs=sim.fs, output="sos")
    motion = sosfiltfilt(motion_band, rng.standard_normal(burst.sum()))
    ecg[burst] += 5.0 * motion / np.std(motion)  # 5 mV SD, five times the R wave

    detected = detect_r_peaks(ecg, sim.fs) / sim.fs
    f1_after, _ = _score(detected[detected > 24.0], sim.r_peak_times_s[sim.r_peak_times_s > 24.0])

    assert f1_after >= 0.98
    assert np.count_nonzero((detected > 20.0) & (detected < 23.0)) <= 3


def test_detect_r_peaks_checks_the_sampling_rate() -> None:
    sim = simulate_ecg(10.0, 200.0, 70, np.random.default_rng(0))
    with pytest.warns(UserWarning, match="250 Hz"):
        detect_r_peaks(sim.ecg_mv, sim.fs)
    with pytest.raises(ValueError, match="too low"):
        detect_r_peaks(sim.ecg_mv, 50.0)
    assert detect_r_peaks(np.zeros(100), 250.0).size == 0


def test_heart_rate_series_flags_missed_extra_and_implausible_beats() -> None:
    times = np.arange(50) * 0.8  # 75 bpm
    times = np.delete(times, [10, 35, 36, 37])  # a missed beat (1.6 s) and a 3.2 s dropout
    times = np.sort(np.append(times, 20.35))  # an extra beat splits 0.8 s into 0.35 + 0.45 s

    series = heart_rate_series(times)

    np.testing.assert_allclose(series.beat_times_s, times[1:])
    # The two 0.8 s intervals share a beat with the dropout, which may be an artifact.
    np.testing.assert_allclose(np.sort(series.rr_s[~series.valid]), [0.35, 0.45, 0.8, 0.8, 1.6, 3.2])
    np.testing.assert_allclose(series.hr_bpm[series.valid], 75.0)
    with pytest.raises(ValueError, match="increasing"):
        heart_rate_series(np.array([1.0, 0.5]))


def test_max_rr_change_sets_the_tolerated_jump() -> None:
    rr = np.full(20, 1.0)
    rr[10] = 1.2
    times = np.concatenate(([0.0], np.cumsum(rr)))

    assert heart_rate_series(times).valid.all()
    strict = heart_rate_series(times, max_rr_change=0.1)
    assert not strict.valid[10]
    assert strict.valid[np.arange(20) != 10].all()
    assert not heart_rate_series(np.arange(10) * 0.25).valid.any()  # 240 bpm is implausible


def test_hr_labels_for_windows_use_valid_intervals_inside_each_window() -> None:
    slow = np.arange(0.0, 20.0, 1.0)  # 60 bpm for 20 s
    fast = 20.0 + np.arange(30) * (2.0 / 3.0)  # then 90 bpm
    times = np.concatenate([slow, fast])
    times = times[~np.isclose(times, 30.0)]  # missed beat: its 1.33 s interval is rejected

    labels = hr_labels_for_windows(times, np.array([0.0, 10.0, 25.0, 45.0]), 10.0)

    np.testing.assert_allclose(labels[:3], [60.0, 60.0, 90.0])
    assert np.isnan(labels[3])  # no beats there
    assert np.isnan(hr_labels_for_windows(times, np.array([0.0]), 3.5)[0])  # only 3 intervals fit
    assert hr_labels_for_windows(times, np.array([0.0]), 3.5, min_valid_beats=3)[0] == pytest.approx(60.0)


def test_hr_labels_need_valid_beats_spanning_enough_of_the_window() -> None:
    times = np.concatenate([np.arange(0.0, 9.0), np.arange(15.0, 21.0)])  # 60 bpm, 7 s dropout
    # 11 valid intervals remain (the two touching the dropout are rejected): 55% of 20 s.

    assert hr_labels_for_windows(times, np.array([0.0]), 20.0)[0] == pytest.approx(60.0)
    assert np.isnan(hr_labels_for_windows(times, np.array([0.0]), 20.0, min_coverage=0.75)[0])


def test_window_label_is_beats_per_minute_not_mean_instantaneous_hr() -> None:
    times = np.concatenate(([0.0], np.cumsum(np.tile([0.9, 1.1], 10))))  # mean RR 1.0 s

    label = hr_labels_for_windows(times, np.array([0.0]), 20.0)[0]

    assert label == pytest.approx(60.0)  # the mean of 60/RR would be 60.6


def test_mechanical_beat_times_shift_by_the_pre_ejection_period() -> None:
    peaks = np.array([1.0, 1.8, 2.7])

    np.testing.assert_allclose(mechanical_beat_times(peaks), peaks + 0.10)
    np.testing.assert_allclose(mechanical_beat_times(peaks, pre_ejection_s=0.08), peaks + 0.08)
    # A constant delay cancels in RR intervals, which is why windowed HR needs no shift.
    np.testing.assert_allclose(heart_rate_series(mechanical_beat_times(peaks)).rr_s, np.diff(peaks))
    with pytest.raises(ValueError, match="seconds"):
        mechanical_beat_times(peaks, pre_ejection_s=105.0)


def _firmware_rows(n: int, start_ms: int = 10_000) -> pd.DataFrame:
    voltage = 1650 + 10 * (np.arange(n) % 9)
    return pd.DataFrame(
        {
            "Timestamp_ms": start_ms + 4 * np.arange(n),
            "RawADC": np.round(voltage * 4095 / 3100).astype(int),
            "Voltage_mV": voltage,
        }
    )


def test_load_plain_firmware_csv(tmp_path: Path) -> None:
    frame = _firmware_rows(40)
    path = tmp_path / "ecg_plain.csv"
    frame.to_csv(path, index=False)

    recording = load_ecg_csv(path)

    assert recording.clock == "device"
    assert recording.start_epoch_s is None
    assert recording.fs == pytest.approx(250.0)
    np.testing.assert_allclose(recording.time_s, 0.004 * np.arange(40))
    np.testing.assert_allclose(recording.voltage_mv, frame["Voltage_mV"])


def test_load_host_timestamped_csv_removes_usb_delivery_jitter(tmp_path: Path) -> None:
    frame = _firmware_rows(40)
    # USB delivers lines in 16 ms bursts: arrival = offset + next flush after the device time.
    offset_ms = 1_784_146_261_000.0
    frame.insert(1, "HostTimestamp_ms", offset_ms + np.ceil(frame["Timestamp_ms"] / 16.0) * 16.0)
    path = tmp_path / "ecg_host.csv"
    frame.to_csv(path, index=False)

    recording = load_ecg_csv(path)

    assert recording.clock == "host"
    assert recording.fs == pytest.approx(250.0)
    np.testing.assert_allclose(np.diff(recording.time_s), 0.004, atol=1e-9)
    assert recording.start_epoch_s == pytest.approx((offset_ms + 10_000.0) / 1000.0, abs=1e-6)


def test_host_arrival_times_are_kept_when_the_device_clock_resets(tmp_path: Path) -> None:
    frame = _firmware_rows(40)
    frame.insert(1, "HostTimestamp_ms", 1.78e12 + 4.0 * np.arange(40))
    frame.loc[20:, "Timestamp_ms"] -= 10_000  # the ESP32 restarted mid-capture
    path = tmp_path / "ecg_reset.csv"
    frame.to_csv(path, index=False)

    recording = load_ecg_csv(path)

    assert recording.clock == "host-arrival"
    assert recording.fs == pytest.approx(250.0)


def test_load_headerless_serial_dump_skips_boot_log(tmp_path: Path) -> None:
    frame = _firmware_rows(30)
    lines = [
        "rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)",
        "I (312) RADAR_ADC: Calibration Success: Line Fitting",
        *(f"{t},{raw},{mv}" for t, raw, mv in frame.itertuples(index=False)),
    ]
    path = tmp_path / "ecg_serial.log"
    path.write_text("\n".join(lines) + "\n")

    recording = load_ecg_csv(path)

    assert recording.clock == "device"
    assert len(recording.voltage_mv) == 30
    assert recording.fs == pytest.approx(250.0)


def test_loader_rejects_unusable_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ecg_csv(tmp_path / "missing.csv")

    no_voltage = tmp_path / "no_voltage.csv"
    _firmware_rows(40).drop(columns="Voltage_mV").to_csv(no_voltage, index=False)
    with pytest.raises(ValueError, match="Voltage_mV"):
        load_ecg_csv(no_voltage)

    short = tmp_path / "short.csv"
    _firmware_rows(5).to_csv(short, index=False)
    with pytest.raises(ValueError, match="usable samples"):
        load_ecg_csv(short)

    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="usable samples"):
        load_ecg_csv(empty)

    reset = _firmware_rows(40)
    reset.loc[20:, "Timestamp_ms"] -= 10_000
    backwards = tmp_path / "backwards.csv"
    reset.to_csv(backwards, index=False)
    with pytest.raises(ValueError, match="backwards"):
        load_ecg_csv(backwards)


def test_end_to_end_from_host_timestamped_csv(tmp_path: Path) -> None:
    sim = simulate_ecg(40.0, 250.0, 72, np.random.default_rng(5), **NOISY)
    device_ms = 5_000 + 4 * np.arange(len(sim.ecg_mv))
    voltage = np.round(1650 + 800 * sim.ecg_mv)  # AD8232 output around mid-supply, integer mV
    path = tmp_path / "ecg_session.csv"
    pd.DataFrame(
        {
            "Timestamp_ms": device_ms,
            "HostTimestamp_ms": 1.78e12 + np.ceil(device_ms / 16.0) * 16.0,
            "RawADC": np.round(voltage * 4095 / 3100),
            "Voltage_mV": voltage,
        }
    ).to_csv(path, index=False)

    recording = load_ecg_csv(path)
    peak_times = recording.time_s[detect_r_peaks(recording.voltage_mv, recording.fs)]
    f1, errors = _score(peak_times, sim.r_peak_times_s)

    assert f1 >= 0.98
    assert np.median(np.abs(errors)) < 0.010
    starts = np.arange(0.0, 30.1, 10.0)
    np.testing.assert_allclose(
        hr_labels_for_windows(peak_times, starts, 10.0),
        hr_labels_for_windows(sim.r_peak_times_s, starts, 10.0),
        atol=0.5,
    )
