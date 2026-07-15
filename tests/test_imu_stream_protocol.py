import struct

import numpy as np
import pytest

from respi_net.imu import BreathCapture, summarize_lsm6ds3_capture_rows
from respi_net.iphone_imu import IPhoneIMUBluetoothCapture


def test_lsm6ds3_timestamped_uart_preserves_device_clock_and_counts_gaps() -> None:
    capture = BreathCapture()

    capture._process_line("10,1000000,0.1000,0.2000,1.0000,1.000,2.000,3.000", host_time_ms=5000.0)
    capture._process_line("12,1020000,0.1100,0.2100,1.0100,1.100,2.100,3.100", host_time_ms=5021.0)

    assert capture.snapshot_data_storage()[0][0] == pytest.approx(5000.0)
    full_rows = capture.snapshot_capture_storage()
    assert full_rows[1][2:4] == [1_020_000.0, 12.0]
    diagnostics = capture.diagnostics()
    assert diagnostics["format"] == "timestamped"
    assert diagnostics["missing_samples"] == 1
    assert diagnostics["device_sample_rate_hz"] == pytest.approx(50.0)


def test_lsm6ds3_legacy_uart_is_kept_but_marked_as_legacy() -> None:
    capture = BreathCapture()
    capture._process_line("0.1000,0.2000,1.0000,1.000,2.000,3.000", host_time_ms=5000.0)

    row = capture.snapshot_capture_storage()[0]
    assert row[0:2] == [5000.0, 5000.0]
    assert np.isnan(row[2]) and np.isnan(row[3])
    assert capture.diagnostics()["format"] == "legacy"


def test_lsm6ds3_counter_wrap_is_not_reported_as_a_loss() -> None:
    rows = [
        [1000.0, 1000.0, 1_000_000.0, 4_294_967_295.0, 0, 0, 1, 0, 0, 0],
        [1010.0, 1010.0, 1_010_000.0, 0.0, 0, 0, 1, 0, 0, 0],
    ]

    diagnostics = summarize_lsm6ds3_capture_rows(rows)

    assert diagnostics["missing_samples"] == 0
    assert diagnostics["counter_resets"] == 0


def test_iphone_capture_counts_missing_and_duplicate_ble_batches() -> None:
    capture = IPhoneIMUBluetoothCapture()

    def packet(sequence: int, time_ms: int) -> bytearray:
        return bytearray(struct.pack("<BBHIhhhhhh", 1, 1, sequence, time_ms, 1000, 0, 1000, 0, 0, 0))

    capture._on_notification(None, packet(4, 10))
    capture._on_notification(None, packet(6, 30))
    capture._on_notification(None, packet(6, 30))

    counters = capture.counter_snapshot()
    assert capture.data_count() == 2
    assert counters["missing_batches"] == 1
    assert counters["dropped_batches"] == 1
    assert counters["duplicate_batches"] == 1
