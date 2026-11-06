from __future__ import annotations

import pytest

from respi_net.heart_strap import heart_strap_row, heart_strap_summary, parse_heart_rate_measurement


def test_an_8_bit_rate_with_rr_intervals_and_contact() -> None:
    # flags 0x16: RR present, contact supported and detected; 75 bpm; RR 812 and 820 units of 1/1024 s
    measurement = parse_heart_rate_measurement(bytes([0x16, 75, 0x2C, 0x03, 0x34, 0x03]))
    assert measurement.heart_rate_bpm == 75
    assert measurement.rr_ms == pytest.approx((793.0, 800.8), abs=0.1)
    assert measurement.sensor_contact is True and measurement.energy_kj is None


def test_a_16_bit_rate_with_energy_and_no_contact_support() -> None:
    # flags 0x09: 16-bit rate, energy expended; 260 bpm, 1000 kJ
    measurement = parse_heart_rate_measurement(bytes([0x09, 0x04, 0x01, 0xE8, 0x03]))
    assert measurement.heart_rate_bpm == 260 and measurement.energy_kj == 1000
    assert measurement.rr_ms == () and measurement.sensor_contact is None


def test_lost_contact_is_reported_and_short_packets_are_refused() -> None:
    assert parse_heart_rate_measurement(bytes([0x04, 60])).sensor_contact is False
    with pytest.raises(ValueError):
        parse_heart_rate_measurement(bytes([0x01, 60]))


def test_rows_and_summary() -> None:
    rows = [
        heart_strap_row(1000.0 * (1 + second), parse_heart_rate_measurement(bytes([0x16, 70 + second % 3, 0x00, 0x04])), b"")
        for second in range(10)
    ]
    assert rows[0][2] == "1000.0" and rows[0][3] == 1
    summary = heart_strap_summary(rows, duration_s=10.0)
    assert summary["notifications"] == 10 and summary["rr_beats"] == 10
    assert summary["median_heart_rate_bpm"] == 71 and summary["largest_gap_s"] == 1.0
    assert summary["coverage_percent"] == 90.0 and summary["no_contact_share"] == 0.0
