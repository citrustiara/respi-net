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


def _strap_rows(seconds, rate, extra_beat_after_s=None):
    rows = []
    for second in range(seconds):
        rr = [60000.0 / rate]
        if extra_beat_after_s is not None and second >= extra_beat_after_s:
            rr = [320.0, 60000.0 / rate - 320.0]  # a spurious beat splits the interval
        rows.append([1.79e12 + second * 1000.0, rate, ";".join(f"{value:.1f}" for value in rr), 1, "", ""])
    return rows


def test_heart_strap_quality_trusts_a_clean_strap():
    from respi_net.heart_strap import heart_strap_quality

    quality = heart_strap_quality(_strap_rows(120, 60))
    assert quality["hr_ok"].all() and quality["rr_ok"].all()
    assert (quality["rr_artifact_share"] == 0).all()


def test_heart_strap_quality_flags_spurious_beats_and_lost_contact():
    from respi_net.heart_strap import heart_strap_quality, heart_strap_summary

    rows = _strap_rows(240, 60, extra_beat_after_s=120)
    rows[10][3] = 0  # skin contact lost
    quality = heart_strap_quality(rows)
    times_s = (quality["host_time_ms"] - quality["host_time_ms"][0]) / 1000.0
    assert quality["hr_ok"][times_s < 80].drop(index=10).all() and not quality["hr_ok"][10]
    assert not quality["hr_ok"][times_s > 160].any()  # every interval is an artefact there
    assert not quality["rr_ok"][times_s > 120].any()
    summary = heart_strap_summary(rows)
    assert 0.3 < summary["hr_ok_share"] < 0.7 and summary["rr_ok_share"] < summary["hr_ok_share"]


def test_heart_strap_quality_sidecar(tmp_path):
    from respi_net.heart_strap import write_heart_strap_csv, write_heart_strap_quality

    recording = tmp_path / "heart_strap_test.csv"
    write_heart_strap_csv(recording, _strap_rows(30, 70))
    path, quality = write_heart_strap_quality(recording)
    assert path.name == "heart_strap_test_quality.csv" and path.exists() and len(quality) == 30
