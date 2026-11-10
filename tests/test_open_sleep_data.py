import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _write_edf(path: Path, signals: dict[str, np.ndarray], fs: int, record_s: int = 1) -> None:
    ns, n_records = len(signals), len(next(iter(signals.values()))) // (fs * record_s)
    field = lambda value, width: str(value).ljust(width)[:width].encode("ascii")
    header = field("0", 8) + field("X", 80) + field("X", 80) + field("01.01.15", 8) + field("12.00.00", 8)
    header += field(256 * (ns + 1), 8) + field("", 44) + field(n_records, 8) + field(record_s, 8) + field(ns, 4)
    names = list(signals)
    for width, values in ((16, names), (80, [""] * ns), (8, ["mV"] * ns), (8, ["-100"] * ns), (8, ["100"] * ns),
                          (8, ["-32768"] * ns), (8, ["32767"] * ns), (80, [""] * ns), (8, [fs * record_s] * ns), (32, [""] * ns)):
        header += b"".join(field(v, width) for v in values)
    data = b""
    for r in range(n_records):
        for name in names:
            chunk = signals[name][r * fs * record_s : (r + 1) * fs * record_s]
            digital = np.round((chunk + 100) / 200 * 65535 - 32768).astype("<i2")
            data += digital.tobytes()
    path.write_bytes(header + data)


def test_edf_reader_returns_physical_values(tmp_path) -> None:
    from build_open_sleep_epochs import read_edf

    t = np.arange(200) / 50
    _write_edf(tmp_path / "x.edf", {"S1": 50 * np.sin(t), "S2": -20 * np.ones(200)}, fs=50)
    signals, rates = read_edf(tmp_path / "x.edf")
    assert rates == {"S1": 50.0, "S2": 50.0}
    np.testing.assert_allclose(signals["S1"], 50 * np.sin(t), atol=0.01)
    np.testing.assert_allclose(signals["S2"], -20, atol=0.01)


def test_wfdb_annotations_carry_the_stage_text(tmp_path) -> None:
    from build_open_sleep_epochs import read_wfdb_annotations

    data = b""
    for k, stage in enumerate(("W", "2", "R")):
        if k:
            data += struct.pack("<H", 59 << 10) + struct.pack("<HH", 0, 1500)  # SKIP 1500 samples
        data += struct.pack("<H", (1 << 10) | 0)  # NORMAL at the current sample
        data += struct.pack("<H", (63 << 10) | 1) + stage.encode() + b"\x00"  # AUX, padded to even
    data += struct.pack("<H", 0)
    (tmp_path / "a.atr").write_bytes(data)
    assert read_wfdb_annotations(tmp_path / "a.atr") == [(0, "W"), (1500, "2"), (3000, "R")]


def test_bidsleep_night_finds_the_local_time_offset() -> None:
    from build_open_sleep_epochs import _bidsleep_night

    start_utc = pd.Timestamp("2021-12-03 04:00:00").timestamp()  # recStart 23:00 local, 5 h behind UTC
    t = start_utc + np.arange(0, 3600, 0.1)
    acc = pd.DataFrame({"Timestamp": t, "x": 0.01 * np.random.default_rng(0).standard_normal(len(t)), "y": 0.0, "z": -1.0})
    hr = pd.DataFrame({"t": start_utc + np.arange(0, 3600, 5.0) - 7200, "hr": 60.0})  # heart rate started 2 h early
    labels = {"expert_label": np.array([[0] * 40 + [2] * 80]), "recStart": np.array(["2021-12-02 23:00:00"])}
    night = _bidsleep_night(hr, acc, labels, "x")
    assert night["motion"].notna().mean() > 0.95
    assert list(night["stage"][:2]) == ["awake", "awake"] and night["stage"].iloc[-1] == "light"


def test_radar_circle_demodulation_recovers_displacement() -> None:
    from build_radar_ecg_windows import WAVELENGTH_MM, displacement_mm

    t = np.arange(4000) / 200
    disp = 3.0 * np.sin(2 * np.pi * 0.25 * t)
    phase = 4 * np.pi * disp / WAVELENGTH_MM
    i, q = 5.0 + 2.0 * np.cos(phase), -3.0 + 2.0 * np.sin(phase)  # offset centre, as a CW receiver adds
    out = displacement_mm(i, q)
    np.testing.assert_allclose(out - out.mean(), disp - disp.mean(), atol=0.05)


def test_lag_and_hold_counting_of_the_external_test() -> None:
    from breath_external_test import FS, HOLD, holds_found, lag_samples

    t = np.arange(int(60 * FS)) / FS
    radar = np.sin(2 * np.pi * 0.25 * t)
    reference = np.roll(radar, int(0.5 * FS))  # the reference lags by 0.5 s
    assert lag_samples(radar, reference) == int(0.5 * FS)
    y = np.zeros(int(30 * FS), int); y[int(5 * FS) : int(15 * FS)] = HOLD
    p = np.zeros_like(y); p[int(6 * FS) : int(14 * FS)] = HOLD
    assert holds_found(y, p) == (1, 1)
    assert holds_found(y, np.zeros_like(y)) == (1, 0)


def test_apnea_detector_finds_a_breathing_stop_and_scores_it() -> None:
    from apnea_feasibility import detect_events, envelope, inject, match

    fs = 20.0
    t = np.arange(int(1200 * fs)) / fs
    rng = np.random.default_rng(1)
    chest = 4.0 * np.sin(2 * np.pi * 0.25 * t) + 0.05 * rng.standard_normal(len(t))
    quiet = detect_events(envelope(chest, fs))
    assert quiet == []  # steady breathing: no event
    held = inject(chest, fs, 600.0, 20.0, 0.0)  # 20 s of no breathing
    events = detect_events(envelope(held, fs))
    found, true = match(events, [{"start": 600.0, "end": 620.0}])
    assert found.all() and true.all()
    assert events[0]["kind"] == "apnea" and 15 <= events[0]["end"] - events[0]["start"] <= 30
