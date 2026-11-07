import json

from respi_net.strain_belt import parse_strain_belt_packet, strain_belt_checksum


def test_json_packet_with_valid_checksum():
    payload = {"seq": 7, "raw": 1234.5, "ts": 70123}
    payload["checksum"] = strain_belt_checksum(7, 1234.5, 70123)
    packet = parse_strain_belt_packet(json.dumps(payload).encode())
    assert packet.seq == 7 and packet.raw == 1234.5 and packet.ts == 70123
    assert packet.checksum_ok is True


def test_json_packet_with_bad_checksum():
    packet = parse_strain_belt_packet(b'{"seq":1,"raw":2.0,"ts":3,"checksum":"deadbeef"}')
    assert packet.checksum_ok is False


def test_plain_number_payload():
    packet = parse_strain_belt_packet(b" -15.25\r\n")
    assert packet.raw == -15.25 and packet.seq is None and packet.checksum_ok is None


def _belt_pattern():
    from respi_net.breathing import load_pattern

    return load_pattern("belt_holds_1m")


def test_belt_pattern_is_one_minute_with_top_and_bottom_holds():
    from respi_net.strain_belt import phase_label

    pattern = _belt_pattern()
    assert pattern.duration_s == 60
    labels = [phase_label(pattern.phases, i) for i in range(len(pattern.phases))]
    assert labels.count("hold_in") == 2 and labels.count("hold_out") == 2


def test_label_coach_rows_skips_pause_and_shifts_elapsed():
    from respi_net.strain_belt import label_coach_rows

    pattern = _belt_pattern()
    start = 1000.0
    # Paused from 5 s to 15 s of wall time; samples every second for 30 s.
    rows = [[(start + t) * 1000.0, "", "", float(t), "", str(t)] for t in range(-2, 30)]
    labelled = label_coach_rows(rows, pattern, start, start + 25.0, pauses=[(start + 5.0, start + 15.0)])
    elapsed = [row[6] for row in labelled]
    assert elapsed[:5] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert elapsed[5] == 5.0  # wall 15 s -> run 5 s after the 10 s pause
    assert labelled[-1][3] == 25.0 and labelled[-1][6] == 15.0
    assert labelled[0][8] == "normal" and labelled[-1][8] in {"inhale", "exhale"}


def test_capture_drops_hx711_error_value():
    from respi_net.strain_belt import StrainBeltCapture

    capture = StrainBeltCapture()
    for payload in (b"-5312279", b"-1", b"-5312753"):
        capture._on_notification(None, bytearray(payload))
    assert [row[3] for row in capture.snapshot_rows()] == [-5312279.0, -5312753.0]
    assert capture.glitches == 1


def _synthetic_belt_csv(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    fs = 10.0
    t = np.arange(0, 60, 1 / fs)
    # Breaths of depth 6000 with a 4 s hold at the bottom every third breath, on a creeping belt.
    volume = 3000 * (1 - np.cos(2 * np.pi * t / 5))
    volume[(t % 15) > 11] = 0.0
    raw = -5.3e6 + volume + 150 * t + rng.normal(0, 40, len(t))
    raw[100] = -1  # failed HX711 read
    raw[300] += 40000  # glitch
    host_ms = (1.79e12 + t * 1000 + rng.uniform(0, 120, len(t))).round(1)
    path = tmp_path / "strain_belt_test.csv"
    pd.DataFrame({"host_time_ms": np.sort(host_ms), "raw": raw}).to_csv(path, index=False)
    return path, t, volume


def test_belt_chest_signal_removes_errors_spikes_and_drift(tmp_path):
    import numpy as np

    from respi_net.chest_signal import strain_belt_chest_signal

    path, t, volume = _synthetic_belt_csv(tmp_path)
    signal = strain_belt_chest_signal(path)
    assert signal.meta["hx711_errors"] == 1 and signal.meta["spikes"] >= 1
    assert abs(signal.fs - 10.0) < 0.05
    assert signal.meta["drift_counts"] > 6000  # ~150 counts/s over a minute was found ...
    truth = np.interp(signal.time_s, t, volume)
    assert np.max(np.abs(signal.chest - truth)[20:-20]) < 900  # ... and taken off, without the spike
    linear = strain_belt_chest_signal(path, drift="none")
    assert np.ptp(linear.chest) > np.ptp(signal.chest) + 5000


def test_label_tool_reads_a_radar_session_belt_file(tmp_path):
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import label_strain_belt_runs as labeller

    from respi_net.breathing import load_pattern

    pattern = load_pattern("belt_holds_1m")
    start_ms = 1.79e12
    session = tmp_path / "a121_belt" / "a121_belt_test"
    session.mkdir(parents=True)
    pd.DataFrame(
        [
            {"start_s": p.start_s, "end_s": p.end_s, "kind": p.kind, "cue": p.cue, "detail": p.detail,
             "section": p.section, "start_wall_ms": start_ms + p.start_s * 1000, "end_wall_ms": start_ms + p.end_s * 1000}
            for p in pattern.phases
        ]
    ).to_csv(session / "run_01_belt_holds_1m_cues.csv", index=False)
    # The subject follows each cue 0.3 s late on a creeping belt.
    t = np.arange(-0.5, 61.0, 0.1)
    volume = np.zeros_like(t)
    for i, p in enumerate(pattern.phases):
        inside = (t - 0.3 >= p.start_s) & (t - 0.3 < p.end_s)
        u = (t[inside] - 0.3 - p.start_s) / p.duration_s
        level = {"inhale": u, "exhale": 1 - u}.get(p.kind)
        if level is None:
            level = 1.0 if p.kind == "hold" and pattern.phases[i - 1].kind == "inhale" else 0.0
        volume[inside] = level
    raw = -5.3e6 + 7000 * volume + 120 * t + np.random.default_rng(2).normal(0, 30, len(t))
    pd.DataFrame({"host_time_ms": start_ms + t * 1000, "seq": "", "device_ts": "", "raw": raw, "checksum_ok": "", "payload": ""}).to_csv(
        session / "run_01_belt_holds_1m_belt.csv", index=False
    )

    row = labeller.label_run(session / "run_01_belt_holds_1m_belt.csv", tmp_path / "out", labeller.BELT_MIN_HOLD_S)

    assert row["run"] == "a121_belt_test__run_01_belt_holds_1m_belt" and row["status"] == "labelled"
    assert abs(row["cue_lag_s"] - 0.3) < 0.15
    assert row["cue_agreement"] > 0.85
    labels = pd.read_csv(tmp_path / "out" / "labels" / "a121_belt_test__run_01_belt_holds_1m_belt.csv")
    assert set(labels["label"]) >= {0, 1, 2} and 3 not in set(labels["label"])  # one hold
    assert (tmp_path / "out" / "plots" / "a121_belt_test__run_01_belt_holds_1m_belt.png").exists()


def test_strain_belt_gaps_finds_only_real_dropouts():
    from respi_net.strain_belt import strain_belt_gaps

    stamps = [0, 100, 450, 550, 9550, 9650]  # a 0.35 s BLE burst gap is normal, 9 s is a dropout
    assert strain_belt_gaps(stamps) == [(550.0, 9550.0)]
    assert strain_belt_gaps(stamps[:2]) == []


def test_belt_chest_signal_leaves_a_dropout_out(tmp_path):
    import numpy as np
    import pandas as pd

    from respi_net.chest_signal import strain_belt_chest_signal

    path, t, volume = _synthetic_belt_csv(tmp_path)
    frame = pd.read_csv(path)
    lost = (frame.host_time_ms - frame.host_time_ms[0] > 20_000) & (frame.host_time_ms - frame.host_time_ms[0] < 35_000)
    frame[~lost].to_csv(path, index=False)
    signal = strain_belt_chest_signal(path)
    assert signal.meta["gaps"] == 1 and 14 < signal.meta["gap_s"] < 16
    assert abs(signal.fs - 10.0) < 0.05  # the rate comes from the stretches with data, not the span
    valid = signal.valid_mask()
    assert not valid[(signal.time_s > 19) & (signal.time_s < 36)].any()
    assert valid[(signal.time_s > 5) & (signal.time_s < 15)].all() and valid[signal.time_s > 40].all()
    truth = np.interp(signal.time_s, t, volume)
    assert np.max(np.abs(signal.chest - truth)[valid][20:-20]) < 900


def test_capture_reconnects_after_the_link_drops(monkeypatch, tmp_path):
    import asyncio

    from respi_net import strain_belt

    sessions = []

    class FakeScanner:
        @staticmethod
        async def find_device_by_filter(matches, timeout):
            return type("Device", (), {"name": strain_belt.STRAIN_BELT_DEVICE_NAME, "address": "AA"})()

    class FakeClient:
        def __init__(self, device):
            self.is_connected = False

        async def __aenter__(self):
            self.is_connected = True
            sessions.append(self)
            return self

        async def __aexit__(self, *exc):
            self.is_connected = False

        async def start_notify(self, uuid, callback):
            async def stream():
                for value in range(5):
                    callback(None, bytearray(str(-5_300_000 + value).encode()))
                    await asyncio.sleep(0.01)
                if len(sessions) == 1:
                    self.is_connected = False  # the first link falls
            self.task = asyncio.ensure_future(stream())

        async def stop_notify(self, uuid):
            pass

    monkeypatch.setattr(strain_belt, "_load_bleak", lambda: (FakeScanner, FakeClient))
    monkeypatch.setattr(strain_belt, "STRAIN_BELT_RECONNECT_PAUSE_S", 0.05)
    capture = strain_belt.StrainBeltCapture(output_dir=tmp_path)
    assert capture.connect()
    for _ in range(100):
        if len(sessions) >= 2 and capture.data_count() >= 10:
            break
        import time
        time.sleep(0.02)
    capture.stop()
    assert len(sessions) == 2 and capture.disconnects == 1
    assert capture.data_count() == 10
    assert "disconnects 1" in strain_belt.strain_belt_gap_summary(capture)


def test_capture_retries_a_failed_first_connection(monkeypatch, tmp_path):
    import time

    from respi_net import strain_belt

    tries = []

    class FakeScanner:
        @staticmethod
        async def find_device_by_filter(matches, timeout):
            return type("Device", (), {"name": strain_belt.STRAIN_BELT_DEVICE_NAME, "address": "AA"})()

    class FakeClient:
        def __init__(self, device):
            self.is_connected = False

        async def __aenter__(self):
            tries.append(1)
            if len(tries) == 1:
                raise RuntimeError("disconnected")  # what macOS reports at a weak signal
            self.is_connected = True
            return self

        async def __aexit__(self, *exc):
            self.is_connected = False

        async def start_notify(self, uuid, callback):
            callback(None, bytearray(b"-5300000"))

        async def stop_notify(self, uuid):
            pass

    monkeypatch.setattr(strain_belt, "_load_bleak", lambda: (FakeScanner, FakeClient))
    monkeypatch.setattr(strain_belt, "STRAIN_BELT_RECONNECT_PAUSE_S", 0.01)
    capture = strain_belt.StrainBeltCapture(output_dir=tmp_path)
    assert capture.connect()
    time.sleep(0.05)
    capture.stop()
    assert len(tries) == 2 and capture.data_count() == 1 and capture.disconnects == 0
