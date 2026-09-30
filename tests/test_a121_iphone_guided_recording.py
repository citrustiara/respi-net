import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402
from pathlib import Path  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import a121_iphone_guided_recording as tool  # noqa: E402
from respi_net.a121 import A121_CAPTURE_COLUMNS  # noqa: E402
from respi_net.breathing import PatternError, load_pattern, pattern_from_dict  # noqa: E402
from respi_net.breathing_coach import LUNGS_EMPTY, LUNGS_FULL, phase_shapes  # noqa: E402
from respi_net.imu import IMU_COLUMNS  # noqa: E402


NN_PATTERNS = ["nn_free", "nn_varied_rates", "nn_holds", "nn_shallow_deep", "nn_irregular"]


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


# -- fake sensors: rows appear as wall-clock time passes, stamped like the real captures --


def _now_ms() -> float:
    return time.time() * 1000.0


class FakeA121Capture:
    frame_rate_hz = 20.0
    backdate_ms = 1000.0  # frames already captured when connect() returns

    def __init__(self, output_dir: Any = None, config: Any = None, **_: Any) -> None:
        self.config = config
        self.running = False
        self.dropped_results = 0
        self.rows: list[list[Any]] = []
        self.sensor_config = SimpleNamespace(start_point=80, num_points=101, step_length=4, sweeps_per_frame=8)
        self.distances_m = np.linspace(0.2, 1.2, 101)
        self._t0: float | None = None
        self._stopped_at: float | None = None
        self._frame = 0

    def connect(self, port_name: str | None = None) -> bool:
        self.port = port_name
        self.running = True
        self._t0 = _now_ms() - self.backdate_ms
        return True

    def _sync(self) -> None:
        if self._t0 is None:
            return
        now = self._stopped_at if self._stopped_at is not None else _now_ms()
        period = 1000.0 / self.frame_rate_hz
        while self._t0 + self._frame * period <= now:
            values = {column: 0.0 for column in A121_CAPTURE_COLUMNS}
            values.update(
                {
                    "Timestamp_ms": self._t0 + self._frame * period,
                    "Frame": self._frame,
                    "PeakDistance_m": 0.8,
                    "Distances_m": "[0.8,0.81]",
                    "Amplitude": "[1000.0,900.0]",
                    "Phase": "[0.1,0.2]",
                    "Real": "[1.0,1.0]",
                    "Imag": "[0.0,0.0]",
                    "AcconeerAppState": "ESTIMATE_BREATHING_RATE",
                    "AcconeerPresenceDetected": 1,
                    "AcconeerTargetDistance_m": 0.79,
                }
            )
            self.rows.append([values[column] for column in A121_CAPTURE_COLUMNS])
            self._frame += 1

    def data_count(self) -> int:
        self._sync()
        return len(self.rows)

    def snapshot_data_since(self, index: int) -> list[list[Any]]:
        self._sync()
        return [list(row) for row in self.rows[index:]]

    def stop(self) -> None:
        if self._t0 is not None and self._stopped_at is None:
            self._sync()
            self._stopped_at = _now_ms()
        self.running = False


class SlowA121Capture(FakeA121Capture):
    frame_rate_hz = 4.0
    backdate_ms = 3000.0


class FakeIPhoneCapture:
    rate_hz = 100.0
    stream_for_s: float | None = None  # stop sending this long after connecting

    def __init__(self, output_dir: Any = None, device: Any = None, scan_timeout_s: float = 10.0, autostart: bool = True):
        self.running = False
        self.rows: list[list[float]] = []
        self._t0: float | None = None
        self._connected_at = 0.0
        self._stopped_at: float | None = None
        self._n = 0

    def connect(self, port_name: str | None = None) -> bool:
        self.running = True
        self._connected_at = _now_ms()
        self._t0 = self._connected_at - 500.0
        return True

    def _sync(self) -> None:
        if self._t0 is None:
            return
        limit = self._stopped_at if self._stopped_at is not None else _now_ms()
        if self.stream_for_s is not None:
            limit = min(limit, self._connected_at + self.stream_for_s * 1000.0)
        period = 1000.0 / self.rate_hz
        while self._t0 + self._n * period <= limit:
            self.rows.append([self._t0 + self._n * period, 0.01, 0.02, -1.0, 0.1, 0.2, 0.3])
            self._n += 1

    def data_count(self) -> int:
        self._sync()
        return len(self.rows)

    def snapshot_data_since(self, index: int) -> list[list[float]]:
        self._sync()
        return [list(row) for row in self.rows[index:]]

    def counter_snapshot(self) -> dict[str, int]:
        return {
            "received_batches": len(self.rows) // 10,
            "dropped_batches": 0,
            "invalid_batches": 0,
            "missing_batches": 0,
            "duplicate_batches": 0,
            "sequence_resets": 0,
        }

    def stop(self) -> None:
        if self._t0 is not None and self._stopped_at is None:
            self._sync()
            self._stopped_at = _now_ms()
        self.running = False


class StallingIPhoneCapture(FakeIPhoneCapture):
    stream_for_s = 0.5


def _quick_trial() -> "tool.Trial":
    pattern = pattern_from_dict(
        {
            "name": "Szybki test",
            "blocks": [
                {"kind": "normal", "duration_seconds": 0.3},
                {"kind": "paced", "cycles": 1, "inhale_seconds": 0.3, "hold_after_inhale_seconds": 0.2, "exhale_seconds": 0.3},
            ],
        },
        pattern_id="quick",
    )
    return tool.Trial(1, pattern)


def _run_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, a121: type, iphone: type, abort_on_progress: bool = False):
    monkeypatch.setattr(tool, "A121Capture", a121)
    monkeypatch.setattr(tool, "ArrivalLoggingIPhoneCapture", iphone)
    trial = _quick_trial()
    settings = tool.SessionSettings(session_dir=tmp_path, a121_port="TEST", prep_seconds=0.1, post_roll_s=0.1)
    worker = tool.MeasurementWorker(trial=trial, settings=settings)
    events: dict[str, list[Any]] = {"finished": [], "failed": [], "aborted": [], "prep": [], "progress": [], "status": []}
    worker.finished.connect(events["finished"].append)
    worker.failed.connect(events["failed"].append)
    worker.aborted.connect(lambda: events["aborted"].append(True))
    worker.prep_changed.connect(events["prep"].append)
    worker.progress_changed.connect(lambda elapsed, total: events["progress"].append((elapsed, total)))
    worker.status_changed.connect(events["status"].append)
    if abort_on_progress:
        worker.progress_changed.connect(lambda *_: worker.request_abort())
    # Leftovers of an earlier attempt at the same trial must never survive a new one.
    for path in worker.paths.values():
        path.write_text("stale", encoding="utf-8")
    worker.run()
    return trial, worker, events


# -- plan -------------------------------------------------------------------------------


def test_default_plan_is_the_nn_patterns_in_a_fixed_order() -> None:
    trials = tool.build_trials()

    assert [trial.pattern_id for trial in trials] == list(tool.DEFAULT_PATTERN_IDS) == NN_PATTERNS
    assert [trial.number for trial in trials] == [1, 2, 3, 4, 5]
    assert all(trial.duration_s == trial.pattern.duration_s for trial in trials)
    assert trials[2].as_dict()["pattern_source"] == "configs/breathing_patterns/nn_holds.json"


def test_patterns_option_takes_ids_and_paths() -> None:
    references = tool.parse_pattern_list(" nn_holds , configs/breathing_patterns/box_breathing.json,, ")
    trials = tool.build_trials(references)

    assert [trial.pattern_id for trial in trials] == ["nn_holds", "box_breathing"]
    assert trials[1].duration_s == load_pattern("box_breathing").duration_s
    with pytest.raises(PatternError, match="Available: .*nn_irregular"):
        tool.build_trials(["no_such_pattern"])
    with pytest.raises(ValueError, match="co najmniej jeden"):
        tool.parse_pattern_list(" , ")


def test_output_names_follow_run_number_and_pattern_id(tmp_path: Path) -> None:
    trial = tool.build_trials(["nn_free", "nn_holds"])[1]

    assert {key: path.name for key, path in tool.output_paths(tmp_path, trial).items()} == {
        "a121": "run_02_nn_holds_a121.csv",
        "iphone": "run_02_nn_holds_iphone.csv",
        "cues": "run_02_nn_holds_cues.csv",
    }
    odd = tool.Trial(12, pattern_from_dict({"blocks": [{"kind": "normal", "duration_seconds": 5}]}, pattern_id="my pattern/1"))
    assert tool.output_paths(tmp_path, odd)["cues"] == tmp_path / "run_12_my_pattern_1_cues.csv"


def test_a121_range_follows_the_distance_or_the_overrides() -> None:
    default = tool.a121_config_for()
    assert (default.start_m, default.end_m) == (0.2, 2.5)
    assert (default.profile, default.hwaas, default.sweeps_per_frame, default.frame_rate_hz) == (3, 32, 8, 20.0)

    at_2m = tool.a121_config_for(distance_cm=200)
    assert (at_2m.start_m, at_2m.end_m) == (1.5, 2.5)  # the 2 m retest's range
    assert (tool.a121_config_for(distance_cm=30).start_m, tool.a121_config_for(distance_cm=30).end_m) == (0.1, 0.8)
    overridden = tool.a121_config_for(distance_cm=100, end_m=1.3)
    assert (overridden.start_m, overridden.end_m) == (0.5, 1.3)
    with pytest.raises(ValueError, match="pusty"):
        tool.a121_config_for(start_m=1.0, end_m=0.9)


def test_dry_run_prints_the_plan_without_hardware(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert tool.main(["--dry-run", "--distance-cm", "120", "--posture", "Sitting", "--output-dir", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    for pattern_id in NN_PATTERNS:
        assert pattern_id in output
    total_s = sum(load_pattern(pattern_id).duration_s for pattern_id in NN_PATTERNS)
    assert f"Łącznie 5 prób: {int(total_s) // 60}:{int(total_s) % 60:02d} nagrań" in output
    assert "0.70–1.70 m" in output and "sitting (siedzenie)" in output
    assert list(tmp_path.iterdir()) == []  # nothing written

    with pytest.raises(SystemExit):
        tool.main(["--dry-run", "--patterns", "no_such_pattern"])
    with pytest.raises(SystemExit):
        tool.main(["--dry-run", "--start-trial", "6"])


def test_cue_rows_carry_pattern_and_host_wall_times() -> None:
    pattern = load_pattern("nn_holds")
    rows = tool.cue_rows(pattern, 1_000_000.0)

    assert len(rows) == len(pattern.phases)
    assert list(rows[0]) == tool.CUE_COLUMNS
    assert rows[5]["start_wall_ms"] == pytest.approx(1_000_000.0 + pattern.phases[5].start_s * 1000.0)
    assert rows[-1]["end_wall_ms"] == pytest.approx(1_000_000.0 + pattern.duration_s * 1000.0)
    assert [row["section"] for row in rows] == [phase.section for phase in pattern.phases]


# -- pattern files -------------------------------------------------------------------


def _paced_sections(pattern_id: str) -> list[tuple[float, Any, list[Any]]]:
    pattern = load_pattern(pattern_id)
    result = []
    for section in pattern.sections:
        if section.kind != "paced":
            continue
        phases = pattern.phases[section.first_phase : section.stop_phase]
        cycles = sum(1 for phase in phases if phase.kind == "inhale")
        result.append((round(60.0 * cycles / section.duration_s, 3), section, phases))
    return result


def test_nn_patterns_load_and_last_two_to_four_minutes() -> None:
    for pattern_id in NN_PATTERNS:
        pattern = load_pattern(pattern_id)
        assert 120.0 <= pattern.duration_s <= 240.0, pattern_id
        assert pattern.name and pattern.description
        assert pattern.phases[0].kind in ("settle", "normal")


def test_nn_varied_rates_covers_the_rates_with_different_splits() -> None:
    paced = _paced_sections("nn_varied_rates")

    assert [rate for rate, _, _ in paced] == [6.0, 10.0, 15.0, 20.0, 25.0]
    splits = {round(phases[0].duration_s / phases[1].duration_s, 2) for _, _, phases in paced}
    assert len(splits) == 5 and max(splits) > 1.0  # some inhales outlast their exhale
    pattern = load_pattern("nn_varied_rates")
    assert {phase.kind for phase in pattern.phases} == {"normal", "inhale", "exhale"}
    kinds = [section.kind for section in pattern.sections]
    assert all(kinds[index + 1] == "normal" for index, kind in enumerate(kinds[:-1]) if kind == "paced")


def test_nn_holds_holds_after_inhaling_and_after_exhaling() -> None:
    pattern = load_pattern("nn_holds")
    shapes = phase_shapes(pattern)
    holds = [(pattern.phases[index - 1].kind, phase.duration_s, shapes[index]) for index, phase in enumerate(pattern.phases) if phase.kind == "hold"]

    after_inhale = [seconds for before, seconds, _ in holds if before == "inhale"]
    after_exhale = [seconds for before, seconds, _ in holds if before == "exhale"]
    assert len(after_inhale) + len(after_exhale) == len(holds)  # never straight after free breathing
    assert min(after_inhale + after_exhale) >= 3.0 and max(after_inhale + after_exhale) <= 15.0
    assert max(after_inhale) >= 10.0 and max(after_exhale) >= 10.0
    assert all(shape == (LUNGS_FULL, LUNGS_FULL) for before, _, shape in holds if before == "inhale")
    assert all(shape == (LUNGS_EMPTY, LUNGS_EMPTY) for before, _, shape in holds if before == "exhale")


def test_nn_shallow_deep_alternates_depth_at_the_same_rates() -> None:
    depth_by_rate: dict[float, list[str]] = {}
    for rate, section, phases in _paced_sections("nn_shallow_deep"):
        details = " ".join(phase.detail.lower() for phase in phases)
        depth = "płytko" if "płytko" in details else "głęboko"
        assert ("głęboko" in details) != ("płytko" in details), section.label
        depth_by_rate.setdefault(rate, []).append(depth)

    assert sorted(depth_by_rate) == [6.0, 10.0, 15.0, 20.0, 25.0]
    assert all(sorted(depths) == ["głęboko", "płytko"] for depths in depth_by_rate.values())


def test_nn_irregular_rhythm_cannot_be_anticipated() -> None:
    pattern = load_pattern("nn_irregular")
    breaths = [phase for phase in pattern.phases if phase.kind in ("inhale", "exhale")]
    cycles = [(breaths[index].duration_s, breaths[index + 1].duration_s) for index in range(0, len(breaths), 2)]

    assert {phase.kind for phase in pattern.phases} == {"normal", "inhale", "exhale"}
    assert [phase.kind for phase in breaths] == ["inhale", "exhale"] * len(cycles)
    assert len(cycles) >= 30
    assert len({inhale for inhale, _ in cycles}) >= 15 and len({exhale for _, exhale in cycles}) >= 15
    assert all(current != following for current, following in zip(cycles, cycles[1:]))
    assert any(inhale > exhale for inhale, exhale in cycles)
    lengths = [inhale + exhale for inhale, exhale in cycles]
    assert max(lengths) / min(lengths) >= 3.0


def test_nn_free_settles_then_breathes_freely() -> None:
    pattern = load_pattern("nn_free")

    assert [phase.kind for phase in pattern.phases] == ["settle", "normal"]
    assert pattern.phases[1].duration_s >= 150.0


# -- quality gates -------------------------------------------------------------------


def test_a121_gate_counts_frames_inside_the_trial_window() -> None:
    def rows(stamps_ms: np.ndarray) -> list[list[Any]]:
        return [[stamp] + [0.0] * (len(A121_CAPTURE_COLUMNS) - 1) for stamp in stamps_ms]

    start_ms = 1_000_000.0
    full = start_ms + np.arange(-20, 220) * 50.0  # 20 Hz from 1 s before to 1 s after a 10 s window
    summary = tool.summarize_a121_rows(rows(full), window_start_ms=start_ms, duration_s=10.0, frame_rate_hz=20.0)
    assert summary["frames_in_window"] == summary["expected_frames"] == 200
    assert summary["sample_rate_hz"] == pytest.approx(20.0)
    assert tool.a121_quality_error(summary, duration_s=10.0) is None

    holed = full[(full < start_ms + 3000.0) | (full >= start_ms + 4000.0)]
    summary = tool.summarize_a121_rows(rows(holed), window_start_ms=start_ms, duration_s=10.0, frame_rate_hz=20.0)
    assert "luk" in tool.a121_quality_error(summary, duration_s=10.0)

    sparse = full[::4]
    summary = tool.summarize_a121_rows(rows(sparse), window_start_ms=start_ms, duration_s=10.0, frame_rate_hz=20.0)
    assert "ramek" in tool.a121_quality_error(summary, duration_s=10.0)


def test_restarted_phone_stream_is_rejected() -> None:
    first = [[1000.0 + 10.0 * index, 0, 0, 1, 0, 0, 0] for index in range(50)]
    restarted = first + [[1100.0 + 10.0 * index, 0, 0, 1, 0, 0, 0] for index in range(50)]

    assert tool.iphone_time_axis_error(first, {"sequence_resets": 0}) is None
    assert "cofają" in tool.iphone_time_axis_error(restarted, {"sequence_resets": 0})
    assert "zrestartowała" in tool.iphone_time_axis_error(first, {"sequence_resets": 1})


def test_iphone_capture_keeps_packet_arrival_lags(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = tool.ArrivalLoggingIPhoneCapture()
    clock_ms = [1_000_120.0]
    monkeypatch.setattr(tool.time, "time", lambda: clock_ms[0] / 1000.0)

    def packet(sequence: int, first_ms: int, count: int = 10) -> bytearray:
        samples = b"".join(struct.pack("<Ihhhhhh", first_ms + 10 * index, 0, 0, 1000, 0, 0, 0) for index in range(count))
        return bytearray(struct.pack("<BBH", 1, count, sequence) + samples)

    # Ten samples 10 ms apart per packet, arriving 30, 30 and then 55 ms after their last sample.
    capture._on_notification(None, packet(0, 0))
    clock_ms[0] = 1_000_220.0
    capture._on_notification(None, packet(1, 100))
    clock_ms[0] = 1_000_345.0
    capture._on_notification(None, packet(2, 200))
    capture._on_notification(None, packet(2, 200))  # duplicate: dropped, so not logged

    assert capture.data_count() == 30
    assert capture.snapshot_data_storage()[0][0] == pytest.approx(1_000_120.0)  # stamped with the first arrival
    assert capture.snapshot_arrival_lags_since(0) == pytest.approx([-90.0, -90.0, -65.0])
    assert capture.arrival_count() == 3
    lags = tool.arrival_lag_summary(capture.snapshot_arrival_lags_since(1))
    assert lags["packets"] == 2
    # The true lateness is 120 ms (90 ms packet span + 30 ms latency); the estimate misses only the latency.
    assert tool.arrival_lag_summary(capture.snapshot_arrival_lags_since(0))["stamp_lateness_estimate_ms"] == pytest.approx(90.0)
    assert tool.arrival_lag_summary([]) is None


# -- worker with fake sensors --------------------------------------------------------


def test_worker_records_three_csvs_and_a_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    trial, worker, events = _run_worker(tmp_path, monkeypatch, a121=FakeA121Capture, iphone=FakeIPhoneCapture)

    assert events["failed"] == [] and events["aborted"] == []
    assert len(events["finished"]) == 1
    summary = events["finished"][0]
    assert events["prep"] == [1]
    assert events["progress"][-1] == (pytest.approx(trial.duration_s), pytest.approx(trial.duration_s))
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "run_01_quick_a121.csv",
        "run_01_quick_cues.csv",
        "run_01_quick_iphone.csv",
    ]
    assert summary["files"] == {key: str(path) for key, path in worker.paths.items()}

    a121 = pd.read_csv(worker.paths["a121"])
    iphone = pd.read_csv(worker.paths["iphone"])
    cues = pd.read_csv(worker.paths["cues"])
    assert list(a121.columns) == A121_CAPTURE_COLUMNS
    assert list(iphone.columns) == IMU_COLUMNS
    assert list(cues.columns) == tool.CUE_COLUMNS
    assert list(cues["kind"]) == [phase.kind for phase in trial.pattern.phases] == ["normal", "inhale", "hold", "exhale"]
    start_ms = summary["measurement_start_wall_ms"]
    assert cues["start_wall_ms"].iloc[1] == pytest.approx(start_ms + 300.0)
    assert cues["end_wall_ms"].iloc[-1] == pytest.approx(summary["measurement_end_wall_ms"])
    # Both files cover the whole trial window, with host-epoch time stamps.
    assert a121["Timestamp_ms"].min() <= start_ms + 50.0 and a121["Timestamp_ms"].max() >= start_ms + 1000.0
    assert iphone["Time_ms"].min() <= start_ms + 10.0 and iphone["Time_ms"].max() >= start_ms + 1000.0

    assert summary["trial"]["pattern_id"] == "quick"
    assert summary["a121"]["port"] == "TEST"
    assert summary["a121"]["frames"] == len(a121)
    assert summary["a121"]["frames_in_window"] >= 18
    assert summary["a121"]["sample_rate_hz"] == pytest.approx(20.0)
    assert summary["a121"]["dropped_results"] == 0
    assert summary["a121"]["effective"]["num_points"] == 101
    assert summary["a121"]["config"]["frame_rate_hz"] == 20.0
    assert summary["iphone"]["rows"] == len(iphone)
    assert summary["iphone"]["sample_rate_hz"] == pytest.approx(100.0)
    assert summary["iphone"]["window_coverage"]["sample_coverage_percent"] >= 95.0
    assert summary["iphone"]["ble_arrival"] is None  # the fake has no packet log
    assert summary["setup"]["posture"] == "supine"
    text = tool.format_summary(summary)
    assert "A121:" in text and "iPhone:" in text and "run_01_quick_cues.csv" in text
    json.dumps(summary)  # manifest-ready


def test_incomplete_iphone_stream_is_rejected_and_files_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    _, worker, events = _run_worker(tmp_path, monkeypatch, a121=FakeA121Capture, iphone=StallingIPhoneCapture)

    assert events["finished"] == [] and events["aborted"] == []
    assert len(events["failed"]) == 1 and "iPhone" in events["failed"][0]
    assert not any(path.exists() for path in worker.paths.values())


def test_too_few_a121_frames_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    _, worker, events = _run_worker(tmp_path, monkeypatch, a121=SlowA121Capture, iphone=FakeIPhoneCapture)

    assert events["finished"] == []
    assert len(events["failed"]) == 1 and "A121" in events["failed"][0] and "ramek" in events["failed"][0]
    assert not any(path.exists() for path in worker.paths.values())


def test_abort_during_the_recording_discards_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    _, worker, events = _run_worker(
        tmp_path, monkeypatch, a121=FakeA121Capture, iphone=FakeIPhoneCapture, abort_on_progress=True
    )

    assert events["aborted"] == [True]
    assert events["finished"] == [] and events["failed"] == []
    assert not any(path.exists() for path in worker.paths.values())


# -- window and manifest -------------------------------------------------------------


def test_window_shows_the_pattern_on_the_coach_and_keeps_the_manifest(tmp_path: Path, qapp: QApplication) -> None:
    trials = tool.build_trials()
    session_dir = tmp_path / "session"
    settings = tool.SessionSettings(
        session_dir=session_dir, distance_cm=80.0, posture="side", a121_config=tool.a121_config_for(distance_cm=80.0)
    )
    window = tool.GuidedRecordingWindow(trials=trials, settings=settings)
    try:
        assert window.coach.pattern is trials[0].pattern
        assert window.coach.cue_label.text() == f"PO STARCIE: {trials[0].pattern.phases[0].cue}"
        assert "10 cm" in window.instructions.text() and "leżenie na boku" in window.instructions.text()
        assert window.start_button.isEnabled() and not window.keep_button.isEnabled()

        manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["tool_version"] == tool.TOOL_VERSION
        assert [item["pattern_id"] for item in manifest["trials"]] == NN_PATTERNS
        assert manifest["distance_cm"] == 80.0 and manifest["posture"] == "side"
        assert manifest["a121_config"]["start_m"] == pytest.approx(0.3)
        assert "10 cm" in manifest["phone_placement"] and manifest["measurements"] == []

        window.on_preparation(3)
        assert window.coach.cue_label.text() == "PRZYGOTUJ SIĘ"
        window.on_progress(20.0, trials[0].duration_s)
        assert window.coach.cue_label.text() == "ODDYCHAJ SWOBODNIE"

        # A finished recording that is thrown away removes its files and stays on the same trial.
        files = {key: session_dir / name for key, name in (("a121", "a.csv"), ("iphone", "i.csv"), ("cues", "c.csv"))}
        for path in files.values():
            path.write_text("x", encoding="utf-8")
        summary = {"trial": trials[0].as_dict(), "files": {key: str(path) for key, path in files.items()}}
        window.on_finished(summary)
        assert window.keep_button.isEnabled() and window.discard_button.isEnabled()
        window.discard_pending()
        assert not any(path.exists() for path in files.values())
        assert window.current_trial() is trials[0]

        window.on_finished(summary)
        window.accept_pending()
        manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
        assert [(item["trial"]["number"], item["status"]) for item in manifest["measurements"]] == [(1, "accepted")]
        assert not (session_dir / "manifest.tmp").exists()
        assert window.coach.pattern is trials[1].pattern
        assert [section.label for section in window.coach.pattern.sections] == [
            section.label for section in trials[1].pattern.sections
        ]
    finally:
        window.close()

    # Resuming the session keeps what was accepted; another plan is refused.
    resumed = tool.load_or_create_manifest(session_dir / "manifest.json", tool.new_manifest(trials, settings))
    assert len(resumed["measurements"]) == 1
    with pytest.raises(ValueError, match="inny plan"):
        tool.load_or_create_manifest(session_dir / "manifest.json", tool.new_manifest(trials[:2], settings))


def test_window_runs_a_trial_in_its_worker_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication) -> None:
    monkeypatch.setattr(tool, "A121Capture", FakeA121Capture)
    monkeypatch.setattr(tool, "ArrivalLoggingIPhoneCapture", FakeIPhoneCapture)
    beeps: list[int] = []
    monkeypatch.setattr(tool.QApplication, "beep", staticmethod(lambda: beeps.append(1)))
    trial = _quick_trial()
    settings = tool.SessionSettings(session_dir=tmp_path, a121_port="TEST", prep_seconds=0.2, post_roll_s=0.1)
    window = tool.GuidedRecordingWindow(trials=[trial], settings=settings)
    try:
        window.start_measurement()
        assert not window.start_button.isEnabled() and window.abort_button.isEnabled()
        deadline = time.monotonic() + 15.0
        while (window.pending_summary is None or window.active_thread is not None) and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)

        assert window.pending_summary is not None, window.status.text()
        assert 2 <= len(beeps) <= len(trial.pattern.phases)  # a beep when a new phase starts
        assert window.coach.cue_label.text() == "KONIEC"
        assert window.keep_button.isEnabled() and not window.abort_button.isEnabled()
        window.accept_pending()
        entry = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["measurements"][0]
        assert entry["status"] == "accepted" and entry["trial"]["pattern_id"] == "quick"
        assert all(Path(path).exists() for path in entry["files"].values())
        assert window.current_trial() is None and window.coach.cue_label.text() == "KONIEC SESJI"
    finally:
        window.close()


def test_recording_an_accepted_trial_again_drops_its_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    monkeypatch.setattr(tool, "A121Capture", FakeA121Capture)
    monkeypatch.setattr(tool, "ArrivalLoggingIPhoneCapture", FakeIPhoneCapture)
    trial = _quick_trial()
    settings = tool.SessionSettings(session_dir=tmp_path, a121_port="TEST", prep_seconds=5.0)
    paths = tool.output_paths(tmp_path, trial)
    for path in paths.values():
        path.write_text("accepted earlier", encoding="utf-8")
    manifest = tool.new_manifest([trial], settings)
    manifest["measurements"] = [
        {"trial": trial.as_dict(), "status": "accepted", "files": {key: str(path) for key, path in paths.items()}}
    ]
    tool.save_manifest(tmp_path / "manifest.json", manifest)

    window = tool.GuidedRecordingWindow(trials=[trial], settings=settings)
    try:
        assert "już zachowana" in window.status.text()
        window.start_measurement()
        window.abort_current()
        deadline = time.monotonic() + 10.0
        while window.active_thread is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)

        assert window.coach.cue_label.text() == "ODRZUCONO"
        # The redo deleted the old files, so the manifest must not point at them any more.
        assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["measurements"] == []
        assert not any(path.exists() for path in paths.values())
    finally:
        window.close()


def test_window_after_the_last_trial_shows_the_end(tmp_path: Path, qapp: QApplication) -> None:
    trials = tool.build_trials(["nn_free"])
    window = tool.GuidedRecordingWindow(trials=trials, settings=tool.SessionSettings(session_dir=tmp_path), start_trial=2)
    try:
        assert window.coach.pattern is None
        assert window.coach.cue_label.text() == "KONIEC SESJI"
        assert not window.start_button.isEnabled()
    finally:
        window.close()



def test_a_second_recorder_on_the_same_session_is_refused(tmp_path: Path) -> None:
    lock = tool.claim_session(tmp_path)
    try:
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
        # the same process may claim it again; a different live process may not
        tool.claim_session(tmp_path)
        lock.write_text(str(os.getppid()), encoding="utf-8")
        with pytest.raises(ValueError, match="używana przez inny rejestrator"):
            tool.claim_session(tmp_path)
    finally:
        lock.unlink(missing_ok=True)


def test_a_stale_lock_is_taken_over(tmp_path: Path) -> None:
    (tmp_path / tool.LOCK_NAME).write_text("999999999", encoding="utf-8")
    lock = tool.claim_session(tmp_path)
    try:
        assert lock.read_text(encoding="utf-8") == str(os.getpid())
    finally:
        lock.unlink(missing_ok=True)
