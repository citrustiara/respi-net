"""Pure protocol helpers for the simultaneous LSM6DS3--iPhone IMU study.

Keeping these definitions outside the Qt runner makes the prescribed timing,
file naming and discard behaviour independently testable on a headless host.
The breathing schedules themselves are the shared pattern files
``natural_hold`` and ``paced_12_hold`` in ``configs/breathing_patterns``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .breathing import cue_at as _cue_at
from .breathing import load_pattern


@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    kind: str
    title: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Trial:
    number: int
    trial_id: str
    label: str
    duration_s: float
    cues: tuple[Cue, ...]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cues"] = [cue.as_dict() for cue in self.cues]
        return data


def _pattern_cues(pattern_id: str, duration_s: float) -> tuple[Cue, ...]:
    pattern = load_pattern(pattern_id)
    if not math.isclose(pattern.duration_s, duration_s):
        raise AssertionError(f"Breathing pattern {pattern_id} must last {duration_s:g} s, got {pattern.duration_s:g} s")
    return tuple(
        Cue(start_s=phase.start_s, end_s=phase.end_s, kind=phase.kind, title=phase.cue, detail=phase.detail)
        for phase in pattern.phases
    )


def natural_hold_cues() -> tuple[Cue, ...]:
    return _pattern_cues("natural_hold", 60.0)


def paced_cues() -> tuple[Cue, ...]:
    return _pattern_cues("paced_12_hold", 90.0)


def build_trials() -> list[Trial]:
    natural = natural_hold_cues()
    paced = paced_cues()
    return [
        Trial(1, "natural-hold-r1", "Oddech naturalny + wstrzymanie — 1/2", 60.0, natural),
        Trial(2, "natural-hold-r2", "Oddech naturalny + wstrzymanie — 2/2", 60.0, natural),
        Trial(3, "paced-r1", "Rytm 12/min + wstrzymanie — 1/2", 90.0, paced),
        Trial(4, "paced-r2", "Rytm 12/min + wstrzymanie — 2/2", 90.0, paced),
    ]


def cue_at(cues: tuple[Cue, ...], elapsed_s: float) -> tuple[Cue, float] | None:
    return _cue_at(cues, elapsed_s)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in value).strip("-._")


def output_paths(session_dir: Path, trial: Trial) -> dict[str, Path]:
    stem = f"run_{trial.number:02d}_{_safe_name(trial.trial_id)}"
    return {
        "lsm6ds3": session_dir / f"{stem}_lsm6ds3.csv",
        "iphone": session_dir / f"{stem}_iphone.csv",
        "cues": session_dir / f"{stem}_cues.csv",
    }


def delete_trial_outputs(paths: dict[str, Path]) -> None:
    """Remove only the three files of the current, rejected trial."""

    for path in paths.values():
        path.unlink(missing_ok=True)


def sample_timing_summary(rows: list[list[float]]) -> dict[str, float | int | None]:
    """Infer time-axis holes for a source that has no sample counter."""

    if len(rows) < 2:
        return {"rows": len(rows), "sample_rate_hz": None, "estimated_missing_samples": 0, "estimated_missing_percent": 0.0}
    values = np.asarray(rows, dtype=float)
    time_ms = values[:, 0]
    dt = np.diff(time_ms)
    valid = dt[np.isfinite(dt) & (dt > 0)]
    if not len(valid):
        return {"rows": len(rows), "sample_rate_hz": None, "estimated_missing_samples": 0, "estimated_missing_percent": 0.0}
    median_dt = float(np.median(valid))
    expected_steps = np.maximum(1, np.rint(valid / median_dt).astype(int))
    missing = int(np.sum(expected_steps - 1))
    expected = len(rows) + missing
    return {
        "rows": len(rows),
        "sample_rate_hz": float(1000.0 / median_dt),
        "estimated_missing_samples": missing,
        "estimated_missing_percent": float(100.0 * missing / expected) if expected else 0.0,
    }


def stream_coverage_summary(
    rows: list[list[float]],
    *,
    window_start_ms: float,
    expected_duration_s: float,
    expected_sample_rate_hz: float,
) -> dict[str, float | int | None]:
    """Describe whether an IMU stream actually covers a capture window.

    A short, contiguous burst can have a perfectly regular local ``dt`` while
    still missing nearly the whole trial.  Unlike :func:`sample_timing_summary`,
    this helper compares the rows with the wall-clock recording interval and
    exposes the largest time hole.  It is intentionally independent of BLE so
    it can also guard other timestamped sources without a sample counter.
    """

    expected_duration_s = max(0.0, float(expected_duration_s))
    expected_rate_hz = max(0.0, float(expected_sample_rate_hz))
    expected_rows = int(round(expected_duration_s * expected_rate_hz))
    if not rows:
        return {
            "rows": 0,
            "expected_rows": expected_rows,
            "sample_coverage_percent": 0.0,
            "time_coverage_s": 0.0,
            "time_coverage_percent": 0.0,
            "first_sample_offset_s": None,
            "last_sample_offset_s": None,
            "largest_gap_s": None,
        }

    values = np.asarray(rows, dtype=float)
    timestamps_ms = values[:, 0]
    finite = timestamps_ms[np.isfinite(timestamps_ms)]
    if not len(finite):
        return {
            "rows": len(rows),
            "expected_rows": expected_rows,
            "sample_coverage_percent": 0.0,
            "time_coverage_s": 0.0,
            "time_coverage_percent": 0.0,
            "first_sample_offset_s": None,
            "last_sample_offset_s": None,
            "largest_gap_s": None,
        }

    finite.sort()
    gaps_s = np.diff(finite) / 1000.0
    positive_gaps_s = gaps_s[np.isfinite(gaps_s) & (gaps_s > 0)]
    first_offset_s = float((finite[0] - window_start_ms) / 1000.0)
    last_offset_s = float((finite[-1] - window_start_ms) / 1000.0)
    time_coverage_s = max(0.0, float((finite[-1] - finite[0]) / 1000.0))
    return {
        "rows": len(rows),
        "expected_rows": expected_rows,
        "sample_coverage_percent": float(100.0 * len(rows) / expected_rows) if expected_rows else 100.0,
        "time_coverage_s": time_coverage_s,
        "time_coverage_percent": float(100.0 * time_coverage_s / expected_duration_s) if expected_duration_s else 100.0,
        "first_sample_offset_s": first_offset_s,
        "last_sample_offset_s": last_offset_s,
        "largest_gap_s": float(np.max(positive_gaps_s)) if len(positive_gaps_s) else 0.0,
    }


def counter_delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: max(0, int(after.get(key, 0)) - int(before.get(key, 0))) for key in after}
