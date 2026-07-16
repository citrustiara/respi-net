"""Pure protocol helpers for the simultaneous LSM6DS3--iPhone IMU study.

Keeping these definitions outside the Qt runner makes the prescribed timing,
file naming and discard behaviour independently testable on a headless host.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np


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


def _append_cue(cues: list[Cue], kind: str, duration_s: float, title: str, detail: str) -> None:
    start_s = cues[-1].end_s if cues else 0.0
    cues.append(Cue(start_s, start_s + duration_s, kind, title, detail))


def natural_hold_cues() -> tuple[Cue, ...]:
    cues: list[Cue] = []
    _append_cue(cues, "settle", 10.0, "USPOKÓJ POZYCJĘ", "Leż spokojnie, bez celowych ruchów.")
    _append_cue(cues, "normal", 35.0, "ODDYCHAJ SWOBODNIE", "Oddychaj naturalnie, bez narzuconego rytmu.")
    _append_cue(
        cues,
        "hold",
        15.0,
        "WSTRZYMAJ ODDECH",
        "Zatrzymaj oddech po wydechu; przy dyskomforcie przerwij próbę.",
    )
    if not math.isclose(cues[-1].end_s, 60.0):
        raise AssertionError("Natural/hold protocol must last 60 s")
    return tuple(cues)


def paced_cues() -> tuple[Cue, ...]:
    cues: list[Cue] = []
    _append_cue(cues, "normal", 10.0, "ODDYCHAJ SWOBODNIE", "Ustabilizuj pozycję.")
    for _ in range(7):
        _append_cue(cues, "inhale", 2.0, "WDECH", "Spokojny wdech przez 2 s.")
        _append_cue(cues, "exhale", 3.0, "WYDECH", "Spokojny wydech przez 3 s.")
    _append_cue(
        cues,
        "hold",
        15.0,
        "WSTRZYMAJ ODDECH",
        "Wstrzymaj oddech po wydechu; przy dyskomforcie przerwij próbę.",
    )
    for _ in range(6):
        _append_cue(cues, "inhale", 2.0, "WDECH", "Spokojny wdech przez 2 s.")
        _append_cue(cues, "exhale", 3.0, "WYDECH", "Spokojny wydech przez 3 s.")
    if not math.isclose(cues[-1].end_s, 90.0):
        raise AssertionError("Paced protocol must last 90 s")
    return tuple(cues)


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
    for index, cue in enumerate(cues):
        if elapsed_s < cue.end_s or index == len(cues) - 1:
            return cue, max(0.0, cue.end_s - elapsed_s)
    return None


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
