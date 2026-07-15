#!/usr/bin/env python
"""Overlay Garmin FIT sleep/vitals references on an A121 gated sleep-vitals trend.

The script also adds a first-pass radar-only sleep phase heuristic.  It is designed for overnight
A121 trend CSVs produced by tools/analyze_a121_sleep_vitals_gated.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .garmin_fit import GarminReferenceData, extract_garmin_reference

RADAR_PHASE_LEVELS = {"deep": 0.0, "light": 1.0, "rem": 2.0, "awake": 3.0}
RADAR_PHASE_LABELS = {0.0: "Deep", 1.0: "Light", 2.0: "REM", 3.0: "Awake"}
RADAR_PHASE_COLORS = {"deep": "#312e81", "light": "#2563eb", "rem": "#a855f7", "awake": "#f97316"}

# FIT Profile sleep_level enum used by Garmin sleep exports.
GARMIN_SLEEP_LEVEL_NAMES = {0: "unmeasurable", 1: "awake", 2: "light", 3: "deep", 4: "rem"}
GARMIN_TO_RADAR_LEVEL = {0: np.nan, 1: 3.0, 2: 1.0, 3: 0.0, 4: 2.0}


@dataclass(frozen=True)
class SleepClassification:
    frame: pd.DataFrame
    onset_idx: int
    wake_idx: int
    onset_reason: str
    wake_reason: str
    step_s: float
    terminal_rest_start_idx: int | None = None
    terminal_rest_reason: str | None = None


@dataclass(frozen=True)
class RadarSleepScore:
    score: int
    quality: str
    raw_score: float
    duration_cap: float
    duration_points: float
    continuity_points: float
    stage_balance_points: float
    restfulness_points: float
    recovery_points: float
    total_sleep_minutes: float
    sleep_period_minutes: float
    awake_minutes: float
    light_minutes: float
    deep_minutes: float
    rem_minutes: float
    sleep_efficiency_pct: float
    deep_pct_of_sleep: float
    rem_pct_of_sleep: float
    mean_movement_fraction: float
    mean_no_presence_fraction: float
    awake_segments: int
    note: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A121 radar vitals overlaid with Garmin FIT HR/RR and sleep phases.")
    parser.add_argument("trend_csv", type=Path, help="*_gated_sleep_vitals.csv produced by analyze_a121_sleep_vitals_gated.py")
    parser.add_argument("garmin_dir", type=Path, help="Directory containing Garmin *.fit files and optional Sleep.csv")
    parser.add_argument("--output", type=Path, help="Output PNG. Defaults beside trend_csv.")
    parser.add_argument("--csv-output", type=Path, help="Output merged phase/vitals CSV. Defaults beside trend_csv.")
    parser.add_argument("--score-output", type=Path, help="Output radar sleep-score JSON. Defaults beside trend_csv.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _default_outputs(trend_csv: Path, *, garmin: bool = False) -> tuple[Path, Path, Path]:
    stem = trend_csv.stem
    suffix = "_gated_sleep_vitals"
    base = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    tag = "garmin_overlay_sleep_phases" if garmin else "sleep_phases"
    return (
        trend_csv.with_name(f"{base}_{tag}.png"),
        trend_csv.with_name(f"{base}_{tag}.csv"),
        trend_csv.with_name(f"{base}_radar_sleep_score.json"),
    )


def empty_garmin_reference() -> GarminReferenceData:
    return GarminReferenceData(
        pd.DataFrame(columns=["time_utc", "local_time", "garmin_hr_bpm", "source_file"]),
        pd.DataFrame(columns=["time_utc", "local_time", "garmin_rr_bpm", "source_file"]),
        pd.DataFrame(columns=["time_utc", "local_time", "sleep_level_code", "source_file"]),
        pd.DataFrame(columns=["time_utc", "local_time", "event", "event_code", "event_type", "source_file"]),
    )


def load_garmin_reference(sources: Path | Iterable[Path] | None) -> GarminReferenceData:
    if sources is None:
        return empty_garmin_reference()
    if isinstance(sources, (str, Path)):
        source_list = [Path(sources)]
    else:
        source_list = [Path(source) for source in sources]
    if not source_list:
        return empty_garmin_reference()
    return extract_garmin_reference(source_list)


def _numeric(df: pd.DataFrame, col: str, default: float = math.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _rolling_mode(values: list[str], window: int) -> list[str]:
    if window <= 1 or len(values) == 0:
        return list(values)
    out: list[str] = []
    half = window // 2
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        counts = Counter(values[lo:hi])
        # Prefer the center label on ties to avoid phase jitter.
        best_count = counts[values[idx]]
        best = values[idx]
        for label, count in counts.items():
            if count > best_count:
                best = label
                best_count = count
        out.append(best)
    return out


def _merge_short_non_awake_segments(phases: list[str], *, min_len: int, sleep_start: int, sleep_end: int) -> list[str]:
    if min_len <= 1:
        return phases
    out = list(phases)
    changed = True
    while changed:
        changed = False
        segments: list[tuple[int, int, str]] = []
        start = 0
        current = out[0]
        for idx, label in enumerate(out[1:], 1):
            if label != current:
                segments.append((start, idx - 1, current))
                start = idx
                current = label
        segments.append((start, len(out) - 1, current))

        for seg_idx, (start, end, label) in enumerate(segments):
            if end < sleep_start or start > sleep_end or label == "awake" or (end - start + 1) >= min_len:
                continue
            left = segments[seg_idx - 1][2] if seg_idx > 0 else None
            right = segments[seg_idx + 1][2] if seg_idx + 1 < len(segments) else None
            if left == right and left is not None:
                replacement = left
            elif right is not None and right != "awake":
                replacement = right
            else:
                replacement = left
            if replacement is None or replacement == label:
                continue
            for idx in range(start, end + 1):
                out[idx] = replacement
            changed = True
            break
    return out


def _bridge_direct_deep_rem_transitions(
    phases: list[str],
    *,
    min_len: int,
    sleep_start: int,
    sleep_end: int,
) -> tuple[list[str], list[bool]]:
    """Insert a short light-sleep bridge between direct deep/REM transitions.

    The radar classifier operates on coarse surrogate features and can otherwise jump directly
    between deep and REM sleep.  Because the available signals cannot support that abrupt staging
    decision reliably, an intermediate lighter stage is the more conservative representation.
    The correction is a structural prior only; it does not use Garmin labels or tune any feature
    threshold.
    """

    out = list(phases)
    corrected = [False] * len(out)
    if len(out) < 2 or min_len <= 0:
        return out, corrected

    idx = max(1, sleep_start + 1)
    last = min(sleep_end, len(out) - 1)
    while idx <= last:
        if {out[idx - 1], out[idx]} != {"deep", "rem"}:
            idx += 1
            continue

        destination = out[idx]
        run_end = idx
        while run_end + 1 <= last and out[run_end + 1] == destination:
            run_end += 1
        bridge_end = min(run_end, idx + min_len - 1)
        for pos in range(idx, bridge_end + 1):
            out[pos] = "light"
            corrected[pos] = True
        idx = bridge_end + 1

    return out, corrected


def _detect_sleep_interval(trend: pd.DataFrame, step_s: float) -> tuple[int, int, str, str]:
    state = trend.get("state", pd.Series([""] * len(trend), index=trend.index)).fillna("").astype(str)
    still = _numeric(trend, "still_fraction", 0.0)
    movement = _numeric(trend, "movement_fraction", 1.0)
    no_presence = _numeric(trend, "no_presence_fraction", 0.0)
    hr_ok = _numeric(trend, "hr_bpm_connected").notna() | _numeric(trend, "hr_bpm").notna() | _numeric(trend, "hr_bpm_plot_connected").notna()
    rr_ok = _numeric(trend, "rr_bpm_connected").notna() | _numeric(trend, "rr_bpm").notna()

    stable_sleep = (
        (still >= 0.80)
        & (movement <= 0.20)
        & (no_presence <= 0.05)
        & (~state.isin(["movement", "no_presence"]))
        & (hr_ok | rr_ok)
    )
    sustain_n = max(10, int(round(30.0 * 60.0 / max(step_s, 1.0))))
    future_stable = stable_sleep.iloc[::-1].rolling(sustain_n, min_periods=max(3, sustain_n // 2)).mean().iloc[::-1]
    candidates = np.flatnonzero(future_stable.to_numpy(dtype=float) >= 0.80)
    if len(candidates) == 0:
        first_ok = np.flatnonzero((hr_ok | rr_ok).to_numpy(dtype=bool))
        onset_idx = int(first_ok[0]) if len(first_ok) else 0
        onset_reason = "fallback_first_vitals"
    else:
        first = int(candidates[0])
        severe_motion = (
            (no_presence > 0.15)
            | (movement > 0.50)
            | state.eq("no_presence")
            | (state.eq("movement") & (still < 0.50))
        ).to_numpy(dtype=bool)
        search_end = min(len(trend), first + sustain_n)
        bad_indices = [idx for idx in range(first, search_end) if severe_motion[idx]]
        onset_idx = (max(bad_indices) + 1) if bad_indices else first
        onset_idx = min(onset_idx, len(trend) - 1)
        onset_reason = "first_30min_sustained_still_vitals_after_last_motion"

    # If the recording contains a terminal sustained awake/movement run, mark wake at its start.
    # Otherwise the only defensible wake cue is the recording stop time.
    terminal_awake = ((no_presence > 0.20) | (movement > 0.35) | state.eq("no_presence") | (state.eq("movement") & (movement > 0.20))).to_numpy(dtype=bool)
    wake_run_n = max(10, int(round(20.0 * 60.0 / max(step_s, 1.0))))
    wake_idx = len(trend) - 1
    wake_reason = "recording_end_no_terminal_wake_run"
    if len(trend) > wake_run_n and np.mean(terminal_awake[-wake_run_n:]) >= 0.65:
        tail_start = len(trend) - wake_run_n
        bad_tail = np.flatnonzero(terminal_awake[tail_start:])
        if len(bad_tail):
            wake_idx = int(tail_start + bad_tail[0])
            wake_reason = "terminal_20min_awake_motion_run"
    wake_idx = max(wake_idx, onset_idx)
    return onset_idx, wake_idx, onset_reason, wake_reason


def _detect_terminal_fragmented_rest(
    *,
    onset_idx: int,
    wake_idx: int,
    step_s: float,
    still: pd.Series,
    movement: pd.Series,
    no_presence: pd.Series,
    state: pd.Series,
) -> int | None:
    """Detect a late-night quiet-but-fragmented in-bed period.

    This is meant for cases like lying still in bed with a phone after the main sleep episode.  Low
    body motion can look REM-like, but repeated small movement/arousal cues near the end of the
    night make light/quiet wake a safer label than REM unless stronger REM evidence is available.
    """

    sleep_len = max(1, wake_idx - onset_idx)
    if sleep_len * step_s < 3.0 * 60.0 * 60.0:
        return None
    late_start = onset_idx + int(round(0.88 * sleep_len))
    fragment = (movement > 0.10) | (no_presence > 0.05) | state.eq("movement") | state.eq("no_presence")
    future_window = max(10, int(round(30.0 * 60.0 / max(step_s, 1.0))))
    future_fragment = fragment.iloc[::-1].rolling(future_window, min_periods=max(3, future_window // 2)).mean().iloc[::-1]
    still_window = max(5, int(round(10.0 * 60.0 / max(step_s, 1.0))))
    still_med = still.rolling(still_window, center=True, min_periods=1).median()
    for idx in range(max(onset_idx, late_start), wake_idx + 1):
        if future_fragment.iloc[idx] >= 0.12 and still_med.iloc[idx] >= 0.70:
            return idx
    return None


def classify_radar_sleep(trend: pd.DataFrame) -> SleepClassification:
    work = trend.copy()
    work["clock_time_dt"] = pd.to_datetime(work["clock_time"])
    elapsed = _numeric(work, "elapsed_s")
    step_s = float(np.nanmedian(np.diff(elapsed.to_numpy(dtype=float)))) if len(work) > 1 else 30.0
    if not np.isfinite(step_s) or step_s <= 0:
        step_s = 30.0

    onset_idx, wake_idx, onset_reason, wake_reason = _detect_sleep_interval(work, step_s)

    hr = _numeric(work, "hr_bpm_connected").combine_first(_numeric(work, "hr_bpm")).combine_first(_numeric(work, "hr_bpm_plot_connected"))
    rr = _numeric(work, "rr_bpm_connected").combine_first(_numeric(work, "rr_bpm"))
    hr_i = hr.interpolate(limit_direction="both")
    rr_i = rr.interpolate(limit_direction="both")

    win5 = max(3, int(round(5.0 * 60.0 / step_s)))
    win10 = max(5, int(round(10.0 * 60.0 / step_s)))
    hr_smooth = hr_i.rolling(win5, center=True, min_periods=1).median()
    rr_smooth = rr_i.rolling(win5, center=True, min_periods=1).median()
    hr_var = hr_i.rolling(win10, center=True, min_periods=3).std().fillna(0.0)
    rr_var = rr_i.rolling(win10, center=True, min_periods=3).std().fillna(0.0)
    move_roll = _numeric(work, "movement_fraction", 0.0).rolling(win5, center=True, min_periods=1).mean()

    sleep_slice = slice(onset_idx, wake_idx + 1)
    hr_sleep = hr_smooth.iloc[sleep_slice].dropna()
    rr_var_sleep = rr_var.iloc[sleep_slice].dropna()
    hr_var_sleep = hr_var.iloc[sleep_slice].dropna()
    if hr_sleep.empty:
        hr_base = 55.0
        hr_iqr = 4.0
        deep_hr_threshold = 55.0
    else:
        hr_base = float(np.nanmedian(hr_sleep))
        q25 = float(np.nanpercentile(hr_sleep, 25))
        q75 = float(np.nanpercentile(hr_sleep, 75))
        hr_iqr = max(q75 - q25, 1.0)
        deep_hr_threshold = float(np.nanpercentile(hr_sleep, 50))
    hr_var_70 = float(np.nanpercentile(hr_var_sleep, 70)) if not hr_var_sleep.empty else 2.5
    rr_var_60 = float(np.nanpercentile(rr_var_sleep, 60)) if not rr_var_sleep.empty else 1.5
    rr_var_75 = float(np.nanpercentile(rr_var_sleep, 75)) if not rr_var_sleep.empty else 2.0

    state = work.get("state", pd.Series([""] * len(work), index=work.index)).fillna("").astype(str)
    still = _numeric(work, "still_fraction", 0.0)
    movement = _numeric(work, "movement_fraction", 0.0)
    no_presence = _numeric(work, "no_presence_fraction", 0.0)
    phases: list[str] = []
    sleep_elapsed0 = float(elapsed.iloc[onset_idx]) if len(elapsed) else 0.0
    sleep_len = max(1, wake_idx - onset_idx)

    for idx, row in work.iterrows():
        if idx < onset_idx or idx > wake_idx:
            phases.append("awake")
            continue
        minutes_since_onset = (float(elapsed.iloc[idx]) - sleep_elapsed0) / 60.0
        night_progress = (idx - onset_idx) / sleep_len
        awake_arousal = (
            no_presence.iloc[idx] > 0.20
            or movement.iloc[idx] > 0.35
            or state.iloc[idx] == "no_presence"
            or (state.iloc[idx] == "movement" and movement.iloc[idx] > 0.20)
        )
        if awake_arousal:
            phases.append("awake")
            continue

        deep = (
            minutes_since_onset > 20.0
            and move_roll.iloc[idx] < 0.10
            and hr_smooth.iloc[idx] <= deep_hr_threshold
            and rr_var.iloc[idx] <= rr_var_60
            and night_progress < 0.70
        )
        rem = (
            minutes_since_onset > 75.0
            and night_progress > 0.18
            and move_roll.iloc[idx] < 0.15
            and (
                hr_var.iloc[idx] >= hr_var_70
                or rr_var.iloc[idx] >= rr_var_75
                or (hr_smooth.iloc[idx] >= hr_base + 0.50 * hr_iqr and hr_var.iloc[idx] >= hr_var_70)
            )
        )
        if deep and not (rem and night_progress > 0.55):
            phases.append("deep")
        elif rem:
            phases.append("rem")
        else:
            phases.append("light")

    # Keep brief awake/movement arousals; only merge tiny non-awake islands so the hypnogram is
    # readable without erasing wakefulness evidence.
    phases = _merge_short_non_awake_segments(
        phases,
        min_len=max(2, int(round(4.0 * 60.0 / step_s))),
        sleep_start=onset_idx,
        sleep_end=wake_idx,
    )
    phases, direct_transition_bridge = _bridge_direct_deep_rem_transitions(
        phases,
        min_len=max(2, int(round(4.0 * 60.0 / step_s))),
        sleep_start=onset_idx,
        sleep_end=wake_idx,
    )

    terminal_rest_start_idx = _detect_terminal_fragmented_rest(
        onset_idx=onset_idx,
        wake_idx=wake_idx,
        step_s=step_s,
        still=still,
        movement=movement,
        no_presence=no_presence,
        state=state,
    )
    rem_downgraded = [False] * len(phases)
    terminal_rest_reason = None
    if terminal_rest_start_idx is not None:
        terminal_rest_reason = "late_quiet_fragmented_rest_rem_candidates_downgraded_to_light"
        for pos in range(terminal_rest_start_idx, wake_idx + 1):
            if phases[pos] == "rem":
                phases[pos] = "light"
                rem_downgraded[pos] = True

    work["radar_sleep_phase"] = phases
    work["radar_sleep_phase_level"] = [RADAR_PHASE_LEVELS[label] for label in phases]
    work["radar_terminal_fragmented_rest"] = False
    work["radar_rem_downgraded_to_light"] = rem_downgraded
    work["radar_direct_transition_bridge_to_light"] = direct_transition_bridge
    if terminal_rest_start_idx is not None:
        work.loc[work.index[terminal_rest_start_idx : wake_idx + 1], "radar_terminal_fragmented_rest"] = True
    work["radar_sleep_onset"] = False
    work["radar_wake"] = False
    work.loc[work.index[onset_idx], "radar_sleep_onset"] = True
    work.loc[work.index[wake_idx], "radar_wake"] = True
    work["radar_hr_smooth_for_phase"] = hr_smooth
    work["radar_rr_smooth_for_phase"] = rr_smooth
    work["radar_hr_var_10min"] = hr_var
    work["radar_rr_var_10min"] = rr_var
    return SleepClassification(work, onset_idx, wake_idx, onset_reason, wake_reason, step_s, terminal_rest_start_idx, terminal_rest_reason)


def _phase_duration_minutes(phases: Iterable[str], step_s: float) -> dict[str, float]:
    counts = Counter(phases)
    return {phase: counts.get(phase, 0) * step_s / 60.0 for phase in ("awake", "light", "deep", "rem")}


def _quality_label(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _plateau_quality(value: float, low_zero: float, low_best: float, high_best: float, high_zero: float) -> float:
    if value <= low_zero or value >= high_zero:
        return 0.0
    if low_best <= value <= high_best:
        return 1.0
    if value < low_best:
        return _clip01((value - low_zero) / max(low_best - low_zero, 1e-9))
    return _clip01((high_zero - value) / max(high_zero - high_best, 1e-9))


def _count_phase_segments(phases: list[str], label: str, min_len: int = 1) -> int:
    count = 0
    idx = 0
    while idx < len(phases):
        current = phases[idx]
        start = idx
        while idx + 1 < len(phases) and phases[idx + 1] == current:
            idx += 1
        if current == label and idx - start + 1 >= min_len:
            count += 1
        idx += 1
    return count


def _duration_cap(total_sleep_minutes: float) -> float:
    """Short-sleep cap inspired by Garmin-like scoring behavior.

    A very efficient 6-hour night should still usually be capped around Fair because total sleep
    duration dominates consumer sleep scores.
    """

    if total_sleep_minutes < 240.0:
        return 30.0 + 30.0 * _clip01(total_sleep_minutes / 240.0)
    if total_sleep_minutes < 420.0:
        return 60.0 + 20.0 * _clip01((total_sleep_minutes - 240.0) / 180.0)
    if total_sleep_minutes < 480.0:
        return 80.0 + 20.0 * _clip01((total_sleep_minutes - 420.0) / 60.0)
    return 100.0


def _estimate_resting_hr_bpm(classified: SleepClassification) -> tuple[float | None, int]:
    """Estimate resting HR from quiet in-sleep radar HR windows."""

    frame = classified.frame.iloc[classified.onset_idx : classified.wake_idx + 1].copy()
    frame = frame[frame["radar_sleep_phase"].astype(str) != "awake"]
    hr = pd.to_numeric(frame.get("radar_hr_smooth_for_phase", pd.Series(dtype=float)), errors="coerce")
    move = pd.to_numeric(frame.get("movement_fraction", pd.Series(dtype=float)), errors="coerce")
    quiet_hr = hr[move.fillna(1.0) <= 0.10].dropna()
    pool = quiet_hr if not quiet_hr.empty else hr.dropna()
    if pool.empty:
        return None, 0
    return round(float(pool.quantile(0.10)), 1), int(len(pool))


def score_radar_sleep(classified: SleepClassification) -> RadarSleepScore:
    """Compute a Garmin-like 0-100 sleep score from the radar hypnogram and vitals.

    This is not Garmin's proprietary formula.  It mirrors the broad behavior: duration caps the
    maximum score, then continuity, stage balance, restfulness, and recovery vitals tune the result.
    """

    frame = classified.frame
    sleep_frame = frame.iloc[classified.onset_idx : classified.wake_idx + 1]
    phases = sleep_frame["radar_sleep_phase"].astype(str).tolist()
    minutes = _phase_duration_minutes(phases, classified.step_s)
    total_sleep = minutes["light"] + minutes["deep"] + minutes["rem"]
    sleep_period = total_sleep + minutes["awake"]
    efficiency = total_sleep / sleep_period if sleep_period > 0 else 0.0
    deep_pct = minutes["deep"] / total_sleep if total_sleep > 0 else 0.0
    rem_pct = minutes["rem"] / total_sleep if total_sleep > 0 else 0.0

    duration_points = 30.0 * _clip01(total_sleep / 480.0)
    if total_sleep > 540.0:
        duration_points *= _clip01(1.0 - (total_sleep - 540.0) / 180.0)

    awake_segments = _count_phase_segments(phases, "awake", min_len=max(1, int(round(60.0 / max(classified.step_s, 1.0)))))
    hours = max(sleep_period / 60.0, 1e-9)
    awake_segments_per_hour = awake_segments / hours
    continuity_points = (
        15.0 * _clip01((efficiency - 0.75) / 0.18)
        + 3.0 * _clip01(1.0 - minutes["awake"] / 60.0)
        + 2.0 * _clip01(1.0 - awake_segments_per_hour / 4.0)
    )

    deep_quality = _plateau_quality(deep_pct, 0.04, 0.13, 0.23, 0.33)
    rem_quality = _plateau_quality(rem_pct, 0.08, 0.18, 0.27, 0.40)
    stage_balance_points = 10.0 * deep_quality + 10.0 * rem_quality

    movement = pd.to_numeric(sleep_frame.get("movement_fraction", pd.Series(dtype=float)), errors="coerce")
    no_presence = pd.to_numeric(sleep_frame.get("no_presence_fraction", pd.Series(dtype=float)), errors="coerce")
    mean_movement = float(movement.mean()) if not movement.dropna().empty else 0.0
    mean_no_presence = float(no_presence.mean()) if not no_presence.dropna().empty else 0.0
    restfulness_points = (
        8.0 * _clip01(1.0 - mean_movement / 0.08)
        + 3.0 * _clip01(1.0 - mean_no_presence / 0.03)
        + 4.0 * _clip01(1.0 - awake_segments_per_hour / 6.0)
    )

    hr = pd.to_numeric(sleep_frame.get("radar_hr_smooth_for_phase", pd.Series(dtype=float)), errors="coerce").dropna()
    rr = pd.to_numeric(sleep_frame.get("radar_rr_smooth_for_phase", pd.Series(dtype=float)), errors="coerce").dropna()
    if hr.empty:
        hr_points = 4.0
        hr_stability_points = 2.0
    else:
        median_hr = float(hr.median())
        hr_std = float(hr.std()) if len(hr) > 1 else 0.0
        hr_points = 7.0 * _clip01(1.0 - max(0.0, median_hr - 55.0) / 25.0)
        hr_stability_points = 4.0 * _clip01(1.0 - max(0.0, hr_std - 4.0) / 8.0)
    if rr.empty:
        rr_stability_points = 2.0
    else:
        rr_std = float(rr.std()) if len(rr) > 1 else 0.0
        rr_stability_points = 4.0 * _clip01(1.0 - max(0.0, rr_std - 1.2) / 3.0)
    recovery_points = hr_points + hr_stability_points + rr_stability_points

    raw_score = duration_points + continuity_points + stage_balance_points + restfulness_points + recovery_points
    cap = _duration_cap(total_sleep)
    score = int(round(np.clip(min(raw_score, cap), 0.0, 100.0)))
    if cap < raw_score:
        note = "Score capped by short total sleep duration; component raw score was higher."
    else:
        note = "Score from duration, continuity, stage balance, restfulness, and recovery vitals."
    return RadarSleepScore(
        score=score,
        quality=_quality_label(score),
        raw_score=round(float(raw_score), 2),
        duration_cap=round(float(cap), 2),
        duration_points=round(float(duration_points), 2),
        continuity_points=round(float(continuity_points), 2),
        stage_balance_points=round(float(stage_balance_points), 2),
        restfulness_points=round(float(restfulness_points), 2),
        recovery_points=round(float(recovery_points), 2),
        total_sleep_minutes=round(float(total_sleep), 1),
        sleep_period_minutes=round(float(sleep_period), 1),
        awake_minutes=round(float(minutes["awake"]), 1),
        light_minutes=round(float(minutes["light"]), 1),
        deep_minutes=round(float(minutes["deep"]), 1),
        rem_minutes=round(float(minutes["rem"]), 1),
        sleep_efficiency_pct=round(float(efficiency * 100.0), 1),
        deep_pct_of_sleep=round(float(deep_pct * 100.0), 1),
        rem_pct_of_sleep=round(float(rem_pct * 100.0), 1),
        mean_movement_fraction=round(float(mean_movement), 4),
        mean_no_presence_fraction=round(float(mean_no_presence), 4),
        awake_segments=awake_segments,
        note=note,
    )


def _read_sleep_csv_summary(garmin_sources: Path | Iterable[Path] | None) -> dict[str, str]:
    if garmin_sources is None:
        return {}
    if isinstance(garmin_sources, (str, Path)):
        source_list = [Path(garmin_sources)]
    else:
        source_list = [Path(source) for source in garmin_sources]
    candidates: list[Path] = []
    for source in source_list:
        candidates.append(source / "Sleep.csv" if source.is_dir() else source.parent / "Sleep.csv")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if "," not in raw_line:
            continue
        key, value = raw_line.split(",", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            out[key] = value
    return out


def _duration_to_minutes(text: str | None) -> float | None:
    if not text:
        return None
    hours = 0
    minutes = 0
    h_match = re.search(r"(\d+)\s*h", text)
    m_match = re.search(r"(\d+)\s*m", text)
    if h_match:
        hours = int(h_match.group(1))
    if m_match:
        minutes = int(m_match.group(1))
    if not h_match and not m_match:
        return None
    return float(hours * 60 + minutes)


def _format_minutes(minutes: float | None) -> str:
    if minutes is None or not np.isfinite(minutes):
        return "--"
    return f"{int(minutes // 60)}h {int(round(minutes % 60)):02d}m" if minutes >= 60 else f"{minutes:.0f}m"


def _garmin_sleep_bounds(data: GarminReferenceData) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    events = data.sleep_events
    if events.empty:
        return None, None
    sleep_events = events[events["event_code"] == 74].sort_values("local_time")
    starts = sleep_events[sleep_events["event_type"] == "start"]
    stops = sleep_events[sleep_events["event_type"] == "stop"]
    start = pd.Timestamp(starts["local_time"].iloc[0]) if not starts.empty else None
    stop = pd.Timestamp(stops["local_time"].iloc[-1]) if not stops.empty else None
    return start, stop


def _plot_step(ax: plt.Axes, times: pd.Series, levels: pd.Series | np.ndarray, *, step_s: float, **kwargs: object) -> None:
    if len(times) == 0:
        return
    x = list(pd.to_datetime(times))
    y = list(np.asarray(levels, dtype=float))
    x.append(x[-1] + pd.Timedelta(seconds=step_s))
    y.append(y[-1])
    ax.step(x, y, where="post", **kwargs)


def _plot_garmin_sleep_levels(
    ax: plt.Axes,
    data: GarminReferenceData,
    *,
    step_s: float,
    label: str = "Garmin FIT sleep_level (decoded)",
) -> None:
    levels = data.sleep_levels.copy()
    if levels.empty:
        return
    levels["level"] = levels["sleep_level_code"].map(GARMIN_TO_RADAR_LEVEL)
    levels = levels.dropna(subset=["level"]).sort_values("local_time")
    if levels.empty:
        return
    start, stop = _garmin_sleep_bounds(data)
    plot_times = list(pd.to_datetime(levels["local_time"]))
    plot_levels = list(levels["level"].astype(float))
    if start is not None and start < plot_times[0]:
        plot_times.insert(0, start)
        plot_levels.insert(0, RADAR_PHASE_LEVELS["awake"])
    if stop is not None and stop > plot_times[-1]:
        plot_times.append(stop)
        plot_levels.append(plot_levels[-1])
    _plot_step(
        ax,
        pd.Series(plot_times),
        np.asarray(plot_levels),
        step_s=step_s,
        color="#111827",
        lw=1.7,
        linestyle="--",
        alpha=0.75,
        label=label,
    )


def _merge_garmin_nearest(classified: pd.DataFrame, data: GarminReferenceData) -> pd.DataFrame:
    out = classified.copy().sort_values("clock_time_dt")
    left = out[["clock_time_dt"]].rename(columns={"clock_time_dt": "local_time"})
    if not data.heart_rate.empty:
        hr = data.heart_rate[["local_time", "garmin_hr_bpm"]].copy().sort_values("local_time")
        hr["local_time"] = pd.to_datetime(hr["local_time"])
        out["garmin_hr_bpm"] = pd.merge_asof(left, hr, on="local_time", tolerance=pd.Timedelta("2min"), direction="nearest")["garmin_hr_bpm"].to_numpy()
    else:
        out["garmin_hr_bpm"] = np.nan
    if not data.respiration_rate.empty:
        rr = data.respiration_rate[["local_time", "garmin_rr_bpm"]].copy().sort_values("local_time")
        rr["local_time"] = pd.to_datetime(rr["local_time"])
        out["garmin_rr_bpm"] = pd.merge_asof(left, rr, on="local_time", tolerance=pd.Timedelta("2min"), direction="nearest")["garmin_rr_bpm"].to_numpy()
    else:
        out["garmin_rr_bpm"] = np.nan
    return out


def write_sleep_outputs(
    trend_csv: Path,
    garmin_sources: Path | Iterable[Path] | None = None,
    *,
    csv_output: Path | None = None,
    score_output: Path | None = None,
    include_score: bool = True,
) -> tuple[Path | None, Path | None, RadarSleepScore | None]:
    """Classify sleep phases/score from a gated A121 trend CSV and optionally write CSV/JSON."""

    trend = pd.read_csv(trend_csv)
    classified = classify_radar_sleep(trend)
    data = load_garmin_reference(garmin_sources)
    merged = _merge_garmin_nearest(classified.frame, data)
    radar_score = score_radar_sleep(classified) if include_score else None
    if radar_score is not None:
        merged["radar_sleep_score"] = radar_score.score
        merged["radar_sleep_quality"] = radar_score.quality
        merged["radar_sleep_total_minutes"] = radar_score.total_sleep_minutes
        merged["radar_sleep_efficiency_pct"] = radar_score.sleep_efficiency_pct
    if csv_output is not None:
        csv_cols = [
            "clock_time",
            "elapsed_s",
            "state",
            "still_fraction",
            "movement_fraction",
            "no_presence_fraction",
            "hr_bpm",
            "hr_bpm_connected",
            "rr_bpm",
            "rr_bpm_connected",
            "garmin_hr_bpm",
            "garmin_rr_bpm",
            "radar_sleep_phase",
            "radar_sleep_phase_level",
            "radar_sleep_score",
            "radar_sleep_quality",
            "radar_sleep_total_minutes",
            "radar_sleep_efficiency_pct",
            "radar_terminal_fragmented_rest",
            "radar_rem_downgraded_to_light",
            "radar_direct_transition_bridge_to_light",
            "radar_sleep_onset",
            "radar_wake",
            "radar_hr_smooth_for_phase",
            "radar_rr_smooth_for_phase",
            "radar_hr_var_10min",
            "radar_rr_var_10min",
        ]
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        merged[[col for col in csv_cols if col in merged.columns]].to_csv(csv_output, index=False)
    if score_output is not None and radar_score is not None:
        score_output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(radar_score)
        payload["onset_time"] = classified.frame["clock_time"].iloc[classified.onset_idx]
        payload["wake_time"] = classified.frame["clock_time"].iloc[classified.wake_idx]
        payload["onset_reason"] = classified.onset_reason
        payload["wake_reason"] = classified.wake_reason
        payload["terminal_rest_start_time"] = (
            classified.frame["clock_time"].iloc[classified.terminal_rest_start_idx] if classified.terminal_rest_start_idx is not None else None
        )
        payload["terminal_rest_reason"] = classified.terminal_rest_reason
        resting_hr_bpm, resting_hr_samples = _estimate_resting_hr_bpm(classified)
        payload["estimated_resting_hr_bpm"] = resting_hr_bpm
        payload["estimated_resting_hr_samples"] = resting_hr_samples
        payload["estimated_resting_hr_method"] = "10th percentile of quiet non-awake radar_hr_smooth_for_phase windows during the inferred sleep interval"
        score_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_output, score_output if radar_score is not None else None, radar_score


def _create_vitals_only_plot(trend: pd.DataFrame, data: GarminReferenceData, output: Path, *, dpi: int = 150) -> Path:
    times = pd.to_datetime(trend["clock_time_dt"])
    start_time = times.iloc[0]
    end_time = times.iloc[-1]
    has_garmin = not data.heart_rate.empty or not data.respiration_rate.empty
    fig = plt.figure(figsize=(18, 9), dpi=dpi)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.35, 0.85], hspace=0.26, top=0.84)
    fig.suptitle(
        f"A121 radar sleep vitals{' vs Garmin FIT' if has_garmin else ''}\n"
        f"Recording {start_time:%Y-%m-%d %H:%M}–{end_time:%H:%M} local",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(grid[0])
    ax1.plot(times, trend["rr_bpm_connected"], color="#15803d", lw=2.0, label="A121 RR final")
    ax1.scatter(times, trend["rr_bpm"], s=8, color="#86efac", alpha=0.35, label="A121 RR raw/smoothed windows")
    if not data.respiration_rate.empty:
        rr = data.respiration_rate.copy()
        rr["local_time"] = pd.to_datetime(rr["local_time"])
        ax1.plot(rr["local_time"], rr["garmin_rr_bpm"], color="#0f766e", lw=1.5, alpha=0.85, label="Garmin RR FIT")
    ax1.set_ylabel("Respiratory rate (brpm)")
    ax1.set_ylim(5, 24)
    ax1.set_title("Respiration overlay" if has_garmin else "Respiration")
    ax1.grid(True, alpha=0.30)
    ax1.legend(loc="upper right", ncol=3, fontsize=8)

    ax2 = fig.add_subplot(grid[1], sharex=ax1)
    ax2.plot(times, trend["hr_bpm_plot_connected"], color="#b91c1c", lw=2.2, alpha=0.88, label="A121 HR display line")
    ax2.scatter(times, trend["hr_bpm_raw"] if "hr_bpm_raw" in trend else trend["hr_bpm"], s=9, color="#fca5a5", alpha=0.35, label="A121 HR valid windows")
    if not data.heart_rate.empty:
        hr = data.heart_rate.copy()
        hr["local_time"] = pd.to_datetime(hr["local_time"])
        ax2.plot(hr["local_time"], hr["garmin_hr_bpm"], color="#1d4ed8", lw=1.45, alpha=0.82, label="Garmin HR FIT")
    ax2.set_ylabel("Heart rate (bpm)")
    ax2.set_title("Heart-rate overlay" if has_garmin else "Heart rate")
    ax2.grid(True, alpha=0.30)
    ax2.legend(loc="upper right", ncol=3, fontsize=8)

    ax3 = fig.add_subplot(grid[2], sharex=ax1)
    ax3.plot(times, trend["still_fraction"], color="#16a34a", lw=1.2, label="still fraction")
    ax3.plot(times, trend["movement_fraction"], color="#f59e0b", lw=1.2, label="movement fraction")
    ax3.plot(times, trend["no_presence_fraction"], color="#6b7280", lw=1.0, alpha=0.85, label="no-presence fraction")
    ax3.set_ylim(-0.03, 1.03)
    ax3.set_ylabel("Fraction")
    ax3.set_xlabel("Clock time")
    ax3.set_title("Radar movement/presence")
    ax3.grid(True, alpha=0.30)
    ax3.legend(loc="upper right", ncol=3, fontsize=8)

    for axis in (ax1, ax2, ax3):
        axis.set_xlim(start_time, end_time)
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def create_plot(
    trend_csv: Path,
    garmin_dir: Path | Iterable[Path] | None,
    output: Path,
    csv_output: Path | None = None,
    score_output: Path | None = None,
    *,
    dpi: int = 150,
    include_sleep: bool = True,
    include_score: bool = True,
) -> tuple[Path, Path | None, Path | None]:
    trend = pd.read_csv(trend_csv)
    if "clock_time" not in trend.columns:
        raise ValueError(f"{trend_csv} is missing clock_time")
    data = load_garmin_reference(garmin_dir)
    summary = _read_sleep_csv_summary(garmin_dir)

    if not include_sleep:
        work = trend.copy()
        work["clock_time_dt"] = pd.to_datetime(work["clock_time"])
        merged = _merge_garmin_nearest(work, data)
        if csv_output is not None:
            csv_output.parent.mkdir(parents=True, exist_ok=True)
            cols = [
                "clock_time",
                "elapsed_s",
                "state",
                "still_fraction",
                "movement_fraction",
                "no_presence_fraction",
                "hr_bpm",
                "hr_bpm_connected",
                "rr_bpm",
                "rr_bpm_connected",
                "garmin_hr_bpm",
                "garmin_rr_bpm",
            ]
            merged[[col for col in cols if col in merged.columns]].to_csv(csv_output, index=False)
        _create_vitals_only_plot(work, data, output, dpi=dpi)
        return output, csv_output, None

    classified = classify_radar_sleep(trend)
    radar_sleep_phases = classified.frame.loc[classified.onset_idx : classified.wake_idx, "radar_sleep_phase"]
    radar_minutes = _phase_duration_minutes(radar_sleep_phases, classified.step_s)
    radar_score = score_radar_sleep(classified) if include_score else None
    merged = _merge_garmin_nearest(classified.frame, data)
    if radar_score is not None:
        merged["radar_sleep_score"] = radar_score.score
        merged["radar_sleep_quality"] = radar_score.quality
        merged["radar_sleep_total_minutes"] = radar_score.total_sleep_minutes
        merged["radar_sleep_efficiency_pct"] = radar_score.sleep_efficiency_pct

    csv_cols = [
        "clock_time",
        "elapsed_s",
        "state",
        "still_fraction",
        "movement_fraction",
        "no_presence_fraction",
        "hr_bpm",
        "hr_bpm_connected",
        "rr_bpm",
        "rr_bpm_connected",
        "garmin_hr_bpm",
        "garmin_rr_bpm",
        "radar_sleep_phase",
        "radar_sleep_phase_level",
        "radar_sleep_score",
        "radar_sleep_quality",
        "radar_sleep_total_minutes",
        "radar_sleep_efficiency_pct",
        "radar_terminal_fragmented_rest",
        "radar_rem_downgraded_to_light",
        "radar_direct_transition_bridge_to_light",
        "radar_sleep_onset",
        "radar_wake",
        "radar_hr_smooth_for_phase",
        "radar_rr_smooth_for_phase",
        "radar_hr_var_10min",
        "radar_rr_var_10min",
    ]
    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        merged[[col for col in csv_cols if col in merged.columns]].to_csv(csv_output, index=False)
    if score_output is not None and radar_score is not None:
        score_output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(radar_score)
        payload["onset_time"] = classified.frame["clock_time"].iloc[classified.onset_idx]
        payload["wake_time"] = classified.frame["clock_time"].iloc[classified.wake_idx]
        payload["onset_reason"] = classified.onset_reason
        payload["wake_reason"] = classified.wake_reason
        payload["terminal_rest_start_time"] = (
            classified.frame["clock_time"].iloc[classified.terminal_rest_start_idx] if classified.terminal_rest_start_idx is not None else None
        )
        payload["terminal_rest_reason"] = classified.terminal_rest_reason
        resting_hr_bpm, resting_hr_samples = _estimate_resting_hr_bpm(classified)
        payload["estimated_resting_hr_bpm"] = resting_hr_bpm
        payload["estimated_resting_hr_samples"] = resting_hr_samples
        payload["estimated_resting_hr_method"] = "10th percentile of quiet non-awake radar_hr_smooth_for_phase windows during the inferred sleep interval"
        score_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    times = pd.to_datetime(classified.frame["clock_time_dt"])
    start_time = times.iloc[0]
    end_time = times.iloc[-1]
    onset_time = times.iloc[classified.onset_idx]
    wake_time = times.iloc[classified.wake_idx]
    garmin_start, garmin_stop = _garmin_sleep_bounds(data)

    garmin_summary_minutes = {
        "awake": _duration_to_minutes(summary.get("Awake Time")),
        "light": _duration_to_minutes(summary.get("Light Sleep Duration")),
        "deep": _duration_to_minutes(summary.get("Deep Sleep Duration")),
        "rem": _duration_to_minutes(summary.get("REM Duration")),
    }

    fig = plt.figure(figsize=(18, 12), dpi=dpi)
    grid = fig.add_gridspec(4, 1, height_ratios=[0.95, 1.25, 1.35, 0.85], hspace=0.24, top=0.82)
    title = (
        f"A121 radar vs Garmin FIT sleep vitals/phases\n"
        f"Radar recording {start_time:%Y-%m-%d %H:%M}–{end_time:%H:%M} local | "
        f"radar sleep onset {onset_time:%H:%M}, wake/end {wake_time:%H:%M}"
    )
    if garmin_start is not None and garmin_stop is not None:
        title += f" | Garmin sleep event {garmin_start:%H:%M}–{garmin_stop:%H:%M}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax0 = fig.add_subplot(grid[0])
    _plot_step(
        ax0,
        classified.frame["clock_time_dt"],
        classified.frame["radar_sleep_phase_level"],
        step_s=classified.step_s,
        color="#7c3aed",
        lw=2.4,
        label="Radar heuristic phase",
    )
    _plot_garmin_sleep_levels(ax0, data, step_s=classified.step_s)
    ax0.axvline(onset_time, color="#16a34a", lw=1.4, alpha=0.90, label="radar sleep onset")
    ax0.axvline(wake_time, color="#ea580c", lw=1.4, alpha=0.90, label="radar wake/end")
    if classified.terminal_rest_start_idx is not None:
        terminal_rest_time = times.iloc[classified.terminal_rest_start_idx]
        ax0.axvline(terminal_rest_time, color="#f59e0b", lw=1.2, linestyle=":", alpha=0.90, label="terminal quiet rest: REM→light")
    if garmin_start is not None:
        ax0.axvline(garmin_start, color="#111827", lw=1.0, alpha=0.45)
    if garmin_stop is not None:
        ax0.axvline(garmin_stop, color="#111827", lw=1.0, alpha=0.45)
    ax0.set_yticks([0, 1, 2, 3])
    ax0.set_yticklabels(["Deep", "Light", "REM", "Awake"])
    ax0.set_ylim(-0.35, 3.35)
    ax0.set_title("Sleep phase timeline (radar heuristic plus decoded Garmin FIT stage changes)")
    ax0.grid(True, alpha=0.30)
    ax0.legend(loc="upper right", ncol=4, fontsize=8)

    ax1 = fig.add_subplot(grid[1], sharex=ax0)
    ax1.plot(times, classified.frame["rr_bpm_connected"], color="#15803d", lw=2.0, label="A121 RR final")
    ax1.scatter(times, classified.frame["rr_bpm"], s=8, color="#86efac", alpha=0.35, label="A121 RR raw/smoothed windows")
    if not data.respiration_rate.empty:
        rr = data.respiration_rate.copy()
        rr["local_time"] = pd.to_datetime(rr["local_time"])
        ax1.plot(rr["local_time"], rr["garmin_rr_bpm"], color="#0f766e", lw=1.5, alpha=0.85, label="Garmin RR FIT")
    ax1.set_ylabel("Respiratory rate (brpm)")
    ax1.set_ylim(5, 24)
    ax1.set_title("Respiration overlay")
    ax1.grid(True, alpha=0.30)
    ax1.legend(loc="upper right", ncol=3, fontsize=8)

    ax2 = fig.add_subplot(grid[2], sharex=ax0)
    ax2.plot(times, classified.frame["hr_bpm_plot_connected"], color="#b91c1c", lw=2.2, alpha=0.88, label="A121 HR display line")
    ax2.scatter(times, classified.frame["hr_bpm_raw"] if "hr_bpm_raw" in classified.frame else classified.frame["hr_bpm"], s=9, color="#fca5a5", alpha=0.35, label="A121 HR valid windows")
    if not data.heart_rate.empty:
        hr = data.heart_rate.copy()
        hr["local_time"] = pd.to_datetime(hr["local_time"])
        ax2.plot(hr["local_time"], hr["garmin_hr_bpm"], color="#1d4ed8", lw=1.45, alpha=0.82, label="Garmin HR FIT")
    hr_vals = pd.concat([
        pd.to_numeric(classified.frame.get("hr_bpm_plot_connected", pd.Series(dtype=float)), errors="coerce"),
        data.heart_rate["garmin_hr_bpm"] if not data.heart_rate.empty else pd.Series(dtype=float),
    ]).dropna()
    if not hr_vals.empty:
        lo = max(35, math.floor(float(np.nanpercentile(hr_vals, 1)) - 4))
        hi = min(110, math.ceil(float(np.nanpercentile(hr_vals, 99)) + 5))
        if hi - lo < 18:
            mid = 0.5 * (hi + lo)
            lo, hi = mid - 9, mid + 9
        ax2.set_ylim(lo, hi)
    ax2.set_ylabel("Heart rate (bpm)")
    ax2.set_title("Heart-rate overlay")
    ax2.grid(True, alpha=0.30)
    ax2.legend(loc="upper right", ncol=3, fontsize=8)

    ax3 = fig.add_subplot(grid[3], sharex=ax0)
    ax3.plot(times, classified.frame["still_fraction"], color="#16a34a", lw=1.2, label="still fraction")
    ax3.plot(times, classified.frame["movement_fraction"], color="#f59e0b", lw=1.2, label="movement fraction")
    ax3.plot(times, classified.frame["no_presence_fraction"], color="#6b7280", lw=1.0, alpha=0.85, label="no-presence fraction")
    ax3.set_ylim(-0.03, 1.03)
    ax3.set_ylabel("Fraction")
    ax3.set_xlabel("Clock time")
    ax3.set_title("Radar movement/presence signals used for sleep/wake detection")
    ax3.grid(True, alpha=0.30)
    ax3.legend(loc="upper right", ncol=3, fontsize=8)

    for axis in (ax0, ax1, ax2, ax3):
        axis.set_xlim(start_time, end_time)
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))

    if radar_score is not None:
        radar_summary = (
            f"Radar sleep score: {radar_score.score} ({radar_score.quality}); "
            f"sleep {_format_minutes(radar_score.total_sleep_minutes)} "
            f"[awake {_format_minutes(radar_minutes['awake'])}, "
            f"light {_format_minutes(radar_minutes['light'])}, "
            f"deep {_format_minutes(radar_minutes['deep'])}, "
            f"REM {_format_minutes(radar_minutes['rem'])}]"
        )
        score_note = f"Score components raw {radar_score.raw_score:.1f}, duration cap {radar_score.duration_cap:.1f}."
    else:
        radar_summary = (
            "Radar sleep phases: "
            f"awake {_format_minutes(radar_minutes['awake'])}, "
            f"light {_format_minutes(radar_minutes['light'])}, "
            f"deep {_format_minutes(radar_minutes['deep'])}, "
            f"REM {_format_minutes(radar_minutes['rem'])}"
        )
        score_note = "Sleep score disabled."
    garmin_summary = ""
    if any(value is not None for value in garmin_summary_minutes.values()):
        garmin_summary = (
            "\nGarmin Connect summary: "
            f"awake {_format_minutes(garmin_summary_minutes['awake'])}, "
            f"light {_format_minutes(garmin_summary_minutes['light'])}, "
            f"deep {_format_minutes(garmin_summary_minutes['deep'])}, "
            f"REM {_format_minutes(garmin_summary_minutes['rem'])}, "
            f"score {summary.get('Sleep Score', '--')} ({summary.get('Quality', '--')})"
        )
    terminal_note = ""
    if classified.terminal_rest_start_idx is not None:
        terminal_note = f"  Terminal quiet-rest from {times.iloc[classified.terminal_rest_start_idx]:%H:%M}: REM candidates downgraded to light."
    notes = (
        f"{radar_summary}{garmin_summary}\n"
        f"FIT extracted: {len(data.heart_rate)} HR samples, {len(data.respiration_rate)} RR samples, "
        f"{len(data.sleep_levels)} sleep-level changes.  "
        f"{score_note}{terminal_note}\n"
        f"Onset rule: {classified.onset_reason}; wake rule: {classified.wake_reason}."
    )
    fig.text(0.5, 0.895, notes, ha="center", va="top", fontsize=9, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88})

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output, csv_output, score_output if radar_score is not None else None


def main() -> None:
    args = _parse_args()
    default_png, default_csv, default_score = _default_outputs(args.trend_csv, garmin=True)
    png = args.output or default_png
    csv = args.csv_output or default_csv
    score_json = args.score_output or default_score
    output, csv_output, score_output = create_plot(args.trend_csv, args.garmin_dir, png, csv, score_json, dpi=args.dpi)
    print(f"Saved overlay plot: {output}")
    print(f"Saved merged CSV: {csv_output}")
    if score_output is not None:
        print(f"Saved radar sleep score: {score_output}")


if __name__ == "__main__":
    main()
