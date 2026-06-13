#!/usr/bin/env python
"""Gated long-duration A121 sleep HR/RR analysis.

Rules implemented for sleep vitals:
- no presence -> no vitals
- presence + movement/reacquisition -> no vitals at that time
- short movement gaps are connected by interpolation for plotting/CSV
- long movement/no-presence gaps stay as breaks

The script intentionally does not classify sleep phases.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
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

import respi_net.a121_vitals as a121_vitals
from respi_net.a121_vitals import analyze_a121_vitals, sample_rate_from_ms

STATE_NO_PRESENCE = 0
STATE_MOVEMENT = 1
STATE_STILL = 2
STATE_LABELS = {
    STATE_NO_PRESENCE: "no_presence",
    STATE_MOVEMENT: "movement",
    STATE_STILL: "still",
}

SCALAR_COLUMNS = [
    "Timestamp_ms",
    "Frame",
    "PeakDistance_m",
    "PeakAmplitude",
    "MeanAmplitude",
    "AcconeerAppState",
    "AcconeerPresenceDetected",
    "AcconeerPresenceDistance_m",
    "AcconeerTargetDistance_m",
    "AcconeerRangeStartIndex",
    "AcconeerRangeEndIndex",
    "AcconeerRangeStart_m",
    "AcconeerRangeEnd_m",
    "AcconeerBreathingRate_BPM",
]


@dataclass(frozen=True)
class Segment:
    state: int
    start_s: float
    end_s: float
    start_idx: int
    end_idx: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class WindowPlan:
    index: int
    end_s: float
    start_s: float
    start_idx: int
    end_idx: int
    end_state: int
    still_fraction: float
    movement_fraction: float
    no_presence_fraction: float
    long_movement_fraction: float
    allowed: bool
    blocked_reason: str


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _state_codes(scalar_df: pd.DataFrame) -> np.ndarray:
    state_text = scalar_df.get("AcconeerAppState", pd.Series([""] * len(scalar_df))).fillna("").astype(str)
    state_upper = state_text.str.upper()
    presence = _numeric(scalar_df.get("AcconeerPresenceDetected", pd.Series([0] * len(scalar_df)))) > 0
    no_presence = (~presence) | state_upper.str.startswith("NO_PRESENCE").to_numpy(dtype=bool)
    still = presence & state_text.eq("ESTIMATE_BREATHING_RATE").to_numpy(dtype=bool) & (~no_presence)
    codes = np.full(len(scalar_df), STATE_MOVEMENT, dtype=np.int8)
    codes[no_presence] = STATE_NO_PRESENCE
    codes[still] = STATE_STILL
    return codes


def _segments(time_s: np.ndarray, codes: np.ndarray, fs: float) -> list[Segment]:
    if len(time_s) == 0:
        return []
    out: list[Segment] = []
    start = 0
    sample_dt = 1.0 / max(fs, 1e-9)
    for idx in range(1, len(codes)):
        if int(codes[idx]) != int(codes[start]):
            out.append(Segment(int(codes[start]), float(time_s[start]), float(time_s[idx - 1] + sample_dt), start, idx - 1))
            start = idx
    out.append(Segment(int(codes[start]), float(time_s[start]), float(time_s[-1] + sample_dt), start, len(codes) - 1))
    return out


def _long_movement_mask(n: int, segments: Iterable[Segment], threshold_s: float) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for seg in segments:
        if seg.state == STATE_MOVEMENT and seg.duration_s >= threshold_s:
            mask[seg.start_idx : seg.end_idx + 1] = True
    return mask


def _make_window_plans(
    time_s: np.ndarray,
    codes: np.ndarray,
    *,
    window_s: float,
    step_s: float,
    min_still_fraction: float,
    max_no_presence_fraction: float,
    short_movement_s: float,
    long_movement_mask: np.ndarray,
) -> list[WindowPlan]:
    if len(time_s) == 0:
        return []
    duration_s = float(time_s[-1])
    ends = np.arange(window_s, duration_s + 1e-9, step_s, dtype=float)
    plans: list[WindowPlan] = []
    for idx, end_s in enumerate(ends):
        start_s = float(max(0.0, end_s - window_s))
        start_idx = int(np.searchsorted(time_s, start_s, side="left"))
        end_idx = int(np.searchsorted(time_s, end_s, side="right") - 1)
        if end_idx < start_idx or end_idx < 0:
            continue
        end_idx = min(end_idx, len(time_s) - 1)
        win = codes[start_idx : end_idx + 1]
        still_fraction = float(np.mean(win == STATE_STILL)) if len(win) else 0.0
        movement_fraction = float(np.mean(win == STATE_MOVEMENT)) if len(win) else 0.0
        no_presence_fraction = float(np.mean(win == STATE_NO_PRESENCE)) if len(win) else 0.0
        long_movement_fraction = float(np.mean(long_movement_mask[start_idx : end_idx + 1])) if len(win) else 0.0
        end_state = int(codes[end_idx])
        allowed = True
        reason = "valid_window"
        if end_state == STATE_NO_PRESENCE:
            allowed = False
            reason = "no_presence_at_window_end"
        elif end_state == STATE_MOVEMENT:
            allowed = False
            reason = "movement_at_window_end"
        elif no_presence_fraction > max_no_presence_fraction:
            allowed = False
            reason = "no_presence_inside_window"
        elif long_movement_fraction > 0.0:
            allowed = False
            reason = "long_movement_inside_window"
        elif still_fraction < min_still_fraction:
            allowed = False
            reason = "too_much_motion_inside_window"
        plans.append(
            WindowPlan(
                index=idx,
                end_s=float(end_s),
                start_s=start_s,
                start_idx=start_idx,
                end_idx=end_idx,
                end_state=end_state,
                still_fraction=still_fraction,
                movement_fraction=movement_fraction,
                no_presence_fraction=no_presence_fraction,
                long_movement_fraction=long_movement_fraction,
                allowed=allowed,
                blocked_reason=reason,
            )
        )
    return plans


def _robust_window_rr(rr_bpm: np.ndarray, codes: np.ndarray, start_idx: int, end_idx: int, fs: float) -> tuple[float, int]:
    rr = rr_bpm[start_idx : end_idx + 1]
    state = codes[start_idx : end_idx + 1]
    mask = (state == STATE_STILL) & np.isfinite(rr) & (rr >= 6.0) & (rr <= 30.0)
    vals = rr[mask]
    min_samples = max(10, int(round(fs * 5.0)))
    if len(vals) < min_samples:
        return math.nan, int(len(vals))
    # Trim occasional harmonic spikes before taking the median.
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med))) + 1e-9
    trimmed = vals[np.abs(vals - med) <= max(4.0, 4.0 * 1.4826 * mad)]
    if len(trimmed) >= min_samples:
        vals = trimmed
    return float(np.nanmedian(vals)), int(len(vals))


def _rolling_median_keep_nan(values: np.ndarray, width: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if width <= 1 or len(x) == 0:
        return x.copy()
    out = np.full(len(x), np.nan, dtype=float)
    half = width // 2
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        lo = max(0, i - half)
        hi = min(len(x), i + half + 1)
        vals = x[lo:hi]
        vals = vals[np.isfinite(vals)]
        out[i] = float(np.nanmedian(vals)) if len(vals) else x[i]
    return out


def _correct_rr_high_harmonics(values: np.ndarray, *, high_bpm: float = 22.0) -> np.ndarray:
    """Fold obvious high RR harmonics back to the local breathing-rate track.

    The persisted Acconeer breathing rate occasionally jumps to ~2x the visible sleep breathing
    rhythm while the person is still.  We only fold high values down by 2 when that is closer to
    the recent/global RR track; low values are left untouched because slow sleep breathing can be
    real.
    """
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return out
    seed_values = np.asarray([v * 0.5 if v >= high_bpm and 6.0 <= v * 0.5 <= 30.0 else v for v in finite], dtype=float)
    target = float(np.nanmedian(seed_values)) if len(seed_values) else 14.0
    for idx, value in enumerate(x):
        if not np.isfinite(value):
            continue
        candidates = [float(value)]
        folded = float(value) * 0.5
        if value >= high_bpm and 6.0 <= folded <= 30.0:
            candidates.append(folded)
        best = min(candidates, key=lambda candidate: abs(candidate - target) + (0.10 if candidate != value else 0.0))
        out[idx] = best
        target = 0.85 * target + 0.15 * best
    return out


def _movement_overlap_ok(
    segments: list[Segment],
    left_s: float,
    right_s: float,
    *,
    short_movement_s: float,
) -> bool:
    total_movement = 0.0
    for seg in segments:
        if seg.end_s < left_s or seg.start_s > right_s:
            continue
        overlap = max(0.0, min(right_s, seg.end_s) - max(left_s, seg.start_s))
        if overlap <= 0:
            continue
        if seg.state == STATE_NO_PRESENCE:
            return False
        if seg.state == STATE_MOVEMENT:
            if seg.duration_s >= short_movement_s:
                return False
            total_movement += overlap
    return total_movement > 0.0 and total_movement <= short_movement_s


def _connect_short_movement_gaps(
    times_s: np.ndarray,
    values: np.ndarray,
    segments: list[Segment],
    *,
    short_movement_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaN runs only when the underlying missing interval is short movement."""
    y = np.asarray(values, dtype=float).copy()
    filled = np.zeros(len(y), dtype=bool)
    finite = np.isfinite(y)
    i = 0
    while i < len(y):
        if finite[i]:
            i += 1
            continue
        run_start = i
        while i < len(y) and not finite[i]:
            i += 1
        run_end = i - 1
        left = run_start - 1
        right = run_end + 1
        if left < 0 or right >= len(y) or not finite[left] or not finite[right]:
            continue
        if not _movement_overlap_ok(segments, float(times_s[left]), float(times_s[right]), short_movement_s=short_movement_s):
            continue
        interp_t = times_s[run_start : run_end + 1]
        y[run_start : run_end + 1] = np.interp(interp_t, [times_s[left], times_s[right]], [y[left], y[right]])
        filled[run_start : run_end + 1] = True
    return y, filled


def _connect_all_gaps(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill all missing samples for a continuous display line.

    This is for visualization only: raw/gated values remain in separate columns.  Interior gaps are
    linearly interpolated; leading/trailing gaps use the nearest finite value.
    """
    x = np.asarray(values, dtype=float)
    y = x.copy()
    finite = np.isfinite(x)
    filled = ~finite
    if not np.any(finite):
        return y, filled
    idx = np.arange(len(x), dtype=float)
    y[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return y, filled


def _local_datetimes_from_ms(ms: np.ndarray | pd.Series) -> list[datetime]:
    return [datetime.fromtimestamp(float(value) / 1000.0) for value in np.asarray(ms, dtype=float)]


def _zoom_ylim(values: np.ndarray, *, lower_bound: float, upper_bound: float, pad: float = 3.0, min_span: float = 14.0) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return lower_bound, upper_bound
    lo = float(np.nanpercentile(vals, 1.0) - pad)
    hi = float(np.nanpercentile(vals, 99.0) + pad)
    if hi - lo < min_span:
        center = 0.5 * (lo + hi)
        lo = center - min_span * 0.5
        hi = center + min_span * 0.5
    return max(lower_bound, math.floor(lo)), min(upper_bound, math.ceil(hi))


def _shade_segments(
    ax: plt.Axes,
    segments: list[Segment],
    *,
    short_movement_s: float,
    xscale: float = 60.0,
    start_timestamp_ms: float | None = None,
) -> None:
    labels_done: set[str] = set()
    for seg in segments:
        if seg.state == STATE_STILL:
            continue
        if seg.state == STATE_NO_PRESENCE:
            color = "#6b7280"
            alpha = 0.20
            label = "no presence"
        elif seg.duration_s >= short_movement_s:
            color = "#dc2626"
            alpha = 0.18
            label = f"long movement ≥{short_movement_s:.0f}s"
        else:
            color = "#f59e0b"
            alpha = 0.13
            label = "short movement"
        if start_timestamp_ms is None:
            x0 = seg.start_s / xscale
            x1 = seg.end_s / xscale
        else:
            x0 = datetime.fromtimestamp((start_timestamp_ms + seg.start_s * 1000.0) / 1000.0)
            x1 = datetime.fromtimestamp((start_timestamp_ms + seg.end_s * 1000.0) / 1000.0)
        ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, label=None if label in labels_done else label)
        labels_done.add(label)


def _downsample_indices(n: int, max_points: int = 5000) -> np.ndarray:
    step = max(1, int(math.ceil(n / max_points)))
    return np.arange(0, n, step, dtype=int)


def _finite_summary(values: np.ndarray) -> str:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return "no valid values"
    return (
        f"median {np.nanmedian(vals):.1f}, IQR "
        f"{np.nanpercentile(vals, 25):.1f}-{np.nanpercentile(vals, 75):.1f}, "
        f"range {np.nanmin(vals):.1f}-{np.nanmax(vals):.1f}"
    )


def analyze(
    csv_path: Path,
    output_dir: Path,
    *,
    window_s: float = 60.0,
    step_s: float = 30.0,
    short_movement_s: float = 30.0,
    min_still_fraction: float = 0.65,
    max_no_presence_fraction: float = 0.05,
    min_hr_confidence: float = 5.0,
    hr_min_bpm: float = 42.0,
    hr_max_bpm: float = 120.0,
    dpi: int = 140,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hr_min_bpm = float(hr_min_bpm)
    hr_max_bpm = float(hr_max_bpm)
    if not (0.0 < hr_min_bpm < hr_max_bpm):
        raise ValueError("HR band must satisfy 0 < hr_min_bpm < hr_max_bpm")
    # The A121 analyzer stores the heart band as a module-level Hz tuple.  Set it here so offline
    # runs can widen the range for non-human subjects (for example cats up to ~200 BPM).
    a121_vitals.HEART_BAND_HZ = (hr_min_bpm / 60.0, hr_max_bpm / 60.0)

    scalar_df = pd.read_csv(csv_path, usecols=lambda col: col in SCALAR_COLUMNS)
    if scalar_df.empty:
        raise ValueError(f"No rows in {csv_path}")

    timestamps_ms = _numeric(scalar_df["Timestamp_ms"])
    fs = sample_rate_from_ms(timestamps_ms, default=20.0)
    time_s = (timestamps_ms - float(timestamps_ms[0])) / 1000.0
    duration_s = float(time_s[-1]) if len(time_s) else 0.0
    codes = _state_codes(scalar_df)
    segments = _segments(time_s, codes, fs)
    long_movement = _long_movement_mask(len(codes), segments, short_movement_s)
    plans = _make_window_plans(
        time_s,
        codes,
        window_s=window_s,
        step_s=step_s,
        min_still_fraction=min_still_fraction,
        max_no_presence_fraction=max_no_presence_fraction,
        short_movement_s=short_movement_s,
        long_movement_mask=long_movement,
    )

    rr_frame = _numeric(scalar_df.get("AcconeerBreathingRate_BPM", pd.Series([math.nan] * len(scalar_df))))
    target_distance = _numeric(scalar_df.get("AcconeerTargetDistance_m", pd.Series([math.nan] * len(scalar_df))))
    peak_distance = _numeric(scalar_df.get("PeakDistance_m", pd.Series([math.nan] * len(scalar_df))))
    peak_amplitude = _numeric(scalar_df.get("PeakAmplitude", pd.Series([math.nan] * len(scalar_df))))

    results: list[dict[str, object]] = []
    rows: deque[dict[str, str]] = deque()
    next_plan = 0
    first_ts = float(timestamps_ms[0])
    prior_hr_hz: float | None = None
    last_valid_hr_s: float | None = None

    print(
        f"Streaming {csv_path.name}: {len(scalar_df)} rows, {duration_s / 3600.0:.2f} h, "
        f"fs {fs:.3f} Hz, {len(plans)} windows...",
        flush=True,
    )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            rows.append(row)
            row_t = (float(row["Timestamp_ms"]) - first_ts) / 1000.0
            # Keep a little slack so windows landing on timestamp jitter still have all rows.
            while rows and (float(rows[0]["Timestamp_ms"]) - first_ts) / 1000.0 < row_t - window_s - max(2.0, step_s):
                rows.popleft()

            while next_plan < len(plans) and row_t >= plans[next_plan].end_s:
                plan = plans[next_plan]
                status = STATE_LABELS[plan.end_state]
                analysis_status = plan.blocked_reason
                hr_bpm = math.nan
                hr_conf = math.nan
                rr_bpm = math.nan
                rr_conf = math.nan
                analysis_target_m = math.nan
                analysis_quality = math.nan
                candidate_bins = 0
                rr_source = "none"

                if plan.allowed:
                    win_rows = [
                        buffered
                        for buffered in rows
                        if plan.start_s - 1e-6 <= (float(buffered["Timestamp_ms"]) - first_ts) / 1000.0 <= plan.end_s + 1e-6
                    ]
                    if len(win_rows) >= max(30, int(round(fs * window_s * 0.50))):
                        # Reset the weak heart prior after a long gap so a stale lock cannot dominate.
                        if last_valid_hr_s is None or plan.end_s - last_valid_hr_s > max(600.0, 4.0 * window_s):
                            prior_hr_hz = None
                        work = pd.DataFrame(win_rows)
                        analysis = analyze_a121_vitals(
                            work,
                            max_frames=len(work),
                            heart_window_s=window_s,
                            heart_prior_hz=prior_hr_hz,
                            heart_prior_std_hz=0.20 if prior_hr_hz is not None else None,
                        )
                        hr_conf = float(analysis.heart_confidence)
                        rr_conf = float(analysis.resp_confidence)
                        analysis_target_m = float(analysis.target_distance_m)
                        analysis_quality = float(analysis.signal_quality)
                        candidate_bins = int(analysis.candidate_bins)
                        if min_hr_confidence <= hr_conf and hr_min_bpm <= analysis.heart_bpm <= hr_max_bpm:
                            candidate_hr = float(analysis.heart_bpm)
                            # Suppress weak, sudden HR jumps but allow high-confidence changes.
                            if prior_hr_hz is not None and abs(candidate_hr - prior_hr_hz * 60.0) > 24.0 and hr_conf < 15.0:
                                analysis_status = "heart_rejected_jump"
                            else:
                                hr_bpm = candidate_hr
                                prior_hr_hz = hr_bpm / 60.0
                                last_valid_hr_s = plan.end_s
                        rr_bpm, rr_count = _robust_window_rr(rr_frame, codes, plan.start_idx, plan.end_idx, fs)
                        if np.isfinite(rr_bpm):
                            rr_source = f"recorded_median_{rr_count}_samples"
                        elif 6.0 <= analysis.resp_bpm <= 30.0 and analysis.resp_confidence >= 4.0:
                            rr_bpm = float(analysis.resp_bpm)
                            rr_source = "analyzer_fallback"
                        if np.isfinite(hr_bpm) or np.isfinite(rr_bpm):
                            analysis_status = "valid"
                        elif analysis_status == "valid_window":
                            analysis_status = "low_quality_no_vitals"
                    else:
                        analysis_status = "not_enough_rows_in_window"

                results.append(
                    {
                        "elapsed_s": float(plan.end_s),
                        "elapsed_min": float(plan.end_s / 60.0),
                        "state": status,
                        "analysis_status": analysis_status,
                        "window_start_s": float(plan.start_s),
                        "window_rows": int(plan.end_idx - plan.start_idx + 1),
                        "still_fraction": float(plan.still_fraction),
                        "movement_fraction": float(plan.movement_fraction),
                        "no_presence_fraction": float(plan.no_presence_fraction),
                        "long_movement_fraction": float(plan.long_movement_fraction),
                        "hr_bpm_raw": float(hr_bpm) if np.isfinite(hr_bpm) else math.nan,
                        "hr_confidence": float(hr_conf) if np.isfinite(hr_conf) else math.nan,
                        "rr_bpm_raw": float(rr_bpm) if np.isfinite(rr_bpm) else math.nan,
                        "rr_confidence": float(rr_conf) if np.isfinite(rr_conf) else math.nan,
                        "rr_source": rr_source,
                        "target_distance_m": float(target_distance[plan.end_idx]) if np.isfinite(target_distance[plan.end_idx]) else math.nan,
                        "peak_distance_m": float(peak_distance[plan.end_idx]) if np.isfinite(peak_distance[plan.end_idx]) else math.nan,
                        "peak_amplitude": float(peak_amplitude[plan.end_idx]) if np.isfinite(peak_amplitude[plan.end_idx]) else math.nan,
                        "analysis_target_distance_m": analysis_target_m,
                        "signal_quality": analysis_quality,
                        "candidate_bins": candidate_bins,
                    }
                )
                next_plan += 1
                if next_plan % 50 == 0 or next_plan == len(plans):
                    print(f"  windows {next_plan}/{len(plans)}", flush=True)

            if next_plan >= len(plans):
                break

    trend = pd.DataFrame(results)
    if trend.empty:
        raise ValueError("No analysis windows were produced")

    trend_t = trend["elapsed_s"].to_numpy(dtype=float)
    hr_raw = trend["hr_bpm_raw"].to_numpy(dtype=float)
    rr_raw = trend["rr_bpm_raw"].to_numpy(dtype=float)
    hr_smooth = _rolling_median_keep_nan(hr_raw, 3)
    rr_corrected = _correct_rr_high_harmonics(rr_raw)
    rr_smooth = _rolling_median_keep_nan(rr_corrected, 3)
    hr_connected, hr_interpolated = _connect_short_movement_gaps(trend_t, hr_smooth, segments, short_movement_s=short_movement_s)
    rr_connected, rr_interpolated = _connect_short_movement_gaps(trend_t, rr_smooth, segments, short_movement_s=short_movement_s)
    hr_plot_connected, hr_plot_interpolated = _connect_all_gaps(hr_connected)
    trend_timestamp_ms = first_ts + trend_t * 1000.0
    trend["timestamp_ms"] = trend_timestamp_ms
    trend["clock_time"] = [value.isoformat(timespec="seconds") for value in _local_datetimes_from_ms(trend_timestamp_ms)]
    trend["hr_bpm"] = hr_smooth
    trend["rr_bpm_harmonic_corrected"] = rr_corrected
    trend["rr_bpm"] = rr_smooth
    trend["hr_bpm_connected"] = hr_connected
    trend["hr_bpm_plot_connected"] = hr_plot_connected
    trend["rr_bpm_connected"] = rr_connected
    trend["hr_interpolated_short_movement"] = hr_interpolated
    trend["hr_interpolated_plot"] = hr_plot_interpolated
    trend["rr_interpolated_short_movement"] = rr_interpolated

    csv_out = output_dir / f"{csv_path.stem}_gated_sleep_vitals.csv"
    trend.to_csv(csv_out, index=False)

    # Plot.
    start_dt = datetime.fromtimestamp(first_ts / 1000.0)
    end_dt = datetime.fromtimestamp((first_ts + duration_s * 1000.0) / 1000.0)
    trend_times = _local_datetimes_from_ms(trend_timestamp_ms)
    trend_times_arr = np.asarray(trend_times, dtype=object)
    ds = _downsample_indices(len(time_s), max_points=4500)
    sample_times = _local_datetimes_from_ms(timestamps_ms[ds])
    hr_ylim = _zoom_ylim(hr_plot_connected, lower_bound=max(0.0, hr_min_bpm - 5.0), upper_bound=hr_max_bpm + 10.0, pad=3.0, min_span=16.0)

    fig = plt.figure(figsize=(18, 11), dpi=dpi)
    # Larger HR panel + headroom for title/summary.
    grid = fig.add_gridspec(4, 1, height_ratios=[0.70, 0.90, 1.70, 0.55], hspace=0.30, top=0.80)
    fig.suptitle(
        f"A121 gated sleep vitals: {csv_path.name}\n"
        f"{start_dt:%Y-%m-%d %H:%M}–{end_dt:%H:%M} local | "
        f"{duration_s / 3600.0:.2f} h, {len(scalar_df)} rows, {fs:.2f} Hz | "
        f"HR {window_s:.0f}s batch / {step_s:.0f}s step / band {hr_min_bpm:.0f}-{hr_max_bpm:.0f} BPM",
        fontsize=14,
        fontweight="bold",
    )

    ax0 = fig.add_subplot(grid[0])
    _shade_segments(ax0, segments, short_movement_s=short_movement_s, start_timestamp_ms=first_ts)
    ax0.plot(sample_times, peak_distance[ds], color="#0f172a", lw=0.75, alpha=0.70, label="peak distance")
    if np.isfinite(target_distance).any():
        ax0.plot(sample_times, target_distance[ds], color="#2563eb", lw=0.9, alpha=0.85, label="Acconeer target")
    ax0.set_ylabel("Distance (m)")
    ax0.set_title("Presence / movement gate and range tracking")
    ax0.grid(True, alpha=0.35)
    ax0.set_xlim(start_dt, end_dt)
    ax0.legend(loc="upper right", ncol=5, fontsize=8)

    ax1 = fig.add_subplot(grid[1], sharex=ax0)
    _shade_segments(ax1, segments, short_movement_s=short_movement_s, start_timestamp_ms=first_ts)
    ax1.scatter(trend_times, trend["rr_bpm_raw"], s=10, color="#86efac", alpha=0.35, label="RR raw valid window")
    ax1.plot(trend_times, trend["rr_bpm_connected"], color="#15803d", lw=1.8, label="RR final (smoothed, high harmonics folded)")
    interp_rr = trend["rr_interpolated_short_movement"].to_numpy(dtype=bool)
    if np.any(interp_rr):
        ax1.scatter(trend_times_arr[interp_rr], trend.loc[interp_rr, "rr_bpm_connected"], s=28, facecolors="none", edgecolors="#14532d", linewidths=1.0, label="RR interpolated")
    ax1.set_ylabel("Respiratory rate (BPM)")
    ax1.set_ylim(5, 24)
    ax1.set_title("Respiratory rate: no presence/movement excluded")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="upper right", ncol=4, fontsize=8)

    ax2 = fig.add_subplot(grid[2], sharex=ax0)
    _shade_segments(ax2, segments, short_movement_s=short_movement_s, start_timestamp_ms=first_ts)
    ax2.scatter(trend_times, trend["hr_bpm_raw"], s=12, color="#fca5a5", alpha=0.45, label="HR raw valid 60s window")
    ax2.plot(trend_times, trend["hr_bpm_plot_connected"], color="#b91c1c", lw=2.3, label="HR continuous display line")
    ax2.set_ylabel("Heart rate (BPM)")
    ax2.set_ylim(*hr_ylim)
    ax2.set_title("Heart rate: connected for display, zoomed")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right", ncol=3, fontsize=8)

    ax3 = fig.add_subplot(grid[3], sharex=ax0)
    _shade_segments(ax3, segments, short_movement_s=short_movement_s, start_timestamp_ms=first_ts)
    ax3.plot(trend_times, trend["still_fraction"], color="#16a34a", lw=1.1, label="still fraction in window")
    ax3.plot(trend_times, trend["movement_fraction"], color="#f59e0b", lw=1.1, label="movement fraction")
    if trend["hr_confidence"].notna().any():
        hr_conf_norm = np.clip(trend["hr_confidence"].to_numpy(dtype=float) / 30.0, 0.0, 1.0)
        ax3.plot(trend_times, hr_conf_norm, color="#dc2626", lw=0.9, alpha=0.75, label="HR confidence / 30")
    ax3.set_ylim(-0.03, 1.03)
    ax3.set_xlabel("Clock time (HH:MM from Timestamp_ms)")
    ax3.set_ylabel("Fraction / norm")
    ax3.set_title("Window quality")
    ax3.grid(True, alpha=0.35)
    ax3.legend(loc="upper right", ncol=4, fontsize=8)

    for axis in (ax0, ax1, ax2, ax3):
        axis.set_xlim(start_dt, end_dt)
        axis.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axis.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))

    summary = (
        f"HR raw: {_finite_summary(hr_raw)} | displayed y-range {hr_ylim[0]:.0f}–{hr_ylim[1]:.0f} BPM\n"
        f"RR final: {_finite_summary(rr_connected)} (high harmonics folded)\n"
        "HR line is continuous for readability; raw/gated HR points remain shown separately."
    )
    fig.text(0.5, 0.895, summary, ha="center", va="top", fontsize=9, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82})

    png_out = output_dir / f"{csv_path.stem}_gated_sleep_vitals.png"
    fig.savefig(png_out, bbox_inches="tight")
    plt.close(fig)

    state_minutes = {
        label: sum(seg.duration_s for seg in segments if seg.state == code) / 60.0
        for code, label in STATE_LABELS.items()
    }
    movement_durations = np.asarray([seg.duration_s for seg in segments if seg.state == STATE_MOVEMENT], dtype=float)
    short_count = int(np.sum(movement_durations < short_movement_s)) if len(movement_durations) else 0
    long_count = int(np.sum(movement_durations >= short_movement_s)) if len(movement_durations) else 0
    print(f"Saved plot: {png_out}")
    print(f"Saved trend CSV: {csv_out}")
    print(
        "State durations: "
        f"still {state_minutes['still']:.1f} min, "
        f"movement {state_minutes['movement']:.1f} min, "
        f"no presence {state_minutes['no_presence']:.1f} min"
    )
    print(f"Movement episodes: {len(movement_durations)} total; {short_count} short, {long_count} long (threshold {short_movement_s:.0f}s)")
    print(f"HR windows: {np.isfinite(hr_raw).sum()}/{len(hr_raw)} valid | {_finite_summary(hr_raw)}")
    print(f"RR windows: {np.isfinite(rr_raw).sum()}/{len(rr_raw)} valid | raw {_finite_summary(rr_raw)} | final {_finite_summary(rr_connected)}")
    return png_out, csv_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create gated A121 sleep HR/RR graph; no sleep-phase classification.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "plots")
    parser.add_argument("--window-s", type=float, default=60.0, help="Trailing analysis window length in seconds")
    parser.add_argument("--step-s", type=float, default=30.0, help="Trend spacing in seconds")
    parser.add_argument("--short-movement-s", type=float, default=30.0, help="Movement gaps shorter than this are connected")
    parser.add_argument("--min-still-fraction", type=float, default=0.65, help="Minimum still samples inside an analysis window")
    parser.add_argument("--max-no-presence-fraction", type=float, default=0.05, help="Maximum no-presence samples tolerated inside a window")
    parser.add_argument("--min-hr-confidence", type=float, default=5.0)
    parser.add_argument("--hr-min-bpm", type=float, default=42.0, help="Minimum HR accepted/analyzed, in BPM")
    parser.add_argument("--hr-max-bpm", type=float, default=120.0, help="Maximum HR accepted/analyzed, in BPM")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    analyze(
        args.csv_path,
        args.output_dir,
        window_s=args.window_s,
        step_s=args.step_s,
        short_movement_s=args.short_movement_s,
        min_still_fraction=args.min_still_fraction,
        max_no_presence_fraction=args.max_no_presence_fraction,
        min_hr_confidence=args.min_hr_confidence,
        hr_min_bpm=args.hr_min_bpm,
        hr_max_bpm=args.hr_max_bpm,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
