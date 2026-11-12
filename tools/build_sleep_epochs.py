#!/usr/bin/env python
"""One row per 30 s of every A121 night: breathing, heart and movement features, and the watch's values.

Input per night (``tools/analyze_a121_sleep_vitals_gated.py --hr-min-bpm 36`` and ``tools/extract_a121_night.py``,
both into ``data/processed/a121_nights/``): the gated trend (a 60 s window every 30 s: heart rate, Acconeer's
breathing rate, still / movement / no-presence shares) and the compact IQ of the night.  Each trend window is one
epoch, and from the IQ of its 60 s the chest displacement is formed -- the 3 bins around Acconeer's target,
unwrapped and weighted by echo -- for:

* ``breath_amp_mm`` -- 5-95th percentile spread of the breathing band (0.1-0.7 Hz),
* ``breath_regularity`` -- the dominant rhythm's share of the 0.1-2.5 Hz power (:func:`~respi_net.phase_baseline.window_rhythm`),
* ``breath_interval_cv`` / ``breath_depth_cv`` -- variation of the breath-to-breath intervals and depths,
* ``motion_energy`` -- log10 of the frame-to-frame displacement scatter (fidgeting, turning over).

Every base feature also gets its median and spread over the surrounding 10 min (``_med10`` / ``_sd10``), all are
scaled per night (median and inter-quartile range: the bed setup changes their size from night to night), and the
epoch keeps its time since the recording started.  ``heuristic_stage`` is the rule-based classifier of the sleep
chapter on the same trend.  Where a Garmin export exists, each epoch gets the watch's sleep stage at its centre
(only inside the watch's sleep interval) and its heart and breathing rate within 2 min.

    uv run python tools/build_sleep_epochs.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from respi_net.a121_sleep import _garmin_sleep_bounds, _merge_garmin_nearest, classify_radar_sleep, load_garmin_reference
from respi_net.chest_signal import A121_CHEST_SIGN, PHASE_TO_MM
from respi_net.garmin_fit import fit_datetime_local_naive, fit_paths_from_sources, parse_fit_files
from respi_net.phase_baseline import window_rhythm

NIGHT_DIR = ROOT / "data" / "processed" / "a121_nights"
GARMIN_DIR = ROOT / "data" / "raw" / "a121"
OUT = ROOT / "data" / "processed" / "sleep_epochs" / "epochs.csv"
STAGES = ("awake", "light", "deep", "rem")
GARMIN_STAGE = {1: "awake", 2: "light", 3: "deep", 4: "rem"}  # code 0: the watch could not tell
BREATH_BAND_HZ = (0.1, 0.7)
CONTEXT_EPOCHS = 21  # 10 min of 30 s epochs, centred


@dataclass(frozen=True)
class Night:
    name: str
    stem: str
    garmin: str | None
    setup: str


ALL_NIGHTS = (
    Night("2026-06-08", "a121_sleep_sleep-night-1_1-real_2026-06-08_02-50-10", "garmin_2026-06-08", "edge, plano-convex"),
    Night("2026-06-09", "a121_sleep_sleep-night-2-real_2026-06-09_02-18-14", "garmin_2026-06-09", "edge, FZP"),
    Night("2026-06-10", "a121_sleep_sleep-night-3-real-flatcover_2026-06-10_02-35-45", "garmin_2026-06-10", "flat cover, pilot"),
    Night("2026-06-11", "a121_sleep_sleep-night-4-real_2026-06-11_02-49-31", "garmin_reference_a121_sleep_sleep-night-4-real_2026-06-11_02-49-31", "hyperbolic, 1.6 m"),
    Night("2026-06-13", "a121_sleep_sleep-night-5-real_2026-06-13_02-58-40", "garmin_a121_sleep_sleep-night-5-real_2026-06-13_02-58-40", "flat cover, over bed"),
    Night("2026-06-15", "a121_sleep_sleep-night-5-real_2026-06-15_02-41-38", "garmin_2026-06-15", "flat cover, over bed"),
    Night("2026-06-16", "a121_sleep_sleep-night-7-real_2026-06-16_03-55-01", "garmin_2026-06-16", "flat cover, over bed"),
    Night("2026-10-11", "a121_sleep_noc-2026-10-11_2026-10-11_03-47-35", None, "October"),
    Night("2026-10-12", "a121_sleep_noc-2026-10-12_2026-10-12_03-31-31", None, "October"),
    Night("2026-10-13", "a121_sleep_noc-2026-10-13_2026-10-13_03-45-56", None, "October"),
    Night("2026-11-10", "a121_sleep_noc-suunto-2026-11-10_2026-11-10_04-03-05", "garmin_2026-11-10", "November, 0.5-1.8 m"),
    Night("2026-11-11", "a121_sleep_noc-2026-11-12_2026-11-11_01-30-46", "garmin_2026-11-11", "November, 0.2-1.2 m"),
    Night("2026-11-12", "a121_sleep_noc-zegarek-2026-11-12_2026-11-12_02-40-55", "garmin_2026-11-12", "November, 0.2-1.2 m"),
)
# Recorded but left out of every analysis (the night stays in ALL_NIGHTS for inspection, e.g. hr_distance_scan.py).
EXCLUDED = {
    "2026-11-10": "chest nearer than the recorded range: 0.5-1.8 m recorded, the chest at ~0.35-0.45 m (the 11 November "
                  "optimum with the same bed and radar; the night's best heart-rate bin was the range's nearest one)",
}
NIGHTS = tuple(n for n in ALL_NIGHTS if n.name not in EXCLUDED)
BASE_FEATURES = ("hr_bpm", "rr_bpm", "breath_amp_mm", "breath_regularity", "breath_interval_cv", "breath_depth_cv",
                 "motion_energy", "movement_fraction", "no_presence_fraction", "radar_rmssd_ms", "radar_sdnn_ms")
# The watch's own signals, to see which of them its sleep stages follow: heart rate and breathing rate every minute,
# its stress level every minute (Firstbeat's index from heart-rate variability, 0-100; negative = not measured) and
# the overnight heart-rate variability every 5 min (RMSSD, FIT message 371, ms x 128).
WATCH_FEATURES = ("garmin_hr_bpm", "garmin_rr_bpm", "garmin_stress", "garmin_hrv_ms")
WATCH_SETS = {
    "watch_hr": ("garmin_hr_bpm",),
    "watch_rr": ("garmin_rr_bpm",),
    "watch_hrv": ("garmin_stress", "garmin_hrv_ms"),
    "watch_all": WATCH_FEATURES,
}
FEATURE_SETS = {
    "breathing": ("rr_bpm", "breath_amp_mm", "breath_regularity", "breath_interval_cv", "breath_depth_cv"),
    "heart": ("hr_bpm",),
    "hrv": ("radar_rmssd_ms", "radar_sdnn_ms"),
    "movement": ("motion_energy", "movement_fraction", "no_presence_fraction", "minutes_since_movement"),
}


def chest_displacement(iq: np.ndarray, distances: np.ndarray, target_m: np.ndarray) -> np.ndarray:
    """Displacement in mm (rising on inhale) of the 3 bins around the window's median Acconeer target."""

    complex_iq = iq[..., 0].astype(np.float32) + 1j * iq[..., 1].astype(np.float32)
    amplitude = np.median(np.abs(complex_iq), axis=0)
    finite = target_m[np.isfinite(target_m) & (target_m > 0)]
    if len(finite):
        centre = int(np.argmin(np.abs(distances - float(np.median(finite)))))
    else:
        centre = int(np.argmax(np.where(distances >= 0.2, amplitude, -np.inf)))
    gate = np.arange(max(0, centre - 1), min(len(distances), centre + 2))
    phase = np.unwrap(np.angle(complex_iq[:, gate]), axis=0)
    weight = amplitude[gate] / (float(amplitude[gate].sum()) + 1e-12)
    return A121_CHEST_SIGN * (phase - phase[:1]) @ weight * PHASE_TO_MM


def breathing_features(chest_mm: np.ndarray, fs: float) -> dict[str, float]:
    nan = {key: math.nan for key in ("breath_amp_mm", "breath_regularity", "breath_interval_cv", "breath_depth_cv", "motion_energy")}
    if len(chest_mm) < int(30 * fs) or not np.all(np.isfinite(chest_mm)):
        return nan
    sos = butter(2, BREATH_BAND_HZ, "bandpass", fs=fs, output="sos")
    breath = sosfiltfilt(sos, chest_mm - np.mean(chest_mm))
    rhythm = window_rhythm(chest_mm, fs)
    peaks, props = find_peaks(breath, distance=int(1.5 * fs), prominence=0.3 * float(np.std(breath)) + 1e-9)
    intervals = np.diff(peaks) / fs
    depths = props["prominences"] if len(peaks) else np.zeros(0)
    cv = lambda x: float(np.std(x) / np.mean(x)) if len(x) >= 3 and np.mean(x) > 0 else math.nan
    return {
        "breath_amp_mm": float(np.subtract(*np.percentile(breath, [95, 5]))),
        "breath_regularity": float(np.atleast_1d(rhythm.share)[0]),
        "breath_interval_cv": cv(intervals),
        "breath_depth_cv": cv(depths),
        "motion_energy": float(np.log10(np.std(np.diff(chest_mm)) + 1e-6)),
    }


def heart_variability(chest_mm: np.ndarray, fs: float, hr_bpm: float) -> dict[str, float]:
    """Beat-to-beat variability from the radar's cardiac band: RMSSD and SDNN of the intervals between beats.

    The chest displacement is band-passed around the heart rate (0.8-2.5 Hz), beats are its peaks at least 60% of
    a beat apart, refined to sub-frame time with a parabola through the three samples around each peak (frames are
    50 ms apart; the variability sought is tens of ms).  Intervals more than 30% off the window's median, or off
    the pipeline's heart rate when it has one, are dropped; fewer than 20 left gives NaN.
    """

    nan = {"radar_rmssd_ms": math.nan, "radar_sdnn_ms": math.nan, "radar_beats": 0.0}
    if len(chest_mm) < int(30 * fs) or not np.all(np.isfinite(chest_mm)):
        return nan
    sos = butter(2, (0.8, 2.5), "bandpass", fs=fs, output="sos")
    cardiac = sosfiltfilt(sos, chest_mm - np.mean(chest_mm))
    beat_s = 60.0 / hr_bpm if np.isfinite(hr_bpm) and hr_bpm > 0 else 1.0
    peaks, _ = find_peaks(cardiac, distance=max(1, int(0.6 * beat_s * fs)))
    peaks = peaks[(peaks > 0) & (peaks < len(cardiac) - 1)]
    if len(peaks) < 5:
        return nan
    a, b, c = cardiac[peaks - 1], cardiac[peaks], cardiac[peaks + 1]
    denom = a - 2 * b + c
    shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / denom, 0.0).clip(-0.5, 0.5)
    times = (peaks + shift) / fs
    ibi = np.diff(times) * 1000.0
    centre = np.median(ibi) if not np.isfinite(hr_bpm) else 60000.0 / hr_bpm
    good = np.abs(ibi - centre) <= 0.3 * centre
    if good.sum() < 20:
        return {**nan, "radar_beats": float(good.sum())}
    successive = np.diff(ibi)[good[1:] & good[:-1]]
    return {"radar_rmssd_ms": float(np.sqrt(np.mean(successive**2))) if len(successive) >= 10 else math.nan,
            "radar_sdnn_ms": float(np.std(ibi[good])), "radar_beats": float(good.sum())}


def garmin_stage_at(times: pd.Series, garmin) -> pd.Series:
    levels = garmin.sleep_levels
    if levels.empty:
        return pd.Series([None] * len(times), index=times.index, dtype=object)
    levels = levels.sort_values("local_time")
    start, stop = _garmin_sleep_bounds(garmin)
    level_times = pd.to_datetime(levels["local_time"]).to_numpy()
    codes = levels["sleep_level_code"].to_numpy()
    index = np.searchsorted(level_times, pd.to_datetime(times).to_numpy(), side="right") - 1
    stage = [GARMIN_STAGE.get(int(codes[i])) if i >= 0 else None for i in index]
    inside = np.ones(len(times), dtype=bool)
    if start is not None:
        inside &= pd.to_datetime(times).to_numpy() >= np.datetime64(start)
    if stop is not None:
        inside &= pd.to_datetime(times).to_numpy() <= np.datetime64(stop)
    return pd.Series([s if ok else None for s, ok in zip(stage, inside)], index=times.index, dtype=object)


def garmin_stress_hrv(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(stress every minute, RMSSD every 5 min) from a watch export, as local time and value."""

    stress, hrv = [], []
    for message in parse_fit_files(fit_paths_from_sources(folder)):
        fields = message.fields
        if message.global_num == 227 and fields.get(1) is not None and fields.get(0) is not None and fields[0] >= 0:
            stress.append((fit_datetime_local_naive(fields[1]), float(fields[0])))
        elif message.global_num == 371 and fields.get(253) is not None and fields.get(0) is not None:
            hrv.append((fit_datetime_local_naive(fields[253]), float(fields[0]) / 128.0))
    to_frame = lambda rows, name: pd.DataFrame(rows, columns=["local_time", name]).drop_duplicates("local_time").sort_values("local_time")
    return to_frame(stress, "garmin_stress"), to_frame(hrv, "garmin_hrv_ms")


def _nearest(times: pd.Series, frame: pd.DataFrame, column: str, tolerance: str) -> np.ndarray:
    if frame.empty:
        return np.full(len(times), np.nan)
    left = pd.DataFrame({"local_time": pd.to_datetime(times).astype("datetime64[us]"), "order": np.arange(len(times))}).sort_values("local_time")
    right = frame.assign(local_time=pd.to_datetime(frame["local_time"]).astype("datetime64[us]"))
    merged = pd.merge_asof(left, right, on="local_time", tolerance=pd.Timedelta(tolerance), direction="nearest")
    return merged.sort_values("order")[column].to_numpy(dtype=float)


def _robust_scale(values: pd.Series) -> pd.Series:
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    spread = (q3 - q1) if np.isfinite(q3 - q1) and q3 - q1 > 0 else 1.0
    return (values - median) / spread


def night_epochs(night: Night, night_dir: Path = NIGHT_DIR, garmin_dir: Path = GARMIN_DIR) -> pd.DataFrame:
    trend = pd.read_csv(night_dir / f"{night.stem}_gated_sleep_vitals.csv")
    with np.load(night_dir / f"{night.stem}.npz") as raw:
        timestamps, distances, iq, target = raw["timestamp_ms"], raw["distances_m"], raw["iq"], raw["acconeer_target_m"]
    t_s = (timestamps - timestamps[0]) / 1000.0
    fs = float(1.0 / np.median(np.diff(t_s)))
    rows = []
    for start_s, end_s, hr in zip(trend["window_start_s"], trend["elapsed_s"], pd.to_numeric(trend["hr_bpm"], errors="coerce")):
        lo, hi = np.searchsorted(t_s, [start_s, end_s])
        if hi - lo < int(30 * fs):
            rows.append({**breathing_features(np.zeros(0), fs), **heart_variability(np.zeros(0), fs, hr)})
            continue
        chest = chest_displacement(iq[lo:hi], distances, target[lo:hi])
        rows.append({**breathing_features(chest, fs), **heart_variability(chest, fs, hr)})
    epochs = pd.concat([trend.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    epochs["night"], epochs["setup"] = night.name, night.setup
    epochs["clock_time_dt"] = pd.to_datetime(epochs["clock_time"])
    epochs["centre_time"] = epochs["clock_time_dt"] - pd.Timedelta(seconds=15)
    epochs["hours_since_start"] = epochs["elapsed_s"] / 3600.0
    moving = (epochs["movement_fraction"] > 0.2) | (epochs["no_presence_fraction"] > 0.2)
    last = epochs["elapsed_s"].where(moving).ffill()
    epochs["minutes_since_movement"] = ((epochs["elapsed_s"] - last) / 60.0).fillna(60.0).clip(upper=60.0)
    epochs["heuristic_stage"] = classify_radar_sleep(trend).frame["radar_sleep_phase"].to_numpy()
    epochs["garmin_stage"] = None
    for column in WATCH_FEATURES:
        epochs[column] = np.nan
    if night.garmin is not None:
        garmin = load_garmin_reference(garmin_dir / night.garmin)
        epochs["garmin_stage"] = garmin_stage_at(epochs["centre_time"], garmin)
        merged = _merge_garmin_nearest(epochs[["clock_time_dt"]].copy(), garmin)
        epochs["garmin_hr_bpm"] = merged["garmin_hr_bpm"].to_numpy()
        epochs["garmin_rr_bpm"] = merged["garmin_rr_bpm"].to_numpy()
        stress, hrv = garmin_stress_hrv(garmin_dir / night.garmin)
        epochs["garmin_stress"] = _nearest(epochs["centre_time"], stress, "garmin_stress", "2min")
        epochs["garmin_hrv_ms"] = _nearest(epochs["centre_time"], hrv, "garmin_hrv_ms", "3min")
    for feature in (*BASE_FEATURES, *WATCH_FEATURES):
        values = pd.to_numeric(epochs[feature], errors="coerce")
        epochs[f"{feature}_med10"] = values.rolling(CONTEXT_EPOCHS, center=True, min_periods=3).median()
        epochs[f"{feature}_sd10"] = values.rolling(CONTEXT_EPOCHS, center=True, min_periods=3).std()
    for column in [c for c in epochs.columns if c.startswith((*BASE_FEATURES, *WATCH_FEATURES)) or c == "minutes_since_movement"]:
        epochs[f"z_{column}"] = _robust_scale(pd.to_numeric(epochs[column], errors="coerce"))
    return epochs


def feature_columns(feature_set: str) -> list[str]:
    """Scaled columns of one set ("breathing", "heart", "movement", "all"), with their 10-minute context."""

    sets = {**FEATURE_SETS, **WATCH_SETS}
    names = list(dict.fromkeys(n for s in FEATURE_SETS.values() for n in s)) if feature_set == "all" else list(sets[feature_set])
    columns = []
    for name in names:
        columns.append(f"z_{name}")
        if name in BASE_FEATURES or name in WATCH_FEATURES:
            columns += [f"z_{name}_med10", f"z_{name}_sd10"]
    return columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    frames = []
    for night in NIGHTS:
        if not (NIGHT_DIR / f"{night.stem}.npz").exists():
            print(f"skip {night.name}: not extracted")
            continue
        epochs = night_epochs(night)
        frames.append(epochs)
        labelled = epochs["garmin_stage"].notna().sum()
        print(f"{night.name} ({night.setup}): {len(epochs)} epochs, {labelled} with a Garmin stage, "
              f"{epochs['garmin_hr_bpm'].notna().sum()} with Garmin HR")
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["clock_time_dt"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} epochs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
