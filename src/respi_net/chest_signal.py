"""Chest-motion signals for breathing phases, filtered as little as possible.

Phase labels and the phase network need the *shape* of each breath: where the
chest turns, where it stops and how long it stays still.  The respiration-rate
pipeline band-passes at 0.1-0.5 Hz, which is right for finding a rate and
wrong for this.  The 0.1 Hz lower edge is a high-pass filter: it keeps only
changes faster than about ten seconds per cycle and takes everything slower
away.  A held breath is the chest standing still for 15 s -- to that filter,
a slow change -- so the output is pulled back toward zero within a couple of
seconds (2nd-order high-pass, time constant sqrt(2)/(2*pi*0.1) ~ 2.2 s), and a
zero-phase filter does it from both ends.  The plateau turns into a slow hump
that looks like breathing.  The 0.5 Hz upper edge also rounds off the sharp
corners where a breath stops or starts.

So here the only filter is a low-pass (default 2 Hz) against noise and the
heartbeat.  A short run loses its slow drift with one straight line, which
cannot bend a plateau, and velocity -- which drift barely touches -- carries
the direction of motion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfilt, sosfilt_zi, sosfiltfilt

from .a121 import parse_json_array

A121_CENTRE_FREQUENCY_HZ = 60.5e9
SPEED_OF_LIGHT_M_S = 299_792_458.0
A121_WAVELENGTH_MM = SPEED_OF_LIGHT_M_S / A121_CENTRE_FREQUENCY_HZ * 1000.0
PHASE_TO_MM = A121_WAVELENGTH_MM / (4.0 * np.pi)
# The recorded unwrapped phase falls while the chest moves toward the radar,
# i.e. during an inhale.  Flipping it makes every chest signal here rise on
# inhale, the convention of the labels, the synthetic data and the network.
A121_CHEST_SIGN = -1.0

DEFAULT_LOW_PASS_HZ = 2.0
RATE_BAND_HZ = (0.10, 0.50)


@dataclass(frozen=True)
class ChestSignal:
    source: str
    time_s: np.ndarray
    fs: float
    chest: np.ndarray
    units: str
    raw: np.ndarray
    echo_db: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def velocity(self) -> np.ndarray:
        """Rate of change of :attr:`chest` per second."""

        return np.gradient(self.chest) * self.fs


def sample_rate(time_s: np.ndarray) -> float:
    dt = np.diff(np.asarray(time_s, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        raise ValueError("The signal has no usable time axis.")
    return float(1.0 / np.median(dt))


def light_filter(
    x: np.ndarray,
    fs: float,
    low_pass_hz: float = DEFAULT_LOW_PASS_HZ,
    *,
    causal: bool = False,
    axis: int = -1,
) -> np.ndarray:
    """Low-pass only: holds stay flat and the corners of each breath stay sharp.

    ``causal`` uses a one-sided filter, as a real-time detector would, started
    from the first sample's level so it does not ring at the beginning.
    """

    x = np.asarray(x, dtype=float)
    cutoff = min(float(low_pass_hz), 0.45 * fs)
    sos = butter(2, cutoff, btype="lowpass", fs=fs, output="sos")
    if not causal:
        return sosfiltfilt(sos, x, axis=axis)
    moved = np.moveaxis(x, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    zi = sosfilt_zi(sos)
    out = np.vstack([sosfilt(sos, row, zi=zi * row[0])[0] for row in flat])
    return np.moveaxis(out.reshape(moved.shape), -1, axis)


def rate_band_filter(x: np.ndarray, fs: float, band: tuple[float, float] = RATE_BAND_HZ) -> np.ndarray:
    """The respiration-rate band-pass, kept for comparisons -- not for phases."""

    sos = butter(2, list(band), btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=float))


def regular_frame_times_ms(timestamps_ms: np.ndarray, tolerance_ms: float = 1.5) -> np.ndarray:
    """Frame times with the driver's rounding drift removed.

    The A121 driver stamps frames in whole milliseconds and its tick clock runs about 0.17 percent fast: frames
    that are 50.00 ms apart get steps of 50 ms with an extra millisecond every ~12 frames (50.085 ms on
    average).  Over a 12-minute recording the stamps then end 1.3 s after the host clock and the phone, and
    the radar slides against any other sensor.  Steps within ``tolerance_ms`` of the median step are set to
    it; larger gaps (dropped frames) stay as they are.
    """

    times = np.asarray(timestamps_ms, dtype=float)
    steps = np.diff(times)
    usable = steps[np.isfinite(steps) & (steps > 0)]
    if len(usable) < 10:
        return times
    median = float(np.median(usable))
    regular = np.where(np.abs(steps - median) <= tolerance_ms, median, steps)
    return times[0] + np.concatenate([[0.0], np.cumsum(regular)])


def _load_matrix(series: pd.Series, width: int | None = None) -> np.ndarray:
    rows = [parse_json_array(value) for value in series]
    width = width or min(len(row) for row in rows)
    return np.vstack([row[:width] for row in rows])


def _a121_gate(df: pd.DataFrame, distances: np.ndarray, median_amplitude: np.ndarray) -> tuple[np.ndarray, str]:
    """The chest range bins: Acconeer's own selection when it was recorded."""

    width = len(distances)
    starts = pd.to_numeric(df.get("AcconeerRangeStartIndex"), errors="coerce")
    ends = pd.to_numeric(df.get("AcconeerRangeEndIndex"), errors="coerce")
    if starts is not None and ends is not None:
        pairs = [
            (int(start), int(end))
            for start, end in zip(starts, ends, strict=True)
            if np.isfinite(start) and np.isfinite(end) and 0 <= int(start) < int(end) <= width
        ]
        if pairs:
            (start, end), _ = Counter(pairs).most_common(1)[0]
            return np.arange(start, end, dtype=int), "acconeer-range"
    target = pd.to_numeric(df.get("AcconeerTargetDistance_m"), errors="coerce")
    if target is not None:
        target = target[np.isfinite(target) & (target > 0)]
        if len(target):
            centre = int(np.argmin(np.abs(distances - float(np.median(target)))))
            return np.arange(max(0, centre - 1), min(width, centre + 2), dtype=int), "acconeer-target"
    centre = int(np.argmax(median_amplitude))
    return np.arange(max(0, centre - 1), min(width, centre + 2), dtype=int), "strongest-bin"


def a121_chest_signal(
    source: Path | str | pd.DataFrame,
    *,
    origin_ms: float | None = None,
    low_pass_hz: float = DEFAULT_LOW_PASS_HZ,
    detrend_linear: bool = True,
    causal: bool = False,
    regular_time: bool = True,
) -> ChestSignal:
    """Chest displacement in mm from an A121 Sparse IQ recording, rising on inhale.

    The chest bins are combined by their median echo strength after each
    bin's phase is unwrapped.  ``detrend_linear`` removes one straight line
    over the whole recording -- right for runs of a few minutes, not for a
    night, where drift is not a line.
    """

    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source
    missing = {"Timestamp_ms", "Distances_m", "Real", "Imag"}.difference(df.columns)
    if missing:
        raise ValueError(f"A121 recording is missing {', '.join(sorted(missing))}")
    timestamps_ms = pd.to_numeric(df["Timestamp_ms"], errors="coerce").to_numpy(dtype=float)
    if regular_time and np.all(np.isfinite(timestamps_ms)):
        timestamps_ms = regular_frame_times_ms(timestamps_ms)
    origin = float(timestamps_ms[0]) if origin_ms is None else float(origin_ms)
    time_s = (timestamps_ms - origin) / 1000.0
    fs = sample_rate(time_s)

    distances = parse_json_array(df["Distances_m"].iloc[0])
    real = _load_matrix(df["Real"])
    imag = _load_matrix(df["Imag"])
    width = min(len(distances), real.shape[1], imag.shape[1])
    if width == 0:
        raise ValueError("A121 recording has no complex range bins")
    distances = np.asarray(distances[:width], dtype=float)
    profile = real[:, :width] + 1j * imag[:, :width]
    amplitude = np.abs(profile)
    median_amplitude = np.median(amplitude, axis=0)
    gate, gate_source = _a121_gate(df, distances, median_amplitude)
    weights = np.maximum(median_amplitude[gate], 0.0)
    weights = weights / (float(np.sum(weights)) + 1e-12)

    phase = np.unwrap(np.angle(profile[:, gate]), axis=0)
    phase = detrend(phase, axis=0, type="linear") if detrend_linear else phase - phase[:1]
    raw = A121_CHEST_SIGN * (phase @ weights) * PHASE_TO_MM
    chest = light_filter(raw, fs, low_pass_hz, causal=causal)
    echo = amplitude[:, gate] @ weights
    echo_db = 20.0 * np.log10(np.maximum(echo, 1e-9))
    return ChestSignal(
        source="a121",
        time_s=time_s,
        fs=fs,
        chest=chest,
        units="mm",
        raw=raw,
        echo_db=echo_db,
        meta={
            "gate_source": gate_source,
            "gate_start_m": float(distances[int(gate[0])]),
            "gate_end_m": float(distances[int(gate[-1])]),
            "gate_bins": int(len(gate)),
            "frames": int(len(df)),
            "low_pass_hz": float(low_pass_hz),
            "causal": bool(causal),
        },
    )


def _first_principal_component(values: np.ndarray, direction_from: np.ndarray | None = None) -> np.ndarray:
    """Project ``values`` on their main axis, or on the main axis of ``direction_from``."""

    basis = values if direction_from is None else direction_from
    _, _, vt = np.linalg.svd(basis - basis.mean(axis=0, keepdims=True), full_matrices=False)
    return (values - values.mean(axis=0, keepdims=True)) @ vt[0]


def imu_chest_signal(
    source: Path | str | pd.DataFrame,
    *,
    origin_ms: float | None = None,
    low_pass_hz: float = DEFAULT_LOW_PASS_HZ,
    detrend_linear: bool = True,
    name: str = "imu",
) -> ChestSignal:
    """Breathing motion from a chest/abdomen IMU, in milli-g along its main axis.

    A phone or sensor lying on the body tilts as the torso rises and falls, so
    the direction of gravity in its accelerometer follows the breath -- a
    volume-like signal that, unlike a gyroscope's rate, keeps a hold flat.
    The main axis comes from PCA, whose sign is arbitrary: callers orient the
    result against the cues or the radar.
    """

    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source
    missing = {"Time_ms", "ax", "ay", "az"}.difference(df.columns)
    if missing:
        raise ValueError(f"IMU recording is missing {', '.join(sorted(missing))}")
    time_ms = pd.to_numeric(df["Time_ms"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(time_ms)
    df = df.loc[keep]
    time_ms = time_ms[keep]
    order = np.argsort(time_ms, kind="stable")
    time_ms = time_ms[order]
    origin = float(time_ms[0]) if origin_ms is None else float(origin_ms)
    time_s = (time_ms - origin) / 1000.0
    fs = sample_rate(time_s)
    accel = df[["ax", "ay", "az"]].to_numpy(dtype=float)[order] * 1000.0
    filtered = light_filter(accel, fs, low_pass_hz, axis=0)
    # The breathing *direction* is found in the breathing band, where slow
    # drift and settling cannot outweigh it; the signal projected on that
    # direction is only low-passed, so holds keep their shape.
    band = butter(2, list(RATE_BAND_HZ), btype="bandpass", fs=fs, output="sos")
    breathing_band = sosfiltfilt(band, accel, axis=0)
    motion = _first_principal_component(filtered, direction_from=breathing_band)
    raw = _first_principal_component(accel, direction_from=breathing_band)
    if detrend_linear:
        motion = detrend(motion, type="linear")
        raw = detrend(raw, type="linear")
    return ChestSignal(
        source=name,
        time_s=time_s,
        fs=fs,
        chest=motion,
        units="mg",
        raw=raw,
        echo_db=np.full(len(time_s), np.nan),
        meta={"samples": int(len(time_s)), "low_pass_hz": float(low_pass_hz)},
    )


# The strain belt's HX711 count rises as the chest expands.
STRAIN_BELT_CHEST_SIGN = 1.0
STRAIN_BELT_ERROR_VALUE = -1.0


def _regular_belt_times(host_s: np.ndarray, smooth_samples: int = 21) -> tuple[np.ndarray, float]:
    """Even sample times for belt packets stamped on arrival.

    The belt sends no clock of its own, and BLE delivers its 10 Hz packets in
    bursts (arrival gaps of 2-210 ms).  The firmware's rate is steady, so the
    time of sample ``k`` is ``k / fs`` plus an offset that only moves when a
    packet is lost; a running median of that offset drops the burst jitter and
    still follows a gap.
    """

    host_s = np.asarray(host_s, dtype=float)
    k = np.arange(len(host_s), dtype=float)
    fs = float((len(host_s) - 1) / (host_s[-1] - host_s[0]))
    offset = pd.Series(host_s - k / fs).rolling(smooth_samples, center=True, min_periods=1).median().to_numpy()
    return k / fs + offset, fs


def despike(x: np.ndarray, window: int = 7, n_sigmas: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Hampel filter: replace samples far from their neighbours' median by interpolation."""

    x = np.asarray(x, dtype=float)
    series = pd.Series(x)
    median = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - median).abs().rolling(window, center=True, min_periods=1).median()
    # A floor from the whole trace, so a flat hold (MAD ~ 0) does not flag its own noise.
    floor = float(np.median(np.abs(np.diff(x)))) or 1.0
    spikes = ((series - median).abs() > n_sigmas * 1.4826 * np.maximum(mad, floor)).to_numpy()
    if spikes.any():
        good = ~spikes
        x = x.copy()
        x[spikes] = np.interp(np.flatnonzero(spikes), np.flatnonzero(good), x[good])
    return x, spikes


def _local_robust_level(times: np.ndarray, levels: np.ndarray, at: float, half_window: float) -> float:
    from scipy.stats import theilslopes

    distance = np.abs(times - at)
    near = distance <= half_window
    if near.sum() < 3:
        # Too few troughs in the window (an edge, or slow breathing): use the three nearest.
        near = np.zeros(len(times), dtype=bool)
        near[np.argsort(distance)[:3]] = True
    if near.sum() < 3:
        return float(np.median(levels[near]))
    slope, intercept, _, _ = theilslopes(levels[near], times[near])
    return float(intercept + slope * at)


def end_expiratory_baseline(
    chest: np.ndarray,
    time_s: np.ndarray,
    fs: float,
    *,
    min_depth_fraction: float = 0.3,
    window_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow drift of the belt, followed through the bottom of each breath.

    The end of a relaxed exhale is the most repeatable level of the chest, so
    the troughs (and holds after exhale, which are long flat troughs) trace
    the belt's creep.  The baseline at each trough is a robust (Theil-Sen)
    line through the troughs within ``window_s`` -- several breaths -- so it
    follows the drift without flattening one deliberately deep or shallow
    exhale, and, unlike a running median, is not pulled inward at the ends of
    the recording where the window is one-sided.  Returns the baseline
    and the trough sample indices.
    """

    from scipy.signal import find_peaks

    chest = np.asarray(chest, dtype=float)
    straight = detrend(chest, type="linear")
    depth = float(np.subtract(*np.percentile(straight, [95, 5])))
    troughs, _ = find_peaks(-chest, prominence=max(depth * min_depth_fraction, 1e-9), distance=max(1, int(1.0 * fs)))
    if len(troughs) < 2:
        trend = chest - straight
        return trend, troughs
    levels = chest[troughs]
    times = time_s[troughs]
    smoothed = np.array([_local_robust_level(times, levels, t, window_s / 2) for t in times])
    baseline = np.interp(time_s, times, smoothed)
    # Beyond the first and last trough, carry the drift on at its fitted slope.
    slope = np.polyfit(times, smoothed, 1)[0] if len(times) >= 3 else 0.0
    before, after = time_s < times[0], time_s > times[-1]
    baseline[before] = smoothed[0] + slope * (time_s[before] - times[0])
    baseline[after] = smoothed[-1] + slope * (time_s[after] - times[-1])
    return light_filter(baseline, fs, 0.1), troughs


def strain_belt_chest_signal(
    source: Path | str | pd.DataFrame,
    *,
    low_pass_hz: float = DEFAULT_LOW_PASS_HZ,
    drift: str = "troughs",
    origin_ms: float | None = None,
    name: str = "strain_belt",
) -> ChestSignal:
    """Chest expansion from the ESP32/HX711 strain belt, rising on inhale.

    Drops the firmware's failed-read value (-1) and isolated spikes, puts the
    samples on an even 10 Hz grid, low-passes, and removes drift:
    ``troughs`` (default) follows the end-expiratory level, ``linear`` takes
    one straight line off, ``none`` keeps it.  Time runs from ``origin_ms``
    (host epoch) when given, else from the coach's ``elapsed_s``, else from
    the first sample.  ``chest`` is in HX711 counts
    above the end-expiratory level; ``meta['breath_depth']`` is a typical
    breath in the same counts, for scaling.
    """

    df = pd.read_csv(source) if not isinstance(source, pd.DataFrame) else source
    missing = {"host_time_ms", "raw"}.difference(df.columns)
    if missing:
        raise ValueError(f"Strain-belt recording is missing {', '.join(sorted(missing))}")
    host_s = pd.to_numeric(df["host_time_ms"], errors="coerce").to_numpy(dtype=float) / 1000.0
    raw = pd.to_numeric(df["raw"], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(host_s) & np.isfinite(raw) & (raw != STRAIN_BELT_ERROR_VALUE)
    errors = int((raw == STRAIN_BELT_ERROR_VALUE).sum())
    host_s, raw = host_s[keep], raw[keep]
    if len(raw) < 20:
        raise ValueError("Strain-belt recording is too short.")
    if origin_ms is not None:
        origin = float(origin_ms) / 1000.0
    elif "elapsed_s" in df.columns:
        # Coached runs: keep the coach's time axis so cues line up.
        elapsed = pd.to_numeric(df["elapsed_s"], errors="coerce").to_numpy(dtype=float)[keep]
        origin = float(host_s[0] - elapsed[0])
    else:
        origin = float(host_s[0])
    stamp_s, fs_nominal = _regular_belt_times(host_s)
    clean, spikes = despike(raw)

    # Resample onto an exact grid so ``fs`` is the true spacing for filters.
    time_s = np.arange(np.ceil((stamp_s[0] - origin) * fs_nominal), np.floor((stamp_s[-1] - origin) * fs_nominal) + 1) / fs_nominal
    values = STRAIN_BELT_CHEST_SIGN * np.interp(time_s, stamp_s - origin, clean)
    fs = fs_nominal
    chest = light_filter(values, fs, low_pass_hz)
    troughs = np.array([], dtype=int)
    if drift == "troughs":
        baseline, troughs = end_expiratory_baseline(chest, time_s, fs)
    elif drift == "linear":
        baseline = chest - detrend(chest, type="linear")
    elif drift == "none":
        baseline = np.full_like(chest, float(np.median(chest)))
    else:
        raise ValueError(f"Unknown drift mode {drift!r}; use troughs, linear or none.")
    chest = chest - baseline
    depth = float(np.subtract(*np.percentile(chest, [95, 5])))
    # Sensor noise: what a 2.5 Hz low-pass takes off the raw counts.  Even fast breathing
    # (60/min = 1 Hz) and its first harmonic stay below that.
    residual = values - light_filter(values, fs, 2.5)
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual)))) or 1.0
    return ChestSignal(
        source=name,
        time_s=time_s,
        fs=fs,
        chest=chest,
        units="counts",
        raw=values - baseline,
        echo_db=np.full(len(time_s), np.nan),
        meta={
            "samples": int(len(time_s)),
            "low_pass_hz": float(low_pass_hz),
            "drift": drift,
            "hx711_errors": errors,
            "spikes": int(spikes.sum()),
            "troughs": int(len(troughs)),
            "breath_depth": depth,
            "noise": noise,
            "depth_to_noise": depth / noise,
            "drift_counts": float(baseline[-1] - baseline[0]),
        },
    )
