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


def _first_principal_component(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[0]


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
    motion = _first_principal_component(filtered)
    raw = _first_principal_component(accel)
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
