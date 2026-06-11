from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt, welch, lfilter

from .a121 import parse_json_array

RESP_BAND_HZ = (0.10, 0.50)
# Narrowed for resting vitals: 6-30 breaths/minute (0.10-0.50 Hz).
A121_RESP_BAND_HZ = (0.10, 0.50)
# Narrowed for resting vitals: 42-120 beats/minute (0.70-2.00 Hz).
HEART_BAND_HZ = (0.70, 2.00)
DEFAULT_GATE_HALF_WIDTH_M = 0.05
A121_RATE_WINDOW_S = 20.0
A121_MIN_TARGET_DISTANCE_M = 0.28
RESP_RATE_CONFIDENCE_MIN = 4.0
HEART_RATE_CONFIDENCE_MIN = 5.0
HEART_RATE_STRONG_LOCK_CONFIDENCE = 30.0
HEART_RATE_LOCK_MIN_WIDTH_HZ = 0.32
HEART_RATE_LOCK_HALF_WIDTH_HZ = HEART_RATE_LOCK_MIN_WIDTH_HZ * 0.5
# Real A121 recordings often show the clearest cardiac micro-motion on the near edge of the
# torso return, not necessarily at the maximum-amplitude distance bin used for respiration.
HEART_RANGE_HALF_WIDTH_M = 0.18
HEART_RANGE_MAX_BINS = 13
HEART_BIN_CLUSTER_TOLERANCE_HZ = 0.10
HEART_BIN_MIN_CONFIDENCE = 4.0
HEART_RANGE_OVERRIDE_MIN_CONFIDENCE = 12.0


@dataclass(frozen=True)
class A121VitalAnalysis:
    sample_rate_hz: float
    target_distance_m: float
    gate_min_m: float
    gate_max_m: float
    peak_distance_m: float
    peak_amplitude: float
    mean_amplitude: float
    presence_score: float
    present: bool
    resp_bpm: float
    heart_bpm: float
    resp_hz: float
    heart_hz: float
    times_s: np.ndarray
    raw_phase: np.ndarray
    resp_signal: np.ndarray
    heart_signal: np.ndarray
    raw_i: np.ndarray
    raw_q: np.ndarray
    distances_m: np.ndarray
    latest_amplitude: np.ndarray
    selected_index: int
    resp_confidence: float = 0.0
    heart_confidence: float = 0.0
    signal_quality: float = 0.0
    candidate_bins: int = 0


@dataclass(frozen=True)
class _RecordedAcconeerGate:
    target_distance_m: float | None = None
    gate_min_m: float | None = None
    gate_max_m: float | None = None
    range_start_idx: int | None = None
    range_end_idx: int | None = None
    app_state: str = ""
    presence_detected: bool | None = None
    breathing_rate_bpm: float | None = None


@dataclass(frozen=True)
class A121LiveTraceResult:
    """Append-only live traces for plotting.

    Unlike :func:`analyze_a121_vitals`, this result is produced by a stateful processor that
    keeps IIR filter memory between GUI ticks. Existing samples are never re-filtered, so the
    graph scrolls left without the historical waveform changing shape.
    """

    sample_rate_hz: float
    target_distance_m: float
    selected_index: int
    times_s: np.ndarray
    raw_phase: np.ndarray
    raw_i: np.ndarray
    raw_q: np.ndarray
    resp_signal: np.ndarray
    heart_signal: np.ndarray


class A121LiveTraceProcessor:
    """Stateful Acconeer-style A121 phase processor for live plotting only.

    The batch analyzer intentionally reprocesses a window to estimate rates. That is fine for
    CSV/offline analysis, but terrible for a live scrolling plot: every new frame changes filter
    initial conditions, detrending, selected-bin weights, and therefore also the already-drawn
    past. This processor mirrors the causal part of Acconeer's breathing processor and appends one
    new filtered sample per radar frame while preserving all previous samples verbatim.
    """

    def __init__(self, max_history_s: float = 180.0) -> None:
        self.max_history_s = float(max_history_s)
        self.reset()

    def reset(self) -> None:
        self.fs: float = 0.0
        self.m: int = 0
        self.distances = np.asarray([], dtype=float)
        self.selected_index: int | None = None
        self.target_distance_m: float = 0.0
        self.last_frame: int | None = None
        self.last_timestamp_ms: float | None = None
        self.first_timestamp_ms: float | None = None
        self.last_time_s: float | None = None
        self.processed_count: int = 0

        self.b_static: np.ndarray | None = None
        self.a_static: np.ndarray | None = None
        self.static_x: np.ndarray | None = None
        self.static_y: np.ndarray | None = None
        # DC-blocking high-pass pre-filter: strips the massive cumulative drift from
        # angle_unwrapped before feeding the bandpass, so the IIR state stays near zero.
        self.b_dc: np.ndarray | None = None
        self.a_dc: np.ndarray | None = None
        self.dc_x: np.ndarray | None = None
        self.dc_y: np.ndarray | None = None
        self.b_resp: np.ndarray | None = None
        self.a_resp: np.ndarray | None = None
        self.resp_x: np.ndarray | None = None
        self.resp_y: np.ndarray | None = None
        self.b_heart: np.ndarray | None = None
        self.a_heart: np.ndarray | None = None
        self.heart_x: np.ndarray | None = None
        self.heart_y: np.ndarray | None = None
        self.prev_angle: np.ndarray | None = None
        self.angle_unwrapped: np.ndarray | None = None
        self.lp_filt_ampl: np.ndarray | None = None
        self.sf: float = 0.0

        maxlen = max(32, int(round(self.max_history_s * 100.0)))
        self.times: deque[float] = deque(maxlen=maxlen)
        self.raw_phase: deque[float] = deque(maxlen=maxlen)
        self.raw_i: deque[float] = deque(maxlen=maxlen)
        self.raw_q: deque[float] = deque(maxlen=maxlen)
        self.resp_signal: deque[float] = deque(maxlen=maxlen)
        self.heart_signal: deque[float] = deque(maxlen=maxlen)

    def _resize_history(self) -> None:
        maxlen = max(32, int(round(self.max_history_s * max(self.fs, 1.0) * 1.2)))
        for name in ("times", "raw_phase", "raw_i", "raw_q", "resp_signal", "heart_signal"):
            old = getattr(self, name)
            setattr(self, name, deque(old, maxlen=maxlen))

    def _initialize_filters(self, fs: float, distances: np.ndarray) -> None:
        self.fs = float(fs)
        self.distances = np.asarray(distances, dtype=float)
        self.m = int(len(self.distances))
        self._resize_history()

        static_band = _valid_band_for_fs((A121_RESP_BAND_HZ[0], A121_RESP_BAND_HZ[0] * 1.05), self.fs, min_width=1e-5)
        static_cutoff = A121_RESP_BAND_HZ[0] if static_band is None else static_band[0]
        self.b_static, self.a_static = butter(2, static_cutoff, btype="lowpass", fs=self.fs)
        self.static_x = np.zeros((len(self.b_static), self.m), dtype=np.complex128)
        self.static_y = np.zeros((len(self.a_static) - 1, self.m), dtype=np.complex128)

        # DC-blocking high-pass: 1st-order Butterworth at 0.04 Hz removes the slow cumulative
        # drift from angle_unwrapped.  The cutoff is well below the respiration band (0.10 Hz)
        # so vital-sign energy is fully preserved, but the massive monotonic ramp that corrupts
        # the downstream bandpass IIR state is eliminated.
        dc_cutoff = min(0.04, self.fs * 0.45)
        if dc_cutoff > 0 and dc_cutoff < self.fs * 0.5:
            self.b_dc, self.a_dc = butter(1, dc_cutoff, btype="highpass", fs=self.fs)
            self.dc_x = np.zeros((len(self.b_dc), self.m), dtype=float)
            self.dc_y = np.zeros((len(self.a_dc) - 1, self.m), dtype=float)
        else:
            self.b_dc = self.a_dc = self.dc_x = self.dc_y = None

        resp_band = _valid_band_for_fs(A121_RESP_BAND_HZ, self.fs)
        if resp_band is not None:
            self.b_resp, self.a_resp = butter(2, resp_band, btype="bandpass", fs=self.fs)
            self.resp_x = np.zeros((len(self.b_resp), self.m), dtype=float)
            self.resp_y = np.zeros((len(self.a_resp) - 1, self.m), dtype=float)
        else:
            self.b_resp = self.a_resp = self.resp_x = self.resp_y = None

        heart_band = _valid_band_for_fs(HEART_BAND_HZ, self.fs)
        if heart_band is not None:
            self.b_heart, self.a_heart = butter(3, heart_band, btype="bandpass", fs=self.fs)
            self.heart_x = np.zeros((len(self.b_heart), self.m), dtype=float)
            self.heart_y = np.zeros((len(self.a_heart) - 1, self.m), dtype=float)
        else:
            self.b_heart = self.a_heart = self.heart_x = self.heart_y = None

        self.prev_angle = None
        self.angle_unwrapped = np.zeros(self.m, dtype=float)
        self.lp_filt_ampl = None
        self.sf = float(np.exp(-1.0 / max(self.fs * A121_RATE_WINDOW_S, 1e-9)))

    @staticmethod
    def _iir_step(
        x: np.ndarray,
        b: np.ndarray,
        a: np.ndarray,
        x_hist: np.ndarray,
        y_hist: np.ndarray,
    ) -> np.ndarray:
        x_hist[1:] = x_hist[:-1].copy()
        x_hist[0] = x
        y = np.sum(b[:, None] * x_hist, axis=0) - np.sum(a[1:][:, None] * y_hist, axis=0)
        y_hist[1:] = y_hist[:-1].copy()
        y_hist[0] = y
        return y

    def _select_index(self, amplitude: np.ndarray, target_distance_m: float | None) -> int:
        if self.m <= 0:
            return 0
        if target_distance_m is not None and np.isfinite(target_distance_m) and target_distance_m > 0:
            idx = int(np.argmin(np.abs(self.distances - float(target_distance_m))))
            self.selected_index = idx
            self.target_distance_m = float(self.distances[idx])
            return idx
        if self.selected_index is None:
            search = self.distances >= A121_MIN_TARGET_DISTANCE_M if len(self.distances) == len(amplitude) else np.ones(len(amplitude), dtype=bool)
            if not np.any(search):
                search = np.ones(len(amplitude), dtype=bool)
            idx = int(np.argmax(np.where(search, amplitude, -np.inf)))
            self.selected_index = idx
            self.target_distance_m = float(self.distances[idx]) if len(self.distances) > idx else 0.0
            return idx
        return int(np.clip(self.selected_index, 0, self.m - 1))

    def _candidate_indices(
        self,
        selected_idx: int,
        gate_half_width_m: float,
        use_gating: bool,
        *,
        max_bins: int = 3,
    ) -> np.ndarray:
        if self.m <= 0:
            return np.asarray([], dtype=int)
        if use_gating:
            half = max(0.0, float(gate_half_width_m))
            idx = np.flatnonzero(np.abs(self.distances - self.distances[selected_idx]) <= half)
            if len(idx):
                if len(idx) > max_bins:
                    order = np.argsort(np.abs(idx - selected_idx), kind="stable")[:max_bins]
                    idx = np.sort(idx[order])
                return idx.astype(int)
        lo = max(0, selected_idx - 1)
        hi = min(self.m, selected_idx + 2)
        idx = np.arange(lo, hi, dtype=int)
        if len(idx) > max_bins:
            order = np.argsort(np.abs(idx - selected_idx), kind="stable")[:max_bins]
            idx = np.sort(idx[order])
        return idx

    def process_rows(
        self,
        rows: list[list[Any]],
        *,
        target_distance_m: float | None = None,
        use_gating: bool = False,
        gate_half_width_m: float = DEFAULT_GATE_HALF_WIDTH_M,
    ) -> A121LiveTraceResult | None:
        if not rows:
            return self.result()

        times_ms = np.asarray([float(row[0]) for row in rows], dtype=float)
        fs = sample_rate_from_ms(times_ms, default=self.fs or 20.0)
        latest_distances = parse_json_array(rows[-1][6])
        if len(latest_distances) == 0:
            return self.result()
        if self.m != len(latest_distances) or self.fs <= 0 or abs(fs - self.fs) / max(self.fs, 1e-9) > 0.20:
            old_selected = self.selected_index
            self.reset()
            self.selected_index = old_selected if old_selected is not None and old_selected < len(latest_distances) else None
            self._initialize_filters(fs, latest_distances)

        assert self.static_x is not None and self.static_y is not None
        assert self.b_static is not None and self.a_static is not None
        for row in rows:
            try:
                frame = int(row[1])
            except Exception:
                frame = None
            timestamp_ms = float(row[0])
            if frame is not None and self.last_frame is not None and frame <= self.last_frame:
                continue
            if frame is None and self.last_timestamp_ms is not None and timestamp_ms <= self.last_timestamp_ms:
                continue

            real = parse_json_array(row[9])
            imag = parse_json_array(row[10])
            if len(real) < self.m or len(imag) < self.m:
                continue
            z = real[: self.m] + 1j * imag[: self.m]
            # Keep the low-pass clutter estimate warm for diagnostics/future use, but do not
            # take phase from z-static.  The residual vector frequently crosses the origin for
            # normal sub-mm chest motion, which makes atan2 jump by pi and produces random flat
            # segments after band-pass filtering.  Inter-frame phase from the raw complex return
            # is much more stable for live traces.
            self._iir_step(z, self.b_static, self.a_static, self.static_x, self.static_y)
            amp = np.abs(z)
            if self.lp_filt_ampl is None:
                self.lp_filt_ampl = amp.copy()
            else:
                self.lp_filt_ampl = self.sf * self.lp_filt_ampl + (1.0 - self.sf) * amp

            angle = np.angle(z)
            if self.prev_angle is None or self.angle_unwrapped is None:
                self.prev_angle = angle.copy()
                self.angle_unwrapped = np.zeros(self.m, dtype=float)
            else:
                diff = angle - self.prev_angle
                diff[diff > np.pi] -= 2.0 * np.pi
                diff[diff < -np.pi] += 2.0 * np.pi
                self.angle_unwrapped = self.angle_unwrapped + diff
                self.prev_angle = angle.copy()

            selected_idx = self._select_index(self.lp_filt_ampl if self.lp_filt_ampl is not None else amp, target_distance_m)
            candidate_idx = self._candidate_indices(selected_idx, gate_half_width_m, use_gating)
            weights = self.lp_filt_ampl[candidate_idx] if self.lp_filt_ampl is not None and len(candidate_idx) else np.asarray([1.0])
            weights = np.maximum(weights.astype(float), 0.0)
            weights = weights / (float(np.sum(weights)) + 1e-12)
            if len(candidate_idx) > 1:
                selected_pos = np.flatnonzero(candidate_idx == selected_idx)
                if len(selected_pos):
                    # Live processing cannot retrospectively sign-align neighboring range bins
                    # like the batch analyzer can. Keep the selected bin dominant so a nearby
                    # multipath bin with opposite phase cannot cancel the trace into a flat line.
                    weights *= 0.35
                    weights[int(selected_pos[0])] += 0.65
                    weights = weights / (float(np.sum(weights)) + 1e-12)

            # DC-block the cumulative phase before bandpass filtering.  Without this the
            # IIR bandpass state tracks the massive monotonic ramp (~1400+ rad) and produces
            # random/flat output whenever the drift slope changes.
            if self.b_dc is not None and self.a_dc is not None and self.dc_x is not None and self.dc_y is not None:
                phase_ac = self._iir_step(self.angle_unwrapped, self.b_dc, self.a_dc, self.dc_x, self.dc_y)
            else:
                phase_ac = self.angle_unwrapped

            if self.b_resp is not None and self.a_resp is not None and self.resp_x is not None and self.resp_y is not None:
                resp_all = self._iir_step(phase_ac, self.b_resp, self.a_resp, self.resp_x, self.resp_y)
            else:
                resp_all = phase_ac
            if self.b_heart is not None and self.a_heart is not None and self.heart_x is not None and self.heart_y is not None:
                heart_all = self._iir_step(phase_ac, self.b_heart, self.a_heart, self.heart_x, self.heart_y)
            else:
                heart_all = np.zeros_like(self.angle_unwrapped)

            if self.first_timestamp_ms is None:
                self.first_timestamp_ms = timestamp_ms
            time_s = (timestamp_ms - self.first_timestamp_ms) / 1000.0
            if self.last_time_s is not None and time_s <= self.last_time_s:
                time_s = self.last_time_s + 1.0 / max(self.fs, 1e-9)
            self.last_time_s = float(time_s)
            self.last_timestamp_ms = timestamp_ms
            if frame is not None:
                self.last_frame = frame

            self.processed_count += 1
            self.times.append(float(time_s))
            self.raw_phase.append(float(self.angle_unwrapped[selected_idx]))
            self.raw_i.append(float(np.real(z[selected_idx])))
            self.raw_q.append(float(np.imag(z[selected_idx])))
            if len(candidate_idx):
                self.resp_signal.append(float(np.sum(resp_all[candidate_idx] * weights)))
                self.heart_signal.append(float(np.sum(heart_all[candidate_idx] * weights)))
            else:
                self.resp_signal.append(float(resp_all[selected_idx]))
                self.heart_signal.append(float(heart_all[selected_idx]))

        return self.result()

    def result(self) -> A121LiveTraceResult | None:
        if not self.times or self.selected_index is None:
            return None
        return A121LiveTraceResult(
            sample_rate_hz=float(self.fs),
            target_distance_m=float(self.target_distance_m),
            selected_index=int(self.selected_index),
            times_s=np.asarray(self.times, dtype=float),
            raw_phase=np.asarray(self.raw_phase, dtype=float),
            raw_i=np.asarray(self.raw_i, dtype=float),
            raw_q=np.asarray(self.raw_q, dtype=float),
            resp_signal=np.asarray(self.resp_signal, dtype=float),
            heart_signal=np.asarray(self.heart_signal, dtype=float),
        )


class HeartRateKalmanTracker:
    """Small constant-acceleration Kalman tracker for live A121 heart-rate display.

    The analysis function is intentionally stateless so it also works for CSV/history views.
    The Qt live view keeps one instance of this tracker and feeds its current state back as
    an adaptive search prior, which prevents the displayed cardiac peak from jumping to short
    respiratory harmonics or motion spikes.
    """

    def __init__(self, band_hz: tuple[float, float] = HEART_BAND_HZ):
        self.band_hz = band_hz
        self.reset()

    def reset(self) -> None:
        self.x = np.asarray([0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([0.18**2, 0.08**2, 0.04**2]).astype(float)
        self.locked = False
        self.missed = 0

    @property
    def current_hz(self) -> float | None:
        return float(self.x[0]) if self.locked and self.band_hz[0] <= self.x[0] <= self.band_hz[1] else None

    @property
    def current_std_hz(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 1e-6))) if self.locked else 0.35

    def search_band(self) -> tuple[float, float]:
        h = self.current_hz
        if h is None:
            return self.band_hz
        half = max(HEART_RATE_LOCK_HALF_WIDTH_HZ, 3.0 * self.current_std_hz)
        return _clamp_band((h - half, h + half), self.band_hz, min_width=HEART_RATE_LOCK_MIN_WIDTH_HZ)

    def update(self, measurement_hz: float, dt_s: float, *, confidence: float = 0.0, quality: float = 0.0) -> float:
        low, high = self.band_hz
        valid = bool(low <= measurement_hz <= high and confidence >= HEART_RATE_CONFIDENCE_MIN and quality >= 0.25)
        if not self.locked:
            if valid:
                initial_std = 0.045 if confidence >= HEART_RATE_STRONG_LOCK_CONFIDENCE else 0.10
                self.x[:] = [measurement_hz, 0.0, 0.0]
                self.P = np.diag([initial_std**2, 0.05**2, 0.025**2]).astype(float)
                self.locked = True
                self.missed = 0
                return float(measurement_hz)
            return 0.0

        dt = float(np.clip(dt_s, 0.05, 2.0))
        A = np.asarray([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], dtype=float)
        G = np.asarray([[0.5 * dt * dt], [dt], [1.0]], dtype=float)
        rho_a = 0.065  # Hz/s^2, allows physiological drift without following spikes
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + (G @ G.T) * rho_a**2
        self.x[0] = float(np.clip(self.x[0], low, high))

        if valid:
            H = np.asarray([[1.0, 0.0, 0.0]], dtype=float)
            # Stronger spectral confidence gives lower measurement variance.
            meas_std = float(np.clip(0.22 / np.sqrt(max(confidence, 1.0)), 0.035, 0.22))
            R = meas_std**2
            innovation = float(measurement_hz - (H @ self.x)[0])
            S = float((H @ self.P @ H.T)[0, 0] + R)
            d2 = (innovation * innovation) / max(S, 1e-9)
            if d2 <= 9.0 or self.missed >= 4:  # chi-square gate, with recovery widening
                K = (self.P @ H.T) / max(S, 1e-9)
                self.x = self.x + (K[:, 0] * innovation)
                self.P = (np.eye(3) - K @ H) @ self.P
                self.x[0] = float(np.clip(self.x[0], low, high))
                if confidence >= HEART_RATE_STRONG_LOCK_CONFIDENCE:
                    self.P[0, 0] = min(float(self.P[0, 0]), 0.040**2)
                self.missed = 0
            else:
                self.missed += 1
        else:
            self.missed += 1

        if self.missed >= 12:
            self.reset()
            return 0.0
        return float(self.x[0]) if self.locked else 0.0


def sample_rate_from_ms(times_ms: np.ndarray, default: float = 20.0) -> float:
    """Estimate frame rate from timestamps, handling A121 UART burst delivery.

    Older recordings used host receive timestamps. The Acconeer client often delivers frames in
    bursts, producing many duplicate/sub-millisecond deltas followed by long gaps. A median-delta
    estimator then underestimates the real frame rate and shifts the vital FFT bands. Prefer the
    total frame count over elapsed time whenever the timestamps look bursty; otherwise keep a
    robust mean of sane inter-frame deltas.
    """
    times = np.asarray(times_ms, dtype=float)
    times = times[np.isfinite(times)]
    if len(times) < 2:
        return default

    elapsed_s = float((times[-1] - times[0]) / 1000.0)
    total_fs = float((len(times) - 1) / elapsed_s) if elapsed_s > 0 else 0.0

    raw_diffs = np.diff(times) / 1000.0
    finite = raw_diffs[np.isfinite(raw_diffs)]
    positive = finite[finite > 0]
    if len(positive) == 0:
        return total_fs or default

    median = float(np.median(positive))
    very_short = positive < max(0.002, median * 0.20)
    non_positive_fraction = float(np.count_nonzero(finite <= 0) / max(len(finite), 1))
    bursty_fraction = non_positive_fraction + float(np.count_nonzero(very_short) / max(len(finite), 1))
    if total_fs > 0 and (bursty_fraction > 0.05 or abs((1.0 / median) - total_fs) / max(total_fs, 1e-9) > 0.25):
        return total_fs

    good = positive[(positive > median * 0.25) & (positive < median * 4.0)] if median > 0 else positive
    robust = float(1.0 / np.mean(good if len(good) else positive))
    return robust if np.isfinite(robust) and robust > 0 else (total_fs or default)


def _times_seconds_from_ms(times_ms: np.ndarray, fs: float) -> np.ndarray:
    times = np.asarray(times_ms, dtype=float)
    if len(times) == 0:
        return np.asarray([], dtype=float)
    wall = (times - float(times[0])) / 1000.0
    if len(times) < 3 or fs <= 0:
        return wall
    diffs = np.diff(times) / 1000.0
    finite = diffs[np.isfinite(diffs)]
    if len(finite) == 0:
        return np.arange(len(times), dtype=float) / fs
    positive = finite[finite > 0]
    median = float(np.median(positive)) if len(positive) else 0.0
    bursty = float(np.count_nonzero(finite <= 0) / max(len(finite), 1)) > 0.05
    if len(positive) and median > 0:
        bursty = bursty or float(np.count_nonzero(positive < max(0.002, median * 0.20)) / max(len(finite), 1)) > 0.05
    nominal = np.arange(len(times), dtype=float) / fs
    if bursty and wall[-1] > 0:
        # Preserve the total recording duration while removing burst/duplicate timestamp jitter.
        return nominal * (wall[-1] / max(nominal[-1], 1e-9))
    return wall


def _interp_nonfinite(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float).copy()
    if len(x) == 0:
        return x
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return x


def _despike(values: np.ndarray, sigma: float = 7.0) -> np.ndarray:
    x = _interp_nonfinite(values)
    if len(x) < 8:
        return x - float(np.mean(x)) if len(x) else x
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad > 1e-12:
        x = np.clip(x, med - sigma * mad, med + sigma * mad)
    return x


def _poly_detrend_1d(values: np.ndarray, degree: int = 5) -> np.ndarray:
    x = _interp_nonfinite(values)
    if len(x) < 24:
        return x - float(np.mean(x)) if len(x) else x
    deg = int(min(degree, max(1, len(x) // 50), len(x) - 2))
    t = np.linspace(-1.0, 1.0, len(x), dtype=float)
    try:
        coeff = np.polyfit(t, x, deg=deg)
        return x - np.polyval(coeff, t)
    except Exception:
        try:
            return detrend(x, type="linear")
        except Exception:
            return x - float(np.mean(x))


def clean_signal(values: np.ndarray) -> np.ndarray:
    x = _despike(values)
    if len(x) < 3:
        return x
    try:
        from scipy.signal import detrend
        return detrend(x, type="linear")
    except Exception:
        return x - float(np.mean(x))


def bandpass_filter(values: np.ndarray, fs: float, band_hz: tuple[float, float], order: int = 2) -> np.ndarray:
    x = clean_signal(values)
    if len(x) < 12 or fs <= 0:
        return x
    low, high = band_hz
    nyquist = fs * 0.5
    high = min(high, nyquist * 0.92)
    low = max(low, 1.0 / max(len(x) / fs, 1e-9))
    if not (0 < low < high < nyquist):
        return x
    try:
        sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
        return sosfiltfilt(sos, x)
    except ValueError:
        return x


def estimate_band_peak(
    values: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    reject_hz: tuple[float, ...] = (),
) -> tuple[float, float]:
    x = clean_signal(values)
    if len(x) < 16 or fs <= 0:
        return 0.0, 0.0
    duration_s = len(x) / fs
    low, high = band_hz
    low = max(low, 1.0 / max(duration_s, 1e-9))
    high = min(high, fs * 0.46)
    if low >= high:
        return 0.0, 0.0
    nperseg = min(len(x), max(64, int(round(fs * min(24.0, duration_s)))))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, scaling="spectrum")
    valid = (freqs >= low) & (freqs <= high)
    if not np.any(valid):
        return 0.0, 0.0
    f = freqs[valid]
    p = psd[valid].astype(float)
    score = p.copy()
    for rejected in reject_hz:
        if rejected <= 0:
            continue
        # Wider notch with deeper suppression for respiratory harmonics.
        # At typical resting breathing rates (~0.15 Hz), harmonics land every
        # ~0.15 Hz throughout the heart band.  A narrow/shallow notch misses
        # most of the harmonic energy, causing the peak detector to jump.
        width = max(0.040, rejected * 0.030)
        score *= 1.0 - 0.75 * np.exp(-0.5 * ((f - rejected) / width) ** 2)
    # Suppress edge bins: the first and last few FFT bins at the band boundary
    # often accumulate filter roll-off energy and cause false peak locks (e.g.
    # locking onto 0.70 Hz / 42 BPM at the heart band edge).  Apply a smooth
    # taper over ~3 bins from each edge so the peak detector ignores them.
    freq_span = float(f[-1] - f[0]) if len(f) > 1 else 1.0
    bin_width = freq_span / max(len(f) - 1, 1)
    edge_bins = max(3, int(0.05 / max(bin_width, 1e-9)))  # ~0.05 Hz margin
    if len(f) > 2 * edge_bins:
        taper = np.ones(len(f), dtype=float)
        for i in range(edge_bins):
            w = float(i + 1) / float(edge_bins + 1)
            taper[i] *= w
            taper[-(i + 1)] *= w
        score *= taper
    idx = int(np.argmax(score))
    peak_hz = float(f[idx])
    if 0 < idx < len(f) - 1 and len(f) > 2:
        # Quadratic interpolation in log power improves rate stability between FFT bins.
        y0, y1, y2 = np.log(score[idx - 1 : idx + 2] + 1e-24)
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            peak_hz = float(np.clip(f[idx] + delta * (f[1] - f[0]), low, high))
    noise = float(np.median(p)) + 1e-18
    confidence = float(p[idx] / noise)
    return peak_hz, confidence


def estimate_autocorr_peak(
    values: np.ndarray,
    fs: float,
    band_hz: tuple[float, float]
) -> tuple[float, float]:
    x = clean_signal(values)
    if len(x) < 12 or fs <= 0:
        return 0.0, 0.0
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n-1:]
    acf /= (acf[0] + 1e-12)
    
    # Search for peak in the lag range corresponding to band_hz
    min_lag = int(np.floor(fs / band_hz[1]))
    max_lag = int(np.ceil(fs / band_hz[0]))
    min_lag = max(1, min_lag)
    max_lag = min(n - 2, max_lag)
    if min_lag >= max_lag:
        return 0.0, 0.0
        
    # Search for local maxima in acf within [min_lag, max_lag]
    peaks = []
    for i in range(min_lag, max_lag + 1):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1]:
            peaks.append(i)
            
    if not peaks:
        # Fallback to absolute maximum in range
        best_lag = int(min_lag + np.argmax(acf[min_lag:max_lag+1]))
    else:
        best_lag = max(peaks, key=lambda idx: acf[idx])
        
    # Quadratic interpolation on ACF peak for sub-sample precision
    y0, y1, y2 = acf[best_lag-1], acf[best_lag], acf[best_lag+1]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.0
    if abs(denom) > 1e-12:
        delta = 0.5 * (y0 - y2) / denom
    refined_lag = float(best_lag) + delta
    
    peak_hz = float(fs / refined_lag)
    confidence = float(acf[best_lag])
    return peak_hz, confidence


def _frequency_is_rejected(frequency_hz: float, reject_hz: tuple[float, ...]) -> bool:
    if frequency_hz <= 0:
        return False
    return any(rejected > 0 and abs(frequency_hz - rejected) <= max(0.040, rejected * 0.030) for rejected in reject_hz)


def _periodic_peak_score(values: np.ndarray, fs: float, frequency_hz: float) -> float:
    """Validate an FFT candidate with time-domain peak spacing.

    This is intentionally only a validator, not the primary estimator: breathing harmonics can be
    spectrally strong in the heart band, but they often fail to form a stable beat-to-beat peak
    train once the signal is narrowed around the candidate.  A high score boosts/locks a heart
    candidate; a low score especially demotes candidates sitting exactly on respiration harmonics.
    """
    if frequency_hz <= 0 or fs <= 0:
        return 0.0
    x = clean_signal(values)
    duration_s = len(x) / fs if fs > 0 else 0.0
    expected_period_s = 1.0 / frequency_hz
    if len(x) < 24 or duration_s < max(6.0, expected_period_s * 4.0):
        return 0.0

    half_width = max(0.12, min(0.22, frequency_hz * 0.18))
    narrow_band = _clamp_band((frequency_hz - half_width, frequency_hz + half_width), HEART_BAND_HZ, min_width=0.18)
    narrowed = _bandpass_matrix(x, fs, narrow_band, order=2, zero_phase=True).ravel()
    narrowed = clean_signal(narrowed)
    sigma = float(np.std(narrowed))
    if not np.isfinite(sigma) or sigma <= 1e-9:
        return 0.0

    expected_samples = max(fs / frequency_hz, 1.0)
    min_distance = max(1, int(round(expected_samples * 0.52)))
    prominence = max(sigma * 0.18, 1e-12)

    best = 0.0
    for candidate_signal in (narrowed, -narrowed):
        peaks, _ = find_peaks(candidate_signal, distance=min_distance, prominence=prominence)
        if len(peaks) < 4:
            continue
        intervals = np.diff(peaks).astype(float) / fs
        if len(intervals) < 3:
            continue
        rel_error = np.abs(intervals - expected_period_s) / max(expected_period_s, 1e-9)
        good_fraction = float(np.mean(rel_error <= 0.32))
        mean_interval = float(np.mean(intervals))
        cv = float(np.std(intervals) / max(mean_interval, 1e-9))
        regularity = float(np.clip(1.0 - cv / 0.42, 0.0, 1.0))
        expected_count = max(duration_s * frequency_hz, 1.0)
        coverage = float(np.clip((len(peaks) - 1) / expected_count, 0.0, 1.0))
        span = float((peaks[-1] - peaks[0]) / fs) if len(peaks) else 0.0
        span_fraction = float(np.clip(span / max(duration_s * 0.75, 1e-9), 0.0, 1.0))
        score = 0.38 * good_fraction + 0.27 * regularity + 0.20 * coverage + 0.15 * span_fraction
        best = max(best, score)
    return float(np.clip(best, 0.0, 1.0))


def _spectral_peak_candidates(
    values: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    reject_hz: tuple[float, ...],
    *,
    max_candidates: int = 6,
) -> list[tuple[float, float, float]]:
    x = clean_signal(values)
    if len(x) < 16 or fs <= 0:
        return []
    duration_s = len(x) / fs
    low, high = band_hz
    low = max(low, 1.0 / max(duration_s, 1e-9))
    high = min(high, fs * 0.46)
    if low >= high:
        return []
    nperseg = min(len(x), max(64, int(round(fs * min(24.0, duration_s)))))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, scaling="spectrum")
    valid = (freqs >= low) & (freqs <= high)
    if not np.any(valid):
        return []
    f = freqs[valid]
    p = psd[valid].astype(float)
    score = p.copy()
    for rejected in reject_hz:
        if rejected <= 0:
            continue
        width = max(0.040, rejected * 0.030)
        score *= 1.0 - 0.78 * np.exp(-0.5 * ((f - rejected) / width) ** 2)

    freq_span = float(f[-1] - f[0]) if len(f) > 1 else 1.0
    bin_width = freq_span / max(len(f) - 1, 1)
    edge_bins = max(3, int(0.05 / max(bin_width, 1e-9)))
    apply_edge_taper = (float(band_hz[1]) - float(band_hz[0])) >= 0.60
    if apply_edge_taper and len(f) > 2 * edge_bins:
        taper = np.ones(len(f), dtype=float)
        for i in range(edge_bins):
            w = float(i + 1) / float(edge_bins + 1)
            taper[i] *= w
            taper[-(i + 1)] *= w
        score *= taper

    if len(score) >= 3:
        local_idx = [idx for idx in range(1, len(score) - 1) if score[idx] >= score[idx - 1] and score[idx] >= score[idx + 1]]
    else:
        local_idx = []
    global_idx = int(np.argmax(score))
    if global_idx not in local_idx:
        local_idx.append(global_idx)
    local_idx = sorted(local_idx, key=lambda idx: float(score[idx]), reverse=True)[:max_candidates]

    noise = float(np.median(p)) + 1e-18
    score_noise = float(np.median(score)) + 1e-18
    candidates: list[tuple[float, float, float]] = []
    for idx in local_idx:
        peak_hz = float(f[idx])
        if 0 < idx < len(f) - 1 and len(f) > 2:
            y0, y1, y2 = np.log(score[idx - 1 : idx + 2] + 1e-24)
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
                peak_hz = float(np.clip(f[idx] + delta * (f[1] - f[0]), low, high))
        raw_confidence = float(p[idx] / noise)
        scored_confidence = float(score[idx] / score_noise)
        candidates.append((peak_hz, raw_confidence, scored_confidence))
    return candidates


def estimate_band_peak_fused(
    values: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    reject_hz: tuple[float, ...] = (),
) -> tuple[float, float]:
    candidates = _spectral_peak_candidates(values, fs, band_hz, reject_hz)
    if not candidates:
        return 0.0, 0.0

    best_hz = 0.0
    best_confidence = 0.0
    best_score = -1.0
    for hz, raw_conf, scored_conf in candidates:
        periodicity = _periodic_peak_score(values, fs, hz)
        near_reject = _frequency_is_rejected(hz, reject_hz)
        harmonic_guard = 0.62 if near_reject and periodicity < 0.45 else 1.0
        combined_score = harmonic_guard * (scored_conf * (0.72 + 0.55 * periodicity) + 2.8 * periodicity)
        if combined_score > best_score:
            best_score = combined_score
            best_hz = hz
            best_confidence = max(scored_conf, min(raw_conf, scored_conf * 1.6)) * (0.85 + 0.35 * periodicity) * harmonic_guard

    acf_hz, acf_conf = estimate_autocorr_peak(values, fs, band_hz)
    if _frequency_is_rejected(acf_hz, reject_hz):
        acf_conf = 0.0

    if acf_hz > 0 and acf_conf >= 0.22 and best_hz > 0 and best_confidence >= 2.0:
        # Use autocorrelation only as a confirmer. When the FFT is weak, the ACF often locks
        # onto motion bursts or respiratory harmonics and makes the live heart rate jump.
        if abs(acf_hz - best_hz) <= 0.22:
            blended = 0.70 * best_hz + 0.30 * acf_hz
            return float(blended), float(min(max(best_confidence, acf_conf * 10.0), 100.0))

    return float(best_hz), float(min(best_confidence, 100.0))


def _json_matrix(series: pd.Series) -> np.ndarray:
    if len(series) == 0:
        return np.empty((0, 0), dtype=float)
    first = series.iloc[0]
    if isinstance(first, np.ndarray):
        return np.vstack(series.values).astype(float)
    if isinstance(first, list):
        return np.vstack(series.values).astype(float)
    import json
    parsed = [json.loads(val) if isinstance(val, str) else val for val in series]
    return np.vstack(parsed).astype(float)


def _smooth_bins(score: np.ndarray) -> np.ndarray:
    if len(score) < 5:
        return score
    kernel = np.array([0.12, 0.20, 0.36, 0.20, 0.12], dtype=float)
    return np.convolve(score, kernel / kernel.sum(), mode="same")


def _circle_fit_center(i_values: np.ndarray, q_values: np.ndarray) -> tuple[float, float, float, float]:
    x = _interp_nonfinite(i_values)
    y = _interp_nonfinite(q_values)
    if len(x) < 8:
        return 0.0, 0.0, float(np.median(np.hypot(x, y))) if len(x) else 0.0, 1.0
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    u = x - xm
    v = y - ym
    try:
        mat = np.column_stack([u, v, np.ones_like(u)])
        rhs = -(u * u + v * v)
        a, b, c = np.linalg.lstsq(mat, rhs, rcond=None)[0]
        cx = xm - 0.5 * float(a)
        cy = ym - 0.5 * float(b)
        r2 = max(0.0, 0.25 * float(a * a + b * b) - float(c))
        radius = float(np.sqrt(r2))
        radial = np.hypot(x - cx, y - cy)
        rms = float(np.sqrt(np.mean(np.square(radial - radius))) / (radius + 1e-12))
        if not np.isfinite(radius) or radius <= 1e-9 or not np.isfinite(rms):
            return 0.0, 0.0, float(np.median(np.hypot(x, y))), 1.0
        return cx, cy, radius, rms
    except Exception:
        return 0.0, 0.0, float(np.median(np.hypot(x, y))), 1.0


def _center_complex_profile(real: np.ndarray, imag: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    N, M = real.shape
    if N < 8:
        centered = real + 1j * imag
        quality = np.full(M, 0.25, dtype=float)
        return centered, quality, 0.25

    xm = np.mean(real, axis=0)
    ym = np.mean(imag, axis=0)
    u = real - xm
    v = imag - ym

    suu = np.sum(u * u, axis=0)
    svv = np.sum(v * v, axis=0)
    suv = np.sum(u * v, axis=0)

    w = -(u * u + v * v)
    suw = np.sum(u * w, axis=0)
    svw = np.sum(v * w, axis=0)
    sw = np.sum(w, axis=0)

    det = suu * svv - suv * suv
    valid = det > 1e-12

    a = np.zeros(M)
    b = np.zeros(M)
    c = np.zeros(M)

    a[valid] = (svv[valid] * suw[valid] - suv[valid] * svw[valid]) / det[valid]
    b[valid] = (-suv[valid] * suw[valid] + suu[valid] * svw[valid]) / det[valid]
    c[valid] = sw[valid] / N

    cx = xm - 0.5 * a
    cy = ym - 0.5 * b
    r2 = np.maximum(0.25 * (a * a + b * b) - c, 0.0)
    radius = np.sqrt(r2)

    radial = np.hypot(real - cx[None, :], imag - cy[None, :])
    rms = np.sqrt(np.mean(np.square(radial - radius[None, :]), axis=0)) / (radius + 1e-12)

    med_amp = np.median(np.hypot(real, imag), axis=0)
    plausible = valid & (radius > med_amp * 0.05) & (radius < med_amp * 20.0) & (rms < 0.45)

    centered = np.where(plausible[None, :], (real - cx[None, :]) + 1j * (imag - cy[None, :]), real + 1j * imag)
    quality = np.where(plausible, np.clip(1.0 - rms / 0.45, 0.0, 1.0), 0.25)
    
    return centered, quality, float(np.mean(quality)) if len(quality) else 0.0


def _phase_matrix(centered_complex: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(centered_complex), axis=0)
    phase = phase - np.median(phase, axis=0, keepdims=True)
    return np.apply_along_axis(clean_signal, 0, phase)


def _band_energy_ratio(phase: np.ndarray, fs: float) -> np.ndarray:
    if phase.shape[0] < 16 or fs <= 0:
        return np.ones(phase.shape[1], dtype=float)
    window = np.hanning(phase.shape[0])[:, None]
    spec = np.square(np.abs(np.fft.rfft(phase * window, axis=0)))
    freqs = np.fft.rfftfreq(phase.shape[0], d=1.0 / fs)
    vital = (freqs >= A121_RESP_BAND_HZ[0]) & (freqs <= min(HEART_BAND_HZ[1], fs * 0.46))
    background = (freqs >= 0.03) & (freqs <= min(fs * 0.46, 5.0)) & ~vital
    if not np.any(vital) or not np.any(background):
        return np.ones(phase.shape[1], dtype=float)
    in_energy = np.mean(spec[vital], axis=0)
    out_energy = np.mean(spec[background], axis=0) + 1e-18
    return in_energy / out_energy


def _select_msp_weights(
    score: np.ndarray,
    gate: np.ndarray,
    energy_ratio: np.ndarray,
    selected_idx: int,
    *,
    max_bins: int = 9,
) -> tuple[np.ndarray, int, int]:
    weights = np.zeros_like(score, dtype=float)
    if len(score) == 0:
        return weights, selected_idx, 0
    gate = np.asarray(gate, dtype=bool)
    if not np.any(gate):
        gate = np.zeros_like(score, dtype=bool)
        gate[int(np.clip(selected_idx, 0, len(score) - 1))] = True
    gated_score = np.where(gate, np.maximum(score, 0.0), 0.0)
    if float(np.sum(gated_score)) <= 0.0:
        gated_score[gate] = 1.0
    local = gated_score[gate]
    threshold = float(np.mean(local) + 0.20 * np.max(local)) if len(local) else 0.0
    candidate = gate & (gated_score >= threshold) & (energy_ratio >= 1.8)
    if np.count_nonzero(candidate) == 0:
        cutoff = float(np.percentile(local, 65.0)) if len(local) else 0.0
        candidate = gate & (gated_score >= cutoff)
    candidate_idx = np.flatnonzero(candidate)
    if len(candidate_idx) == 0:
        candidate_idx = np.asarray([int(np.clip(selected_idx, 0, len(score) - 1))])
    if len(candidate_idx) > max_bins:
        order = np.argsort(gated_score[candidate_idx])[-max_bins:]
        candidate_idx = candidate_idx[order]
    weights[candidate_idx] = gated_score[candidate_idx]
    if float(np.sum(weights)) <= 0.0:
        weights[candidate_idx] = 1.0
    weights /= float(np.sum(weights)) + 1e-12
    best_idx = int(candidate_idx[int(np.argmax(gated_score[candidate_idx]))])
    return weights, best_idx, int(len(candidate_idx))


def _differential_phase_matrix(complex_profile: np.ndarray) -> np.ndarray:
    if complex_profile.shape[0] == 0:
        return np.empty(complex_profile.shape, dtype=float)
    unit = complex_profile / (np.abs(complex_profile) + 1e-12)
    delta = np.angle(unit[1:] * np.conj(unit[:-1])) if complex_profile.shape[0] > 1 else np.empty((0, complex_profile.shape[1]))
    phase = np.vstack([np.zeros((1, complex_profile.shape[1]), dtype=float), np.cumsum(delta, axis=0)])
    try:
        from scipy.signal import detrend
        return detrend(phase, axis=0)
    except Exception:
        return phase - np.mean(phase, axis=0, keepdims=True)


def _coherent_differential_phase(complex_profile: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(weights > 0)
    if len(indices) == 0 or complex_profile.shape[0] == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    z = complex_profile[:, indices]
    w = weights[indices]
    w = w / (float(np.sum(w)) + 1e-12)
    unit = z / (np.abs(z) + 1e-12)
    if len(unit) > 1:
        # Acconeer's breathing reference uses cumulative inter-frame phase changes.
        # This is much less noisy for short arcs than re-fitting an absolute IQ circle
        # every GUI update, and it avoids random arctangent jumps from weak bins.
        cross = np.sum(w[None, :] * unit[1:] * np.conj(unit[:-1]), axis=1)
        raw_phase = np.concatenate([[0.0], np.cumsum(np.angle(cross))])
    else:
        raw_phase = np.zeros(len(unit), dtype=float)
    raw_phase = clean_signal(raw_phase)

    ref_idx = int(indices[int(np.argmax(weights[indices]))])
    ref = complex_profile[:, ref_idx]
    combined = np.zeros(complex_profile.shape[0], dtype=np.complex128)
    for idx in indices:
        series = complex_profile[:, idx]
        align = np.mean(series * np.conj(ref))
        offset = float(np.angle(align)) if np.isfinite(align) and abs(align) > 1e-12 else 0.0
        combined += float(weights[idx]) * series * np.exp(-1j * offset)
    return raw_phase, np.real(combined), np.imag(combined)


def _subtract_resp_harmonics(values: np.ndarray, fs: float, resp_hz: float, max_order: int = 6) -> np.ndarray:
    x = clean_signal(values)
    if len(x) < 24 or fs <= 0 or resp_hz <= 0:
        return x
    duration = len(x) / fs
    t = np.arange(len(x), dtype=float) / fs
    cols: list[np.ndarray] = []
    for order in range(2, max_order + 1):
        freq = resp_hz * order
        if HEART_BAND_HZ[0] * 0.85 <= freq <= min(HEART_BAND_HZ[1] * 1.05, fs * 0.46) and duration * freq >= 2.0:
            cols.append(np.sin(2.0 * np.pi * freq * t))
            cols.append(np.cos(2.0 * np.pi * freq * t))
    if not cols or len(x) <= len(cols) + 8:
        return x
    design = np.column_stack(cols)
    try:
        coeff = np.linalg.lstsq(design, x, rcond=None)[0]
        return clean_signal(x - design @ coeff)
    except Exception:
        return x


def _remove_resp_correlated_motion(values: np.ndarray, resp_signal: np.ndarray, fs: float) -> np.ndarray:
    """Subtract motion components that are coherent with the breathing waveform.

    Normal breathing creates large nonlinear chest motion; its harmonics can sit directly inside
    the heart band.  Instead of blindly notching every respiratory harmonic, fit only components
    that are actually correlated with the measured respiration waveform and subtract them softly.
    This preserves a real pulse near a harmonic better than a hard sinusoid comb filter.
    """
    x_full = clean_signal(np.asarray(values, dtype=float))
    r_full = clean_signal(np.asarray(resp_signal, dtype=float))
    n = min(len(x_full), len(r_full))
    if fs <= 0 or n < max(160, int(round(fs * 10.0))):
        return x_full

    x = x_full[-n:]
    r = r_full[-n:]
    r_std = float(np.std(r))
    if not np.isfinite(r_std) or r_std <= 1e-9:
        return x_full
    r = (r - float(np.mean(r))) / r_std

    columns: list[np.ndarray] = []
    # Powers 2-6 model common nonlinear respiratory harmonics without explicitly deleting every
    # sin/cos harmonic.  The subtraction is intentionally soft so a real pulse near a harmonic
    # can survive and be recovered by the peak-spacing validator.
    for power in (2, 3, 4, 5, 6):
        feature = np.power(r, power)
        feature = clean_signal(feature)
        feature = _bandpass_matrix(feature, fs, HEART_BAND_HZ, order=2, zero_phase=True).ravel()
        scale = float(np.std(feature))
        if np.isfinite(scale) and scale > 1e-9:
            columns.append(feature / scale)
    if not columns:
        return x_full

    design = np.column_stack(columns)
    if design.shape[0] <= design.shape[1] + 8:
        return x_full
    try:
        xtx = design.T @ design
        ridge = max(float(np.trace(xtx)) / max(xtx.shape[0], 1), 1e-9) * 0.03
        coeff = np.linalg.solve(xtx + np.eye(xtx.shape[0]) * ridge, design.T @ x)
        artifact = design @ coeff
        residual = clean_signal(x - 0.70 * artifact)
        out = x_full.copy()
        out[-n:] = residual
        return out
    except Exception:
        return x_full


def _spectral_entropy(values: np.ndarray, fs: float, band_hz: tuple[float, float]) -> float:
    x = clean_signal(values)
    if len(x) < 16 or fs <= 0:
        return 1.0
    nperseg = min(len(x), max(64, int(round(fs * min(30.0, len(x) / fs)))))
    freqs, power = welch(x, fs=fs, nperseg=nperseg, scaling="spectrum")
    valid = (freqs >= band_hz[0]) & (freqs <= min(band_hz[1], fs * 0.46))
    if not np.any(valid):
        return 1.0
    p = np.maximum(power[valid], 0.0)
    total = float(np.sum(p))
    if total <= 0.0:
        return 1.0
    prob = p / total
    if len(prob) <= 1:
        return 1.0
    return float(-np.sum(prob * np.log(prob + 1e-18)) / np.log(len(prob)))


def _clamp_band(band: tuple[float, float], limits: tuple[float, float], min_width: float = 0.25) -> tuple[float, float]:
    low = max(float(band[0]), float(limits[0]))
    high = min(float(band[1]), float(limits[1]))
    if high - low < min_width:
        center = float(np.clip((low + high) * 0.5, limits[0] + min_width * 0.5, limits[1] - min_width * 0.5))
        low = max(limits[0], center - min_width * 0.5)
        high = min(limits[1], center + min_width * 0.5)
    return low, high


def _heart_search_band(heart_prior_hz: float | None, heart_prior_std_hz: float | None) -> tuple[float, float]:
    if heart_prior_hz is None or not np.isfinite(heart_prior_hz) or heart_prior_hz <= 0:
        return HEART_BAND_HZ
    half = max(HEART_RATE_LOCK_HALF_WIDTH_HZ, 3.0 * float(heart_prior_std_hz if heart_prior_std_hz is not None else 0.12))
    return _clamp_band(
        (float(heart_prior_hz) - half, float(heart_prior_hz) + half),
        HEART_BAND_HZ,
        min_width=HEART_RATE_LOCK_MIN_WIDTH_HZ,
    )


def _valid_band_for_fs(band_hz: tuple[float, float], fs: float, *, min_width: float = 0.02) -> tuple[float, float] | None:
    if fs <= 0:
        return None
    nyquist = 0.5 * fs
    low = max(float(band_hz[0]), 1e-5)
    high = min(float(band_hz[1]), nyquist * 0.92)
    if high - low < min_width or not (0 < low < high < nyquist):
        return None
    return low, high


def _bandpass_matrix(values: np.ndarray, fs: float, band_hz: tuple[float, float], *, order: int = 2, zero_phase: bool = False) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < 8:
        return x.copy()
    band = _valid_band_for_fs(band_hz, fs)
    cleaned = np.apply_along_axis(clean_signal, 0, x)
    if band is None:
        return cleaned
    try:
        if zero_phase and x.shape[0] >= max(24, order * 9):
            sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
            return sosfiltfilt(sos, cleaned, axis=0)
        b, a = butter(order, band, btype="bandpass", fs=fs)
        return lfilter(b, a, cleaned, axis=0)
    except ValueError:
        return cleaned


def _lowpass_static_complex(values: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    z = np.asarray(values, dtype=np.complex128)
    if z.shape[0] < 4 or fs <= 0:
        return np.zeros_like(z)
    cutoff = min(max(float(cutoff_hz), 1e-5), fs * 0.45)
    if not (0 < cutoff < fs * 0.5):
        return np.zeros_like(z)
    try:
        b, a = butter(2, cutoff, btype="lowpass", fs=fs)
        return lfilter(b, a, z, axis=0)
    except ValueError:
        return np.zeros_like(z)


def _smooth_amplitude(values: np.ndarray, fs: float, time_constant_s: float = A121_RATE_WINDOW_S) -> np.ndarray:
    amp = np.asarray(values, dtype=float)
    if amp.ndim == 1:
        amp = amp[:, None]
    if amp.shape[0] == 0:
        return amp.copy()
    if fs <= 0 or time_constant_s <= 0:
        return amp.copy()
    sf = float(np.exp(-1.0 / max(fs * time_constant_s, 1e-9)))
    out = np.empty_like(amp)
    out[0] = amp[0]
    for idx in range(1, amp.shape[0]):
        out[idx] = sf * out[idx - 1] + (1.0 - sf) * amp[idx]
    return out


def _unwrap_phase_matrix(complex_profile: np.ndarray) -> np.ndarray:
    if complex_profile.shape[0] == 0:
        return np.empty(complex_profile.shape, dtype=float)
    angle = np.angle(complex_profile)
    unwrapped = np.zeros_like(angle, dtype=float)
    if angle.shape[0] > 1:
        diff = np.diff(angle, axis=0)
        diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        unwrapped[1:] = np.cumsum(diff, axis=0)
    return unwrapped


def _numeric_column(work: pd.DataFrame, column: str) -> np.ndarray:
    if column not in work.columns:
        return np.full(len(work), np.nan, dtype=float)
    return pd.to_numeric(work[column], errors="coerce").to_numpy(dtype=float)


def _last_index(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(idx[-1]) if len(idx) else None


def _recorded_acconeer_gate(work: pd.DataFrame, distances: np.ndarray) -> _RecordedAcconeerGate | None:
    """Return the current Acconeer reference-app selection stored in a new-format CSV/DB row.

    The latest Acconeer app state is authoritative.  If the latest rows say no presence/reference
    error, do not reuse an older ESTIMATE gate from earlier in the same recording window.
    """
    if work.empty or not any(
        col in work.columns
        for col in (
            "AcconeerAppState",
            "AcconeerTargetDistance_m",
            "AcconeerPresenceDistance_m",
            "AcconeerRangeStartIndex",
            "AcconeerRangeStart_m",
            "AcconeerBreathingRate_BPM",
        )
    ):
        return None

    app_states = work["AcconeerAppState"].fillna("").astype(str).to_numpy() if "AcconeerAppState" in work.columns else np.asarray([""] * len(work))
    latest_state = str(app_states[-1]) if len(app_states) else ""
    latest_state_upper = latest_state.upper()

    presence_detected: bool | None = None
    presence_numeric = _numeric_column(work, "AcconeerPresenceDetected")
    latest_presence_idx = _last_index(np.isfinite(presence_numeric))
    if latest_presence_idx is not None:
        presence_detected = bool(presence_numeric[latest_presence_idx] > 0)

    state_break_mask = np.asarray(
        [state.upper().startswith("NO_PRESENCE") or state.upper().startswith("REFERENCE_") for state in app_states],
        dtype=bool,
    )
    last_break_idx = _last_index(state_break_mask)
    active_mask = np.ones(len(work), dtype=bool)
    if last_break_idx is not None:
        active_mask[: last_break_idx + 1] = False

    if latest_state_upper.startswith("NO_PRESENCE") or latest_state_upper.startswith("REFERENCE_"):
        return _RecordedAcconeerGate(app_state=latest_state, presence_detected=presence_detected)
    if latest_state and latest_state_upper not in {"ESTIMATE_BREATHING_RATE", "DETERMINE_DISTANCE_ESTIMATE"}:
        return _RecordedAcconeerGate(app_state=latest_state, presence_detected=presence_detected)

    estimate_mask = (app_states == "ESTIMATE_BREATHING_RATE") & active_mask
    determine_mask = (app_states == "DETERMINE_DISTANCE_ESTIMATE") & active_mask
    provisional_mask = estimate_mask | determine_mask

    target_values = _numeric_column(work, "AcconeerTargetDistance_m")
    presence_values = _numeric_column(work, "AcconeerPresenceDistance_m")
    target_valid = np.isfinite(target_values) & (target_values > 0) & active_mask
    # Presence distance is only a provisional gate during Acconeer's distance-determination state.
    presence_valid = np.isfinite(presence_values) & (presence_values > 0) & determine_mask

    target_idx = _last_index(estimate_mask & target_valid)
    if target_idx is None:
        target_idx = _last_index(target_valid)
    if target_idx is None:
        target_idx = _last_index(presence_valid)

    target_distance_m: float | None = None
    if target_idx is not None:
        if target_valid[target_idx]:
            target_distance_m = float(target_values[target_idx])
        elif presence_valid[target_idx]:
            target_distance_m = float(presence_values[target_idx])

    start_idx_values = _numeric_column(work, "AcconeerRangeStartIndex")
    end_idx_values = _numeric_column(work, "AcconeerRangeEndIndex")
    range_valid = np.isfinite(start_idx_values) & np.isfinite(end_idx_values) & (end_idx_values > start_idx_values) & estimate_mask
    range_idx = _last_index(range_valid)

    range_start_idx: int | None = None
    range_end_idx: int | None = None
    if range_idx is not None and len(distances):
        range_start_idx = int(np.clip(int(round(start_idx_values[range_idx])), 0, max(len(distances) - 1, 0)))
        range_end_idx = int(np.clip(int(round(end_idx_values[range_idx])), range_start_idx + 1, len(distances)))
        if target_distance_m is None:
            center_idx = int(np.clip((range_start_idx + range_end_idx - 1) // 2, 0, len(distances) - 1))
            target_distance_m = float(distances[center_idx])

    start_m_values = _numeric_column(work, "AcconeerRangeStart_m")
    end_m_values = _numeric_column(work, "AcconeerRangeEnd_m")
    range_m_valid = np.isfinite(start_m_values) & np.isfinite(end_m_values) & (end_m_values >= start_m_values) & estimate_mask
    range_m_idx = _last_index(range_m_valid)

    gate_min_m: float | None = None
    gate_max_m: float | None = None
    if range_m_idx is not None:
        gate_min_m = float(start_m_values[range_m_idx])
        gate_max_m = float(end_m_values[range_m_idx])
    elif range_start_idx is not None and range_end_idx is not None and len(distances):
        gate_min_m = float(distances[range_start_idx])
        gate_max_m = float(distances[int(np.clip(range_end_idx - 1, range_start_idx, len(distances) - 1))])

    breathing_values = _numeric_column(work, "AcconeerBreathingRate_BPM")
    breathing_valid = estimate_mask & np.isfinite(breathing_values) & (breathing_values > 0)
    breathing_rate_bpm: float | None = None
    if np.any(breathing_valid):
        # Acconeer's per-frame RR can occasionally jump to a harmonic for a handful of frames
        # while the visible waveform remains stable.  Reuse the recent robust value rather than
        # the single latest sample so offline/history analysis is not dominated by a terminal blip.
        recent_valid = breathing_valid.copy()
        if "Timestamp_ms" in work.columns:
            ts = _numeric_column(work, "Timestamp_ms")
            finite_ts = np.isfinite(ts)
            if np.any(finite_ts):
                latest_ts = float(ts[np.flatnonzero(finite_ts)[-1]])
                recent_valid &= finite_ts & (ts >= latest_ts - 10_000.0)
        recent_values = breathing_values[recent_valid]
        if len(recent_values) < 5:
            recent_values = breathing_values[breathing_valid]
        breathing_rate_bpm = float(np.median(recent_values))

    if target_distance_m is None and gate_min_m is None and gate_max_m is None and breathing_rate_bpm is None and not np.any(provisional_mask):
        return None
    return _RecordedAcconeerGate(
        target_distance_m=target_distance_m,
        gate_min_m=gate_min_m,
        gate_max_m=gate_max_m,
        range_start_idx=range_start_idx,
        range_end_idx=range_end_idx,
        app_state=latest_state,
        presence_detected=presence_detected,
        breathing_rate_bpm=breathing_rate_bpm,
    )


def _select_target_index(
    distances: np.ndarray,
    latest_amplitude: np.ndarray,
    median_amplitude: np.ndarray,
    phase_energy_ratio: np.ndarray,
    target_distance_m: float | None,
    *,
    use_gating: bool,
    latest_peak_m: float,
) -> int:
    m = len(distances)
    if m == 0:
        return 0
    if use_gating and target_distance_m is not None and np.isfinite(target_distance_m):
        return int(np.argmin(np.abs(distances - float(target_distance_m))))

    search_mask = distances >= A121_MIN_TARGET_DISTANCE_M
    if not np.any(search_mask):
        search_mask = np.ones(m, dtype=bool)

    amp = np.asarray(median_amplitude if len(median_amplitude) == m else latest_amplitude, dtype=float)
    if len(amp) != m:
        amp = np.ones(m, dtype=float)
    amp_floor = float(np.median(amp[search_mask])) + 1e-12
    score = np.where(search_mask, amp / amp_floor, 0.0)
    if len(phase_energy_ratio) == m:
        # Phase-band energy is only a tie-breaker. Large unrelated motion in a weaker bin should
        # not outrank the actual high-SNR chest return.
        score += 0.15 * np.clip(phase_energy_ratio, 0.0, 5.0)
    if latest_peak_m > 0 and np.isfinite(latest_peak_m):
        # Prefer the reported peak when its range is plausible, but do not blindly lock to close leakage.
        peak_idx = int(np.argmin(np.abs(distances - latest_peak_m)))
        if search_mask[peak_idx]:
            score[peak_idx] += 0.5 * float(np.max(score))
    return int(np.argmax(score))


def _candidate_weights(
    distances: np.ndarray,
    selected_idx: int,
    amplitude_weight: np.ndarray,
    *,
    gate_half_width_m: float,
    use_gating: bool,
    max_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    m = len(distances)
    if m == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    selected_idx = int(np.clip(selected_idx, 0, m - 1))
    if use_gating:
        half = max(0.0, float(gate_half_width_m))
        candidate = np.flatnonzero(np.abs(distances - distances[selected_idx]) <= half)
    else:
        # Acconeer's breathing reference app analyzes a small segment centered on the target.
        lo = max(0, selected_idx - 1)
        hi = min(m, selected_idx + 2)
        candidate = np.arange(lo, hi, dtype=int)
    if len(candidate) == 0:
        candidate = np.asarray([selected_idx], dtype=int)
    if len(candidate) > max_bins:
        # Match Acconeer's reference behavior: analyze a compact segment centered on the
        # selected presence distance, rather than cherry-picking distant high-amplitude bins.
        order = np.argsort(np.abs(candidate - selected_idx), kind="stable")[:max_bins]
        candidate = np.sort(candidate[order])
    amp = amplitude_weight[candidate] if len(amplitude_weight) == m else np.ones(len(candidate))
    weights = np.maximum(np.asarray(amp, dtype=float), 0.0)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(candidate), dtype=float)
    weights /= float(np.sum(weights)) + 1e-12
    return candidate.astype(int), weights


def _aligned_weighted_average(signals: np.ndarray, weights: np.ndarray, ref_col: int = 0) -> np.ndarray:
    x = np.asarray(signals, dtype=float)
    if x.ndim == 1:
        return x.copy()
    if x.shape[1] == 0:
        return np.asarray([], dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(w) != x.shape[1] or float(np.sum(w)) <= 0.0:
        w = np.ones(x.shape[1], dtype=float) / x.shape[1]
    else:
        w = w / (float(np.sum(w)) + 1e-12)
    ref_col = int(np.clip(ref_col, 0, x.shape[1] - 1))
    ref = clean_signal(x[:, ref_col])
    aligned = np.empty_like(x)
    for col in range(x.shape[1]):
        sig = clean_signal(x[:, col])
        corr = float(np.dot(ref, sig)) if len(sig) == len(ref) else 0.0
        aligned[:, col] = -sig if corr < 0 else sig
    return np.sum(aligned * w[None, :], axis=1)


def _heart_range_candidate_indices(
    distances: np.ndarray,
    selected_idx: int,
    amplitude_weight: np.ndarray,
    gate_half_width_m: float,
) -> np.ndarray:
    m = len(distances)
    if m == 0:
        return np.asarray([], dtype=int)
    selected_idx = int(np.clip(selected_idx, 0, m - 1))
    selected_distance = float(distances[selected_idx])
    half_width_m = max(HEART_RANGE_HALF_WIDTH_M, float(gate_half_width_m) * 3.0)
    local = np.flatnonzero(
        (np.abs(distances - selected_distance) <= half_width_m)
        & (distances >= A121_MIN_TARGET_DISTANCE_M)
    )
    if len(local) == 0:
        return np.asarray([selected_idx], dtype=int)

    amp = np.asarray(amplitude_weight, dtype=float)
    if len(amp) != m:
        amp = np.ones(m, dtype=float)
    local_amp = np.maximum(amp[local], 0.0)
    local_peak = float(np.max(local_amp)) if len(local_amp) else 0.0
    if not np.isfinite(local_peak) or local_peak <= 0.0:
        local_peak = 1.0
    amp_norm = local_amp / (local_peak + 1e-12)
    proximity = 1.0 - np.clip(np.abs(distances[local] - selected_distance) / max(half_width_m, 1e-9), 0.0, 1.0)

    # Keep the torso return around the selected range.  Cardiac motion often appears on the
    # near edge of that return, so do not use only the maximum-amplitude bin.
    keep = (amp_norm >= 0.25) | (proximity >= 0.65)
    candidate = local[keep]
    if len(candidate) == 0:
        candidate = np.asarray([selected_idx], dtype=int)

    if len(candidate) > HEART_RANGE_MAX_BINS:
        cand_amp_norm = np.maximum(amp[candidate], 0.0) / (local_peak + 1e-12)
        cand_proximity = 1.0 - np.clip(
            np.abs(distances[candidate] - selected_distance) / max(half_width_m, 1e-9),
            0.0,
            1.0,
        )
        rank_score = 0.70 * cand_amp_norm + 0.30 * cand_proximity
        order = np.argsort(rank_score, kind="stable")[-HEART_RANGE_MAX_BINS:]
        candidate = candidate[np.sort(order)]
    return np.asarray(np.sort(candidate), dtype=int)


def _estimate_heart_from_range_bins(
    angle_unwrapped: np.ndarray,
    distances: np.ndarray,
    selected_idx: int,
    amplitude_weight: np.ndarray,
    resp_signal: np.ndarray,
    fs: float,
    heart_band: tuple[float, float],
    reject_hz: tuple[float, ...],
    gate_half_width_m: float,
    heart_window_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Find heart rate from a small range-bin cluster instead of only the peak bin.

    In real foil/chest recordings the strongest HR bin can sit several centimeters in front of
    the maximum-amplitude range.  Estimate each nearby bin independently, cluster agreeing rates,
    then recombine the best bins.  This rejects isolated false locks while recovering the real
    pulse when breathing motion shifts the peak-distance bin.
    """
    x = np.asarray(angle_unwrapped, dtype=float)
    if x.ndim != 2 or x.shape[0] < 24 or fs <= 0 or len(distances) == 0:
        return None
    m = min(x.shape[1], len(distances))
    if m <= 0:
        return None
    distances = distances[:m]
    x = x[:, :m]
    candidate_idx = _heart_range_candidate_indices(distances, selected_idx, amplitude_weight, gate_half_width_m)
    candidate_idx = candidate_idx[(0 <= candidate_idx) & (candidate_idx < m)]
    if len(candidate_idx) == 0:
        return None

    amp = np.asarray(amplitude_weight, dtype=float)
    if len(amp) != m:
        amp = np.ones(m, dtype=float)
    local_peak = float(np.max(np.maximum(amp[candidate_idx], 0.0))) if len(candidate_idx) else 1.0
    if not np.isfinite(local_peak) or local_peak <= 0.0:
        local_peak = 1.0
    selected_distance = float(distances[int(np.clip(selected_idx, 0, m - 1))])
    half_width_m = max(HEART_RANGE_HALF_WIDTH_M, float(gate_half_width_m) * 3.0)

    bin_results: list[tuple[int, float, float, float]] = []
    for idx in candidate_idx:
        heart_source = _remove_resp_correlated_motion(x[:, int(idx)], resp_signal, fs)
        heart_signal = _bandpass_matrix(heart_source, fs, HEART_BAND_HZ, order=3, zero_phase=True).ravel()
        rate_signal = _heart_rate_signal_window(heart_signal, resp_signal, fs, heart_window_s)
        hz, confidence = estimate_band_peak_fused(rate_signal, fs, heart_band, reject_hz=reject_hz)
        if hz <= 0.0 or confidence < HEART_BIN_MIN_CONFIDENCE:
            continue
        amp_norm = float(np.clip(max(float(amp[int(idx)]), 0.0) / (local_peak + 1e-12), 0.0, 1.0))
        proximity = float(
            1.0
            - np.clip(abs(float(distances[int(idx)]) - selected_distance) / max(half_width_m, 1e-9), 0.0, 1.0)
        )
        score = float(confidence * (0.55 + 0.35 * amp_norm + 0.10 * proximity))
        bin_results.append((int(idx), float(hz), float(confidence), score))

    if not bin_results:
        return None

    best_members: list[tuple[int, float, float, float]] = []
    best_score = -1.0
    for result in bin_results:
        hz = result[1]
        members = [other for other in bin_results if abs(other[1] - hz) <= HEART_BIN_CLUSTER_TOLERANCE_HZ]
        cluster_score = float(sum(other[3] for other in members))
        if len(members) == 1:
            cluster_score *= 0.45
        else:
            cluster_score *= 1.0 + 0.08 * min(len(members) - 1, 4)
        if cluster_score > best_score:
            best_score = cluster_score
            best_members = members

    if not best_members:
        return None
    top_members = sorted(best_members, key=lambda item: item[3], reverse=True)[:3]
    top_members = sorted(top_members, key=lambda item: item[0])
    top_idx = np.asarray([item[0] for item in top_members], dtype=int)
    weights = np.asarray([max(item[3], 1e-6) for item in top_members], dtype=float)
    weights /= float(np.sum(weights)) + 1e-12
    ref_col = int(np.argmax(weights))

    heart_source = _aligned_weighted_average(x[:, top_idx], weights, ref_col)
    heart_source = _remove_resp_correlated_motion(heart_source, resp_signal, fs)
    heart_signal = _bandpass_matrix(heart_source, fs, HEART_BAND_HZ, order=3, zero_phase=True).ravel()
    rate_signal = _heart_rate_signal_window(heart_signal, resp_signal, fs, heart_window_s)
    heart_hz, heart_confidence = estimate_band_peak_fused(rate_signal, fs, heart_band, reject_hz=reject_hz)

    member_weight_sum = float(np.sum([item[3] for item in top_members])) + 1e-12
    clustered_hz = float(sum(item[1] * item[3] for item in top_members) / member_weight_sum)
    clustered_confidence = float(max(item[2] for item in top_members))
    if heart_hz <= 0.0 or abs(heart_hz - clustered_hz) > HEART_BIN_CLUSTER_TOLERANCE_HZ * 1.35:
        heart_hz = clustered_hz
        heart_confidence = max(heart_confidence, clustered_confidence * 0.85)
    else:
        heart_confidence = max(heart_confidence, clustered_confidence * 0.75)
    return heart_source, heart_signal, float(heart_hz), float(min(heart_confidence, 100.0))


def _weighted_fft_peak(
    signals: np.ndarray,
    weights: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    *,
    reject_hz: tuple[float, ...] = (),
    min_duration_s: float = A121_RATE_WINDOW_S,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    x = np.asarray(signals, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < 16 or fs <= 0:
        return 0.0, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float)
    duration_s = x.shape[0] / fs
    if duration_s < min_duration_s:
        return 0.0, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float)
    n = min(x.shape[0], int(round(fs * A121_RATE_WINDOW_S)))
    if n < 16:
        return 0.0, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float)
    source = np.apply_along_axis(clean_signal, 0, x[-n:])
    windowed = source * np.hamming(n)[:, None]
    padded_len = 2 ** (int(np.log2(n)) + 1)
    psd = np.abs(np.fft.rfft(windowed, axis=0, n=padded_len))
    w = np.asarray(weights, dtype=float)
    if len(w) != psd.shape[1] or float(np.sum(w)) <= 0.0:
        w = np.ones(psd.shape[1], dtype=float) / psd.shape[1]
    else:
        w = w / (float(np.sum(w)) + 1e-12)
    weighted = np.sum(psd * w[None, :], axis=1)
    freqs = np.fft.rfftfreq(padded_len, d=1.0 / fs)
    low, high = band_hz
    high = min(float(high), fs * 0.46)
    low = max(float(low), 1.0 / max(duration_s, 1e-9))
    valid = (freqs >= low) & (freqs <= high)
    if not np.any(valid):
        return 0.0, 0.0, freqs, weighted
    f = freqs[valid]
    p = weighted[valid]
    score = p.copy()
    for rejected in reject_hz:
        if rejected <= 0:
            continue
        width = max(0.035, rejected * 0.025)
        score *= 1.0 - 0.92 * np.exp(-0.5 * ((f - rejected) / width) ** 2)
    # Edge-bin suppression: taper score near band boundaries to avoid false
    # peak locks at the first/last FFT bins (filter roll-off energy).
    freq_span = float(f[-1] - f[0]) if len(f) > 1 else 1.0
    bin_width = freq_span / max(len(f) - 1, 1)
    edge_bins = max(3, int(0.05 / max(bin_width, 1e-9)))
    if len(f) > 2 * edge_bins:
        taper = np.ones(len(f), dtype=float)
        for i in range(edge_bins):
            w_t = float(i + 1) / float(edge_bins + 1)
            taper[i] *= w_t
            taper[-(i + 1)] *= w_t
        score *= taper
    local_idx = int(np.argmax(score))
    peak_loc = int(np.flatnonzero(valid)[local_idx])
    peak_hz = float(freqs[peak_loc])
    if 0 < peak_loc < len(freqs) - 1:
        try:
            peak_hz = float(np.clip(_peak_interpolation(weighted[peak_loc - 1 : peak_loc + 2], freqs[peak_loc - 1 : peak_loc + 2]), low, high))
        except Exception:
            pass
    noise = float(np.median(p)) + 1e-18
    confidence = float(p[local_idx] / noise)
    return peak_hz, confidence, freqs, weighted


def _acconeer_breathing_estimate(
    complex_profile: np.ndarray,
    fs: float,
    candidate_idx: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Run Acconeer's own A121 BreathingProcessor on a selected range segment.

    The Exploration Tool processor first averages sweeps inside each frame. Our CSV/live rows
    already store that mean complex sweep, so each row is supplied as a one-sweep frame. The rate
    estimate, static-clutter IIR, phase unwrap, band-pass, Hamming window, FFT padding, and
    quadratic peak interpolation are therefore Acconeer's implementation rather than our previous
    reimplementation.
    """
    z = np.asarray(complex_profile, dtype=np.complex128)
    indices = np.asarray(candidate_idx, dtype=int)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0 or len(indices) == 0 or fs <= 0:
        return 0.0, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    indices = indices[(0 <= indices) & (indices < z.shape[1])]
    if len(indices) == 0:
        return 0.0, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    segment = z[:, indices]
    try:
        from acconeer.exptool import a121
        from acconeer.exptool.a121.algo.breathing._processor import (
            BreathingProcessor,
            BreathingProcessorConfig,
        )

        processor = BreathingProcessor(
            sensor_config=a121.SensorConfig(num_points=int(segment.shape[1]), frame_rate=float(fs)),
            processor_config=BreathingProcessorConfig(
                lowest_breathing_rate=float(A121_RESP_BAND_HZ[0] * 60.0),
                highest_breathing_rate=float(A121_RESP_BAND_HZ[1] * 60.0),
                time_series_length_s=float(A121_RATE_WINDOW_S),
            ),
        )
        result = None
        for mean_sweep in segment:
            result = processor.process(SimpleNamespace(frame=mean_sweep[np.newaxis, :]))
        if result is None:
            return 0.0, 0.0, np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

        extra = result.extra_result
        signal = np.asarray(extra.breathing_motion, dtype=float)
        if len(signal):
            signal = signal[-min(len(signal), segment.shape[0]) :]
        freqs = np.asarray(extra.frequencies, dtype=float)
        psd = np.asarray(extra.psd, dtype=float)
        rate_bpm = result.breathing_rate
        if rate_bpm is None or not np.isfinite(float(rate_bpm)):
            return 0.0, 0.0, signal, freqs, psd

        rate_hz = float(rate_bpm) / 60.0
        # Acconeer's quadratic interpolation can land just outside the configured FFT band.
        in_range = A121_RESP_BAND_HZ[0] * 0.90 <= rate_hz <= min(A121_RESP_BAND_HZ[1] * 1.05, fs * 0.46)
        valid = (freqs >= A121_RESP_BAND_HZ[0]) & (freqs <= min(A121_RESP_BAND_HZ[1], fs * 0.46))
        if in_range and np.any(valid) and len(psd) == len(freqs):
            peak_idx = int(np.argmin(np.abs(freqs - rate_hz)))
            confidence = float(psd[peak_idx] / (np.median(psd[valid]) + 1e-18))
        else:
            confidence = 0.0
        return rate_hz, confidence, signal, freqs, psd
    except Exception:
        # Keep analysis functional if a future acconeer-exptool package moves the private class.
        phase = _unwrap_phase_matrix(segment)
        resp_matrix = _bandpass_matrix(phase, fs, A121_RESP_BAND_HZ, order=2, zero_phase=False)
        weights = np.ones(resp_matrix.shape[1], dtype=float) / max(resp_matrix.shape[1], 1)
        resp_signal = _aligned_weighted_average(resp_matrix, weights, 0)
        resp_hz, confidence, freqs, psd = _weighted_fft_peak(
            resp_matrix,
            weights,
            fs,
            A121_RESP_BAND_HZ,
            min_duration_s=A121_RATE_WINDOW_S,
        )
        return resp_hz, confidence, resp_signal, freqs, psd


def _heart_rate_signal_window(
    heart_signal: np.ndarray,
    resp_signal: np.ndarray,
    fs: float,
    heart_window_s: float | None,
) -> np.ndarray:
    """Return the signal segment used for heart-rate estimation."""
    signal = np.asarray(heart_signal, dtype=float)
    if heart_window_s is not None and np.isfinite(float(heart_window_s)) and heart_window_s > 0:
        samples = int(round(max(float(heart_window_s), 0.0) * max(fs, 1e-9)))
        if samples > 0 and len(signal) > samples:
            return signal[-samples:]
        return signal
    if len(resp_signal) >= int(round(fs * 10.0)) and len(resp_signal) < len(signal):
        # The respiration processor keeps a compact rolling window.  Estimate heart rate on the
        # same recent window by default so older uncorrected breathing motion cannot dominate.
        return signal[-len(resp_signal) :]
    return signal


def _near_resp_harmonic(frequency_hz: float, resp_hz: float, max_order: int = 7) -> bool:
    if frequency_hz <= 0 or resp_hz <= 0:
        return False
    for order in range(2, max_order + 1):
        harmonic = resp_hz * order
        if HEART_BAND_HZ[0] * 0.85 <= harmonic <= HEART_BAND_HZ[1] * 1.05:
            if abs(frequency_hz - harmonic) <= max(0.030, harmonic * 0.020):
                return True
    return False


def _empty_analysis(df: pd.DataFrame | None = None) -> A121VitalAnalysis:
    latest_peak = float(df["PeakDistance_m"].iloc[-1]) if df is not None and len(df) and "PeakDistance_m" in df else 0.0
    latest_amp = float(df["PeakAmplitude"].iloc[-1]) if df is not None and len(df) and "PeakAmplitude" in df else 0.0
    mean_amp = float(df["MeanAmplitude"].iloc[-1]) if df is not None and len(df) and "MeanAmplitude" in df else 0.0
    return A121VitalAnalysis(
        sample_rate_hz=0.0,
        target_distance_m=latest_peak,
        gate_min_m=max(0.0, latest_peak - DEFAULT_GATE_HALF_WIDTH_M),
        gate_max_m=latest_peak + DEFAULT_GATE_HALF_WIDTH_M,
        peak_distance_m=latest_peak,
        peak_amplitude=latest_amp,
        mean_amplitude=mean_amp,
        presence_score=0.0,
        present=False,
        resp_bpm=0.0,
        heart_bpm=0.0,
        resp_hz=0.0,
        heart_hz=0.0,
        times_s=np.asarray([], dtype=float),
        raw_phase=np.asarray([], dtype=float),
        resp_signal=np.asarray([], dtype=float),
        heart_signal=np.asarray([], dtype=float),
        raw_i=np.asarray([], dtype=float),
        raw_q=np.asarray([], dtype=float),
        distances_m=np.asarray([], dtype=float),
        latest_amplitude=np.asarray([], dtype=float),
        selected_index=0,
    )


def _peak_interpolation(y: np.ndarray, x: np.ndarray) -> float:
    a = (x[0] * (y[2] - y[1]) + x[1] * (y[0] - y[2]) + x[2] * (y[1] - y[0])) / (
        (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2]) + 1e-12
    )
    b = (y[1] - y[0]) / (x[1] - x[0] + 1e-12) - a * (x[0] + x[1])
    if abs(a) > 1e-12:
        peak_loc = -b / (2 * a)
        return float(peak_loc)
    return float(x[1])


def analyze_a121_vitals(
    df: pd.DataFrame,
    *,
    auto_gate: bool = True,
    gate_half_width_m: float = DEFAULT_GATE_HALF_WIDTH_M,
    max_frames: int = 900,
    target_distance_m: float | None = None,
    heart_prior_hz: float | None = None,
    heart_prior_std_hz: float | None = None,
    use_gating: bool = True,
    heart_window_s: float | None = None,
) -> A121VitalAnalysis:
    """Estimate A121 presence, target range, respiration, and cardiac activity from Sparse IQ rows.

    This stateless batch/window analyzer estimates target, rates, and confidence. The GUI uses
    A121LiveTraceProcessor for append-only live waveforms so historical samples are not redrawn by
    this sliding-window rate analysis.
    """
    if df.empty or "Timestamp_ms" not in df:
        return _empty_analysis(df)

    sort_col = "Frame" if "Frame" in df.columns else "Timestamp_ms"
    work = df.sort_values(sort_col, kind="mergesort").tail(max_frames).reset_index(drop=True)
    times_ms = work["Timestamp_ms"].to_numpy(dtype=float)
    fs = sample_rate_from_ms(times_ms, default=20.0)
    times_s = _times_seconds_from_ms(times_ms, fs)

    latest_peak = float(work["PeakDistance_m"].iloc[-1]) if "PeakDistance_m" in work else 0.0
    latest_amp = float(work["PeakAmplitude"].iloc[-1]) if "PeakAmplitude" in work else 0.0
    mean_amp = float(work["MeanAmplitude"].iloc[-1]) if "MeanAmplitude" in work else 0.0

    distances = parse_json_array(work["Distances_m"].iloc[-1]) if "Distances_m" in work else np.asarray([], dtype=float)
    latest_amplitude = parse_json_array(work["Amplitude"].iloc[-1]) if "Amplitude" in work else np.asarray([], dtype=float)
    recorded_gate = _recorded_acconeer_gate(work, distances) if bool(use_gating) else None
    explicit_target_distance_m = target_distance_m
    if target_distance_m is None and recorded_gate is not None and recorded_gate.target_distance_m is not None:
        target_distance_m = recorded_gate.target_distance_m

    if {"Real", "Imag"}.issubset(work.columns):
        real = _json_matrix(work["Real"])
        imag = _json_matrix(work["Imag"])
        n = min(len(work), real.shape[0], imag.shape[0])
        m = min(real.shape[1] if real.ndim == 2 else 0, imag.shape[1] if imag.ndim == 2 else 0, len(distances) or 10**9)
        if n > 0 and m > 0:
            real = real[-n:, :m]
            imag = imag[-n:, :m]
            distances = distances[:m] if len(distances) >= m else np.arange(m, dtype=float)
            latest_amplitude = latest_amplitude[:m] if len(latest_amplitude) >= m else np.hypot(real[-1], imag[-1])
            complex_profile = real + 1j * imag

            # Use clutter removal for target/weight amplitudes. Respiration rate itself is
            # estimated below by Acconeer's BreathingProcessor; this centered differential phase
            # path remains for target scoring, plotting, and the experimental heart candidate.
            static_cutoff = _valid_band_for_fs(A121_RESP_BAND_HZ, fs)
            zm_profile = complex_profile - _lowpass_static_complex(
                complex_profile,
                fs,
                static_cutoff[0] if static_cutoff is not None else A121_RESP_BAND_HZ[0],
            )
            zm_amplitude = np.abs(zm_profile)
            lp_filt_ampl = _smooth_amplitude(zm_amplitude, fs)
            median_amp = np.median(zm_amplitude, axis=0)
            centered_profile, _, _ = _center_complex_profile(real, imag)
            angle_unwrapped = _differential_phase_matrix(centered_profile)
            phase_energy_ratio = _band_energy_ratio(angle_unwrapped, fs)

            half_width = max(0.01, float(gate_half_width_m))
            use_recorded_gate = False
            if recorded_gate is not None and recorded_gate.target_distance_m is not None and bool(use_gating):
                if explicit_target_distance_m is None:
                    use_recorded_gate = True
                else:
                    distance_step = float(np.median(np.diff(distances))) if len(distances) > 1 else half_width
                    use_recorded_gate = abs(float(explicit_target_distance_m) - recorded_gate.target_distance_m) <= max(half_width * 2.0, abs(distance_step) * 3.0)
                if use_recorded_gate and recorded_gate.gate_min_m is not None and recorded_gate.gate_max_m is not None:
                    half_width = max(
                        0.0025,
                        abs(recorded_gate.target_distance_m - recorded_gate.gate_min_m),
                        abs(recorded_gate.gate_max_m - recorded_gate.target_distance_m),
                    )

            selected_idx = _select_target_index(
                distances,
                latest_amplitude,
                median_amp,
                phase_energy_ratio,
                target_distance_m,
                use_gating=bool(use_gating),
                latest_peak_m=latest_peak,
            )
            selected_idx = int(np.clip(selected_idx, 0, m - 1))
            target_distance = float(distances[selected_idx]) if len(distances) else latest_peak
            if use_recorded_gate and recorded_gate is not None and recorded_gate.gate_min_m is not None and recorded_gate.gate_max_m is not None:
                gate_min = float(recorded_gate.gate_min_m)
                gate_max = float(recorded_gate.gate_max_m)
            else:
                gate_min = target_distance - half_width
                gate_max = target_distance + half_width

            amplitude_weight = lp_filt_ampl[-1] if len(lp_filt_ampl) else median_amp
            if (
                use_recorded_gate
                and recorded_gate is not None
                and recorded_gate.range_start_idx is not None
                and recorded_gate.range_end_idx is not None
            ):
                start_idx = int(np.clip(recorded_gate.range_start_idx, 0, m - 1))
                end_idx = int(np.clip(recorded_gate.range_end_idx, start_idx + 1, m))
                candidate_idx = np.arange(start_idx, end_idx, dtype=int)
                if len(candidate_idx) == 0:
                    candidate_idx = np.asarray([selected_idx], dtype=int)
                amp = amplitude_weight[candidate_idx] if len(amplitude_weight) == m else np.ones(len(candidate_idx))
                weights = np.maximum(np.asarray(amp, dtype=float), 0.0)
                if float(np.sum(weights)) <= 0.0:
                    weights = np.ones(len(candidate_idx), dtype=float)
                weights /= float(np.sum(weights)) + 1e-12
            else:
                candidate_idx, weights = _candidate_weights(
                    distances,
                    selected_idx,
                    amplitude_weight,
                    gate_half_width_m=half_width,
                    use_gating=bool(use_gating),
                    max_bins=3,
                )
            candidate_bins = int(len(candidate_idx))
            ref_matches = np.flatnonzero(candidate_idx == selected_idx)
            ref_col = int(ref_matches[0]) if len(ref_matches) else int(np.argmax(weights))

            raw_phase = angle_unwrapped[:, selected_idx]
            raw_i = np.real(zm_profile[:, selected_idx])
            raw_q = np.imag(zm_profile[:, selected_idx])

            resp_hz_candidate, resp_confidence, resp_signal, _, _ = _acconeer_breathing_estimate(
                complex_profile,
                fs,
                candidate_idx,
            )
            resp_hz = resp_hz_candidate
            duration_s = len(raw_phase) / max(fs, 1e-9)
            recorded_resp_hz = 0.0
            if recorded_gate is not None and recorded_gate.breathing_rate_bpm is not None:
                recorded_resp_hz = float(recorded_gate.breathing_rate_bpm) / 60.0
                if A121_RESP_BAND_HZ[0] * 0.90 <= recorded_resp_hz <= A121_RESP_BAND_HZ[1] * 1.05:
                    # This is Acconeer's own live BreathingProcessor result persisted with the
                    # frame. Prefer it over re-estimating RR from the averaged CSV IQ profile.
                    resp_hz = recorded_resp_hz
                    resp_confidence = max(float(resp_confidence), 20.0)
                else:
                    recorded_resp_hz = 0.0
            # Harmonic rejection should trust only Acconeer's own rate confidence.  The fallback
            # display rate below can be useful for short slow-breath recordings, but should not
            # notch heart candidates near a questionable respiration harmonic.
            resp_confidence_for_harmonics = float(resp_confidence)

            # No-breath detection: if the RMS of the respiration signal is very
            # low, the person is likely not breathing normally (breath-hold, away
            # from sensor, etc.).  Zero out the rate so the display doesn't show
            # a random frequency from noise.
            if recorded_resp_hz <= 0.0 and len(resp_signal) > 0:
                resp_rms = float(np.sqrt(np.mean(np.square(resp_signal))))
                # Threshold: below this RMS the "breathing" is indistinguishable
                # from sensor noise.  Empirically ~0.005 rad works for A121.
                if resp_rms < 0.005:
                    resp_hz = 0.0
                    resp_confidence = 0.0
                elif duration_s >= A121_RATE_WINDOW_S and resp_confidence < RESP_RATE_CONFIDENCE_MIN:
                    # In short real recordings with slow breathing, Acconeer's private
                    # BreathingProcessor can return a low-confidence rate even though the motion
                    # trace has a dominant low-frequency peak.  Use this only when the respiration
                    # waveform has substantial RMS, so static clutter stays suppressed.
                    resp_candidates = _spectral_peak_candidates(
                        resp_signal,
                        fs,
                        A121_RESP_BAND_HZ,
                        (),
                        max_candidates=3,
                    )
                    if resp_candidates:
                        fallback_hz, fallback_raw_conf, fallback_scored_conf = resp_candidates[0]
                        acf_hz, acf_conf = estimate_autocorr_peak(resp_signal, fs, A121_RESP_BAND_HZ)
                        acf_agrees = acf_hz > 0 and acf_conf >= 0.18 and abs(acf_hz - fallback_hz) <= 0.08
                        if fallback_raw_conf >= 8.0 and (acf_agrees or fallback_raw_conf >= 20.0):
                            resp_hz = float(fallback_hz)
                            resp_confidence = float(max(resp_confidence, min(fallback_scored_conf, 100.0)))

            resp_harmonic_source_hz = recorded_resp_hz if recorded_resp_hz > 0.0 else resp_hz_candidate
            resp_hz_for_harmonics = (
                resp_harmonic_source_hz
                if (recorded_resp_hz > 0.0 or duration_s >= A121_RATE_WINDOW_S)
                and resp_confidence_for_harmonics >= RESP_RATE_CONFIDENCE_MIN
                and A121_RESP_BAND_HZ[0] * 0.90 <= resp_harmonic_source_hz <= min(A121_RESP_BAND_HZ[1] * 1.05, fs * 0.46)
                else 0.0
            )

            harmonic_rejects: tuple[float, ...]
            if resp_hz_for_harmonics > 0:
                harmonic_rejects = tuple(
                    resp_hz_for_harmonics * order
                    for order in range(2, 8)
                    if HEART_BAND_HZ[0] * 0.85 <= resp_hz_for_harmonics * order <= HEART_BAND_HZ[1] * 1.05
                )
            else:
                harmonic_rejects = ()

            heart_source = _aligned_weighted_average(angle_unwrapped[:, candidate_idx], weights, ref_col)
            # Breathing can dominate the heart band through nonlinear harmonics.  Remove only the
            # part of the heart source that is coherent with the measured respiration waveform,
            # then let the FFT/peak-spacing validator decide which cardiac candidate to trust.
            heart_source = _remove_resp_correlated_motion(heart_source, resp_signal, fs)
            heart_signal = _bandpass_matrix(heart_source, fs, HEART_BAND_HZ, order=3, zero_phase=True).ravel()
            heart_rate_signal = _heart_rate_signal_window(heart_signal, resp_signal, fs, heart_window_s)
            heart_band = _heart_search_band(heart_prior_hz, heart_prior_std_hz)
            heart_hz, heart_confidence = estimate_band_peak_fused(
                heart_rate_signal,
                fs,
                heart_band,
                reject_hz=harmonic_rejects,
            )
            range_heart = _estimate_heart_from_range_bins(
                angle_unwrapped,
                distances,
                selected_idx,
                median_amp,
                resp_signal,
                fs,
                heart_band,
                harmonic_rejects,
                half_width,
                heart_window_s=heart_window_s,
            )
            if range_heart is not None:
                range_source, range_signal, range_hz, range_confidence = range_heart
                if range_confidence >= max(
                    heart_confidence * 1.10,
                    HEART_RATE_CONFIDENCE_MIN,
                    HEART_RANGE_OVERRIDE_MIN_CONFIDENCE,
                ):
                    heart_source = range_source
                    heart_signal = range_signal
                    heart_rate_signal = _heart_rate_signal_window(heart_signal, resp_signal, fs, heart_window_s)
                    heart_hz = range_hz
                    heart_confidence = range_confidence
            # NOTE: _near_resp_harmonic post-check was removed. It zeroed out valid heart
            # rate estimates when they happened to be near a respiratory harmonic, which is
            # extremely common. The narrowed FFT notch above provides sufficient suppression
            # for true harmonic artifacts without blanking the real heart rate.
            if len(heart_rate_signal) / max(fs, 1e-9) < 10.0:
                heart_hz = 0.0
                heart_confidence = 0.0

            search_mask = distances >= A121_MIN_TARGET_DISTANCE_M if len(distances) else np.asarray([], dtype=bool)
            if not np.any(search_mask):
                search_mask = np.ones_like(distances, dtype=bool) if len(distances) else np.asarray([], dtype=bool)
            amp_floor = float(np.median(median_amp[search_mask])) + 1e-12 if np.any(search_mask) else float(np.median(median_amp)) + 1e-12
            amp_ratio = float(median_amp[selected_idx] / amp_floor) if amp_floor > 0 else 0.0
            phase_ratio = float(phase_energy_ratio[selected_idx]) if len(phase_energy_ratio) == m else 1.0
            presence_score = float(
                np.clip(
                    (amp_ratio - 1.0) * 35.0
                    + min(candidate_bins, 5) * 6.0
                    + min(max(phase_ratio, 0.0), 5.0) * 10.0,
                    0.0,
                    100.0,
                )
            )
            present = bool(presence_score >= 20.0 and (amp_ratio >= 1.08 or phase_ratio >= 1.30))
            if recorded_gate is not None:
                state_upper = recorded_gate.app_state.upper()
                if recorded_gate.presence_detected is True or state_upper == "ESTIMATE_BREATHING_RATE":
                    presence_score = max(presence_score, 60.0)
                    present = True
                elif recorded_gate.presence_detected is False and state_upper.startswith("NO_PRESENCE"):
                    presence_score = 0.0
                    present = False
                if state_upper and state_upper != "ESTIMATE_BREATHING_RATE":
                    resp_hz = 0.0
                    heart_hz = 0.0
                    resp_confidence = 0.0
                    heart_confidence = 0.0

            if recorded_resp_hz <= 0.0 and (duration_s < A121_RATE_WINDOW_S or resp_confidence < RESP_RATE_CONFIDENCE_MIN):
                resp_hz = 0.0
            if heart_confidence < HEART_RATE_CONFIDENCE_MIN or (
                recorded_resp_hz <= 0.0
                and resp_confidence < RESP_RATE_CONFIDENCE_MIN
                and heart_confidence < HEART_RATE_STRONG_LOCK_CONFIDENCE
            ):
                heart_hz = 0.0
                heart_confidence = 0.0
            if not present:
                resp_hz = 0.0
                heart_hz = 0.0
                resp_confidence = 0.0
                heart_confidence = 0.0

            signal_quality = float(
                np.clip(
                    0.18 * np.log1p(max(amp_ratio - 1.0, 0.0))
                    + 0.22 * min(resp_confidence / 5.0, 1.0)
                    + 0.22 * min(heart_confidence / 5.0, 1.0)
                    + 0.20 * min(max(phase_ratio, 0.0) / 4.0, 1.0)
                    + (0.18 if present else 0.0),
                    0.0,
                    1.0,
                )
            )

            plot_len = min(len(times_s), len(resp_signal), len(heart_signal))
            if plot_len > 0:
                plot_times_s = times_s[-plot_len:]
                resp_signal = resp_signal[-plot_len:]
                heart_signal = heart_signal[-plot_len:]
            else:
                plot_times_s = np.asarray([], dtype=float)
                resp_signal = np.asarray([], dtype=float)
                heart_signal = np.asarray([], dtype=float)

            return A121VitalAnalysis(
                sample_rate_hz=fs,
                target_distance_m=target_distance,
                gate_min_m=gate_min,
                gate_max_m=gate_max,
                peak_distance_m=latest_peak,
                peak_amplitude=latest_amp,
                mean_amplitude=mean_amp,
                presence_score=presence_score,
                present=present,
                resp_bpm=float(resp_hz * 60.0),
                heart_bpm=float(heart_hz * 60.0),
                resp_hz=float(resp_hz),
                heart_hz=float(heart_hz),
                times_s=plot_times_s,
                raw_phase=raw_phase,
                resp_signal=resp_signal,
                heart_signal=heart_signal,
                raw_i=raw_i,
                raw_q=raw_q,
                distances_m=distances,
                latest_amplitude=lp_filt_ampl[-1] if len(lp_filt_ampl) else latest_amplitude,
                selected_index=selected_idx,
                resp_confidence=float(resp_confidence),
                heart_confidence=float(heart_confidence),
                signal_quality=signal_quality,
                candidate_bins=candidate_bins,
            )

    if "PeakPhase_rad" not in work:
        return _empty_analysis(work)

    # Backup logic if no IQ data. Keep it conservative: do not report rates until there is
    # enough history to resolve the vital bands, and never create invalid filters at low FPS.
    raw_phase = np.unwrap(work["PeakPhase_rad"].to_numpy(dtype=float))
    raw_phase = clean_signal(raw_phase)
    resp_signal = _bandpass_matrix(raw_phase, fs, A121_RESP_BAND_HZ, order=2, zero_phase=False).ravel()
    resp_hz, resp_confidence, _, _ = _weighted_fft_peak(resp_signal, np.asarray([1.0]), fs, A121_RESP_BAND_HZ, min_duration_s=A121_RATE_WINDOW_S)
    heart_source = _subtract_resp_harmonics(raw_phase, fs, resp_hz, max_order=6)
    heart_signal = _bandpass_matrix(heart_source, fs, HEART_BAND_HZ, order=3, zero_phase=False).ravel()
    heart_rate_signal = _heart_rate_signal_window(heart_signal, resp_signal, fs, heart_window_s)
    heart_hz, heart_confidence = estimate_band_peak_fused(
        heart_rate_signal,
        fs,
        _heart_search_band(heart_prior_hz, heart_prior_std_hz),
        reject_hz=tuple(resp_hz * order for order in range(2, 7)) if resp_hz > 0 else (),
    )
    duration_s = len(raw_phase) / max(fs, 1e-9)
    recorded_resp_hz = 0.0
    if recorded_gate is not None and recorded_gate.breathing_rate_bpm is not None:
        recorded_resp_hz = float(recorded_gate.breathing_rate_bpm) / 60.0
        if A121_RESP_BAND_HZ[0] * 0.90 <= recorded_resp_hz <= A121_RESP_BAND_HZ[1] * 1.05:
            resp_hz = recorded_resp_hz
            resp_confidence = max(float(resp_confidence), 20.0)
        else:
            recorded_resp_hz = 0.0
    if recorded_resp_hz <= 0.0 and (duration_s < A121_RATE_WINDOW_S or resp_confidence < RESP_RATE_CONFIDENCE_MIN):
        resp_hz = 0.0
    if duration_s < 10.0 or heart_confidence < HEART_RATE_CONFIDENCE_MIN:
        heart_hz = 0.0

    target_distance = float(target_distance_m) if target_distance_m is not None else float(np.median(work["PeakDistance_m"].to_numpy(dtype=float))) if "PeakDistance_m" in work else latest_peak
    gate_min = target_distance - gate_half_width_m
    gate_max = target_distance + gate_half_width_m
    if recorded_gate is not None:
        if recorded_gate.target_distance_m is not None:
            target_distance = float(recorded_gate.target_distance_m)
        if recorded_gate.gate_min_m is not None and recorded_gate.gate_max_m is not None:
            gate_min = float(recorded_gate.gate_min_m)
            gate_max = float(recorded_gate.gate_max_m)
        else:
            gate_min = target_distance - gate_half_width_m
            gate_max = target_distance + gate_half_width_m
    amp = work["PeakAmplitude"].to_numpy(dtype=float) if "PeakAmplitude" in work else np.asarray([latest_amp])
    amp_ratio = float(np.median(amp) / (np.median(np.abs(amp - np.median(amp))) * 1.4826 + np.median(amp) + 1e-9))
    presence_score = float(np.clip(amp_ratio * 40.0, 0.0, 100.0))

    present = presence_score >= 20.0
    if recorded_gate is not None:
        state_upper = recorded_gate.app_state.upper()
        if recorded_gate.presence_detected is True or state_upper == "ESTIMATE_BREATHING_RATE":
            presence_score = max(presence_score, 60.0)
            present = True
        elif recorded_gate.presence_detected is False and state_upper.startswith("NO_PRESENCE"):
            presence_score = 0.0
            present = False
        if state_upper and state_upper != "ESTIMATE_BREATHING_RATE":
            resp_hz = 0.0
            heart_hz = 0.0
            resp_confidence = 0.0
            heart_confidence = 0.0
    if not present:
        resp_hz = 0.0
        heart_hz = 0.0
        resp_confidence = 0.0
        heart_confidence = 0.0
    signal_quality = float(np.clip(0.25 * min(resp_confidence / 5.0, 1.0) + 0.25 * min(heart_confidence / 5.0, 1.0) + (0.25 if present else 0.0), 0.0, 1.0))

    return A121VitalAnalysis(
        sample_rate_hz=fs,
        target_distance_m=target_distance,
        gate_min_m=gate_min,
        gate_max_m=gate_max,
        peak_distance_m=latest_peak,
        peak_amplitude=latest_amp,
        mean_amplitude=mean_amp,
        presence_score=presence_score,
        present=present,
        resp_bpm=float(resp_hz * 60.0),
        heart_bpm=float(heart_hz * 60.0),
        resp_hz=float(resp_hz),
        heart_hz=float(heart_hz),
        times_s=times_s,
        raw_phase=raw_phase,
        resp_signal=resp_signal,
        heart_signal=heart_signal,
        raw_i=np.asarray([], dtype=float),
        raw_q=np.asarray([], dtype=float),
        distances_m=distances,
        latest_amplitude=latest_amplitude,
        selected_index=0,
        resp_confidence=float(resp_confidence),
        heart_confidence=float(heart_confidence),
        signal_quality=signal_quality,
        candidate_bins=1,
    )
