from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt, welch, lfilter

from .a121 import parse_json_array

RESP_BAND_HZ = (0.10, 0.70)
HEART_BAND_HZ = (0.65, 3.00)
DEFAULT_GATE_HALF_WIDTH_M = 0.05
A121_RATE_WINDOW_S = 20.0
A121_MIN_TARGET_DISTANCE_M = 0.28
RESP_RATE_CONFIDENCE_MIN = 4.0
HEART_RATE_CONFIDENCE_MIN = 8.0


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

        static_band = _valid_band_for_fs((RESP_BAND_HZ[0], RESP_BAND_HZ[0] * 1.05), self.fs, min_width=1e-5)
        static_cutoff = RESP_BAND_HZ[0] if static_band is None else static_band[0]
        self.b_static, self.a_static = butter(2, static_cutoff, btype="lowpass", fs=self.fs)
        self.static_x = np.zeros((len(self.b_static), self.m), dtype=np.complex128)
        self.static_y = np.zeros((len(self.a_static) - 1, self.m), dtype=np.complex128)

        resp_band = _valid_band_for_fs(RESP_BAND_HZ, self.fs)
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
        if self.selected_index is None or self.processed_count < max(5, int(round(self.fs * 3.0))):
            search = self.distances >= A121_MIN_TARGET_DISTANCE_M if len(self.distances) == len(amplitude) else np.ones(len(amplitude), dtype=bool)
            if not np.any(search):
                search = np.ones(len(amplitude), dtype=bool)
            idx = int(np.argmax(np.where(search, amplitude, -np.inf)))
            self.selected_index = idx
            self.target_distance_m = float(self.distances[idx]) if len(self.distances) > idx else 0.0
            return idx
        return int(np.clip(self.selected_index, 0, self.m - 1))

    def _candidate_indices(self, selected_idx: int, gate_half_width_m: float, use_gating: bool) -> np.ndarray:
        if self.m <= 0:
            return np.asarray([], dtype=int)
        if use_gating:
            half = max(0.0, float(gate_half_width_m))
            idx = np.flatnonzero(np.abs(self.distances - self.distances[selected_idx]) <= half)
            if len(idx):
                return idx.astype(int)
        lo = max(0, selected_idx - 1)
        hi = min(self.m, selected_idx + 2)
        return np.arange(lo, hi, dtype=int)

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
            static = self._iir_step(z, self.b_static, self.a_static, self.static_x, self.static_y)
            zm = z - static
            amp = np.abs(zm)
            if self.lp_filt_ampl is None:
                self.lp_filt_ampl = amp.copy()
            else:
                self.lp_filt_ampl = self.sf * self.lp_filt_ampl + (1.0 - self.sf) * amp

            angle = np.angle(zm)
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

            if self.b_resp is not None and self.a_resp is not None and self.resp_x is not None and self.resp_y is not None:
                resp_all = self._iir_step(self.angle_unwrapped, self.b_resp, self.a_resp, self.resp_x, self.resp_y)
            else:
                resp_all = self.angle_unwrapped
            if self.b_heart is not None and self.a_heart is not None and self.heart_x is not None and self.heart_y is not None:
                heart_all = self._iir_step(self.angle_unwrapped, self.b_heart, self.a_heart, self.heart_x, self.heart_y)
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
            self.raw_i.append(float(np.real(zm[selected_idx])))
            self.raw_q.append(float(np.imag(zm[selected_idx])))
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
        half = max(0.18, 3.0 * self.current_std_hz)
        return _clamp_band((h - half, h + half), self.band_hz, min_width=0.35)

    def update(self, measurement_hz: float, dt_s: float, *, confidence: float = 0.0, quality: float = 0.0) -> float:
        low, high = self.band_hz
        valid = bool(low <= measurement_hz <= high and confidence >= HEART_RATE_CONFIDENCE_MIN and quality >= 0.25)
        if not self.locked:
            if valid:
                self.x[:] = [measurement_hz, 0.0, 0.0]
                self.P = np.diag([0.10**2, 0.06**2, 0.03**2]).astype(float)
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
                self.missed = 0
            else:
                self.missed += 1
        else:
            self.missed += 1

        if self.missed >= 24:
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
        width = max(0.035, rejected * 0.025)
        score *= 1.0 - 0.92 * np.exp(-0.5 * ((f - rejected) / width) ** 2)
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


def estimate_band_peak_fused(
    values: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    reject_hz: tuple[float, ...] = (),
) -> tuple[float, float]:
    fft_hz, fft_conf = estimate_band_peak(values, fs, band_hz, reject_hz=reject_hz)
    acf_hz, acf_conf = estimate_autocorr_peak(values, fs, band_hz)
    for rejected in reject_hz:
        if rejected > 0 and abs(acf_hz - rejected) <= max(0.04, rejected * 0.03):
            acf_conf = 0.0
            break
    
    if acf_hz > 0 and acf_conf >= 0.22 and fft_hz > 0 and fft_conf >= 2.0:
        # Use autocorrelation only as a confirmer. When the FFT is weak, the ACF often locks
        # onto motion bursts or respiratory harmonics and makes the live heart rate jump.
        if abs(acf_hz - fft_hz) <= 0.25:
            return acf_hz, max(fft_conf, acf_conf * 10.0)

    return fft_hz, fft_conf


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
    vital = (freqs >= RESP_BAND_HZ[0]) & (freqs <= min(HEART_BAND_HZ[1], fs * 0.46))
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
    half = max(0.18, 3.0 * float(heart_prior_std_hz if heart_prior_std_hz is not None else 0.12))
    return _clamp_band((float(heart_prior_hz) - half, float(heart_prior_hz) + half), HEART_BAND_HZ, min_width=0.35)


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

            # Acconeer-style clutter removal and inter-frame phase unwrapping.  This mirrors the
            # A121 breathing reference processor much more closely than refitting absolute IQ
            # circles on every GUI refresh.
            static_cutoff = _valid_band_for_fs(RESP_BAND_HZ, fs)
            zm_profile = complex_profile - _lowpass_static_complex(
                complex_profile,
                fs,
                static_cutoff[0] if static_cutoff is not None else RESP_BAND_HZ[0],
            )
            zm_amplitude = np.abs(zm_profile)
            lp_filt_ampl = _smooth_amplitude(zm_amplitude, fs)
            median_amp = np.median(zm_amplitude, axis=0)
            angle_unwrapped = _unwrap_phase_matrix(zm_profile)
            phase_energy_ratio = _band_energy_ratio(angle_unwrapped, fs)

            half_width = max(0.01, float(gate_half_width_m))
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
            gate_min = target_distance - half_width
            gate_max = target_distance + half_width

            candidate_idx, weights = _candidate_weights(
                distances,
                selected_idx,
                lp_filt_ampl[-1] if len(lp_filt_ampl) else median_amp,
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

            resp_phase_matrix = _bandpass_matrix(angle_unwrapped[:, candidate_idx], fs, RESP_BAND_HZ, order=2, zero_phase=False)
            resp_signal = _aligned_weighted_average(resp_phase_matrix, weights, ref_col)
            resp_hz, resp_confidence, _, _ = _weighted_fft_peak(
                resp_phase_matrix,
                weights,
                fs,
                RESP_BAND_HZ,
                min_duration_s=A121_RATE_WINDOW_S,
            )

            harmonic_rejects: tuple[float, ...]
            if resp_hz > 0:
                harmonic_rejects = tuple(
                    resp_hz * order
                    for order in range(2, 7)
                    if HEART_BAND_HZ[0] * 0.85 <= resp_hz * order <= HEART_BAND_HZ[1] * 1.05
                )
            else:
                harmonic_rejects = ()

            heart_source = _aligned_weighted_average(angle_unwrapped[:, candidate_idx], weights, ref_col)
            heart_source = _subtract_resp_harmonics(heart_source, fs, resp_hz, max_order=6)
            heart_signal = _bandpass_matrix(heart_source, fs, HEART_BAND_HZ, order=3, zero_phase=False).ravel()
            heart_band = _heart_search_band(heart_prior_hz, heart_prior_std_hz)
            heart_hz, heart_confidence = estimate_band_peak_fused(
                heart_signal,
                fs,
                heart_band,
                reject_hz=harmonic_rejects,
            )
            if len(heart_signal) / max(fs, 1e-9) < 10.0:
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

            duration_s = len(raw_phase) / max(fs, 1e-9)
            if duration_s < A121_RATE_WINDOW_S or resp_confidence < RESP_RATE_CONFIDENCE_MIN:
                resp_hz = 0.0
            if heart_confidence < HEART_RATE_CONFIDENCE_MIN:
                heart_hz = 0.0
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
                times_s=times_s[-len(raw_phase) :],
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
    resp_signal = _bandpass_matrix(raw_phase, fs, RESP_BAND_HZ, order=2, zero_phase=False).ravel()
    resp_hz, resp_confidence, _, _ = _weighted_fft_peak(resp_signal, np.asarray([1.0]), fs, RESP_BAND_HZ, min_duration_s=A121_RATE_WINDOW_S)
    heart_source = _subtract_resp_harmonics(raw_phase, fs, resp_hz, max_order=6)
    heart_signal = _bandpass_matrix(heart_source, fs, HEART_BAND_HZ, order=3, zero_phase=False).ravel()
    heart_hz, heart_confidence = estimate_band_peak_fused(
        heart_signal,
        fs,
        _heart_search_band(heart_prior_hz, heart_prior_std_hz),
        reject_hz=tuple(resp_hz * order for order in range(2, 7)) if resp_hz > 0 else (),
    )
    duration_s = len(raw_phase) / max(fs, 1e-9)
    if duration_s < A121_RATE_WINDOW_S or resp_confidence < RESP_RATE_CONFIDENCE_MIN:
        resp_hz = 0.0
    if duration_s < 10.0 or heart_confidence < HEART_RATE_CONFIDENCE_MIN:
        heart_hz = 0.0

    target_distance = float(target_distance_m) if target_distance_m is not None else float(np.median(work["PeakDistance_m"].to_numpy(dtype=float))) if "PeakDistance_m" in work else latest_peak
    amp = work["PeakAmplitude"].to_numpy(dtype=float) if "PeakAmplitude" in work else np.asarray([latest_amp])
    amp_ratio = float(np.median(amp) / (np.median(np.abs(amp - np.median(amp))) * 1.4826 + np.median(amp) + 1e-9))
    presence_score = float(np.clip(amp_ratio * 40.0, 0.0, 100.0))

    present = presence_score >= 20.0
    if not present:
        resp_hz = 0.0
        heart_hz = 0.0
        resp_confidence = 0.0
        heart_confidence = 0.0
    signal_quality = float(np.clip(0.25 * min(resp_confidence / 5.0, 1.0) + 0.25 * min(heart_confidence / 5.0, 1.0) + (0.25 if present else 0.0), 0.0, 1.0))

    return A121VitalAnalysis(
        sample_rate_hz=fs,
        target_distance_m=target_distance,
        gate_min_m=target_distance - gate_half_width_m,
        gate_max_m=target_distance + gate_half_width_m,
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
