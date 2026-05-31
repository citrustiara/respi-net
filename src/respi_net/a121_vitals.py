from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt, welch

from .a121 import parse_json_array

RESP_BAND_HZ = (0.08, 0.70)
HEART_BAND_HZ = (0.65, 3.00)
DEFAULT_GATE_HALF_WIDTH_M = 0.05


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
        valid = bool(low <= measurement_hz <= high and confidence >= 1.3 and quality >= 0.08)
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
    times = np.asarray(times_ms, dtype=float)
    if len(times) < 2:
        return default
    diffs = np.diff(times) / 1000.0
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return default
    median = float(np.median(diffs))
    good = diffs[(diffs > median * 0.25) & (diffs < median * 4.0)] if median > 0 else diffs
    return float(1.0 / np.mean(good if len(good) else diffs))


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
            peak_hz = float(f[idx] + delta * (f[1] - f[0]))
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
    
    if acf_hz > 0 and acf_conf >= 0.22:
        # If they agree within 15 BPM, return the more precise ACF estimate
        if abs(acf_hz - fft_hz) <= 0.25:
            return acf_hz, max(fft_conf, acf_conf * 10.0)
        else:
            # If they disagree, but FFT has low confidence, favor ACF
            if fft_conf < 1.8:
                return acf_hz, acf_conf * 8.0
                
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

    The pipeline is tailored for 60 GHz phase sensitivity: range-gated multi-bin selection,
    algebraic circle centering, coherent spatial combining, respiratory harmonic subtraction,
    and optional Kalman-prior cardiac search-band narrowing for live closed-loop tracking.
    """
    if df.empty or "Timestamp_ms" not in df:
        return _empty_analysis(df)

    work = df.sort_values("Timestamp_ms").tail(max_frames).reset_index(drop=True)
    times_ms = work["Timestamp_ms"].to_numpy(dtype=float)
    fs = sample_rate_from_ms(times_ms, default=20.0)
    times_s = (times_ms - float(times_ms[0])) / 1000.0 if len(times_ms) else np.asarray([], dtype=float)

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
            amplitude = np.abs(complex_profile)
            median_amp = np.median(amplitude, axis=0)
            # Determine gating first to restrict heavy calculations to the active zone
            half_width = max(0.01, float(gate_half_width_m))
            # To prevent locking onto direct TX-RX antenna coupling leakage at the start of the range,
            # we restrict the peak search to distances >= 0.28 m if available.
            search_mask = (distances >= 0.28) if len(distances) else np.asarray([], dtype=bool)
            if not np.any(search_mask):
                search_mask = np.ones_like(distances, dtype=bool) if len(distances) else np.asarray([], dtype=bool)

            if latest_peak and len(distances):
                latest_peak_idx = int(np.argmin(np.abs(distances - latest_peak)))
            elif len(latest_amplitude):
                masked_amp = np.where(search_mask, latest_amplitude, -1e9)
                latest_peak_idx = int(np.argmax(masked_amp))
            else:
                masked_med = np.where(search_mask, median_amp, -1e9)
                latest_peak_idx = int(np.argmax(masked_med))

            if target_distance_m is not None and len(distances):
                selected_idx = int(np.argmin(np.abs(distances - float(target_distance_m))))
            elif auto_gate:
                selected_idx = latest_peak_idx
            elif latest_peak and len(distances):
                selected_idx = latest_peak_idx
            else:
                masked_med = np.where(search_mask, median_amp, -1e9)
                selected_idx = int(np.argmax(masked_med))

            target_distance = float(distances[selected_idx]) if len(distances) else latest_peak
            if use_gating:
                gate_min = target_distance - half_width
                gate_max = target_distance + half_width
                gate = (distances >= gate_min) & (distances <= gate_max) if len(distances) else np.zeros(m, dtype=bool)
                if not np.any(gate):
                    gate = np.zeros(m, dtype=bool)
                    gate[selected_idx] = True
            else:
                gate_min = float(distances[0]) if len(distances) else 0.0
                gate_max = float(distances[-1]) if len(distances) else 10.0
                gate = np.ones(m, dtype=bool)

            # Gated slices
            complex_profile_gated = complex_profile[:, gate]
            real_gated = real[:, gate]
            imag_gated = imag[:, gate]

            # Compute phase matrix only for gated bins
            phase_gated = _differential_phase_matrix(complex_profile_gated)
            phase_motion_gated = np.std(phase_gated, axis=0)

            # Compute energy ratio only for gated bins
            energy_ratio_gated = _band_energy_ratio(phase_gated, fs)

            # Circle fitting only for gated bins
            centered_gated, circle_quality_bins_gated, circle_quality_mean = _center_complex_profile(real_gated, imag_gated)

            # Construct score for gated bins
            amp_floor = float(np.median(median_amp)) + 1e-9
            amp_weight_gated = np.sqrt(np.maximum(median_amp[gate] / amp_floor, 0.0))
            spectral_prominence_gated = np.clip(energy_ratio_gated, 0.5, 25.0)
            score_gated = phase_motion_gated * spectral_prominence_gated * amp_weight_gated * (0.5 + circle_quality_bins_gated)

            # Reconstruct full score and energy_ratio arrays for back-compatibility with other functions
            score = np.zeros(m, dtype=float)
            score[gate] = score_gated
            score = _smooth_bins(score)

            energy_ratio = np.zeros(m, dtype=float)
            energy_ratio[gate] = energy_ratio_gated

            weights, best_gate_idx, candidate_bins = _select_msp_weights(score, gate, energy_ratio, selected_idx)
            raw_phase, raw_i, raw_q = _coherent_differential_phase(complex_profile, weights)
            if len(raw_phase) == 0:
                raw_phase = np.sum(phase_gated * (weights[gate] / (np.sum(weights[gate]) + 1e-12))[None, :], axis=1)
                raw_i = np.sum(real_gated * (weights[gate] / (np.sum(weights[gate]) + 1e-12))[None, :], axis=1)
                raw_q = np.sum(imag_gated * (weights[gate] / (np.sum(weights[gate]) + 1e-12))[None, :], axis=1)

            score_med = float(np.median(score[gate])) if np.any(gate) else 0.0
            score_mad = float(np.median(np.abs(score[gate] - score_med))) * 1.4826 + 1e-9 if np.any(gate) else 1e-9
            dynamic_z = max(0.0, (float(score[best_gate_idx]) - score_med) / score_mad)
            amp_ratio = float(median_amp[best_gate_idx] / amp_floor)
            presence_score = float(np.clip(dynamic_z * 11.0 + max(0.0, amp_ratio - 1.0) * 8.0 + min(candidate_bins, 5) * 2.0, 0.0, 100.0))
            present = bool(presence_score >= 22.0 and amp_ratio >= 1.05)

            resp_signal = bandpass_filter(raw_phase, fs, RESP_BAND_HZ)
            resp_hz, resp_conf = estimate_band_peak_fused(resp_signal, fs, RESP_BAND_HZ)
            harmonic_reject = tuple(resp_hz * order for order in range(2, 7) if resp_hz > 0)
            
            # Use harmonic-subtracted phase for rate estimation to avoid breathing harmonics
            harmonic_free_phase = _subtract_resp_harmonics(raw_phase, fs, resp_hz)
            heart_band = _heart_search_band(heart_prior_hz, heart_prior_std_hz)
            
            # Displayed heart signal is filtered directly from raw_phase to keep the waveform clean and visible
            heart_signal = bandpass_filter(raw_phase, fs, HEART_BAND_HZ)
            heart_signal_clean = bandpass_filter(harmonic_free_phase, fs, heart_band)
            heart_hz, heart_conf = estimate_band_peak_fused(heart_signal_clean, fs, heart_band, reject_hz=harmonic_reject)

            entropy = _spectral_entropy(heart_signal_clean, fs, heart_band)
            signal_quality = float(np.clip(0.35 * circle_quality_mean + 0.35 * (1.0 - entropy) + 0.30 * min(heart_conf / 8.0, 1.0), 0.0, 1.0))
            if heart_conf < 2.0 and signal_quality < 0.35:
                heart_hz = 0.0

            if not present:
                resp_hz = 0.0
                heart_hz = 0.0
                signal_quality = 0.0

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
                resp_hz=resp_hz,
                heart_hz=heart_hz,
                times_s=times_s[-len(raw_phase) :],
                raw_phase=raw_phase,
                resp_signal=resp_signal,
                heart_signal=heart_signal,
                raw_i=raw_i,
                raw_q=raw_q,
                distances_m=distances,
                latest_amplitude=latest_amplitude,
                selected_index=selected_idx,
                resp_confidence=resp_conf,
                heart_confidence=heart_conf,
                signal_quality=signal_quality,
                candidate_bins=candidate_bins,
            )

    if "PeakPhase_rad" not in work:
        return _empty_analysis(work)

    raw_phase = np.unwrap(work["PeakPhase_rad"].to_numpy(dtype=float))
    raw_phase = clean_signal(raw_phase)
    resp_signal = bandpass_filter(raw_phase, fs, RESP_BAND_HZ)
    resp_hz, resp_conf = estimate_band_peak(resp_signal, fs, RESP_BAND_HZ)
    harmonic_reject = tuple(resp_hz * order for order in range(2, 7) if resp_hz > 0)
    harmonic_free_phase = _subtract_resp_harmonics(raw_phase, fs, resp_hz)
    heart_band = _heart_search_band(heart_prior_hz, heart_prior_std_hz)
    heart_signal = bandpass_filter(harmonic_free_phase, fs, heart_band)
    heart_hz, heart_conf = estimate_band_peak(heart_signal, fs, heart_band, reject_hz=harmonic_reject)
    entropy = _spectral_entropy(heart_signal, fs, heart_band)
    signal_quality = float(np.clip(0.35 * (1.0 - entropy) + 0.30 * min(heart_conf / 8.0, 1.0), 0.0, 1.0))
    target_distance = float(target_distance_m) if target_distance_m is not None else float(np.median(work["PeakDistance_m"].to_numpy(dtype=float))) if "PeakDistance_m" in work else latest_peak
    amp = work["PeakAmplitude"].to_numpy(dtype=float) if "PeakAmplitude" in work else np.asarray([latest_amp])
    amp_ratio = float(np.median(amp) / (np.median(np.abs(amp - np.median(amp))) * 1.4826 + np.median(amp) + 1e-9))
    phase_motion = float(np.std(raw_phase))
    presence_score = float(np.clip(phase_motion * 35.0 + max(0.0, amp_ratio - 0.5) * 15.0, 0.0, 100.0))

    return A121VitalAnalysis(
        sample_rate_hz=fs,
        target_distance_m=target_distance,
        gate_min_m=target_distance - gate_half_width_m,
        gate_max_m=target_distance + gate_half_width_m,
        peak_distance_m=latest_peak,
        peak_amplitude=latest_amp,
        mean_amplitude=mean_amp,
        presence_score=presence_score,
        present=presence_score >= 22.0,
        resp_bpm=float(resp_hz * 60.0),
        heart_bpm=float(heart_hz * 60.0),
        resp_hz=resp_hz,
        heart_hz=heart_hz,
        times_s=times_s,
        raw_phase=raw_phase,
        resp_signal=resp_signal,
        heart_signal=heart_signal,
        raw_i=np.asarray([], dtype=float),
        raw_q=np.asarray([], dtype=float),
        distances_m=distances,
        latest_amplitude=latest_amplitude,
        selected_index=0,
        resp_confidence=resp_conf,
        heart_confidence=heart_conf,
        signal_quality=signal_quality,
        candidate_bins=1,
    )
