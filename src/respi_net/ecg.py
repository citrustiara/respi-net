"""AD8232 single-lead ECG: R-peak detection and heart-rate labels for radar models.

Hardware and sampling (checked against ``firmware/esp32_radar_adc/main/radar_main.c``)
-------------------------------------------------------------------------------------
* Wire the AD8232 ``OUTPUT`` to ESP32 GPIO33 (ADC1 channel 5), the pin the HB100
  firmware already samples, so that image can be reused unchanged. It averages 64
  conversions per sample at 12 dB attenuation and prints one headerless line
  ``Timestamp_ms,RawADC,Voltage_mV`` every 4 ms (250 Hz; FreeRTOS tick is 1 kHz) at
  230400 baud. The host capture tools add ``HostTimestamp_ms`` (serial arrival time).
* Power the AD8232 (a 2.0-3.5 V part) from the ESP32 3V3 pin with a common GND. Its
  output then idles near mid-supply (~1.65 V), inside the ADC range. The ADC is
  non-linear above ~2.45 V at this attenuation, which only compresses tall R waves;
  timing, the only thing used here, is unaffected.
* ``LO+``/``LO-`` (digital leads-off flags) are optional: route them to spare GPIOs only
  if the firmware is extended. Without them a detached electrode shows up as a railed
  or noisy trace whose RR intervals :func:`heart_rate_series` rejects.
* Chest electrodes: RA below the right clavicle, LA below the left clavicle, RL on the
  lower right rib cage. Swapping RA/LA inverts the ECG; :func:`detect_r_peaks` detects
  the polarity itself.
* Sample at >= 250 Hz (4 ms R-peak resolution); 500 Hz is better for beat-level
  labels. The current firmware sits exactly at the 250 Hz minimum.

Radar labels: the radar sees the mechanical heartbeat about one pre-ejection period
(PEP, ~105 ms at rest) after the R-peak, so beat-level labels should come from
:func:`mechanical_beat_times`. Windowed HR needs no shift, because a constant delay
cancels in RR intervals.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

MIN_FS_HZ = 250.0  # 4 ms R-peak resolution; below this beat timing degrades
RECOMMENDED_FS_HZ = 500.0
PRE_EJECTION_S = 0.10  # R-peak -> ventricular ejection at rest (PEP ~105 ms)
RR_RANGE_S = (0.3, 2.0)  # 200-30 bpm: outside this an interval is a detection error
QRS_BAND_HZ = (5.0, 15.0)  # QRS energy; rejects baseline wander, P/T waves and mains hum
DEFAULT_MAX_RR_CHANGE = 0.25
DEFAULT_MIN_VALID_BEATS = 4
DEFAULT_MIN_COVERAGE = 0.5  # share of a label window that valid RR intervals must span
MIN_SAMPLES = 10

DEVICE_COLUMN, HOST_COLUMN, VOLTAGE_COLUMN = "Timestamp_ms", "HostTimestamp_ms", "Voltage_mV"
# One firmware line (device ms, raw count, calibrated mV); ESP-IDF log lines do not match.
_FIRMWARE_LINE = re.compile(r"^\s*(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class EcgRecording:
    """ECG samples on one clock; ``time_s`` is seconds since the first sample."""

    time_s: np.ndarray
    voltage_mv: np.ndarray  # AD8232 OUTPUT at the ADC pin, not body-surface millivolts
    fs: float
    clock: str  # "device", "host" (device clock mapped onto host clock) or "host-arrival"
    start_epoch_s: float | None = None  # host wall-clock time of time_s == 0 (radar alignment)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])


@dataclass(frozen=True)
class HeartRateSeries:
    """Beat-to-beat heart rate; element ``i`` is the RR interval ending at ``beat_times_s[i]``."""

    beat_times_s: np.ndarray
    rr_s: np.ndarray
    hr_bpm: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class SimulatedEcg:
    """Synthetic ECG with the true R-wave apex times, for scoring detectors."""

    ecg_mv: np.ndarray
    fs: float
    r_peak_times_s: np.ndarray

    @property
    def time_s(self) -> np.ndarray:
        return np.arange(len(self.ecg_mv)) / self.fs


def load_ecg_csv(path: str | Path) -> EcgRecording:
    """Load an ESP32 ADC capture: ``Timestamp_ms[,HostTimestamp_ms],RawADC,Voltage_mV``.

    Accepts the plain firmware format (with a header, or a raw headerless serial dump) and
    the host-timestamped format of the capture tools. With host timestamps the exact device
    clock is mapped onto the host clock, so beats align with other host-stamped sensors
    without the USB delivery jitter of raw arrival times.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ECG CSV not found: {path}")
    frame = _read_table(path)
    missing = [column for column in (DEVICE_COLUMN, VOLTAGE_COLUMN) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {list(frame.columns)}")
    device = pd.to_numeric(frame[DEVICE_COLUMN], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(frame[VOLTAGE_COLUMN], errors="coerce").to_numpy(dtype=float)
    host = np.full(len(frame), np.nan)
    if HOST_COLUMN in frame.columns:
        host = pd.to_numeric(frame[HOST_COLUMN], errors="coerce").to_numpy(dtype=float)
    use_host = bool(len(host)) and np.isfinite(host).mean() > 0.5
    keep = np.isfinite(device) & np.isfinite(voltage) & (np.isfinite(host) if use_host else True)
    device, host, voltage = device[keep], host[keep], voltage[keep]
    if len(device) < MIN_SAMPLES:
        raise ValueError(f"{path}: {len(device)} usable samples (need >= {MIN_SAMPLES}); not an ADC capture?")
    if use_host:
        order = np.argsort(host, kind="stable")
        device, host, voltage = device[order], host[order], voltage[order]
        if np.all(np.diff(device) >= 0):
            clock, stamps_ms = "host", _device_on_host_clock(device, host)
        else:  # the ESP32 restarted mid-file: only the (jittery) arrival times are consistent
            clock, stamps_ms = "host-arrival", host
    else:
        if np.any(np.diff(device) < 0):
            raise ValueError(f"{path}: Timestamp_ms goes backwards (ESP32 reset?); split the capture")
        clock, stamps_ms = "device", device
    unique = np.concatenate(([True], np.diff(stamps_ms) > 0))
    stamps_ms, voltage = stamps_ms[unique], voltage[unique]
    if len(stamps_ms) < MIN_SAMPLES:
        raise ValueError(f"{path}: only {len(stamps_ms)} distinct timestamps")
    return EcgRecording(
        time_s=(stamps_ms - stamps_ms[0]) / 1000.0,
        voltage_mv=voltage,
        fs=1000.0 / float(np.median(np.diff(stamps_ms))),
        clock=clock,
        start_epoch_s=float(stamps_ms[0]) / 1000.0 if use_host else None,
    )


def _read_table(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8", errors="replace") as handle:
        first = next((line for line in handle if line.strip()), "")
    if DEVICE_COLUMN in first or VOLTAGE_COLUMN in first:
        return pd.read_csv(path)
    # Headerless serial dump straight from the firmware, possibly with ESP-IDF log lines.
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = [match.groups() for match in map(_FIRMWARE_LINE.match, lines) if match]
    return pd.DataFrame(rows, columns=[DEVICE_COLUMN, "RawADC", VOLTAGE_COLUMN])


def _device_on_host_clock(device_ms: np.ndarray, host_ms: np.ndarray, chunk_s: float = 30.0) -> np.ndarray:
    # Host stamps are serial arrival times: in a real 250 Hz HB100 capture the device spacing
    # was exactly 4 ms while host spacing ranged 0.002-12 ms (USB delivers lines in bursts).
    # Arrival can only be late, so a line through the per-chunk lower envelope of
    # (host - device) gives clock offset and drift without the delivery jitter.
    lag = host_ms - device_ms
    elapsed = device_ms - device_ms[0]
    n_chunks = max(1, int(elapsed[-1] / (chunk_s * 1000.0)))
    parts = np.array_split(np.arange(len(lag)), n_chunks)
    x = np.array([np.median(elapsed[part]) for part in parts])
    y = np.array([np.percentile(lag[part], 5) for part in parts])
    slope, offset = np.polyfit(x, y, 1) if n_chunks > 1 else (0.0, float(y[0]))
    return device_ms + offset + slope * elapsed


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Return R-peak sample indices (Pan-Tompkins detection, refined on the band-passed ECG).

    Band-pass 5-15 Hz (zero-phase, so timing is kept), 5-point derivative, squaring and
    150 ms moving integration feed adaptive signal/noise thresholds with a 250 ms
    refractory period, T-wave rejection and search-back. Each detection is then moved to
    the extremum of the band-passed ECG within +-50 ms. Polarity is decided once per
    recording from the dominant QRS deflection, so swapped electrodes still give the R
    apex instead of flipping between R and S from beat to beat. Detections over 3x the
    amplitude of the surrounding beats (motion artifacts) are dropped.
    """
    x = np.asarray(ecg, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("ecg must be a 1-D array of finite samples")
    if fs < 100.0:
        raise ValueError(f"fs={fs:.0f} Hz is too low for R-peak detection; sample at >= {MIN_FS_HZ:.0f} Hz")
    if fs < MIN_FS_HZ:
        warnings.warn(
            f"fs={fs:.0f} Hz < {MIN_FS_HZ:.0f} Hz: R-peak timing will be coarse "
            f"(sample at >= {MIN_FS_HZ:.0f} Hz, ideally {RECOMMENDED_FS_HZ:.0f} Hz)",
            stacklevel=2,
        )
    if len(x) < int(fs):
        return np.array([], dtype=int)
    band = sosfiltfilt(butter(2, QRS_BAND_HZ, btype="bandpass", fs=fs, output="sos"), x)
    slope = np.convolve(band, np.array([1.0, 2.0, 0.0, -2.0, -1.0]) * fs / 8.0, mode="same")
    width = max(1, round(0.150 * fs))
    integrated = np.convolve(slope**2, np.ones(width) / width, mode="same")
    beats = _threshold_qrs(integrated, np.abs(slope), fs)
    if not beats:
        return np.array([], dtype=int)
    half = max(1, round(0.050 * fs))
    windows = [slice(max(beat - half, 0), beat + half + 1) for beat in beats]
    upward = np.median([band[w].max() for w in windows]) >= np.median([-band[w].min() for w in windows])
    polarity = 1.0 if upward else -1.0
    peaks = np.unique([w.start + int(np.argmax(polarity * band[w])) for w in windows])
    # Motion artifacts can pass the energy thresholds but dwarf real QRS complexes: drop
    # detections > 3x the median amplitude of the preceding or the following 60 s, whichever
    # is larger, so a genuine amplitude step (electrode contact change) is kept.
    amplitude = polarity * band[peaks]
    span = round(60 * fs)
    reference = np.empty(len(peaks))
    for k, peak in enumerate(peaks):
        before = amplitude[np.searchsorted(peaks, peak - span) : k + 1]
        after = amplitude[k : np.searchsorted(peaks, peak + span, side="right")]
        reference[k] = max(np.median(before), np.median(after))
    return peaks[amplitude <= 3.0 * reference]


def _typical_qrs_energy(integrated: np.ndarray, fs: float) -> float:
    # Every 2 s block holds a beat at >= 30 bpm, so the median block maximum is a typical QRS
    # level that ignores both isolated artifacts and stretches without beats.
    n_blocks = max(1, len(integrated) // int(2 * fs))
    return float(np.median([part.max() for part in np.array_split(integrated, n_blocks)]))


def _initial_levels(integrated: np.ndarray, start: int, fs: float) -> tuple[float, float]:
    segment = integrated[start : start + int(8 * fs)]
    return _typical_qrs_energy(segment, fs), float(np.median(segment))


def _threshold_qrs(integrated: np.ndarray, slope_abs: np.ndarray, fs: float) -> list[int]:
    """Pan-Tompkins decision rules on the integrated signal; returns QRS sample indices."""
    candidates, _ = find_peaks(integrated, distance=max(1, round(0.1 * fs)))
    refractory, t_wave = round(0.25 * fs), round(0.36 * fs)
    qrs_half = max(1, round(0.075 * fs))

    def max_slope(k: int) -> float:
        return float(slope_abs[max(k - qrs_half, 0) : k + qrs_half + 1].max())

    # A floor tied to the recording's typical QRS energy stops flat or railed stretches
    # (detached electrode) from being learned as signal.
    floor = 0.02 * _typical_qrs_energy(integrated, fs)
    spki, npki = _initial_levels(integrated, 0, fs)
    beats: list[int] = []
    rr: list[int] = []
    searched = relearned = False
    i = 0
    while i < len(candidates):
        peak = int(candidates[i])
        last = beats[-1] if beats else -refractory
        gap = peak - last
        rr_ref = float(np.median(rr[-8:])) if rr else fs
        threshold = max(npki + 0.25 * (spki - npki), floor)
        beat, weight = None, 0.125
        if gap > 1.66 * rr_ref and not searched:
            # Search-back: a beat was probably missed; take the best candidate above half threshold.
            searched = True
            lo = int(np.searchsorted(candidates, last + refractory, side="left"))
            hi = int(np.searchsorted(candidates, last + 1.66 * rr_ref, side="right"))
            back = candidates[lo:hi]
            back = back[integrated[back] > 0.5 * threshold]
            if back.size:
                beat, weight = int(back[np.argmax(integrated[back])]), 0.25
        if beat is None:
            if gap > max(2.5 * rr_ref, 2.0 * fs) and not relearned:
                # Search-back found nothing either (a motion burst inflated the signal level, or
                # the amplitude dropped): re-learn the levels from the next 8 s (offline look-ahead).
                relearned = True
                spki, npki = _initial_levels(integrated, peak, fs)
                threshold = max(npki + 0.25 * (spki - npki), floor)
            if gap < refractory:
                i += 1
                continue
            height = float(integrated[peak])
            t_wave_like = bool(beats) and gap < t_wave and max_slope(peak) < 0.5 * max_slope(last)
            if height <= threshold or t_wave_like:
                npki = 0.125 * height + 0.875 * npki
                i += 1
                continue
            beat = peak
        # A noisy P wave can cross the threshold first and hide its QRS behind the refractory
        # period, so keep the largest integrated peak within the refractory window (offline).
        ahead = candidates[np.searchsorted(candidates, beat) : np.searchsorted(candidates, beat + refractory)]
        beat = int(ahead[np.argmax(integrated[ahead])])
        if beats:
            rr.append(beat - beats[-1])
        beats.append(beat)
        # Cap the update so a single motion artifact cannot lift the threshold above real beats.
        spki = weight * min(float(integrated[beat]), 2.0 * spki) + (1.0 - weight) * spki
        searched = relearned = False
        i = int(np.searchsorted(candidates, beat, side="right"))
    return beats


def heart_rate_series(peak_times_s: np.ndarray, *, max_rr_change: float = DEFAULT_MAX_RR_CHANGE) -> HeartRateSeries:
    """Beat-to-beat RR/HR with a validity mask for use as ground truth.

    An interval is valid when it is plausible (0.3-2.0 s), within ``max_rr_change``
    (relative) of the median of its plausible neighbours (+-4 intervals), and not adjacent
    to an implausible interval. A local median, unlike the previous interval, flags only the
    bad intervals around a missed or extra beat. An implausible neighbour means a dropout
    or a double detection, so the beat shared with it may be an artifact.
    """
    times = np.asarray(peak_times_s, dtype=float)
    if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("peak_times_s must be a 1-D, finite, strictly increasing array")
    rr = np.diff(times)
    plausible = (rr >= RR_RANGE_S[0]) & (rr <= RR_RANGE_S[1])
    neighbours = pd.Series(np.where(plausible, rr, np.nan)).rolling(9, center=True, min_periods=2)
    reference = neighbours.median().to_numpy()
    padded = np.concatenate(([True], plausible, [True]))
    valid = plausible & padded[:-2] & padded[2:] & (np.abs(rr - reference) <= max_rr_change * reference)
    return HeartRateSeries(beat_times_s=times[1:], rr_s=rr, hr_bpm=60.0 / rr, valid=valid)


def hr_labels_for_windows(
    peak_times_s: np.ndarray,
    window_starts_s: np.ndarray,
    window_s: float,
    *,
    min_valid_beats: int = DEFAULT_MIN_VALID_BEATS,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    max_rr_change: float = DEFAULT_MAX_RR_CHANGE,
) -> np.ndarray:
    """Mean HR (bpm) per window ``[start, start + window_s]`` for a windowed radar HR model.

    Uses valid RR intervals lying fully inside the window; NaN when fewer than
    ``min_valid_beats`` of them, or when together they span less than ``min_coverage`` of
    the window (otherwise an artifact-ridden window is labelled from a few seconds). The
    label is 60 / mean(RR), i.e. beats per minute, not the mean of instantaneous HR, which
    RR variability biases upwards. Pass R-peak times: a constant electromechanical delay
    does not change RR intervals.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    starts = np.asarray(window_starts_s, dtype=float)
    labels = np.full(starts.shape, np.nan)
    series = heart_rate_series(peak_times_s, max_rr_change=max_rr_change)
    ends, rr = series.beat_times_s[series.valid], series.rr_s[series.valid]
    begins = ends - rr
    for k, start in enumerate(starts):
        inside = (begins >= start) & (ends <= start + window_s)
        if np.count_nonzero(inside) >= min_valid_beats and np.sum(rr[inside]) >= min_coverage * window_s:
            labels[k] = 60.0 / float(np.mean(rr[inside]))
    return labels


def mechanical_beat_times(peak_times_s: np.ndarray, pre_ejection_s: float = PRE_EJECTION_S) -> np.ndarray:
    """Shift R-peak times to when the radar sees the beat, for beat-level radar labels.

    Ventricular ejection (the chest-wall/aortic motion a radar senses) follows electrical
    activation by the pre-ejection period: ~105 ms at rest, shorter under stress or
    exercise, and subject/posture dependent, so calibrate per subject if beat timing
    matters. Windowed HR labels need no shift.
    """
    if not 0.0 <= pre_ejection_s < 0.5:
        raise ValueError(f"pre_ejection_s={pre_ejection_s} must be seconds in [0, 0.5), typically ~0.10")
    return np.asarray(peak_times_s, dtype=float) + pre_ejection_s


# (offset from the R apex [s], Gaussian sigma [s], amplitude [mV]) per wave. The T offset is
# scaled by sqrt(RR) (QT shortens with heart rate) so fast rhythms keep T before the next P.
_ECG_WAVES = {
    "P": (-0.16, 0.025, 0.12),
    "Q": (-0.03, 0.010, -0.12),
    "R": (0.0, 0.012, 1.0),
    "S": (0.03, 0.012, -0.30),
    "T": (0.30, 0.045, 0.30),
}


def simulate_ecg(
    duration_s: float,
    fs: float,
    hr_bpm: float,
    rng: np.random.Generator,
    *,
    hrv: float = 0.05,
    noise_mv: float = 0.02,
    baseline_wander_mv: float = 0.3,
    mains_mv: float = 0.0,
    inverted: bool = False,
) -> SimulatedEcg:
    """Synthetic single-lead ECG from Gaussian P/Q/R/S/T waves (R ~1 mV).

    ``hrv`` is the relative SD of RR (smooth AR(1) variation, like sinus variability);
    baseline wander mixes a breathing-rate and a slow-drift sinusoid; ``noise_mv`` is white
    noise SD and ``mains_mv`` a 50 Hz hum amplitude. ``inverted`` mimics swapped RA/LA.
    """
    if duration_s <= 0 or fs <= 0 or hr_bpm <= 0:
        raise ValueError("duration_s, fs and hr_bpm must be positive")
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    rr0 = 60.0 / hr_bpm
    beat_times, preceding_rr = [], []
    beat, rr, z = float(rng.uniform(0.0, rr0)), rr0, 0.0
    while beat < duration_s + 1.0:
        beat_times.append(beat)
        preceding_rr.append(rr)
        z = 0.8 * z + 0.6 * float(rng.standard_normal())  # AR(1) with unit variance
        rr = rr0 * (1.0 + hrv * float(np.clip(z, -3.0, 3.0)))
        beat += rr
    ecg = np.zeros(n)
    scales = 1.0 + 0.05 * rng.standard_normal(len(beat_times))  # beat-to-beat amplitude change
    for r_time, rr, scale in zip(beat_times, preceding_rr, scales):
        for name, (offset, sigma, amplitude) in _ECG_WAVES.items():
            centre = r_time + (offset * np.sqrt(rr) if name == "T" else offset)
            lo, hi = max(0, int((centre - 5 * sigma) * fs)), min(n, int((centre + 5 * sigma) * fs) + 2)
            if lo < hi:
                ecg[lo:hi] += scale * amplitude * np.exp(-0.5 * ((t[lo:hi] - centre) / sigma) ** 2)
    phases = rng.uniform(0.0, 2 * np.pi, 3)
    breathing_hz = rng.uniform(0.15, 0.35)
    wander = 0.7 * np.sin(2 * np.pi * breathing_hz * t + phases[0]) + 0.3 * np.sin(2 * np.pi * 0.05 * t + phases[1])
    signal = ecg + baseline_wander_mv * wander + noise_mv * rng.standard_normal(n)
    signal += mains_mv * np.sin(2 * np.pi * 50.0 * t + phases[2])
    r_times = np.asarray(beat_times)
    return SimulatedEcg(
        ecg_mv=-signal if inverted else signal,
        fs=float(fs),
        r_peak_times_s=r_times[(r_times >= 0.0) & (r_times < n / fs)],
    )
