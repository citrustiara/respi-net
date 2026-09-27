"""Synthetic breathing with perfect phase labels, an A121 forward model and augmentations.

Labelled radar recordings are few, and their labels only as good as the coach
cues and the labeller.  Here every sample's phase (:mod:`respi_net.breath_phases`)
is known by construction, in the units and sign of the A121 front end: chest
displacement in mm at ~20 Hz, ``chest_mm`` rising on inhale as the chest moves
toward the radar.

The default ranges are calibrated on this project's recordings:

* coached deep breaths measure 5-12 mm peak to trough and natural sleep
  breaths 0.4-3 mm (median ~0.8 mm), so depths are log-uniform over 0.2-12 mm;
* during coached breath holds the displacement still moves ~0.5 mm RMS in
  0.1-1 Hz, heartbeat included, which sets the top of the noise ranges.

:func:`a121_forward` turns displacement into complex range bins and
:func:`displacement_from_iq` reads it back the way the real front end does, so
training data can carry the front end's own failures (phase-unwrapping slips),
not only additive noise.  The augmentations take signals as ``[T]`` or
``[C, T]`` with labels ``[T]`` (IGNORE allowed), never modify their inputs, and
draw randomness only from the ``rng`` they are given.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import Callable, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .breath_phases import (
    EXHALE,
    HOLD_AFTER_EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    NOISE,
    label_runs,
)

A121_WAVELENGTH_MM = 299_792_458.0 / 60.5e9 * 1e3  # c / 60.5 GHz ≈ 4.955 mm

# Breath-to-breath wobble of the inhale fraction around the recording's own value.
_INHALE_FRACTION_JITTER = 0.03
# How long the body takes to settle into a new posture.
_POSTURE_STEP_S = 1.0
# Drift is smoothed over this long, keeping it far below the slowest breathing (5/min).
_DRIFT_SMOOTH_S = 5.0

# Maps a phase's progress u in [0, 1) to the fraction of its move done.
_Shape = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SynthConfig:
    """What :func:`simulate_breathing` draws from.

    ``(low, high)`` pairs are ranges drawn once per recording unless a comment
    says otherwise, so each recording has a character of its own.  Rates, depths
    and hold lengths are log-uniform: 0.5 mm differs from 1 mm as 5 mm from 10.
    """

    # Rhythm.
    rate_bpm: tuple[float, float] = (5.0, 35.0)
    rate_walk: float = 0.03  # std of each breath's step in log(rate): a slow random walk
    period_jitter: float = 0.08  # breath-to-breath coefficient of variation of the cycle
    inhale_fraction: tuple[float, float] = (0.3, 0.5)  # of inhale + exhale, holds excluded
    # Expiratory flow decay as a fraction of the exhale.  Below ~0.17 the last
    # third of a slow exhale is flat, which would look like a hold yet read EXHALE.
    exhale_tau: tuple[float, float] = (0.17, 0.3)
    # Size, as chest displacement peak to trough.
    depth_mm: tuple[float, float] = (0.2, 12.0)
    depth_variability: tuple[float, float] = (0.1, 0.3)  # breath-to-breath coefficient of variation
    sigh_prob: float = 0.02  # per breath
    sigh_scale: tuple[float, float] = (1.5, 3.0)  # drawn per sigh
    # Holds: decided per breath, lengths drawn per hold.
    hold_after_inhale_prob: float = 0.04
    hold_after_exhale_prob: float = 0.06
    hold_s: tuple[float, float] = (1.0, 20.0)
    hold_sag: float = 0.1  # a held full chest sinks by up to this fraction of the breath
    # Everything else the radar sees; none of it changes the labels except motion bursts.
    heart_rate_bpm: tuple[float, float] = (50.0, 110.0)
    heart_mm: tuple[float, float] = (0.05, 0.4)  # peak to peak
    heart_rate_variability: float = 0.05  # std of the slow relative wander of the heart rate
    drift_mm_per_min: float = 1.0  # steepest baseline drift, see add_drift
    posture_steps_per_min: float = 0.1
    posture_step_mm: tuple[float, float] = (0.1, 1.0)  # drawn per step
    white_noise_mm: tuple[float, float] = (0.005, 0.1)  # std
    lf_noise_mm: tuple[float, float] = (0.02, 0.6)  # std of noise below ~1 Hz
    motion_bursts_per_min: float = 0.2  # labelled NOISE
    motion_burst_s: tuple[float, float] = (1.0, 6.0)  # drawn per burst
    motion_burst_mm: tuple[float, float] = (2.0, 20.0)  # largest excursion, drawn per burst

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            low, high = value if isinstance(item.default, tuple) else (value, value)
            if not 0 <= low <= high:
                raise ValueError(f"SynthConfig.{item.name} must be non-negative with low <= high, got {value!r}")
        if min(self.rate_bpm[0], self.depth_mm[0], self.hold_s[0], self.exhale_tau[0]) <= 0:
            raise ValueError("SynthConfig.rate_bpm, depth_mm, hold_s and exhale_tau must start above zero")
        if not (0 < self.inhale_fraction[0] and self.inhale_fraction[1] < 1):
            raise ValueError("SynthConfig.inhale_fraction must lie strictly between 0 and 1")
        if max(self.hold_after_inhale_prob, self.hold_after_exhale_prob, self.sigh_prob, self.hold_sag) > 1:
            raise ValueError("SynthConfig probabilities and hold_sag cannot exceed 1")


@dataclass(frozen=True)
class SyntheticBreathing:
    """A simulated recording: chest displacement and the phase of every sample.

    ``events`` are the label runs as ``(start_s, end_s, class)`` with the end
    exclusive, so :func:`~respi_net.breath_phases.labels_from_segments` rebuilds
    ``labels`` from them exactly.
    """

    time_s: np.ndarray
    chest_mm: np.ndarray
    labels: np.ndarray  # int8, classes 0-4, never IGNORE
    events: list[tuple[float, float, int]]
    fs: float


def simulate_breathing(
    duration_s: float,
    fs: float,
    rng: np.random.Generator,
    config: SynthConfig = SynthConfig(),
) -> SyntheticBreathing:
    """Simulate ``duration_s`` seconds of chest displacement at ``fs`` Hz with perfect labels.

    Breaths run inhale → [hold after inhale] → exhale → [hold after exhale], all
    steps of :data:`~respi_net.breath_phases.ALLOWED_TRANSITIONS`; motion bursts
    overwrite what they land on with NOISE, which may interrupt any phase.
    Heartbeat, posture steps, drift and noise leave the labels alone.  Zero the
    noise ranges when feeding :func:`a121_forward`, which adds its own.
    """

    if fs <= 0 or duration_s * fs < 0.5:
        raise ValueError(f"Need a positive duration and sample rate, got {duration_s} s at {fs} Hz")
    count = int(round(duration_s * fs))
    breathing, labels = _breathing_track(count, fs, rng, config)
    motion, moving = _motion_bursts(count, fs, rng, config)
    labels[moving] = NOISE
    chest_mm = (
        breathing
        + motion
        + _heartbeat(count, fs, rng, config)
        + _posture_steps(count, fs, rng, config)
        + rng.uniform(*config.lf_noise_mm) * _smooth_noise(count, fs, rng, smooth_s=0.3)
        + rng.uniform(*config.white_noise_mm) * rng.standard_normal(count)
    )
    chest_mm = add_drift(chest_mm, fs, rng, config.drift_mm_per_min)
    events = [(start / fs, stop / fs, label) for start, stop, label in label_runs(labels)]
    return SyntheticBreathing(np.arange(count) / fs, chest_mm, labels, events, float(fs))


class _Track:
    """Phases laid end to end on the sample grid.

    Boundaries come from rounding a running clock rather than each phase's own
    length, so rounding never adds up to a rate error.
    """

    def __init__(self, fs: float) -> None:
        self.fs = fs
        self.clock = 0.0  # in samples
        self.levels: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

    @property
    def length(self) -> int:
        return int(round(self.clock))

    def add(self, label: int, seconds: float, start_mm: float, end_mm: float, shape: _Shape) -> None:
        begin = self.length
        # At least one sample each: a phase lost to rounding could leave, say, a
        # hold after inhale straight after an exhale, which is not a valid sequence.
        self.clock = max(self.clock + seconds * self.fs, begin + 1.0)
        n = self.length - begin
        self.levels.append(start_mm + (end_mm - start_mm) * shape(np.arange(n) / n))
        self.labels.append(np.full(n, label, dtype=np.int8))


def _breathing_track(
    count: int, fs: float, rng: np.random.Generator, config: SynthConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Clean breathing in mm above the end-expiratory level, and its labels."""

    log_rate_range = np.log(config.rate_bpm)
    log_rate = rng.uniform(*log_rate_range)
    inhale_fraction = rng.uniform(*config.inhale_fraction)
    depth_mm = _log_uniform(rng, config.depth_mm)
    variability = rng.uniform(*config.depth_variability)
    exhale_shape = partial(_exhale_shape, tau=rng.uniform(*config.exhale_tau))
    # Start part-way into a breath, so recordings do not all open on an inhale onset.
    skip = int(round(rng.uniform(0.0, 60.0 / np.exp(log_rate)) * fs))
    track = _Track(fs)
    while track.length < skip + count:
        period_s = 60.0 / np.exp(log_rate) * np.exp(config.period_jitter * rng.standard_normal())
        depth = depth_mm * np.exp(variability * rng.standard_normal() - variability**2 / 2)
        if rng.random() < config.sigh_prob:
            scale = rng.uniform(*config.sigh_scale)
            depth, period_s = depth * scale, period_s * np.sqrt(scale)  # deeper, and slower with it
        jitter = _INHALE_FRACTION_JITTER * rng.standard_normal()
        fraction = np.clip(inhale_fraction + jitter, *config.inhale_fraction)
        track.add(INHALE, period_s * fraction, 0.0, depth, _smoothstep)
        level = depth
        if rng.random() < config.hold_after_inhale_prob:
            held = level * (1.0 - config.hold_sag * rng.random())
            track.add(HOLD_AFTER_INHALE, _log_uniform(rng, config.hold_s), level, held, _smoothstep)
            level = held
        track.add(EXHALE, period_s * (1.0 - fraction), level, 0.0, exhale_shape)
        if rng.random() < config.hold_after_exhale_prob:
            track.add(HOLD_AFTER_EXHALE, _log_uniform(rng, config.hold_s), 0.0, 0.0, _smoothstep)
        log_rate = _reflect(log_rate + config.rate_walk * rng.standard_normal(), *log_rate_range)
    levels, labels = np.concatenate(track.levels), np.concatenate(track.labels)
    return levels[skip : skip + count], labels[skip : skip + count]


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """Raised cosine, 0 to 1 with zero slope at both ends: the half-sine flow of quiet inspiration."""

    return 0.5 - 0.5 * np.cos(np.pi * u)


def _exhale_shape(u: np.ndarray, tau: float) -> np.ndarray:
    """Fraction of the breath exhaled by ``u`` when the flow goes as ``u·exp(-u/tau)``.

    Passive recoil empties the lungs fast and then ever slower, so the flow peaks
    early (at ``u = tau``) and decays; starting it from zero rounds the top of the
    breath instead of leaving a corner there.
    """

    return (1.0 - np.exp(-u / tau) * (1.0 + u / tau)) / (1.0 - np.exp(-1.0 / tau) * (1.0 + 1.0 / tau))


def _log_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1]))))


def _reflect(value: float, low: float, high: float) -> float:
    """Fold a random-walk step back into ``[low, high]`` instead of piling up at the edge."""

    value = 2.0 * high - value if value > high else value
    return float(np.clip(2.0 * low - value if value < low else value, low, high))


def _smooth_noise(count: int, fs: float, rng: np.random.Generator, smooth_s: float) -> np.ndarray:
    """Gaussian noise of unit std low-passed by a Gaussian kernel ``smooth_s`` seconds wide."""

    sigma = smooth_s * fs
    noise = gaussian_filter1d(rng.standard_normal(count), sigma, mode="reflect")
    # Divide by the filter's own gain, not the sample std, which a short and
    # nearly constant stretch would inflate without bound.
    return noise / np.sqrt(min(1.0, 1.0 / (2.0 * np.sqrt(np.pi) * sigma)))


def _heartbeat(count: int, fs: float, rng: np.random.Generator, config: SynthConfig) -> np.ndarray:
    """Chest-wall motion of the heartbeat: a fundamental and second harmonic at a wandering rate."""

    rate_hz = rng.uniform(*config.heart_rate_bpm) / 60.0
    wander = 1.0 + config.heart_rate_variability * _smooth_noise(count, fs, rng, smooth_s=3.0)
    phase = 2.0 * np.pi * np.cumsum(rate_hz * wander) / fs + rng.uniform(0.0, 2.0 * np.pi)
    harmonic, lag = rng.uniform(0.2, 0.6), rng.uniform(0.0, 2.0 * np.pi)

    def beat(p: np.ndarray) -> np.ndarray:
        return np.sin(p) + harmonic * np.sin(2.0 * p + lag)

    peak_to_peak = np.ptp(beat(np.linspace(0.0, 2.0 * np.pi, 256)))
    return rng.uniform(*config.heart_mm) * beat(phase) / peak_to_peak


def _posture_steps(count: int, fs: float, rng: np.random.Generator, config: SynthConfig) -> np.ndarray:
    """Small smooth shifts of the resting level: the body settling, too little for a labeller to call motion."""

    time_s = np.arange(count) / fs
    steps = np.zeros(count)
    for _ in range(rng.poisson(config.posture_steps_per_min * count / fs / 60.0)):
        onset_s = rng.uniform(0.0, count / fs)
        size = rng.uniform(*config.posture_step_mm) * rng.choice((-1.0, 1.0))
        steps += size * _smoothstep(np.clip((time_s - onset_s) / _POSTURE_STEP_S, 0.0, 1.0))
    return steps


def _motion_bursts(
    count: int, fs: float, rng: np.random.Generator, config: SynthConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Large movements in mm, and the mask of samples they make unusable."""

    motion = np.zeros(count)
    moving = np.zeros(count, dtype=bool)
    for _ in range(rng.poisson(config.motion_bursts_per_min * count / fs / 60.0)):
        n = max(2, int(round(rng.uniform(*config.motion_burst_s) * fs)))
        start = int(rng.integers(-n + 1, count))  # may run over either end of the recording
        size = rng.uniform(*config.motion_burst_mm)
        jerk = np.hanning(n) * gaussian_filter1d(rng.standard_normal(n), 0.15 * fs)
        # The body rarely comes back exactly where it was: part of the move stays as a new level.
        shift = size * rng.uniform(-0.5, 0.5)
        burst = size * jerk / (np.max(np.abs(jerk)) + 1e-12) + shift * _smoothstep(np.arange(n) / (n - 1))
        lo, hi = max(start, 0), min(start + n, count)
        motion[lo:hi] += burst[lo - start : hi - start]
        motion[start + n :] += shift
        moving[lo:hi] = True
    return motion, moving


def _complex_normal(rng: np.random.Generator, shape: int | tuple[int, ...]) -> np.ndarray:
    """Circular complex Gaussian samples of unit power."""

    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def a121_forward(
    chest_mm: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    *,
    snr_db: float,
    n_bins: int = 5,
    target_bin: int = 2,
    wavelength_mm: float = A121_WAVELENGTH_MM,
    distance_mm: float = 800.0,
    bin_spread: float = 1.0,
    static_clutter: float = 0.1,
    fading: float = 0.2,
) -> np.ndarray:
    """Complex Sparse IQ range bins ``[T, n_bins]`` an A121 would return for ``chest_mm``.

    The chest echo peaks in ``target_bin`` and spreads over its neighbours with a
    Gaussian profile ``bin_spread`` bins wide.  Its phase is
    ``-4π (distance_mm - chest_mm) / λ``: the round trip shortens as the chest
    rises toward the radar on an inhale.  Each bin also holds a static clutter
    phasor (random, RMS ``static_clutter`` × the peak echo) that bends the phase
    away from the displacement, and the echo fades slowly (``fading``: std of its
    log amplitude over seconds).  ``snr_db`` is the peak echo power over the
    complex noise power per frame.

    Read back by :func:`displacement_from_iq`, the defaults recover the chest
    almost exactly above ~25 dB and break down between 20 and 10 dB, as
    noise-dominated side bins pass the front end's amplitude rule.  Clutter near
    the echo's strength (``static_clutter`` ≳ 0.3) sends a bin's IQ circle close
    to the origin, where the phase races and slips at any SNR.
    """

    chest_mm = np.asarray(chest_mm, dtype=float)
    profile = np.exp(-0.5 * ((np.arange(n_bins) - target_bin) / bin_spread) ** 2)
    gain = np.exp(fading * _smooth_noise(len(chest_mm), fs, rng, smooth_s=2.0))
    echo = gain * np.exp(-4j * np.pi * (distance_mm - chest_mm) / wavelength_mm)
    clutter = static_clutter * _complex_normal(rng, n_bins)
    noise = 10.0 ** (-snr_db / 20.0) * _complex_normal(rng, (len(chest_mm), n_bins))
    return echo[:, None] * profile[None, :] + clutter[None, :] + noise


def displacement_from_iq(
    iq: np.ndarray,
    wavelength_mm: float = A121_WAVELENGTH_MM,
    *,
    strong_fraction: float = 0.25,
) -> np.ndarray:
    """Chest displacement in mm from complex range bins ``[T, n_bins]``, read as the real front end does.

    Bins whose median amplitude reaches ``strong_fraction`` of the strongest are
    kept, each one's phase unwrapped over time with its mean removed, and the
    bins averaged weighted by median amplitude; ``phase · λ / 4π`` then gives
    ``chest_mm``, sign included, up to an offset.

    Nothing here is made robust, on purpose: ``np.unwrap`` assumes less than λ/4
    (≈1.24 mm) of motion between frames, so noise at low SNR, or motion faster
    than λ/4 × fs (≈25 mm/s at 20 Hz), slips the phase by whole turns (steps of
    λ/2 in mm), as in real recordings, and the network gets to see those slips.
    """

    iq = np.asarray(iq)
    iq = iq.reshape(len(iq), -1)  # a single bin may come as [T]
    median_amplitude = np.median(np.abs(iq), axis=0)
    strong = median_amplitude >= strong_fraction * median_amplitude.max()
    phase = np.unwrap(np.angle(iq[:, strong]), axis=0)
    phase -= phase.mean(axis=0)
    weights = median_amplitude[strong] / median_amplitude[strong].sum()
    return phase @ weights * wavelength_mm / (4.0 * np.pi)


def scale_depth(x: np.ndarray, factor: float) -> np.ndarray:
    """Breaths ``factor`` times as deep; ~0.05-1 turns coached breaths into sleep-like ones.

    Everything in ``x`` scales, noise included, so scale clean breathing before
    adding noise when the noise level should stay put.
    """

    return np.asarray(x, dtype=float) * float(factor)


def time_warp(x: np.ndarray, labels: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Play a recording ``factor`` times slower (> 1) or faster (< 1) at the same sample rate.

    Every phase lasts ``factor`` times longer, so ``T`` samples become
    ``round(T * factor)``.  Signals are interpolated linearly and labels take the
    nearest input sample, both read at the same positions so they stay aligned.
    Heartbeat and noise in ``x`` are warped too; warp clean breathing first when
    they should keep their rates.
    """

    x, labels = _aligned(x, labels)
    if factor <= 0:
        raise ValueError(f"time_warp factor must be positive, got {factor}")
    n = len(labels)
    # Sample centres map to sample centres, so a run of k samples becomes
    # ~k·factor samples wherever it sits in the recording.
    position = np.clip((np.arange(max(1, round(n * factor))) + 0.5) / factor - 0.5, 0.0, n - 1)
    left = np.floor(position).astype(int)
    right = np.minimum(left + 1, n - 1)
    weight = position - left
    warped = x[..., left] * (1.0 - weight) + x[..., right] * weight
    return warped, labels[np.rint(position).astype(int)]


def add_noise(x: np.ndarray, noise: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """``x + scale · noise``; noise of shape ``[T]`` goes into every channel of a ``[C, T]`` signal."""

    x = np.asarray(x, dtype=float)
    noise = np.asarray(noise, dtype=float)
    if noise.shape[-1] != x.shape[-1]:
        raise ValueError(f"noise has {noise.shape[-1]} samples, the signal {x.shape[-1]}: fit it with NoiseBank.sample")
    return x + float(scale) * noise


class NoiseBank:
    """Recorded noise (e.g. empty-scene or breath-hold displacement) to add to synthetic breathing.

    Each segment is ``[T]`` or ``[C, T]`` with time last.  Its mean is removed:
    a recording's absolute level is arbitrary and would only shift the breathing.
    """

    def __init__(self, segments: Iterable[np.ndarray]) -> None:
        self.segments = tuple(
            segment - segment.mean(axis=-1, keepdims=True)
            for segment in (np.asarray(s, dtype=float) for s in segments)
            if segment.shape[-1] > 0
        )
        if not self.segments:
            raise ValueError("NoiseBank needs at least one non-empty noise segment")
        lengths = np.array([segment.shape[-1] for segment in self.segments], dtype=float)
        self._weights = lengths / lengths.sum()

    def sample(self, length: int, rng: np.random.Generator) -> np.ndarray:
        """A random contiguous ``length``-sample stretch of recorded noise.

        Segments are picked in proportion to their length, so every recorded
        sample is equally likely.  A segment shorter than ``length`` is tiled
        forwards and backwards in turn, which keeps the joins continuous and the
        spectrum unchanged, and the stretch starts at a random point of it.
        """

        segment = self.segments[int(rng.choice(len(self.segments), p=self._weights))]
        n = segment.shape[-1]
        if length <= n:
            start = int(rng.integers(n - length + 1))
            return segment[..., start : start + length].copy()
        start = int(rng.integers(n))
        copies = -(-(start + length) // n)
        tiled = np.concatenate([segment if k % 2 == 0 else segment[..., ::-1] for k in range(copies)], axis=-1)
        return tiled[..., start : start + length]


def add_drift(x: np.ndarray, fs: float, rng: np.random.Generator, max_mm_per_min: float) -> np.ndarray:
    """``x`` plus a slow random wander of the baseline, never steeper than ``max_mm_per_min``.

    A random walk smoothed over ~5 s stays far below the slowest breathing (5/min,
    0.08 Hz), so it moves the level without passing for a breath; its steepest
    slope is drawn uniformly from 0 to ``max_mm_per_min``.  All channels get the
    same drift, since the chest moves as a whole.
    """

    x = np.asarray(x, dtype=float)
    count = x.shape[-1]
    if count < 2 or max_mm_per_min <= 0:
        return x.copy()
    walk = gaussian_filter1d(np.cumsum(rng.standard_normal(count)), _DRIFT_SMOOTH_S * fs, mode="nearest")
    steepest = np.max(np.abs(np.gradient(walk))) * fs * 60.0
    return x + (walk - walk[0]) * rng.uniform(0.0, max_mm_per_min) / (steepest + 1e-12)


def random_crop(
    x: np.ndarray, labels: np.ndarray, length: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A random ``length``-sample window of ``x`` and ``labels``.

    Shorter inputs are placed at a random offset and padded, the signal with its
    edge values and the labels with IGNORE, so the padding carries no loss.
    """

    x, labels = _aligned(x, labels)
    n = len(labels)
    if n >= length:
        start = int(rng.integers(n - length + 1))
        return x[..., start : start + length].copy(), labels[start : start + length].copy()
    before = int(rng.integers(length - n + 1))
    after = length - n - before
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(before, after)], mode="edge")
    return padded, np.pad(labels, (before, after), constant_values=IGNORE)


def _aligned(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    if labels.ndim != 1 or x.ndim not in (1, 2) or x.shape[-1] != len(labels) or not len(labels):
        raise ValueError(f"Expected a signal [T] or [C, T] and labels [T], got {x.shape} and {labels.shape}")
    return x, labels
