"""Artificial labelled runs: augmented copies of real recordings and synthetic A121 runs.

Both come out as :class:`~respi_net.nn_dataset.LabelledRun`, so they are windowed,
normalised and exported exactly like the real recordings.

* :func:`augment_run` plays a real run faster or slower (breathing rate changes,
  labels stretch with it), adds real radar noise at a chosen size relative to
  the breathing, and a slow baseline drift.  Depth itself is not varied: the
  network input is scaled by the breath size anyway, so what matters is how big
  the noise is next to the breathing.
* :func:`synthetic_run` simulates breathing with every phase known
  (:func:`respi_net.breath_synth.simulate_breathing`) and reads it through the
  A121 forward model, so the radar front end's own failures are in the data.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend

from .breath_phases import HOLD_AFTER_EXHALE, HOLD_AFTER_INHALE, label_runs
from .breath_synth import (
    NoiseBank,
    SynthConfig,
    a121_forward,
    add_drift,
    add_noise,
    displacement_from_iq,
    simulate_breathing,
    time_warp,
)
from .chest_signal import ChestSignal, light_filter
from .nn_dataset import LabelledRun, labelled_run

# White noise comes from the A121 model instead, which adds its own per frame.
SYNTH_CONFIG = SynthConfig(white_noise_mm=(0.0, 0.0))


def breath_size(chest: np.ndarray) -> float:
    """Robust breath depth: the 5th-95th percentile spread, as in the network's normalisation."""

    spread = float(np.percentile(chest, 95) - np.percentile(chest, 5))
    return spread if spread > 0 else 1.0


def noise_bank_from_holds(runs: Iterable[LabelledRun], *, min_s: float = 5.0) -> NoiseBank:
    """Real radar noise: the chest signal during labelled holds of at least ``min_s``.

    A held breath leaves what the radar sees anyway -- heartbeat, small body
    motion, sensor noise.  Each segment is straightened (linear trend removed),
    because drift is added separately.
    """

    segments = [
        detrend(run.features[0][start:stop].astype(float))
        for run in runs
        for start, stop, label in label_runs(run.labels)
        if label in (HOLD_AFTER_EXHALE, HOLD_AFTER_INHALE) and stop - start >= min_s * run.fs
    ]
    return NoiseBank(segments)


def augment_run(
    run: LabelledRun,
    stretch: float,
    noise_ratio: float,
    bank: NoiseBank,
    rng: np.random.Generator,
    *,
    drift_per_min: float = 0.3,
) -> LabelledRun:
    """``run`` played ``stretch`` times slower (> 1) or faster (< 1), with noise and drift.

    ``noise_ratio`` is the noise standard deviation as a fraction of the breath
    size (:func:`breath_size`), ``drift_per_min`` the steepest drift in breath
    sizes per minute.  The copy keeps the run's group and subject, so it always
    lands in the same split as the original.
    """

    warped, labels = time_warp(run.features[[0, 2]], run.labels, stretch)
    chest, echo = warped
    size = breath_size(chest)
    noise = bank.sample(len(labels), rng)
    chest = add_noise(chest, noise, noise_ratio * size / (float(np.std(noise)) + 1e-12))
    chest = add_drift(chest, run.fs, rng, drift_per_min * size)
    signal = ChestSignal(
        source=f"{run.meta.get('source', 'real')}+augmented",
        time_s=np.arange(len(labels)) / run.fs,
        fs=run.fs,
        chest=chest,
        units=str(run.meta.get("units", "")),
        raw=chest,
        echo_db=echo,
    )
    meta = {key: value for key, value in run.meta.items() if key not in ("source", "units")}
    meta.update({"augmented_from": run.run_id, "stretch": float(stretch), "noise_ratio": float(noise_ratio)})
    return labelled_run(f"{run.run_id}__x{stretch:g}", signal, labels, group=run.group, subject=run.subject, meta=meta)


def synthetic_run(
    index: int,
    duration_s: float,
    fs: float,
    rng: np.random.Generator,
    *,
    snr_db: tuple[float, float] = (12.0, 40.0),
    config: SynthConfig = SYNTH_CONFIG,
) -> LabelledRun:
    """One simulated A121 recording with perfect labels.

    SNR, static clutter and fading are drawn per run; below ~20 dB the front
    end starts to slip, as real recordings do.  The echo channel is the
    strongest range bin's amplitude in dB.
    """

    simulated = simulate_breathing(duration_s, fs, rng, config)
    snr = float(rng.uniform(*snr_db))
    iq = a121_forward(
        simulated.chest_mm,
        fs,
        rng,
        snr_db=snr,
        static_clutter=float(rng.uniform(0.02, 0.25)),
        fading=float(rng.uniform(0.05, 0.3)),
    )
    chest = displacement_from_iq(iq)
    echo_db = 20.0 * np.log10(np.abs(iq).max(axis=1) + 1e-9)
    signal = ChestSignal("synthetic_a121", simulated.time_s, fs, chest, "mm", chest, echo_db, {"snr_db": snr})
    run_id = f"synthetic_{index:04d}"
    return labelled_run(run_id, signal, simulated.labels, group=run_id, subject="synthetic", meta={"dataset": "synthetic"})


def enhanced_augment_run(
    run: LabelledRun,
    stretch: float,
    noise_ratio: float,
    bank: NoiseBank,
    rng: np.random.Generator,
    *,
    drift_per_min: float,
) -> LabelledRun:
    """Existing augmentation plus a time-varying extra hold-noise component.

    Labels are only time-warped, never re-detected from the corrupted signal.
    Their IGNORE regions and source uncertainty are inherited unchanged.
    """
    copy = augment_run(run, stretch, noise_ratio, bank, rng, drift_per_min=drift_per_min)
    noise = bank.sample(len(copy.labels), rng)
    envelope = gaussian_filter1d(rng.standard_normal(len(noise)), 3 * run.fs)
    envelope = .5 + (envelope - envelope.min()) / (np.ptp(envelope) + 1e-12)
    chest = copy.features[0] + noise * envelope * (.5 * noise_ratio * breath_size(copy.features[0]) / (np.std(noise) + 1e-12))
    signal = ChestSignal(f"{run.meta.get('source', 'real')}+augmented", copy.time_s, copy.fs, chest,
                         str(run.meta.get("units", "")), chest, copy.features[2])
    meta = {**copy.meta, "drift_per_min": drift_per_min,
            "extra_noise_envelope": "0.5–1.5, smoothed over 3 s",
            "label_origin": "source labels; inherited uncertainty; no re-labelling"}
    return labelled_run(copy.run_id, signal, copy.labels, group=run.group, subject=run.subject, meta=meta)


def reduced_power_run(run: LabelledRun, drop_db: float, rng: np.random.Generator) -> LabelledRun:
    """Replay a measured mm displacement through A121 with weaker reflected power.

    Receiver noise stays fixed; SNR is 35 minus drop_db. Arbitrary belt counts
    are not physical millimetres, so this method is restricted to mm signals.
    Labels, times, masks and group come from the original, not the IQ output.
    """
    if run.meta.get("units") != "mm" or not 0 <= drop_db <= 35:
        raise ValueError("Reduced-power replay requires mm displacement and 0–35 dB drop")
    snr = 35.0 - drop_db
    iq = a121_forward(run.features[0], run.fs, rng, snr_db=snr, static_clutter=.08, fading=.15)
    iq *= 10 ** (-drop_db / 20)
    chest = light_filter(displacement_from_iq(iq), run.fs, 2.0)
    echo = 20 * np.log10(np.maximum(np.abs(iq).max(axis=1), 1e-9))
    signal = ChestSignal("empirical_trace+simulated_a121", run.time_s, run.fs, chest, "mm", chest, echo)
    meta = {"augmented_from": run.run_id, "power_drop_db": drop_db, "snr_db": snr,
            "static_clutter": .08, "fading": .15,
            "label_origin": "source labels; inherited uncertainty; no re-labelling",
            "echo_available": True, "echo_units": "relative model dB, not calibrated measured echo"}
    return labelled_run(f"{run.run_id}__iq_drop{drop_db:g}", signal, run.labels.copy(),
                        group=run.group, subject=run.subject, meta=meta)
