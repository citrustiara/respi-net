#!/usr/bin/env python3
"""Compare static-noise metrics of a simultaneous LSM6DS3 and iPhone capture.

The inputs are produced by ``imu_lsm6ds3_iphone_static_capture.py``.  Results
describe the exposed acquisition paths (sensor, firmware/app and transport),
not a laboratory calibration of the two physical MEMS chips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, periodogram, sosfiltfilt


ACCEL_COLUMNS = ["ax", "ay", "az"]
GYRO_COLUMNS = ["gx", "gy", "gz"]
BANDS_HZ = {"respiration": (0.07, 0.60), "cardiac": (0.80, 3.00)}


def _sample_rate_hz(frame: pd.DataFrame) -> float:
    time_s = frame["Time_ms"].to_numpy(dtype=float) / 1000.0
    differences = np.diff(time_s)
    valid = differences[np.isfinite(differences) & (differences > 0)]
    if not len(valid):
        raise ValueError("No usable time differences in IMU CSV.")
    return float(1.0 / np.median(valid))


def _first_pca(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return centered @ vectors[0]


def _vector_rms(values: np.ndarray) -> float:
    centered = detrend(values, axis=0, type="linear")
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _band_pca_rms(values: np.ndarray, fs_hz: float, band_hz: tuple[float, float]) -> float:
    sos = butter(4, band_hz, btype="bandpass", fs=fs_hz, output="sos")
    filtered = np.column_stack([sosfiltfilt(sos, values[:, axis]) for axis in range(3)])
    return float(np.std(_first_pca(filtered)))


def _metrics(frame: pd.DataFrame, device: str) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray, float]:
    fs_hz = _sample_rate_hz(frame)
    acceleration = frame[ACCEL_COLUMNS].to_numpy(dtype=float)
    angular_rate = frame[GYRO_COLUMNS].to_numpy(dtype=float)
    acceleration_pca = _first_pca(detrend(acceleration, axis=0, type="linear"))
    frequencies, power = periodogram(acceleration_pca, fs=fs_hz, window="hann", scaling="density")
    result: dict[str, float | int | str] = {
        "device": device,
        "samples": int(len(frame)),
        "sample_rate_hz": fs_hz,
        "accel_vector_rms_mg": 1000.0 * _vector_rms(acceleration),
        "gyro_vector_rms_dps": _vector_rms(angular_rate),
    }
    for label, band_hz in BANDS_HZ.items():
        result[f"{label}_pca_rms_mg"] = 1000.0 * _band_pca_rms(acceleration, fs_hz, band_hz)
    return result, frequencies, power, fs_hz


def _plot_spectra(results: list[tuple[dict[str, float | int | str], np.ndarray, np.ndarray, float]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 4.7), constrained_layout=True)
    colours = {"LSM6DS3": "#1d4ed8", "iPhone": "#c2410c"}
    for metrics, frequencies, power, _fs_hz in results:
        axis.semilogy(frequencies, power, color=colours[str(metrics["device"])], lw=1.0, label=str(metrics["device"]))
    axis.axvspan(*BANDS_HZ["respiration"], color="#bfdbfe", alpha=0.45, label="pasmo oddechu")
    axis.axvspan(*BANDS_HZ["cardiac"], color="#fecaca", alpha=0.45, label="pasmo tętna")
    axis.set_xlim(0.0, 5.0)
    axis.set_xlabel("Częstotliwość [Hz]")
    axis.set_ylabel("Gęstość widmowa PCA [g²/Hz]")
    axis.set_title("Zapis nieruchomy: widmo PCA przyspieszeń")
    axis.grid(alpha=0.25)
    axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "accepted":
        raise SystemExit("Pomiar statyczny nie został zaakceptowany.")
    files = {name: Path(path) for name, path in manifest["files"].items()}
    lsm_metrics, lsm_frequencies, lsm_power, lsm_fs = _metrics(pd.read_csv(files["lsm6ds3"]), "LSM6DS3")
    iphone_metrics, iphone_frequencies, iphone_power, iphone_fs = _metrics(pd.read_csv(files["iphone"]), "iPhone")
    metrics = pd.DataFrame([lsm_metrics, iphone_metrics])
    metrics_path = args.metrics or session_dir / "analysis_static_imu_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    if args.figure:
        _plot_spectra(
            [(lsm_metrics, lsm_frequencies, lsm_power, lsm_fs), (iphone_metrics, iphone_frequencies, iphone_power, iphone_fs)],
            args.figure,
        )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print(f"\nMetryki: {metrics_path}")
    if args.figure:
        print(f"Wykres: {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
