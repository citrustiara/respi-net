#!/usr/bin/env python
"""60 s windows of chest motion from a 24 GHz radar with the ECG's heart rate (Schellenberger et al. 2020), for
pretraining the sleep heart-rate network.

Each recording (``data/external/schellenberger2020/GDN00xx/*.mat``: 30 healthy people lying down, resting,
Valsalva, tilt up and down; radar I/Q and ECG at 2000 Hz; CC BY 4.0) becomes windows like the A121 ones of
``tools/train_sleep_heart_rate.py``: displacement in mm at 20 Hz, 60 s plus 6 s padding each side, every 10 s.

The radar's I/Q is demodulated per window: a least-squares circle through the I/Q points gives the centre the
phase turns around (a continuous-wave receiver adds offsets), the unwrapped angle around it times
wavelength / 4 pi is the displacement (24 GHz: 12.5 mm).  The label is the mean heart rate over the 60 s from the
ECG's R peaks (:func:`respi_net.ecg.detect_r_peaks`, valid intervals only, :func:`respi_net.ecg.hr_labels_for_windows`).

    uv run python tools/build_radar_ecg_windows.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

from respi_net.ecg import detect_r_peaks, hr_labels_for_windows

SOURCE = ROOT / "data" / "external" / "schellenberger2020"
OUT = ROOT / "data" / "processed" / "radar_ecg" / "schellenberger_windows.npz"
WAVELENGTH_MM = 299.792458 / 24.0
FS = 20.0
WINDOW_S, PAD_S, STEP_S = 60.0, 6.0, 10.0


def circle_centre(i: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Least-squares (Kasa) circle centre of the I/Q points."""

    a = np.column_stack([i, q, np.ones_like(i)])
    b = i**2 + q**2
    (ci2, cq2, _), *_ = np.linalg.lstsq(a, b, rcond=None)
    return ci2 / 2.0, cq2 / 2.0


def displacement_mm(i: np.ndarray, q: np.ndarray) -> np.ndarray:
    ci, cq = circle_centre(i, q)
    angle = np.unwrap(np.arctan2(q - cq, i - ci))
    if not np.all(np.isfinite(angle)):  # degenerate fit: fall back to the plain angle
        angle = np.unwrap(np.arctan2(q - q.mean(), i - i.mean()))
    return angle * WAVELENGTH_MM / (4.0 * np.pi)


def recording_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    m = sio.loadmat(path)
    fs_radar, fs_ecg = float(m["fs_radar"].squeeze()), float(m["fs_ecg"].squeeze())
    i, q = m["radar_i"].ravel(), m["radar_q"].ravel()
    leads = [m[k].ravel().astype(float) for k in ("tfm_ecg2", "tfm_ecg1") if k in m]
    ecg = min(leads, key=lambda lead: int(np.isnan(lead).sum()))  # the lead with fewer gaps
    gaps = ~np.isfinite(ecg)
    if gaps.mean() > 0.2:
        return np.zeros((0, 1, int((WINDOW_S + 2 * PAD_S) * FS)), np.float32), np.zeros(0)
    if gaps.any():  # short dropouts: bridged; windows with too few valid beats lose their label anyway
        ecg[gaps] = np.interp(np.flatnonzero(gaps), np.flatnonzero(~gaps), ecg[~gaps])
    peaks = detect_r_peaks(ecg, fs_ecg) / fs_ecg
    if len(peaks) < 20:
        return np.zeros((0, 1, int((WINDOW_S + 2 * PAD_S) * FS)), np.float32), np.zeros(0)
    duration = len(i) / fs_radar
    starts = np.arange(PAD_S, duration - WINDOW_S - PAD_S, STEP_S)
    labels = hr_labels_for_windows(peaks, starts, WINDOW_S)
    decim = int(round(fs_radar / FS))
    xs, ys = [], []
    for start, label in zip(starts, labels):
        if not np.isfinite(label):
            continue
        lo, hi = int((start - PAD_S) * fs_radar), int((start + WINDOW_S + PAD_S) * fs_radar)
        disp = displacement_mm(i[lo:hi], q[lo:hi])
        xs.append(resample_poly(disp - disp.mean(), 1, decim).astype(np.float32)[None, : int((WINDOW_S + 2 * PAD_S) * FS)])
        ys.append(label)
    width = int((WINDOW_S + 2 * PAD_S) * FS)
    xs = [x for x in xs if x.shape[-1] == width]
    return (np.stack(xs) if xs else np.zeros((0, 1, width), np.float32)), np.asarray(ys[: len(xs)], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    xs, ys, subjects, scenarios = [], [], [], []
    for path in sorted(SOURCE.glob("GDN*/GDN*_*.mat")):
        x, y = recording_windows(path)
        subject, scenario = path.stem.split("_")[0], path.stem.split("_", 2)[-1]
        xs.append(x); ys.append(y); subjects += [subject] * len(y); scenarios += [scenario] * len(y)
        print(f"{path.stem}: {len(y)} windows, HR {np.median(y) if len(y) else float('nan'):.0f} bpm", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, x=np.concatenate(xs), label=np.concatenate(ys), subject=np.array(subjects), scenario=np.array(scenarios))
    print(f"wrote {args.out}: {sum(len(y) for y in ys)} windows from {len(set(subjects))} people")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
