#!/usr/bin/env python
"""The sleep heart-rate network on another 60 GHz radar: TI IWR6843AOP recordings with a finger-PPG pulse rate.

Barvik and Hruby (Zenodo 23762780, CC BY 4.0): raw DCA1000 captures of people seated 1-2 m from the radar, facing
it or sideways, breathing paced, with a SOMNOtouch pulse rate at 4 Hz.  Used: the single-person seated recordings
with a single transmitter (chirp configurations A, B, E; 20 recordings, 3 people's worth of data, mostly one).

Each capture is decoded (16-bit complex samples, two LVDS lanes: I0 I1 Q0 Q1 ...), range-FFT'd per chirp, the first receiver
averaged over the chirps of a frame, and down-sampled to 20 Hz.  The bin that breathes most between 0.4
and 2.6 m and its 6 neighbours give displacement (unwrapped phase x wavelength / 4 pi) -- the input the network
takes from the A121.  Windows of 60 s every 10 s (edges padded for the network's stretch margin); label: the median
pulse rate over the window.

Scored: the final networks trained on the author's watch nights (``tools/train_sleep_heart_rate.py --final``,
without and with radar+ECG pretraining, 3 seeds averaged) and the plain spectral peak of the best bin.

    uv run --extra ml python tools/hr_60ghz_test.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import torch

from train_sleep_heart_rate import BINS, OUT_DIR as HR_DIR, PAD, WINDOW, make_model, predict

DATA = ROOT / "data" / "external" / "barvik_60ghz_ppg"
OUT = ROOT / "runs" / "hr_60ghz"
FS = 20.0
WAVELENGTH_MM = 299.792458 / 60.25
CONFIG = {  # samples per chirp, chirps per frame (set 1 / set 2), range resolution m, frame rate
    "A": (256, {"set1": 5, "set2": 50}, 0.065, 100.0),
    "B": (80, {"set1": 4, "set2": 4}, 0.050, 100.0),
    "E": (256, {"set1": 10, "set2": 10}, 0.070, 100.0),
}
RX = 4


def decode(raw: np.ndarray, samples: int, chirps: int) -> np.ndarray:
    """[frames, bins] range profile (complex) of the first receiver, averaged over the chirps of a frame.

    The receivers see the chest with different path phases: averaging them coherently partly cancels the cardiac
    motion (spectral-peak error 16 bpm with all four, 8 with one on S1-R32)."""

    lanes = np.zeros(len(raw) // 2, complex)
    lanes[0::2] = raw[0::4] + 1j * raw[2::4]
    lanes[1::2] = raw[1::4] + 1j * raw[3::4]
    per_frame = samples * chirps * RX
    frames = len(lanes) // per_frame
    cube = lanes[: frames * per_frame].reshape(frames, chirps, RX, samples)
    profile = np.fft.fft((cube - cube.mean(-1, keepdims=True)) * np.hanning(samples), axis=-1)
    return profile[:, :, 0].mean(axis=1)


def recording_windows(row: pd.Series) -> tuple[np.ndarray, np.ndarray, list[float]]:
    samples, chirps, resolution, fps = CONFIG[row["config"]]
    chirps = chirps[row["set"]]
    with zipfile.ZipFile(DATA / f"{row['recording_id']}.zip") as archive:
        bins = sorted(n for n in archive.namelist() if "_Raw_" in n and n.endswith(".bin"))
        raw = np.concatenate([np.frombuffer(archive.read(n), dtype="<i2") for n in bins])
    profile = decode(raw, samples, chirps)
    step = int(round(fps / FS))
    profile = profile[: len(profile) // step * step].reshape(-1, step, profile.shape[1]).mean(1)
    distance = np.arange(profile.shape[1]) * resolution
    phase = np.unwrap(np.angle(profile), axis=0)
    spectrum = np.abs(np.fft.rfft(phase - phase.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(phase), 1 / FS)
    power = spectrum[(f > 0.1) & (f < 0.6)].sum(0)
    power[(distance < 0.4) | (distance > 2.6)] = 0
    centre = int(np.clip(np.argmax(power), BINS // 2, profile.shape[1] - 1 - BINS // 2))
    disp = (phase[:, centre - BINS // 2 : centre + BINS // 2 + 1] * WAVELENGTH_MM / (4 * np.pi)).T  # [BINS, n]
    disp = np.pad(disp - disp.mean(1, keepdims=True), ((0, 0), (PAD, PAD)), mode="edge")
    starts = list(np.arange(0, disp.shape[1] - (WINDOW + 2 * PAD) + 1, int(10 * FS)))
    xs = np.stack([disp[:, s : s + WINDOW + 2 * PAD] for s in starts]).astype(np.float32) if starts else np.zeros((0, BINS, WINDOW + 2 * PAD), np.float32)
    centres = [(s + PAD + WINDOW / 2) / FS - PAD / FS for s in starts]  # seconds from the recording start
    return xs, np.array(centres), [float(distance[centre])]


def pulse_series() -> pd.DataFrame:
    frames = []
    for name in ("reference_set1.zip", "reference_set2.zip"):
        with zipfile.ZipFile(DATA / name) as archive:
            for member in [n for n in archive.namelist() if n.endswith("Pulse.txt")]:
                text = archive.read(member).decode("latin-1").splitlines()
                date = next(l for l in text if l.startswith("Start Time")).split(":", 1)[1].strip().split()[0]
                rows = [l.split(";") for l in text[text.index("Data:") + 1 :] if ";" in l]
                # set 1 rows hold the time of day, set 2 rows the date and time
                stamps = [t.strip().replace(",", ".") for t, _ in rows]
                times = pd.to_datetime([x if " " in x else f"{date} {x}" for x in stamps], format="%d.%m.%Y %H:%M:%S.%f")
                frames.append(pd.DataFrame({"time": times, "pulse": [float(v) for _, v in rows]}))
    out = pd.concat(frames).sort_values("time")
    return out[out["pulse"] > 0]


def spectral_peak(x: np.ndarray) -> np.ndarray:
    centre = x[:, BINS // 2, PAD : PAD + WINDOW]
    f = np.fft.rfftfreq(WINDOW, 1 / FS)
    band = (f >= 0.7) & (f <= 2.0)
    p = np.abs(np.fft.rfft((centre - centre.mean(1, keepdims=True)) * np.hanning(WINDOW), axis=1)) ** 2
    return f[band][p[:, band].argmax(1)] * 60.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    meta = pd.read_csv(DATA / "recordings.csv")
    use = meta[meta["config"].isin(CONFIG) & meta["activity"].isin(["seated"]) & (meta["persons_in_scene"] == 1) & (meta["duration_s"] >= 60)]
    pulse = pulse_series()
    xs, ys, rows = [], [], []
    for _, row in use.iterrows():
        x, centres, dist = recording_windows(row)
        start = pd.Timestamp(f"{row['date']} {row['start_time_local']}")
        labels = []
        for c in centres:
            t = start + pd.Timedelta(seconds=float(c))
            near = pulse[(pulse["time"] >= t - pd.Timedelta(seconds=30)) & (pulse["time"] <= t + pd.Timedelta(seconds=30))]["pulse"]
            labels.append(float(near.median()) if len(near) > 100 else math.nan)
        labels = np.array(labels)
        keep = np.isfinite(labels)
        xs.append(x[keep]); ys.append(labels[keep])
        rows.append({"recording": row["recording_id"], "config": row["config"], "position": row["protocol_position"], "participant": row["participants"],
                     "chosen_bin_m": dist[0], "windows": int(keep.sum())})
        print(f"{row['recording_id']} cfg {row['config']} {row['protocol_position']}: {keep.sum()} windows, bin {dist[0]:.2f} m, pulse {np.nanmedian(labels):.0f}", flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    device = torch.device("cpu")
    result = {"windows": int(len(y)), "pulse_median": float(np.median(y)), "recordings": rows,
              "spectral_peak_mae": float(np.mean(np.abs(spectral_peak(x) - y)))}
    for variant in ("shared", "shared-pre"):
        preds = []
        for seed in (0, 1, 2):
            path = HR_DIR / f"final_{variant}_s{seed}.pt"
            if path.exists():
                model = make_model("shared"); model.load_state_dict(torch.load(path, map_location="cpu")); model.eval()
                preds.append(predict(model, x, device)[0])
        if preds:
            p = np.mean(preds, axis=0)
            result[variant] = {"mae": float(np.mean(np.abs(p - y))), "bias": float(np.mean(p - y)),
                               "within_5_bpm": float(np.mean(np.abs(p - y) <= 5))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "recordings"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
