#!/usr/bin/env python
"""A network that reads the heart rate off 60 s of radar during sleep, against the night pipeline and the watch.

Input: the displacement of the 7 range bins around Acconeer's target (``tools/extract_a121_night.py``), 60 s at
20 Hz, each bin unwrapped and scaled.  Output: a distribution over 36-120 bpm in 1 bpm steps -- the heart rate is
its peak and the confidence the peak's height, so the network can decline a window like the night pipeline does.
Targets are soft (a Gaussian of 2 bpm around the label: the watch itself is a few bpm off).

Labels: the Garmin watch's heart rate at the window centre (interpolated between its readings, only within 2 min of
one) on the nights with a watch export, and the chest strap's median heart rate (with good contact) on the strap
sessions (readings that pass its quality check).  Only still windows count (no movement or absence by the night pipeline's rules).

Two architectures:

* ``time`` -- a 1D convolutional network on the band-passed signals (0.5-3 Hz), pooled over the window;
* ``spectral`` -- a fully convolutional network over the power spectrum of each bin (1 bpm resolution,
  30-150 bpm, so a heartbeat's second harmonic is in view), one output per bpm: a learned version of the pipeline's
  spectral peak search.

and the ablation ``<arch>-nostretch``: without time-stretch augmentation (a window resampled 0.9-1.1x with its
label scaled by the same factor; it widens the narrow sleep heart rate the network sees and works against simply
predicting the average).

Each watch night is held out once as test and the others train; the last 20% of every training night (after a 60 s
gap, so no window overlaps the training part) validates -- best epoch by validation MAE.  A whole night as the
validation set rewards a network that outputs that night's typical heart rate, whatever the radar says.  3 seeds.  On the test night the network is compared on the night pipeline's windows with the pipeline's own heart
rate (``hr_bpm``) and with the training nights' median heart rate.  The strap sessions are never trained on: every
fold's model is scored on them.

    uv run --extra ml python tools/train_sleep_heart_rate.py
    uv run --extra ml python tools/train_sleep_heart_rate.py --summary-only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import torch
from torch import nn

from build_sleep_epochs import GARMIN_DIR, NIGHT_DIR, NIGHTS
from respi_net.a121_sleep import load_garmin_reference
from respi_net.chest_signal import A121_CHEST_SIGN, PHASE_TO_MM, regular_frame_times_ms

OUT_DIR = ROOT / "runs" / "sleep_heart_rate"
STRAP_DIR = ROOT / "data" / "raw" / "heart_strap"
HR_MIN, HR_MAX = 36, 120
CLASSES = HR_MAX - HR_MIN + 1
SPEC_MIN, SPEC_MAX = 30, 150
BINS = 7
FS = 20.0
WINDOW = int(60 * FS)
PAD = int(6 * FS)  # extra samples on each side for the time stretch
STRETCH = (0.9, 1.1)
TRAIN_STEP_S = 10.0
LABEL_SIGMA = 2.0
# Radar recordings made with the strap on (same Mac, same clock).  The 20:16 session is left out: its strap pieces
# fail the quality check (0% of readings in three of four pieces, the last one at an implausible 122 bpm while the
# radar shows about 51 bpm).
STRAP_SESSIONS = (
    "a121_sleep_radar-pas-hr-2m-2026-11-07_2026-11-07_18-43-04",
    "a121_sleep_radar-belt-hr-alt-2026-11-07_2026-11-07_19-26-08",
)
STRAP_MIN_OK_SHARE = 0.5  # a strap piece counts only when at least half of its readings pass the quality check
ARCHS = ("time", "spectral", "shared")
VARIANTS = ("time", "spectral", "time-nostretch", "spectral-nostretch")
# ``shared``: the time network applied to every bin with the same weights and pooled over bins, so it takes any number
# of bins -- the 7 of the A121 and the single channel of a continuous-wave radar.  ``shared-pre`` first learns on the
# 24 GHz radar windows with ECG heart rate of 30 people (``tools/build_radar_ecg_windows.py``), then fine-tunes on the
# nights; ``shared-zeroshot`` is that pretrained network on the nights as it is.
PRETRAIN_VARIANTS = ("shared", "shared-pre", "shared-zeroshot")
PRETRAIN_WINDOWS = ROOT / "data" / "processed" / "radar_ecg" / "schellenberger_windows.npz"


# ------------------------------------------------------------------- data


@dataclass
class Windows:
    """Windows of one recording: raw displacement with padding [n, BINS, WINDOW + 2 PAD], labels and metadata."""

    x: np.ndarray
    label: np.ndarray  # bpm, NaN where none
    centre_ms: np.ndarray  # regularised time of the window centre
    end_s: np.ndarray  # raw elapsed time of the window end (the night pipeline's clock)
    source: str


def _displacement(iq: np.ndarray, distances: np.ndarray, target: np.ndarray) -> np.ndarray:
    """[n, BINS] displacement in mm of the bins around the median target, each unwrapped."""

    complex_iq = iq[..., 0].astype(np.float32) + 1j * iq[..., 1].astype(np.float32)
    finite = target[np.isfinite(target) & (target > 0)]
    if len(finite):
        centre = int(np.argmin(np.abs(distances - float(np.median(finite)))))
    else:
        amplitude = np.median(np.abs(complex_iq), axis=0)
        centre = int(np.argmax(np.where(distances >= 0.2, amplitude, -np.inf)))
    centre = int(np.clip(centre, BINS // 2, len(distances) - 1 - BINS // 2))
    gate = np.arange(centre - BINS // 2, centre + BINS // 2 + 1)
    return A121_CHEST_SIGN * np.unwrap(np.angle(complex_iq[:, gate]), axis=0) * PHASE_TO_MM


def recording_windows(stem: str, ends_s: np.ndarray, still: np.ndarray, night_dir: Path = NIGHT_DIR) -> Windows:
    """Padded windows ending at ``ends_s`` (raw elapsed seconds) where ``still``; others are dropped."""

    with np.load(night_dir / f"{stem}.npz") as raw:
        stamps, distances, iq, target = raw["timestamp_ms"], raw["distances_m"], raw["iq"], raw["acconeer_target_m"]
    t_s = (stamps - stamps[0]) / 1000.0
    regular = regular_frame_times_ms(stamps)
    xs, centres, ends = [], [], []
    for end, ok in zip(ends_s, still):
        if not ok:
            continue
        hi = int(np.searchsorted(t_s, end)) + PAD
        lo = hi - WINDOW - 2 * PAD
        if lo < 0 or hi > len(t_s):
            continue
        xs.append(_displacement(iq[lo:hi], distances, target[lo:hi]).T.astype(np.float32))
        centres.append(regular[lo + PAD + WINDOW // 2])
        ends.append(end)
    x = np.stack(xs) if xs else np.zeros((0, BINS, WINDOW + 2 * PAD), np.float32)
    return Windows(x, np.full(len(xs), np.nan), np.asarray(centres), np.asarray(ends), stem)


def still_mask(trend: pd.DataFrame) -> np.ndarray:
    return ((trend["still_fraction"] >= 0.65) & (trend["no_presence_fraction"] <= 0.05)).to_numpy()


def garmin_labels(centre_ms: np.ndarray, garmin_folder: str) -> np.ndarray:
    hr = load_garmin_reference(GARMIN_DIR / garmin_folder).heart_rate
    if hr.empty:
        return np.full(len(centre_ms), np.nan)
    epoch = pd.Timestamp("1970-01-01")
    as_ms = lambda stamps: np.asarray((pd.DatetimeIndex(stamps) - epoch) / pd.Timedelta(milliseconds=1), dtype=float)
    times = as_ms(pd.to_datetime(hr["local_time"]))  # local wall-clock time, as the night pipeline's clock_time
    order = np.argsort(times)
    times, values = times[order], hr["garmin_hr_bpm"].to_numpy(dtype=float)[order]
    local_ms = as_ms(pd.to_datetime(centre_ms, unit="ms").tz_localize("UTC").tz_convert("Europe/Warsaw").tz_localize(None))
    label = np.interp(local_ms, times, values)
    nearest = np.min(np.abs(local_ms[:, None] - times[None, :]), axis=1) if len(times) else np.full(len(local_ms), np.inf)
    return np.where(nearest <= 120_000, label, np.nan)


def strap_labels(centre_ms: np.ndarray) -> np.ndarray:
    """Median strap heart rate in the window, from the readings that pass the strap's quality check (``hr_ok`` in
    ``*_quality.csv``), in pieces where at least :data:`STRAP_MIN_OK_SHARE` of the readings pass."""

    frames = []
    for path in sorted(STRAP_DIR.glob("heart_strap_*_quality.csv")):
        quality = pd.read_csv(path)
        ok = quality["hr_ok"].astype(bool) & (quality["heart_rate_bpm"] > 0)
        if len(quality) and ok.mean() >= STRAP_MIN_OK_SHARE:
            frames.append(quality[ok][["host_time_ms", "heart_rate_bpm"]])
    strap = pd.concat(frames).sort_values("host_time_ms") if frames else pd.DataFrame(columns=["host_time_ms", "heart_rate_bpm"])
    t, hr = strap["host_time_ms"].to_numpy(dtype=float), strap["heart_rate_bpm"].to_numpy(dtype=float)
    out = np.full(len(centre_ms), np.nan)
    for i, c in enumerate(centre_ms):
        lo, hi = np.searchsorted(t, [c - 30_000, c + 30_000])
        if hi - lo >= 30:  # about one reading a second: most of the window covered
            out[i] = float(np.median(hr[lo:hi]))
    return out


def night_data(night, step_s: float | None) -> tuple[Windows, pd.DataFrame]:
    """Training windows every ``step_s`` (None: the pipeline's own windows, for scoring) and the pipeline trend."""

    trend = pd.read_csv(NIGHT_DIR / f"{night.stem}_gated_sleep_vitals.csv")
    if step_s is None:
        ends, still = trend["elapsed_s"].to_numpy(), still_mask(trend)
    else:
        ends = np.arange(trend["elapsed_s"].iloc[0], trend["elapsed_s"].iloc[-1], step_s)
        idx = np.clip(np.searchsorted(trend["elapsed_s"].to_numpy(), ends), 0, len(trend) - 1)
        ends, still = ends, still_mask(trend)[idx]
    windows = recording_windows(night.stem, ends, still)
    windows.label = garmin_labels(windows.centre_ms, night.garmin)
    keep = np.isfinite(windows.label)
    return Windows(windows.x[keep], windows.label[keep], windows.centre_ms[keep], windows.end_s[keep], night.name), trend


# ---------------------------------------------------------------- features


def _band_mask(n: int, low_hz: float, high_hz: float, device: torch.device | None = None) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(n, 1.0 / FS, device=device)
    return ((freqs >= low_hz) & (freqs <= high_hz)).float()


def crop_and_stretch(x: torch.Tensor, label: torch.Tensor, stretch: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """[b, BINS, WINDOW + 2 PAD] -> [b, BINS, WINDOW]; with ``stretch`` a random 0.9-1.1x resampling, label scaled."""

    b = x.shape[0]
    factor = torch.empty(b).uniform_(*STRETCH) if stretch else torch.ones(b)
    centre = (x.shape[-1] - 1) / 2.0
    offsets = (torch.arange(WINDOW) - (WINDOW - 1) / 2.0)[None, :] * factor[:, None] + centre  # [b, WINDOW]
    lo = offsets.floor().long().clamp(0, x.shape[-1] - 2)
    frac = (offsets - lo).clamp(0, 1)
    gather = lambda idx: torch.gather(x, 2, idx[:, None, :].expand(-1, x.shape[1], -1))
    out = gather(lo) * (1 - frac[:, None, :]) + gather(lo + 1) * frac[:, None, :]
    return out, label * factor


def time_input(x: torch.Tensor) -> torch.Tensor:
    """Band-passed 0.5-3 Hz and scaled per bin."""

    spectrum = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1) * _band_mask(x.shape[-1], 0.5, 3.0, x.device)
    band = torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)
    return band / (band.std(-1, keepdim=True) + 1e-6)


def spectral_input(x: torch.Tensor) -> torch.Tensor:
    """Log power per bpm, 30-150 bpm, per bin (detrended, Hann window), scaled per window."""

    n = x.shape[-1]
    t = torch.arange(n, dtype=x.dtype, device=x.device)
    centred = x - x.mean(-1, keepdim=True)
    slope = (centred * (t - t.mean())).sum(-1, keepdim=True) / ((t - t.mean()) ** 2).sum()
    detrended = (centred - slope * (t - t.mean())) * torch.hann_window(n, device=x.device)
    power = torch.fft.rfft(detrended, n=n, dim=-1).abs() ** 2  # 1/60 Hz = 1 bpm steps for n = 1200
    spec = torch.log(power[..., SPEC_MIN : SPEC_MAX + 1] + 1e-9)
    return (spec - spec.mean((-1, -2), keepdim=True)) / (spec.std((-1, -2), keepdim=True) + 1e-6)


# ------------------------------------------------------------------- models


class TimeNet(nn.Module):
    def __init__(self, width: int = 48):
        super().__init__()
        layers, channels = [], BINS
        for dilation in (1, 2, 4, 8):
            layers += [nn.Conv1d(channels, width, 7, padding=3 * dilation, dilation=dilation), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)]
            channels = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(2 * width, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(time_input(x))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))


class SpectralNet(nn.Module):
    """Convolutions over frequency, shared by all bpm: one logit per bpm (36-120 cut from the 30-150 view)."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(BINS, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=16, dilation=8), nn.ReLU(),
            nn.Conv1d(width, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.body(spectral_input(x))[:, 0]
        return logits[:, HR_MIN - SPEC_MIN : HR_MAX - SPEC_MIN + 1]


class SharedTimeNet(nn.Module):
    """One encoder for every bin (1 input channel), mean and max over the bins, then the heart-rate head."""

    def __init__(self, width: int = 48):
        super().__init__()
        layers, channels = [], 1
        for dilation in (1, 2, 4, 8):
            layers += [nn.Conv1d(channels, width, 7, padding=3 * dilation, dilation=dilation), nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(2)]
            channels = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(4 * width, CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, bins, n = x.shape
        h = self.body(time_input(x).reshape(b * bins, 1, n))
        per_bin = torch.cat([h.mean(-1), h.amax(-1)], dim=1).reshape(b, bins, -1)
        return self.head(torch.cat([per_bin.mean(1), per_bin.amax(1)], dim=1))


def make_model(arch: str) -> nn.Module:
    return {"time": TimeNet, "spectral": SpectralNet, "shared": SharedTimeNet}[arch]()


def soft_targets(label: torch.Tensor) -> torch.Tensor:
    bpm = torch.arange(HR_MIN, HR_MAX + 1, dtype=torch.float32)
    weights = torch.exp(-0.5 * ((bpm[None, :] - label[:, None]) / LABEL_SIGMA) ** 2)
    return weights / weights.sum(1, keepdim=True)


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """(heart rate in bpm, confidence) per window: the peak of the distribution, refined by its neighbours."""

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb, _ = crop_and_stretch(torch.from_numpy(x[i : i + batch]), torch.zeros(min(batch, len(x) - i)), stretch=False)
            probs.append(torch.softmax(model(xb.to(device)).float(), dim=1).cpu().numpy())
    p = np.concatenate(probs) if probs else np.zeros((0, CLASSES))
    peak = p.argmax(1)
    lo, hi = np.clip(peak - 2, 0, CLASSES - 1), np.clip(peak + 2, 0, CLASSES - 1)
    bpm = np.empty(len(p))
    for i in range(len(p)):
        w = p[i, lo[i] : hi[i] + 1]
        bpm[i] = HR_MIN + float(np.sum(np.arange(lo[i], hi[i] + 1) * w) / max(w.sum(), 1e-9))
    return bpm, p.max(1)


def train_model(arch: str, stretch: bool, train: Windows, val: Windows, seed: int, device: torch.device,
                epochs: int = 60, patience: int = 12, init: dict[str, torch.Tensor] | None = None, lr: float = 1e-3) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(arch)
    if init is not None:
        model.load_state_dict(init)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    x_all, y_all = torch.from_numpy(train.x), torch.from_numpy(train.label.astype(np.float32))
    best, best_state, since, history = math.inf, None, 0, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_all))
        for i in range(0, len(order), 64):
            idx = order[i : i + 64]
            xb, yb = crop_and_stretch(x_all[idx], y_all[idx], stretch)
            xb = xb * torch.empty(len(idx), 1, 1).uniform_(0.5, 2.0)  # echo strength and depth vary between nights
            loss = -(soft_targets(yb).to(device) * torch.log_softmax(model(xb.to(device)).float(), dim=1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        bpm, _ = predict(model, val.x, device)
        mae = float(np.mean(np.abs(bpm - val.label)))
        history.append({"epoch": epoch + 1, "val_mae": mae})
        if mae < best - 1e-3:
            best, since = mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_mae": best, "epochs": len(history), "history": history}


# ------------------------------------------------------------------ scoring


def coverage_scores(bpm: np.ndarray, conf: np.ndarray, label: np.ndarray, coverage: float) -> dict[str, float]:
    """MAE over all windows, and over the most confident ``coverage`` share of them."""

    order = np.argsort(-conf)
    keep = order[: max(1, int(round(coverage * len(order))))]
    return {"mae_all": float(np.mean(np.abs(bpm - label))), "mae_matched": float(np.mean(np.abs(bpm[keep] - label[keep]))),
            "coverage": float(coverage)}


def coverage_at_mae(bpm: np.ndarray, conf: np.ndarray, label: np.ndarray, target_mae: float) -> float:
    """The largest share of the most confident windows whose MAE stays at or below ``target_mae``."""

    errors = np.abs(bpm - label)[np.argsort(-conf)]
    running = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    ok = np.flatnonzero(running <= target_mae)
    return float((ok[-1] + 1) / len(errors)) if len(ok) else 0.0


def trend_scores(times_s: np.ndarray, bpm: np.ndarray, label: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"bin": (times_s // 600).astype(int), "p": bpm, "g": label}).dropna()
    sizes = frame.groupby("bin").size()
    med = frame.groupby("bin").median()[sizes >= 5]
    r = float(np.corrcoef(med["p"], med["g"])[0, 1]) if len(med) > 2 and med["p"].std() > 0 else math.nan
    return {"mae_10min": float(np.mean(np.abs(med["p"] - med["g"]))) if len(med) else math.nan, "r_10min": r}


def split_train_val(nights: list[Windows], val_share: float = 0.2) -> tuple[Windows, Windows]:
    """The last ``val_share`` of every night validates; the 6 windows before it (60 s at a 10 s step) are dropped."""

    gap = int(round(60.0 / TRAIN_STEP_S))
    parts = []
    for w in nights:
        cut = int(len(w.label) * (1 - val_share))
        parts.append(((w.x[: max(0, cut - gap)], w.label[: max(0, cut - gap)]), (w.x[cut:], w.label[cut:])))
    join = lambda k, name: Windows(np.concatenate([p[k][0] for p in parts]), np.concatenate([p[k][1] for p in parts]), np.zeros(0), np.zeros(0), name)
    return join(0, "train"), join(1, "val")


def pretrained_states(seeds: list[int], device: torch.device) -> dict[int, dict[str, torch.Tensor]]:
    """The shared network learned on the 24 GHz radar + ECG windows; 5 of the 30 people validate."""

    data = np.load(PRETRAIN_WINDOWS)
    x, y, subject = data["x"], data["label"].astype(float), data["subject"]
    people = sorted(set(subject))
    val_people = set(people[-5:])
    val_mask = np.isin(subject, list(val_people))
    train = Windows(x[~val_mask], y[~val_mask], np.zeros(0), np.zeros(0), "pretrain")
    val = Windows(x[val_mask], y[val_mask], np.zeros(0), np.zeros(0), "pretrain-val")
    states = {}
    for seed in seeds:
        model, info = train_model("shared", True, train, val, seed, device, epochs=40, patience=8)
        print(f"pretrain seed {seed}: val MAE {info['best_val_mae']:.2f} bpm after {info['epochs']} epochs", flush=True)
        states[seed] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return states


def run(seeds: list[int], variants: list[str], device: torch.device, summary_only: bool) -> dict[str, Any]:
    nights = [n for n in NIGHTS if n.garmin is not None and (NIGHT_DIR / f"{n.stem}.npz").exists()]
    data = {}
    for night in nights:
        train_w, _ = night_data(night, TRAIN_STEP_S)
        eval_w, trend = night_data(night, None)
        if len(eval_w.label):
            data[night.name] = (train_w, eval_w, trend)
            print(f"{night.name}: {len(train_w.label)} training windows, {len(eval_w.label)} scored", flush=True)
    names = list(data)
    strap = []
    for stem in STRAP_SESSIONS:
        if (NIGHT_DIR / f"{stem}.npz").exists():
            trend = pd.read_csv(NIGHT_DIR / f"{stem}_gated_sleep_vitals.csv")
            w = recording_windows(stem, trend["elapsed_s"].to_numpy(), still_mask(trend))
            w.label = strap_labels(w.centre_ms)
            keep = np.isfinite(w.label)
            if not keep.any():
                continue
            hr = trend.set_index("elapsed_s")["hr_bpm"].reindex(w.end_s[keep]).to_numpy()
            strap.append((Windows(w.x[keep], w.label[keep], w.centre_ms[keep], w.end_s[keep], stem), hr))
    print(f"strap: {sum(len(s[0].label) for s in strap)} windows", flush=True)
    results: dict[str, Any] = {"nights": names, "folds": {}, "strap": {}}
    pretrained = pretrained_states(seeds, device) if any(v in ("shared-pre", "shared-zeroshot") for v in variants) else {}
    for test_name in names:
        train_names = [n for n in names if n != test_name]
        train, val = split_train_val([data[n][0] for n in train_names])
        test, trend = data[test_name][1], data[test_name][2]
        pipeline = trend.set_index("elapsed_s")["hr_bpm"].reindex(test.end_s).to_numpy()
        covered = np.isfinite(pipeline)
        median_train = float(np.median(train.label))
        fold: dict[str, Any] = {"val": "last 20% of each training night", "train": train_names, "windows": int(len(test.label)),
                                "pipeline": {"coverage": float(covered.mean()), "mae": float(np.mean(np.abs(pipeline[covered] - test.label[covered]))),
                                             **trend_scores(test.end_s[covered], pipeline[covered], test.label[covered])},
                                "median": {"mae": float(np.mean(np.abs(median_train - test.label)))}}
        for variant in variants:
            arch, stretch = variant.split("-")[0], not variant.endswith("nostretch")
            per_seed = []
            for seed in seeds:
                if variant == "shared-zeroshot":
                    model = make_model("shared"); model.load_state_dict(pretrained[seed]); model = model.to(device)
                    info = {"best_val_mae": float(np.mean(np.abs(predict(model, val.x, device)[0] - val.label))), "epochs": 0}
                elif variant == "shared-pre":
                    model, info = train_model("shared", True, train, val, seed, device, epochs=30, patience=8, init=pretrained[seed], lr=3e-4)
                else:
                    model, info = train_model(arch, stretch, train, val, seed, device)
                bpm, conf = predict(model, test.x, device)
                row = {**coverage_scores(bpm, conf, test.label, float(covered.mean())),
                       "coverage_at_pipeline_mae": coverage_at_mae(bpm, conf, test.label, fold["pipeline"]["mae"]),
                       **trend_scores(test.end_s, bpm, test.label), "best_val_mae": info["best_val_mae"], "epochs": info["epochs"]}
                for k, (windows, _) in enumerate(strap):
                    sb, _ = predict(model, windows.x, device)
                    row[f"strap{k}_mae"] = float(np.mean(np.abs(sb - windows.label)))
                per_seed.append(row)
                print(f"  test {test_name} {variant} seed {seed}: MAE {row['mae_all']:.2f} (pipeline {fold['pipeline']['mae']:.2f} "
                      f"at {100 * covered.mean():.0f}%), matched {row['mae_matched']:.2f}, val {info['best_val_mae']:.2f}", flush=True)
            fold[variant] = {key: float(np.mean([r[key] for r in per_seed])) for key in per_seed[0] if isinstance(per_seed[0][key], float)}
        results["folds"][test_name] = fold
    for k, (windows, hr) in enumerate(strap):
        ok = np.isfinite(hr)
        results["strap"][f"strap{k}"] = {"session": windows.source, "windows": int(len(windows.label)),
                                         "strap_median_bpm": float(np.median(windows.label)), "pipeline_coverage": float(ok.mean()),
                                         "pipeline_mae": float(np.mean(np.abs(hr[ok] - windows.label[ok]))) if ok.any() else math.nan}
    return results


def markdown(results: dict[str, Any], variants: list[str]) -> str:
    folds = results["folds"]
    lines = ["| test night | windows | pipeline coverage % | pipeline MAE | median MAE | " + " | ".join(f"{v} MAE / matched / cov@pipe" for v in variants) + " |",
             "|---|---|---|---|---|" + "---|" * len(variants)]
    for night, f in folds.items():
        cells = [f"{f[v]['mae_all']:.2f} / {f[v]['mae_matched']:.2f} / {100 * f[v]['coverage_at_pipeline_mae']:.0f}%" for v in variants if v in f]
        lines.append(f"| {night} | {f['windows']} | {100 * f['pipeline']['coverage']:.0f} | {f['pipeline']['mae']:.2f} | {f['median']['mae']:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    straps = sorted(results.get("strap", {}))
    lines.append("| variant | mean val MAE | mean test MAE (all) | mean matched MAE | mean coverage at pipeline MAE | MAE 10 min | r 10 min | "
                 + " | ".join(f"strap {results['strap'][k]['session'][11:40]} (~{results['strap'][k]['strap_median_bpm']:.0f} bpm)" for k in straps) + " |")
    lines.append("|---|---|---|---|---|---|---|" + "---|" * len(straps))
    for v in variants:
        rows = [f[v] for f in folds.values() if v in f]
        m = lambda k: float(np.nanmean([r.get(k, math.nan) for r in rows]))
        lines.append(f"| {v} | {m('best_val_mae'):.2f} | {m('mae_all'):.2f} | {m('mae_matched'):.2f} | {100 * m('coverage_at_pipeline_mae'):.0f}% | "
                     f"{m('mae_10min'):.2f} | {m('r_10min'):.2f} | " + " | ".join(f"{m(k + '_mae'):.1f}" for k in straps) + " |")
    p = lambda k: float(np.nanmean([f["pipeline"][k] for f in folds.values()]))
    lines.append(f"| pipeline | - | {p('mae'):.2f} (at {100 * p('coverage'):.0f}% coverage) | - | - | {p('mae_10min'):.2f} | {p('r_10min'):.2f} | "
                 + " | ".join(f"{results['strap'][k]['pipeline_mae']:.1f} ({100 * results['strap'][k]['pipeline_coverage']:.0f}%)" for k in straps) + " |")
    lines.append(f"| training median | - | {float(np.mean([f['median']['mae'] for f in folds.values()])):.2f} | - | - | - | - |" + " - |" * len(straps))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--tag", default="", help="suffix of summary.json / summary.md, e.g. _pretrain")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        results = json.loads((OUT_DIR / f"summary{args.tag}.json").read_text())
    else:
        results = run(seeds, variants, torch.device(args.device), False)
        (OUT_DIR / f"summary{args.tag}.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    table = markdown(results, variants)
    (OUT_DIR / f"summary{args.tag}.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
