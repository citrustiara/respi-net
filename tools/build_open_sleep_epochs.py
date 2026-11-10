#!/usr/bin/env python
"""Open sleep datasets and the author's radar nights in one per-30-s schema, for a whole-night stage model.

Every night becomes a table of 30 s epochs with the same columns, whatever measured it:

* heart: ``hr`` (bpm) -- the watch's optical heart rate (Walch et al. 2019, Bidsleep) or the radar night
  pipeline's (own nights); none from the 4 GHz bioradar of SLEEPBRL;
* breathing: ``rr`` (breaths/min), ``breath_amp``, ``breath_regularity``, ``breath_interval_cv`` -- from the chest
  motion a radar sees (SLEEPBRL, own nights; :func:`build_sleep_epochs.breathing_features`); none from a wrist;
* movement: ``motion`` -- log of the frame-to-frame scatter of the chest signal (radar) or of the wrist's
  acceleration magnitude (watch): different sensors, so only its size *within the night* is used;
* ``hours`` since the recording (lights off) started, and the stage: awake, light (N1+N2), deep (N3/N4), REM.

Scaling per night (median, inter-quartile range) makes the sources comparable: "heart rate high for this night",
"moving a lot for this night".  A missing modality is 0 with a mask column (``has_hr``, ``has_breath``).  Each
base feature also gets its 10-minute median and spread, and ``moving`` (motion more than 2 IQR above the night's
median) gives ``minutes_since_movement``.

Sources (``data/external``, not in git; each keeps its licence -- cite them):

* ``walch`` -- Apple Watch heart rate and acceleration, PSG stages, 31 nights (ODC-By 1.0);
* ``sleepbrl`` -- 4 GHz bioradar I/Q at 50 Hz, PSG stages, 32 nights (ODC-By 1.0);
* ``bidsleep`` -- Apple Watch heart rate and acceleration, Dreem EEG headband stages, 253 nights (ODC-By 1.0);
* ``own`` -- the A121 nights (``tools/build_sleep_epochs.py``), Garmin's stages where the watch was worn.

    uv run python tools/build_open_sleep_epochs.py --sources walch,sleepbrl,own
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
import struct
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from build_sleep_epochs import OUT as OWN_EPOCHS, breathing_features

EXTERNAL = ROOT / "data" / "external"
OUT_DIR = ROOT / "data" / "processed" / "sleep_open"
EPOCH_S = 30.0
STAGES = ("awake", "light", "deep", "rem")
BASE = ("hr", "rr", "breath_amp", "breath_regularity", "breath_interval_cv", "motion",
        # from the breathing-phase network on the own nights (tools/breath_phases_at_night.py); absent elsewhere
        "net_rate", "net_interval_cv", "net_inhale_fraction", "net_hold_share")
NET_FEATURES = ROOT / "data" / "processed" / "sleep_epochs" / "net_breath_features.csv"
CONTEXT = 21  # epochs, 10 min centred
WALCH_DIR = EXTERNAL / "walch_sleep_accel" / "motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"
SLEEPBRL_DIR = EXTERNAL / "sleepbrl" / "sleep-bioradiolocation-database-1.0.0"
BIDSLEEP_ZIP = EXTERNAL / "bidsleep" / "bidsleep-1.0.1.zip"


# ------------------------------------------------------------------ readers


def read_edf(path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Signals of an EDF file in physical units, and each signal's sample rate."""

    raw = path.read_bytes()
    n_records, record_s, ns = int(raw[236:244]), float(raw[244:252]), int(raw[252:256])
    field = lambda start, width, i: raw[start + i * width : start + (i + 1) * width].decode("ascii").strip()
    base = 256
    labels = [field(base, 16, i) for i in range(ns)]
    # per-signal fields, each block ns entries long: label 16, transducer 80, unit 8, pmin 8, pmax 8, dmin 8, dmax 8,
    # prefiltering 80, samples per record 8, reserved 32 bytes
    offsets = {name: base + ns * start for name, start in (("pmin", 104), ("pmax", 112), ("dmin", 120), ("dmax", 128), ("samples", 216))}
    pmin = [float(field(offsets["pmin"], 8, i)) for i in range(ns)]
    pmax = [float(field(offsets["pmax"], 8, i)) for i in range(ns)]
    dmin = [float(field(offsets["dmin"], 8, i)) for i in range(ns)]
    dmax = [float(field(offsets["dmax"], 8, i)) for i in range(ns)]
    samples = [int(field(offsets["samples"], 8, i)) for i in range(ns)]
    header_bytes = 256 * (ns + 1)
    data = np.frombuffer(raw, dtype="<i2", offset=header_bytes, count=n_records * sum(samples)).reshape(n_records, sum(samples))
    signals, rates, start = {}, {}, 0
    for i, name in enumerate(labels):
        digital = data[:, start : start + samples[i]].reshape(-1).astype(np.float64)
        gain = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) if dmax[i] != dmin[i] else 1.0
        signals[name] = (digital - dmin[i]) * gain + pmin[i]
        rates[name] = samples[i] / record_s
        start += samples[i]
    return signals, rates


def read_wfdb_annotations(path: Path) -> list[tuple[int, str]]:
    """(sample, aux text) of a WFDB (MIT format) annotation file; the aux text carries SLEEPBRL's stage."""

    data = path.read_bytes()
    out: list[tuple[int, str]] = []
    sample, i, pending = 0, 0, None
    while i + 1 < len(data):
        word = struct.unpack_from("<H", data, i)[0]
        i += 2
        code, value = word >> 10, word & 0x3FF
        if code == 0 and value == 0:
            break
        if code == 59:  # SKIP: a 32-bit interval follows (high word first)
            high, low = struct.unpack_from("<HH", data, i)
            i += 4
            sample += (high << 16) | low
        elif code == 63:  # AUX: text of the last annotation
            text = data[i : i + value].decode("latin-1").strip("\x00")
            i += value + (value & 1)
            if pending is not None:
                out[pending] = (out[pending][0], text)
        elif code in (60, 61, 62):  # NUM, SUB, CHN: no time step
            continue
        else:
            sample += value
            out.append((sample, ""))
            pending = len(out) - 1
    return out


# ------------------------------------------------------------------ schema


def _robust(values: pd.Series) -> pd.Series:
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    spread = q3 - q1 if np.isfinite(q3 - q1) and q3 - q1 > 0 else 1.0
    return (values - median) / spread


def finish_night(epochs: pd.DataFrame, source: str, night: str) -> pd.DataFrame:
    """Context, per-night scaling, masks and the movement flag of one night's raw epoch table."""

    out = epochs.copy().reset_index(drop=True)
    out["source"], out["night"] = source, night
    for name in BASE:
        if name not in out:
            out[name] = np.nan
        values = pd.to_numeric(out[name], errors="coerce")
        out[f"{name}_med10"] = values.rolling(CONTEXT, center=True, min_periods=3).median()
        out[f"{name}_sd10"] = values.rolling(CONTEXT, center=True, min_periods=3).std()
    for column in [c for c in out.columns if c.startswith(BASE)]:
        out[f"z_{column}"] = _robust(pd.to_numeric(out[column], errors="coerce")).fillna(0.0).clip(-6, 6)
    out["has_hr"] = float(out["hr"].notna().mean() > 0.2)
    out["has_breath"] = float(out["rr"].notna().mean() > 0.2)
    out["has_net"] = float(out["net_rate"].notna().mean() > 0.2)
    moving = out["z_motion"] > 2.0
    last = out["hours"].where(moving).ffill()
    out["minutes_since_movement"] = ((out["hours"] - last) * 60.0).fillna(60.0).clip(upper=60.0)
    return out


def feature_columns() -> list[str]:
    cols = []
    for name in BASE:
        cols += [f"z_{name}", f"z_{name}_med10", f"z_{name}_sd10"]
    return cols + ["minutes_since_movement_scaled", "hours_scaled", "has_hr", "has_breath", "has_net"]


def add_scaled(frame: pd.DataFrame) -> pd.DataFrame:
    frame["minutes_since_movement_scaled"] = frame["minutes_since_movement"] / 60.0
    frame["hours_scaled"] = frame["hours"] / 10.0
    return frame


# ---------------------------------------------------------------- sources


WALCH_STAGE = {0: "awake", 1: "light", 2: "light", 3: "deep", 4: "deep", 5: "rem"}


def walch_nights() -> list[pd.DataFrame]:
    nights = []
    for label_file in sorted((WALCH_DIR / "labels").glob("*_labeled_sleep.txt")):
        sid = label_file.name.split("_")[0]
        labels = pd.read_csv(label_file, sep=" ", header=None, names=["t", "stage"])
        hr = pd.read_csv(WALCH_DIR / "heart_rate" / f"{sid}_heartrate.txt", header=None, names=["t", "hr"])
        acc = pd.read_csv(WALCH_DIR / "motion" / f"{sid}_acceleration.txt", sep=" ", header=None, names=["t", "x", "y", "z"])
        acc = acc[acc["t"] >= -60].sort_values("t")
        magnitude = np.sqrt(acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2).to_numpy()
        at = acc["t"].to_numpy()
        hr = hr.sort_values("t")
        rows = []
        for start, stage in zip(labels["t"], labels["stage"]):
            lo, hi = np.searchsorted(at, [start, start + EPOCH_S])
            motion = float(np.log10(np.std(np.diff(magnitude[lo:hi])) + 1e-6)) if hi - lo > 10 else math.nan
            near = hr[(hr["t"] >= start - 15) & (hr["t"] < start + EPOCH_S + 15)]["hr"]
            rows.append({"hours": start / 3600.0, "hr": float(near.median()) if len(near) else math.nan, "motion": motion,
                         "stage": WALCH_STAGE.get(int(stage))})
        nights.append(finish_night(pd.DataFrame(rows), "walch", sid))
    return nights


SLEEPBRL_STAGE = {"W": "awake", "1": "light", "2": "light", "3": "deep", "4": "deep", "R": "rem"}


def _best_radar_channel(signals: dict[str, np.ndarray], fs: float, lo: int, hi: int) -> np.ndarray:
    """The I or Q channel with the most breathing-band power in this window (CW radar: the chest's motion shows
    in whichever channel is not at a null of the carrier's phase)."""

    best, best_power = None, -1.0
    for name, values in signals.items():
        part = values[lo:hi]
        spectrum = np.abs(np.fft.rfft(part - part.mean())) ** 2
        freqs = np.fft.rfftfreq(len(part), 1.0 / fs)
        power = float(spectrum[(freqs >= 0.1) & (freqs <= 0.7)].sum())
        if power > best_power:
            best, best_power = part, power
    return best


def sleepbrl_nights() -> list[pd.DataFrame]:
    nights = []
    for edf in sorted(SLEEPBRL_DIR.glob("sbj*.edf")):
        signals, rates = read_edf(edf)
        fs = float(next(iter(rates.values())))
        annotations = read_wfdb_annotations(edf.with_suffix(".edf.atr"))
        rows = []
        n = len(next(iter(signals.values())))
        for sample, text in annotations:
            stage = SLEEPBRL_STAGE.get(text.strip()[:1].upper()) if text else None
            centre = sample + int(EPOCH_S * fs / 2)
            lo, hi = max(0, centre - int(30 * fs)), min(n, centre + int(30 * fs))  # 60 s around the epoch
            if hi - lo < int(40 * fs):
                rows.append({"hours": sample / fs / 3600.0, "stage": stage})
                continue
            part = _best_radar_channel(signals, fs, lo, hi)
            chest = resample_poly(part, 2, 5)  # 50 -> 20 Hz, as the A121 features
            feats = breathing_features(chest, 20.0)
            rhythm_rate = _rate_bpm(chest, 20.0)
            rows.append({"hours": sample / fs / 3600.0, "rr": rhythm_rate, "breath_amp": feats["breath_amp_mm"],
                         "breath_regularity": feats["breath_regularity"], "breath_interval_cv": feats["breath_interval_cv"],
                         "motion": feats["motion_energy"], "stage": stage})
        nights.append(finish_night(pd.DataFrame(rows), "sleepbrl", edf.stem))
    return nights


def _rate_bpm(chest: np.ndarray, fs: float) -> float:
    from respi_net.phase_baseline import window_rhythm

    rhythm = window_rhythm(chest, fs)
    rate = float(np.atleast_1d(rhythm.peak_hz)[0]) * 60.0
    return rate if 6.0 <= rate <= 30.0 and float(np.atleast_1d(rhythm.share)[0]) > 0.2 else math.nan


OWN_STAGE = {"awake": "awake", "light": "light", "deep": "deep", "rem": "rem"}


def own_nights() -> list[pd.DataFrame]:
    epochs = pd.read_csv(OWN_EPOCHS, low_memory=False)
    if NET_FEATURES.exists():
        epochs = epochs.merge(pd.read_csv(NET_FEATURES), on=["night", "elapsed_s"], how="left")
    nights = []
    for night, frame in epochs.groupby("night"):
        frame = frame.sort_values("elapsed_s")
        raw = pd.DataFrame({
            "hours": frame["hours_since_start"].to_numpy(), "hr": frame["hr_bpm"].to_numpy(), "rr": frame["rr_bpm"].to_numpy(),
            "breath_amp": frame["breath_amp_mm"].to_numpy(), "breath_regularity": frame["breath_regularity"].to_numpy(),
            "breath_interval_cv": frame["breath_interval_cv"].to_numpy(), "motion": frame["motion_energy"].to_numpy(),
            "stage": frame["garmin_stage"].map(OWN_STAGE).to_numpy(), "setup": frame["setup"].to_numpy(),
            "elapsed_s": frame["elapsed_s"].to_numpy(),
            **{name: frame[f"{name.replace('net_rate', 'net_rate_bpm')}"].to_numpy() if name.replace('net_rate', 'net_rate_bpm') in frame else np.nan
               for name in ("net_rate", "net_interval_cv", "net_inhale_fraction", "net_hold_share")},
        })
        nights.append(finish_night(raw, "own", str(night)))
    return nights


BIDSLEEP_STAGE = {0: "awake", 1: "light", 2: "light", 3: "deep", 4: "rem"}  # 5 = unknown


def bidsleep_nights(limit: int | None = None) -> list[pd.DataFrame]:
    """Read straight from the ZIP (30 GB unpacked): per night ``hr.csv`` (unix time, bpm; no header),
    ``motion.csv`` (Timestamp, x, y, z in g) and ``labels.mat`` (``expert_label`` per 30 s from ``recStart``, local
    time).  The local-to-unix offset of each night is the whole number of hours under which the labels are best
    covered by acceleration data."""

    import scipy.io as sio

    nights = []
    with zipfile.ZipFile(BIDSLEEP_ZIP) as archive:
        groups: dict[str, dict[str, str]] = {}
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) >= 4 and parts[-1] in ("hr.csv", "motion.csv", "labels.mat"):
                groups.setdefault(f"{parts[-3]}_n{parts[-2]}", {})[parts[-1]] = name
        for key, files in sorted(groups.items())[:limit]:
            if len(files) < 3:
                continue
            hr = pd.read_csv(io.BytesIO(archive.read(files["hr.csv"])), header=None, names=["t", "hr"])
            with archive.open(files["motion.csv"]) as handle:
                acc = pd.read_csv(handle)
            labels = sio.loadmat(io.BytesIO(archive.read(files["labels.mat"])))
            nights.append(_bidsleep_night(hr, acc, labels, key))
            if len(nights) % 25 == 0:
                print(f"  bidsleep {len(nights)} nights", flush=True)
    return nights


def _bidsleep_night(hr: pd.DataFrame, acc: pd.DataFrame, labels: dict, key: str) -> pd.DataFrame:
    stages = np.asarray(labels["expert_label"]).ravel().astype(int)
    rec_start = pd.Timestamp(str(np.asarray(labels["recStart"]).ravel()[0]).strip())
    naive_start = (rec_start - pd.Timestamp("1970-01-01")).total_seconds()
    hr = hr.dropna().sort_values("t")
    at = acc["Timestamp"].to_numpy(dtype=float)
    magnitude = np.sqrt(acc["x"].to_numpy(float) ** 2 + acc["y"].to_numpy(float) ** 2 + acc["z"].to_numpy(float) ** 2)
    order = np.argsort(at)
    at, magnitude = at[order], magnitude[order]
    # recStart is local wall-clock time: take the whole-hour offset under which the epochs are best covered by
    # acceleration samples (the heart-rate record need not be centred on the night)
    epoch_starts = naive_start + np.arange(len(stages)) * EPOCH_S
    coverage = {h: int(np.sum(np.searchsorted(at, epoch_starts + h * 3600 + EPOCH_S) - np.searchsorted(at, epoch_starts + h * 3600) > 10))
                for h in range(-12, 15)}
    offset_h = max(coverage, key=coverage.get)
    start = naive_start + offset_h * 3600.0
    ht, hv = hr["t"].to_numpy(float), hr["hr"].to_numpy(float)
    rows = []
    for k, stage in enumerate(stages):
        t0 = start + k * EPOCH_S
        lo, hi = np.searchsorted(at, [t0, t0 + EPOCH_S])
        motion = float(np.log10(np.std(np.diff(magnitude[lo:hi])) + 1e-6)) if hi - lo > 10 else math.nan
        hlo, hhi = np.searchsorted(ht, [t0 - 15, t0 + EPOCH_S + 15])
        rows.append({"hours": k * EPOCH_S / 3600.0, "hr": float(np.median(hv[hlo:hhi])) if hhi > hlo else math.nan,
                     "motion": motion, "stage": BIDSLEEP_STAGE.get(int(stage))})
    return finish_night(pd.DataFrame(rows), "bidsleep", key)


SOURCES = {"walch": walch_nights, "sleepbrl": sleepbrl_nights, "own": own_nights, "bidsleep": bidsleep_nights}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", default="walch,sleepbrl,own")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for source in [s for s in args.sources.split(",") if s]:
        nights = SOURCES[source]()
        frame = add_scaled(pd.concat(nights, ignore_index=True))
        frame.to_csv(OUT_DIR / f"{source}.csv", index=False)
        labelled = frame["stage"].isin(STAGES)
        shares = frame.loc[labelled, "stage"].value_counts(normalize=True).round(2).to_dict()
        print(f"{source}: {frame['night'].nunique()} nights, {len(frame)} epochs, {labelled.sum()} labelled {shares}; "
              f"has hr {frame.groupby('night')['has_hr'].first().mean():.2f}, breath {frame.groupby('night')['has_breath'].first().mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
