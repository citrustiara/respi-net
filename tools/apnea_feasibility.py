#!/usr/bin/env python
"""Is apnea detection from chest motion worth pursuing?  One detector on four kinds of evidence.

The detector works on any chest-motion signal and needs no calibration: the breathing band (0.1-0.7 Hz), its
amplitude every second (5th-95th percentile spread over 8 s), divided by the median amplitude of the surrounding
5 minutes.  An event is a drop below :data:`EVENT_RATIO` lasting at least :data:`MIN_EVENT_S` (gaps under 3 s
bridged, longer than 120 s dropped); it is apnea-like when its amplitude falls below :data:`APNEA_RATIO`, otherwise
hypopnea-like -- the AASM rules for airflow (>=90% / >=30-50% drop for >=10 s) carried over to chest motion.

* ``B`` UCDDB: the thoracic effort belt of 25 adults (event threshold and baseline quantile chosen on 5, scored on 20) with suspected sleep apnea, against the technicians' events
  (obstructive, central, mixed apneas and hypopneas): sensitivity per type, precision, and the estimated against the
  annotated apnea-hypopnea index (events per hour of sleep).  Obstructive events are expected to be the hardest:
  the chest keeps moving against the closed airway.
* ``C`` the author's A121 nights (healthy, no reference): events per hour of radar-estimated sleep = false alarms.
* ``C2`` synthetic events injected into those nights' radar signal (central apnea: breathing removed, heart and noise
  kept; hypopnea: breathing scaled to 30-70%; 10-40 s, a few larger recovery breaths after): sensitivity by kind,
  depth and duration in real radar noise.  Optimistic: real events end in arousal and movement.
* ``A`` Schellenberger et al. 2020, apnea scenario: voluntary breath holds of 30 people lying; the same detector on
  the thoracic impedance gives the reference, on the 24 GHz radar the detection.

    uv run python tools/apnea_feasibility.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, resample_poly, sosfiltfilt

from build_open_sleep_epochs import read_edf
from build_radar_ecg_windows import displacement_mm
from build_sleep_epochs import NIGHT_DIR, NIGHTS
from respi_net.a121_sleep import classify_radar_sleep
from respi_net.chest_signal import A121_CHEST_SIGN, PHASE_TO_MM

UCDDB = next((ROOT / "data" / "external" / "ucddb").glob("*/"), ROOT / "data" / "external" / "ucddb")
SCHELL = ROOT / "data" / "external" / "schellenberger2020"
OUT = ROOT / "runs" / "apnea"
EVENT_RATIO, APNEA_RATIO = 0.6, 0.2  # chosen on the first 5 UCDDB subjects (TUNING), scored on the other 20
MIN_EVENT_S, MAX_EVENT_S, BRIDGE_S = 10.0, 120.0, 3.0
BASELINE_S, ENVELOPE_S = 300.0, 8.0
BASELINE_Q = 0.9  # quantile of the surrounding amplitude taken as normal breathing: events pull a median down
TUNING = 5  # UCDDB subjects (in name order) the two settings were chosen on; left out of the scores
MATCH_TOLERANCE_S = 5.0


# ----------------------------------------------------------------- detector


def envelope(chest: np.ndarray, fs: float) -> np.ndarray:
    """Breathing amplitude every second (5th-95th percentile spread of the breathing band over 8 s)."""

    band = sosfiltfilt(butter(2, (0.1, 0.7), "bandpass", fs=fs, output="sos"), chest - np.nanmean(chest))
    n_s = int(len(band) / fs)
    half = int(ENVELOPE_S * fs / 2)
    out = np.full(n_s, np.nan)
    for t in range(n_s):
        c = int(t * fs)
        part = band[max(0, c - half) : c + half]
        if len(part) > fs:
            out[t] = np.subtract(*np.percentile(part, [95, 5]))
    return out


def detect_events(amplitude: np.ndarray, allowed: np.ndarray | None = None) -> list[dict[str, float]]:
    """Events (start s, end s, lowest ratio, kind) in a 1 Hz amplitude series."""

    baseline = pd.Series(amplitude).rolling(int(BASELINE_S), center=True, min_periods=60).quantile(BASELINE_Q).to_numpy()
    ratio = amplitude / np.where(baseline > 0, baseline, np.nan)
    low = np.nan_to_num(ratio, nan=1.0) < EVENT_RATIO
    if allowed is not None:
        n_common = min(len(low), len(allowed))
        low = low[:n_common] & allowed[:n_common]
        ratio = ratio[:n_common]
    # bridge short gaps
    k, n = 0, len(low)
    while k < n:
        if not low[k]:
            j = k
            while j < n and not low[j]:
                j += 1
            if 0 < k and j < n and (j - k) <= BRIDGE_S:
                low[k:j] = True
            k = j
        else:
            k += 1
    events, k = [], 0
    while k < n:
        if low[k]:
            j = k
            while j < n and low[j]:
                j += 1
            if MIN_EVENT_S <= j - k <= MAX_EVENT_S:
                lowest = float(np.nanmin(ratio[k:j]))
                events.append({"start": float(k), "end": float(j), "ratio": lowest, "kind": "apnea" if lowest < APNEA_RATIO else "hypopnea"})
            k = j
        else:
            k += 1
    return events


def match(detected: list[dict], reference: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(reference found?, detection true?) with :data:`MATCH_TOLERANCE_S` slack on both ends."""

    found = np.zeros(len(reference), bool)
    true = np.zeros(len(detected), bool)
    for i, r in enumerate(reference):
        for j, d in enumerate(detected):
            if d["start"] <= r["end"] + MATCH_TOLERANCE_S and d["end"] >= r["start"] - MATCH_TOLERANCE_S:
                found[i] = True
                true[j] = True
    return found, true


# ---------------------------------------------------------------- B: UCDDB


def _clock_s(text: str) -> float:
    h, m, s = (int(x) for x in re.split("[:.]", text.strip())[:3])
    return h * 3600 + m * 60 + s


def ucddb_subject(rec: Path) -> dict[str, Any]:
    signals, rates = read_edf(rec)
    raw = rec.read_bytes()[:256]
    start = _clock_s(raw[176:184].decode())
    events = []
    for line in rec.with_name(rec.stem + "_respevt.txt").read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].count(":") == 2 and re.match(r"(APNEA|HYP)-", parts[1]):
            duration = float(parts[2]) if re.match(r"^\d+(\.\d+)?$", parts[2]) else float(parts[3])
            t = (_clock_s(parts[0]) - start) % 86400
            events.append({"start": t, "end": t + duration, "type": parts[1]})
    stages = np.array([int(x) for x in rec.with_name(rec.stem + "_stage.txt").read_text().split()])
    asleep = np.repeat(np.isin(stages, (1, 2, 3, 4, 5)), 30)  # UCDDB: 0 wake, 1 REM, 2-5 S1-S4, 6 artifact, 7 indeterminate
    amplitude = envelope(signals["ribcage"], rates["ribcage"])
    n = min(len(amplitude), len(asleep))
    detected = detect_events(amplitude[:n], asleep[:n])
    sleep_h = asleep[:n].sum() / 3600.0
    in_sleep = [e for e in events if e["start"] < n and asleep[int(min(e["start"], n - 1))]]
    found, true = match(detected, in_sleep)
    return {"subject": rec.stem, "sleep_h": sleep_h, "events": in_sleep, "found": found.tolist(), "detected": len(detected),
            "true_detections": int(true.sum()), "ahi_reference": len(in_sleep) / sleep_h if sleep_h else math.nan,
            "ahi_estimated": len(detected) / sleep_h if sleep_h else math.nan}


def part_b() -> dict[str, Any]:
    subjects = [ucddb_subject(p) for p in sorted(UCDDB.glob("ucddb*.rec"))[TUNING:]]
    by_type: dict[str, list[bool]] = {}
    for s in subjects:
        for e, f in zip(s["events"], s["found"]):
            by_type.setdefault(e["type"], []).append(f)
    detected = sum(s["detected"] for s in subjects)
    true = sum(s["true_detections"] for s in subjects)
    ref = np.array([s["ahi_reference"] for s in subjects]); est = np.array([s["ahi_estimated"] for s in subjects])
    severe_ref, severe_est = ref >= 15, est >= 15
    return {"subjects": len(subjects), "sensitivity_by_type": {k: float(np.mean(v)) for k, v in sorted(by_type.items())},
            "events_by_type": {k: len(v) for k, v in sorted(by_type.items())},
            "sensitivity_all": float(np.mean([f for v in by_type.values() for f in v])), "precision": true / detected if detected else math.nan,
            "ahi_r": float(np.corrcoef(ref, est)[0, 1]), "ahi_mae": float(np.mean(np.abs(ref - est))),
            "ahi15_agreement": float(np.mean(severe_ref == severe_est)),
            "per_subject": [{"subject": s["subject"], "ahi_reference": s["ahi_reference"], "ahi_estimated": s["ahi_estimated"]} for s in subjects]}


# ------------------------------------------------------------ C / C2: own


def night_chest(stem: str, chunk_s: float = 3600.0) -> tuple[np.ndarray, float]:
    """The night's chest displacement (mm, rising on inhale), the 3 bins around each hour's median target."""

    with np.load(NIGHT_DIR / f"{stem}.npz") as raw:
        stamps, distances, iq, target = raw["timestamp_ms"], raw["distances_m"], raw["iq"], raw["acconeer_target_m"]
    t = (stamps - stamps[0]) / 1000.0
    fs = float(1.0 / np.median(np.diff(t)))
    parts = []
    for lo in range(0, len(t), int(chunk_s * fs)):
        hi = min(len(t), lo + int(chunk_s * fs))
        z = iq[lo:hi, :, 0].astype(np.float32) + 1j * iq[lo:hi, :, 1].astype(np.float32)
        finite = target[lo:hi][np.isfinite(target[lo:hi]) & (target[lo:hi] > 0)]
        amp = np.median(np.abs(z), axis=0)
        centre = int(np.argmin(np.abs(distances - np.median(finite)))) if len(finite) else int(np.argmax(amp))
        gate = np.arange(max(0, centre - 1), min(len(distances), centre + 2))
        w = amp[gate] / (amp[gate].sum() + 1e-12)
        phase = np.unwrap(np.angle(z[:, gate]), axis=0) @ w
        parts.append(A121_CHEST_SIGN * (phase - phase.mean()) * PHASE_TO_MM)
    return np.concatenate(parts), fs


def own_sleep_mask(night, n_s: int) -> np.ndarray:
    """Seconds inside the radar-estimated sleep interval, non-awake and still (gated trend windows)."""

    trend = pd.read_csv(NIGHT_DIR / f"{night.stem}_gated_sleep_vitals.csv")
    c = classify_radar_sleep(trend)
    frame = c.frame
    mask = np.zeros(n_s, bool)
    for k in range(c.onset_idx, c.wake_idx + 1):
        row = frame.iloc[k]
        if row["radar_sleep_phase"] != "awake" and float(row.get("still_fraction", 1.0)) >= 0.65:
            lo, hi = int(row["elapsed_s"] - 30), int(row["elapsed_s"])
            mask[max(0, lo) : min(n_s, hi)] = True
    return mask


def inject(chest: np.ndarray, fs: float, start_s: float, length_s: float, scale: float) -> np.ndarray:
    """Breathing band scaled to ``scale`` for ``length_s`` (2 s ramps), larger breaths for 10 s after."""

    band = sosfiltfilt(butter(2, (0.1, 0.7), "bandpass", fs=fs, output="sos"), chest - chest.mean())
    gain = np.ones(len(chest))
    a, b = int(start_s * fs), int((start_s + length_s) * fs)
    ramp = int(2 * fs)
    gain[a:b] = scale
    gain[max(0, a - ramp) : a] = np.linspace(1, scale, len(gain[max(0, a - ramp) : a]))
    gain[b : b + ramp] = np.linspace(scale, 1.6, len(gain[b : b + ramp]))
    gain[b + ramp : b + int(10 * fs)] = 1.6
    gain[b + int(10 * fs) : b + int(12 * fs)] = np.linspace(1.6, 1, len(gain[b + int(10 * fs) : b + int(12 * fs)]))
    return chest + (gain - 1.0) * band


def part_c(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    nights, synthetic = [], []
    for night in NIGHTS:
        if not (NIGHT_DIR / f"{night.stem}.npz").exists():
            continue
        chest, fs = night_chest(night.stem)
        amplitude = envelope(chest, fs)
        mask = own_sleep_mask(night, len(amplitude))
        events = detect_events(amplitude, mask)
        hours = mask.sum() / 3600.0
        nights.append({"night": night.name, "sleep_h": hours, "events": len(events), "per_hour": len(events) / hours if hours else math.nan,
                       "apnea_like": sum(e["kind"] == "apnea" for e in events)})
        # C2: 30 events per night in sleep, at least 3 min apart, in stretches with no detection
        candidates = np.flatnonzero(mask)
        busy = np.zeros(len(mask), bool)
        for e in events:
            busy[int(e["start"]) - 60 : int(e["end"]) + 60] = True
        placed: list[float] = []
        for _ in range(400):
            if len(placed) >= 30 or not len(candidates):
                break
            t0 = float(rng.choice(candidates))
            length = float(rng.choice([10, 15, 20, 30, 40]))
            if t0 + length + 40 >= len(mask) or not mask[int(t0) : int(t0 + length)].all() or busy[int(t0) - 30 : int(t0 + length) + 30].any():
                continue
            if any(abs(t0 - p) < 180 for p in placed):
                continue
            placed.append(t0)
        for t0 in placed:
            kind = rng.choice(["apnea", "hypopnea"])
            scale = 0.0 if kind == "apnea" else float(rng.choice([0.3, 0.5, 0.7]))
            length = float(rng.choice([10, 15, 20, 30, 40]))
            lo, hi = int(max(0, t0 - 400)), int(min(len(mask), t0 + length + 400))
            part = inject(chest[int(lo * fs) : int(hi * fs)], fs, t0 - lo, length, scale)
            det = detect_events(envelope(part, fs), mask[lo:hi])
            target = [{"start": t0 - lo, "end": t0 - lo + length}]
            found, _ = match(det, target)
            synthetic.append({"night": night.name, "kind": kind, "scale": scale, "length_s": length, "found": bool(found[0])})
        print(f"C {night.name}: {nights[-1]['per_hour']:.1f} events/h over {hours:.1f} h; C2 {len(placed)} injected", flush=True)
    syn = pd.DataFrame(synthetic)
    table = syn.groupby(["kind", "scale"])["found"].agg(["mean", "count"]).reset_index().to_dict("records")
    by_length = syn.groupby("length_s")["found"].mean().to_dict()
    return {"nights": nights, "false_per_hour_median": float(np.median([n["per_hour"] for n in nights])),
            "synthetic_by_kind": table, "synthetic_by_length": {str(k): float(v) for k, v in by_length.items()},
            "synthetic_total": int(len(syn)), "synthetic_found": float(syn["found"].mean()) if len(syn) else math.nan}


# ------------------------------------------------------ A: Schellenberger


def part_a() -> dict[str, Any]:
    found_all, precision_hits, detected_all, rows = [], 0, 0, []
    for path in sorted(SCHELL.glob("GDN*/GDN*_3_Apnea.mat")):
        m = sio.loadmat(path)
        fs_r, fs_z = float(m["fs_radar"].squeeze()), float(m["fs_z0"].squeeze())
        z = m["tfm_z0"].ravel().astype(float)
        if np.mean(np.isfinite(z)) < 0.9:
            continue
        z = np.where(np.isfinite(z), z, np.nanmedian(z))
        radar = resample_poly(displacement_mm(m["radar_i"].ravel(), m["radar_q"].ravel()), 1, int(round(fs_r / 20)))
        ref = resample_poly(z, 1, int(round(fs_z / 20)))
        reference = detect_events(envelope(ref, 20.0))
        detected = detect_events(envelope(radar, 20.0))
        found, true = match(detected, reference)
        found_all += found.tolist(); precision_hits += int(true.sum()); detected_all += len(detected)
        rows.append({"recording": path.stem, "reference_holds": len(reference), "detected": len(detected), "found": int(found.sum())})
    return {"recordings": len(rows), "reference_holds": len(found_all), "sensitivity": float(np.mean(found_all)) if found_all else math.nan,
            "precision": precision_hits / detected_all if detected_all else math.nan, "per_recording": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", default="a,b,c")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    parts = args.parts.split(",")
    if "a" in parts:
        result["A_schellenberger_holds"] = part_a(); print("A", json.dumps({k: v for k, v in result["A_schellenberger_holds"].items() if k != "per_recording"}), flush=True)
    if "b" in parts:
        result["B_ucddb"] = part_b(); print("B", json.dumps({k: v for k, v in result["B_ucddb"].items() if k != "per_subject"}), flush=True)
    if "c" in parts:
        result["C_own_nights"] = part_c(); print("C", json.dumps({k: v for k, v in result["C_own_nights"].items() if k != "nights"}, default=float), flush=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
