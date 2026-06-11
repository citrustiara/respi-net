#!/usr/bin/env python
"""Create long-duration A121 sleep diagnostics without plotting full raw vitals traces."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt, welch

from respi_net.a121 import parse_json_array
from respi_net.a121_vitals import _times_seconds_from_ms, analyze_a121_vitals, sample_rate_from_ms


@dataclass(frozen=True)
class Peak:
    bpm: float
    confidence: float
    bpm_axis: np.ndarray
    power: np.ndarray


def _load_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack([parse_json_array(value) for value in series])


def _valid_band(band: tuple[float, float], fs: float) -> tuple[float, float]:
    nyq = fs * 0.5
    return max(0.001, band[0]), min(band[1], nyq * 0.98)


def _bandpass(x: np.ndarray, fs: float, band_hz: tuple[float, float], order: int = 3) -> np.ndarray:
    lo, hi = _valid_band(band_hz, fs)
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _psd_peak(signal: np.ndarray, fs: float, band_bpm: tuple[float, float], *, max_nperseg_s: float | None = None) -> Peak:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max(32, int(fs * 8)):
        return Peak(0.0, 0.0, np.asarray([]), np.asarray([]))
    if max_nperseg_s is None:
        nperseg = len(x)
    else:
        nperseg = min(len(x), max(64, int(round(fs * max_nperseg_s))))
    freqs, power = welch(x, fs=fs, nperseg=nperseg, scaling="spectrum")
    bpm_axis = freqs * 60.0
    valid = (bpm_axis >= band_bpm[0]) & (bpm_axis <= band_bpm[1])
    if not np.any(valid):
        return Peak(0.0, 0.0, bpm_axis, power)
    band_power = power[valid]
    band_bpm_axis = bpm_axis[valid]
    idx = int(np.argmax(band_power))
    confidence = float(band_power[idx] / (np.median(band_power) + 1e-18))
    return Peak(float(band_bpm_axis[idx]), confidence, bpm_axis, power)


def _rolling_median(values: np.ndarray, width: int = 5) -> np.ndarray:
    if len(values) == 0 or width <= 1:
        return values
    out = np.empty_like(values, dtype=float)
    half = width // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = np.nanmedian(values[lo:hi])
    return out


def _chunk_peaks(time_s: np.ndarray, signal: np.ndarray, fs: float, *, chunk_s: float, step_s: float, band_bpm: tuple[float, float], nperseg_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers: list[float] = []
    bpms: list[float] = []
    confs: list[float] = []
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0
    for start in np.arange(0.0, max(duration_s - chunk_s, 0.0) + 1e-9, step_s):
        end = start + chunk_s
        mask = (time_s >= start) & (time_s < end)
        if int(np.sum(mask)) < int(fs * chunk_s * 0.65):
            continue
        peak = _psd_peak(signal[mask], fs, band_bpm, max_nperseg_s=nperseg_s)
        centers.append((start + end) * 0.5)
        bpms.append(peak.bpm)
        confs.append(peak.confidence)
    return np.asarray(centers), np.asarray(bpms), np.asarray(confs)


def _analyzer_trend(df: pd.DataFrame, time_s: np.ndarray, *, window_s: float = 60.0, step_s: float = 15.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers: list[float] = []
    hrs: list[float] = []
    hr_confs: list[float] = []
    rrs: list[float] = []
    rr_confs: list[float] = []
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0
    for end in np.arange(window_s, duration_s + 0.1, step_s):
        mask = (time_s >= end - window_s) & (time_s <= end)
        if int(np.sum(mask)) < 30:
            continue
        work = df.loc[mask].copy()
        analysis = analyze_a121_vitals(work, max_frames=len(work), heart_window_s=window_s)
        centers.append(end - window_s * 0.5)
        hrs.append(float(analysis.heart_bpm) if analysis.heart_bpm > 0 else np.nan)
        hr_confs.append(float(analysis.heart_confidence))
        rrs.append(float(analysis.resp_bpm) if analysis.resp_bpm > 0 else np.nan)
        rr_confs.append(float(analysis.resp_confidence))
    return np.asarray(centers), np.asarray(hrs), np.asarray(hr_confs), np.asarray(rrs), np.asarray(rr_confs)


def analyze(csv_path: Path, output_dir: Path, *, dpi: int = 140) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    timestamps_ms = df["Timestamp_ms"].to_numpy(dtype=float)
    fs = sample_rate_from_ms(timestamps_ms)
    time_s = _times_seconds_from_ms(timestamps_ms, fs)
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) else 0.0

    distances = parse_json_array(df["Distances_m"].iloc[0])
    real = _load_matrix(df["Real"])
    imag = _load_matrix(df["Imag"])
    m = min(len(distances), real.shape[1], imag.shape[1])
    distances = distances[:m]
    z = real[:, :m] + 1j * imag[:, :m]
    amplitude = np.abs(z)

    target_series = pd.to_numeric(df.get("AcconeerTargetDistance_m"), errors="coerce") if "AcconeerTargetDistance_m" in df else pd.Series(dtype=float)
    target_m = float(np.nanmedian(target_series)) if target_series.notna().any() else float(distances[int(np.argmax(np.median(amplitude, axis=0)))])
    gate_start = pd.to_numeric(df.get("AcconeerRangeStart_m"), errors="coerce") if "AcconeerRangeStart_m" in df else pd.Series(dtype=float)
    gate_end = pd.to_numeric(df.get("AcconeerRangeEnd_m"), errors="coerce") if "AcconeerRangeEnd_m" in df else pd.Series(dtype=float)
    if gate_start.notna().any() and gate_end.notna().any():
        gate_min_m = float(np.nanmedian(gate_start))
        gate_max_m = float(np.nanmedian(gate_end))
    else:
        gate_min_m, gate_max_m = target_m - 0.04, target_m + 0.04
    gate_mask = (distances >= gate_min_m) & (distances <= gate_max_m)
    if not np.any(gate_mask):
        gate_mask = np.abs(distances - target_m) <= 0.05
    gate_idx = np.flatnonzero(gate_mask)
    if len(gate_idx) == 0:
        gate_idx = np.asarray([int(np.argmin(np.abs(distances - target_m)))])
    weights = np.median(amplitude[:, gate_idx], axis=0)
    weights = weights / (float(np.sum(weights)) + 1e-12)

    phase = detrend(np.unwrap(np.angle(z[:, gate_idx]), axis=0), axis=0, type="linear")
    resp_signal = _bandpass(phase, fs, (0.10, 0.50), order=2) @ weights
    heart_signal = _bandpass(phase, fs, (0.70, 2.00), order=3) @ weights

    full_analysis = analyze_a121_vitals(df, max_frames=len(df), heart_window_s=60.0)
    tail_analysis = analyze_a121_vitals(df, max_frames=min(len(df), int(round(fs * 60.0))), heart_window_s=60.0)
    weighted_rr = _psd_peak(resp_signal, fs, (6.0, 30.0), max_nperseg_s=180.0)
    weighted_hr = _psd_peak(heart_signal, fs, (42.0, 120.0), max_nperseg_s=120.0)

    chunk_t_hr, chunk_hr, chunk_hr_conf = _chunk_peaks(time_s, heart_signal, fs, chunk_s=60.0, step_s=30.0, band_bpm=(42.0, 120.0), nperseg_s=60.0)
    chunk_t_rr, chunk_rr, chunk_rr_conf = _chunk_peaks(time_s, resp_signal, fs, chunk_s=60.0, step_s=30.0, band_bpm=(6.0, 30.0), nperseg_s=60.0)
    an_t, an_hr, an_hr_conf, an_rr, an_rr_conf = _analyzer_trend(df, time_s, window_s=60.0, step_s=15.0)

    ac_rr = pd.to_numeric(df.get("AcconeerBreathingRate_BPM"), errors="coerce") if "AcconeerBreathingRate_BPM" in df else pd.Series(dtype=float)
    state = df["AcconeerAppState"].astype(str) if "AcconeerAppState" in df else pd.Series([""] * len(df))
    present_mask = state.eq("ESTIMATE_BREATHING_RATE").to_numpy()

    # Figure 1: whole-record overview and trends. No full raw respiration/heart waveform.
    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    grid = fig.add_gridspec(4, 1, height_ratios=[1.35, 0.75, 0.9, 0.9], hspace=0.35)
    fig.suptitle(f"A121 sleep overview: {csv_path.name} ({duration_s / 60.0:.1f} min)", fontsize=15, fontweight="bold")
    summary = (
        f"Rows {len(df)} | fs {fs:.3f} Hz | target {target_m:.3f} m | gate {gate_min_m:.3f}-{gate_max_m:.3f} m\n"
        f"Overall analyzer: HR {full_analysis.heart_bpm:.1f} BPM (conf {full_analysis.heart_confidence:.1f}), "
        f"RR {full_analysis.resp_bpm:.1f} BPM (conf {full_analysis.resp_confidence:.1f})\n"
        f"Weighted PSD: HR {weighted_hr.bpm:.1f} BPM (conf {weighted_hr.confidence:.1f}), "
        f"RR {weighted_rr.bpm:.1f} BPM (conf {weighted_rr.confidence:.1f})"
    )
    fig.text(0.5, 0.94, summary, ha="center", va="top", fontsize=10, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.86})

    ax0 = fig.add_subplot(grid[0])
    ds = max(1, int(round(len(df) / 2500)))
    mesh = ax0.pcolormesh(time_s[::ds] / 60.0, distances, amplitude[::ds].T, shading="auto", cmap="viridis")
    peak_dist = pd.to_numeric(df["PeakDistance_m"], errors="coerce").to_numpy(dtype=float)
    ax0.plot(time_s[::ds] / 60.0, peak_dist[::ds], color="white", lw=0.7, alpha=0.8, label="peak distance")
    if target_series.notna().any():
        ax0.plot(time_s[::ds] / 60.0, target_series.to_numpy(dtype=float)[::ds], color="#ff4d4d", lw=1.0, label="Acconeer target")
    ax0.axhspan(gate_min_m, gate_max_m, color="white", alpha=0.10, label="median selected range")
    ax0.set_title("Full 15-min range-amplitude overview")
    ax0.set_ylabel("Distance (m)")
    ax0.set_xlim(0, duration_s / 60.0)
    ax0.grid(True, color="white", alpha=0.18)
    ax0.legend(loc="upper right")
    fig.colorbar(mesh, ax=ax0, pad=0.01, label="Amplitude")

    ax1 = fig.add_subplot(grid[1], sharex=ax0)
    ax1.fill_between(time_s / 60.0, 0, 1, where=present_mask, color="#16a34a", alpha=0.18, transform=ax1.get_xaxis_transform(), label="ESTIMATE_BREATHING_RATE")
    if ac_rr.notna().any():
        ax1.plot(time_s / 60.0, ac_rr.to_numpy(dtype=float), color="#15803d", lw=0.8, alpha=0.65, label="Acconeer RR")
    if len(chunk_t_rr):
        ax1.plot(chunk_t_rr / 60.0, _rolling_median(chunk_rr, 3), "o-", color="#22c55e", lw=1.2, ms=4, label="60s weighted RR")
    if len(an_t):
        ax1.plot(an_t / 60.0, an_rr, "^-", color="#065f46", lw=1.2, ms=4, label="60s analyzer RR")
    ax1.axhline(full_analysis.resp_bpm, color="black", ls="--", lw=1.0, label=f"overall RR {full_analysis.resp_bpm:.1f}")
    ax1.set_ylim(5, 30)
    ax1.set_ylabel("RR (BPM)")
    ax1.set_title("Respiration-rate trend (whole record, no full raw trace)")
    ax1.grid(True, alpha=0.55)
    ax1.legend(loc="upper right", ncol=4, fontsize=8)

    ax2 = fig.add_subplot(grid[2], sharex=ax0)
    if len(chunk_t_hr):
        ax2.plot(chunk_t_hr / 60.0, _rolling_median(chunk_hr, 3), "o-", color="#d62728", lw=1.2, ms=4, label="60s weighted HR PSD")
    if len(an_t):
        ax2.plot(an_t / 60.0, an_hr, "^-", color="#1f77b4", lw=1.2, ms=4, label="60s analyzer HR")
    ax2.axhline(full_analysis.heart_bpm, color="black", ls="--", lw=1.0, label=f"overall HR {full_analysis.heart_bpm:.1f}")
    ax2.axhline(tail_analysis.heart_bpm, color="#777777", ls=":", lw=1.0, label=f"last 60s HR {tail_analysis.heart_bpm:.1f}")
    ax2.set_ylim(40, 120)
    ax2.set_ylabel("HR (BPM)")
    ax2.set_title("Heart-rate trend")
    ax2.grid(True, alpha=0.55)
    ax2.legend(loc="upper right", ncol=4, fontsize=8)

    ax3 = fig.add_subplot(grid[3], sharex=ax0)
    if len(an_t):
        ax3.plot(an_t / 60.0, an_hr_conf, color="#1f77b4", lw=1.2, label="HR confidence")
        ax3.plot(an_t / 60.0, an_rr_conf, color="#15803d", lw=1.2, label="RR confidence")
    if len(chunk_t_hr):
        ax3.plot(chunk_t_hr / 60.0, chunk_hr_conf, color="#d62728", lw=0.9, alpha=0.65, label="weighted HR PSD confidence")
    ax3.set_xlabel("Elapsed time (min)")
    ax3.set_ylabel("Confidence")
    ax3.set_title("Window confidence / quality")
    ax3.grid(True, alpha=0.55)
    ax3.legend(loc="upper right", ncol=3, fontsize=8)
    ax3.set_xlim(0, duration_s / 60.0)

    overview_path = output_dir / f"{csv_path.stem}_sleep_overview.png"
    fig.savefig(overview_path, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: only short excerpts plus full-record PSDs.
    excerpt_centers = [min(60.0, duration_s / 2), duration_s / 2, max(duration_s - 60.0, duration_s / 2)]
    fig2 = plt.figure(figsize=(16, 10.5), dpi=dpi)
    grid2 = fig2.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.1], hspace=0.42, wspace=0.22)
    fig2.suptitle("A121 sleep short waveform excerpts + spectra (avoids plotting all 15 min)", fontsize=15, fontweight="bold")
    for i, center in enumerate(excerpt_centers):
        start = max(0.0, center - 15.0)
        end = min(duration_s, center + 15.0)
        mask = (time_s >= start) & (time_s <= end)
        local_t = time_s[mask] - start
        axr = fig2.add_subplot(grid2[i, 0])
        axh = fig2.add_subplot(grid2[i, 1])
        resp_seg = resp_signal[mask]
        heart_seg = heart_signal[mask]
        if len(resp_seg):
            resp_norm = (resp_seg - np.nanmedian(resp_seg)) / (np.nanstd(resp_seg) + 1e-12)
            axr.plot(local_t, resp_norm, color="#16a34a", lw=1.2)
        if len(heart_seg):
            heart_norm = (heart_seg - np.nanmedian(heart_seg)) / (np.nanstd(heart_seg) + 1e-12)
            axh.plot(local_t, heart_norm, color="#1f77b4", lw=1.0)
        axr.set_title(f"Resp excerpt {start / 60.0:.1f}-{end / 60.0:.1f} min")
        axh.set_title(f"Heart-band excerpt {start / 60.0:.1f}-{end / 60.0:.1f} min")
        for ax in (axr, axh):
            ax.set_xlabel("Seconds in excerpt")
            ax.set_ylabel("z-score")
            ax.grid(True, alpha=0.55)

    axp1 = fig2.add_subplot(grid2[3, 0])
    rr_valid = (weighted_rr.bpm_axis >= 6.0) & (weighted_rr.bpm_axis <= 30.0)
    axp1.plot(weighted_rr.bpm_axis[rr_valid], weighted_rr.power[rr_valid], color="#16a34a", lw=1.4)
    axp1.axvline(full_analysis.resp_bpm, color="black", ls="--", label=f"analyzer {full_analysis.resp_bpm:.1f}")
    axp1.axvline(weighted_rr.bpm, color="#16a34a", ls=":", label=f"weighted PSD {weighted_rr.bpm:.1f}")
    axp1.set_title("Full-record respiration PSD")
    axp1.set_xlabel("Breaths/min")
    axp1.set_ylabel("Power")
    axp1.grid(True, alpha=0.55)
    axp1.legend()

    axp2 = fig2.add_subplot(grid2[3, 1])
    hr_valid = (weighted_hr.bpm_axis >= 42.0) & (weighted_hr.bpm_axis <= 120.0)
    axp2.plot(weighted_hr.bpm_axis[hr_valid], weighted_hr.power[hr_valid], color="#1f77b4", lw=1.4)
    axp2.axvline(full_analysis.heart_bpm, color="black", ls="--", label=f"analyzer {full_analysis.heart_bpm:.1f}")
    axp2.axvline(weighted_hr.bpm, color="#1f77b4", ls=":", label=f"weighted PSD {weighted_hr.bpm:.1f}")
    axp2.set_title("Full-record heart-band PSD")
    axp2.set_xlabel("Beats/min")
    axp2.set_ylabel("Power")
    axp2.grid(True, alpha=0.55)
    axp2.legend()

    excerpts_path = output_dir / f"{csv_path.stem}_sleep_excerpts_psd.png"
    fig2.savefig(excerpts_path, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved: {overview_path}")
    print(f"Saved: {excerpts_path}")
    print(f"Rows: {len(df)} | duration: {duration_s:.1f}s ({duration_s / 60.0:.2f} min) | fs: {fs:.3f} Hz")
    print(f"Target/gate: {target_m:.3f} m, {gate_min_m:.3f}-{gate_max_m:.3f} m")
    print(f"Overall analyzer: HR {full_analysis.heart_bpm:.1f} BPM conf {full_analysis.heart_confidence:.1f}; RR {full_analysis.resp_bpm:.1f} BPM conf {full_analysis.resp_confidence:.1f}")
    print(f"Last 60s analyzer: HR {tail_analysis.heart_bpm:.1f} BPM conf {tail_analysis.heart_confidence:.1f}; RR {tail_analysis.resp_bpm:.1f} BPM conf {tail_analysis.resp_confidence:.1f}")
    print(f"Weighted full PSD: HR {weighted_hr.bpm:.1f} BPM conf {weighted_hr.confidence:.1f}; RR {weighted_rr.bpm:.1f} BPM conf {weighted_rr.confidence:.1f}")
    if len(an_t):
        print(f"Sliding analyzer median: HR {np.nanmedian(an_hr):.1f} BPM; RR {np.nanmedian(an_rr):.1f} BPM")
        print(f"Sliding analyzer IQR: HR {np.nanpercentile(an_hr, 25):.1f}-{np.nanpercentile(an_hr, 75):.1f} BPM; RR {np.nanpercentile(an_rr, 25):.1f}-{np.nanpercentile(an_rr, 75):.1f} BPM")
    if ac_rr.notna().any():
        print(f"Acconeer RR median/IQR: {np.nanmedian(ac_rr):.1f} BPM; {np.nanpercentile(ac_rr, 25):.1f}-{np.nanpercentile(ac_rr, 75):.1f} BPM")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "plots")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    analyze(args.csv_path, args.output_dir, dpi=args.dpi)


if __name__ == "__main__":
    main()
