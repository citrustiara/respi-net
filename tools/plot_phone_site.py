#!/usr/bin/env python3
"""The radar with and without the phone on the middle of the chest, and that phone's own trace (pattern ``phone_site_1m``).

Two runs in the same position, one after the other: the phone strapped on the middle of the chest, screen towards the radar,
and the phone off the body, out of the radar's view.  Prints, per run, the echo at the strongest range bin, the breath size,
the radar detector against the cues and the level kept in the two holds; draws both radar traces and the phone's trace.

    uv run python tools/plot_phone_site.py --with-phone sit_phone_mid_01 --without-phone sit_nophone_01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from respi_net.breath_phases import IGNORE, merge_holds
from respi_net.chest_signal import a121_chest_signal
from respi_net.label_alignment import cue_phases, estimate_lag
from respi_net.phase_baseline import detect_phases
from respi_net.phone_annotation import PHONE_DETECTOR, compare_labels, oriented_phone
from respi_net.phone_orientation import run_orientation_distance

SESSIONS = ROOT / "data" / "raw" / "nn" / "a121_iphone"
RUN = "run_01_phone_site_1m"
GRID_HZ = 20.0
CUE_COLOURS = {"inhale": "#3b82f6", "exhale": "#22c55e", "hold": "#f59e0b"}
# Pattern times: paced breaths 10-25 s, deep inhale 25-28 s, hold after it 28-36 s, hold after exhale 55-63 s.
PACED, INHALE_HOLD, EXHALE_HOLD = (10.0, 25.0), (29.0, 35.0), (56.0, 62.0)


def cue_score(grid: np.ndarray, signal: np.ndarray, records: list[dict], **detector) -> float:
    phases = cue_phases(records)
    lag, _, _ = estimate_lag(grid, np.gradient(signal) * GRID_HZ, phases, np.arange(-1.5, 1.5001, 0.025))
    cues = np.full(len(grid), IGNORE, dtype=np.int8)
    for phase in phases:
        if phase.label != IGNORE:
            cues[(grid >= phase.start_s + lag) & (grid < phase.end_s + lag)] = phase.label
    body = grid >= PACED[0]
    labels = merge_holds(detect_phases(signal, GRID_HZ, **detector))
    return compare_labels(np.where(body, merge_holds(cues), IGNORE), np.where(body, labels, IGNORE), grid, tolerance_s=0.5)["accuracy"]


def kept(grid: np.ndarray, signal: np.ndarray, window: tuple[float, float]) -> float:
    """Median level in ``window`` as a fraction of the paced breaths' 5-95 percentile range, from its bottom."""

    lo, hi = np.percentile(signal[(grid >= PACED[0]) & (grid < PACED[1])], [5, 95])
    return float((np.median(signal[(grid >= window[0]) & (grid < window[1])]) - lo) / (hi - lo))


def peak_echo_db(path: Path) -> tuple[float, float]:
    df = pd.read_csv(path, usecols=["Distances_m", "Amplitude"])
    distances = np.array(json.loads(df["Distances_m"].iloc[0]))
    mean = np.mean([json.loads(a) for a in df["Amplitude"]], axis=0)
    body = distances > 0.4  # skip the direct leakage near the sensor
    best = int(np.argmax(np.where(body, mean, 0.0)))
    return float(distances[best]), float(20 * np.log10(mean[best]))


def load(session: str):
    prefix = SESSIONS / session / RUN
    cues = pd.read_csv(f"{prefix}_cues.csv")
    origin = float(cues["start_wall_ms"].iloc[0])
    grid = np.arange(0.0, float(cues["end_s"].iloc[-1]), 1.0 / GRID_HZ)
    radar = a121_chest_signal(f"{prefix}_a121.csv", origin_ms=origin)
    return prefix, cues, grid, np.interp(grid, radar.time_s, radar.chest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-phone", default="sit_phone_mid_01")
    parser.add_argument("--without-phone", default="sit_nophone_01")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "phone_site_mid_vs_none.png")
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, constrained_layout=True)
    report = {}
    for ax, session, title in ((axes[0], args.without_phone, "Radar, telefon poza klatką"), (axes[1], args.with_phone, "Radar, telefon na środku klatki")):
        prefix, cues, grid, radar = load(session)
        records = cues[["start_s", "end_s", "kind"]].to_dict("records")
        distance_m, echo_db = peak_echo_db(Path(f"{prefix}_a121.csv"))
        report[session] = {
            "peak_echo_m": distance_m,
            "peak_echo_db": echo_db,
            "breath_mm_p5_p95": float(np.subtract(*np.percentile(radar[(grid >= PACED[0]) & (grid < PACED[1])], [95, 5]))),
            "radar_detector_vs_cues": cue_score(grid, radar, records),
            "radar_kept_inhale_hold": kept(grid, radar, INHALE_HOLD),
            "radar_kept_exhale_hold": kept(grid, radar, EXHALE_HOLD),
        }
        for _, cue in cues.iterrows():
            if cue["kind"] in CUE_COLOURS:
                ax.axvspan(cue["start_s"], cue["end_s"], color=CUE_COLOURS[cue["kind"]], alpha=0.15, lw=0)
        ax.plot(grid, radar - np.median(radar), color="#111827", lw=1.3)
        ax.set_title(f"{title} ({session}): maksimum echa {echo_db:.0f} dB", loc="left", fontsize=11)
        ax.set_ylabel("przesunięcie [mm]")
        ax.grid(alpha=0.25)
    prefix, cues, grid, radar = load(args.with_phone)
    records = cues[["start_s", "end_s", "kind"]].to_dict("records")
    phone_grid, trace, sign = oriented_phone(prefix, GRID_HZ)
    orientation = run_orientation_distance(prefix, phone_grid, trace)
    report[args.with_phone].update({
        "phone_sign": sign,
        "phone_velocity_corr_with_radar": float(np.corrcoef(np.gradient(np.interp(grid, phone_grid, trace))[grid >= PACED[0]], np.gradient(radar)[grid >= PACED[0]])[0, 1]),
        "phone_detector_vs_cues": cue_score(phone_grid, trace, records, **PHONE_DETECTOR),
        "orientation_detector_vs_cues": cue_score(phone_grid, orientation, records, **PHONE_DETECTOR),
        "phone_kept_inhale_hold": kept(phone_grid, trace, INHALE_HOLD),
        "phone_kept_exhale_hold": kept(phone_grid, trace, EXHALE_HOLD),
    })
    ax = axes[2]
    for _, cue in cues.iterrows():
        if cue["kind"] in CUE_COLOURS:
            ax.axvspan(cue["start_s"], cue["end_s"], color=CUE_COLOURS[cue["kind"]], alpha=0.15, lw=0)
    lo, hi = np.percentile(trace, [1, 99])
    o_lo, o_hi = np.percentile(orientation, [1, 99])
    ax.plot(phone_grid, trace, color="#ea580c", lw=1.4)
    ax.plot(phone_grid, lo + (orientation - o_lo) / (o_hi - o_lo) * (hi - lo), color="#0891b2", lw=1.0)
    ax.set_title(f"Telefon na środku klatki ({args.with_phone}): jedna oś i orientacja", loc="left", fontsize=11)
    ax.set_ylabel("telefon [mg]")
    ax.set_xlabel("czas od startu nagrania [s]")
    ax.grid(alpha=0.25)
    fig.legend(
        handles=[Line2D([], [], color="#111827", lw=2, label="radar"), Line2D([], [], color="#ea580c", lw=2, label="telefon, jedna oś"),
                 Line2D([], [], color="#0891b2", lw=2, label="telefon, orientacja"), Patch(color=CUE_COLOURS["inhale"], alpha=0.3, label="komenda: wdech"),
                 Patch(color=CUE_COLOURS["exhale"], alpha=0.3, label="komenda: wydech"), Patch(color=CUE_COLOURS["hold"], alpha=0.3, label="komenda: pauza")],
        loc="outside lower center", ncols=6, frameon=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(json.dumps(report, indent=1))
    print(f"Figure: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
