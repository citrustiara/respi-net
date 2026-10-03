#!/usr/bin/env python3
"""The phone's single-axis trace and its orientation distance, each next to the radar, around the holds of one recording.

Top: the usual phone trace (accelerometer on one axis).  Bottom: the distance of the phone's 2-D tilt from its rest orientation
(:mod:`respi_net.phone_orientation`).  The radar (normalised, black) is drawn in both for comparison only; neither phone trace
uses it, except for the sign of the single-axis trace.  The table gives, for each episode, how much of a normal breath's size the
trace keeps in the hold, and when it drops after the radar's exhale.

    uv run python tools/plot_orientation_vs_pca.py --session self_lying_3min_01 --out outputs/orientation_vs_pca.png
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
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from respi_net.chest_signal import a121_chest_signal
from respi_net.phone_annotation import oriented_phone
from respi_net.phone_orientation import breath_troughs, gravity_tilt, orientation_distance

SESSIONS = ROOT / "data" / "raw" / "nn" / "a121_iphone"
RUN = "run_01_nn_self_3min"
GRID_HZ = 20.0
# Episodes read off the radar plateau: (hold start, hold end = start of the radar's exhale, end of that exhale).
EPISODES = {
    "self_lying_3min_01": [(49.65, 57.95, 60.05), (81.2, 93.4, None), (95.0, 103.3, 105.55), (110.3, 124.7, None)],
}


def load(session: str):
    prefix = SESSIONS / session / RUN
    grid, trace, _ = oriented_phone(prefix, GRID_HZ)
    cues = pd.read_csv(f"{prefix}_cues.csv")
    origin = float(cues["start_wall_ms"].iloc[0])
    radar_signal = a121_chest_signal(f"{prefix}_a121.csv", origin_ms=origin)
    radar = np.interp(grid, radar_signal.time_s, radar_signal.chest)
    df = pd.read_csv(f"{prefix}_iphone.csv")
    t = (df["Time_ms"].to_numpy(float) - origin) / 1000.0
    order = np.argsort(t, kind="stable")
    fs = 1.0 / float(np.median(np.diff(t[order])))
    tilt100 = gravity_tilt(df[["ax", "ay", "az"]].to_numpy(float)[order], fs)
    tilt = np.column_stack([np.interp(grid, t[order], tilt100[:, k]) for k in range(2)])
    return grid, trace, radar, tilt


def normalise(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x - lo) / (hi - lo + 1e-12)


def episode_row(grid, signal, base_idx, start, end, exhale_end):
    """Fraction of a normal breath size kept in the middle of the hold, and when the signal falls after the hold ends."""

    before = (grid >= start - 30) & (grid < start - 3)
    crests, _ = find_peaks(signal[before], prominence=0.2 * np.subtract(*np.percentile(signal[before], [95, 5])))
    troughs = breath_troughs(signal[before], GRID_HZ, prominence=0.2)
    size = float(np.median(signal[before][crests]) - np.median(signal[before][troughs])) if len(crests) and len(troughs) else float("nan")
    rest = float(np.median(signal[before][troughs])) if len(troughs) else float(np.percentile(signal[before], 10))
    middle = (grid >= start + 0.25 * (end - start)) & (grid <= end - 0.25 * (end - start))
    level = float(np.median(signal[middle]))
    kept = (level - rest) / size if size > 0 else float("nan")
    after = (grid >= end - 0.5) & (grid <= end + 5)
    below = np.flatnonzero(after & (signal < rest + 0.5 * (level - rest)))
    drop = float(grid[below[0]] - end) if len(below) and level > rest else float("nan")
    return kept, drop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default="self_lying_3min_01")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "orientation_vs_pca.png")
    parser.add_argument("--start", type=float, default=40.0)
    parser.add_argument("--end", type=float, default=130.0)
    args = parser.parse_args()
    grid, trace, radar, tilt = load(args.session)
    troughs = breath_troughs(trace, GRID_HZ)
    distance = orientation_distance(tilt, troughs, GRID_HZ)
    episodes = EPISODES.get(args.session, [])
    window = (grid >= args.start) & (grid <= args.end)
    r_lo, r_hi = np.percentile(radar[window], [1, 99])
    fig = plt.figure(figsize=(13, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.9])
    top, bottom, table = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None), fig.add_subplot(gs[2])
    table.axis("off")
    rows = []
    for ax, signal, colour, title in (
        (top, trace, "#ea580c", "Telefon, rzut na jedną oś (dotychczasowy sygnał) i radar"),
        (bottom, distance, "#0891b2", "Telefon, odległość pełnej orientacji od położenia spoczynkowego i radar"),
    ):
        lo, hi = np.percentile(signal[window], [1, 99])
        ax.plot(grid[window], normalise(radar[window], r_lo, r_hi), color="#111827", lw=1.3, label="radar")
        ax.plot(grid[window], normalise(signal[window], lo, hi), color=colour, lw=1.6, label="telefon")
        for start, end, exhale_end in episodes:
            ax.axvspan(start, end, color="#f59e0b", alpha=0.15, lw=0)
            if exhale_end:
                ax.axvspan(end, exhale_end, color="#22c55e", alpha=0.15, lw=0)
        ax.set_xlim(args.start, args.end)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.25)
        ax.set_ylabel("znormalizowane")
    top.tick_params(labelbottom=False)
    bottom.set_xlabel("czas od startu nagrania [s]")
    cells = []
    for start, end, exhale_end in episodes:
        old = episode_row(grid, trace, None, start, end, exhale_end)
        new = episode_row(grid, distance, None, start, end, exhale_end)
        cells.append([f"{start:.0f}-{end:.0f} s", f"{old[0]*100:.0f}%", f"{new[0]*100:.0f}%", "-" if np.isnan(old[1]) else f"{old[1]:+.1f} s", "-" if np.isnan(new[1]) else f"{new[1]:+.1f} s"])
        rows.append({"hold": [start, end], "kept_old": old[0], "kept_new": new[0], "drop_old_s": old[1], "drop_new_s": new[1]})
    t = table.table(
        cellText=cells,
        colLabels=[
            "pauza\n(z plateau radaru)",
            "poziom w pauzie, jedna oś\n[% zwykłego oddechu]",
            "poziom w pauzie, orientacja\n[% zwykłego oddechu]",
            "spadek po wydechu radaru\njedna oś [s]",
            "spadek po wydechu radaru\norientacja [s]",
        ],
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 2.2)
    fig.legend(
        handles=[Line2D([], [], color="#111827", lw=2, label="radar"), Line2D([], [], color="#ea580c", lw=2, label="telefon, jedna oś"), Line2D([], [], color="#0891b2", lw=2, label="telefon, orientacja"),
                 plt.Rectangle((0, 0), 1, 1, color="#f59e0b", alpha=0.3, label="pauza (plateau radaru)"), plt.Rectangle((0, 0), 1, 1, color="#22c55e", alpha=0.3, label="wydech radaru po pauzie")],
        loc="outside lower center", ncols=5, frameon=False,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(json.dumps(rows, indent=1))
    print(f"Figure: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
