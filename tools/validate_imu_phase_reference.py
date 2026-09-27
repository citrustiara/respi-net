#!/usr/bin/env python3
"""Can an IMU recording be labelled per breath?  iPhone and LSM6DS3, each on its own.

Every chest IMU recording made with the breathing coach is labelled against
its own cues with the same per-breath alignment used for the radar
(:func:`respi_net.label_alignment.align_cues`) and described by the same
numbers as the radar coach runs: the lag behind the cues, how well the motion
follows the cued shape, how often a boundary has no clear turn, and how far
the corrected boundaries move.  The deterministic phase detector is then run
on the same signal without any cues; its agreement with the cue-based labels
says how well free breathing -- which has no cues -- could be labelled from
that sensor.

The iPhone and LSM6DS3 files are separate recordings and are never compared
with or aligned to each other.  Runs whose iPhone stream is incomplete (BLE
batches lost) are reported and skipped.  Writes ``imu_reference_check.csv``
next to the data and, with ``--thesis-figure``,
``docs/thesis/figures/fazy_imu_etykiety.png``.

    uv run python tools/validate_imu_phase_reference.py --thesis-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from respi_net.breath_phases import CLASS_COLOURS, IGNORE, label_runs
from respi_net.chest_signal import ChestSignal, imu_chest_signal
from respi_net.label_alignment import AlignmentResult, align_cues, cue_phases
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import score_phases

SESSIONS = ROOT / "data" / "raw" / "imu" / "lsm6ds3_iphone_guided"
FIGURE = ROOT / "docs" / "thesis" / "figures" / "fazy_imu_etykiety.png"
GRID_HZ = 20.0
MIN_COVERAGE = 0.95
DEVICES = (("iphone", "iPhone"), ("lsm6ds3", "LSM6DS3"))


def _on_grid(signal: ChestSignal, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, signal.time_s, signal.chest)


def check_recording(cue_path: Path, device: str) -> dict[str, object] | None:
    stem = cue_path.name.replace("_cues.csv", "")
    path = cue_path.with_name(f"{stem}_{device}.csv")
    if not path.exists():
        return None
    cues = pd.read_csv(cue_path)
    origin = float(cues["start_wall_ms"].iloc[0])
    duration = float(cues["end_s"].iloc[-1])
    signal = imu_chest_signal(path, origin_ms=origin, name=device)
    coverage = len(signal.time_s) / (duration * signal.fs)
    entry: dict[str, object] = {"session": cue_path.parent.name, "run": stem, "device": device, "coverage": coverage}
    if coverage < MIN_COVERAGE:
        return {**entry, "status": "skipped: incomplete stream"}
    records = cues[["start_s", "end_s", "kind"]].to_dict("records")
    if not any(phase.label != IGNORE and phase.kind in ("inhale", "exhale") for phase in cue_phases(records)):
        return {**entry, "status": "skipped: no cued breaths"}
    grid = np.arange(0.0, duration, 1.0 / GRID_HZ)
    chest = _on_grid(signal, grid)
    result = align_cues(grid, chest, records, fs=GRID_HZ)
    summary = result.summary()
    detected = detect_phases(result.sign * chest, GRID_HZ)
    free = score_phases(result.labels, np.where(detected == IGNORE, -9, detected))
    entry.update(
        {
            "status": "checked",
            "lag_s": result.lag_s,
            "cue_match": result.correlation,
            "fallback_share": summary["fallback_share"],
            "shift_median_s": summary["shift_median_s"],
            "shift_iqr_s": summary["shift_iqr_s"],
            "shift_abs_p90_s": summary["shift_abs_p90_s"],
            "detector_agreement": free.accuracy,
            "detector_macro_f1": free.macro_f1,
            "_plot": (grid, result.sign * chest, result, records),
        }
    )
    return entry


def _draw(ax: plt.Axes, bars: plt.Axes, entry: dict[str, object], name: str) -> None:
    grid, chest, result, records = entry["_plot"]  # type: ignore[misc]
    assert isinstance(result, AlignmentResult)
    scale = np.percentile(chest, 95) - np.percentile(chest, 5)
    chest = (chest - np.median(chest)) / (scale + 1e-12)
    for start, stop, label in label_runs(result.labels):
        if label != IGNORE:
            end = grid[stop] if stop < len(grid) else grid[-1]
            ax.axvspan(grid[start], end, color=CLASS_COLOURS[label], alpha=0.28, lw=0)
    ax.plot(grid, chest, color="#111827", lw=1.1)
    for boundary in result.boundaries:
        ax.axvline(boundary.expected_s, color="#6b7280", lw=0.6, ls=(0, (3, 3)))
    ax.set_ylabel("ruch (znorm.)")
    ax.set_xlim(grid[0], grid[-1])
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.25)
    ax.set_title(
        f"{name}, {entry['run']}: opóźnienie {float(entry['lag_s']):.2f} s, dopasowanie do komend {float(entry['cue_match']):.2f}, "
        f"bez wyraźnego zwrotu {float(entry['fallback_share']) * 100:.0f}%, IQR przesunięć {float(entry['shift_iqr_s']):.2f} s",
        loc="left",
        fontsize=9,
    )
    cue_labels = np.full(len(grid), IGNORE, dtype=np.int8)
    for phase in cue_phases(records):
        if phase.label != IGNORE:
            cue_labels[(grid >= phase.start_s + result.lag_s) & (grid < phase.end_s + result.lag_s)] = phase.label
    for labels, lo, hi in ((cue_labels, 0.52, 0.98), (result.labels, 0.02, 0.48)):
        for start, stop, label in label_runs(labels):
            if label != IGNORE:
                end = grid[stop] if stop < len(grid) else grid[-1]
                bars.axvspan(grid[start], end, ymin=lo, ymax=hi, color=CLASS_COLOURS[label], alpha=0.9, lw=0)
    bars.set_yticks([0.25, 0.75], ["skorygowane", "komendy + opóźn."])
    bars.set_ylim(0, 1)
    bars.set_xlim(grid[0], grid[-1])


def plot(entries: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(
        4, 1, figsize=(13, 8.2), gridspec_kw={"height_ratios": [4, 1, 4, 1], "hspace": 0.35}
    )
    for (ax, bars), entry in zip(((axes[0], axes[1]), (axes[2], axes[3])), entries):
        _draw(ax, bars, entry, dict(DEVICES)[str(entry["device"])])
    axes[-1].set_xlabel("czas od startu komend w danym nagraniu [s]")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", type=Path, default=SESSIONS)
    parser.add_argument("--thesis-figure", action="store_true")
    args = parser.parse_args()
    entries = [
        entry
        for cue_path in sorted(args.sessions.glob("*/run_*_cues.csv"))
        for device, _ in DEVICES
        if (entry := check_recording(cue_path, device))
    ]
    if not entries:
        raise SystemExit("No IMU recordings with cues found.")
    table = pd.DataFrame([{k: v for k, v in entry.items() if not k.startswith("_")} for entry in entries])
    table.to_csv(args.sessions / "imu_reference_check.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    checked = table[table["status"] == "checked"]
    if len(checked):
        print()
        print(checked.groupby("device")[["lag_s", "cue_match", "fallback_share", "shift_iqr_s", "detector_agreement"]].mean().round(3).to_string())
    if args.thesis_figure:
        chosen = [
            next(entry for entry in entries if entry.get("device") == device and "_plot" in entry)
            for device, _ in DEVICES
            if any(entry.get("device") == device and "_plot" in entry for entry in entries)
        ]
        if chosen:
            plot(chosen, FIGURE)
            print(f"Figure: {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
