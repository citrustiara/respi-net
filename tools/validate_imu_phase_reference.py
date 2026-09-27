#!/usr/bin/env python3
"""Can the iPhone IMU label breathing phases?  Check it against the LSM6DS3.

The chest IMU sessions recorded the iPhone (RespiPhoneIMU over BLE) and the
LSM6DS3 side by side on the chest while following the breathing coach.  Both
are turned into per-breath labels independently -- the same cue alignment
used for the radar -- and compared: if two different sensors put the same
boundaries in the same place, either can serve as the labelling reference.
Free breathing, which has no cues, is labelled by the deterministic phase
detector on each sensor and compared the same way.

The iPhone's clock is tied to the host by its first BLE packet, which is off
by up to about a second and a half, differently in every run.  So the check
is made twice: as recorded, and after one clock offset per run is removed
with :func:`respi_net.chest_signal.estimate_time_offset` against the other
sensor -- the role the radar takes in a radar recording.

Runs whose iPhone stream is incomplete (BLE batches lost) are reported and
skipped.  Writes ``imu_reference_check.csv`` next to the data and, with
``--thesis-figure``, ``docs/thesis/figures/fazy_iphone_vs_lsm6ds3.png``.

    uv run python tools/validate_imu_phase_reference.py --thesis-figure
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
import numpy as np
import pandas as pd

from respi_net.breath_phases import CLASS_COLOURS, IGNORE, label_runs
from dataclasses import replace

from respi_net.chest_signal import estimate_time_offset, imu_chest_signal
from respi_net.label_alignment import align_cues
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import score_phases

SESSIONS = ROOT / "data" / "raw" / "imu" / "lsm6ds3_iphone_guided"
FIGURE = ROOT / "docs" / "thesis" / "figures" / "fazy_iphone_vs_lsm6ds3.png"
GRID_HZ = 20.0
MIN_IPHONE_COVERAGE = 0.95


def _on_grid(time_s: np.ndarray, values: np.ndarray, grid: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    if nearest:
        index = np.clip(np.searchsorted(time_s, grid), 0, len(time_s) - 1)
        return values[index]
    return np.interp(grid, time_s, values)


def _z(values: np.ndarray) -> np.ndarray:
    return (values - np.median(values)) / (np.percentile(values, 95) - np.percentile(values, 5) + 1e-12)


def check_run(cue_path: Path) -> dict[str, object] | None:
    stem = cue_path.name.replace("_cues.csv", "")
    iphone_path = cue_path.with_name(f"{stem}_iphone.csv")
    lsm_path = cue_path.with_name(f"{stem}_lsm6ds3.csv")
    if not (iphone_path.exists() and lsm_path.exists()):
        return None
    cues = pd.read_csv(cue_path)
    origin = float(cues["start_wall_ms"].iloc[0])
    duration = float(cues["end_s"].iloc[-1])
    iphone = imu_chest_signal(iphone_path, origin_ms=origin, name="iphone")
    coverage = len(iphone.time_s) / (duration * iphone.fs)
    base = {"session": cue_path.parent.name, "run": stem, "iphone_coverage": coverage}
    if coverage < MIN_IPHONE_COVERAGE:
        return {**base, "status": "skipped: incomplete iPhone stream"}
    lsm = imu_chest_signal(lsm_path, origin_ms=origin, name="lsm6ds3")
    offset_s, offset_corr = estimate_time_offset(lsm, iphone)
    corrected = replace(iphone, time_s=iphone.time_s - offset_s)
    records = cues[["start_s", "end_s", "kind"]].to_dict("records")
    grid = np.arange(0.0, duration, 1.0 / GRID_HZ)
    lsm_chest = _on_grid(lsm.time_s, lsm.chest, grid)
    result: dict[str, object] = {**base, "status": "checked", "iphone_clock_offset_s": offset_s, "offset_match": offset_corr}
    sign_phone = sign_lsm = 1.0
    for tag, phone in (("as_recorded", iphone), ("clock_fixed", corrected)):
        phone_chest = _on_grid(phone.time_s, phone.chest, grid)
        if not cues["kind"].isin(["inhale", "exhale"]).any():
            continue
        by_phone = align_cues(grid, phone_chest, records, fs=GRID_HZ)
        by_lsm = align_cues(grid, lsm_chest, records, fs=GRID_HZ)
        pairs = [
            (a.time_s - b.time_s)
            for a, b in zip(by_phone.boundaries, by_lsm.boundaries)
            if a.method in ("turn", "stop", "start") and b.method in ("turn", "stop", "start")
        ]
        delta = np.asarray(pairs)
        agreement = score_phases(by_lsm.labels, np.where(by_phone.labels == IGNORE, -9, by_phone.labels))
        result.update(
            {
                f"{tag}_lag_phone_s": by_phone.lag_s,
                f"{tag}_boundaries": len(delta),
                f"{tag}_delta_abs_median_s": float(np.median(np.abs(delta))),
                f"{tag}_delta_abs_p90_s": float(np.percentile(np.abs(delta), 90)),
                f"{tag}_label_agreement": agreement.accuracy,
            }
        )
        if tag == "clock_fixed":
            result.update(
                {
                    "lag_lsm_s": by_lsm.lag_s,
                    "fallback_phone": by_phone.summary()["fallback_share"],
                    "fallback_lsm": by_lsm.summary()["fallback_share"],
                    "signal_correlation": float(abs(np.corrcoef(phone_chest, lsm_chest)[0, 1])),
                    "_plot": (grid, by_phone.sign * _z(phone_chest), by_lsm.sign * _z(lsm_chest), by_phone, by_lsm),
                }
            )
            # Free breathing has no cue to lean on: label it with the detector.
            sign_phone, sign_lsm = by_phone.sign, by_lsm.sign
    phone_chest = _on_grid(corrected.time_s, corrected.chest, grid)
    free_phone = detect_phases(sign_phone * phone_chest, GRID_HZ)
    free_lsm = detect_phases(sign_lsm * lsm_chest, GRID_HZ)
    free = score_phases(free_lsm, np.where(free_phone == IGNORE, -9, free_phone))
    result["free_detector_agreement"] = free.accuracy
    return result


def plot(entry: dict[str, object], path: Path) -> None:
    grid, phone, lsm, by_phone, by_lsm = entry["_plot"]  # type: ignore[misc]
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(13, 4.8), sharex=True, gridspec_kw={"height_ratios": [4, 1.1], "hspace": 0.06})
    ax.plot(grid, lsm, color="#111827", lw=1.3, label="LSM6DS3 (akcelerometr, PCA)")
    ax.plot(grid, phone, color="#2563eb", lw=1.1, alpha=0.85, label="iPhone (akcelerometr, PCA)")
    for boundary in by_lsm.boundaries:
        if boundary.method in ("turn", "stop", "start"):
            ax.axvline(boundary.time_s, color="#111827", lw=0.6, alpha=0.5)
    for boundary in by_phone.boundaries:
        if boundary.method in ("turn", "stop", "start"):
            ax.axvline(boundary.time_s, color="#2563eb", lw=0.6, ls=(0, (2, 2)))
    ax.set_ylabel("ruch (znormalizowany)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(
        f"{entry['run']}: iPhone po korekcie zegara o {float(entry['iphone_clock_offset_s']):+.2f} s; granice iPhone vs LSM6DS3: "
        f"mediana |różnicy| {float(entry['clock_fixed_delta_abs_median_s']) * 1000:.0f} ms, "
        f"zgodność etykiet {float(entry['clock_fixed_label_agreement']) * 100:.1f}%",
        loc="left",
        fontsize=10,
    )
    for labels, lo, hi in ((by_lsm.labels, 0.52, 0.98), (by_phone.labels, 0.02, 0.48)):
        for start, stop, label in label_runs(labels):
            if label != IGNORE:
                end = grid[stop] if stop < len(grid) else grid[-1]
                bars.axvspan(grid[start], end, ymin=lo, ymax=hi, color=CLASS_COLOURS[label], alpha=0.9, lw=0)
    bars.set_yticks([0.25, 0.75], ["iPhone", "LSM6DS3"])
    bars.set_ylim(0, 1)
    bars.set_xlim(grid[0], grid[-1])
    bars.set_xlabel("czas od startu komend [s]")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", type=Path, default=SESSIONS)
    parser.add_argument("--thesis-figure", action="store_true")
    args = parser.parse_args()
    entries = [entry for path in sorted(args.sessions.glob("*/run_*_cues.csv")) if (entry := check_run(path))]
    if not entries:
        raise SystemExit("No IMU sessions with cues found.")
    table = pd.DataFrame([{k: v for k, v in entry.items() if not k.startswith("_")} for entry in entries])
    table.to_csv(args.sessions / "imu_reference_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    checked = [entry for entry in entries if "_plot" in entry]
    if checked:
        for tag in ("as_recorded", "clock_fixed"):
            deltas = [float(entry[f"{tag}_delta_abs_median_s"]) * 1000 for entry in checked]
            agreement = [float(entry[f"{tag}_label_agreement"]) * 100 for entry in checked]
            print(f"{tag}: median |iPhone - LSM6DS3| boundary difference per run {np.round(deltas).tolist()} ms, label agreement {np.round(agreement, 1).tolist()} %")
        if args.thesis_figure:
            plot(checked[0], FIGURE)
            print(f"Figure: {FIGURE}")
    summary = {
        "checked_runs": int((table["status"] == "checked").sum()),
        "skipped_runs": int((table["status"] != "checked").sum()),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
