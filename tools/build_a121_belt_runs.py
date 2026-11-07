#!/usr/bin/env python3
"""Labelled A121 runs from radar + strain-belt recordings, labels from the belt alone.

For every recording folder holding ``a121.csv`` and ``belt.csv`` (both on host
epoch milliseconds, as a plain radar + belt capture writes them), or the
folders given:

1. :func:`respi_net.chest_signal.strain_belt_chest_signal` cleans the belt trace;
   a recording whose breaths are not well above the belt noise is skipped;
2. :func:`detect_phases` (offline, 2 s shortest hold, as
   ``tools/label_strain_belt_runs.py``) labels the belt, holds merged into one;
3. the labels go onto the radar's time grid (nearest belt sample); the radar's
   first seconds, belt dropouts and anything outside the belt stream are IGNORE.
   The belt-to-radar delay is measured from the breathing and reported -- on the
   first recording it was 0.0 s, so the labels are not shifted; a recording whose
   radar hardly follows the belt (correlation under
   :data:`MIN_BELT_RADAR_CORRELATION`) is skipped;
4. a check: the share of samples in which the radar chest rises, per belt class.

Each run goes to ``data/processed/breath_phases/a121_belt/runs/<run>.npz``
(nn_dataset format, subject S01), where ``tools/build_artificial_data.py``
takes it into the fine-tuning sets as a training run with its copies.
``plots/<run>.png`` shows the radar with the belt labels.

The belt trace itself, with the same labels, goes to ``belt_runs/<run>__belt.npz``
in the form of the supervisor's belt recordings
(:func:`respi_net.szymanski_dataset.to_labelled_run`): 20 Hz, rising on inhale,
unit spread, zero echo channel.  The builder takes it into the pre-training sets.

    uv run python tools/build_a121_belt_runs.py
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
from scipy.signal import butter, sosfiltfilt

from respi_net.breath_phases import CLASS_COLOURS, CLASS_NAMES, IGNORE, label_runs, merge_holds
from respi_net.chest_signal import ChestSignal, a121_chest_signal, light_filter, strain_belt_chest_signal
from respi_net.nn_dataset import labelled_run, save_run
from respi_net.phase_baseline import detect_phases

RAW_DIR = ROOT / "data" / "raw" / "nn" / "a121_belt"
OUT = ROOT / "data" / "processed" / "breath_phases" / "a121_belt"
BELT_MIN_HOLD_S = 2.0  # as tools/label_strain_belt_runs.py
MIN_DEPTH_TO_NOISE = 50.0  # worn belt ~500, belt on the desk ~13
WARMUP_S = 3.0  # the radar settles after its session starts
MAX_LAG_S = 2.0
# Below this belt-radar correlation (0.05-0.7 Hz) the radar hardly shows the breathing the belt labels: the
# chest is outside the chosen range or the person moved.  Earlier runs had 0.89-0.98; the 2026-11-07 runs at
# 1.3 m had 0.45-0.56, and 0.22 in the first minutes, where the radar rose almost as often in holds as in inhales.
MIN_BELT_RADAR_CORRELATION = 0.3
BELT_RUN_FS = 20.0  # the grid of every run in the datasets


def recording_dirs(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    return sorted(p for p in RAW_DIR.iterdir() if (p / "a121.csv").exists() and (p / "belt.csv").exists())


def run_id(folder: Path) -> str:
    return f"a121_belt_{folder.name}"


def belt_to_radar_lag_s(time_s: np.ndarray, belt: np.ndarray, radar: np.ndarray, fs: float) -> tuple[float, float]:
    """Delay of the radar behind the belt (positive: radar later) and the correlation there, both band-passed 0.05-0.7 Hz."""

    sos = butter(2, (0.05, 0.7), "bandpass", fs=fs, output="sos")
    ok = time_s >= WARMUP_S
    b, r = sosfiltfilt(sos, belt[ok]), sosfiltfilt(sos, radar[ok])
    best = (0.0, 0.0)
    for lag in range(-int(MAX_LAG_S * fs), int(MAX_LAG_S * fs) + 1):
        x, y = (b[: len(b) - lag], r[lag:]) if lag >= 0 else (b[-lag:], r[: len(r) + lag])
        corr = float(np.corrcoef(x, y)[0, 1])
        if abs(corr) > abs(best[1]):
            best = (lag / fs, corr)
    return best


def belt_trace_run(folder: Path, belt: ChestSignal, belt_labels: np.ndarray):
    """The belt trace as a run like the supervisor's belt recordings, labelled with ``belt_labels``."""

    time_s = np.arange(belt.time_s[0], belt.time_s[-1], 1.0 / BELT_RUN_FS)
    values = np.interp(time_s, belt.time_s, belt.chest)
    values = light_filter(values - np.median(values), BELT_RUN_FS, 2.0)
    spread = float(np.percentile(values, 95) - np.percentile(values, 5)) or 1.0
    nearest = np.clip(np.searchsorted(belt.time_s, time_s), 0, len(belt.time_s) - 1)
    signal = ChestSignal("strain_belt", time_s - time_s[0], BELT_RUN_FS, values / spread, "spread", values,
                         np.zeros(len(time_s)), {"echo_available": False})
    name = f"{run_id(folder)}__belt"
    meta = {"dataset": "a121_belt", "sensor": "strain_belt", "recording": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
            "label_origin": f"strain belt, offline detect_phases (min hold {BELT_MIN_HOLD_S:g} s), holds merged"}
    return labelled_run(name, signal, belt_labels[nearest], group=name, meta=meta)


def plot_run(path: Path, time_s: np.ndarray, chest: np.ndarray, labels: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.plot(time_s, chest, color="k", lw=0.6)
    for start, stop, label in label_runs(labels):
        if label != IGNORE:
            ax.axvspan(time_s[start], time_s[min(stop, len(time_s) - 1)], color=CLASS_COLOURS[label], alpha=0.25, lw=0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("A121 chest [mm]")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def build_run(folder: Path, out: Path) -> dict[str, object]:
    radar_frame = pd.read_csv(folder / "a121.csv")
    origin_ms = float(radar_frame["Timestamp_ms"].iloc[0])
    belt = strain_belt_chest_signal(folder / "belt.csv", origin_ms=origin_ms)
    row: dict[str, object] = {"run": run_id(folder), "depth_to_noise": belt.meta.get("depth_to_noise")}
    if belt.meta["depth_to_noise"] < MIN_DEPTH_TO_NOISE:
        row["status"] = "skipped: no breathing above belt noise"
        return row
    radar = a121_chest_signal(radar_frame, origin_ms=origin_ms)
    belt_labels = merge_holds(detect_phases(belt.chest, belt.fs, causal=False, min_hold_s=BELT_MIN_HOLD_S))
    belt_labels[~belt.valid_mask()] = IGNORE  # belt dropouts carry no labels
    nearest = np.clip(np.searchsorted(belt.time_s, radar.time_s), 0, len(belt.time_s) - 1)
    labels = belt_labels[nearest].astype(np.int8)
    labels[(radar.time_s < max(float(belt.time_s[0]), WARMUP_S)) | (radar.time_s > float(belt.time_s[-1]))] = IGNORE
    lag_s, corr = belt_to_radar_lag_s(radar.time_s, belt.chest[nearest], radar.chest, radar.fs)
    if abs(corr) < MIN_BELT_RADAR_CORRELATION:
        # The belt trace is fine on its own: it still goes to pre-training; only the radar run is left out.
        (out / "runs" / f"{run_id(folder)}.npz").unlink(missing_ok=True)
        belt_run = belt_trace_run(folder, belt, belt_labels)
        save_run(belt_run, out / "belt_runs" / f"{belt_run.run_id}.npz")
        row.update({"status": f"belt trace only: belt-radar correlation {corr:+.2f} below {MIN_BELT_RADAR_CORRELATION}",
                    "belt_radar_lag_s": round(lag_s, 3), "belt_minutes": round(belt_run.labelled_s / 60.0, 1)})
        return row
    rising = np.gradient(radar.chest) > 0
    rises = {name: round(float(np.mean(rising[labels == c])), 3) for c, name in enumerate(CLASS_NAMES) if np.any(labels == c)}
    meta = {
        "dataset": "a121_belt",
        "recording": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
        "label_origin": f"strain belt, offline detect_phases (min hold {BELT_MIN_HOLD_S:g} s), holds merged, nearest sample on the radar grid",
        "belt_radar_lag_s": round(lag_s, 3),
        "belt_radar_band_correlation": round(corr, 3),
        "radar_rising_share_by_belt_class": rises,
    }
    run = labelled_run(run_id(folder), radar, labels, group=run_id(folder), meta=meta)
    save_run(run, out / "runs" / f"{run.run_id}.npz")
    belt_run = belt_trace_run(folder, belt, belt_labels)
    save_run(belt_run, out / "belt_runs" / f"{belt_run.run_id}.npz")
    plot_run(out / "plots" / f"{run.run_id}.png", radar.time_s, radar.chest, labels,
             f"{run.run_id}  ·  belt->radar lag {lag_s:+.2f} s, r {corr:+.2f}  ·  radar rising in inhale {rises.get('inhale', 0):.0%}, exhale {rises.get('exhale', 0):.0%}")
    row.update({"status": "labelled", "minutes": round(run.labelled_s / 60.0, 1), "belt_minutes": round(belt_run.labelled_s / 60.0, 1), **{k: v for k, v in meta.items() if k.startswith(("belt_", "radar_"))}})
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recordings", nargs="*", type=Path, help="recording folders with a121.csv and belt.csv")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    folders = recording_dirs(args.recordings)
    if not folders:
        print(f"No a121.csv + belt.csv folders in {RAW_DIR}.")
        return 1
    rows = [build_run(folder, args.output_dir) for folder in folders]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(json.dumps(row))
    print(f"Runs, plots and summary.json in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
