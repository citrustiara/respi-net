#!/usr/bin/env python3
"""Breathing-phase labels from the strain belt, with no hand labelling.

For every ``data/raw/strain_belt/strain_belt_*.csv`` and every belt file of a
guided radar session, ``data/raw/nn/<kind>/<session>/run_*_belt.csv`` (or the
files given):

1. :func:`respi_net.chest_signal.strain_belt_chest_signal` cleans the belt trace
   (failed reads, spikes, BLE arrival jitter) and removes its drift;
2. a run whose breaths are not well above the sensor noise (belt not worn,
   loose) is skipped rather than labelled;
3. :func:`detect_phases` (offline) labels every sample from the belt alone.
   These are *the labels*: the belt trace is clean enough that its turns and
   stops are unambiguous (checked by eye on the first coached run), so no
   uncertain slice is masked around the boundaries;
4. the causal detector is scored against them -- what a live reading loses;
5. on coached runs the cues, aligned with :func:`align_cues`, are compared too.
   That is a *compliance* check -- did the subject do what was asked, and how
   late -- not an accuracy score: the cue labels are the less exact of the two.

Writes under ``data/processed/strain_belt/``: ``labels/<run>.csv`` (time,
chest, ``label`` and the live / cue labels, in the Szymański export codes),
``plots/<run>.png`` and ``summary.csv``.

Run from the repository root::

    uv run python tools/label_strain_belt_runs.py
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from respi_net.breath_phases import CLASS_COLOURS, CLASS_NAMES_PL, IGNORE, export_codes, label_runs
from respi_net.chest_signal import strain_belt_chest_signal
from respi_net.label_alignment import align_cues
from respi_net.paths import DATA_DIR
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import boundary_errors, score_phases
from respi_net.strain_belt import RAW_STRAIN_BELT_DIR, cue_origin_ms, cue_sidecar_path, load_coach_cues

# Guided radar sessions (tools/a121_iphone_guided_recording.py --reference belt|both).
SESSIONS_DIR = DATA_DIR / "raw" / "nn"

OUTPUT_DIR = DATA_DIR / "processed" / "strain_belt"
# The belt shows a stop clearly, so a pause is a hold from 2 s on (the phone
# protocol needs 4 s because its trace is still sliding after a breath).
BELT_MIN_HOLD_S = 2.0
# Typical breath depth over sensor noise.  Worn belt: ~500; belt lying on the desk: ~13.
MIN_DEPTH_TO_NOISE = 50.0


def belt_recordings(paths: list[Path]) -> list[Path]:
    if paths:
        return paths
    coach = [p for p in RAW_STRAIN_BELT_DIR.glob("strain_belt_*.csv") if not p.stem.endswith("_cues")]
    return sorted(coach) + sorted(SESSIONS_DIR.glob("*/*/run_*_belt.csv"))


def run_name(recording: Path) -> str:
    """Unique output name: session files share run names, so they carry their session's."""
    return f"{recording.parent.name}__{recording.stem}" if recording.stem.startswith("run_") else recording.stem


def _has_cues(recording: Path) -> bool:
    return cue_sidecar_path(recording).exists() or recording.stem.startswith("strain_belt_coach_")


def _spans(ax: plt.Axes, time_s: np.ndarray, labels: np.ndarray, ymin: float, ymax: float) -> None:
    for start, stop, label in label_runs(labels):
        if label == IGNORE:
            continue
        end = time_s[stop] if stop < len(time_s) else time_s[-1]
        ax.axvspan(time_s[start], end, ymin=ymin, ymax=ymax, color=CLASS_COLOURS[label], alpha=0.35, lw=0)


def plot_run(path: Path, signal, with_drift: np.ndarray, labels: np.ndarray, cue_labels: np.ndarray | None, title: str) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, height_ratios=(1, 1.4))
    top.plot(signal.time_s, with_drift, color="#7f8c8d", lw=1.2, label="pas po czyszczeniu, z dryfem")
    top.plot(signal.time_s, with_drift - signal.chest, color="#c0392b", lw=1.0, ls="--", label="linia końca wydechu (dryf)")
    top.set_ylabel("HX711 [jedn.]")
    top.legend(loc="upper left", fontsize=8)
    if cue_labels is None:
        _spans(bottom, signal.time_s, labels, 0.0, 1.0)
    else:
        _spans(bottom, signal.time_s, labels, 0.5, 1.0)
        _spans(bottom, signal.time_s, cue_labels, 0.0, 0.5)
        bottom.text(0.002, 0.03, "podpowiedzi trenera (zgodność)", transform=bottom.transAxes, va="bottom", fontsize=8)
    bottom.text(0.002, 0.97, "etykiety (detektor na pasie)", transform=bottom.transAxes, va="top", fontsize=8)
    bottom.plot(signal.time_s, signal.chest, color="#1f4e79", lw=1.6)
    bottom.axhline(0, color="k", lw=0.5)
    bottom.set_ylabel("pas bez dryfu [jedn.]")
    bottom.set_xlabel("czas od startu [s]")
    bottom.legend(handles=[Patch(color=CLASS_COLOURS[i], alpha=0.5, label=CLASS_NAMES_PL[i]) for i in range(4)], loc="upper right", fontsize=8, ncol=4)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def label_run(recording: Path, out: Path, min_hold_s: float) -> dict[str, object]:
    cues = load_coach_cues(recording) if _has_cues(recording) else None
    origin_ms = cue_origin_ms(cues) if cues else None
    signal = strain_belt_chest_signal(recording, origin_ms=origin_ms)
    name = run_name(recording)
    row: dict[str, object] = {"run": name, **{k: v for k, v in signal.meta.items() if k != "low_pass_hz"}}
    if signal.meta["depth_to_noise"] < MIN_DEPTH_TO_NOISE:
        row["status"] = "skipped: no breathing above noise (belt not worn?)"
        return row
    row["status"] = "labelled"
    labels = detect_phases(signal.chest, signal.fs, causal=False, min_hold_s=min_hold_s)
    live = detect_phases(signal.chest, signal.fs, causal=True, min_hold_s=min_hold_s)
    phase = score_phases(labels, live)
    timing = boundary_errors(labels, live, signal.time_s)
    row.update(
        {
            "labelled_share": float(np.mean(labels != IGNORE)),
            "live_accuracy": phase.accuracy,
            "live_macro_f1": phase.macro_f1,
            "live_boundary_median_delay_s": timing.median_abs_error_s,
        }
    )
    columns = {
        "time_s": np.round(signal.time_s, 3),
        "chest": np.round(signal.chest, 1),
        "label": export_codes(labels),
        "label_live": export_codes(live),
    }
    cue_labels = None
    if cues:
        result = align_cues(signal.time_s, signal.chest, cues, fs=signal.fs)
        cue_labels = result.labels
        agreement = score_phases(cue_labels, labels)
        summary = result.summary()
        row.update(
            {
                "cue_correlation": summary["correlation"],
                "cue_lag_s": summary["lag_s"],
                "cue_agreement": agreement.accuracy,
            }
        )
        columns["label_cues"] = export_codes(cue_labels)
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(labels_dir / f"{name}.csv", index=False)
    title = f"{name}  ·  dryf {signal.meta['drift_counts']:+.0f} jedn.  ·  na żywo {row['live_accuracy']:.0%}"
    if cue_labels is not None:
        title += f"  ·  zgodność z trenerem {row['cue_agreement']:.0%}"
    with_drift = strain_belt_chest_signal(recording, drift="none", origin_ms=origin_ms).chest
    plot_run(out / "plots" / f"{name}.png", signal, with_drift, labels, cue_labels, title)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recordings", nargs="*", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-hold", type=float, default=BELT_MIN_HOLD_S, help="Shortest stillness labelled as a hold [s].")
    args = parser.parse_args()
    recordings = belt_recordings(args.recordings)
    if not recordings:
        print(f"No belt recordings in {RAW_STRAIN_BELT_DIR}.")
        return 1
    summary = pd.DataFrame([label_run(path, args.output_dir, args.min_hold) for path in recordings])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    shown = ["run", "status", "depth_to_noise", "drift_counts", "live_accuracy", "live_boundary_median_delay_s", "cue_agreement", "cue_lag_s"]
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.2f}".format):
        print(summary[[c for c in shown if c in summary]].to_string(index=False))
    print(f"Labels, plots and summary.csv in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
