#!/usr/bin/env python3
"""Corrected breathing-phase labels, charts, baseline scores and a dataset from coach runs.

Reads every A121 recording made with the breathing coach:

* the 2 m foil/angle retest (``data/raw/a121/guided/guided_foil2m_*``),
* the A121 files of the HB100 comparison session
  (``data/raw/hb100_a121/guided/hb100_a121_range_*``, A121 only),

turns the cues into per-breath labels with :func:`respi_net.label_alignment.align_cues`
and writes, under ``data/processed/breath_phases/``:

* ``runs/<run>.npz`` -- signals and labels ready for :mod:`respi_net.nn_dataset`,
* ``labels/<run>_labels.csv`` and ``<run>_boundaries.csv`` -- for inspection,
* ``plots/<run>.png`` and ``plots/coach_labels.pdf`` -- every run, labels over the signal,
* ``alignment.csv``, ``baseline.csv``, ``summary.json``,
* ``dataset_coach_v1.npz`` -- 60 s windows (first 20 s context only) split by run.

``--thesis-figures`` also redraws the two figures used in the thesis.

The labels here are snapped to the radar's own turning points, which is fine
for these clean, deep, coached breaths and circular for anything else: the
baseline scores against them are a sanity check, not the benchmark.

Run from the repository root::

    uv run python tools/build_breath_phase_labels.py --thesis-figures
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from respi_net.breath_phases import (
    CLASS_COLOURS,
    CLASS_NAMES,
    CLASS_NAMES_PL,
    EXHALE,
    HOLD_AFTER_EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    export_codes,
    label_runs,
)
from respi_net.chest_signal import a121_chest_signal, light_filter, rate_band_filter
from respi_net.label_alignment import AlignmentResult, align_cues, cue_phases
from respi_net.nn_dataset import export_dataset, labelled_run, save_run
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import boundary_errors, score_phases

FOIL_DIR = ROOT / "data" / "raw" / "a121" / "guided"
RANGE_DIR = ROOT / "data" / "raw" / "hb100_a121" / "guided"
OUT_DIR = ROOT / "data" / "processed" / "breath_phases"
FIGURE_DIR = ROOT / "docs" / "thesis" / "figures"
PHASE_LABELS = (INHALE, HOLD_AFTER_EXHALE, EXHALE, HOLD_AFTER_INHALE)


@dataclass(frozen=True)
class CoachRun:
    run_id: str
    a121_path: Path
    origin_ms: float | None
    cues: list[dict[str, Any]]
    session: str
    distance_cm: float | None
    condition: str


def coach_runs() -> list[CoachRun]:
    runs: list[CoachRun] = []
    for manifest_path in sorted(FOIL_DIR.glob("guided_foil2m_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("measurements", []):
            if item.get("status") != "accepted":
                continue
            path = manifest_path.parent / Path(item["csv_path"]).name
            runs.append(
                CoachRun(
                    run_id=f"a121_2m_{int(item['step']):02d}_{str(item.get('condition_id', '')).lower()}",
                    a121_path=path,
                    origin_ms=None,
                    cues=list(manifest["breathing_cues"]),
                    session=manifest_path.parent.name,
                    distance_cm=float(item.get("distance_cm") or 200.0),
                    condition=str(item.get("condition_id", "")),
                )
            )
    for session in sorted(RANGE_DIR.glob("hb100_a121_range_*")):
        for cue_path in sorted(session.glob("run_*_range-*_cues.csv")):
            a121_path = cue_path.with_name(cue_path.name.replace("_cues.csv", "_a121.csv"))
            if not a121_path.exists():
                continue
            cues = pd.read_csv(cue_path)
            stem = cue_path.name.replace("_cues.csv", "")
            distance = next((part for part in stem.split("_") if part.endswith("cm")), "")
            runs.append(
                CoachRun(
                    run_id=f"a121_{stem.split('_')[0]}_{stem.split('_')[1]}_{distance}",
                    a121_path=a121_path,
                    origin_ms=float(cues["start_wall_ms"].iloc[0]),
                    cues=cues[["start_s", "end_s", "kind"]].to_dict("records"),
                    session=session.name,
                    distance_cm=float(distance[:-2]) if distance else None,
                    condition=stem.split("_")[2] if len(stem.split("_")) > 2 else "",
                )
            )
    return runs


def _spans(ax: plt.Axes, time_s: np.ndarray, labels: np.ndarray, *, alpha: float = 0.28, ymin: float = 0.0, ymax: float = 1.0) -> None:
    for start, stop, label in label_runs(labels):
        if label == IGNORE:
            continue
        end = time_s[stop] if stop < len(time_s) else time_s[-1]
        ax.axvspan(time_s[start], end, ymin=ymin, ymax=ymax, color=CLASS_COLOURS[label], alpha=alpha, lw=0)


def _cue_labels(time_s: np.ndarray, cues: list[dict[str, Any]], lag_s: float) -> np.ndarray:
    labels = np.full(len(time_s), IGNORE, dtype=np.int8)
    for phase in cue_phases(cues):
        if phase.label != IGNORE:
            labels[(time_s >= phase.start_s + lag_s) & (time_s < phase.end_s + lag_s)] = phase.label
    return labels


def plot_run(run: CoachRun, time_s: np.ndarray, chest: np.ndarray, result: AlignmentResult, path: Path | None = None, *, polish: bool = False) -> plt.Figure:
    names = CLASS_NAMES_PL if polish else CLASS_NAMES
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(14, 5.2), sharex=True, gridspec_kw={"height_ratios": [5, 1.1], "hspace": 0.06})
    _spans(ax, time_s, result.labels)
    ax.plot(time_s, chest, color="#111827", lw=1.1)
    top = np.nanmax(chest)
    for boundary in result.boundaries:
        ax.axvline(boundary.expected_s, color="#6b7280", lw=0.7, ls=(0, (3, 3)))
        if boundary.method == "fallback":
            ax.plot(boundary.time_s, top, marker="x", color="#b91c1c", ms=7, mew=2)
        elif boundary.method != "end":
            ax.plot([boundary.expected_s, boundary.time_s], [top, top], color="#111827", lw=1.2)
            ax.plot(boundary.time_s, top, marker="v", color="#111827", ms=4)
    summary = result.summary()
    distance = f"{run.distance_cm:g} cm" if run.distance_cm else ""
    if polish:
        title = (
            f"{run.run_id}  ({distance})   opóźnienie {summary['lag_s']:.2f} s,   przesunięcie granic: mediana "
            f"{summary['shift_median_s']:+.2f} s, IQR {summary['shift_iqr_s']:.2f} s,   bez wyraźnego zwrotu: {summary['fallback_share'] * 100:.0f}%"
        )
        ax.set_ylabel("klatka [mm]")
        bars.set_yticks([0.25, 0.75], ["skorygowane", "komendy + opóźn."])
        bars.set_xlabel("czas od startu komend [s]")
    else:
        title = (
            f"{run.run_id}  ({distance})   lag {summary['lag_s']:.2f} s,   boundary shift median {summary['shift_median_s']:+.2f} s, "
            f"IQR {summary['shift_iqr_s']:.2f} s,   no clear turn: {summary['fallback_share'] * 100:.0f}%"
        )
        ax.set_ylabel("chest [mm]")
        bars.set_yticks([0.25, 0.75], ["corrected", "cue + lag"])
        bars.set_xlabel("time from first cue [s]")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(alpha=0.25)
    _spans(bars, time_s, _cue_labels(time_s, run.cues, result.lag_s), alpha=0.9, ymin=0.52, ymax=0.98)
    _spans(bars, time_s, result.labels, alpha=0.9, ymin=0.02, ymax=0.48)
    bars.set_ylim(0, 1)
    bars.set_xlim(time_s[0], time_s[-1])
    handles = [Patch(color=CLASS_COLOURS[label], alpha=0.6, label=names[label]) for label in PHASE_LABELS]
    marker_labels = (
        ("przesunięta granica", "zwrot nieznaleziony (granica z komendy)", "komenda + opóźnienie")
        if polish
        else ("snapped boundary", "no clear turn (cue fallback)", "cue + lag")
    )
    handles += [
        plt.Line2D([], [], color="#111827", marker="v", ls="-", ms=4, label=marker_labels[0]),
        plt.Line2D([], [], color="#b91c1c", marker="x", ls="", ms=7, mew=2, label=marker_labels[1]),
        plt.Line2D([], [], color="#6b7280", ls=(0, (3, 3)), label=marker_labels[2]),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=8, ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.07))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def evaluate_baseline(time_s: np.ndarray, chest: np.ndarray, fs: float, reference: np.ndarray) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for mode, causal in (("offline", False), ("realtime", True)):
        predicted = detect_phases(chest, fs, causal=causal)
        phase = score_phases(reference, predicted)
        timing = boundary_errors(reference, predicted, time_s)
        scores[mode] = {
            "accuracy": phase.accuracy,
            "macro_f1": phase.macro_f1,
            **{f"f1_{name}": values["f1"] for name, values in phase.per_class.items() if values["support"]},
            "boundary_detection": timing.detection_rate,
            "boundary_median_error_s": timing.median_abs_error_s,
            "false_boundaries_per_min": timing.false_per_minute,
            "samples": phase.samples,
        }
    return scores


def figure_filtering(runs: list[CoachRun], path: Path) -> dict[str, float]:
    """Synthetic and real breath hold through the rate band-pass and the light filter."""

    fs = 20.0
    t = np.arange(0.0, 70.0, 1.0 / fs)
    true = np.zeros_like(t)
    for start in np.arange(0.0, 30.0, 5.0):
        inhale = (t >= start) & (t < start + 2.0)
        exhale = (t >= start + 2.0) & (t < start + 5.0)
        true[inhale] = 0.5 - 0.5 * np.cos(np.pi * (t[inhale] - start) / 2.0)
        true[exhale] = 0.5 + 0.5 * np.cos(np.pi * (t[exhale] - start - 2.0) / 3.0)
    hold = (t >= 30.0) & (t < 45.0)
    true[hold] = 0.0
    for start in np.arange(45.0, 70.0, 5.0):
        inhale = (t >= start) & (t < start + 2.0)
        exhale = (t >= start + 2.0) & (t < start + 5.0)
        true[inhale] = 0.5 - 0.5 * np.cos(np.pi * (t[inhale] - start) / 2.0)
        true[exhale] = 0.5 + 0.5 * np.cos(np.pi * (t[exhale] - start - 2.0) / 3.0)
    true *= 10.0
    band = rate_band_filter(true, fs)
    light = light_filter(true, fs)

    chosen = next((run for run in runs if run.run_id.startswith("a121_2m_12")), runs[0])
    signal = a121_chest_signal(chosen.a121_path, origin_ms=chosen.origin_ms)
    result = align_cues(signal.time_s, signal.chest, chosen.cues, fs=signal.fs)
    real_band = rate_band_filter(signal.raw, signal.fs)
    hold_phase = next(phase for phase in cue_phases(chosen.cues) if phase.label == HOLD_AFTER_EXHALE)
    hold_start, hold_end = hold_phase.start_s + result.lag_s, hold_phase.end_s + result.lag_s

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=False)
    ax = axes[0]
    ax.axvspan(30.0, 45.0, color=CLASS_COLOURS[HOLD_AFTER_EXHALE], alpha=0.18, lw=0)
    ax.plot(t, true, color="#111827", lw=2.2, label="ruch klatki (wzorzec)")
    ax.plot(t, light, color="#2563eb", lw=1.2, ls="--", label="tylko dolnoprzepustowy 2 Hz")
    ax.plot(t, band - np.mean(band) + np.mean(true), color="#dc2626", lw=1.6, label="pasmowy 0,1–0,5 Hz (do częstości)")
    ax.set_title("a) wzorzec: 15 s pauzy po wydechu", loc="left", fontsize=10)
    ax.set_ylabel("przemieszczenie [mm]")
    ax.set_xlim(15, 60)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(alpha=0.25)

    ax = axes[1]
    window = (signal.time_s >= hold_start - 15.0) & (signal.time_s <= hold_end + 15.0)
    ax.axvspan(hold_start, hold_end, color=CLASS_COLOURS[HOLD_AFTER_EXHALE], alpha=0.18, lw=0)
    ax.plot(signal.time_s[window], signal.chest[window], color="#2563eb", lw=1.4, label="A121, tylko dolnoprzepustowy 2 Hz")
    shifted = real_band + np.mean(signal.chest[window]) - np.mean(real_band[window])
    ax.plot(signal.time_s[window], shifted[window], color="#dc2626", lw=1.4, label="A121, pasmowy 0,1–0,5 Hz")
    ax.set_title(f"b) nagranie {chosen.run_id}: pauza po wydechu", loc="left", fontsize=10)
    ax.set_xlabel("czas [s]")
    ax.set_ylabel("przemieszczenie [mm]")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)

    inside = (t >= 32.0) & (t < 44.0)
    real_inside = (signal.time_s >= hold_start + 2.0) & (signal.time_s < hold_end - 1.0)
    breath_depth = float(np.ptp(true[(t >= 5.0) & (t < 25.0)]))
    return {
        "synthetic_hold_level_error_band_mm": float(abs(np.mean(band[inside]) - np.mean(band) - (np.mean(true[inside]) - np.mean(true)))),
        "synthetic_breath_depth_mm": breath_depth,
        "synthetic_hold_motion_band_mm": float(np.ptp(band[inside])),
        "synthetic_hold_motion_light_mm": float(np.ptp(light[inside])),
        "real_hold_motion_band_mm": float(np.ptp(real_band[real_inside])),
        "real_hold_motion_light_mm": float(np.ptp(signal.chest[real_inside])),
        "real_run": chosen.run_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--window-s", type=float, default=60.0)
    parser.add_argument("--stride-s", type=float, default=5.0)
    parser.add_argument("--warmup-s", type=float, default=20.0, help="Context at the start of each window that is not a training target.")
    parser.add_argument("--thesis-figures", action="store_true", help="Also redraw the thesis figures in docs/thesis/figures.")
    args = parser.parse_args()

    runs = coach_runs()
    if not runs:
        raise SystemExit("No coach recordings found.")
    out = args.out_dir
    alignment_rows, baseline_rows, labelled = [], [], []
    shifts: list[float] = []
    pdf_path = out / "plots" / "coach_labels.pdf"
    for folder in ("plots", "labels", "runs"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for run in runs:
            signal = a121_chest_signal(run.a121_path, origin_ms=run.origin_ms)
            result = align_cues(signal.time_s, signal.chest, run.cues, fs=signal.fs)
            summary = result.summary()
            shifts += [b.shift_s for b in result.boundaries if b.method in ("turn", "stop", "start")]
            meta = {"session": run.session, "distance_cm": run.distance_cm, "condition": run.condition, "lag_s": result.lag_s}
            data = labelled_run(run.run_id, signal, result.labels, meta=meta)
            labelled.append(data)
            save_run(data, out / "runs" / f"{run.run_id}.npz")
            pd.DataFrame({"time_s": signal.time_s, "label": export_codes(result.labels)}).to_csv(
                out / "labels" / f"{run.run_id}_labels.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "cue_s": b.cue_s,
                        "expected_s": b.expected_s,
                        "time_s": b.time_s,
                        "shift_s": b.shift_s,
                        "before": CLASS_NAMES[b.before] if b.before >= 0 else "none",
                        "after": CLASS_NAMES[b.after] if b.after >= 0 else "none",
                        "method": b.method,
                    }
                    for b in result.boundaries
                ]
            ).to_csv(out / "labels" / f"{run.run_id}_boundaries.csv", index=False)
            fig = plot_run(run, signal.time_s, signal.chest, result, out / "plots" / f"{run.run_id}.png")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            alignment_rows.append({"run": run.run_id, "session": run.session, "distance_cm": run.distance_cm, **summary})
            for mode, scores in evaluate_baseline(signal.time_s, signal.chest, signal.fs, result.labels).items():
                baseline_rows.append({"run": run.run_id, "mode": mode, **scores})

    alignment = pd.DataFrame(alignment_rows)
    baseline = pd.DataFrame(baseline_rows)
    alignment.to_csv(out / "alignment.csv", index=False)
    baseline.to_csv(out / "baseline.csv", index=False)
    dataset_meta = export_dataset(
        labelled,
        out / "dataset_coach_v1.npz",
        window_s=args.window_s,
        stride_s=args.stride_s,
        warmup_s=args.warmup_s,
        description="Coach recordings (A121, one subject), labels snapped to the radar's own turning points.",
    )
    shifts_arr = np.asarray(shifts)
    summary = {
        "runs": len(runs),
        "labelled_minutes": float(sum(run.labelled_s for run in labelled) / 60.0),
        "lag_s": {"min": float(alignment.lag_s.min()), "median": float(alignment.lag_s.median()), "max": float(alignment.lag_s.max())},
        "shift_s": {
            "median": float(np.median(shifts_arr)),
            "iqr": float(np.subtract(*np.percentile(shifts_arr, [75, 25]))),
            "abs_p90": float(np.percentile(np.abs(shifts_arr), 90)),
            "abs_mean": float(np.mean(np.abs(shifts_arr))),
        },
        "fallback_share": float(alignment.fallback_share.mean()),
        "baseline": {
            mode: {column: float(group[column].mean()) for column in group.columns if column not in ("run", "mode")}
            for mode, group in baseline.groupby("mode")
        },
        "dataset": {name: {k: v for k, v in counts.items() if k != "runs"} | {"runs": len(counts["runs"])} for name, counts in dataset_meta["splits"].items()},
    }
    if args.thesis_figures:
        best = next((run for run in runs if run.run_id.startswith("a121_2m_12")), runs[0])
        signal = a121_chest_signal(best.a121_path, origin_ms=best.origin_ms)
        result = align_cues(signal.time_s, signal.chest, best.cues, fs=signal.fs)
        fig = plot_run(best, signal.time_s, signal.chest, result, FIGURE_DIR / "fazy_etykiety_skorygowane.png", polish=True)
        plt.close(fig)
        summary["filtering"] = figure_filtering(runs, FIGURE_DIR / "fazy_filtracja_pauza.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(alignment.round(3).to_string(index=False))
    print()
    print(baseline.groupby("mode").mean(numeric_only=True).round(3).to_string())
    print()
    print(json.dumps({k: v for k, v in summary.items() if k != "baseline"}, indent=2, ensure_ascii=False))
    print(f"\nCharts: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
