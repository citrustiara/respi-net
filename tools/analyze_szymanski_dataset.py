#!/usr/bin/env python3
"""What the supervisor's respiratory-phase dataset holds, and how our detector does on it.

Reads the unzipped dataset of Szymański et al. (Sci Data 2025, doi
10.34808/ray2-4g53) from ``data/external/szymanski2025/unpacked`` through
:mod:`respi_net.szymanski_dataset` and reports:

* recordings, minutes, sample rates and subjects per sensor,
* class balance, phase changes and hold durations of the labels as published,
* where the belt labels put the end of each breath relative to the belt's own
  turn, as published and after :func:`respi_net.szymanski_dataset.corrected_labels` (the published ones
  are 0.2 s early),
* the deterministic phase detector scored against the corrected hand labels
  of every recording (and the published belt labels) on this project's 20 Hz
  grid: offline, offline with holds from 1 s instead of 2 s (their holds are
  short), and real time -- with how late the real-time decisions come and
  how much accuracy is left once that delay is taken out,
* the same detector settings on the corrected coach labels, when
  ``tools/build_breath_phase_labels.py`` has been run, to show what each
  hold definition costs on our own data.

Writes ``analysis.json`` next to the data, with ``--thesis-figure``
``docs/thesis/figures/fazy_zbior_szymanski.png`` and ``fazy_szymanski_przesuniecie.png``
(why the published labels are early), with ``--notes-figure PNG`` the latter in
English, and with ``--export`` the belt
recordings as an nn_dataset ``.npz`` for pre-training (corrected hand labels;
the echo channel is zero, so train on ``--inputs chest,velocity`` or let the
model learn to do without it).

    uv run python tools/analyze_szymanski_dataset.py --thesis-figure --export
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from respi_net.breath_phases import (
    fold_short_holds,
    CLASS_COLOURS,
    CLASS_NAMES,
    CLASS_NAMES_PL,
    EXHALE,
    HOLD_AFTER_EXHALE,
    HOLD_AFTER_INHALE,
    IGNORE,
    INHALE,
    NOISE,
    label_runs,
)
from respi_net.chest_signal import light_filter
from respi_net.nn_dataset import LabelledRun, export_dataset, runs_in
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import boundary_errors, score_phases
from respi_net.szymanski_dataset import DEFAULT_ROOT, SzymanskiRecording, orientation, recordings, to_labelled_run

FIGURE = ROOT / "docs" / "thesis" / "figures" / "fazy_zbior_szymanski.png"
TIMING_FIGURE = ROOT / "docs" / "thesis" / "figures" / "fazy_szymanski_przesuniecie.png"
TIMING_TEXT = {
    "pl": {
        "raw": "surowy sygnał pasa",
        "copy": "wygładzona kopia widziana przy etykietowaniu",
        "published": "opublikowane",
        "corrected": "po korekcie",
        "a": "a) zwykły oddech: kopia wyprzedza sygnał o 2 próbki (0,2 s), a z nią etykiety",
        "b": "b) głęboki wdech i pauza: kopia ma wartość +1, zanim klatka się zatrzyma",
        "time": "czas [s]",
        "classes": {INHALE: "wdech", EXHALE: "wydech", HOLD_AFTER_INHALE: "pauza po wdechu", HOLD_AFTER_EXHALE: "pauza po wydechu"},
    },
    "en": {
        "raw": "raw belt signal",
        "copy": "smoothed copy the labellers saw",
        "published": "published",
        "corrected": "corrected",
        "a": "a) normal breath: the copy runs 2 samples (0.2 s) ahead of the signal, and so do the labels",
        "b": "b) deep inhale into a hold: the copy sits at +1 before the chest stops",
        "time": "time [s]",
        "classes": {INHALE: "inhale", EXHALE: "exhale", HOLD_AFTER_INHALE: "hold after inhale", HOLD_AFTER_EXHALE: "hold after exhale"},
    },
}
COACH_RUNS = ROOT / "data" / "processed" / "breath_phases" / "runs"
EXPORT = ROOT / "data" / "processed" / "breath_phases" / "dataset_szymanski_belt_v1.npz"
# "offline" is the detector as it is now (hold from 4 s, slow-motion check) against the labels brought to the same rule
# (holds shorter than 4 s folded into the breaths around them); the rows below show it against the labels as published and the
# older settings against the labels as published.
MODES = {
    "offline": {"causal": False, "fold_s": 4.0},
    "offline_published_holds": {"causal": False},
    "offline_2s": {"causal": False, "min_hold_s": 2.0, "slow_motion_fraction": None},
    "offline_short_holds": {"causal": False, "min_hold_s": 1.0, "slow_motion_fraction": None},
    "realtime": {"causal": True},
}


def score_run(run: LabelledRun, *, causal: bool = False, fold_s: float | None = None, **settings: float | None) -> dict[str, float]:
    predicted = detect_phases(run.features[0], run.fs, causal=causal, **settings)
    reference = np.where(run.labels == NOISE, IGNORE, run.labels)
    if fold_s is not None:
        reference = fold_short_holds(reference, run.fs, fold_s)
    phase = score_phases(reference, predicted)
    timing = boundary_errors(reference, predicted, run.time_s)
    entry = {
        "accuracy": phase.accuracy,
        "macro_f1": phase.macro_f1,
        **{f"f1_{name}": values["f1"] for name, values in phase.per_class.items() if values["support"]},
        "boundary_detection": timing.detection_rate,
        "boundary_median_error_s": timing.median_abs_error_s,
        "boundary_delay_s": timing.median_delay_s,
    }
    if causal:
        # How much of the gap to offline is plain delay: the accuracy once the
        # output is moved back by the shift (up to 2 s) that suits it best.
        entry["accuracy_delay_removed"], entry["best_shift_s"] = max(
            (score_phases(reference[: len(reference) - k], predicted[k:]).accuracy, k / run.fs)
            for k in range(0, int(2 * run.fs) + 1, 2)
        )
    return entry


def mean_scores(entries: list[dict[str, float]]) -> dict[str, float]:
    return pd.DataFrame(entries).mean(numeric_only=True).to_dict()


def turn_offsets(recording: SzymanskiRecording, window_s: float = 1.0) -> dict[str, list[float]]:
    """Labelled breath ends minus the nearest turn of the signal itself.

    A breath ends where the chest turns: an inhale at the highest point within
    ``window_s``, an exhale at the lowest.  Negative means the label is early.
    """

    chest = orientation(recording) * recording.raw
    time_s = recording.time_s
    offsets: dict[str, list[float]] = {}
    runs = label_runs(recording.labels)
    for (_, _, before), (start, _, after) in zip(runs[:-1], runs[1:]):
        if before not in (INHALE, EXHALE) or after in (before, IGNORE, NOISE):
            continue
        near = np.abs(time_s - time_s[start]) <= window_s
        turn = np.argmax(chest[near]) if before == INHALE else np.argmin(chest[near])
        name = f"{CLASS_NAMES[before]} -> {'hold' if after in (HOLD_AFTER_INHALE, HOLD_AFTER_EXHALE) else CLASS_NAMES[after]}"
        offsets.setdefault(name, []).append(float(time_s[start] - time_s[near][turn]))
    return offsets


def settles(recording: SzymanskiRecording, *, motion_fraction: float = 0.2, gap_s: float = 0.3) -> dict[str, list[float]]:
    """How long the chest keeps moving back at the start of each hold that follows its breath.

    A settle is motion against the breath just ended, faster than
    ``motion_fraction`` of the typical breathing speed, starting within
    ``gap_s`` of the hold's first sample.  Holds without one count as 0 s.
    """

    chest = orientation(recording) * recording.raw
    velocity = np.gradient(light_filter(chest - np.median(chest), recording.fs, 2.0)) * recording.fs
    threshold = motion_fraction * float(np.percentile(np.abs(velocity), 80))
    found: dict[str, list[float]] = {}
    runs = label_runs(recording.labels)
    for (_, _, before), (start, stop, after) in zip(runs[:-1], runs[1:]):
        if (before, after) not in ((INHALE, HOLD_AFTER_INHALE), (EXHALE, HOLD_AFTER_EXHALE)):
            continue
        back = -velocity[start:stop] if before == INHALE else velocity[start:stop]
        first = next((index for index in range(min(len(back), int(round(gap_s * recording.fs)))) if back[index] > threshold), None)
        length = 0
        if first is not None:
            length = first
            while length < len(back) and back[length] > threshold:
                length += 1
            length -= first
        found.setdefault(CLASS_NAMES[after], []).append(length / recording.fs)
    return found


def plot_label_timing(published: SzymanskiRecording, corrected: SzymanskiRecording, root: Path, path: Path, language: str = "pl") -> None:
    """The raw belt against the smoothed copy the labels were drawn on, with both label versions.

    No text sits inside the plots; what they show is said in the titles and legend.
    """

    text = TIMING_TEXT[language]
    stored = pd.read_csv(root / "data" / published.group / "labelled" / f"tens_{published.name}.txt", header=None, names=["norm", "code", "sec"])
    time_s = published.time_s
    raw = orientation(published) * published.raw
    copy = stored["norm"].to_numpy()[: len(time_s)]
    fig = plt.figure(figsize=(11, 8.2))
    outer = fig.add_gridspec(2, 1, hspace=0.3)
    pairs = []
    for cell in outer:
        inner = cell.subgridspec(2, 1, height_ratios=[3, 0.9], hspace=0.08)
        pairs.append((fig.add_subplot(inner[0]), fig.add_subplot(inner[1])))
    for index, ((ax, bars), (lo, hi), title) in enumerate(zip(pairs, ((29.6, 32.4), (90.0, 93.6)), (text["a"], text["b"]))):
        sel = (time_s >= lo) & (time_s <= hi)
        fit = sel & (np.abs(copy) < 0.999)
        slope, offset = np.polyfit(raw[fit], copy[fit], 1)  # the raw belt on the copy's scale, to overlay them
        ax.plot(time_s[sel], slope * raw[sel] + offset, "o-", color="#111827", ms=3.5, lw=1.4)
        ax.plot(time_s[sel], copy[sel], "o-", color="#f97316", ms=3.5, lw=1.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(alpha=0.25)
        ax.tick_params(labelbottom=False)
        ax.set_title(title, loc="left", fontsize=10)
        for row, labels in enumerate((published.labels, corrected.labels)):
            top = 0.96 - row * 0.5
            for start, stop, label in label_runs(labels):
                begin, end = time_s[start], time_s[min(stop, len(time_s) - 1)]
                if label != IGNORE and end > lo and begin < hi:
                    bars.axvspan(max(begin, lo), min(end, hi), ymin=top - 0.42, ymax=top, color=CLASS_COLOURS[label], lw=0)
        bars.set_xlim(lo, hi)
        bars.set_ylim(0, 1)
        bars.set_yticks([0.75, 0.25], [text["published"], text["corrected"]])
        if index == len(pairs) - 1:
            bars.set_xlabel(text["time"])
    lines = [plt.Line2D([], [], color="#111827", marker="o", ms=3.5), plt.Line2D([], [], color="#f97316", marker="o", ms=3.5)]
    patches = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[label]) for label in text["classes"]]
    fig.legend(lines + patches, [text["raw"], text["copy"], *text["classes"].values()], loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_example(run: LabelledRun, published: LabelledRun, name: str, path: Path) -> None:
    predicted = detect_phases(run.features[0], run.fs)
    window = (run.time_s >= 20.0) & (run.time_s < 110.0)
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True, gridspec_kw={"height_ratios": [4, 1.6], "hspace": 0.06})
    t = run.time_s[window]
    for start, stop, label in label_runs(run.labels[window]):
        if label != IGNORE:
            ax.axvspan(t[start], t[min(stop, len(t) - 1)], color=CLASS_COLOURS[label], alpha=0.25, lw=0)
    ax.plot(t, run.features[0][window], color="#111827", lw=1.1)
    ax.set_ylabel("pas (znorm.)")
    ax.grid(alpha=0.25)
    ax.set_title(f"Zbiór Szymańskiego i in., pas tensometryczny, nagranie „{name}”: etykiety ręczne przed i po korekcie czasu oraz detektor", loc="left", fontsize=10)
    rows = ((published.labels[window], "ręczne (opublikowane)"), (run.labels[window], "ręczne (po korekcie)"), (predicted[window], "detektor"))
    for index, (labels, _) in enumerate(rows):
        hi = 1.0 - index / len(rows) - 0.02
        lo = hi - 1.0 / len(rows) + 0.04
        for start, stop, label in label_runs(labels):
            if label != IGNORE:
                bars.axvspan(t[start], t[min(stop, len(t) - 1)], ymin=lo, ymax=hi, color=CLASS_COLOURS[label], alpha=0.9, lw=0)
    bars.set_yticks([1.0 - (index + 0.5) / len(rows) for index in range(len(rows))], [title for _, title in rows])
    bars.set_ylim(0, 1)
    bars.set_xlim(t[0], t[-1])
    bars.set_xlabel("czas [s]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS[label]) for label in (INHALE, EXHALE, HOLD_AFTER_INHALE, HOLD_AFTER_EXHALE)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#9ca3af"))
    names = [CLASS_NAMES_PL[label] for label in (INHALE, EXHALE, HOLD_AFTER_INHALE, HOLD_AFTER_EXHALE)] + ["bez etykiety"]
    fig.legend(handles, names, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--thesis-figure", action="store_true")
    parser.add_argument("--notes-figure", type=Path, default=None, metavar="PNG", help="the label timing picture in English, for local notes")
    parser.add_argument("--export", type=Path, nargs="?", const=EXPORT, default=None, metavar="NPZ",
                        help=f"write the belt recordings as a dataset (default {EXPORT.relative_to(ROOT)})")
    args = parser.parse_args()

    published = recordings(args.root, corrected=False)
    found = recordings(args.root)
    rows, changes, holds = [], Counter(), {HOLD_AFTER_EXHALE: [], HOLD_AFTER_INHALE: []}
    for recording in published:
        counts = np.bincount(recording.labels, minlength=5)
        runs = label_runs(recording.labels)
        for (_, _, before), (_, _, after) in zip(runs[:-1], runs[1:]):
            changes[f"{CLASS_NAMES[before]} -> {CLASS_NAMES[after]}"] += 1
        for start, stop, label in runs:
            if label in holds:
                holds[label].append((stop - start) / recording.fs)
        rows.append(
            {
                "sensor": recording.sensor,
                "group": recording.group,
                "recording": recording.name,
                "subject": recording.subject,
                "fs_hz": round(recording.fs, 1),
                "minutes": round(float(recording.time_s[-1] - recording.time_s[0]) / 60.0, 2),
                **{CLASS_NAMES[k]: int(counts[k]) for k in range(5)},
            }
        )

    # The detector against the corrected labels of every sensor, and against
    # the belt labels as published, to show what the correction changes.
    scores: dict[str, list[dict[str, float]]] = {}
    timing: dict[str, dict[str, list[float]]] = {}
    for version, recording in [("corrected", r) for r in found] + [("published", r) for r in published if r.sensor == "tensometer"]:
        key = recording.sensor if version == "corrected" else f"{recording.sensor}_published"
        run = to_labelled_run(recording)
        for mode, settings in MODES.items():
            scores.setdefault(f"{key}/{mode}", []).append(score_run(run, **settings))
        if recording.sensor == "tensometer":
            for name, values in turn_offsets(recording).items():
                timing.setdefault(version, {}).setdefault(name, []).extend(values)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))
    per_sensor = table.groupby("sensor").agg(recordings=("recording", "count"), minutes=("minutes", "sum"))
    print("\n" + per_sensor.round(1).to_string())
    print("\nBelt minutes per subject:", table[table.sensor == "tensometer"].groupby("subject")["minutes"].sum().round(1).to_dict())
    print("\nMost common phase changes:", changes.most_common(10))
    hold_summary = {
        CLASS_NAMES[label]: {"count": len(values), "p10_s": float(np.percentile(values, 10)), "median_s": float(np.median(values)), "p90_s": float(np.percentile(values, 90))}
        for label, values in holds.items()
        if values
    }
    print("Hold durations:", json.dumps(hold_summary, indent=None))
    summary = {"recordings": len(found), "per_sensor": per_sensor.reset_index().to_dict("records"), "holds": hold_summary, "changes": dict(changes)}
    summary["label_timing_belt"] = {
        version: {
            name: {"median_s": float(np.median(values)), "within_0_1_s": float(np.mean(np.abs(values) <= 0.1001)), "count": len(values)}
            for name, values in sorted(entries.items())
        }
        for version, entries in timing.items()
    }
    print("\nBelt label boundaries minus the belt's own turn (published vs corrected):")
    for version, entries in summary["label_timing_belt"].items():
        print(f"  {version:10}", {name: (round(v["median_s"], 2), f"{v['within_0_1_s']:.0%}") for name, v in entries.items()})
    corrected_belt = [r for r in found if r.sensor == "tensometer"]
    settle_times: dict[str, list[float]] = {}
    for recording in corrected_belt:
        for name, values in settles(recording).items():
            settle_times.setdefault(name, []).extend(values)
    summary["settles_belt"] = {
        name: {"holds": len(values), "with_settle": float(np.mean(np.array(values) > 0)), "median_s": float(np.median([v for v in values if v > 0]))}
        for name, values in settle_times.items()
    }
    print("\nHolds that start with a settle back (corrected belt labels):", json.dumps(summary["settles_belt"]))
    published_belt = {(r.group, r.name): r.labels for r in published if r.sensor == "tensometer"}
    summary["correction_belt"] = {
        "samples_changed": float(np.mean([np.mean(r.labels != published_belt[(r.group, r.name)]) for r in corrected_belt])),
        "samples_unlabelled": float(np.mean([np.mean((r.labels == IGNORE) & (published_belt[(r.group, r.name)] != IGNORE)) for r in corrected_belt])),
    }
    summary["detector"] = {key: mean_scores(entries) for key, entries in scores.items()}
    if COACH_RUNS.is_dir():
        coach = list(runs_in(COACH_RUNS))
        for mode, settings in MODES.items():
            summary["detector"][f"coach_a121/{mode}"] = mean_scores([score_run(run, **settings) for run in coach])
    columns = ["accuracy", "macro_f1", "f1_hold_after_inhale", "f1_hold_after_exhale", "boundary_detection", "boundary_delay_s", "accuracy_delay_removed", "best_shift_s"]
    print("\nDeterministic detector vs hand labels (means over recordings):")
    print(pd.DataFrame(summary["detector"]).T.reindex(columns=columns).round(3).to_string())
    (args.root.parent / "analysis.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    if args.export:
        belt = [to_labelled_run(recording) for recording in corrected_belt]
        meta = export_dataset(
            belt,
            args.export,
            window_s=60.0,
            stride_s=5.0,
            warmup_s=20.0,
            description="Szymanski et al. 2025 belt recordings, hand labels with the timing corrected (CC BY-NC-ND: local use only)",
        )
        print(f"\nDataset: {args.export} ({', '.join(f'{name} {part['windows']}' for name, part in meta['splits'].items())} windows)")

    if args.thesis_figure:
        chosen = next(r for r in found if r.sensor == "tensometer" and r.name == "test")
        original = next(r for r in published if r.group == chosen.group and r.name == chosen.name)
        plot_example(to_labelled_run(chosen), to_labelled_run(original), chosen.name, FIGURE)
        plot_label_timing(original, chosen, args.root, TIMING_FIGURE, "pl")
        print(f"\nFigures: {FIGURE}, {TIMING_FIGURE}")
    if args.notes_figure:
        chosen = next(r for r in found if r.group == "tens/synchronous_with_inode_acc" and r.name == "test")
        original = next(r for r in published if r.group == chosen.group and r.name == chosen.name)
        plot_label_timing(original, chosen, args.root, args.notes_figure, "en")
        print(f"Notes figure: {args.notes_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
