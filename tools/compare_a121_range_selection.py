#!/usr/bin/env python3
"""Does choosing the A121 chest bins by breathing score change any labels, and for the better?

For every A121 recording with something to compare the radar with, the chest signal is built twice --
``range_selection="acconeer"`` (the recorded choice, as before) and the default ``"breathing"``
(:mod:`respi_net.a121_range`) -- and the offline phase detector labels both.  Where the gate is the same the
labels are bit-identical; where it moved, both label sets are scored against the most independent reference
the recording has, each at its own best delay within +-1.5 s (the sensors and the cues are not in step):

* coach runs (2 m foil retest, HB100 range runs, cued radar + phone / belt runs): the cues, shifted by the
  delay that matches them to the trace (:func:`respi_net.label_alignment.estimate_lag`, whose correlation is
  reported too) -- coarse, people do not follow cues to the tenth of a second, but the same for both;
* radar + belt: the belt labels of ``tools/label_strain_belt_runs.py`` (made here the same way for a session
  folder with a plain ``a121.csv`` + ``belt.csv``, recorded without the guided recorder) and the hand labels of
  the belt trace;
* radar + phone: the hand labels of the phone trace where there are any, and the phone detector's labels;

and the correlation of the chest trace with the belt or phone trace.  Recordings with no reference (long
close-range runs, the 90 s watch check) are listed with their gates and scores only.

Which way up each trace is: ``cue_r`` is the cue template's correlation with the chest velocity at its best delay,
signed (negative: the chest falls on inhale), ``belt_r`` the signed correlation with the belt in 0.05-0.7 Hz at the
delay of the largest magnitude; the phone's axis is turned to the radar, so it cannot tell.  The polarity the
selection reads from the breath shape (:func:`respi_net.a121_range.polarity`, +1 right way up, -1 upside down,
0 unsure) is listed for the old gate and the best group with its rise share and the share of windows under one
half.  ``--polarity-margin inf`` gives the rule before the polarity (a group moving against the old gate never
replaces it), ``--flip`` turns a chosen trace over when its polarity says upside down, ``--ignore-direction`` switches
on the score alone.

    uv run python tools/compare_a121_range_selection.py
    uv run python tools/compare_a121_range_selection.py --extra path/to/a121.csv --csv out.csv --score-unchanged
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

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from respi_net import a121_range
from respi_net.breath_phases import IGNORE, from_export_codes, merge_holds
from respi_net.chest_signal import ChestSignal, a121_chest_signal, imu_chest_signal, strain_belt_chest_signal
from respi_net.label_alignment import cue_phases, estimate_lag
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import score_phases
from respi_net.phone_annotation import PHONE_DETECTOR, load_segments, segments_to_labels
from label_strain_belt_runs import BELT_MIN_HOLD_S

GRID_HZ = 20.0
MAX_LAG_S = 1.5
ANNOTATIONS = ROOT / "annotations"
BELT_LABELS = ROOT / "data" / "processed" / "strain_belt" / "labels"


@dataclass
class Recording:
    kind: str
    name: str
    a121: Path
    origin_ms: float | None = None
    cues: list[dict[str, Any]] | None = None
    prefix: Path | None = None  # run prefix of a guided radar + phone / belt session
    belt: Path | None = None  # belt file of a plain session folder (no cues, no processed labels)


def recordings(extra: list[Path]) -> list[Recording]:
    found: list[Recording] = []
    for manifest_path in sorted((ROOT / "data/raw/a121/guided").glob("guided_foil2m_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("measurements", []):
            if item.get("status") == "accepted":
                found.append(Recording("coach 2 m", f"2m_{int(item['step']):02d}", manifest_path.parent / Path(item["csv_path"]).name,
                                       cues=list(manifest["breathing_cues"])))
    for cue_path in sorted((ROOT / "data/raw/hb100_a121/guided").glob("hb100_a121_range_*/run_*_range-*_cues.csv")):
        a121 = cue_path.with_name(cue_path.name.replace("_cues.csv", "_a121.csv"))
        if a121.exists():
            cues = pd.read_csv(cue_path)
            found.append(Recording("coach range", cue_path.name.split("_range-")[1].replace("_cues.csv", ""), a121,
                                   float(cues["start_wall_ms"].iloc[0]), cues[["start_s", "end_s", "kind"]].to_dict("records")))
    for a121 in sorted((ROOT / "data/raw/nn").glob("a121_*/*/run_*_a121.csv")):
        prefix = Path(str(a121)[: -len("_a121.csv")])
        cues = pd.read_csv(f"{prefix}_cues.csv")
        records = cues[["start_s", "end_s", "kind"]].to_dict("records")
        cued = bool(cues["kind"].isin(["inhale", "exhale"]).any())
        found.append(Recording(a121.parent.parent.name, a121.parent.name, a121, float(cues["start_wall_ms"].iloc[0]), records if cued else None, prefix))
    for a121 in sorted((ROOT / "data/raw/nn").glob("a121_*/*/a121.csv")):
        origin = float(pd.read_csv(a121, nrows=1)["Timestamp_ms"].iloc[0])  # host epoch, as the belt's
        belt = a121.with_name("belt.csv")
        found.append(Recording(a121.parent.parent.name, a121.parent.name, a121, origin, belt=belt if belt.exists() else None))
    for a121 in sorted((ROOT / "data/raw/a121").glob("a121_sparse_iq_*.csv")):
        if not a121.stem.endswith("_annotations"):
            found.append(Recording("close range", a121.stem.replace("a121_sparse_iq_", ""), a121))
    found += [Recording("extra", path.parent.name, path) for path in extra]
    return found


def on_grid(signal: ChestSignal, grid: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """``labels`` (nearest sample) or the chest trace (interpolated) of ``signal`` on ``grid``."""

    if labels is None:
        return np.interp(grid, signal.time_s, signal.chest)
    index = np.clip(np.searchsorted(signal.time_s, grid), 0, len(signal.time_s) - 1)
    out = labels[index].astype(np.int8)
    out[(grid < signal.time_s[0]) | (grid > signal.time_s[-1])] = IGNORE
    return out


def best_lag_accuracy(reference: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """``(accuracy, lag_s)`` at the delay of ``predicted`` that agrees best with ``reference``."""

    best = (float("nan"), 0.0)
    reach = int(round(MAX_LAG_S * GRID_HZ))
    for shift in range(-reach, reach + 1):
        ref = reference[max(0, shift) : len(reference) + min(0, shift)]
        pred = predicted[max(0, -shift) : len(predicted) + min(0, -shift)]
        accuracy = score_phases(ref, pred).accuracy
        if np.isfinite(accuracy) and not accuracy <= best[0]:
            best = (accuracy, shift / GRID_HZ)
    return best


def signed_band_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of the two traces in 0.05-0.7 Hz at the delay (within +-``MAX_LAG_S``) of the largest magnitude,
    with its sign: negative when one is upside down against the other."""

    sos = butter(2, (0.05, 0.7), "bandpass", fs=GRID_HZ, output="sos")
    fa, fb = sosfiltfilt(sos, a), sosfiltfilt(sos, b)
    reach = int(round(MAX_LAG_S * GRID_HZ))
    best = 0.0
    for shift in range(-reach, reach + 1):
        x = fa[max(0, shift) : len(fa) + min(0, shift)]
        y = fb[max(0, -shift) : len(fb) + min(0, -shift)]
        if np.std(x) > 0 and np.std(y) > 0:
            value = float(np.corrcoef(x, y)[0, 1])
            best = value if abs(value) > abs(best) else best
    return best


def best_lag_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Largest |correlation| of the two velocities within +-``MAX_LAG_S``."""

    va, vb = np.gradient(a), np.gradient(b)
    reach = int(round(MAX_LAG_S * GRID_HZ))
    best = 0.0
    for shift in range(-reach, reach + 1):
        x = va[max(0, shift) : len(va) + min(0, shift)]
        y = vb[max(0, -shift) : len(vb) + min(0, -shift)]
        if np.std(x) > 0 and np.std(y) > 0:
            best = max(best, abs(float(np.corrcoef(x, y)[0, 1])))
    return best


def cue_reference(grid: np.ndarray, trace: np.ndarray, cues: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    """Cue labels at the delay that matches them to ``trace``, and the template correlation there, signed (negative
    when ``trace`` falls on inhale)."""

    phases = cue_phases(cues)
    lag, match, sign = estimate_lag(grid, np.gradient(trace) * GRID_HZ, phases, np.arange(-1.5, 1.5001, 0.025))
    labels = np.full(len(grid), IGNORE, dtype=np.int8)
    for phase in phases:
        if phase.label != IGNORE:
            labels[(grid >= phase.start_s + lag) & (grid < phase.end_s + lag)] = phase.label
    return merge_holds(labels), sign * match


def references(recording: Recording, grid: np.ndarray) -> list[tuple[str, np.ndarray | None, np.ndarray | None]]:
    """``(name, labels or None, trace or None)`` of each reference for this recording, on ``grid``."""

    out: list[tuple[str, np.ndarray | None, np.ndarray | None]] = []
    prefix = recording.prefix
    if recording.belt is not None:
        belt = strain_belt_chest_signal(recording.belt, origin_ms=recording.origin_ms)
        labels = merge_holds(detect_phases(belt.chest, belt.fs, min_hold_s=BELT_MIN_HOLD_S))
        index = np.clip(np.searchsorted(belt.time_s, grid), 0, len(belt.time_s) - 1)
        out.append(("belt", labels[index], np.interp(grid, belt.time_s, belt.chest)))
    if prefix is not None and Path(f"{prefix}_belt.csv").exists():
        belt = strain_belt_chest_signal(f"{prefix}_belt.csv", origin_ms=recording.origin_ms)
        trace = np.interp(grid, belt.time_s, belt.chest)
        name = f"{prefix.parent.name}__{prefix.name}_belt"
        label_file = BELT_LABELS / f"{name}.csv"
        labels = None
        if label_file.exists():
            table = pd.read_csv(label_file)
            index = np.clip(np.searchsorted(table["time_s"].to_numpy(), grid), 0, len(table) - 1)
            labels = merge_holds(from_export_codes(table["label"].to_numpy()[index]))
        out.append(("belt", labels, trace))
        hand = sorted(ANNOTATIONS.glob(f"{prefix.parent.name}__{prefix.name}_belt__*__pass1.csv"))
        if hand:
            out.append(("belt hand", _hand_labels(hand, grid), None))
    if prefix is not None and Path(f"{prefix}_iphone.csv").exists():
        phone = imu_chest_signal(f"{prefix}_iphone.csv", origin_ms=recording.origin_ms, name="iphone")
        trace = np.interp(grid, phone.time_s, phone.chest)
        # The phone detector's labels are made per radar trace, from the phone trace turned to move with it.
        out.append(("phone", None, trace))
        hand = sorted(ANNOTATIONS.glob(f"{prefix.parent.name}__{prefix.name}__*__pass1.csv"))
        if hand:
            out.append(("phone hand", _hand_labels(hand, grid), None))
    return out


def _hand_labels(paths: list[Path], grid: np.ndarray) -> np.ndarray:
    labels = np.full(len(grid), IGNORE, dtype=np.int8)
    for path in paths:
        part = segments_to_labels(load_segments(path), grid)
        labels = np.where(part != IGNORE, part, labels)
    return merge_holds(labels)


def _oriented(trace: np.ndarray, radar: np.ndarray) -> np.ndarray:
    """A phone axis points either way; turn it so it moves with the radar (as the phone tools do)."""

    return trace if np.corrcoef(np.gradient(trace), np.gradient(radar))[0, 1] >= 0 else -trace


def compare(recording: Recording, score_unchanged: bool = False) -> dict[str, Any]:
    old = a121_chest_signal(recording.a121, origin_ms=recording.origin_ms, range_selection="acconeer")
    new = a121_chest_signal(recording.a121, origin_ms=recording.origin_ms)
    meta = new.meta
    row: dict[str, Any] = {
        "kind": recording.kind,
        "recording": recording.name,
        "minutes": round((old.time_s[-1] - old.time_s[0]) / 60.0, 1),
        "old_gate": f"{old.meta['gate_start_m']:.3f}-{old.meta['gate_end_m']:.3f}",
        "new_gate": f"{meta['gate_start_m']:.3f}-{meta['gate_end_m']:.3f}",
        "source": meta["gate_source"],
        "old_score": meta.get("old_gate_score", float("nan")),
        "best_score": meta.get("best_group_score", float("nan")),
        "best_gate": "-".join(f"{value:.3f}" for value in meta.get("best_group_m", [])),
        "in_phase": meta.get("best_group_in_phase"),
        "old_pol": meta.get("old_gate_polarity"),
        "old_rise": meta.get("old_gate_rise_share"),
        "old_below": meta.get("old_gate_rise_below"),
        "best_pol": meta.get("best_group_polarity"),
        "best_rise": meta.get("best_group_rise_share"),
        "best_below": meta.get("best_group_rise_below"),
        "flipped": "turned over" in str(meta.get("range_note", "")),
    }
    changed = (old.meta["gate_start_m"], old.meta["gate_end_m"]) != (meta["gate_start_m"], meta["gate_end_m"]) or row["flipped"]
    old_labels = merge_holds(detect_phases(old.chest, old.fs))
    new_labels = merge_holds(detect_phases(new.chest, new.fs))
    row["changed"] = changed
    row["labels_identical"] = bool(np.array_equal(old_labels, new_labels)) if not changed or len(old_labels) == len(new_labels) else False
    if not changed and not score_unchanged:
        return row
    start = max(0.0, float(old.time_s[0]))
    grid = np.arange(start, float(old.time_s[-1]), 1.0 / GRID_HZ)
    old_trace, new_trace = on_grid(old, grid), on_grid(new, grid)
    old_on, new_on = on_grid(old, grid, old_labels), on_grid(new, grid, new_labels)
    row["old_vs_new_agreement"] = round(score_phases(old_on, new_on).accuracy, 3)
    refs = references(recording, grid)
    if recording.cues:
        old_cues, old_match = cue_reference(grid, old_trace, recording.cues)
        new_cues, new_match = cue_reference(grid, new_trace, recording.cues)
        row |= {"cue_r_old": round(old_match, 3), "cue_r_new": round(new_match, 3)}
        row |= {"cue_acc_old": round(best_lag_accuracy(old_cues, old_on)[0], 3), "cue_acc_new": round(best_lag_accuracy(new_cues, new_on)[0], 3)}
    for name, labels, trace in refs:
        key = name.replace(" ", "_")
        if trace is not None:
            row |= {f"{key}_corr_old": round(best_lag_correlation(old_trace, trace), 3), f"{key}_corr_new": round(best_lag_correlation(new_trace, trace), 3)}
            if name == "belt":
                row |= {"belt_r_old": round(signed_band_correlation(old_trace, trace), 3), "belt_r_new": round(signed_band_correlation(new_trace, trace), 3)}
        if labels is not None or name == "phone":
            if name == "phone":
                labels_old = merge_holds(detect_phases(_oriented(trace, old_trace), GRID_HZ, **PHONE_DETECTOR))
                labels_new = merge_holds(detect_phases(_oriented(trace, new_trace), GRID_HZ, **PHONE_DETECTOR))
            else:
                labels_old = labels_new = labels
            row |= {f"{key}_acc_old": round(best_lag_accuracy(labels_old, old_on)[0], 3), f"{key}_acc_new": round(best_lag_accuracy(labels_new, new_on)[0], 3)}
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extra", type=Path, action="append", default=[], help="another A121 CSV to list (no reference)")
    parser.add_argument("--csv", type=Path, help="also write the table here")
    parser.add_argument("--score-unchanged", action="store_true", help="score the references where the gate did not move too")
    parser.add_argument("--keep-old-fraction", type=float, help=f"try another KEEP_OLD_FRACTION (default {a121_range.KEEP_OLD_FRACTION})")
    parser.add_argument("--group-half-width", type=int, help=f"try another GROUP_HALF_WIDTH (default {a121_range.GROUP_HALF_WIDTH})")
    parser.add_argument("--polarity-margin", type=float, help=f"try another POLARITY_MARGIN (default {a121_range.POLARITY_MARGIN}; inf: no polarity)")
    parser.add_argument("--polarity-agreement", type=float, help=f"try another POLARITY_AGREEMENT (default {a121_range.POLARITY_AGREEMENT})")
    parser.add_argument("--flip", action="store_true", help="turn a chosen trace over when its polarity says upside down")
    parser.add_argument("--ignore-direction", action="store_true", help="switch on the score alone, whichever way the group moves")
    parser.add_argument("--only", action="append", default=[], help="only recordings whose name contains this (repeatable)")
    args = parser.parse_args()
    if args.polarity_margin is not None:
        a121_range.POLARITY_MARGIN = args.polarity_margin
    if args.polarity_agreement is not None:
        a121_range.POLARITY_AGREEMENT = args.polarity_agreement
    a121_range.POLARITY_FLIP = bool(args.flip)
    if args.ignore_direction:
        a121_range.clearly_better = lambda new, old, *_: old < a121_range.KEEP_OLD_FRACTION * new
    if args.keep_old_fraction is not None:
        a121_range.KEEP_OLD_FRACTION = args.keep_old_fraction
    if args.group_half_width is not None:
        a121_range.GROUP_HALF_WIDTH = args.group_half_width

    rows = []
    for recording in recordings(args.extra):
        if args.only and not any(part in recording.name for part in args.only):
            continue
        rows.append(compare(recording, args.score_unchanged))
        print(f"{recording.kind:12s} {recording.name:28s} {rows[-1]['source']:15s} {rows[-1]['old_gate']} -> {rows[-1]['new_gate']}", flush=True)
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 320, "display.max_columns", 60):
        print()
        print(table.drop(columns=["labels_identical"]).to_string(index=False))
    unchanged = table[~table["changed"]]
    print(f"\n{len(table)} recordings, gate changed in {int(table['changed'].sum())}; labels identical in "
          f"{int(unchanged['labels_identical'].sum())} of {len(unchanged)} unchanged.")
    for column in sorted(c for c in table.columns if c.endswith("_old") and c.replace("_old", "_new") in table.columns):
        pair = table.loc[table["changed"], [column, column.replace("_old", "_new")]].dropna()
        if len(pair):
            change = pair.iloc[:, 1] - pair.iloc[:, 0]
            print(f"  {column[:-4]:18s} {len(pair)} changed recordings: mean {pair.iloc[:, 0].mean():.3f} -> {pair.iloc[:, 1].mean():.3f} "
                  f"(worst {change.min():+.3f}, best {change.max():+.3f})")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
