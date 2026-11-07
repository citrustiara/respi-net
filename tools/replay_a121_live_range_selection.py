#!/usr/bin/env python3
"""The live A121 chest trace replayed from recordings: does the live range selection pick the chest, right way up?

Each recording of ``tools/compare_a121_range_selection.py`` is fed frame by frame (a few frames per call, as the
app's timer does) through :class:`respi_net.a121_vitals.A121LiveTraceProcessor`, once with
``range_selection="acconeer"`` (Acconeer's target bin, as before) and once with the default ``"breathing"``
(:class:`respi_net.a121_range.LiveRangeSelector`).  The target is Acconeer's per frame (its presence distance while
it determines the distance), with its range as the gate, as the app passes them.  For each run: when the selection
moved and where it ended, the polarity the selector holds for the bins it started and ended on (+1 right way up,
-1 upside down, 0 unsure; rise share and share of updates under one half), the share of the time it was frozen,
and -- where the recording has a reference -- the offline phase detector on both live traces (detrended) scored
against the cues, the belt and the phone as in the offline comparison, with the signed correlation with the belt
or the cue template.

    uv run python tools/replay_a121_live_range_selection.py
    uv run python tools/replay_a121_live_range_selection.py --only 50cm --csv out.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
from scipy.signal import detrend

from compare_a121_range_selection import (
    GRID_HZ,
    Recording,
    _oriented,
    best_lag_accuracy,
    cue_reference,
    on_grid,
    recordings,
    references,
    signed_band_correlation,
)
from respi_net import a121_range
from respi_net.a121 import A121_CAPTURE_COLUMNS
from respi_net.a121_vitals import A121LiveTraceProcessor
from respi_net.breath_phases import merge_holds
from respi_net.chest_signal import ChestSignal, regular_frame_times_ms
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import score_phases
from respi_net.phone_annotation import PHONE_DETECTOR

FRAMES_PER_CALL = 5


def _target(row: pd.Series) -> tuple[float | None, float]:
    """Acconeer's target (its presence distance while it determines the distance) and the gate half width."""

    target = pd.to_numeric(row.get("AcconeerTargetDistance_m"), errors="coerce")
    if not (np.isfinite(target) and target > 0) and str(row.get("AcconeerAppState", "")) == "DETERMINE_DISTANCE_ESTIMATE":
        target = pd.to_numeric(row.get("AcconeerPresenceDistance_m"), errors="coerce")
    if not (np.isfinite(target) and target > 0):
        return None, 0.05
    lo, hi = (pd.to_numeric(row.get(key), errors="coerce") for key in ("AcconeerRangeStart_m", "AcconeerRangeEnd_m"))
    half = max(abs(target - lo), abs(hi - target), 0.0025) if np.isfinite(lo) and np.isfinite(hi) else 0.05
    return float(target), float(half)


def replay(frame: pd.DataFrame, selection: str) -> dict[str, Any]:
    """Feed the recording through the live processor; the chest trace and how the chest bin moved."""

    table = frame.reindex(columns=A121_CAPTURE_COLUMNS).astype(object)
    rows = table.where(table.notna(), None).to_numpy().tolist()
    processor = A121LiveTraceProcessor(max_history_s=1e5, range_selection=selection)
    selected, frozen = [], []
    for start in range(0, len(rows), FRAMES_PER_CALL):
        chunk = rows[start : start + FRAMES_PER_CALL]
        target, half = _target(frame.iloc[start + len(chunk) - 1])
        result = processor.process_rows(chunk, target_distance_m=target, use_gating=target is not None, gate_half_width_m=half)
        if result is None:
            continue
        selected += [result.selected_index] * len(chunk)
        frozen += [result.range_frozen] * len(chunk)
    result = processor.result()
    assert result is not None
    return {"result": result, "processor": processor, "selected": np.asarray(selected), "frozen": np.asarray(frozen)}


def _polarity_text(selector: a121_range.LiveRangeSelector | None, index: int) -> str:
    if selector is None:
        return ""
    sign = int(selector.polarity()[index])
    return f"{sign:+d} ({selector.rise_share[index]:.3f}, {selector.rise_below[index]:.2f}, {selector.polarity_s[index]:.0f} s)"


def evaluate(recording: Recording) -> dict[str, Any]:
    frame = pd.read_csv(recording.a121)
    stamps = regular_frame_times_ms(pd.to_numeric(frame["Timestamp_ms"], errors="coerce").to_numpy(dtype=float))
    origin = float(stamps[0]) if recording.origin_ms is None else float(recording.origin_ms)
    runs = {selection: replay(frame, selection) for selection in ("acconeer", "breathing")}
    new = runs["breathing"]
    processor, selected = new["processor"], new["selected"]
    distances = processor.distances
    selector = processor.range_selector
    time_s = (stamps - origin) / 1000.0
    prior = runs["acconeer"]["selected"]
    # Moves of the selector itself: the live bin changes while Acconeer's does not.
    moves = np.flatnonzero((np.diff(selected) != 0) & (np.diff(prior) == 0)) + 1
    apart = selected != prior
    row: dict[str, Any] = {
        "kind": recording.kind,
        "recording": recording.name,
        "minutes": round((time_s[-1] - time_s[0]) / 60.0, 1),
        "acconeer_end_m": round(float(distances[prior[-1]]), 3),
        "live_end_m": round(float(distances[selected[-1]]), 3),
        "moves": len(moves),
        "moved_at_s": " ".join(f"{time_s[index] - time_s[0]:.0f}:{distances[selected[index]]:.3f}" for index in moves[:5]),
        "apart_share": round(float(np.mean(apart)), 3),
        "frozen_share": round(float(np.mean(new["frozen"])), 3),
        "acconeer_polarity": _polarity_text(selector, int(prior[-1])),
        "live_polarity": _polarity_text(selector, int(selected[-1])),
    }
    signals = {}
    for selection, run in runs.items():
        chest = np.asarray(run["result"].chest_mm, dtype=float)
        if len(chest) != len(time_s):
            raise ValueError(f"{recording.name}: {len(chest)} live samples for {len(time_s)} frames")
        signals[selection] = ChestSignal("a121", time_s, run["result"].sample_rate_hz, detrend(chest), "mm", chest, np.zeros(len(chest)))
    old, live = signals["acconeer"], signals["breathing"]
    row["changed"] = bool(apart.any())
    grid = np.arange(max(0.0, float(time_s[0])), float(time_s[-1]), 1.0 / GRID_HZ)
    old_trace, new_trace = on_grid(old, grid), on_grid(live, grid)
    old_on = on_grid(old, grid, merge_holds(detect_phases(old.chest, old.fs)))
    new_on = on_grid(live, grid, merge_holds(detect_phases(live.chest, live.fs)))
    row["old_vs_new_agreement"] = round(score_phases(old_on, new_on).accuracy, 3)
    if recording.cues:
        old_cues, old_match = cue_reference(grid, old_trace, recording.cues)
        new_cues, new_match = cue_reference(grid, new_trace, recording.cues)
        row |= {"cue_r_old": round(old_match, 3), "cue_r_new": round(new_match, 3)}
        row |= {"cue_acc_old": round(best_lag_accuracy(old_cues, old_on)[0], 3), "cue_acc_new": round(best_lag_accuracy(new_cues, new_on)[0], 3)}
    for name, labels, trace in references(recording, grid):
        key = name.replace(" ", "_")
        if name == "belt" and trace is not None:
            row |= {"belt_r_old": round(signed_band_correlation(old_trace, trace), 3), "belt_r_new": round(signed_band_correlation(new_trace, trace), 3)}
        if name == "phone":
            labels_old = merge_holds(detect_phases(_oriented(trace, old_trace), GRID_HZ, **PHONE_DETECTOR))
            labels_new = merge_holds(detect_phases(_oriented(trace, new_trace), GRID_HZ, **PHONE_DETECTOR))
        elif labels is not None:
            labels_old = labels_new = labels
        else:
            continue
        row |= {f"{key}_acc_old": round(best_lag_accuracy(labels_old, old_on)[0], 3), f"{key}_acc_new": round(best_lag_accuracy(labels_new, new_on)[0], 3)}
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[], help="only recordings whose name contains this (repeatable)")
    parser.add_argument("--skip", action="append", default=[], help="leave out recordings whose name contains this")
    parser.add_argument("--polarity-margin", type=float, help=f"try another POLARITY_MARGIN (default {a121_range.POLARITY_MARGIN}; inf: no polarity)")
    parser.add_argument("--keep-old-fraction", type=float, help=f"try another KEEP_OLD_FRACTION (default {a121_range.KEEP_OLD_FRACTION})")
    parser.add_argument("--csv", type=Path, help="also write the table here")
    args = parser.parse_args()
    if args.polarity_margin is not None:
        a121_range.POLARITY_MARGIN = args.polarity_margin
    if args.keep_old_fraction is not None:
        a121_range.KEEP_OLD_FRACTION = args.keep_old_fraction
    rows = []
    for recording in recordings([]):
        if (args.only and not any(part in recording.name for part in args.only)) or any(part in recording.name for part in args.skip):
            continue
        rows.append(evaluate(recording))
        print(f"{recording.kind:12s} {recording.name:40s} {rows[-1]['acconeer_end_m']:.3f} -> {rows[-1]['live_end_m']:.3f} ({rows[-1]['moves']} moves)", flush=True)
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 320, "display.max_columns", 60, "display.max_colwidth", 60):
        print()
        print(table.to_string(index=False))
    print(f"\n{len(table)} recordings, live chest bin moved in {int(table['changed'].sum())}.")
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
