#!/usr/bin/env python3
"""How reliable is the hand-labelled reference, and how close is the phone detector to it?

``reliability``: two labelling passes over the same stretch (``--pass 1`` and ``--pass 2`` of
``annotate_phone_phases.py``).  Their disagreement -- sample agreement away from the boundaries, how many boundaries
are found, how far they move -- is the uncertainty of the reference itself.

``belt``: the belt labels of ``tools/label_strain_belt_runs.py`` against a hand annotation of the belt trace
(``annotate_phone_phases.py --belt``): how good the belt is as the reference for the radar.  Both hold classes of the
belt count as one hold, as in the hand labels.

``validate``: the deterministic detector on the phone trace (merged holds, several minimum hold lengths) against
one annotation.  The detector is as good as the reference allows when it is about as close to the annotation as the
two passes are to each other.

    uv run python tools/score_phone_annotations.py reliability annotations/A_pass1.csv annotations/A_pass2.csv
    uv run python tools/score_phone_annotations.py belt annotations/<run>__maciek__0-75__pass1.csv
    uv run python tools/score_phone_annotations.py validate --session self_lying_3min_01 --run run_01_nn_self_3min annotations/A_pass1.csv
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

import numpy as np

from respi_net.breath_phases import IGNORE, merge_holds
from respi_net.phase_baseline import detect_phases
from respi_net.phone_annotation import (
    GUARD_S,
    HOLD_GUARD_S,
    PHONE_DETECTOR,
    compare_labels,
    load_segments,
    oriented_phone,
    segments_to_labels,
)

SESSIONS = ROOT / "data" / "raw" / "nn" / "a121_iphone"
GRID_HZ = 20.0


def reliability(first: Path, second: Path) -> dict[str, object]:
    a, b = load_segments(first), load_segments(second)
    end = max(segment.end_s for segment in [*a, *b])
    grid = np.arange(0.0, end + 1.0, 1.0 / GRID_HZ)
    return {
        "first_vs_second": compare_labels(segments_to_labels(a, grid), segments_to_labels(b, grid), grid),
        "first_vs_second_guarded": compare_labels(
            segments_to_labels(a, grid), segments_to_labels(b, grid), grid, guard_s=GUARD_S, hold_guard_s=HOLD_GUARD_S
        ),
    }


def validate(session: str, run: str, annotation: Path, min_holds: list[float], slow_motion: bool = True) -> dict[str, object]:
    grid, phone, sign = oriented_phone(SESSIONS / session / run, GRID_HZ)
    segments = load_segments(annotation)
    reference = segments_to_labels(segments, grid)
    out: dict[str, object] = {"phone_sign": sign, "annotated_s": float(np.sum(reference != IGNORE) / GRID_HZ)}
    for minimum in min_holds:
        detected = merge_holds(detect_phases(phone, GRID_HZ, min_hold_s=minimum, slow_motion_fraction=PHONE_DETECTOR["slow_motion_fraction"] if slow_motion else None))
        out[f"min_hold_{minimum:g}s"] = compare_labels(reference, detected, grid)
        out[f"min_hold_{minimum:g}s_guarded"] = compare_labels(reference, detected, grid, guard_s=GUARD_S, hold_guard_s=HOLD_GUARD_S)
    return out


BELT_LABELS = ROOT / "data" / "processed" / "strain_belt" / "labels"


def belt(annotations: list[Path]) -> dict[str, object]:
    """Belt labels against hand annotations of the belt trace, per annotation and over all of them pooled."""

    from respi_net.breath_phases import NOISE, NOISE_EXPORT_CODE
    import pandas as pd

    out: dict[str, object] = {}
    for annotation in annotations:
        name = "__".join(annotation.stem.split("__")[:-3])
        table = pd.read_csv(BELT_LABELS / f"{name}.csv")
        grid = table["time_s"].to_numpy(dtype=float)
        codes = table["label"].to_numpy()
        belt_labels = merge_holds(np.where(codes == NOISE_EXPORT_CODE, NOISE, codes).astype(np.int8))
        reference = segments_to_labels(load_segments(annotation), grid)
        out[annotation.name] = {
            "annotated_s": float(np.sum(reference != IGNORE) * np.median(np.diff(grid))),
            "agreement": compare_labels(reference, belt_labels, grid),
            "agreement_guarded": compare_labels(reference, belt_labels, grid, guard_s=GUARD_S, hold_guard_s=HOLD_GUARD_S),
        }
    if len(annotations) > 1:  # sample-weighted over all annotations
        scores = [entry["agreement"] for entry in out.values()]
        weights = np.array([score["samples"] for score in scores], dtype=float)
        out["pooled"] = {
            key: float(np.average([score[key] for score in scores], weights=weights))
            for key in ("accuracy", "macro_f1", "boundaries_found", "boundary_median_abs_error_s")
        } | {"samples": int(weights.sum()), "boundaries": int(sum(score["boundaries"] for score in scores))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rel = sub.add_parser("reliability", help="compare two labelling passes of one stretch")
    rel.add_argument("first", type=Path)
    rel.add_argument("second", type=Path)
    val = sub.add_parser("validate", help="the phone detector against one annotation")
    val.add_argument("--session", required=True)
    val.add_argument("--run", required=True)
    val.add_argument("annotation", type=Path)
    val.add_argument("--min-hold", type=float, nargs="+", default=[2.0, 3.0, 4.0], help="minimum hold lengths to try [s]")
    blt = sub.add_parser("belt", help="the strain-belt labels against hand annotations of the belt trace")
    blt.add_argument("annotations", type=Path, nargs="+")
    val.add_argument("--no-slow-motion", action="store_true", help="the plain detector, without the slow-motion check")
    args = parser.parse_args()
    if args.command == "belt":
        print(json.dumps(belt(args.annotations), indent=2))
        return 0
    result = reliability(args.first, args.second) if args.command == "reliability" else validate(args.session, args.run, args.annotation, args.min_hold, not args.no_slow_motion)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
