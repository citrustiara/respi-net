#!/usr/bin/env python3
"""How reliable is the hand-labelled reference, and how close is the phone detector to it?

``reliability``: two labelling passes over the same stretch (``--pass 1`` and ``--pass 2`` of
``annotate_phone_phases.py``).  Their disagreement -- sample agreement away from the boundaries, how many boundaries
are found, how far they move -- is the uncertainty of the reference itself.

``validate``: the deterministic detector on the phone trace (merged holds, several minimum hold lengths) against
one annotation.  The detector is as good as the reference allows when it is about as close to the annotation as the
two passes are to each other.

    uv run python tools/score_phone_annotations.py reliability annotations/A_pass1.csv annotations/A_pass2.csv
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
from respi_net.phone_annotation import compare_labels, load_segments, oriented_phone, segments_to_labels

SESSIONS = ROOT / "data" / "raw" / "nn" / "a121_iphone"
GRID_HZ = 20.0


def reliability(first: Path, second: Path) -> dict[str, object]:
    a, b = load_segments(first), load_segments(second)
    end = max(segment.end_s for segment in [*a, *b])
    grid = np.arange(0.0, end + 1.0, 1.0 / GRID_HZ)
    labels_a, labels_b = segments_to_labels(a, grid), segments_to_labels(b, grid)
    return {"first_vs_second": compare_labels(labels_a, labels_b, grid), "second_vs_first": compare_labels(labels_b, labels_a, grid)}


def validate(session: str, run: str, annotation: Path, min_holds: list[float]) -> dict[str, object]:
    grid, phone, sign = oriented_phone(SESSIONS / session / run, GRID_HZ)
    reference = segments_to_labels(load_segments(annotation), grid)
    out: dict[str, object] = {"phone_sign": sign, "annotated_s": float(np.sum(reference != IGNORE) / GRID_HZ)}
    for minimum in min_holds:
        detected = merge_holds(detect_phases(phone, GRID_HZ, min_hold_s=minimum))
        out[f"min_hold_{minimum:g}s"] = compare_labels(reference, detected, grid)
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
    args = parser.parse_args()
    result = reliability(args.first, args.second) if args.command == "reliability" else validate(args.session, args.run, args.annotation, args.min_hold)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
