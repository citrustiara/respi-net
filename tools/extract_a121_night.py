#!/usr/bin/env python
"""A whole A121 night in one compact file: the complex echo of every range bin, the frame times and states.

A night's CSV is 1-2 GB of text (every frame's IQ as JSON strings); the analyses of the sleep chapter -- breathing
features, the heart-rate network -- need the IQ again and again, and the raw files live in a 5 GB archive on a
nearly full disk.  This reads the CSV once, in chunks, and writes ``<stem>.npz`` with

* ``timestamp_ms`` [N] (float64, as recorded), ``distances_m`` [B],
* ``iq`` [N, B, 2] (float16 real and imaginary part; echoes stay below 65 000 and keep 11 bits),
* ``state`` [N] (int8: 0 no presence, 1 movement, 2 still, by the rules of the night pipeline),
* ``acconeer_target_m``, ``acconeer_rr_bpm`` [N] (float32, NaN where not recorded),

about 130 MB for an 8 h night of 56 bins.

    uv run python tools/extract_a121_night.py data/raw/a121/a121_sleep_noc-2026-10-11_2026-10-11_03-47-35.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd

from analyze_a121_sleep_vitals_gated import _state_codes

OUT_DIR = ROOT / "data" / "processed" / "a121_nights"
SCALARS = ["Timestamp_ms", "AcconeerAppState", "AcconeerPresenceDetected", "AcconeerTargetDistance_m", "AcconeerBreathingRate_BPM"]


def _matrix(texts: pd.Series, width: int) -> np.ndarray:
    """JSON arrays ("[1.0,2.0,...]") of one column into an [n, width] float32 matrix."""

    joined = ",".join(text[1:-1] for text in texts.astype(str))
    values = np.array(joined.split(","), dtype=np.float32)
    if values.size != len(texts) * width:
        rows = [np.array(text[1:-1].split(","), dtype=np.float32)[:width] for text in texts.astype(str)]
        return np.vstack(rows)
    return values.reshape(len(texts), width)


def extract(csv_path: Path, out_dir: Path = OUT_DIR, chunk_rows: int = 50_000) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [*SCALARS, "Distances_m", "Real", "Imag"]
    distances: np.ndarray | None = None
    parts: dict[str, list[np.ndarray]] = {"iq": [], "timestamp_ms": [], "state": [], "acconeer_target_m": [], "acconeer_rr_bpm": []}
    for chunk in pd.read_csv(csv_path, usecols=lambda name: name in columns, chunksize=chunk_rows):
        if distances is None:
            distances = np.array(str(chunk["Distances_m"].iloc[0])[1:-1].split(","), dtype=np.float64)
        width = len(distances)
        real, imag = _matrix(chunk["Real"], width), _matrix(chunk["Imag"], width)
        parts["iq"].append(np.stack([real, imag], axis=-1).astype(np.float16))
        parts["timestamp_ms"].append(pd.to_numeric(chunk["Timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64))
        parts["state"].append(_state_codes(chunk).astype(np.int8))
        for key, column in (("acconeer_target_m", "AcconeerTargetDistance_m"), ("acconeer_rr_bpm", "AcconeerBreathingRate_BPM")):
            values = pd.to_numeric(chunk[column], errors="coerce") if column in chunk else pd.Series(np.nan, index=chunk.index)
            parts[key].append(values.to_numpy(dtype=np.float32))
    if distances is None:
        raise ValueError(f"No rows in {csv_path}")
    out = out_dir / f"{csv_path.stem}.npz"
    np.savez(out, distances_m=distances, **{key: np.concatenate(value) for key, value in parts.items()})
    return out


def load_night(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    for path in args.csv:
        out = extract(path, args.out_dir)
        night = load_night(out)
        hours = (night["timestamp_ms"][-1] - night["timestamp_ms"][0]) / 3.6e6
        print(f"{out}: {len(night['timestamp_ms'])} frames, {hours:.2f} h, {len(night['distances_m'])} bins, "
              f"{out.stat().st_size / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
