#!/usr/bin/env python3
"""Launch the guided 2 m A121 foil/angle retest.

The experiment uses only the hyperbolic lens and follows the sequence defined
in ``configs/a121_foil_2m_retest.json``: all six foil runs are recorded first,
then all six no-foil runs, while the two geometries are interleaved inside each
block. Each 90-second run includes visual and audible inhale/exhale/hold cues.
Press Esc or click the red discard button at any time to stop and delete the
current run.

Run from the repository root with::

    uv run python tools/a121_foil_2m_experiment.py

Use ``--dry-run --port TEST`` to inspect the sequence without opening the GUI
or connecting to the sensor.
"""

from __future__ import annotations

from pathlib import Path

from a121_guided_experiment import main


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "a121_foil_2m_retest.json"


if __name__ == "__main__":
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Missing experiment configuration: {CONFIG_PATH}")
    raise SystemExit(
        main(
            default_config_path=CONFIG_PATH,
            description=__doc__,
            enable_default_config_write=False,
        )
    )
