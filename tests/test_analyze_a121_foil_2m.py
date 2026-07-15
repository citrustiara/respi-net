from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_a121_foil_2m import (  # noqa: E402
    cue_template,
    estimate_cue_delay,
    paired_effects,
)


CUES = [
    {"kind": "normal", "start_s": 0.0, "end_s": 10.0},
    {"kind": "inhale", "start_s": 10.0, "end_s": 12.0},
    {"kind": "exhale", "start_s": 12.0, "end_s": 15.0},
    {"kind": "inhale", "start_s": 15.0, "end_s": 17.0},
    {"kind": "exhale", "start_s": 17.0, "end_s": 20.0},
]


def test_cue_template_has_expected_phase_orientation() -> None:
    time_s = np.asarray([10.4, 12.4, 15.4])
    template = cue_template(time_s, CUES, delay_s=0.4)

    assert np.allclose(template, [1.0, -1.0, 1.0])


def test_estimate_cue_delay_recovers_known_shift() -> None:
    time_s = np.arange(0.0, 20.0, 0.025)
    expected_delay_s = 0.425
    signal = cue_template(time_s, CUES, delay_s=expected_delay_s)
    signal = np.nan_to_num(signal)

    delay_s, correlation = estimate_cue_delay(time_s, signal, CUES)

    assert np.isclose(delay_s, expected_delay_s)
    assert correlation > 0.999


def test_paired_effects_match_repeats_before_converting_to_db() -> None:
    rows = []
    values = {
        "N0": (100.0, 100.0, 100.0),
        "NF": (200.0, 100.0, 50.0),
        "P0": (200.0, 400.0, 800.0),
        "PF": (400.0, 400.0, 400.0),
    }
    for condition, repeats in values.items():
        for repeat, value in enumerate(repeats, start=1):
            rows.append(
                {
                    "condition": condition,
                    "repeat": repeat,
                    "echo_peak_median": value,
                }
            )

    effects = paired_effects(pd.DataFrame(rows), "echo_peak_median")

    assert np.allclose(
        effects["foil_natural"],
        [20.0 * np.log10(2.0), 0.0, -20.0 * np.log10(2.0)],
    )
    assert np.allclose(
        effects["angle_no_foil"],
        [20.0 * np.log10(2.0), 20.0 * np.log10(4.0), 20.0 * np.log10(8.0)],
    )
