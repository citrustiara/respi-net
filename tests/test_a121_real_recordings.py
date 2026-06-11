from pathlib import Path

import pandas as pd
import pytest

from respi_net.a121_vitals import analyze_a121_vitals


def test_a121_real_recording_75hr_watch_reference() -> None:
    path = Path("data/raw/a121/a121_test_20s_75hr_2026-06-05_21-26-55.csv")
    if not path.exists():
        pytest.skip("local A121 75 HR real recording fixture is not available")

    analysis = analyze_a121_vitals(pd.read_csv(path))

    assert analysis.present
    assert 6.0 <= analysis.resp_bpm <= 20.0
    assert 74.0 <= analysis.heart_bpm <= 79.0
    assert analysis.heart_confidence >= 30.0


def test_a121_real_recording_80s_watch_reference() -> None:
    path = Path("data/raw/a121/a121_test_80s_80hr_2026-06-05_23-02-19.csv")
    if not path.exists():
        pytest.skip("local A121 80 s real recording fixture is not available")

    df = pd.read_csv(path)
    full_window = analyze_a121_vitals(df, max_frames=2000)
    gui_window = analyze_a121_vitals(df, max_frames=1200)

    assert full_window.present
    assert 6.0 <= full_window.resp_bpm <= 12.0
    assert 70.0 <= full_window.heart_bpm <= 80.0
    assert full_window.heart_confidence >= 30.0
    assert 70.0 <= gui_window.heart_bpm <= 80.0
    assert gui_window.heart_confidence >= 30.0
