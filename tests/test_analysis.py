from pathlib import Path

import numpy as np
import pandas as pd

from respi_net.imu import analyze_imu_csv
from respi_net.radar import analyze_radar_csv


def test_analyze_radar_csv(tmp_path: Path) -> None:
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    csv_path = tmp_path / "radar_raw_sample.csv"
    pd.DataFrame(
        {
            "Timestamp_ms": t * 1000,
            "RawADC": 2048 + 20 * np.sin(2 * np.pi * 2.0 * t),
            "Voltage_mV": 1650 + 50 * np.sin(2 * np.pi * 2.0 * t),
        }
    ).to_csv(csv_path, index=False)

    result = analyze_radar_csv(csv_path, output_dir=tmp_path)

    assert result.plot_path and result.plot_path.exists()
    assert 95 <= result.sample_rate_hz <= 105
    assert 1.8 <= result.peak_frequency_hz <= 2.2


def test_analyze_imu_csv(tmp_path: Path) -> None:
    fs = 100.0
    t = np.arange(0, 30, 1 / fs)
    breathing = 0.03 * np.sin(2 * np.pi * 0.25 * t)
    heart = 0.005 * np.sin(2 * np.pi * 1.2 * t)
    csv_path = tmp_path / "respiratory_6axis_raw_sample.csv"
    pd.DataFrame(
        {
            "Time_ms": t * 1000,
            "ax": breathing + heart,
            "ay": 0.5 * breathing,
            "az": 1.0 + 0.2 * breathing,
            "gx": 0.01 * np.cos(2 * np.pi * 0.25 * t),
            "gy": 0.01 * np.sin(2 * np.pi * 0.25 * t),
            "gz": 0.01 * np.cos(2 * np.pi * 0.1 * t),
        }
    ).to_csv(csv_path, index=False)

    result = analyze_imu_csv(csv_path, output_dir=tmp_path)

    assert result.plot_path and result.plot_path.exists()
    assert 95 <= result.sample_rate_hz <= 105
    assert result.heart_bpm >= 40

