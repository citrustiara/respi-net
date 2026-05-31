from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMU_DIR = DATA_DIR / "raw" / "imu"
RAW_RADAR_DIR = DATA_DIR / "raw" / "radar"
RAW_A121_DIR = DATA_DIR / "raw" / "a121"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMU_PLOTS_DIR = OUTPUT_DIR / "plots" / "imu"
RADAR_PLOTS_DIR = OUTPUT_DIR / "plots" / "radar"

