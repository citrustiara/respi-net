from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMU_DIR = DATA_DIR / "raw" / "imu"
RAW_RADAR_DIR = DATA_DIR / "raw" / "radar"
RAW_A121_DIR = DATA_DIR / "raw" / "a121"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMU_PLOTS_DIR = OUTPUT_DIR / "plots" / "imu"
RADAR_PLOTS_DIR = OUTPUT_DIR / "plots" / "radar"
SLEEP_PLOTS_DIR = OUTPUT_DIR / "plots" / "a121_sleep"

RECORDINGS_DB_FILENAME = "respi_recordings.sqlite3"
PROJECT_RECORDINGS_DB_PATH = DATA_DIR / RECORDINGS_DB_FILENAME
RECORDINGS_DB_ENV_VARS = ("RESPI_RECORDINGS_DB_PATH", "RESPI_DB_PATH")


def _windows_e_drive_recordings_db_path() -> Path | None:
    """Return the preferred Windows E: drive DB path, or None off Windows."""
    if os.name != "nt":
        return None
    return Path("E:/") / RECORDINGS_DB_FILENAME


def resolve_recordings_db_path() -> Path:
    """Choose the SQLite recordings DB path.

    Priority:
    1. An explicit environment override (RESPI_RECORDINGS_DB_PATH, then RESPI_DB_PATH).
    2. E:/respi_recordings.sqlite3 on Windows, but only if that file already exists.
    3. The project-local data/respi_recordings.sqlite3 fallback, which also works on macOS/Linux.
    """
    for env_var in RECORDINGS_DB_ENV_VARS:
        raw_path = os.environ.get(env_var)
        if raw_path:
            return Path(raw_path).expanduser()

    external_db_path = _windows_e_drive_recordings_db_path()
    if external_db_path is not None and external_db_path.is_file():
        return external_db_path

    return PROJECT_RECORDINGS_DB_PATH


DB_PATH = resolve_recordings_db_path()
