#!/usr/bin/env python3
"""Capture a short, headless static-noise recording from LSM6DS3 and iPhone.

Both IMUs should be fixed on the same rigid, motionless surface.  The tool
records only after both transports pass a small preflight, then writes the
two CSV files and a manifest with UART/BLE and time-coverage diagnostics.

Example:

    uv run python tools/imu_lsm6ds3_iphone_static_capture.py \\
      --lsm-port /dev/cu.usbserial-XXXX --iphone-device RespiPhoneIMU
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from respi_net.imu import DEFAULT_IMU_BAUD, IMU_COLUMNS, LSM6DS3_CAPTURE_COLUMNS, BreathCapture
from respi_net.imu_guided_protocol import counter_delta, sample_timing_summary, stream_coverage_summary
from respi_net.iphone_imu import IPhoneIMUBluetoothCapture


DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "imu" / "lsm6ds3_iphone_static"


def _wait_for_rows(capture: Any, minimum_rows: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if capture.data_count() >= minimum_rows:
            return True
        time.sleep(0.05)
    return capture.data_count() >= minimum_rows


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "-" for character in value).strip("-._")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lsm-port", required=True, help="Dokładny port ESP32 z LSM6DS3.")
    parser.add_argument("--lsm-baud", type=int, default=DEFAULT_IMU_BAUD)
    parser.add_argument("--iphone-device", default=None, help="Nazwa albo adres BLE aplikacji RespiPhoneIMU.")
    parser.add_argument("--duration", type=float, default=60.0, help="Długość właściwego zapisu [s], domyślnie 60.")
    parser.add_argument("--prep-seconds", type=float, default=3.0, help="Czas ustabilizowania po połączeniu [s].")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--session-name", default=None)
    args = parser.parse_args()

    duration_s = max(5.0, float(args.duration))
    session_name = args.session_name or f"imu_static_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    session_dir = Path(args.output_dir) / _safe_name(session_name)
    session_dir.mkdir(parents=True, exist_ok=False)

    lsm = BreathCapture(baud=args.lsm_baud, output_dir=session_dir)
    iphone = IPhoneIMUBluetoothCapture(output_dir=session_dir, device=args.iphone_device, autostart=True)
    manifest: dict[str, Any] = {
        "experiment": "Simultaneous static LSM6DS3 and iPhone IMU noise recording",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instruction": "Both IMUs fixed on the same rigid, motionless surface; do not touch them during capture.",
        "duration_requested_s": duration_s,
        "status": "failed",
    }

    try:
        print(f"Łączenie z LSM6DS3: {args.lsm_port} ({args.lsm_baud} baud)…", flush=True)
        if not lsm.connect(args.lsm_port, exact_port=True) or not _wait_for_rows(lsm, 20, 4.0):
            raise RuntimeError("LSM6DS3 nie dostarcza próbek.")
        if lsm.diagnostics()["format"] != "timestamped":
            raise RuntimeError("LSM6DS3 nie używa formatu UART z czasem i licznikiem.")

        print("Łączenie z iPhone przez BLE…", flush=True)
        if not iphone.connect(args.iphone_device) or not _wait_for_rows(iphone, 20, 4.0):
            raise RuntimeError("iPhone nie dostarcza próbek BLE.")

        print(f"Ustabilizowanie: {args.prep_seconds:.0f} s…", flush=True)
        time.sleep(max(0.0, float(args.prep_seconds)))
        lsm_start = lsm.data_count()
        iphone_start = iphone.data_count()
        iphone_counters_before = iphone.counter_snapshot()
        start_wall_ms = time.time() * 1000.0
        print(f"Zapis nieruchomy: {duration_s:.0f} s…", flush=True)
        time.sleep(duration_s)
        end_wall_ms = time.time() * 1000.0
        lsm_rows = lsm.snapshot_capture_since(lsm_start)
        iphone_rows = iphone.snapshot_data_since(iphone_start)
        iphone_counters = counter_delta(iphone.counter_snapshot(), iphone_counters_before)
        lsm_diagnostics = lsm.diagnostics(lsm_start)
        iphone_timing = sample_timing_summary(iphone_rows)
        iphone_coverage = stream_coverage_summary(
            iphone_rows,
            window_start_ms=start_wall_ms,
            expected_duration_s=duration_s,
            expected_sample_rate_hz=100.0,
        )
        if len(lsm_rows) < duration_s * 80 or len(iphone_rows) < duration_s * 80:
            raise RuntimeError("Za mało próbek w jednym z torów; pomiar nie zostanie uznany za poprawny.")

        lsm_path = session_dir / "lsm6ds3_static.csv"
        iphone_path = session_dir / "iphone_static.csv"
        pd.DataFrame(lsm_rows, columns=LSM6DS3_CAPTURE_COLUMNS).to_csv(lsm_path, index=False)
        pd.DataFrame(iphone_rows, columns=IMU_COLUMNS).to_csv(iphone_path, index=False)
        manifest.update(
            {
                "status": "accepted",
                "measurement_start_utc": datetime.fromtimestamp(start_wall_ms / 1000.0, tz=timezone.utc).isoformat(),
                "measurement_start_wall_ms": start_wall_ms,
                "measurement_end_wall_ms": end_wall_ms,
                "actual_duration_s": (end_wall_ms - start_wall_ms) / 1000.0,
                "files": {"lsm6ds3": str(lsm_path.resolve()), "iphone": str(iphone_path.resolve())},
                "lsm6ds3": lsm_diagnostics,
                "iphone": {**iphone_timing, "window_coverage": iphone_coverage, "ble_batches": iphone_counters},
            }
        )
        print(
            f"Zapisano: {session_dir}\n"
            f"LSM6DS3: {len(lsm_rows)} próbek, luki licznika: {lsm_diagnostics['missing_samples']}\n"
            f"iPhone: {len(iphone_rows)} próbek, luki BLE: {iphone_counters['missing_batches']}, "
            f"maks. Δt: {iphone_coverage['largest_gap_s']:.3f} s",
            flush=True,
        )
        return 0
    except Exception as exc:
        manifest["error"] = str(exc)
        print(f"Błąd pomiaru: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        lsm.stop()
        iphone.stop()
        (session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
