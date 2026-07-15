# RespiNet App and CLI Guide

This guide documents the unified desktop app, supported sensors, recording outputs, and useful CLI commands.

## Setup

Install/update the Python environment with `uv`:

```powershell
uv sync
```

The project dependencies include:

- `PySide6` - Qt desktop UI
- `pyqtgraph` - interactive live/history plots
- `pyserial` - ESP32/serial devices
- `acconeer-exptool[app]` - Acconeer A121 low-level Python API
- `numpy`, `pandas`, `scipy`, `matplotlib` - analysis and offline plots

## Supported sensors

### HB100 radar via ESP32 ADC

Use this for the existing analog HB100 radar pipeline. The current ESP32
firmware streams strict ASCII at 230400 baud and a stable 250 Hz in CSV-like
serial rows:

```text
Timestamp_ms,RawADC,Voltage_mV
```

Default output folder:

```text
data/raw/radar/
```

### Acconeer / Waveshare A121 radar

Use this for the Waveshare A121 Range Sensor connected over USB-C/UART.

Important: the Waveshare board exposes two serial ports. Use the first one, usually shown as Interface A:

- Windows: `USB-Enhanced-SERIAL-A CH342 (COMx)`
- Linux/macOS: usually `/dev/ttyUSB0` or similar

The app captures Sparse IQ data from the A121 and stores per-frame:

```text
Timestamp_ms,Frame,PeakDistance_m,PeakAmplitude,PeakPhase_rad,MeanAmplitude,
Distances_m,Amplitude,Phase,Real,Imag
```

`Distances_m`, `Amplitude`, `Phase`, `Real`, and `Imag` are JSON arrays in the CSV so the full distance profile can be replayed later.

Default output folder:

```text
data/raw/a121/
```

### IMU

Use this for the ESP32/LSM6DS3 comparison path. The current firmware streams a
sample counter and its own microsecond clock before the six axes; the desktop
CSV preserves both of them as well as the host arrival time:

```text
Time_ms,HostTime_ms,DeviceTime_us,SampleIndex,ax,ay,az,gx,gy,gz
```

`Time_ms` is the device clock aligned to the host epoch at first reception.
For legacy six-axis firmware rows, the same reader remains compatible, but the
counter and device-clock columns are empty and exact UART-loss detection is not
available.

Default output folder:

```text
data/raw/imu/
```

### iPhone IMU over BLE

Use this with the native SwiftUI companion app in `ios/RespiPhoneIMU/`. The iPhone advertises a custom BLE service named `RespiPhoneIMU`, samples CoreMotion at 20-100 Hz, and sends compact binary batches. The desktop receiver expands those batches to the same IMU CSV schema used by the ESP32 path:

```text
Time_ms,ax,ay,az,gx,gy,gz
```

Acceleration is in `g`, gyroscope values are in `deg/s`, and `Time_ms` is aligned to the desktop host clock from the first received phone sample.

## Unified desktop app

Open the app:

```powershell
uv run respi app
```

Open the app with a specific sensor and port:

```powershell
uv run respi app --sensor radar --port COM6
uv run respi app --sensor a121 --port COM3
uv run respi app --sensor imu --port COM6
uv run respi app --sensor iphone-imu
```

Available app sensor choices:

- `HB100 Radar`
- `A121 Radar`
- `IMU`
- `iPhone IMU (BLE)`

### App controls

Common controls:

- Sensor selector
- Serial port selector with refresh button
- Baud rate selector for ESP32-based sensors
- Recording target:
  - `CSV + SQLite`
  - `CSV only`
  - `SQLite only`
- Live window length in seconds
- Start/Stop recording buttons

A121-specific controls:

- `A121 target selection` - uses Acconeer's breathing reference state machine/presence-distance selector by default
- `A121 gate display` - overlays the currently selected compact range segment on the amplitude plot
- `A121 breath phases` - automatically shades detected inhale/exhale phases on A121 respiration plots and saves a `*_breath_annotations.csv` sidecar for CSV recordings
- `Offline/fallback gate` - compact segment width used only when no recorded/live Acconeer selected range is available
- `A121 start` - start distance in meters, e.g. `0.20 m`
- `A121 end` - end distance in meters, e.g. `1.50 m`
- `A121 profile` - Acconeer profile `1..5`; profile 3 is the default breathing-reference profile
- `A121 HWAAS` - hardware averaging; higher values reduce noise but reduce update rate
- `A121 sweeps` / `A121 fps` - defaults are 16 sweeps at 20 Hz to avoid UART backlog and GUI lag

### App graphs

Graphs are rendered with `pyqtgraph` and are interactive:

- drag to pan
- mouse wheel to zoom
- right-click for plot options

Live graph behavior:

- HB100: voltage over time + FFT
- A121: latest amplitude vs distance plus selectable filtered vitals, rate FFT, or raw IQ/phase view. The live time-domain A121 traces use stateful causal filters, so previous samples are not recomputed as new frames arrive.
- IMU: accelerometer axes + gyroscope axes

### Live stats

The stats panel shows sensor-dependent metrics:

- HB100:
  - sample rate
  - estimated respiration-band BPM
  - peak frequency
  - Doppler speed estimate
- A121:
  - frame rate
  - peak distance/amplitude and range gate
  - compact range-bin count and signal-quality index
  - Acconeer `BreathingProcessor` respiration-rate estimate
  - heart candidate confidence
  - Kalman-gated tracked heart estimate, shown as `acquiring` until stable enough
- IMU:
  - sample rate
  - respiration-band estimate
  - heart-band estimate
  - accelerometer RMS
  - gyroscope RMS

### Recordings tab

The `Recordings` tab can open:

- saved CSV files from `data/raw/radar/`, `data/raw/a121/`, and `data/raw/imu/`
- sessions stored in SQLite

SQLite database path priority:

1. `RESPI_RECORDINGS_DB_PATH` (or `RESPI_DB_PATH`) if set.
2. `E:/respi_recordings.sqlite3` on Windows, when that file exists.
3. `data/respi_recordings.sqlite3` in the project folder as the macOS/Linux-safe fallback.

The project-local fallback database is ignored by Git.

## CLI commands

Show all commands:

```powershell
uv run respi --help
```

List serial ports:

```powershell
uv run respi ports
```

Open the unified app:

```powershell
uv run respi app --sensor radar --port COM6
uv run respi app --sensor a121 --port COM3
uv run respi app --sensor imu --port COM6
```

Test the A121 without opening the UI:

```powershell
uv run respi test-a121 --port COM3 --frames 20 --start-m 0.2 --end-m 1.0 --profile 3 --hwaas 32 --sweeps-per-frame 16 --frame-rate-hz 20
```

Record a real-data A121 test CSV. Omit `--seconds` to record until Ctrl+C; if supplied, `--seconds` has no upper cap:

```powershell
uv run respi record-a121-test --port COM3 --label foil-chest
uv run respi record-a121-test --port COM3 --seconds 60 --label foil-chest
# Configure A121 range/profile/HWAAS/sweeps/frame-rate explicitly:
uv run respi record-a121-test --port COM3 --seconds 80 --label 80hr --start-m 0.20 --end-m 1.50 --profile 3 --hwaas 32 --sweeps-per-frame 16 --frame-rate-hz 20
# Automatic breath phase annotations, avoiding key-press vibration artifacts:
uv run respi record-a121-test --port COM3 --seconds 80 --label auto-breath --auto-breaths
# Manual breath phase markers are still available: press i for inhale start/end, o for exhale start/end.
uv run respi record-a121-test --port COM3 --seconds 80 --label breath-marked --mark-breaths --inhale-key i --exhale-key o
# Generate automatic breath annotations for an existing A121 CSV:
uv run respi annotate-a121-breaths data/raw/a121/a121_test_80s_auto-breath_YYYY-MM-DD_HH-MM-SS.csv
# Alias:
uv run respi capture-a121 --port COM3 --label foil-chest
uv run respi capture-a121 --port COM3 --seconds 60 --label foil-chest
```

This saves a new non-overwriting file under `data/raw/a121/`, for example `a121_test_60s_foil-chest_YYYY-MM-DD_HH-MM-SS.csv`. `--auto-breaths` or `annotate-a121-breaths` saves a sidecar `*_breath_annotations.csv` in the same five-column format as manual markers: `Timestamp_ms,Elapsed_s,Frame,Event,Key`. Events are `inhale_start`, `inhale_end`, `exhale_start`, and `exhale_end`; automatic rows use `Key=auto`. Manual `--mark-breaths` is still supported, but it can shake the sensor if key presses are hard.

Record an overnight/sleep A121 CSV. The sleep command streams rows directly to disk instead of keeping the full night in RAM. Omit `--hours` to record until you manually stop with Ctrl+C; include it only as a safety limit. By default, stopping the sleep recording also runs gated vitals extraction, classifies radar sleep phases, computes a Garmin-like radar sleep score, and writes a plot plus CSV/JSON sidecars under `data/plots/`.

```powershell
uv run respi record-a121-sleep --port COM3 --label sleep-night-1
uv run respi record-a121-sleep --port COM3 --label sleep-night-1 --hours 8
# Disable automatic post-processing:
uv run respi record-a121-sleep --port COM3 --label sleep-night-1 --no-auto-analyze
# Keep automatic vitals analysis but skip the sleep/comparison plot:
uv run respi record-a121-sleep --port COM3 --label sleep-night-1 --no-auto-plot
# Include Garmin FIT overlay after recording if you already have exported FIT files:
uv run respi record-a121-sleep --port COM3 --label sleep-night-1 --garmin-fit path\to\WELLNESS.fit --garmin-fit path\to\SLEEP_DATA.fit
```

Offline A121 sleep analysis and Garmin FIT comparison can be run later on existing files:

```powershell
# Full raw A121 CSV -> gated HR/RR trend -> sleep phases/score/plot
uv run respi analyze-a121-sleep data\raw\a121\a121_sleep_YYYY-MM-DD_HH-MM-SS.csv
# Existing gated trend CSV -> radar-only phase/score plot
uv run respi plot-a121-sleep data\plots\a121_sleep_YYYY-MM-DD_HH-MM-SS_gated_sleep_vitals.csv
# Existing trend + one Garmin FIT file or a directory of FIT files; timestamps are aligned in local time
uv run respi plot-a121-sleep data\plots\a121_sleep_YYYY-MM-DD_HH-MM-SS_gated_sleep_vitals.csv --garmin-fit path\to\GarminExportDir
uv run respi plot-a121-sleep data\plots\a121_sleep_YYYY-MM-DD_HH-MM-SS_gated_sleep_vitals.csv --garmin-fit path\to\WELLNESS.fit --garmin-fit path\to\SLEEP_DATA.fit
# Vitals-only plot with no phase/score panel
uv run respi plot-a121-sleep data\plots\a121_sleep_YYYY-MM-DD_HH-MM-SS_gated_sleep_vitals.csv --no-sleep-phases
# Decode Garmin FIT tables only
uv run respi decode-garmin-fit path\to\WELLNESS.fit path\to\SLEEP_DATA.fit --stem my_sleep
```

Reusable code/scripts:

- Garmin FIT decoding: `src/respi_net/garmin_fit.py`, CLI `respi decode-garmin-fit`
- Radar sleep phases/scoring/plotting: `src/respi_net/a121_sleep.py`, CLI `respi plot-a121-sleep`
- Raw overnight A121 gated vitals extraction: `tools/analyze_a121_sleep_vitals_gated.py`, CLI `respi analyze-a121-sleep`
- Direct repository wrapper: `tools/plot_a121_garmin_sleep_overlay.py`

Expected output looks like:

```text
A121 session started: 0.200-1.010 m, 55 points, profile 3, HWAAS 32.
Frame 000 | peak=0.458 m | amp=192.3 | phase=2.10 rad
```

Capture HB100 radar from serial until Ctrl+C:

```powershell
uv run respi capture-radar --port COM6
```

Capture IMU from serial until Ctrl+C:

```powershell
uv run respi capture-imu --port COM6
```

Scan for and capture iPhone IMU data over BLE:

```powershell
uv run respi iphone-imu-devices
uv run respi capture-iphone-imu --seconds 60
uv run python tools/imu_lsm6ds3_iphone_guided_experiment.py --lsm-port /dev/cu.usbserial-XXXX
```

`imu_lsm6ds3_iphone_guided_experiment.py` is a separate guided protocol for
the four supine LSM6DS3--iPhone trials. It stores both sensor files, shared
cue timestamps, loss diagnostics and a manifest; Esc deletes the active
trial, while a completed one can be discarded and repeated before acceptance.

Generate an offline HB100 radar plot:

```powershell
uv run respi plot-radar data\raw\radar\radar_raw_YYYY-MM-DD_HH-MM-SS.csv
```

Generate an offline IMU plot:

```powershell
uv run respi plot-imu data\raw\imu\respiratory_6axis_raw_YYYY-MM-DD_HH-MM-SS.csv
```

Batch-generate plots:

```powershell
uv run respi batch-radar
uv run respi batch-imu
```

Compatibility alias:

```powershell
uv run respi live-radar --port COM6
```

`live-radar` now opens the unified app in HB100 radar mode.

## A121 implementation notes

The A121 capture code uses Acconeer's low-level client:

```python
from acconeer.exptool import a121
client = a121.Client.open(serial_port="COM3")
```

Internally, the app creates an `a121.SensorConfig` using approximate distance-to-point conversion:

```text
start_point ~= start_m / 0.0025
num_points  ~= (end_m - start_m) / (0.0025 * step_length)
```

After `setup_session`, the actual physical distance bins are computed from Acconeer metadata via `get_distances_m(...)`.

For every frame:

- raw complex Sparse IQ frame is read
- amplitude is computed with `np.abs(...)`
- phase is computed with `np.angle(...)`
- strongest amplitude bin becomes `PeakDistance_m`
- full arrays are saved as JSON strings in CSV/SQLite
- Acconeer reference-app state, presence distance, selected compact range, and breathing rate are saved per frame

For live A121 respiration, target acquisition/reacquisition follows Acconeer's breathing reference app state machine and presence-distance selection. New CSV/SQLite recordings persist that Acconeer selection, so history/offline analysis reuses the same selected gate instead of rediscovering a weaker host-side gate from averaged IQ. The `Offline/fallback gate` is now only for disabled/unavailable Acconeer selection or very old/core-only files. Heart-rate extraction is not provided by Acconeer here; the app treats it as an experimental, conservatively gated candidate. The A121 result buffer is limited to about `num_points * sweeps_per_frame <= 4095`; the app automatically reduces sweeps/frame or increases step length when a requested range would exceed sensor/serial limits.

## Troubleshooting

### A121 does not connect

1. Run:

   ```powershell
   uv run respi ports
   ```

2. Pick the `USB-Enhanced-SERIAL-A CH342` port, not `SERIAL-B`.
3. Close Acconeer GUI or any other program that may be using the port.
4. Try the CLI test:

   ```powershell
   uv run respi test-a121 --port COM3 --frames 5
   ```

### No ESP32 data appears

- Check the correct COM port.
- Check firmware matches selected app sensor.
- Confirm the sensor-specific baud rate: HB100 defaults to `230400`, while the
  ESP32 IMU remains at `921600`.
- Use `uv run respi ports` to verify the port is visible.

### Empty or noisy plots

- For A121, narrow the range with `A121 start` / `A121 end`.
- Increase A121 HWAAS to reduce noise.
- Try A121 profile 3 first, then adjust profiles for closer/farther targets.
- For HB100, check amplifier clipping and ADC wiring.

## Source files

Main implementation files:

```text
src/respi_net/app.py      # Unified Qt/pyqtgraph app
src/respi_net/a121.py     # Acconeer A121 Sparse IQ capture
src/respi_net/radar.py    # HB100 radar analysis/capture
src/respi_net/imu.py      # IMU analysis/capture
src/respi_net/iphone_imu.py # iPhone BLE IMU batch receiver
src/respi_net/cli.py      # Click CLI commands
src/respi_net/paths.py    # Data/output paths
```
