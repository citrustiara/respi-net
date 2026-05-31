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

Use this for the existing analog HB100 radar pipeline. The ESP32 streams CSV-like serial rows:

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

Use this for the ESP32/LSM6DS3 comparison path. The ESP32 streams 6-axis rows and the app adds host timestamps:

```text
Time_ms,ax,ay,az,gx,gy,gz
```

Default output folder:

```text
data/raw/imu/
```

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
```

Available app sensor choices:

- `HB100 Radar`
- `A121 Radar`
- `IMU`

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

- `A121 start` - start distance in meters, e.g. `0.20 m`
- `A121 end` - end distance in meters, e.g. `1.50 m`
- `A121 profile` - Acconeer profile `1..5`; profile 3 is a good general default
- `A121 HWAAS` - hardware averaging; higher values reduce noise but reduce update rate

### App graphs

Graphs are rendered with `pyqtgraph` and are interactive:

- drag to pan
- mouse wheel to zoom
- right-click for plot options

Live graph behavior:

- HB100: voltage over time + FFT
- A121: latest amplitude vs distance + peak distance over time
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
  - multi-scattering-point (MSP) bin count and signal-quality index
  - circle-centered respiration-band estimate
  - harmonic-suppressed raw heart estimate
  - Kalman-gated tracked heart estimate
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

SQLite database path:

```text
data/respi_recordings.sqlite3
```

This file is ignored by Git.

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
uv run respi test-a121 --port COM3 --frames 20 --start-m 0.2 --end-m 1.0 --profile 2 --hwaas 64 --sweeps-per-frame 12 --frame-rate-hz 50
```

Expected output looks like:

```text
A121 session started: 0.200-1.501 m, 521 points, profile 2, HWAAS 64.
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

For vital signs, the analyzer then applies algebraic IQ circle centering, MSP multi-bin coherent combining inside the selected gate, respiration harmonic subtraction, and an optional live Kalman prior for the cardiac search band. The A121 result buffer is limited to about `num_points * sweeps_per_frame <= 4095`; the app automatically reduces sweeps/frame when a requested range would exceed that limit.

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
- Confirm baud rate, default `921600`.
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
src/respi_net/cli.py      # Click CLI commands
src/respi_net/paths.py    # Data/output paths
```
