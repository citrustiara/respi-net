# Neural Networks for Respiratory Analysis

This repository contains the source code for my Bachelor's Thesis (Praca Inżynierska). The project focuses on non-contact respiratory sensing with an HB100 microwave radar module, using an ESP32-based acquisition pipeline and Python tools for recording, visualization, and signal analysis.

## Project Overview

The main objective is to evaluate how effectively low-cost Doppler radar can capture breathing-related motion and produce clean respiratory signals for later machine-learning experiments. The current work is centered on reliable radar data acquisition, analog signal conditioning, repeatable CSV recordings, and fast chart generation for inspection.

The IMU path is still present as a secondary comparison channel, but the main experimental direction is now the radar pipeline.

Current Progress:
- **HB100 10.525 GHz analog radar integrated** with custom amplification and filtering.
- ESP32 ADC firmware captures raw radar voltage at high serial throughput.
- Click CLI supports serial capture, batch processing, and chart generation.
- Unified Qt desktop app (`respi app`) combines HB100 radar, Acconeer A121 radar, and IMU live viewing, controls, history browsing, CSV recording, and SQLite recording.
- Radar CSV recordings are organized under `data/raw/radar/` and A121 Sparse IQ recordings under `data/raw/a121/`.
- Radar plots are generated under `outputs/plots/radar/`.
- IMU capture and analysis remain available for comparison experiments.
- Calibrated HB100 analog frontend schematic is documented and rendered in [`hardware/hb100_calibrated_schematic.svg`](hardware/hb100_calibrated_schematic.svg).

## Hardware Setup

- Microcontroller: ESP32 (LilyGO T-Display)
- Main sensor: **HB100 10.525 GHz Microwave Motion Sensor**
- Optional 60 GHz range radar: **Waveshare / Acconeer A121 Range Sensor** over USB-UART CH342 Interface A
- Signal conditioning: custom active filter and amplifier stage
- ADC input: ESP32 GPIO 33 / ADC1_CH5, using the conditioned radar IF signal
- Optional comparison sensor: LSM6DS3 accelerometer + gyroscope

Radar connections:
- VCC -> 5V
- GND -> GND
- IF (Signal) -> calibrated two-stage MCP6002 amplifier/filter -> ESP32 GPIO 33 (ADC1_CH5)

### Calibrated HB100 Analog Frontend

![Rendered HB100 calibrated schematic](hardware/hb100_calibrated_schematic.svg)

The current radar frontend uses two symmetrical non-inverting MCP6002 stages. Each stage has `Rf = 100 kΩ`, `Rg = 10 kΩ`, and a `22 nF` (`"223"`) capacitor across the feedback loop, giving about `11×` per stage (`~121×` total). The AC-coupled inputs are restored to the `1.65 V` virtual ground through `1 MΩ` bias resistors, and the output is protected by `1 kΩ` before ESP32 GPIO 33.

Optional IMU connections:
- 3V3 -> 3V
- GND -> GND
- GPIO 22 -> SCL
- GPIO 21 -> SDA

## Signal Processing Pipeline

### Radar Pipeline (HB100)

- High-speed acquisition: captures raw ADC and voltage samples from the ESP32.
- Live visualization: the desktop app plots recent voltage samples and a live FFT during recording with interactive pan/zoom via pyqtgraph.
- Offline analysis: generates time-domain and Welch PSD charts from saved CSV files.
- Doppler helper axis: maps frequency to estimated speed for quick movement interpretation.
- Batch processing: regenerates plots for all radar recordings in one command.

### Acconeer A121 Pipeline

- Uses `acconeer-exptool` Low-Level Client API over the CH342 Interface A serial port.
- Captures Sparse IQ frames, storing per-frame distance bins, amplitude, phase, real, and imaginary arrays.
- UI controls distance focus (`start_m`, `end_m`), profile, HWAAS, sweeps/frame, and frame rate.
- Defaults follow Acconeer's breathing reference app for stable live operation (profile 3, HWAAS 32, 16 sweeps/frame, 20 Hz) and auto-clamp sweeps/frame to the A121 4095-sample buffer/serial limits for wide ranges.
- Live visualization plots latest amplitude vs distance, Acconeer-selected target range, respiration band, heart band, rate FFTs, and raw IQ. Live time-domain traces are append-only causal filters so old samples do not change shape while the plot scrolls.
- A121 target acquisition now follows Acconeer's breathing reference app state machine/presence-distance selection, and respiratory-rate extraction runs Acconeer's own `BreathingProcessor` on that compact range segment; heart-rate display remains experimental and conservatively gated.

### IMU Pipeline (Secondary)

- Orientation-independent analysis with PCA over accelerometer axes.
- Butterworth filtering for respiratory and cardiac bands.
- Plot generation for comparison with radar recordings.

## Usage

Full app/CLI documentation is in [`docs/APP_AND_CLI.md`](docs/APP_AND_CLI.md).

This project uses `uv` for Python dependency and command management.

```powershell
uv sync
uv run respi --help
```

Open the unified desktop app:

```powershell
uv run respi app --sensor radar --port COM6   # HB100/ESP32 ADC
uv run respi app --sensor a121 --port COM3    # Acconeer A121/Waveshare CH342 Interface A
```

The app has HB100 Radar/A121 Radar/IMU sensor selection, serial port controls, Start/Stop buttons, live stats, interactive graphs (drag to pan, mouse wheel to zoom, right-click plot menu), and a Recordings tab for opening saved CSV files or SQLite sessions. Live sessions can be recorded to CSV, SQLite, or both. The SQLite database is stored at `data/respi_recordings.sqlite3`.

Generate a chart from one file:

```powershell
uv run respi plot-radar data\raw\radar\radar_raw_2026-03-13_18-30-09.csv
```

Batch-generate radar charts:

```powershell
uv run respi batch-radar
```

Capture or view radar data from serial:

```powershell
uv run respi ports
uv run respi app --sensor radar --port COM6
uv run respi app --sensor a121 --port COM3
uv run respi test-a121 --port COM3 --frames 20 --start-m 0.2 --end-m 1.5
uv run respi capture-radar --port COM6
uv run respi live-radar --port COM6  # compatibility alias for the unified app
```

IMU comparison commands are still available when needed:

```powershell
uv run respi plot-imu data\raw\imu\respiratory_6axis_raw_2026-03-08_02-37-19.csv
uv run respi batch-imu
uv run respi capture-imu --port COM6
```

## Repository Structure

- `src/respi_net/` - Python package for capture, analysis, plotting, and CLI commands.
- `data/raw/radar/` - HB100 radar CSV recordings.
- `data/raw/a121/` - Acconeer A121 Sparse IQ CSV recordings.
- `outputs/plots/radar/` - Generated radar charts.
- `firmware/esp32_radar_adc/` - ESP32 firmware for high-speed radar data acquisition.
- `hardware/` - Hardware design files, including the calibrated HB100 amplifier schematic source and rendered SVG.
- `data/raw/imu/` - Optional IMU comparison recordings.
- `outputs/plots/imu/` - Generated IMU comparison charts.
- `firmware/esp32_imu_stream/` - Optional ESP32 firmware for IMU streaming.
- `docs/` - Reports, notes, and logs.
- `tools/docx_generator/` - Legacy Node-based report generator.

## Future Plans

1. Validate the calibrated low-gain radar amplifier across distances, body positions, and movement intensity.
2. Collect a broader radar dataset across distances, body positions, and breathing patterns.
3. Compare selected radar recordings against IMU reference measurements.
4. Design and train neural network models to classify respiratory patterns and detect anomalies.
5. Explore edge inference on ESP32 after the radar signal pipeline is stable.

License
This project is created as part of a Bachelor's Thesis. All rights reserved.
