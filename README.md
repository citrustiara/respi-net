# Neural Networks for Respiratory Analysis

This repository contains the source code for my Bachelor's Thesis (Praca Inżynierska). The project focuses on non-contact respiratory sensing with an HB100 microwave radar module, using an ESP32-based acquisition pipeline and Python tools for recording, visualization, and signal analysis.

## Project Overview

The main objective is to evaluate how effectively low-cost Doppler radar can capture breathing-related motion and produce clean respiratory signals for later machine-learning experiments. The current work is centered on reliable radar data acquisition, analog signal conditioning, repeatable CSV recordings, and fast chart generation for inspection.

The IMU path is still present as a secondary comparison channel, but the main experimental direction is now the radar pipeline.

Current Progress:
- **HB100 10.525 GHz analog radar integrated** with custom amplification and filtering.
- ESP32 ADC firmware captures raw radar voltage at high serial throughput.
- Click CLI supports live radar viewing, serial capture, batch processing, and chart generation.
- Radar CSV recordings are organized under `data/raw/radar/`.
- Radar plots are generated under `outputs/plots/radar/`.
- IMU capture and analysis remain available for comparison experiments.
- Hardware schematics are in progress.

## Hardware Setup

- Microcontroller: ESP32 (LilyGO T-Display)
- Main sensor: **HB100 10.525 GHz Microwave Motion Sensor**
- Signal conditioning: custom active filter and amplifier stage
- ADC input: ESP32 ADC, using the conditioned radar IF signal
- Optional comparison sensor: LSM6DS3 accelerometer + gyroscope

Radar connections:
- VCC -> 5V
- GND -> GND
- IF (Signal) -> Amplifier Input -> ESP32 ADC (GPIO 34 or similar)

Optional IMU connections:
- 3V3 -> 3V
- GND -> GND
- GPIO 22 -> SCL
- GPIO 21 -> SDA

## Signal Processing Pipeline

### Radar Pipeline (HB100)

- High-speed acquisition: captures raw ADC and voltage samples from the ESP32.
- Live visualization: plots recent voltage samples and a live FFT during recording.
- Offline analysis: generates time-domain and Welch PSD charts from saved CSV files.
- Doppler helper axis: maps frequency to estimated speed for quick movement interpretation.
- Batch processing: regenerates plots for all radar recordings in one command.

### IMU Pipeline (Secondary)

- Orientation-independent analysis with PCA over accelerometer axes.
- Butterworth filtering for respiratory and cardiac bands.
- Plot generation for comparison with radar recordings.

## Usage

This project uses `uv` for Python dependency and command management.

```powershell
uv sync
uv run respi --help
```

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
uv run respi capture-radar --port COM6
uv run respi live-radar --port COM6
```

IMU comparison commands are still available when needed:

```powershell
uv run respi plot-imu data\raw\imu\respiratory_6axis_raw_2026-03-08_02-37-19.csv
uv run respi batch-imu
uv run respi capture-imu --port COM6
```

## Repository Structure

- `src/respi_net/` - Python package for capture, analysis, plotting, and CLI commands.
- `data/raw/radar/` - Radar CSV recordings.
- `outputs/plots/radar/` - Generated radar charts.
- `firmware/esp32_radar_adc/` - ESP32 firmware for high-speed radar data acquisition.
- `hardware/` - KiCad hardware design files including the radar amplifier.
- `data/raw/imu/` - Optional IMU comparison recordings.
- `outputs/plots/imu/` - Generated IMU comparison charts.
- `firmware/esp32_imu_stream/` - Optional ESP32 firmware for IMU streaming.
- `docs/` - Reports, notes, and logs.
- `tools/docx_generator/` - Legacy Node-based report generator.

## Future Plans

1. Fine-tune the radar amplifier and filtering stage to prevent clipping during deep breaths.
2. Collect a broader radar dataset across distances, body positions, and breathing patterns.
3. Compare selected radar recordings against IMU reference measurements.
4. Design and train neural network models to classify respiratory patterns and detect anomalies.
5. Explore edge inference on ESP32 after the radar signal pipeline is stable.

License
This project is created as part of a Bachelor's Thesis. All rights reserved.
