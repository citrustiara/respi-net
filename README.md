# Neural Networks for Respiratory Analysis

This repository contains the source code for my Bachelor's Thesis (Praca Inżynierska). The project explores using Neural Networks and IMU sensors (accelerometer and gyroscope) to detect and analyze respiratory patterns in real-time.

## Project Overview

The main objective is to see how effectively we can use a standard IMU sensor (LSM6DS3) to measure breathing and potentially heart rate. While the ultimate goal involves machine learning for classification, the current focus is on reliable data acquisition and signal processing to extract clear respiratory signals.

Current Progress:
- ESP32 (LilyGO T-Display) integrated with LSM6DS3 IMU.
- Reliable streaming of 6-axis raw data (accelerometer and gyroscope).
- Advanced signal processing implemented, including PCA (Principal Component Analysis) to ensure measurements are orientation-independent.
- **HB100 Analog Radar integrated** with custom amplification and filtering.
- **`radar_viewer.py` developed** for live raw voltage visualization and high-speed data recording (921,600 baud).
- Basic peak detection successfully counts breaths.
- Heart rate signal is visible and somewhat reliable depending on placement.
- Hardware schematics in progress

## Hardware Setup

- Microcontroller: ESP32 (LilyGO T-Display)
- Sensor 1: LSM6DS3 (Accelerometer + Gyroscope)
- Sensor 2: **HB100 10.525GHz Microwave Motion Sensor** (Radar)
- Radar Amplifier: Custom active filter and amplifier stage for signal conditioning.

Connections (IMU):
- 3V3 -> 3V
- GND -> GND
- GPIO 22 -> SCL
- GPIO 21 -> SDA

Connections (Radar):
- VCC -> 5V
- GND -> GND
- IF (Signal) -> Amplifier Input -> ESP32 ADC (GPIO 34 or similar)

## Signal Processing Pipeline

The project uses two primary data paths:

### 1. IMU Pipeline (LSM6DS3)
- Orientation Independence (PCA): Extracts dominant moving vectors across 3 axes.
- Frequency Filtering (Butterworth): Isolates Respiratory (0.1 - 0.5 Hz) and Cardiac (0.8 - 4.0 Hz) bands.
- Peak Detection: Counts breaths and heartbeats from smoothed data.

### 2. Radar Pipeline (HB100)
- High-Speed Acquisition: Captures raw voltage via 12-bit ADC at high sampling rates.
- Real-time Visualization: Live plotting of raw voltage for signal quality monitoring.
- Post-Processing: FFT-based analysis and bandpass filtering (0.05 - 4.0 Hz) for BPM estimation.

## Repository Structure

- `script.py` - Python script for IMU data acquisition and PCA processing.
- `radar_viewer.py` - Real-time visualization and recording tool for HB100 radar.
- `esp32_imu_stream/` - ESP32 firmware for IMU streaming.
- `esp32_radar_adc/` - ESP32 firmware for high-speed radar data acquisition.
- `schematics/` - KiCad hardware design files including the radar amplifier.

## Future Plans

1. Collect a comprehensive dataset comparing IMU and Radar signals simultaneously.
2. Fine-tune hardware amplification to prevent clipping at deep breaths.
3. Design and train Neural Network models (CNN/LSTM) to classify respiratory patterns and detect anomalies.
4. Implement edge inference on ESP32 using TensorFlow Lite for Microcontrollers.

License
This project is created as part of a Bachelor's Thesis. All rights reserved.
