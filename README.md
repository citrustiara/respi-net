# Neural Networks for Respiratory Analysis

This repository contains the source code for my Bachelor's Thesis (Praca Inżynierska). The project explores using Neural Networks and IMU sensors (accelerometer and gyroscope) to detect and analyze respiratory patterns in real-time.

## Project Overview

The main objective is to see how effectively we can use a standard IMU sensor (LSM6DS3) to measure breathing and potentially heart rate. While the ultimate goal involves machine learning for classification, the current focus is on reliable data acquisition and signal processing to extract clear respiratory signals.

Current Progress:
- ESP32 (LilyGO T-Display) integrated with LSM6DS3 IMU.
- Reliable streaming of 6-axis raw data (accelerometer and gyroscope).
- Advanced signal processing implemented, including PCA (Principal Component Analysis) to ensure measurements are orientation-independent.
- Basic peak detection successfully counts breaths.
- Heart rate signal is visible and somewhat reliable depending on placement.
- Hardware schematics in progress.

## Hardware Setup

- Microcontroller: ESP32 (LilyGO T-Display)
- Sensor: LSM6DS3 (Accelerometer + Gyroscope, medium-end sensor, comparable or slightly better than standard smartphones)
- Radar: Pending (HB100 analog radar available, awaiting parts to build an amplifier)

Connections:
- 3V3 -> 3V
- GND -> GND
- GPIO 22 -> SCL
- GPIO 21 -> SDA

## Signal Processing Pipeline

The project currently uses a robust filtering pipeline to extract clean respiratory and cardiac data from noisy raw IMU signals:

1. Orientation Independence (PCA): Instead of relying on a single axis (which changes when the user moves), Principal Component Analysis extracts the most dominant moving vector across all 3 axes of both the accelerometer and the calculated 3D tilt angle.
2. Frequency Filtering (Butterworth): 
   - Respiratory Band: A bandpass filter (typically 0.1 - 0.5 Hz) isolates the chest movement frequencies associated with breathing, removing high-frequency noise and sudden movements.
   - Cardiac Band: A separate bandpass filter (typically 0.8 - 4.0 Hz) isolates the subtle, faster vibrations of the heart beating from the accelerometer data.
3. Peak Detection: `scipy.signal.find_peaks` is used on the smoothed, isolated signals to accurately timestamp individual breaths and heartbeats.

## Repository Structure

- `script.py` - Main Python script for data acquisition, PCA signal processing, and real-time visualization.
- `esp32_imu_stream/` - ESP-IDF C++ firmware for the ESP32.
- `schematics/` - Hardware design files.

## Future Plans

1. Collect a comprehensive dataset of breathing patterns.
2. Build and integrate the amplifier for the HB100 radar to compare its performance against the IMU.
3. Design and train Neural Network models (CNN/LSTM) to classify the collected data.
4. Implement the trained model directly on the ESP32 for edge inference.

License
This project is created as part of a Bachelor's Thesis. All rights reserved.
