# Respiratory Analysis AI (Draft)

![Status](https://img.shields.io/badge/Status-Work_in_Progress-orange)
![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%7C%20LSM6DS3-blue)
![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green)

This repository contains the source code and hardware design for my Bachelor's Thesis (Praca Inżynierska). The ultimate goal of this project is to develop a system for **respiratory and physiological analysis using Neural Networks** based on IMU (Inertial Measurement Unit) data.

## 📌 Project Overview
The project explores the viability of using a simple, non-invasive IMU sensor (accelerometer and gyroscope) to detect, measure, and analyze respiratory patterns (and potentially heart rate) in real-time. The collected signals will eventually be processed by machine learning models to classify breathing patterns or detect anomalies.

**Current State:**
- ✅ ESP32 (LilyGO T-Display) integrated with LSM6DS3 IMU.
- ✅ Data streaming (6-axis raw data: G-force and tilt/angle).
- ✅ Signal processing using Principal Component Analysis (PCA) to make measurements orientation-independent.
- ✅ Basic peak detection and breath counting.
- 🚧 Hardware schematic design is in progress (KiCad).
- ⏳ Neural Network implementation (Planned).
- ⏳ Additional sensors integration (Planned).

## 🛠️ Hardware Setup
- **Microcontroller:** ESP32 (LilyGO T-Display)
- **Sensor:** LSM6DS3 (Accelerometer + Gyroscope)
- **Connections:**
  - `3V3` -> `3V`
  - `GND` -> `GND`
  - `GPIO 22` -> `SCL`
  - `GPIO 21` -> `SDA`

Hardware schematics and PCB designs will be available in the `schematics/` directory once finalized.

## 📂 Repository Structure
* `script.py` - Main Python script for data acquisition, PCA signal processing, and real-time visualization.
* `esp32_imu_stream/` - ESP-IDF C++ firmware for the ESP32 to read the IMU and stream data.
* `schematics/` - KiCad project files for the hardware design.
* `*.csv` - Raw captured dataset files (ignored in git for size).

## 🚀 Future Roadmap
1. Finalize hardware schematic and PCB layout.
2. Collect an extensive labeled dataset of various breathing patterns.
3. Design and train a Neural Network model (e.g., CNN or LSTM) for respiratory analysis.
4. Port the trained AI model to the ESP32 for edge edge computing/inference (TinyML).

## 📄 License
This project is created as part of a Bachelor's Thesis. All rights reserved.
