# respi-net

Contactless breathing measurement with a 60 GHz Acconeer A121 radar, and the groundwork for a causal neural network
that recognises breathing phases (inhale, exhale, hold, noise) from the radar alone. Bachelor's thesis project
(praca inżynierska).

The project started with an HB100 Doppler module and IMUs (ESP32 + LSM6DS3). Those were a proof of concept; the
current work is A121-only.

## What is here

- **A121 signal chain** (`src/respi_net/chest_signal.py`): Sparse IQ → presence and chest range gates (Acconeer's
  breathing reference app) → unwrapped phase per gate → echo-weighted average → chest motion in mm (inhale up) →
  driver clock drift correction (0.17 %) → a 2 Hz low-pass as the only filter, so breath holds stay flat.
- **Desktop app** (`respi app`): live A121 view of the signal the network gets (range profile, chest motion and a
  small velocity strip, echo strength in the stats; ~0.2–0.3 s lag), CSV/SQLite recording, and a Recordings tab with curated demo recordings
  (`configs/demo_recordings.json`). Inhale/exhale shading and the older band-pass breathing/heart view are optional,
  under the *View* menu. Also supports HB100, the ESP32 IMU and the iPhone app.
- **iPhone app** (`ios/RespiPhoneIMU/`): streams CoreMotion accelerometer/gyro over BLE, saves every trial on the
  phone as well, remote start/stop from the desktop. Used as a timing helper for labels, never as a network input.
- **Breathing coach** (`respi coach`): the whole session drawn as the breath itself, countdown and next cue; JSON
  patterns in `configs/breathing_patterns/`. Guided recorders log cue times next to the radar and phone data.
- **Breathing-phase data**: classes after Szymański et al. (Sci Data 2025): 0 exhale, 1 hold after exhale,
  2 inhale, 3 hold after inhale, 4 noise, −1 ignored. Per-breath label correction, hand-annotation viewer, a loader
  for the Szymański respiratory-belt dataset (with its 0.2 s label offset corrected), augmented copies and a
  synthetic breathing generator passed through an A121 radar model.
- **Phase models** (`src/respi_net/phase_model.py`): causal TCN (~44k parameters, 25.5 s receptive field) and
  causal GRU (~38k), Viterbi / streaming decoding with phase-transition rules, and a deterministic baseline.
  Nothing has been trained yet; the target recordings are still being collected.
- **Earlier experiments**: lens comparison (hyperbolic, Fresnel zone plate, flat cover; aluminium foil gives no
  gain), sleep nights against a Garmin watch, heart rate from cardiac motion, HB100 vs A121.

## Quick start

```bash
uv sync
uv run respi app --sensor a121                      # live radar (serial port is auto-detected)
uv run respi app --sensor a121 --open data/raw/...csv   # open a recording, no hardware needed
uv run respi coach paced_12_hold                    # breathing coach
uv run respi --help                                 # all commands
uv run pytest -q
```

On macOS the Waveshare A121 board shows up as two ports, `/dev/cu.usbmodem…1` (interface A, the one to use) and
`…3`. Suggested app settings at 0.5–1.5 m: start 0.30 m, end 1.50 m, profile 3, HWAAS 32, 16 sweeps, 20 Hz,
live window 30 s.

Other useful commands:

```bash
uv run respi record-a121-test -p /dev/cu.usbmodemXXXX1 --label my-run    # record A121 until Ctrl+C
uv run respi record-a121-sleep -p /dev/cu.usbmodemXXXX1 --label night-1  # overnight recording + sleep analysis
uv run respi capture-iphone-imu --seconds 60        # iPhone IMU over BLE
uv run python tools/a121_iphone_guided_recording.py # A121 + iPhone with the coach
uv run python tools/annotate_phone_phases.py --session self_lying_3min_01 --run run_01_nn_self_3min --start 40 --length 60
uv run python tools/build_artificial_data.py        # augmented + synthetic training sets
uv run python tools/train_breath_phase_model.py --help
```

Full app and CLI reference: [`docs/APP_AND_CLI.md`](docs/APP_AND_CLI.md).

## Hardware

- Waveshare Acconeer A121 module (60 GHz pulsed coherent radar, USB), with a hyperbolic lens.
- iPhone with the RespiPhoneIMU app (labelling helper).
- Legacy: HB100 10.525 GHz module with a two-stage MCP6002 front end on an ESP32
  ([schematic](hardware/hb100_calibrated_schematic.svg)), LSM6DS3 IMU, AD8232 ECG for future heart-rate labels.

## Layout

- `src/respi_net/` – capture, signal processing, app, coach, datasets, models.
- `tools/` – guided recorders, analyses, dataset builders, training and figure scripts.
- `configs/` – breathing patterns, experiment configs, demo recordings list.
- `annotations/` – reviewed radar labels and small generated example datasets.
- `docs/thesis/` – thesis source (`praca_inzynierska.tex`) and figures; `docs/datasets/` – dataset notes.
- `ios/RespiPhoneIMU/` – iPhone app; `firmware/` – ESP32 firmware for the legacy sensors; `hardware/` – schematics.
- `data/` – recordings (large CSVs and third-party datasets are not committed).

## License

MIT, see [LICENSE](LICENSE). The Szymański et al. dataset is CC BY-NC-ND and is not redistributed here.
