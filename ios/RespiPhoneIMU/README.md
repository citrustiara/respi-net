# RespiPhoneIMU

Native SwiftUI iPhone companion app for streaming phone IMU data into RespiNet over Bluetooth Low Energy.

## Install on an iPhone

You need a Mac with Xcode for the final build/sign/install step. Apple does not ship the iOS SDK, device installer, or signing toolchain for Windows.

1. Copy or clone this repository on a Mac.
2. Install Xcode from the Mac App Store, then open Xcode once so it finishes installing components.
3. Connect the iPhone with USB, unlock it, and trust the Mac when prompted.
4. Open `ios/RespiPhoneIMU/RespiPhoneIMU.xcodeproj` in Xcode.
5. Select the `RespiPhoneIMU` target, then `Signing & Capabilities`.
6. Choose your Apple ID team. A free Apple ID is usually enough for direct development installs, though the app may expire and need reinstalling.
7. Select your physical iPhone as the run destination.
8. Press Run in Xcode.
9. If iOS asks for Developer Mode, enable it in Settings, restart the phone if prompted, then run again.

The simulator cannot provide real Bluetooth peripheral advertising or useful motion data, so use a physical iPhone. (In the simulator the app draws a made-up breath so its screens can be checked.)

## Using it for recordings

The app is meant to be strapped to the body, screen outwards, while the desktop recorder runs:

- **Before a trial** the main screen shows a checklist (Bluetooth, visible to the Mac, Mac connected, battery
  at least 30%, Low Power Mode off) and a **placement check**: a live line of the chest's tilt over the last
  30 s and its size in mg. With the strap tight, it rises and falls smoothly with each breath.
- **While streaming** a dark full-screen view shows the elapsed time, whether the Mac is connected, the live
  line and the measured rate, readable from a few metres. The screen dims (optional). Nothing on it reacts to
  a quick touch: stopping from the phone takes a 2-second hold, so a strap or shirt cannot stop a trial. The
  recorder normally starts and stops the stream itself.
- **If the Mac drops out** the phone vibrates and shows a red banner; it keeps saving.
- **Every trial is also saved on the phone** as `respi_imu_<date>_<time>.csv` in the same columns as the
  desktop files (`Time_ms,ax,ay,az,gx,gy,gz`, `Time_ms` in Unix ms from the phone's clock). They are listed
  under "Saved on this phone" with share (AirDrop) and delete buttons, and appear in the Files app under
  On My iPhone › Respi IMU. A trial the recorder rejects because Bluetooth lost samples is still complete here.

## Desktop capture

From this repository on the desktop:

```powershell
uv sync
uv run respi iphone-imu-devices
uv run respi capture-iphone-imu --seconds 60
```

Or open the desktop UI:

```powershell
uv run respi app --sensor iphone-imu
```

The saved CSV uses the same schema as the ESP32 IMU path:

```text
Time_ms,ax,ay,az,gx,gy,gz
```

Acceleration is in `g`; gyroscope values are in `deg/s`.

## BLE protocol

- Device name: `RespiPhoneIMU`
- Service UUID: `7B61B4E2-F5B4-4C90-8C7F-A7B2F1E8F4D0`
- Notify characteristic: `7B61B4E3-F5B4-4C90-8C7F-A7B2F1E8F4D0`
- Control characteristic: `7B61B4E4-F5B4-4C90-8C7F-A7B2F1E8F4D0`

The desktop writes ASCII `START` or `STOP` to the control characteristic. Notifications use little-endian binary payloads:

```text
uint8  version = 1
uint8  sample_count
uint16 sequence
repeat sample_count:
  uint32 time_ms_since_stream_start
  int16  ax_mg
  int16  ay_mg
  int16  az_mg
  int16  gx_centi_deg_per_s
  int16  gy_centi_deg_per_s
  int16  gz_centi_deg_per_s
```

The app adapts batch size to the central's `maximumUpdateValueLength`. With a 20-byte BLE payload it sends one sample per notification; with a larger MTU it sends multiple samples per notification.
