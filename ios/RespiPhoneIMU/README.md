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

The simulator cannot provide real Bluetooth peripheral advertising or useful motion data, so use a physical iPhone.

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
