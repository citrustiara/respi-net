# ESP-IDF Environment & Build Guide

This guide documents the specific configurations and fixes needed to build and flash ESP32 projects on this system, avoiding the "Environment not set up" errors encountered during development.

## 1. Environment Paths

The primary ESP-IDF installation and tools are located here:
- **IDF_PATH**: `C:\espressif\frameworks\esp-idf-v5.5.3`
- **IDF_TOOLS_PATH**: `C:\Users\macie\.espressif` (Required for compilers, ninja, ccache)

## 2. Successful Activation (PowerShell)

Standard activation via `. .\export.ps1` may fail due to specific shell configurations or dot-sourcing issues. Use this sequence for a guaranteed setup:

```powershell
# 1. Explicitly set the tools path (Crucial for finding the compiler)
$env:IDF_TOOLS_PATH = 'C:\Users\macie\.espressif'

# 2. Navigate to the ESP-IDF root
cd C:\espressif\frameworks\esp-idf-v5.5.3

# 3. Source the export script
. .\export.ps1

# 4. Return to your project and build
cd path\to\your\project
idf.py build
```

## 3. ESP-IDF v5 ADC API Tips

If using ESP-IDF **v5.x**, the older `esp_adc_cal` component is deprecated.

### CMakeLists.txt
Use `esp_adc` instead of `esp_adc_cal`:
```cmake
idf_component_register(SRCS "main.c"
                    PRIV_REQUIRES driver esp_timer esp_adc)
```

### C Code (radar_main.c)
- Use `#include "esp_adc/adc_oneshot.h"`.
- Structs use `_cfg_t` suffix instead of `_config_t` (e.g., `adc_oneshot_unit_init_cfg_t`).
- Use `ADC_ATTEN_DB_12` instead of the deprecated `ADC_ATTEN_DB_11`.

## 4. Common Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `idf.py: The term 'idf.py' is not recognized` | `export.ps1` failed or wasn't run. | Ensure the Activation sequence in Step 2 is followed exactly. |
| `ninja: error: rebuild 'build.ninja' failed` | Component dependency error or toolchain not found. | Check `IDF_TOOLS_PATH` and verify `PRIV_REQUIRES` in `CMakeLists.txt`. |
| `xtensa-esp32-elf-gcc: not found` | Compiler path missing from $env:PATH. | Running `. .\export.ps1` with `$env:IDF_TOOLS_PATH` set fixes this. |

## 5. Helpful Commands

- **List Serial Ports**: `[System.IO.Ports.SerialPort]::GetPortNames()`
- **Build & Flash**: `idf.py -p (PORT) flash monitor`
- **Clean Build**: `idf.py fullclean`
