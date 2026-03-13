---
description: ESP-IDF build and flash commands
---

1. Build the project:
// turbo
run_command ". C:\espressif\frameworks\esp-idf-v5.5.3\export.ps1 ; idf.py build"

2. Flash to device:
run_command ". C:\espressif\frameworks\esp-idf-v5.5.3\export.ps1 ; idf.py -p (PORT) flash"

3. Monitor output:
run_command ". C:\espressif\frameworks\esp-idf-v5.5.3\export.ps1 ; idf.py -p (PORT) monitor"

4. Full Build, Flash, and Monitor:
run_command ". C:\espressif\frameworks\esp-idf-v5.5.3\export.ps1 ; idf.py -p (PORT) flash monitor"
