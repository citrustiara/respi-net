# A121 aluminium-foil experiment brief

## Purpose

Test whether a **5 x 5 cm kitchen aluminium-foil patch attached to the chest with tape** improves the usable A121 respiratory signal, and whether the effect depends on the lens. The analysis should compare foil-attached and foil-removed recordings at the same lens and distance; raw amplitude alone is not sufficient because a stronger reflection can still have worse respiratory SNR.

## Hardware and acquisition

- Sensor: Acconeer A121 over WCH USB Dual_Serial Interface A.
- Lenses: Hyperbolic, FZP, and Flat Cover.
- Distances: 50 cm and 100 cm.
- A121 profile: 3.
- HWAAS: 32.
- Sweeps/frame: 8.
- Frame rate: 20 Hz.
- Requested range: 0.20–1.50 m.
- Recording duration: 60 seconds per accepted measurement, after a 10-second preparation period.
- Three repeats per condition.

## Design and data completeness

There are 12 conditions: 2 patch states x 2 distances x 3 lenses. Each has 3 repeats, for **36 accepted CSV recordings**:

1. Runs 1–18: foil patch attached with tape.
2. Runs 19–36: foil patch removed.

The recordings are stored under `data/raw/a121/guided/`:

- `guided_2026-07-13_23-45-25/`: run 1, created with the earlier filename/manifest format.
- `guided_2026-07-13_23-53-27/`: runs 2–18.
- `guided_2026-07-14_17-58-54/`: runs 19–36.
- `guided_2026-07-13_23-39-59/`: empty smoke-test session; ignore it.

Each recording contains approximately 1,200 frames. The current manifests mark all 36 measurements as accepted. Run 1 maps to condition 1, repeat 1: foil attached, 50 cm, Hyperbolic lens.

## Suggested analysis

Match recordings by `(distance, lens, repeat)` and compare foil-attached versus foil-removed values. Useful primary measures are respiratory-band SNR or PSD peak prominence in approximately 0.10–0.50 Hz, respiration-rate confidence, signal-quality score, target-distance stability, and valid/presence-frame percentage. Report peak amplitude as a secondary measure.

The order was not randomized: all foil-attached measurements were recorded before all foil-removed measurements. Therefore, lens/distance comparisons are useful, but foil effects should be reported with the possible influence of time, posture, sensor alignment, and subject drift explicitly acknowledged.
