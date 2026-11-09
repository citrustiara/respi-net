# Sleep experiments: log

One entry per experiment of the sleep part (heart rate, breathing, movement, stages, apnea at night): what was
tried, on what data, the numbers, and what follows from it -- negative results too.  Newest last.  Results live in
`runs/` (not in git); the commands below rebuild them.  The work queue is in `PLAN_NEURAL_NETWORK.md` (local).

Data used throughout: 10 A121 nights of the author (S01) -- 7 in June 2026 with Garmin Fenix 7 exports (heart and
breathing rate on 5 of them, sleep stages on all 7), 3 in October without the watch (it overwrote them before
syncing); the thesis numbers the June nights 1-5 (08, 09, 13, 15, 16 June), plus a short pilot (10 June) and a
failed-geometry night at 1.6 m (11 June).  Chest-strap (Suunto) sessions of 2026-11-07 with the father (S02).

---

## 2026-11-09 -- Heart-rate band of the night pipeline: 36 or 42 bpm

- **Goal:** let the night pipeline report heart rates down to 36 bpm (the user's request for the network range).
- **Data:** the 5 nights with the watch's heart rate; sleep windows, Garmin value within 2 min.
- **Result:** with 36 bpm the pipeline agrees worse in every night: bias -3.4 to -5.5 bpm (42: -2.5 to +1.1), MAE
  4.8-6.3 bpm (42: 3.4-4.0).  The 36-42 bpm band collects higher harmonics of breathing.
- **Decision:** the pipeline keeps 42 bpm; the heart-rate network covers 36-120 bpm.
- **Reproduce:** `tools/analyze_a121_sleep_vitals_gated.py <night>.csv --hr-min-bpm 36` vs default.

## 2026-11-09 -- Chest range chosen by breathing in the night pipeline

- **Goal:** the pipeline follows Acconeer's target; the strongest echo sits 0.10-0.15 m from the sensor (housing,
  cover, bedding), so the target may not be the chest all night.  Test the breathing-based range choice of the
  breathing-phase chapter in every 60 s window.
- **Data:** the 5 watch nights, both selections, same sleep windows.
- **Result:** the range changes in 23-31% of windows, agreement with the watch does not change consistently
  (10-minute r up in 3 nights, down in 2; MAE within 0.8 bpm).  Breathing rate is identical (the analyzer takes
  Acconeer's recorded rate).
- **Side result:** the weaker nights are useful -- where the radar gives a heart rate it agrees with the watch about
  as well as on the best night (MAE 3.4-4.0 bpm per window, 1.9-3.1 bpm on 10-minute medians); a worse setup mainly
  lowers coverage (down to 45% at 1.6 m).  Breathing rate agrees except at 1.6 m (bias -2.3/min, no correlation).
- **Decision:** the pipeline keeps Acconeer's range.
- **Reproduce:** `tools/analyze_a121_sleep_vitals_gated.py --range-selection breathing`, then
  `tools/compare_a121_sleep_range_selection.py`.  Commit 6778f81.

## 2026-11-09 -- All nights in one compact file and 30 s epochs

- `tools/extract_a121_night.py`: the IQ of a night (1-2 GB CSV) as float16 in an npz (about 120 MB), states, Acconeer
  target; the June raw files are unpacked from the July archive one at a time and deleted after.
- `tools/build_sleep_epochs.py`: one row per 30 s -- heart rate, breathing (rate, depth, regularity, interval and depth
  variation), movement (motion energy, movement share, minutes since movement), 10-minute context, per-night scaling,
  the watch's stage / heart rate / breathing rate / stress / RMSSD where available.  8 916 epochs, 5 257 with a stage.

## 2026-11-09 -- Heart-rate network, first version (discarded)

- **What:** distribution over 36-120 bpm from 60 s of the 7 bins around the chest; time-domain CNN and spectral CNN,
  with/without time stretch; each watch night tested once, *the next night validating*.
- **Result:** the network output about 56 bpm on every night, also on its training nights (true 50-52): a whole
  night as the validation set rewards outputting that night's typical rate.  Its apparent wins (nights at ~57 bpm)
  were chance.
- **Fix:** validation = the last 20% of every training night (60 s gap).

## 2026-11-09 -- Heart-rate network, block validation

- **Data:** 5 watch nights, each held out once; 3 seeds; scored on the pipeline's windows.
- **Result (time-domain CNN with stretch, chosen by validation MAE):** MAE 4.22 bpm on all windows vs the pipeline's
  3.85 on its 65%; at the pipeline's coverage 3.61 (wins on 3 of 5 nights); keeps the pipeline's error on 81% of
  windows; 10-minute medians: 3.27 bpm, r 0.54 (pipeline 2.76, r 0.51); a constant training median: 5.03.
  Without stretch 4.44 / 57%; the spectral network 4.68.
- **Strap (session 19:26, 19 windows, ~56 bpm):** network 2.6 bpm on all windows, pipeline 1.5 on 89%.
- **Reproduce:** `tools/train_sleep_heart_rate.py`.  Commits a7e6f79, 19a1901.

## 2026-11-09 -- Strap quality

- Only strap pieces where >= 50% of readings pass the strap's own check (`hr_ok`) count.  The 20:16 session is
  dropped (0% in three pieces, the last at 122 bpm while the radar shows ~51); the 2 m session drops out (4%).

## 2026-11-09 -- Sleep stages, per-epoch logistic regression (7 watch nights)

- **Method:** multinomial logistic regression on feature sets, classes weighted, hidden-Markov smoothing over the
  night; leave-one-night-out.  Reference: Garmin's stages (not a ground truth).
- **Result (kappa):** always-light 0, time of night 0.111, rule-based classifier 0.083, breathing 0.015, heart 0.027,
  radar HRV -0.069, movement 0.015, all 0.042, all+time 0.054.  The watch's own heart rate differs by 3 bpm between
  its stages (52-55).
- **Bug on the way:** dividing the weighted classifier's output by the class prior again pushed every epoch out of
  light sleep; fixed.
- **Reproduce:** `tools/sleep_stage_crossval.py`.

## 2026-11-09 -- What the watch's stages follow (5 nights)

- **Method:** the same model on the watch's exported signals: heart rate and breathing rate (1/min), stress (1/min,
  from HRV), RMSSD (5 min).
- **Result (kappa):** heart 0.04, breathing 0.03, stress+RMSSD 0.125, all 0.132, all+time 0.141 -- even the watch's
  own export explains its stages weakly; it must use beat-level HRV and wrist acceleration not in the export.  On the
  same 5 nights radar movement reaches 0.18 (on 7 nights 0.02: unstable with so few nights).
- **Reproduce:** `tools/sleep_stage_crossval.py --watch`.

## 2026-11-09 -- Beat-to-beat variability from the radar

- **Method:** beats from the cardiac band, sub-frame peak timing, RMSSD/SDNN per 60 s.
- **Result:** RMSSD ~217 ms against the watch's 68 ms (timing jitter at 20 frames/s dominates); on the best night
  (13 June) 10-minute medians follow the watch's RMSSD (r 0.50) and stress (r -0.53), on the others not; as a stage
  feature it does not help.  A faster frame rate would be needed.
- **Reproduce:** part of `tools/build_sleep_epochs.py` (`heart_variability`).  Commit 19a1901.

## 2026-11-09 -- Open datasets

| set | content | reference | licence | use |
|---|---|---|---|---|
| Walch et al. 2019 (PhysioNet sleep-accel) | 31 nights, Apple Watch HR + acceleration | PSG | ODC-By 1.0 | stage pretraining |
| SLEEPBRL (PhysioNet) | 32 nights, 4 GHz CW bioradar I/Q 50 Hz | PSG | ODC-By 1.0 | stages, radar vs PSG |
| Bidsleep (PhysioNet, 2026) | 253 nights of 47 people, Apple Watch HR + acceleration | Dreem EEG headband | ODC-By 1.0 | stage pretraining |
| Schellenberger et al. 2020 (figshare) | 30 people lying, 24 GHz radar, ECG, thoracic impedance, apnea scenario | ECG, impedance | CC BY 4.0 | HR pretraining, breathing test, apnea A |
| Shi et al. 2020 (figshare) | 11 people, 24 GHz radar, nasal airflow, ECG, holds after inhale/exhale | airflow, ECG | CC BY 4.0 | breathing test, HR test |
| UCDDB (PhysioNet) | 25 suspected-apnea adults, thoracic/abdominal belts, events by type | technicians | ODC-By 1.0 | apnea B |
| Barvik et al. (Zenodo 23762780) | 60 GHz FMCW raw ADC, mostly one seated person at 1-2 m | finger PPG pulse rate | CC BY 4.0 | HR test at 60 GHz |

Not used: DREAMT (restricted, 20 GB, too slow to download), MESA/SHHS (application needed).  Precedent for moving
stage models to radar: Krauss et al. 2026 (MESA pretraining, MCC 0.25 -> 0.47 on radar).

## 2026-11-09 -- Thoracic impedance tracks radar breathing (Schellenberger)

- Band-passed impedance `tfm_z0` and the radar's displacement: same breathing rate, correlation 0.83-0.96 on three
  recordings (resting, apnea) -> usable as an independent breathing reference on radar for 30 people.

## 2026-11-09 -- Whole-night stage model (in progress)

- **Method:** bidirectional GRU over the 30 s epochs (common schema of `tools/build_open_sleep_epochs.py`: heart,
  breathing, movement scaled per night, masks for missing modalities), random 2-4 h crops, modality dropout, classes
  weighted.  Evaluations: SLEEPBRL radar vs PSG (4 folds, with/without Walch); own nights vs Garmin (scratch,
  zero-shot, fine-tuned).
- **First result:** SLEEPBRL radar (breathing + movement) vs PSG: kappa 0.46-0.49, agreement 61-62%; adding Walch
  changes little.  Own nights pending.
- **Reproduce:** `tools/build_open_sleep_epochs.py`, `tools/sleep_stage_sequence.py`.

## 2026-11-09 -- Heart-rate network with radar + ECG pretraining (in progress)

- **Data:** 7 731 windows of 60 s from Schellenberger (30 people, ECG heart rate 49-87 bpm); the plain spectral peak
  already gives 3.2 bpm MAE at rest -- the heart shows clearly.
- **Method:** one encoder shared by all bins, pooled over bins (takes 1 CW channel or 7 A121 bins); pretrain on 25
  people (5 validate), then fine-tune on the nights; also zero-shot.
- **First fold (night 8 June, pipeline 3.67 bpm):** pretrain + fine-tune 3.3-3.5 bpm, zero-shot 3.0-4.3 bpm.
- **Reproduce:** `tools/build_radar_ecg_windows.py`, `tools/train_sleep_heart_rate.py --variants
  shared,shared-pre,shared-zeroshot --tag _pretrain`.
