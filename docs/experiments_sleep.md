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

## 2026-11-09 -- Whole-night stage model

- **Method:** bidirectional GRU over the 30 s epochs (common schema of `tools/build_open_sleep_epochs.py`: heart,
  breathing, movement scaled per night, masks for missing modalities), random 2-4 h crops, modality dropout, classes
  weighted.  Evaluations: SLEEPBRL radar vs PSG (4 folds, with/without Walch); own nights vs Garmin (scratch,
  zero-shot, fine-tuned).
- **SLEEPBRL radar (breathing + movement) vs PSG:** kappa 0.470 +- 0.006 (radar nights only), 0.480 +- 0.005 with
  Walch; agreement 62%; F1 awake 0.72, light 0.57, deep 0.59, REM 0.60.  Radar alone stages sleep moderately against
  a real reference.
- **Own nights vs Garmin (7 nights, each held out):** from scratch kappa 0.139 +- 0.029 (best so far: logistic
  0.054, time of night 0.111); fine-tuned from open data 0.131; open data only (zero-shot) -0.055.
- **Reading:** a model that stages radar nights against PSG does *not* agree with the watch on the own nights --
  either the transfer from the 4 GHz radar and wrist data to the A121 nights fails, or the watch's stages differ from
  PSG-style staging (both are estimates; without PSG on the own nights the two cannot be told apart).  Learning on
  the own nights with Garmin labels gives the best agreement with Garmin, as expected.
- **Not checked yet:** stage shares and timing of the zero-shot predictions on the own nights.
- **Reproduce:** `tools/build_open_sleep_epochs.py`, `tools/sleep_stage_sequence.py`.

## 2026-11-09 -- Heart-rate network with radar + ECG pretraining

- **Data:** 7 731 windows of 60 s from Schellenberger (30 people, ECG heart rate 49-87 bpm); the plain spectral peak
  already gives 3.2 bpm MAE at rest -- the heart shows clearly.
- **Method:** one encoder shared by all bins, pooled over bins (takes 1 CW channel or 7 A121 bins); pretrain on 25
  people (5 validate), then fine-tune on the nights; also zero-shot.
- **Pretraining:** validation MAE 1.0-1.2 bpm on the 5 held-out people.
- **Result on the 5 watch nights (each held out, 3 seeds; pipeline 3.85 bpm on its 65% of windows, 10-minute
  medians 2.76 bpm, r 0.51):**

  | variant | MAE all windows | MAE at the pipeline's coverage | windows at the pipeline's MAE | 10-min MAE | 10-min r | strap 19:26 |
  |---|---|---|---|---|---|---|
  | shared encoder, own nights only | 3.70 | 2.94 | 89% | 2.76 | 0.65 | 1.4 |
  | shared encoder, pretrained + fine-tuned | **3.50** | **2.62** | **99%** | **2.38** | **0.71** | 1.5 |
  | pretrained only (zero-shot) | 6.20 | 4.50 | 59% | 4.62 | 0.30 | 2.5 |

- **Reading:** the shared-encoder network now beats the pipeline -- lower error on *all* windows than the pipeline on
  the 65% it accepts, and a better trend over the night.  Most of the gain comes from the architecture (one encoder
  for every bin, pooled; the earlier 7-channel network: 4.22 bpm); pretraining adds a little on average and most on
  the hardest night (1.6 m: 3.60 vs 4.68 bpm).  Validation MAE (end of the training nights) slightly prefers the
  network without pretraining (3.69 vs 3.85), so the honest pick by validation is a near tie.  Zero-shot transfer
  from the 24 GHz radar fails: fine-tuning on A121 data is needed.
- **Reproduce:** `tools/build_radar_ecg_windows.py`, `tools/train_sleep_heart_rate.py --variants
  shared,shared-pre,shared-zeroshot --tag _pretrain`.

## 2026-11-10 -- The breathing-phase network on the nights

- **Method:** the chosen phase network (TCN pretrained + fine-tuned, 3 seeds averaged) on every 60 s epoch window of
  the 10 nights, input formed as in training (3 bins around the target, 2 Hz low-pass, causal 30 s normalisation
  over 90 s), Viterbi path; features from the last 40 s.
- **Does it find breathing in sleep?** Its breathing rate (inhale onsets per minute) vs Acconeer's: bias +0.6/min,
  MAE 1.74, r 0.32 (7 489 still epochs); vs Garmin: MAE 2.02, r 0.15 (Acconeer's rate vs Garmin: MAE 1.3-2.0).  Inhale
  share 0.50 on average; pauses rare (median 0, mean 3.4% of time).  So it does track breathing at night, about as
  well as the existing rate estimate -- trained on sitting daytime data only.
- **Reproduce:** `tools/breath_phases_at_night.py`.  Its features enter the stage model as group "net".

## 2026-11-10 -- Stages from heart, breathing or movement alone, and together (open data, true references)

- **Question (supervisor):** how much of the sleep stage does each signal carry -- heart rate, breathing, movement --
  and all of them together?
- **Method:** the whole-night BiGRU with only the chosen feature groups (others zeroed; time of night a separate
  group), 1000 steps, 3 seeds; 4 folds over the nights of each set; kappa pooled over the held-out nights, and per
  night the minutes of each stage against the reference (correlation over nights, mean absolute difference).
- **Result (kappa):**

  | signals | Walch (watch, PSG, 31 nights) | Bidsleep (watch, Dreem EEG, 253 nights) | SLEEPBRL (4 GHz radar, PSG, 32 nights) |
  |---|---|---|---|
  | time of night only | 0.136 | 0.230 | 0.119 |
  | heart rate | 0.373 | 0.384 | -- |
  | breathing | -- | -- | **0.475** |
  | movement | 0.226 | 0.335 | 0.308 |
  | heart + movement / breathing + movement | 0.418 | 0.423 | 0.443 |
  | the same + time of night | 0.470 | 0.447 | 0.476 |

- **Reading:** each signal alone stages sleep clearly better than the time of night; the radar's breathing alone
  (rate, depth, regularity, interval variation from the chest) matches what heart rate and movement from a watch give
  together; movement adds to the watch's heart rate but nothing to the radar's breathing (the breathing features
  already show restlessness).  Per-night stage minutes: REM and light sleep follow the reference best (r 0.6-0.8),
  deep sleep worst (r 0.1-0.3).  Agreement with the true reference is moderate (kappa 0.42-0.48); compare with
  published consumer-device figures before claiming where it stands.
- **Reproduce:** `tools/sleep_stage_sequence.py --ablation walch|bidsleep|sleepbrl`; tables in
  `runs/sleep_stage_sequence/ablation_*.md`.

## 2026-11-10 -- External test of the breathing-phase network (other people, another radar)

- **Data:** Schellenberger et al. 2020 (135 recordings, 30 people lying, 24 GHz CW radar; reference thoracic
  impedance, inhale = up) and Shi et al. 2020 (208 recordings with the radar at the chest, 11 people; reference nasal
  temperature airflow, turned over).  Labels: the deterministic detector (offline) on the reference, as the belt
  labels of the author's data.  Radar polarity decided from the radar alone by the shape of the breath: agrees with
  the reference in 99% / 81% of the decided recordings; radar-reference lag -0.1 / -0.2 s.
- **Method:** the chosen network unchanged (TCN pretrained + fine-tuned, 3 seeds, echo channel 0) over each whole
  recording, Viterbi; scored after the first 20 s; the detector on the radar offline and causal.
- **Result (all recordings; live comparison = network vs causal detector):**

  | set | network acc / macro-F1 / false boundaries per min | causal detector | offline detector | holds >= 5 s found (net / causal / offline) |
  |---|---|---|---|---|
  | Schellenberger | **72.8% / 0.643 / 6.4** | 67.1% / 0.612 / 17.4 | 82.3% / 0.797 / 2.3 | 315 / 439 / 610 of 703 |
  | Shi | **65.4% / 0.517 / 8.6** | 58.0% / 0.498 / 24.8 | 67.6% / 0.616 / 6.6 | 7 / 16 / 30 of 52 |

- **Reading:** the network, trained on one person sitting in front of the A121, carries over to 41 other people and a
  different radar: it beats the real-time detector there too, with 2.7-2.9x fewer false boundaries, though below its
  81.9% on the author's own test.  Its weakness is the same as at home -- holds (F1 0.17-0.40): it misses most breath
  holds that the detector finds.  As in the thesis, the offline detector looks best partly because the labels come
  from the same detector (here on the reference sensor).
- **Bug on the way:** the polarity test was given chest motion while it expects A121 phase (it multiplies by the A121
  sign): every radar trace came out upside down (agreement 1%, lag stuck at the search limit).  Fixed by passing
  ``chest_sign=1``.
- **Reproduce:** `tools/breath_external_test.py`; `runs/breath_external/summary.md` has every scenario.

## 2026-11-10 -- Apnea feasibility

- **Detector:** breathing-band amplitude every second (8 s spread) over the 90th percentile of the surrounding 5 min;
  an event is a drop below 0.6 for >= 10 s (gaps < 3 s bridged), apnea-like below 0.2.  Threshold and baseline
  quantile chosen on the first 5 UCDDB subjects, scored on the other 20.
- **A, voluntary breath holds on a 24 GHz radar (Schellenberger apnea scenario, 24 recordings, holds found by the
  same detector on the thoracic impedance):** sensitivity 0.87, precision 0.88 (62 holds).  A radar sees breathing stop.
- **B, clinical events on a thoracic effort belt (UCDDB, 20 held-out adults with suspected apnea, 2 363 annotated
  events):** sensitivity 0.79 overall -- central apneas 0.87, mixed 0.93, obstructive 0.79; hypopneas central 0.81,
  mixed 0.95, obstructive 0.72; precision 0.53; estimated vs annotated AHI r 0.85, MAE 10.3/h, AHI >= 15 classified
  alike in 70%.  Chest motion alone carries most clinical events, obstructive hypopneas least -- as expected.
- **C, false alarms on the author's 10 A121 nights (healthy, no reference):** 17-30 events per hour of sleep (median
  23), 3-11 apnea-like.  The radar's breathing amplitude in sleep varies far more than a belt's; the belt-tuned
  threshold is far too sensitive for it.  Following Acconeer's target every 30 s instead of a fixed hourly gate makes
  it worse (phase jumps at each gate change).
- **C2, synthetic events in those nights:** 288 injected, 60% found (apnea 68%, hypopnea to 30% / 50% / 70%: 61% / 56%
  / 36%; 10 s events 23%, 15-40 s 64-78%).
- **Reading:** the path is plausible -- a radar sees breathing stop (A) and a chest-motion signal carries most clinical
  events (B) -- but on the A121 sleep recordings the detector is not usable as it is: a radar-specific operating point
  needs real apnea events recorded with the radar, which this work does not have.
- **Reproduce:** `tools/apnea_feasibility.py` (`runs/apnea/summary.json`).
