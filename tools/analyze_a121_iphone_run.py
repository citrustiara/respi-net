#!/usr/bin/env python3
"""Can a phone on the body label the radar's breathing?  One A121 + iPhone coach run at a time.

The recorder (``a121_iphone_guided_recording.py``) writes the radar and the phone
stream of one trial with the same clock.  For every run this script

1. takes the chest motion of each sensor on a common 20 Hz grid (radar: phase of
   the chest range bins in mm; phone: main axis of the accelerometer in mg),
2. turns the phone signal the right way up by its correlation with the radar,
3. measures the delay between the two sensors three ways that do not depend on
   each other (cross-correlation of the velocity, the moment of the fastest
   motion of every cued inhale and exhale, the moment of every turn), and
   checks that the delay is the same in the first and in the second half,
4. shifts the phone by that delay and labels it with the same deterministic
   detector as the radar, and scores both detectors against the radar's
   labels made per breath from the coach cues
   (:func:`respi_net.label_alignment.align_cues`), which is the reference used
   for all coach runs.

Recordings without cued inhales and exhales (``nn_self_12min``) are analysed in a second mode, ``analyse_free``:
the delay is measured minute by minute (it has to stay put for the phone to be a usable reference), the two
detectors are compared per block of the pattern, and the long holds of the holds block are counted.  The radar's
time stamps are first regularised (:func:`respi_net.chest_signal.regular_frame_times_ms`): the driver's stamps
run 0.17 percent fast, which is 1.3 s over 12 minutes and looks like a delay that grows by 0.11 s every minute.

Writes one JSON per run to ``data/processed/breath_phases/a121_iphone/`` and, with
``--thesis-figure``, ``docs/thesis/figures/fazy_telefon_radar.png``.

    uv run python tools/analyze_a121_iphone_run.py --session pilot_coach_1m --thesis-figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from respi_net.breath_phases import CLASS_COLOURS, CLASS_NAMES_PL, IGNORE, label_runs
from respi_net.chest_signal import a121_chest_signal, imu_chest_signal, light_filter
from respi_net.label_alignment import align_cues, cue_phases, estimate_lag
from respi_net.phase_baseline import detect_phases
from respi_net.phase_metrics import boundary_errors, score_phases

SESSIONS = ROOT / "data" / "raw" / "nn" / "a121_iphone"
OUT = ROOT / "data" / "processed" / "breath_phases" / "a121_iphone"
FIGURE = ROOT / "docs" / "thesis" / "figures" / "fazy_telefon_radar.png"
FIGURE_FREE = ROOT / "docs" / "thesis" / "figures" / "fazy_telefon_12min.png"
GRID_HZ = 20.0
MAX_LAG_S = 1.5
EVENT_LOW_PASS_HZ = 1.5
MIN_COVERAGE = 0.95


def _peak(lags: np.ndarray, values: np.ndarray, index: int) -> float:
    """Lag of the extreme at ``index`` refined by a parabola through its neighbours."""

    if 0 < index < len(values) - 1:
        a, b, c = values[index - 1], values[index], values[index + 1]
        denominator = a - 2.0 * b + c
        if denominator != 0:
            return float(lags[index] + 0.5 * (a - c) / denominator * (lags[1] - lags[0]))
    return float(lags[index])


def cross_correlation(reference: np.ndarray, other: np.ndarray, fs: float, max_lag_s: float = MAX_LAG_S) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of ``other(t) = reference(t - lag)`` for lags in ``[-max_lag_s, max_lag_s]``."""

    n = len(reference)
    lags, values = [], []
    for k in range(-int(max_lag_s * fs), int(max_lag_s * fs) + 1):
        x, y = (reference[: n - k], other[k:]) if k >= 0 else (reference[-k:], other[: n + k])
        lags.append(k / fs)
        values.append(float(np.corrcoef(x, y)[0, 1]))
    return np.asarray(lags), np.asarray(values)


def best_lag(reference: np.ndarray, other: np.ndarray, fs: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    lags, values = cross_correlation(reference, other, fs)
    index = int(np.argmax(np.abs(values)))
    return _peak(lags, values * np.sign(values[index]), index), float(values[index]), lags, values


def shifted(time_s: np.ndarray, signal: np.ndarray, delay_s: float) -> np.ndarray:
    """``signal`` moved earlier by ``delay_s`` (the sensor that reports late is brought back)."""

    return np.interp(time_s + delay_s, time_s, signal)


def event_differences(
    time_s: np.ndarray, radar: np.ndarray, phone: np.ndarray, cues: pd.DataFrame, fs: float
) -> dict[str, np.ndarray]:
    """Phone minus radar, in seconds, for the fastest motion of each cued breath and for each turn."""

    smooth_r = light_filter(radar, fs, EVENT_LOW_PASS_HZ)
    smooth_p = light_filter(phone, fs, EVENT_LOW_PASS_HZ)
    vel_r, vel_p = np.gradient(smooth_r) * fs, np.gradient(smooth_p) * fs
    found: dict[str, list[tuple[float, float]]] = {"inhale_speed": [], "exhale_speed": [], "peak": [], "trough": []}
    for row in cues.itertuples():
        if row.kind not in ("inhale", "exhale"):
            continue
        inside = (time_s >= row.start_s - 0.5) & (time_s <= row.end_s + 0.5)
        if inside.sum() < 5:
            continue
        pick = np.argmax if row.kind == "inhale" else np.argmin
        found[f"{row.kind}_speed"].append((row.start_s, float(time_s[inside][pick(vel_p[inside])] - time_s[inside][pick(vel_r[inside])])))
        turn = (time_s >= row.end_s - 1.0) & (time_s <= row.end_s + 1.5)
        if turn.sum() < 5:
            continue
        pick_turn = np.argmax if row.kind == "inhale" else np.argmin
        found["peak" if row.kind == "inhale" else "trough"].append(
            (row.end_s, float(time_s[turn][pick_turn(smooth_p[turn])] - time_s[turn][pick_turn(smooth_r[turn])]))
        )
    return {key: np.asarray(value, dtype=float).reshape(-1, 2) for key, value in found.items()}


def describe(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"n": 0, "median_s": float("nan"), "iqr_s": float("nan")}
    return {
        "n": int(len(values)),
        "median_s": float(np.median(values)),
        "iqr_s": float(np.subtract(*np.percentile(values, [75, 25]))),
    }


def scores(reference: np.ndarray, predicted: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    phases = score_phases(reference, np.where(predicted == IGNORE, -9, predicted))
    boundaries = boundary_errors(reference, predicted, grid)
    return {
        "accuracy": float(phases.accuracy),
        "macro_f1": float(phases.macro_f1),
        "samples": int(phases.samples),
        "boundary_detection_rate": float(boundaries.detection_rate),
        "boundary_median_abs_error_s": float(boundaries.median_abs_error_s),
        "boundary_median_delay_s": float(boundaries.median_delay_s),
        "false_boundaries_per_min": float(boundaries.false_per_minute),
    }


def stamp_lateness_s(run_prefix: Path) -> float:
    """Lower bound on how late the phone's time stamps are (recorder's estimate from BLE arrival times), or NaN."""

    try:
        number = int(run_prefix.name.split("_")[1])
        manifest = json.loads((run_prefix.parent / "manifest.json").read_text())
        for entry in manifest["measurements"]:
            if entry["trial"]["number"] == number:
                return float(entry["iphone"]["ble_arrival"]["stamp_lateness_estimate_ms"]) / 1000.0
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return float("nan")


def analyse(run_prefix: Path) -> dict[str, object]:
    cues = pd.read_csv(f"{run_prefix}_cues.csv")
    origin = float(cues["start_wall_ms"].iloc[0])
    duration = float(cues["end_s"].iloc[-1])
    radar_signal = a121_chest_signal(f"{run_prefix}_a121.csv", origin_ms=origin)
    phone_signal = imu_chest_signal(f"{run_prefix}_iphone.csv", origin_ms=origin, name="iphone")
    coverage = len(phone_signal.time_s) / (duration * phone_signal.fs)
    grid = np.arange(0.0, duration, 1.0 / GRID_HZ)
    radar = np.interp(grid, radar_signal.time_s, radar_signal.chest)
    phone = np.interp(grid, phone_signal.time_s, phone_signal.chest)
    records = cues[["start_s", "end_s", "kind"]].to_dict("records")
    phases = cue_phases(records)

    # 1. orientation and delay between the two sensors
    body = grid >= 10.0  # the first seconds are free breathing, still breathing, but settling
    lag_vel, corr_vel, lags, corr_curve = best_lag(np.gradient(radar)[body] * GRID_HZ, np.gradient(phone)[body] * GRID_HZ, GRID_HZ)
    sign = 1.0 if corr_vel >= 0 else -1.0
    phone = sign * phone
    lag_vel, corr_vel, lags, corr_curve = best_lag(np.gradient(radar)[body] * GRID_HZ, np.gradient(phone)[body] * GRID_HZ, GRID_HZ)
    lag_pos, corr_pos, _, _ = best_lag(radar[body], phone[body], GRID_HZ)
    events = event_differences(grid, radar, phone, cues, GRID_HZ)
    pooled = np.concatenate([values[:, 1] for values in events.values() if len(values)]) if any(len(v) for v in events.values()) else np.array([])
    half = duration / 2.0
    first = np.concatenate([v[v[:, 0] < half, 1] for v in events.values() if len(v)]) if len(pooled) else np.array([])
    second = np.concatenate([v[v[:, 0] >= half, 1] for v in events.values() if len(v)]) if len(pooled) else np.array([])
    delay = float(np.median(pooled)) if len(pooled) else lag_vel

    # 2. labels
    radar_lag, radar_match, _ = estimate_lag(grid, np.gradient(radar) * GRID_HZ, phases, np.arange(-1.5, 1.5001, 0.025))
    reference = align_cues(grid, radar, records, fs=GRID_HZ, lag_s=radar_lag)
    phone_lag, phone_match, _ = estimate_lag(grid, np.gradient(phone) * GRID_HZ, phases, np.arange(-1.5, 1.5001, 0.025))
    phone_cues = align_cues(grid, phone, records, fs=GRID_HZ, lag_s=phone_lag)
    phone_aligned = shifted(grid, phone, delay)
    radar_detected = detect_phases(radar, GRID_HZ)
    phone_detected = detect_phases(phone, GRID_HZ)
    phone_shifted_detected = detect_phases(phone_aligned, GRID_HZ)
    cue_labels = np.full(len(grid), IGNORE, dtype=np.int8)
    for phase in phases:
        if phase.label != IGNORE:
            cue_labels[(grid >= phase.start_s + radar_lag) & (grid < phase.end_s + radar_lag)] = phase.label

    result: dict[str, object] = {
        "ble_stamp_lateness_s": stamp_lateness_s(run_prefix),
        "run": run_prefix.name,
        "session": run_prefix.parent.name,
        "duration_s": duration,
        "phone_coverage": coverage,
        "radar_gate_m": [radar_signal.meta["gate_start_m"], radar_signal.meta["gate_end_m"]],
        "radar_echo_db": float(np.nanmean(radar_signal.echo_db)),
        "radar_mm_p5_p95": float(np.subtract(*np.percentile(radar, [95, 5]))),
        "phone_mg_p5_p95": float(np.subtract(*np.percentile(phone, [95, 5]))),
        "phone_sign_flipped": bool(sign < 0),
        "delay": {
            "chosen_s": delay,
            "velocity_xcorr_s": lag_vel,
            "velocity_xcorr_corr": corr_vel,
            "position_xcorr_s": lag_pos,
            "position_xcorr_corr": corr_pos,
            "events": {key: describe(values[:, 1]) for key, values in events.items()},
            "events_pooled": describe(pooled),
            "first_half": describe(first),
            "second_half": describe(second),
        },
        "cue_match": {
            "radar": {**reference.summary(), "lag_s": radar_lag, "template_corr": radar_match},
            "phone": {**phone_cues.summary(), "lag_s": phone_lag, "template_corr": phone_match},
        },
        "against_radar_reference": {
            "radar_detector": scores(reference.labels, radar_detected, grid),
            "phone_detector_unshifted": scores(reference.labels, phone_detected, grid),
            "phone_detector_shifted": scores(reference.labels, phone_shifted_detected, grid),
            "phone_cue_labels_unshifted": scores(reference.labels, phone_cues.labels, grid),
        },
        "phone_detector_vs_radar_detector": scores(np.where(radar_detected == IGNORE, IGNORE, radar_detected), phone_shifted_detected, grid),
        "_plot": {
            "grid": grid,
            "radar": radar,
            "phone": phone,
            "phone_aligned": phone_aligned,
            "lags": lags,
            "corr_curve": corr_curve,
            "events": events,
            "delay": delay,
            "cue_lines": np.asarray(cues["start_s"], dtype=float),
            "labels": {
                "cue": cue_labels,
                "reference": reference.labels,
                "radar": radar_detected,
                "phone": phone_shifted_detected,
            },
        },
    }
    return result


SHORT_BLOCKS = {
    "USPOKÓJ POZYCJĘ": "ułożenie",
    "SPOKOJNY ODDECH": "spokojnie",
    "WOLNO I GŁĘBOKO": "wolno",
    "SZYBKO": "szybko",
    "BARDZO PŁYTKO": "płytko",
    "PAUZY": "pauzy",
    "NIEREGULARNIE": "nieregularnie",
    "RUCH I KASZEL": "ruch",
}
CHUNK_S = 60.0
GOOD_CORRELATION = 0.5
LONG_HOLD_S = 4.0


def has_cued_breaths(cues: pd.DataFrame) -> bool:
    return bool(cues["kind"].isin(["inhale", "exhale"]).any())


def chunk_delays(time_s: np.ndarray, radar: np.ndarray, phone: np.ndarray, start_s: float, fs: float) -> list[dict[str, float]]:
    """Delay (phone minus radar) and correlation of the velocities in consecutive chunks."""

    velocity_r, velocity_p = np.gradient(radar) * fs, np.gradient(phone) * fs
    rows = []
    for lo in np.arange(start_s, time_s[-1] - CHUNK_S + 1e-9, CHUNK_S):
        inside = (time_s >= lo) & (time_s < lo + CHUNK_S)
        lag, corr, _, _ = best_lag(velocity_r[inside], velocity_p[inside], fs)
        rows.append({"start_s": float(lo), "delay_s": lag, "correlation": abs(corr)})
    return rows


def long_holds(labels: np.ndarray, time_s: np.ndarray, minimum_s: float = LONG_HOLD_S) -> list[tuple[float, float, int]]:
    out = []
    for start, stop, label in label_runs(labels):
        if label in (1, 3):
            begin, end = float(time_s[start]), float(time_s[min(stop, len(time_s)) - 1])
            if end - begin >= minimum_s:
                out.append((begin, end, int(label)))
    return out


def analyse_free(run_prefix: Path) -> dict[str, object]:
    cues = pd.read_csv(f"{run_prefix}_cues.csv")
    origin = float(cues["start_wall_ms"].iloc[0])
    duration = float(cues["end_s"].iloc[-1])
    phone_signal = imu_chest_signal(f"{run_prefix}_iphone.csv", origin_ms=origin, name="iphone")
    radar_fixed = a121_chest_signal(f"{run_prefix}_a121.csv", origin_ms=origin)
    radar_raw = a121_chest_signal(f"{run_prefix}_a121.csv", origin_ms=origin, regular_time=False)
    grid = np.arange(0.0, duration, 1.0 / GRID_HZ)
    phone = np.interp(grid, phone_signal.time_s, phone_signal.chest)
    radar = np.interp(grid, radar_fixed.time_s, radar_fixed.chest)
    radar_before = np.interp(grid, radar_raw.time_s, radar_raw.chest)
    start = float(cues["end_s"].iloc[0])  # after the settling block
    body = grid >= start
    _, corr_sign, _, _ = best_lag(np.gradient(radar)[body] * GRID_HZ, np.gradient(phone)[body] * GRID_HZ, GRID_HZ)
    sign = 1.0 if corr_sign >= 0 else -1.0
    phone = sign * phone
    after = chunk_delays(grid, radar, phone, start, GRID_HZ)
    before = chunk_delays(grid, radar_before, phone, start, GRID_HZ)
    good = [row for row in after if row["correlation"] >= GOOD_CORRELATION]
    good_before = [row for row in before if row["correlation"] >= GOOD_CORRELATION]
    delays = np.array([row["delay_s"] for row in good])
    delay = float(np.median(delays)) if len(delays) else float("nan")
    drift_before = float(np.polyfit([r["start_s"] for r in good_before], [r["delay_s"] for r in good_before], 1)[0] * 60.0) if len(good_before) >= 3 else float("nan")
    drift_after = float(np.polyfit([r["start_s"] for r in good], [r["delay_s"] for r in good], 1)[0] * 60.0) if len(good) >= 3 else float("nan")
    phone_aligned = shifted(grid, phone, delay)
    radar_labels = detect_phases(radar, GRID_HZ)
    phone_labels = detect_phases(phone_aligned, GRID_HZ)
    blocks = []
    for row in cues.itertuples():
        inside = (grid >= max(row.start_s, start)) & (grid < row.end_s)
        if not inside.any():
            continue
        a, b = radar_labels[inside], phone_labels[inside]
        both = (a >= 0) & (b >= 0)
        blocks.append(
            {
                "block": SHORT_BLOCKS.get(row.cue, row.cue),
                "start_s": float(row.start_s),
                "end_s": float(row.end_s),
                "radar_mm_p5_p95": float(np.subtract(*np.percentile(radar[inside], [95, 5]))),
                "phone_mg_p5_p95": float(np.subtract(*np.percentile(phone[inside], [95, 5]))),
                "detector_agreement": float((a[both] == b[both]).mean()) if both.any() else float("nan"),
            }
        )
    pause_block = next((b for b in blocks if b["block"] == "pauzy"), None)
    holds_phone = long_holds(phone_labels, grid)
    holds_radar = long_holds(radar_labels, grid)

    def inside_block(holds: list[tuple[float, float, int]]) -> tuple[int, int]:
        if pause_block is None:
            return 0, len(holds)
        inner = sum(1 for begin, end, _ in holds if begin >= pause_block["start_s"] - 1 and end <= pause_block["end_s"] + 1)
        return inner, len(holds) - inner

    result: dict[str, object] = {
        "run": run_prefix.name,
        "session": run_prefix.parent.name,
        "duration_s": duration,
        "phone_coverage": len(phone_signal.time_s) / (duration * phone_signal.fs),
        "ble_stamp_lateness_s": stamp_lateness_s(run_prefix),
        "radar_gate_m": [radar_fixed.meta["gate_start_m"], radar_fixed.meta["gate_end_m"]],
        "radar_echo_db": float(np.nanmean(radar_fixed.echo_db)),
        "radar_stamp_span_s": float(radar_raw.time_s[-1] - radar_raw.time_s[0]),
        "radar_regular_span_s": float(radar_fixed.time_s[-1] - radar_fixed.time_s[0]),
        "phone_sign_flipped": bool(sign < 0),
        "delay": {
            "chosen_s": delay,
            "good_minutes": len(good),
            "minutes": len(after),
            "spread_s": float(np.std(delays)) if len(delays) else float("nan"),
            "drift_s_per_min_before": drift_before,
            "drift_s_per_min_after": drift_after,
            "chunks_after": after,
            "chunks_before": before,
        },
        "blocks": blocks,
        "agreement_overall": float(
            np.mean(radar_labels[body & (radar_labels >= 0) & (phone_labels >= 0)] == phone_labels[body & (radar_labels >= 0) & (phone_labels >= 0)])
        ),
        "long_holds": {
            "phone": {"total": len(holds_phone), "in_pause_block": inside_block(holds_phone)[0], "elsewhere": inside_block(holds_phone)[1]},
            "radar": {"total": len(holds_radar), "in_pause_block": inside_block(holds_radar)[0], "elsewhere": inside_block(holds_radar)[1]},
        },
        "_plot_free": {
            "grid": grid,
            "radar": radar,
            "phone": phone_aligned,
            "after": after,
            "before": before,
            "delay": delay,
            "blocks": cues[["start_s", "end_s", "cue"]].to_dict("records"),
            "labels": {"radar": radar_labels, "phone": phone_labels},
            "holds": {"phone": holds_phone, "radar": holds_radar},
        },
    }
    return result


def plot_free(result: dict[str, object], path: Path) -> None:
    data = result["_plot_free"]  # type: ignore[assignment]
    grid = data["grid"]
    fig = plt.figure(figsize=(13, 10.2), constrained_layout=True)
    grid_spec = fig.add_gridspec(4, 1, height_ratios=[0.55, 3.4, 2.6, 1.7])
    names = fig.add_subplot(grid_spec[0])
    signals = fig.add_subplot(grid_spec[1], sharex=names)
    delays = fig.add_subplot(grid_spec[2], sharex=names)
    bars = fig.add_subplot(grid_spec[3], sharex=names)

    palette = ["#e5e7eb", "#f3f4f6"]
    for index, block in enumerate(data["blocks"]):
        names.axvspan(block["start_s"], block["end_s"], color=palette[index % 2], lw=0)
        names.text(0.5 * (block["start_s"] + block["end_s"]), 0.5, SHORT_BLOCKS.get(block["cue"], block["cue"]), ha="center", va="center", fontsize=8)
        for ax in (signals, delays, bars):
            ax.axvline(block["start_s"], color="#9ca3af", lw=0.5, ls=(0, (3, 3)))
    names.set_yticks([])
    names.set_xlim(grid[0], grid[-1])
    names.tick_params(labelbottom=False)
    names.set_title("Bloki wzorca nn_self_12min", loc="left", fontsize=10)

    radar, phone = data["radar"], data["phone"]
    signals.plot(grid, _z(radar, radar), color="#111827", lw=0.9)
    signals.plot(grid, _z(phone, phone), color="#ea580c", lw=0.9, alpha=0.9)
    signals.set_ylabel("ruch (znorm.)")
    signals.set_title(
        f"Radar i telefon po korekcie zegara radaru (telefon przesunięty o {abs(data['delay']):.2f} s)", loc="left", fontsize=10
    )
    signals.grid(alpha=0.25)
    signals.tick_params(labelbottom=False)

    for rows, colour, label in ((data["before"], "#9ca3af", "znaczniki radaru bez korekty"), (data["after"], "#2563eb", "po korekcie zegara")):
        good = [row for row in rows if row["correlation"] >= GOOD_CORRELATION]
        weak = [row for row in rows if row["correlation"] < GOOD_CORRELATION]
        delays.scatter([r["start_s"] + CHUNK_S / 2 for r in good], [r["delay_s"] for r in good], color=colour, s=34, label=label)
        delays.scatter([r["start_s"] + CHUNK_S / 2 for r in weak], [r["delay_s"] for r in weak], facecolors="none", edgecolors=colour, s=34)
    delays.axhline(0.0, color="#111827", lw=0.6)
    delays.set_ylabel("telefon − radar [s]")
    delays.set_title(
        "Opóźnienie w kolejnych minutach (pełne punkty: korelacja ≥ 0,5; puste: sygnał radaru zbyt słaby)", loc="left", fontsize=10
    )
    delays.legend(loc="lower left", fontsize=8, framealpha=0.9)
    delays.grid(alpha=0.25)
    delays.tick_params(labelbottom=False)

    _bars(bars, grid, [("radar:\ndetektor", data["labels"]["radar"]), ("telefon:\ndetektor", data["labels"]["phone"])])
    bars.set_xlabel("czas od startu nagrania [s]")
    bars.set_title("Fazy oddechu z detektora deterministycznego", loc="left", fontsize=10)
    fig.legend(
        handles=[
            Line2D([], [], color="#111827", lw=1.6, label="radar"),
            Line2D([], [], color="#ea580c", lw=1.6, label="telefon"),
            *[Patch(color=CLASS_COLOURS[i], label=CLASS_NAMES_PL[i]) for i in range(4)],
        ],
        loc="outside lower center",
        ncols=6,
        fontsize=9,
        frameon=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)



def _bars(ax: plt.Axes, grid: np.ndarray, rows: list[tuple[str, np.ndarray]]) -> None:
    height = 1.0 / len(rows)
    for position, (_, labels) in enumerate(rows):
        low = 1.0 - (position + 1) * height
        for start, stop, label in label_runs(labels):
            if label == IGNORE or label < 0:
                continue
            end = grid[stop] if stop < len(grid) else grid[-1]
            ax.axvspan(grid[start], end, ymin=low + 0.06 * height, ymax=low + 0.94 * height, color=CLASS_COLOURS[label], alpha=0.92, lw=0)
    ax.set_yticks([1.0 - (i + 0.5) * height for i in range(len(rows))], [name for name, _ in rows])
    ax.set_ylim(0, 1)
    ax.set_xlim(grid[0], grid[-1])


def _shift_title(delay: float) -> str:
    if delay < 0:
        return f"Po opóźnieniu telefonu o {-delay:.2f} s (telefon wyprzedzał radar)"
    return f"Po cofnięciu telefonu o {delay:.2f} s (telefon spóźniał się względem radaru)"


def _z(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    spread = np.subtract(*np.percentile(reference, [95, 5])) + 1e-12
    return (x - np.median(reference)) / spread


def plot(result: dict[str, object], path: Path) -> None:
    data = result["_plot"]  # type: ignore[assignment]
    grid, radar = data["grid"], data["radar"]
    delay = data["delay"]
    fig = plt.figure(figsize=(13, 10.6), constrained_layout=True)
    grid_spec = fig.add_gridspec(4, 2, height_ratios=[3.0, 3.0, 2.6, 1.9])
    top = fig.add_subplot(grid_spec[0, :])
    middle = fig.add_subplot(grid_spec[1, :], sharex=top)
    hist = fig.add_subplot(grid_spec[2, 0])
    corr = fig.add_subplot(grid_spec[2, 1])
    bars = fig.add_subplot(grid_spec[3, :], sharex=top)

    for ax, phone, title in (
        (top, data["phone"], "Sygnały z jednego zegara komputera, bez wyrównania"),
        (middle, data["phone_aligned"], _shift_title(delay)),
    ):
        for line in data["cue_lines"]:
            ax.axvline(line, color="#9ca3af", lw=0.5, ls=(0, (3, 3)))
        ax.plot(grid, _z(radar, radar), color="#111827", lw=1.3)
        ax.plot(grid, _z(phone, phone), color="#ea580c", lw=1.3)
        ax.set_ylabel("ruch (znorm.)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=0.25)
    top.tick_params(labelbottom=False)
    middle.tick_params(labelbottom=False)
    middle.set_xlim(grid[0], grid[-1])

    events = data["events"]
    all_values = np.concatenate([v[:, 1] for v in events.values() if len(v)])
    edges = np.arange(np.floor(all_values.min() * 10) / 10 - 0.05, np.ceil(all_values.max() * 10) / 10 + 0.11, 0.1)
    hist.hist(all_values, bins=edges, color="#ea580c", alpha=0.85, edgecolor="white")
    hist.axvline(delay, color="#111827", lw=1.4)
    hist.set_xlabel("telefon − radar [s]: moment najszybszego ruchu i zwrotu w każdym oddechu")
    hist.set_ylabel("liczba zdarzeń")
    hist.set_title(f"Opóźnienie w pojedynczych oddechach (mediana {delay:.2f} s, n = {len(all_values)})", loc="left", fontsize=10)
    hist.grid(alpha=0.25)

    corr.plot(data["lags"], data["corr_curve"], color="#111827", lw=1.3)
    corr.axvline(result["delay"]["velocity_xcorr_s"], color="#ea580c", lw=1.4)  # type: ignore[index]
    corr.set_xlabel("przesunięcie telefonu względem radaru [s]")
    corr.set_ylabel("korelacja prędkości")
    corr.set_title(
        f"Korelacja wzajemna: szczyt przy {result['delay']['velocity_xcorr_s']:.2f} s, r = {abs(result['delay']['velocity_xcorr_corr']):.2f}",  # type: ignore[index]
        loc="left",
        fontsize=10,
    )
    corr.grid(alpha=0.25)

    rows = [
        ("komendy\n(+ opóźnienie)", data["labels"]["cue"]),
        ("radar + komendy\n(odniesienie)", data["labels"]["reference"]),
        ("radar:\ndetektor", data["labels"]["radar"]),
        ("telefon:\ndetektor", data["labels"]["phone"]),
    ]
    _bars(bars, grid, rows)
    bars.set_xlabel("czas od startu komend [s]")
    bars.set_title("Fazy oddechu", loc="left", fontsize=10)
    fig.legend(
        handles=[
            Line2D([], [], color="#111827", lw=1.6, label="radar (przemieszczenie klatki)"),
            Line2D([], [], color="#ea580c", lw=1.6, label="telefon (przyspieszenie, oś główna)"),
            *[Patch(color=CLASS_COLOURS[i], label=CLASS_NAMES_PL[i]) for i in range(4)],
        ],
        loc="outside lower center",
        ncols=6,
        fontsize=9,
        frameon=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _clean(value: object) -> object:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if not key.startswith("_")}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default="pilot_coach_1m", help="folder name under data/raw/nn/a121_iphone")
    parser.add_argument("--run", default=None, help="run stem, for example run_01_paced_12_hold (default: every run)")
    parser.add_argument("--thesis-figure", action="store_true", help="draw the figure of the first checked run to the thesis folder")
    parser.add_argument("--figure", type=Path, default=None, help="where to draw the figure (default: next to the JSON)")
    args = parser.parse_args()
    session_dir = SESSIONS / args.session
    prefixes = sorted({Path(str(p).removesuffix("_cues.csv")) for p in session_dir.glob("run_*_cues.csv")})
    if args.run:
        prefixes = [p for p in prefixes if p.name == args.run]
    if not prefixes:
        raise SystemExit(f"No runs found in {session_dir}")
    OUT.mkdir(parents=True, exist_ok=True)
    drawn = False
    for prefix in prefixes:
        if not (Path(f"{prefix}_a121.csv").exists() and Path(f"{prefix}_iphone.csv").exists()):
            print(f"{prefix.name}: sensor files missing, skipped")
            continue
        free = not has_cued_breaths(pd.read_csv(f"{prefix}_cues.csv"))
        result = analyse_free(prefix) if free else analyse(prefix)
        if float(result["phone_coverage"]) < MIN_COVERAGE:  # type: ignore[arg-type]
            print(f"{prefix.name}: phone stream incomplete, skipped")
            continue
        out = OUT / f"{args.session}_{prefix.name}.json"
        out.write_text(json.dumps(_clean(result), indent=2, ensure_ascii=False))
        print(json.dumps(_clean(result), indent=2, ensure_ascii=False))
        draw = plot_free if free else plot
        figure = args.figure or out.with_suffix(".png")
        draw(result, figure)
        print(f"Figure: {figure}")
        if args.thesis_figure and not drawn:
            target = FIGURE_FREE if free else FIGURE
            draw(result, target)
            print(f"Thesis figure: {target}")
            drawn = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
