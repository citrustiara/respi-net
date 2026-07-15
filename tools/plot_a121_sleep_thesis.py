#!/usr/bin/env python
"""Create separate, thesis-ready A121 overnight sleep plots.

Each ``--night`` argument contains a panel label, a gated sleep-vitals CSV and either a Garmin
directory/file or ``-`` when no watch export is available.  The script deliberately writes sleep
phases, heart rate and respiration to separate files.  Long invalid radar gaps remain gaps; unlike
the live diagnostic view, they are not connected solely for visual continuity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from respi_net.a121_sleep import (
    _merge_garmin_nearest,
    _plot_garmin_sleep_levels,
    _plot_step,
    classify_radar_sleep,
    load_garmin_reference,
)
from respi_net.garmin_fit import GarminReferenceData


@dataclass(frozen=True)
class NightPanel:
    label: str
    trend_path: Path
    garmin_path: Path | None
    classified: pd.DataFrame
    step_s: float
    garmin: GarminReferenceData


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create separate overnight A121 thesis plots.")
    parser.add_argument(
        "--night",
        action="append",
        nargs=3,
        required=True,
        metavar=("LABEL", "TREND_CSV", "GARMIN_OR_DASH"),
        help="Repeat for every panel; use '-' when no Garmin export is available.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", default="Całonocny pomiar A121")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _load_panel(label: str, trend_text: str, garmin_text: str) -> NightPanel:
    trend_path = Path(trend_text)
    trend = pd.read_csv(trend_path)
    classified = classify_radar_sleep(trend)
    garmin_path = None if garmin_text == "-" else Path(garmin_text)
    garmin = load_garmin_reference(garmin_path)
    merged = _merge_garmin_nearest(classified.frame, garmin)
    return NightPanel(label, trend_path, garmin_path, merged, classified.step_s, garmin)


def _setup_axes(panels: list[NightPanel], title: str, metric: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, raw_axes = plt.subplots(
        len(panels),
        1,
        figsize=(14.5, 2.75 * len(panels)),
        squeeze=False,
        constrained_layout=True,
    )
    axes = list(raw_axes[:, 0])
    fig.suptitle(f"{title} — {metric}", fontsize=15, fontweight="bold")
    return fig, axes


def _format_clock_axis(ax: plt.Axes, panel: NightPanel) -> None:
    times = pd.to_datetime(panel.classified["clock_time_dt"])
    ax.set_xlim(times.iloc[0], times.iloc[-1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
    ax.grid(True, alpha=0.27)
    ax.set_xlabel("Czas lokalny")


def _panel_note(panel: NightPanel, value_column: str) -> str:
    values = pd.to_numeric(panel.classified.get(value_column, pd.Series(dtype=float)), errors="coerce")
    valid_pct = 100.0 * float(values.notna().mean()) if len(values) else 0.0
    target = pd.to_numeric(panel.classified.get("target_distance_m", pd.Series(dtype=float)), errors="coerce").dropna()
    target_text = f"mediana odległości celu {target.median():.2f} m" if not target.empty else "odległość celu niedostępna"
    return f"{target_text}; ważne okna {valid_pct:.0f}%"


def _plot_phases(panels: list[NightPanel], output: Path, *, title: str, dpi: int) -> None:
    fig, axes = _setup_axes(panels, title, "estymowane fazy snu")
    for ax, panel in zip(axes, panels):
        frame = panel.classified
        times = pd.to_datetime(frame["clock_time_dt"])
        _plot_step(
            ax,
            times,
            frame["radar_sleep_phase_level"],
            step_s=panel.step_s,
            color="#7c3aed",
            lw=2.0,
            label="A121 — heurystyka",
        )
        _plot_garmin_sleep_levels(
            ax,
            panel.garmin,
            step_s=panel.step_s,
            label="Garmin Fenix 7 — estymata faz",
        )
        onset = times[frame["radar_sleep_onset"].astype(bool)]
        wake = times[frame["radar_wake"].astype(bool)]
        if not onset.empty:
            ax.axvline(onset.iloc[0], color="#16a34a", lw=1.0, alpha=0.8, label="początek snu A121")
        if not wake.empty:
            ax.axvline(wake.iloc[-1], color="#ea580c", lw=1.0, alpha=0.8, label="koniec/wybudzenie A121")
        corrected = frame.get("radar_direct_transition_bridge_to_light", pd.Series(False, index=frame.index)).astype(bool)
        if corrected.any():
            indices = np.flatnonzero(corrected.to_numpy())
            for idx in indices:
                ax.axvspan(times.iloc[idx], times.iloc[idx] + pd.Timedelta(seconds=panel.step_s), color="#fde68a", alpha=0.20)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Głęboki", "Lekki", "REM", "Czuwanie"])
        ax.set_ylim(-0.35, 3.35)
        ax.set_title(panel.label, loc="left", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", ncol=4, fontsize=7.5)
        _format_clock_axis(ax, panel)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_vital(
    panels: list[NightPanel],
    output: Path,
    *,
    title: str,
    dpi: int,
    radar_connected: str,
    radar_raw: str,
    garmin_column: str,
    metric_title: str,
    ylabel: str,
    colors: tuple[str, str, str],
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, axes = _setup_axes(panels, title, metric_title)
    for ax, panel in zip(axes, panels):
        frame = panel.classified
        times = pd.to_datetime(frame["clock_time_dt"])
        connected = pd.to_numeric(frame.get(radar_connected, pd.Series(dtype=float)), errors="coerce")
        raw = pd.to_numeric(frame.get(radar_raw, pd.Series(dtype=float)), errors="coerce")
        ax.plot(times, connected, color=colors[0], lw=1.65, label="A121 — ważne okna i krótkie luki")
        ax.scatter(times, raw, s=7, color=colors[1], alpha=0.25, linewidths=0, label="A121 — estymaty okienne")
        garmin_values = pd.to_numeric(frame.get(garmin_column, pd.Series(dtype=float)), errors="coerce")
        if garmin_values.notna().any():
            ax.plot(times, garmin_values, color=colors[2], lw=1.25, alpha=0.85, label="Garmin Fenix 7")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(panel.label, loc="left", fontsize=11, fontweight="bold")
        ax.text(
            0.995,
            0.04,
            _panel_note(panel, radar_connected),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
        )
        ax.legend(loc="upper right", ncol=3, fontsize=7.5)
        _format_clock_axis(ax, panel)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    panels = [_load_panel(*night) for night in args.night]
    prefix = args.output_prefix
    phase_output = prefix.with_name(f"{prefix.name}_fazy.png")
    hr_output = prefix.with_name(f"{prefix.name}_tetno.png")
    rr_output = prefix.with_name(f"{prefix.name}_oddech.png")

    _plot_phases(panels, phase_output, title=args.title, dpi=args.dpi)
    _plot_vital(
        panels,
        hr_output,
        title=args.title,
        dpi=args.dpi,
        radar_connected="hr_bpm_connected",
        radar_raw="hr_bpm",
        garmin_column="garmin_hr_bpm",
        metric_title="tętno",
        ylabel="Tętno [bpm]",
        colors=("#b91c1c", "#fca5a5", "#1d4ed8"),
        ylim=(42.0, 90.0),
    )
    _plot_vital(
        panels,
        rr_output,
        title=args.title,
        dpi=args.dpi,
        radar_connected="rr_bpm_connected",
        radar_raw="rr_bpm",
        garmin_column="garmin_rr_bpm",
        metric_title="częstość oddechu",
        ylabel="Oddechy/min",
        colors=("#15803d", "#86efac", "#0f766e"),
        ylim=(5.0, 24.0),
    )
    print(f"Saved phase plot: {phase_output}")
    print(f"Saved heart-rate plot: {hr_output}")
    print(f"Saved respiration plot: {rr_output}")


if __name__ == "__main__":
    main()
