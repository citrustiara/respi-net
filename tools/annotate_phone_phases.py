#!/usr/bin/env python3
"""Mark breathing phases by hand on a phone IMU trace (the reference for the radar network).

You see the phone trace only: no radar, no detector output.  Click where a new phase starts; the class of that
phase is the one you pressed a key for just before the click, otherwise the next one in the usual order
(inhale, exhale, inhale, ...; after a hold, the breath that was not held).

    e wydech (exhale)     i wdech (inhale)     h pauza (hold)     n szum (noise: movement, cough)
    .  then click         end of the annotated stretch (otherwise the window end)
    Backspace, u, Ctrl/Cmd+Z   undo the last click      right click   remove the mark nearest to the click
    s   save (also on closing the window)      + / -   zoom around the pointer      left / right arrow   pan

Inhale and exhale boundaries snap to the nearest turning point of the trace (a trough starts an inhale, a crest an exhale, within
0.8 s) and the pointer shows where the mark will land (a vertical line, a circle on the trace): what you see is what is saved, the raw
click time is kept in the ``clicked_start_s`` column.  Holds and noise are placed exactly where you click.  ``--no-snap`` turns this off.

A second, thinner teal line (on by default, ``--no-orientation`` hides it) is the phone's orientation distance from its rest
position (:mod:`respi_net.phone_orientation`, scaled to the main trace).  The main trace falls back to its baseline soon after a deep
inhale even when the breath is held; the teal line stays up through such a hold and falls with the real exhale, so use it for
the start and end of holds after a deep inhale.

A hold is a stillness of at least 4 s (natural rests between breaths last 1-3 s and belong to the breath).  The first click is the
start of the first segment.  Use the toolbar to zoom; clicks do not count while a toolbar tool is active.

    uv run python tools/annotate_phone_phases.py --session self_lying_3min_01 --run run_01_nn_self_3min --start 40 --length 60

With ``--belt <file>`` the trace is the strain belt instead (a ``run_XX_*_belt.csv`` of a guided session or a
``data/raw/strain_belt`` coach file), on the time axis of ``tools/label_strain_belt_runs.py``, so the hand labels can
be scored against the belt labels (``score_phone_annotations.py belt``).  The belt shows a stop clearly: a hold is a
stillness of at least 2 s there, as in the belt labeller.  You see the belt trace only, never its labels::

    uv run python tools/annotate_phone_phases.py --belt data/raw/nn/a121_belt/<session>/run_01_nn_holds_belt.csv --start 0 --length 75

Saves ``annotations/<session>__<run>__<annotator>__<start>-<end>__pass<N>.csv`` (with ``--belt``, ``<session>__<run>``
is the belt labeller's run name).  Label the same stretch again a
day later with ``--pass 2``: the difference between the two passes is the uncertainty of the reference.
``--demo out.png`` draws a scripted example on synthetic data instead of opening a window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib
import numpy as np

from respi_net.breath_phases import CLASS_COLOURS, EXHALE, HOLD, INHALE, NOISE
from respi_net.chest_signal import light_filter
from respi_net.phone_annotation import KEY_TO_CLASS, Segment, marks_to_segments, next_label, save_segments, turning_point

SESSIONS = ROOT / "data" / "raw" / "nn" / "a121_iphone"
ANNOTATIONS = ROOT / "annotations"
UNDO_KEYS = {"backspace", "u", "ctrl+z", "cmd+z", "super+z", "meta+z"}
MIN_SEGMENT_S = 0.3  # marks closer than this are one mark: the later click replaces the earlier
NAMES_PL = {EXHALE: "wydech", HOLD: "pauza", INHALE: "wdech", NOISE: "szum"}


class AnnotationSession:
    """The clicks and keys of one annotation, without any plotting."""

    def __init__(self, window_start_s: float, window_end_s: float, snapper=None) -> None:
        self.window_start_s = window_start_s
        self.window_end_s = window_end_s
        self.snapper = snapper  # (time_s, label) -> time of the turning point, or None for no snapping
        self.marks: list[tuple[float, int]] = []
        self.clicks: list[float] = []  # raw click time of every mark
        self.pending: int | None = None
        self.end_s: float | None = None
        self.setting_end = False

    def key(self, key: str) -> None:
        if key in KEY_TO_CLASS:
            self.pending = KEY_TO_CLASS[key]
            self.setting_end = False
        elif key == ".":
            self.setting_end = True
        elif key in UNDO_KEYS:
            self.undo()

    def click(self, time_s: float) -> None:
        time_s = float(min(max(time_s, self.window_start_s), self.window_end_s))
        if self.setting_end:
            self.end_s = time_s
            self.setting_end = False
            return
        landing, label = self.resolve(time_s)
        near = [i for i, (at, _) in enumerate(self.marks) if abs(at - landing) < MIN_SEGMENT_S]
        if near:  # a second click on (almost) the same spot corrects that mark instead of adding a sliver of a phase
            self.marks.pop(near[0])
            self.clicks.pop(near[0])
            if self.pending is None and self.snapper is not None:
                landing, label = self.resolve(time_s)
        self.marks.append((landing, label))
        self.clicks.append(time_s)
        self.pending = None

    def resolve(self, time_s: float) -> tuple[float, int]:
        """Where a click at ``time_s`` lands and what it starts.

        Between breaths (no key pressed) the click goes to the nearest real turning point, a trough or a crest, and the
        class follows from it: a trough starts an inhale, a crest an exhale.  A pressed ``i``/``e`` looks only for its
        own kind.  With no turning point within reach, or next to a hold or noise, the mark stays where you click.
        """

        label = self.next_class()
        previous = self.marks[-1][1] if self.marks else None
        if self.snapper is None or label not in (INHALE, EXHALE) or previous not in (None, INHALE, EXHALE):
            return time_s, label
        kinds = (label,) if self.pending is not None else (INHALE, EXHALE)
        found = [(abs(at - time_s), at, kind) for kind in kinds if (at := self.snapper(time_s, kind)) is not None]
        if not found:
            return time_s, label
        _, at, kind = min(found)
        return float(at), kind

    def landing(self, time_s: float) -> float:
        return self.resolve(time_s)[0]

    def remove_near(self, time_s: float, reach_s: float = 1.5) -> None:
        """Remove the mark closest to ``time_s`` (a wrong click anywhere in the stretch), if one is within ``reach_s``."""

        if not self.marks:
            return
        index = min(range(len(self.marks)), key=lambda i: abs(self.marks[i][0] - time_s))
        if abs(self.marks[index][0] - time_s) <= reach_s:
            self.marks.pop(index)
            self.clicks.pop(index)

    def undo(self) -> None:
        if self.end_s is not None:
            self.end_s = None
        elif self.marks:
            self.marks.pop()
            self.clicks.pop()

    def next_class(self) -> int:
        return self.pending if self.pending is not None else next_label([label for _, label in self.marks])

    def segments(self) -> list[Segment]:
        return marks_to_segments(self.marks, self.end_s if self.end_s is not None else self.window_end_s)

    def clicked(self) -> list[float | None]:
        """Raw click time per segment, in the order of :meth:`segments`."""

        order = sorted(range(len(self.marks)), key=lambda i: self.marks[i][0])
        return [self.clicks[i] for i in order][: len(self.segments())]

    def status(self) -> str:
        if self.setting_end:
            return "Kliknij koniec oznaczonego fragmentu"
        return f"Następny odcinek: {NAMES_PL[self.next_class()]}   (e wydech, i wdech, h pauza, n szum, . koniec, Cmd/Ctrl+Z lub Backspace cofnij, prawy klik usuwa znacznik, s zapisz)"


def scaled_to(signal: np.ndarray, trace: np.ndarray) -> np.ndarray:
    """``signal`` rescaled to the 1-99 percentile range of ``trace``, to draw both on one axis."""

    s_lo, s_hi = np.percentile(signal, [1, 99])
    t_lo, t_hi = np.percentile(trace, [1, 99])
    return t_lo + (signal - s_lo) / (s_hi - s_lo + 1e-12) * (t_hi - t_lo)


def draw(
    ax,
    grid: np.ndarray,
    trace: np.ndarray,
    session: AnnotationSession,
    xlim: tuple[float, float] | None = None,
    orientation: np.ndarray | None = None,
    ylabel: str = "ruch telefonu (rośnie na wdechu)",
) -> None:
    ax.clear()
    left, right = xlim if xlim is not None else (session.window_start_s, session.window_end_s)
    inside = (grid >= min(left, session.window_start_s)) & (grid <= max(right, session.window_end_s))
    for segment in session.segments():
        ax.axvspan(segment.start_s, segment.end_s, color=CLASS_COLOURS[segment.label], alpha=0.3, lw=0)
    for time_s, _ in session.marks:
        ax.axvline(time_s, color="#374151", lw=0.8, ls=(0, (3, 3)))
    if orientation is not None:
        ax.plot(grid[inside], orientation[inside], color="#0891b2", lw=0.9, alpha=0.85)
    ax.plot(grid[inside], trace[inside], color="#ea580c", lw=1.3)
    ax.set_xlim(left, right)
    ax.set_xlabel("czas od startu nagrania [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(session.status(), loc="left", fontsize=10)
    ax.grid(alpha=0.25)


def output_path(session_name: str, run: str, annotator: str, start_s: float, end_s: float, pass_number: int) -> Path:
    return ANNOTATIONS / f"{session_name}__{run}__{annotator}__{start_s:.0f}-{end_s:.0f}__pass{pass_number}.csv"


def belt_run_name(recording: Path) -> str:
    """The belt labeller's name for a recording: session files share run names, so they carry their session's."""
    return f"{recording.parent.name}__{recording.stem}" if recording.stem.startswith("run_") else recording.stem


def belt_trace(recording: Path, grid_hz: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """``(grid, belt chest trace)`` on the time axis of ``tools/label_strain_belt_runs.py`` (from the pattern start when cued)."""

    from respi_net.chest_signal import strain_belt_chest_signal
    from respi_net.strain_belt import cue_origin_ms, cue_sidecar_path, load_coach_cues

    cued = cue_sidecar_path(recording).exists() or recording.stem.startswith("strain_belt_coach_")
    cues = load_coach_cues(recording) if cued else None
    signal = strain_belt_chest_signal(recording, origin_ms=cue_origin_ms(cues) if cues else None)
    grid = np.arange(float(signal.time_s[0]), float(signal.time_s[-1]), 1.0 / grid_hz)
    return grid, np.interp(grid, signal.time_s, signal.chest)


def make_snapper(grid: np.ndarray, trace: np.ndarray, reach_s: float = 0.8):
    """``(time, label) -> time of the nearest turning point`` on the trace smoothed to 1 Hz, the system's boundary convention.

    ``None`` when the extreme within reach sits at the edge of the reach: that is a slope, not a turn.
    """

    smooth = light_filter(np.asarray(trace, dtype=float), 1.0 / float(np.median(np.diff(grid))), 1.0)
    step = float(np.median(np.diff(grid)))

    def snap(time_s: float, label: int) -> float | None:
        at = turning_point(grid, smooth, time_s, label, reach_s=reach_s)
        return None if abs(at - time_s) >= reach_s - 1.5 * step else at

    return snap


def run_viewer(args: argparse.Namespace) -> int:
    from respi_net.phone_annotation import oriented_phone
    from respi_net.phone_orientation import run_orientation_distance

    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    for name in list(plt.rcParams):
        if name.startswith("keymap."):
            plt.rcParams[name] = []
    if args.belt is not None:
        grid, trace = belt_trace(args.belt)
        orientation, sign, ylabel = None, 1.0, "pas tensometryczny (rośnie na wdechu)"
        name = belt_run_name(args.belt)
    else:
        prefix = SESSIONS / args.session / args.run
        grid, trace, sign = oriented_phone(prefix)
        orientation = None if args.no_orientation else scaled_to(run_orientation_distance(prefix, grid, trace), trace)
        ylabel = "ruch telefonu (rośnie na wdechu)"
        name = f"{args.session}__{args.run}"
    start, end = max(float(args.start), float(grid[0])), min(float(args.start) + float(args.length), float(grid[-1]))
    session = AnnotationSession(start, end, None if args.no_snap else make_snapper(grid, trace))
    path = ANNOTATIONS / f"{name}__{args.annotator}__{start:.0f}-{end:.0f}__pass{args.pass_number}.csv"
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.9)
    view = [start, end]
    hover: list = []

    def save() -> None:
        save_segments(path, session.segments(), None if args.no_snap else session.clicked())
        print(f"Zapisano {path}")

    def redraw() -> None:
        hover.clear()
        draw(ax, grid, trace, session, (view[0], view[1]), orientation, ylabel)
        fig.canvas.draw_idle()

    def zoom(factor: float, centre: float | None) -> None:
        centre = 0.5 * (view[0] + view[1]) if centre is None else centre
        low = max(start, centre - (centre - view[0]) * factor)
        high = min(end, centre + (view[1] - centre) * factor)
        if high - low >= 3.0:
            view[0], view[1] = low, high

    def on_key(event) -> None:
        if event.key == "s":
            save()
            return
        if event.key in ("+", "=", "-", "_"):
            zoom(0.5 if event.key in ("+", "=") else 2.0, event.xdata)
        elif event.key in ("left", "right"):
            shift = (view[1] - view[0]) * 0.25 * (-1 if event.key == "left" else 1)
            low = min(max(start, view[0] + shift), end - (view[1] - view[0]))
            view[0], view[1] = low, low + (view[1] - view[0])
        else:
            session.key(event.key)
        redraw()

    def on_move(event) -> None:
        toolbar = getattr(fig.canvas, "toolbar", None)
        for artist in hover:
            artist.remove()
        hover.clear()
        if event.inaxes is not ax or event.xdata is None or (toolbar is not None and getattr(toolbar, "mode", "")):
            fig.canvas.draw_idle()
            return
        landing, label = session.resolve(float(event.xdata))
        colour = CLASS_COLOURS[label]
        hover.append(ax.axvline(landing, color=colour, lw=1.4))
        hover.append(ax.text(landing, ax.get_ylim()[1], f" {NAMES_PL[label]}", color=colour, va="top", fontsize=9))
        hover.append(ax.scatter([landing], [float(np.interp(landing, grid, trace))], s=60, facecolor="white", edgecolor=colour, zorder=5))
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        toolbar = getattr(fig.canvas, "toolbar", None)
        if event.inaxes is not ax or event.xdata is None or (toolbar is not None and getattr(toolbar, "mode", "")):
            return
        if event.button == 3:
            session.remove_near(float(event.xdata))
        elif event.button == 1:
            session.click(float(event.xdata))
        else:
            return
        redraw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("close_event", lambda _event: save() if session.marks else None)
    redraw()
    if args.belt is not None:
        print(f"Pas: {name}, {start:.0f}-{end:.0f} s. Pauza = bezruch od 2 s. Zapis: {path}")
    else:
        print(f"Faza telefonu: {args.session}/{args.run}, {start:.0f}-{end:.0f} s (znak osi {sign:+.0f}). Zapis: {path}")
    print("Granice wdech/wydech przyciągają się do zwrotu przebiegu; linia i kółko w kolorze klasy pokazują, gdzie i jaki znacznik wyląduje."
          if not args.no_snap else "Bez przyciągania do zwrotów (--no-snap).")
    plt.show()
    return 0


def run_demo(figure: Path) -> int:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.arange(0.0, 60.0, 0.05)
    trace = np.sin(2 * np.pi * grid / 5.0) * np.where((grid > 25) & (grid < 38), 0.0, 1.0)
    session = AnnotationSession(0.0, 60.0)
    # the trace rises from a trough at 3.75 s to a peak at 6.25 s: an inhale starts at a trough, an exhale at a peak
    for step in ("click:3.75", "click:6.25", "click:8.75", "click:11.25", "click:13.75", "click:16.25", "click:18.75", "click:21.25",
                 "click:23.75", "key:h", "click:25.0", "click:38.0", "key:.", "click:48.75"):
        kind, value = step.split(":", 1)
        session.key(value) if kind == "key" else session.click(float(value))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.9)
    draw(ax, grid, trace, session)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=110)
    print(f"Demo: {figure}; segments: {[(round(s.start_s, 1), round(s.end_s, 1), NAMES_PL[s.label]) for s in session.segments()]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", help="folder name under data/raw/nn/a121_iphone")
    parser.add_argument("--run", help="run stem, for example run_01_nn_self_3min")
    parser.add_argument("--start", type=float, default=0.0, help="window start [s]")
    parser.add_argument("--length", type=float, default=60.0, help="window length [s]")
    parser.add_argument("--annotator", default="maciek")
    parser.add_argument("--pass", dest="pass_number", type=int, default=1, help="1 for the first labelling, 2 for the repeat")
    parser.add_argument("--no-snap", action="store_true", help="place every mark exactly where you click (no pull to the turning point)")
    parser.add_argument("--no-orientation", action="store_true", help="hide the orientation line (teal)")
    parser.add_argument("--belt", type=Path, default=None, help="label a strain-belt recording (..._belt.csv or a coach file) instead of a phone run")
    parser.add_argument("--demo", type=Path, default=None, help="draw a scripted example to this png and exit")
    args = parser.parse_args()
    if args.demo is not None:
        return run_demo(args.demo)
    if args.belt is None and (not args.session or not args.run):
        parser.error("--session and --run (or --belt) are required")
    return run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
