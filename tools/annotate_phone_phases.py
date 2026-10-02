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

A hold is a stillness of at least 4 s (natural rests between breaths last 1-3 s and belong to the breath).  The first click is the
start of the first segment.  Use the toolbar to zoom; clicks do not count while a toolbar tool is active.

    uv run python tools/annotate_phone_phases.py --session self_lying_3min_01 --run run_01_nn_self_3min --start 40 --length 60

Saves ``annotations/<session>__<run>__<annotator>__<start>-<end>__pass<N>.csv``.  Label the same stretch again a
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
        label = self.next_class()
        self.marks.append((self.landing(time_s, label), label))
        self.clicks.append(time_s)
        self.pending = None

    def landing(self, time_s: float, label: int | None = None) -> float:
        """Where a click at ``time_s`` lands: at the nearest turning point for a boundary that starts the other breath."""

        label = self.next_class() if label is None else label
        previous = self.marks[-1][1] if self.marks else None
        opposite = {INHALE: EXHALE, EXHALE: INHALE}.get(label)
        if self.snapper is not None and label in (INHALE, EXHALE) and (previous is None or previous == opposite):
            return float(self.snapper(time_s, label))
        return time_s

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


def draw(ax, grid: np.ndarray, trace: np.ndarray, session: AnnotationSession, xlim: tuple[float, float] | None = None) -> None:
    ax.clear()
    left, right = xlim if xlim is not None else (session.window_start_s, session.window_end_s)
    inside = (grid >= min(left, session.window_start_s)) & (grid <= max(right, session.window_end_s))
    for segment in session.segments():
        ax.axvspan(segment.start_s, segment.end_s, color=CLASS_COLOURS[segment.label], alpha=0.3, lw=0)
    for time_s, _ in session.marks:
        ax.axvline(time_s, color="#374151", lw=0.8, ls=(0, (3, 3)))
    ax.plot(grid[inside], trace[inside], color="#ea580c", lw=1.3)
    ax.set_xlim(left, right)
    ax.set_xlabel("czas od startu nagrania [s]")
    ax.set_ylabel("ruch telefonu (rośnie na wdechu)")
    ax.set_title(session.status(), loc="left", fontsize=10)
    ax.grid(alpha=0.25)


def output_path(session_name: str, run: str, annotator: str, start_s: float, end_s: float, pass_number: int) -> Path:
    return ANNOTATIONS / f"{session_name}__{run}__{annotator}__{start_s:.0f}-{end_s:.0f}__pass{pass_number}.csv"


def make_snapper(grid: np.ndarray, trace: np.ndarray):
    """``(time, label) -> time of the nearest turning point`` on the trace smoothed to 1 Hz, the system's boundary convention."""

    smooth = light_filter(np.asarray(trace, dtype=float), 1.0 / float(np.median(np.diff(grid))), 1.0)
    return lambda time_s, label: turning_point(grid, smooth, time_s, label)


def run_viewer(args: argparse.Namespace) -> int:
    from respi_net.phone_annotation import oriented_phone

    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    for name in list(plt.rcParams):
        if name.startswith("keymap."):
            plt.rcParams[name] = []
    prefix = SESSIONS / args.session / args.run
    grid, trace, sign = oriented_phone(prefix)
    start, end = float(args.start), min(float(args.start) + float(args.length), float(grid[-1]))
    session = AnnotationSession(start, end, None if args.no_snap else make_snapper(grid, trace))
    path = output_path(args.session, args.run, args.annotator, start, end, args.pass_number)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.9)
    view = [start, end]
    hover: list = []

    def save() -> None:
        save_segments(path, session.segments(), None if args.no_snap else session.clicked())
        print(f"Zapisano {path}")

    def redraw() -> None:
        hover.clear()
        draw(ax, grid, trace, session, (view[0], view[1]))
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
        landing = session.landing(float(event.xdata))
        hover.append(ax.axvline(landing, color="#2563eb", lw=1.2))
        hover.append(ax.scatter([landing], [float(np.interp(landing, grid, trace))], s=60, facecolor="white", edgecolor="#2563eb", zorder=5))
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
    print(f"Faza telefonu: {args.session}/{args.run}, {start:.0f}-{end:.0f} s (znak osi {sign:+.0f}). Zapis: {path}")
    print("Granice wdech/wydech przyciągają się do zwrotu przebiegu; niebieska linia i kółko pokazują, gdzie wyląduje znacznik."
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
    parser.add_argument("--demo", type=Path, default=None, help="draw a scripted example to this png and exit")
    args = parser.parse_args()
    if args.demo is not None:
        return run_demo(args.demo)
    if not args.session or not args.run:
        parser.error("--session and --run are required")
    return run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
