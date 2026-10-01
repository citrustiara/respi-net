#!/usr/bin/env python3
"""Mark breathing phases by hand on a phone IMU trace (the reference for the radar network).

You see the phone trace only: no radar, no detector output.  Click where a new phase starts; the class of that
phase is the one you pressed a key for just before the click, otherwise the next one in the usual order
(inhale, exhale, inhale, ...; after a hold, the breath that was not held).

    e wydech (exhale)     i wdech (inhale)     h pauza (hold)     n szum (noise: movement, cough)
    .  then click         end of the annotated stretch (otherwise the window end)
    Backspace             undo the last click      s   save (also on closing the window)

A hold is a stillness of at least 1.5 s; a shorter rest belongs to the breath around it.  The first click is the
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
from respi_net.phone_annotation import KEY_TO_CLASS, MIN_HOLD_S, Segment, marks_to_segments, next_label, save_segments

SESSIONS = ROOT / "data" / "raw" / "nn" / "a121_iphone"
ANNOTATIONS = ROOT / "annotations"
NAMES_PL = {EXHALE: "wydech", HOLD: "pauza", INHALE: "wdech", NOISE: "szum"}


class AnnotationSession:
    """The clicks and keys of one annotation, without any plotting."""

    def __init__(self, window_start_s: float, window_end_s: float) -> None:
        self.window_start_s = window_start_s
        self.window_end_s = window_end_s
        self.marks: list[tuple[float, int]] = []
        self.pending: int | None = None
        self.end_s: float | None = None
        self.setting_end = False

    def key(self, key: str) -> None:
        if key in KEY_TO_CLASS:
            self.pending = KEY_TO_CLASS[key]
            self.setting_end = False
        elif key == ".":
            self.setting_end = True
        elif key == "backspace":
            self.undo()

    def click(self, time_s: float) -> None:
        time_s = float(min(max(time_s, self.window_start_s), self.window_end_s))
        if self.setting_end:
            self.end_s = time_s
            self.setting_end = False
            return
        self.marks.append((time_s, self.next_class()))
        self.pending = None

    def undo(self) -> None:
        if self.end_s is not None:
            self.end_s = None
        elif self.marks:
            self.marks.pop()

    def next_class(self) -> int:
        return self.pending if self.pending is not None else next_label([label for _, label in self.marks])

    def segments(self) -> list[Segment]:
        return marks_to_segments(self.marks, self.end_s if self.end_s is not None else self.window_end_s)

    def status(self) -> str:
        if self.setting_end:
            return "Kliknij koniec oznaczonego fragmentu"
        return f"Następny odcinek: {NAMES_PL[self.next_class()]}   (e wydech, i wdech, h pauza, n szum, . koniec, Backspace cofnij, s zapisz)"


def draw(ax, grid: np.ndarray, trace: np.ndarray, session: AnnotationSession) -> None:
    ax.clear()
    inside = (grid >= session.window_start_s) & (grid <= session.window_end_s)
    for segment in session.segments():
        ax.axvspan(segment.start_s, segment.end_s, color=CLASS_COLOURS[segment.label], alpha=0.3, lw=0)
    for time_s, _ in session.marks:
        ax.axvline(time_s, color="#374151", lw=0.8, ls=(0, (3, 3)))
    ax.plot(grid[inside], trace[inside], color="#ea580c", lw=1.3)
    ax.set_xlim(session.window_start_s, session.window_end_s)
    ax.set_xlabel("czas od startu nagrania [s]")
    ax.set_ylabel("ruch telefonu (rośnie na wdechu)")
    ax.set_title(session.status(), loc="left", fontsize=10)
    ax.grid(alpha=0.25)


def output_path(session_name: str, run: str, annotator: str, start_s: float, end_s: float, pass_number: int) -> Path:
    return ANNOTATIONS / f"{session_name}__{run}__{annotator}__{start_s:.0f}-{end_s:.0f}__pass{pass_number}.csv"


def run_viewer(args: argparse.Namespace) -> int:
    from respi_net.phone_annotation import oriented_phone

    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
    for name in list(plt.rcParams):
        if name.startswith("keymap."):
            plt.rcParams[name] = []
    prefix = SESSIONS / args.session / args.run
    grid, trace, sign = oriented_phone(prefix)
    start, end = float(args.start), min(float(args.start) + float(args.length), float(grid[-1]))
    session = AnnotationSession(start, end)
    path = output_path(args.session, args.run, args.annotator, start, end, args.pass_number)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.9)

    def save() -> None:
        save_segments(path, session.segments())
        print(f"Zapisano {path}")

    def on_key(event) -> None:
        if event.key == "s":
            save()
            return
        session.key(event.key)
        draw(ax, grid, trace, session)
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        toolbar = getattr(fig.canvas, "toolbar", None)
        if event.inaxes is not ax or event.button != 1 or event.xdata is None or (toolbar is not None and getattr(toolbar, "mode", "")):
            return
        session.click(float(event.xdata))
        draw(ax, grid, trace, session)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", lambda _event: save() if session.marks else None)
    draw(ax, grid, trace, session)
    print(f"Faza telefonu: {args.session}/{args.run}, {start:.0f}-{end:.0f} s (znak osi {sign:+.0f}). Zapis: {path}")
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
    parser.add_argument("--demo", type=Path, default=None, help="draw a scripted example to this png and exit")
    args = parser.parse_args()
    if args.demo is not None:
        return run_demo(args.demo)
    if not args.session or not args.run:
        parser.error("--session and --run are required")
    return run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
