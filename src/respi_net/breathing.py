"""Breathing patterns: one file format for every guided-breathing screen.

A pattern is what the subject is asked to do, and for how long.  The
standalone coach (``respi coach``) plays any pattern file, and the guided
recording tools read their protocols from the same files, so the schedule a
subject follows and the schedule saved beside a recording come from one
definition instead of one hand-written loop per tool.

Pattern files live in ``configs/breathing_patterns/`` and are JSON objects
with a ``name``, an optional ``description`` and a list of ``blocks``,
expanded in order:

``{"kind": "normal", "duration_seconds": 10}``
    One phase.  ``kind`` is ``settle``, ``normal``, ``inhale``, ``exhale``,
    ``hold`` or ``interference``; ``cue``, ``detail`` and ``label`` replace the
    default wording.
``{"kind": "paced", "cycles": 7, "inhale_seconds": 2, "exhale_seconds": 3}``
    Breathing cycles.  ``hold_after_inhale_seconds`` and
    ``hold_after_exhale_seconds`` add a pause inside every cycle, which is all
    box breathing is; ``inhale_cue``, ``exhale_cue``, ``hold_cue`` and the
    matching ``*_detail`` keys replace the default wording.
``{"kind": "repeat", "times": 3, "blocks": [...]}``
    The nested blocks, ``times`` times over.

Every block other than ``repeat`` is one *section*: the stretch of the session
the coach names on its timeline ("12/min × 7", "Wstrzymanie 15 s").  Keys the
format does not know are an error rather than ignored, so a misspelt
``hold_after_inhale_second`` stops the file loading instead of quietly
producing a pattern without holds.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .paths import BREATHING_PATTERNS_DIR, PROJECT_ROOT

PHASE_KINDS = ("settle", "normal", "inhale", "exhale", "hold", "interference")

DEFAULT_CUES = {
    "settle": "USPOKÓJ POZYCJĘ",
    "normal": "ODDYCHAJ SWOBODNIE",
    "inhale": "WDECH",
    "exhale": "WYDECH",
    "hold": "WSTRZYMAJ ODDECH",
    "interference": "PUSTA I NIERUCHOMA SCENA",
}

# What a one-phase section is called on the timeline, before its length.
SECTION_NAMES = {
    "settle": "Uspokojenie",
    "normal": "Swobodnie",
    "inhale": "Wdech",
    "exhale": "Wydech",
    "hold": "Wstrzymanie",
    "interference": "Pusta scena",
}

PACED_HOLD_CUE = "ZATRZYMAJ"
HOLD_AFTER_INHALE_DETAIL = "Pełne płuca — nie wypuszczaj powietrza."
HOLD_AFTER_EXHALE_DETAIL = "Puste płuca — nie nabieraj powietrza."

MAX_NESTING = 8
MAX_PHASES = 20_000

_PATTERN_KEYS = frozenset({"name", "description", "blocks"})
_PHASE_KEYS = frozenset({"kind", "duration_seconds", "cue", "detail", "label"})
_PACED_KEYS = frozenset(
    {
        "kind",
        "cycles",
        "inhale_seconds",
        "exhale_seconds",
        "hold_after_inhale_seconds",
        "hold_after_exhale_seconds",
        "inhale_cue",
        "exhale_cue",
        "hold_cue",
        "inhale_detail",
        "exhale_detail",
        "hold_detail",
        "label",
    }
)
_REPEAT_KEYS = frozenset({"kind", "times", "blocks"})


class PatternError(ValueError):
    """A pattern that cannot be expanded; the message names the offending key."""


@dataclass(frozen=True)
class BreathPhase:
    """One instruction held for a fixed time, in seconds from the session start."""

    start_s: float
    end_s: float
    kind: str
    cue: str
    detail: str
    section: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class PatternSection:
    """A run of phases named as one stretch of the session."""

    index: int
    start_s: float
    end_s: float
    first_phase: int
    stop_phase: int
    kind: str
    label: str
    short_label: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class CoachPosition:
    """Everything a live display needs about one moment of a pattern."""

    elapsed_s: float
    index: int
    phase: BreathPhase
    remaining_s: float
    fraction: float
    next_phase: BreathPhase | None
    section: PatternSection
    next_section: PatternSection | None
    finished: bool

    @property
    def until_next_section_s(self) -> float | None:
        if self.next_section is None:
            return None
        return max(0.0, self.next_section.start_s - self.elapsed_s)


@dataclass(frozen=True)
class BreathingPattern:
    pattern_id: str
    name: str
    description: str
    phases: tuple[BreathPhase, ...]
    sections: tuple[PatternSection, ...]
    source: Path | None = None

    @classmethod
    def from_phases(
        cls,
        phases: Iterable[BreathPhase],
        *,
        name: str = "",
        pattern_id: str = "",
        description: str = "",
    ) -> BreathingPattern:
        """Wrap phases built elsewhere, naming sections from the phases alone."""

        phases = tuple(phases)
        return cls(pattern_id, name, description, phases, build_sections(phases))

    @property
    def duration_s(self) -> float:
        return self.phases[-1].end_s if self.phases else 0.0

    @cached_property
    def _phase_ends(self) -> list[float]:
        return [phase.end_s for phase in self.phases]

    @cached_property
    def _section_starts(self) -> list[int]:
        return [section.first_phase for section in self.sections]

    def phase_index_at(self, elapsed_s: float) -> int | None:
        """Index of the phase running at ``elapsed_s``; see :func:`cue_at`."""

        if not self.phases:
            return None
        return min(bisect_right(self._phase_ends, max(0.0, float(elapsed_s))), len(self.phases) - 1)

    def section_index_of(self, phase_index: int) -> int:
        return max(0, bisect_right(self._section_starts, phase_index) - 1)

    def position(self, elapsed_s: float) -> CoachPosition | None:
        index = self.phase_index_at(elapsed_s)
        if index is None:
            return None
        elapsed = min(max(0.0, float(elapsed_s)), self.duration_s)
        phase = self.phases[index]
        section_index = self.section_index_of(index)
        length = phase.duration_s
        fraction = 1.0 if length <= 0 else min(1.0, max(0.0, (elapsed - phase.start_s) / length))
        return CoachPosition(
            elapsed_s=elapsed,
            index=index,
            phase=phase,
            remaining_s=max(0.0, phase.end_s - elapsed),
            fraction=fraction,
            next_phase=self.phases[index + 1] if index + 1 < len(self.phases) else None,
            section=self.sections[section_index],
            next_section=self.sections[section_index + 1] if section_index + 1 < len(self.sections) else None,
            finished=elapsed >= self.duration_s,
        )


def format_seconds(seconds: float) -> str:
    """``2`` rather than ``2.0``, ``2.5`` rather than ``2.50``."""

    return f"{seconds:.2f}".rstrip("0").rstrip(".")


def format_clock(seconds: float, *, round_up: bool = False) -> str:
    """``m:ss``; elapsed time rounds down, time left rounds up."""

    whole = math.ceil(seconds - 1e-6) if round_up else math.floor(seconds + 1e-6)
    whole = max(0, int(whole))
    return f"{whole // 60}:{whole % 60:02d}"


def _format_rate(per_minute: float) -> str:
    return f"{per_minute:.1f}".rstrip("0").rstrip(".")


def _describe_section(members: Sequence[BreathPhase]) -> tuple[str, str, str]:
    """``(kind, label, short label)`` for one section's phases.

    The short label is what still fits a narrow stretch of the timeline: the
    rhythm of a run of cycles, or the name of a single phase.
    """

    first = members[0]
    length = members[-1].end_s - first.start_s
    kinds = {phase.kind for phase in members}
    cycles = sum(1 for phase in members if phase.kind == "inhale")
    if len(members) > 1 and cycles and kinds <= {"inhale", "exhale", "hold"}:
        per_cycle, leftover = divmod(len(members), cycles)
        if "hold" in kinds and not leftover:
            rhythm = "-".join(format_seconds(phase.duration_s) for phase in members[:per_cycle]) + " s"
        else:
            rhythm = f"{_format_rate(60.0 * cycles / length)}/min"
        return "paced", f"{rhythm} × {cycles}", rhythm
    name = SECTION_NAMES.get(first.kind) or first.cue.capitalize() or first.kind
    return first.kind, f"{name} {format_seconds(length)} s", name


def build_sections(
    phases: Sequence[BreathPhase],
    labels: Sequence[str | None] = (),
) -> tuple[PatternSection, ...]:
    """Group consecutive phases sharing a ``section`` id; ``labels`` override names by id."""

    sections: list[PatternSection] = []
    first = 0
    for stop in range(1, len(phases) + 1):
        if stop < len(phases) and phases[stop].section == phases[first].section:
            continue
        members = phases[first:stop]
        kind, label, short_label = _describe_section(members)
        section_id = members[0].section
        explicit = labels[section_id] if 0 <= section_id < len(labels) else None
        sections.append(
            PatternSection(
                index=len(sections),
                start_s=members[0].start_s,
                end_s=members[-1].end_s,
                first_phase=first,
                stop_phase=stop,
                kind=kind,
                label=explicit or label,
                short_label=explicit or short_label,
            )
        )
        first = stop
    return tuple(sections)


def _check_keys(raw: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(str(key) for key in raw if key not in allowed)
    if unknown:
        listed = ", ".join(repr(key) for key in unknown)
        raise PatternError(f"{where}: unknown key {listed}; allowed: {', '.join(sorted(allowed))}.")


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PatternError(f"{where} must be a number of seconds.")
    number = float(value)
    if not math.isfinite(number):
        raise PatternError(f"{where} must be finite.")
    return number


def _seconds(value: Any, where: str) -> float:
    if value is None:
        raise PatternError(f"{where} is required.")
    seconds = _number(value, where)
    if seconds <= 0:
        raise PatternError(f"{where} must be positive.")
    return seconds


def _optional_seconds(value: Any, where: str) -> float:
    if value is None:
        return 0.0
    seconds = _number(value, where)
    if seconds < 0:
        raise PatternError(f"{where} cannot be negative.")
    return seconds


def _count(value: Any, where: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float(value).is_integer()
        or value < 1
    ):
        raise PatternError(f"{where} must be a whole number of at least 1.")
    return int(value)


def _text(raw: Mapping[str, Any], key: str, default: str, where: str) -> str:
    value = raw.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise PatternError(f"{where}.{key} must be text.")
    return value.strip() or default


class _Expander:
    def __init__(self) -> None:
        self.phases: list[BreathPhase] = []
        self.labels: list[str | None] = []
        self.cursor = 0.0

    def blocks(self, blocks: Any, where: str, depth: int) -> None:
        if not isinstance(blocks, list) or not blocks:
            raise PatternError(f"{where} must be a non-empty list of blocks.")
        for index, raw in enumerate(blocks):
            self.block(raw, f"{where}[{index}]", depth)

    def block(self, raw: Any, where: str, depth: int) -> None:
        if not isinstance(raw, Mapping):
            raise PatternError(f"{where} must be an object with a 'kind'.")
        kind = raw.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise PatternError(f"{where} needs a 'kind'.")
        kind = kind.strip().lower()
        if kind == "repeat":
            _check_keys(raw, _REPEAT_KEYS, where)
            if depth >= MAX_NESTING:
                raise PatternError(f"{where}: repeat blocks nest more than {MAX_NESTING} levels deep.")
            times = _count(raw.get("times"), f"{where}.times")
            for _ in range(times):
                self.blocks(raw.get("blocks"), f"{where}.blocks", depth + 1)
        elif kind == "paced":
            self.paced(raw, where)
        elif kind in DEFAULT_CUES:
            _check_keys(raw, _PHASE_KEYS, where)
            seconds = _seconds(raw.get("duration_seconds"), f"{where}.duration_seconds")
            self.section(raw, where)
            self.phase(kind, seconds, _text(raw, "cue", DEFAULT_CUES[kind], where), _text(raw, "detail", "", where))
        else:
            raise PatternError(f"{where}.kind {kind!r} is not one of: paced, repeat, {', '.join(PHASE_KINDS)}.")

    def paced(self, raw: Mapping[str, Any], where: str) -> None:
        _check_keys(raw, _PACED_KEYS, where)
        cycles = _count(raw.get("cycles"), f"{where}.cycles")
        inhale_s = _seconds(raw.get("inhale_seconds"), f"{where}.inhale_seconds")
        exhale_s = _seconds(raw.get("exhale_seconds"), f"{where}.exhale_seconds")
        hold_in_s = _optional_seconds(raw.get("hold_after_inhale_seconds"), f"{where}.hold_after_inhale_seconds")
        hold_out_s = _optional_seconds(raw.get("hold_after_exhale_seconds"), f"{where}.hold_after_exhale_seconds")
        inhale_cue = _text(raw, "inhale_cue", DEFAULT_CUES["inhale"], where)
        exhale_cue = _text(raw, "exhale_cue", DEFAULT_CUES["exhale"], where)
        hold_cue = _text(raw, "hold_cue", PACED_HOLD_CUE, where)
        inhale_detail = _text(raw, "inhale_detail", f"Spokojny wdech przez {format_seconds(inhale_s)} s.", where)
        exhale_detail = _text(raw, "exhale_detail", f"Spokojny wydech przez {format_seconds(exhale_s)} s.", where)
        hold_detail = _text(raw, "hold_detail", "", where)
        self.section(raw, where)
        for _ in range(cycles):
            self.phase("inhale", inhale_s, inhale_cue, inhale_detail)
            if hold_in_s > 0:
                self.phase("hold", hold_in_s, hold_cue, hold_detail or HOLD_AFTER_INHALE_DETAIL)
            self.phase("exhale", exhale_s, exhale_cue, exhale_detail)
            if hold_out_s > 0:
                self.phase("hold", hold_out_s, hold_cue, hold_detail or HOLD_AFTER_EXHALE_DETAIL)

    def section(self, raw: Mapping[str, Any], where: str) -> None:
        self.labels.append(_text(raw, "label", "", where) or None)

    def phase(self, kind: str, seconds: float, cue: str, detail: str) -> None:
        if len(self.phases) >= MAX_PHASES:
            raise PatternError(f"The pattern expands to more than {MAX_PHASES} phases.")
        start = self.cursor
        self.phases.append(BreathPhase(start, start + seconds, kind, cue, detail, len(self.labels) - 1))
        self.cursor += seconds


def expand_blocks(
    blocks: Any,
    *,
    where: str = "blocks",
) -> tuple[tuple[BreathPhase, ...], tuple[str | None, ...]]:
    """Phases for a list of blocks, plus each section's explicit label (or ``None``)."""

    expander = _Expander()
    expander.blocks(blocks, where, 0)
    return tuple(expander.phases), tuple(expander.labels)


def pattern_from_blocks(
    blocks: Any,
    *,
    name: str = "",
    pattern_id: str = "",
    where: str = "blocks",
) -> BreathingPattern:
    """A pattern from a bare block list, e.g. one embedded in an experiment config."""

    phases, labels = expand_blocks(blocks, where=where)
    return BreathingPattern(pattern_id, name, "", phases, build_sections(phases, labels))


def pattern_from_dict(
    payload: Any,
    *,
    pattern_id: str = "",
    source: Path | None = None,
) -> BreathingPattern:
    if not isinstance(payload, Mapping):
        raise PatternError("A breathing pattern must be a JSON object with 'blocks'.")
    _check_keys(payload, _PATTERN_KEYS, "pattern")
    name = payload.get("name", "")
    description = payload.get("description", "")
    if not isinstance(name, str) or not isinstance(description, str):
        raise PatternError("pattern: 'name' and 'description' must be text.")
    phases, labels = expand_blocks(payload.get("blocks"))
    return BreathingPattern(
        pattern_id=pattern_id,
        name=name.strip() or pattern_id or "Wzorzec oddechu",
        description=description.strip(),
        phases=phases,
        sections=build_sections(phases, labels),
        source=source,
    )


def pattern_path(name_or_path: str | Path, directory: Path = BREATHING_PATTERNS_DIR) -> Path:
    """Where a pattern named on the command line or in a config lives.

    A bare name is a file in ``directory``; anything that looks like a path is
    one, relative to the working directory or else to the repository root.
    """

    candidate = Path(name_or_path).expanduser()
    bare = len(candidate.parts) == 1
    if bare and candidate.suffix.lower() != ".json":
        return directory / f"{candidate.name}.json"
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if bare and (directory / candidate.name).is_file():
        return directory / candidate.name
    return PROJECT_ROOT / candidate


def load_pattern(name_or_path: str | Path, directory: Path = BREATHING_PATTERNS_DIR) -> BreathingPattern:
    path = pattern_path(name_or_path, directory)
    if not path.is_file():
        known = ", ".join(item.stem for item in sorted(directory.glob("*.json"))) or "none"
        raise PatternError(f"No breathing pattern {str(name_or_path)!r} at {path}. Available: {known}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PatternError(f"{path}: {exc.strerror or exc}.") from None
    except json.JSONDecodeError as exc:
        raise PatternError(f"{path.name}: not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}.") from None
    try:
        return pattern_from_dict(payload, pattern_id=path.stem, source=path.resolve())
    except PatternError as exc:
        raise PatternError(f"{path.name}: {exc}") from None


def discover_patterns(
    directory: Path = BREATHING_PATTERNS_DIR,
) -> tuple[list[BreathingPattern], list[tuple[Path, str]]]:
    """Every pattern file in ``directory``, and the ones that failed to load with why."""

    patterns: list[BreathingPattern] = []
    problems: list[tuple[Path, str]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            patterns.append(load_pattern(path, directory))
        except PatternError as exc:
            problems.append((path, str(exc)))
    return patterns, problems


def cue_at(cues: Sequence[Any], elapsed_s: float) -> tuple[Any, float] | None:
    """The cue running at ``elapsed_s`` and the seconds left in it.

    Works on anything with an ``end_s`` -- the recording tools' own cue records
    as well as :class:`BreathPhase`.  A moment exactly on a boundary belongs to
    the cue starting there; past the end it is the last cue, with nothing left.
    """

    if not cues:
        return None
    elapsed = max(0.0, float(elapsed_s))
    index = min(bisect_right([cue.end_s for cue in cues], elapsed), len(cues) - 1)
    cue = cues[index]
    return cue, max(0.0, cue.end_s - elapsed)


def phases_from_cues(cues: Iterable[Any]) -> tuple[BreathPhase, ...]:
    """Phases for the coach from a recording tool's cue records.

    The records carry the cue text as ``cue`` or ``title`` and no section, so
    one is inferred: a run of inhales and exhales is one stretch of paced
    breathing and anything else stands alone, which is how every fixed
    recording protocol is built.
    """

    phases: list[BreathPhase] = []
    section = -1
    previous_kind: str | None = None
    for cue in cues:
        kind = str(cue.kind)
        if not (kind in ("inhale", "exhale") and previous_kind in ("inhale", "exhale")):
            section += 1
        text = getattr(cue, "cue", None) or getattr(cue, "title", "")
        phases.append(
            BreathPhase(
                start_s=float(cue.start_s),
                end_s=float(cue.end_s),
                kind=kind,
                cue=str(text),
                detail=str(getattr(cue, "detail", "") or ""),
                section=section,
            )
        )
        previous_kind = kind
    return tuple(phases)
