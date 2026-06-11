"""Small Garmin FIT reader for wellness/sleep reference exports.

This is intentionally dependency-free.  It implements the FIT record/definition layer and extracts
only the wellness data needed by the A121 sleep comparison tooling:

- monitoring heart-rate samples (global 55, timestamp_16 + heart_rate)
- respiration-rate samples (global 297)
- sleep level changes (global 275)
- sleep start/stop events (global 21, event 74 in Garmin sleep exports)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import struct
from typing import Any, Iterable

import pandas as pd

FIT_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)

# name, byte-size, struct format, invalid value
_BASE_TYPES: dict[int, tuple[str, int, str, int | None]] = {
    0x00: ("enum", 1, "B", 0xFF),
    0x01: ("sint8", 1, "b", 0x7F),
    0x02: ("uint8", 1, "B", 0xFF),
    0x83: ("sint16", 2, "h", 0x7FFF),
    0x84: ("uint16", 2, "H", 0xFFFF),
    0x85: ("sint32", 4, "i", 0x7FFFFFFF),
    0x86: ("uint32", 4, "I", 0xFFFFFFFF),
    0x07: ("string", 1, "s", 0x00),
    0x88: ("float32", 4, "f", 0xFFFFFFFF),
    0x89: ("float64", 8, "d", 0xFFFFFFFFFFFFFFFF),
    0x0A: ("uint8z", 1, "B", 0x00),
    0x8B: ("uint16z", 2, "H", 0x0000),
    0x8C: ("uint32z", 4, "I", 0x00000000),
    0x0D: ("byte", 1, "B", None),
    0x8E: ("sint64", 8, "q", 0x7FFFFFFFFFFFFFFF),
    0x8F: ("uint64", 8, "Q", 0xFFFFFFFFFFFFFFFF),
    0x90: ("uint64z", 8, "Q", 0x0000000000000000),
}


@dataclass(frozen=True)
class FitMessage:
    global_num: int
    fields: dict[int, Any]
    path: Path


@dataclass(frozen=True)
class _FieldDef:
    number: int
    size: int
    base_type: int


@dataclass(frozen=True)
class _DevFieldDef:
    number: int
    size: int
    developer_index: int


@dataclass(frozen=True)
class _Definition:
    endian: str
    global_num: int
    fields: tuple[_FieldDef, ...]
    developer_fields: tuple[_DevFieldDef, ...]


@dataclass(frozen=True)
class GarminReferenceData:
    heart_rate: pd.DataFrame
    respiration_rate: pd.DataFrame
    sleep_levels: pd.DataFrame
    sleep_events: pd.DataFrame


def fit_datetime(timestamp_s: int | float) -> datetime:
    """Convert a FIT date_time value to a timezone-aware UTC datetime."""

    return FIT_EPOCH + timedelta(seconds=float(timestamp_s))


def fit_datetime_local_naive(timestamp_s: int | float) -> datetime:
    """Convert FIT date_time seconds to the machine-local naive datetime used by plots."""

    return fit_datetime(timestamp_s).astimezone().replace(tzinfo=None)


def _base_info(base_type: int) -> tuple[str, int, str, int | None]:
    # Base types are normally exact values above.  The fallback keeps the parser tolerant of files
    # that set non-profile flag bits while preserving the low base-type id.
    return _BASE_TYPES.get(base_type) or _BASE_TYPES.get(base_type & 0xDF) or ("byte", 1, "B", None)


def _read_value(data: bytes, offset: int, size: int, base_type: int, endian: str) -> Any:
    name, unit_size, fmt, invalid = _base_info(base_type)
    raw = data[offset : offset + size]
    if name == "string":
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
    if name == "byte":
        values = list(raw)
        return values[0] if len(values) == 1 else values
    if unit_size <= 0 or size % unit_size != 0:
        return raw.hex()
    count = size // unit_size
    if unit_size == 1:
        values = list(raw)
        if fmt == "b":
            values = [struct.unpack("b", bytes([value]))[0] for value in values]
    else:
        values = list(struct.unpack(endian + (fmt * count), raw))
    cleaned: list[Any] = [None if invalid is not None and value == invalid else value for value in values]
    return cleaned[0] if count == 1 else cleaned


def parse_fit_file(path: Path) -> list[FitMessage]:
    """Parse a FIT file into generic messages keyed by global message and field numbers."""

    path = Path(path)
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError(f"{path} is too small to be a FIT file")
    header_size = data[0]
    if header_size < 12 or len(data) < header_size:
        raise ValueError(f"{path} has an invalid FIT header")
    data_size = struct.unpack_from("<I", data, 4)[0]
    if data[8:12] != b".FIT":
        raise ValueError(f"{path} does not contain FIT magic bytes")

    offset = header_size
    data_end = min(len(data), header_size + data_size)
    definitions: dict[int, _Definition] = {}
    messages: list[FitMessage] = []
    last_timestamp: int | None = None

    while offset < data_end:
        header = data[offset]
        offset += 1
        compressed_timestamp: int | None = None

        if header & 0x80:
            is_definition = False
            has_developer_fields = False
            local_num = (header >> 5) & 0x03
            time_offset = header & 0x1F
            if last_timestamp is not None:
                compressed_timestamp = (last_timestamp & ~0x1F) + time_offset
                if compressed_timestamp <= last_timestamp:
                    compressed_timestamp += 32
        else:
            is_definition = bool(header & 0x40)
            has_developer_fields = bool(header & 0x20)
            local_num = header & 0x0F

        if is_definition:
            if offset + 5 > data_end:
                raise ValueError(f"Truncated FIT definition in {path}")
            # reserved byte, architecture byte
            architecture = data[offset + 1]
            offset += 2
            endian = ">" if architecture else "<"
            global_num = struct.unpack_from(endian + "H", data, offset)[0]
            offset += 2
            field_count = data[offset]
            offset += 1
            fields: list[_FieldDef] = []
            for _ in range(field_count):
                if offset + 3 > data_end:
                    raise ValueError(f"Truncated FIT field definition in {path}")
                fields.append(_FieldDef(data[offset], data[offset + 1], data[offset + 2]))
                offset += 3
            developer_fields: list[_DevFieldDef] = []
            if has_developer_fields:
                developer_count = data[offset]
                offset += 1
                for _ in range(developer_count):
                    if offset + 3 > data_end:
                        raise ValueError(f"Truncated FIT developer field definition in {path}")
                    developer_fields.append(_DevFieldDef(data[offset], data[offset + 1], data[offset + 2]))
                    offset += 3
            definitions[local_num] = _Definition(endian, global_num, tuple(fields), tuple(developer_fields))
            continue

        definition = definitions.get(local_num)
        if definition is None:
            raise ValueError(f"FIT data message in {path} references undefined local message {local_num}")
        values: dict[int, Any] = {}
        for field in definition.fields:
            if offset + field.size > data_end:
                raise ValueError(f"Truncated FIT data field in {path}")
            values[field.number] = _read_value(data, offset, field.size, field.base_type, definition.endian)
            offset += field.size
        for field in definition.developer_fields:
            # Developer fields are not required for the Garmin sleep/wellness exports handled here.
            offset += field.size
        if compressed_timestamp is not None and 253 not in values:
            values[253] = compressed_timestamp
        timestamp = values.get(253)
        if isinstance(timestamp, (int, float)) and timestamp > 0:
            last_timestamp = int(timestamp)
        messages.append(FitMessage(definition.global_num, values, path))

    return messages


def parse_fit_files(paths: Iterable[Path]) -> list[FitMessage]:
    messages: list[FitMessage] = []
    for path in paths:
        messages.extend(parse_fit_file(path))
    return messages


def _valid_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _timestamp16_to_full(timestamp16: int | float, anchor: int | float | None) -> int | None:
    if anchor is None:
        return None
    low = int(timestamp16) & 0xFFFF
    anchor_i = int(anchor)
    candidate = (anchor_i & ~0xFFFF) + low
    while candidate < anchor_i - 32768:
        candidate += 65536
    while candidate > anchor_i + 32768:
        candidate -= 65536
    return candidate


def _dedupe_time_rows(rows: list[dict[str, Any]], value_col: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["time_utc", "local_time", value_col, "source_file"])
    df = pd.DataFrame(rows).sort_values("time_utc")
    df = df.drop_duplicates(subset=["time_utc", value_col], keep="last")
    return df.reset_index(drop=True)


def fit_paths_from_sources(sources: Path | Iterable[Path]) -> list[Path]:
    """Resolve one or more FIT files/directories to a sorted list of .fit paths."""

    if isinstance(sources, (str, Path)):
        source_list = [Path(sources)]
    else:
        source_list = [Path(source) for source in sources]
    paths: list[Path] = []
    for source in source_list:
        if source.is_dir():
            paths.extend(sorted(source.glob("*.fit")))
        elif source.is_file():
            paths.append(source)
        else:
            raise FileNotFoundError(f"Garmin FIT source does not exist: {source}")
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            out.append(path)
            seen.add(resolved)
    return sorted(out)


def extract_garmin_reference(fit_dir: Path | Iterable[Path]) -> GarminReferenceData:
    """Extract HR, RR, and sleep-level/event data from Garmin FIT file(s) or directory(s)."""

    paths = fit_paths_from_sources(fit_dir)
    heart_rows: list[dict[str, Any]] = []
    rr_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for path in paths:
        anchor_timestamp: int | None = None
        last_hr_timestamp: int | None = None
        for message in parse_fit_file(path):
            fields = message.fields
            timestamp = _valid_number(fields.get(253))
            if timestamp is not None:
                anchor_timestamp = int(timestamp)

            # Garmin stress-level messages in these wellness files carry their timestamp in field 1.
            if message.global_num == 227:
                stress_timestamp = _valid_number(fields.get(1))
                if stress_timestamp is not None:
                    anchor_timestamp = int(stress_timestamp)

            if message.global_num == 55:
                # FIT monitoring message.  Garmin stores minute-level optical HR as field 27 with
                # timestamp_16 in field 26 for compact wellness files.
                hr = _valid_number(fields.get(27))
                if hr is not None and 25.0 <= hr <= 220.0:
                    full_timestamp = _valid_number(fields.get(253))
                    if full_timestamp is None:
                        timestamp16 = _valid_number(fields.get(26))
                        full_timestamp = _timestamp16_to_full(timestamp16, last_hr_timestamp or anchor_timestamp) if timestamp16 is not None else None
                    if full_timestamp is not None:
                        full_i = int(full_timestamp)
                        heart_rows.append(
                            {
                                "time_utc": fit_datetime(full_i),
                                "local_time": fit_datetime_local_naive(full_i),
                                "garmin_hr_bpm": float(hr),
                                "source_file": path.name,
                            }
                        )
                        last_hr_timestamp = full_i
                        anchor_timestamp = full_i

            elif message.global_num == 297:
                # Garmin respiration-rate samples are in hundredths of breaths/minute.
                rr_raw = _valid_number(fields.get(0))
                if timestamp is not None and rr_raw is not None:
                    rr = rr_raw / 100.0
                    if 4.0 <= rr <= 40.0:
                        rr_rows.append(
                            {
                                "time_utc": fit_datetime(int(timestamp)),
                                "local_time": fit_datetime_local_naive(int(timestamp)),
                                "garmin_rr_bpm": float(rr),
                                "source_file": path.name,
                            }
                        )

            elif message.global_num == 275:
                sleep_level = _valid_number(fields.get(0))
                if timestamp is not None and sleep_level is not None:
                    level_rows.append(
                        {
                            "time_utc": fit_datetime(int(timestamp)),
                            "local_time": fit_datetime_local_naive(int(timestamp)),
                            "sleep_level_code": int(sleep_level),
                            "source_file": path.name,
                        }
                    )

            elif message.global_num == 21:
                # Sleep start/stop event in Garmin sleep FIT exports observed here: event 74,
                # event_type 0=start and 1=stop.
                event = _valid_number(fields.get(0))
                event_type = _valid_number(fields.get(1))
                if timestamp is not None and event is not None:
                    event_name = "sleep" if int(event) == 74 else f"event_{int(event)}"
                    if event_type == 0:
                        type_name = "start"
                    elif event_type == 1:
                        type_name = "stop"
                    else:
                        type_name = f"type_{int(event_type)}" if event_type is not None else "unknown"
                    event_rows.append(
                        {
                            "time_utc": fit_datetime(int(timestamp)),
                            "local_time": fit_datetime_local_naive(int(timestamp)),
                            "event": event_name,
                            "event_code": int(event),
                            "event_type": type_name,
                            "source_file": path.name,
                        }
                    )

    heart = _dedupe_time_rows(heart_rows, "garmin_hr_bpm")
    respiration = _dedupe_time_rows(rr_rows, "garmin_rr_bpm")
    levels = _dedupe_time_rows(level_rows, "sleep_level_code")
    events = _dedupe_time_rows(event_rows, "event_code")
    return GarminReferenceData(heart, respiration, levels, events)


def write_garmin_reference_csvs(data: GarminReferenceData, output_dir: Path, *, stem: str = "garmin_fit") -> dict[str, Path]:
    """Write decoded Garmin reference tables to CSV files and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "heart_rate": data.heart_rate,
        "respiration_rate": data.respiration_rate,
        "sleep_levels": data.sleep_levels,
        "sleep_events": data.sleep_events,
    }
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = output_dir / f"{stem}_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths
