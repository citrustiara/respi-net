from __future__ import annotations

from typing import Sequence

import serial.tools.list_ports


def list_serial_ports() -> list[str]:
    return [port.device for port in serial.tools.list_ports.comports()]


def ordered_ports(available_ports: Sequence[str], preferred_port: str | None = None) -> list[str]:
    ports: list[str] = []
    if preferred_port and preferred_port in available_ports:
        ports.append(preferred_port)

    for port in reversed(available_ports):
        if port not in ports:
            ports.append(port)

    return ports

