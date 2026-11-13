"""Is the machine busy with something else?  For long training runs that share the laptop with a local language
model (e.g. Qwen served by Ollama, llama.cpp, MLX or LM Studio) and should back off while it works.

``python tools/machine_load.py --wait`` blocks until the machine is free (before a GPU run that cannot throttle itself).

``machine_busy(own)`` is true when a local-model process is using CPU, or when the 1-minute load average minus this
run's own ``own`` busy workers exceeds half the cores.
"""

from __future__ import annotations

import os
import subprocess

LLM_HINTS = ("ollama", "llama-server", "llama-cli", "llama.cpp", "mlx_lm", "lm studio", "lmstudio", "qwen", "vllm")
LLM_CPU_PERCENT = 20.0


def local_llm_busy() -> bool:
    try:
        out = subprocess.run(["ps", "-Ao", "pcpu=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        cpu, _, command = line.strip().partition(" ")
        try:
            busy = float(cpu) >= LLM_CPU_PERCENT
        except ValueError:
            continue
        if busy and any(hint in command.lower() for hint in LLM_HINTS):
            return True
    return False


def machine_busy(own: int = 0) -> bool:
    if local_llm_busy():
        return True
    cores = os.cpu_count() or 4
    return os.getloadavg()[0] - own > cores / 2


def wait_until_free(checks: int = 3, every_s: float = 60.0) -> None:
    """Block until the machine has been free (``machine_busy()`` false) on ``checks`` checks in a row."""
    import time

    free = 0
    while free < checks:
        free = 0 if machine_busy() else free + 1
        if free < checks:
            time.sleep(every_s)


if __name__ == "__main__":
    import sys

    if "--wait" in sys.argv:
        print("waiting for the machine to be free...", flush=True)
        wait_until_free()
        print("machine free", flush=True)
    else:
        print("busy" if machine_busy() else "free")
