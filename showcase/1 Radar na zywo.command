#!/bin/zsh
# Otwiera aplikację z radarem A121 na żywo (port wykrywany automatycznie).
cd "$(dirname "$0")/.." || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
PORT=$(ls /dev/cu.usbmodem*1 2>/dev/null | head -1)
if [ -n "$PORT" ]; then
  echo "Radar A121 na porcie: $PORT"
  uv run respi app --sensor a121 --port "$PORT"
else
  echo "Nie widzę radaru (podłącz USB-C). Otwieram aplikację bez portu."
  uv run respi app --sensor a121
fi
