#!/bin/zsh
# Nagrywa radar A121 przez noc (zasięg 0,3–1,3 m), a po Ctrl+C liczy oddech, tętno i fazy snu.
# Mac nie zaśnie w trakcie (caffeinate), ale nie zamykaj klapy i podłącz zasilacz.
cd "$(dirname "$0")/.." || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
PORT=$(ls /dev/cu.usbmodem*1 2>/dev/null | head -1)
if [ -z "$PORT" ]; then
  echo "Nie widzę radaru (podłącz USB-C) — przerywam."
  read -k1 "?Naciśnij dowolny klawisz, aby zamknąć."
  exit 1
fi
echo "Radar A121 na porcie: $PORT. Zatrzymanie rano: Ctrl+C (potem poczekaj na analizę)."
caffeinate -i uv run respi record-a121-sleep --port "$PORT" --start-m 0.3 --end-m 1.3 --label "noc-$(date +%Y-%m-%d)"
read -k1 "?Gotowe. Naciśnij dowolny klawisz, aby zamknąć."
