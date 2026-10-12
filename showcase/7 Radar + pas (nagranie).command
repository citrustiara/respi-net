#!/bin/zsh
# Sesja z trenerem: radar A121 + pas tensometryczny zapisywane razem (wzorzec nn_holds, ~4 min).
# Pas załóż i włącz przed startem; radar podłącz przez USB-C. Odległość radaru zmień w --distance-cm.
cd "$(dirname "$0")/.." || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
uv run python tools/a121_iphone_guided_recording.py --reference belt --patterns nn_holds --distance-cm 80 --posture supine
uv run python tools/label_strain_belt_runs.py
read -k1 "?Gotowe. Naciśnij dowolny klawisz, aby zamknąć."
