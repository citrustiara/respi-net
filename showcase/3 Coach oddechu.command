#!/bin/zsh
# Przewodnik oddechu: 1:30, 12 oddechów/min i 15 s pauzy. Spacja = start/pauza, Esc = koniec.
cd "$(dirname "$0")/.." || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
uv run respi coach paced_12_hold
