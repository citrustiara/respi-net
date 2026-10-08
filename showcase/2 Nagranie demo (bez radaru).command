#!/bin/zsh
# Plan B: aplikacja z gotowym nagraniem (3 min, dwie pauzy po wdechu, ~0,4 m) - radar niepotrzebny.
cd "$(dirname "$0")/.." || exit 1
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"
uv run respi app --sensor a121 --open data/raw/nn/a121_iphone/phase1_halfside_02/run_01_nn_self_3min_a121.csv
