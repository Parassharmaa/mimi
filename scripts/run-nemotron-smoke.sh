#!/bin/zsh
set -euo pipefail

# Opt-in physical-Mac smoke test. It never downloads a model: install the
# explicit Nemotron pack in Mimi first, then run this while granting the one
# microphone permission if macOS asks. A full-Xcode build (or the explicit
# MIMI_MLX_METALLIB developer override) is required to include MLX shaders.

ROOT="${0:A:h:h}"
LANGUAGE="${1:-en}"
"$ROOT/scripts/build-app.sh" debug
set +e
output=$("$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-engine-smoke nemotron --e2e-language "$LANGUAGE" 2>&1)
exit_code=$?
set -e
print -r -- "$output"

if [[ $exit_code -ne 0 || "$output" != *"Nemotron 3.5 MLX (756 MB) smoke passed:"* ]]; then
  exit 1
fi
