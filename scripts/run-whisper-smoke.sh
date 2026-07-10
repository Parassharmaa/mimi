#!/bin/zsh
set -euo pipefail

# Local-only: requires a person to explicitly download Whisper Large-v3 first
# and a granted microphone permission. It never triggers a model download.
ROOT="${0:A:h:h}"
cd "$ROOT"

scripts/build-app.sh debug >/dev/null
set +e
output=$("$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-engine-smoke whisper 2>&1)
exit_code=$?
set -e
print -r -- "$output"

if [[ $exit_code -ne 0 || "$output" != *"Whisper Large-v3 (626 MB) smoke passed:"* ]]; then
  exit 1
fi
