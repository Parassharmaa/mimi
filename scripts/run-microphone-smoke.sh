#!/bin/zsh
set -euo pipefail

# A deliberately local-only physical-Mac check. It requests microphone access,
# captures for one second, retains no source audio, and must not run in CI.
ROOT="${0:A:h:h}"
cd "$ROOT"

scripts/build-app.sh debug >/dev/null
set +e
output=$("$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-microphone-smoke 2>&1)
exit_code=$?
set -e
print -r -- "$output"

if [[ $exit_code -ne 0 || "$output" != *"Mimi microphone smoke passed:"* ]]; then
  exit 1
fi
