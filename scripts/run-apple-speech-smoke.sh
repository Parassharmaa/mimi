#!/bin/zsh
set -euo pipefail

# Local-only: requires an explicitly downloaded Apple Speech language asset
# and a granted microphone permission. It never downloads an asset itself.
ROOT="${0:A:h:h}"
cd "$ROOT"

scripts/build-app.sh debug >/dev/null
set +e
output=$("$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-engine-smoke apple 2>&1)
exit_code=$?
set -e
print -r -- "$output"

if [[ $exit_code -ne 0 || "$output" != *"Apple Speech smoke passed:"* ]]; then
  exit 1
fi
