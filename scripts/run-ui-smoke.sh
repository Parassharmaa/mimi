#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

scripts/build-app.sh debug >/dev/null
for scenario in "menu ready" "menu recording" "menu backpressure" "menu failed" "menu clear-confirmation" "transcript ready" "transcript recording" "transcript backpressure" "transcript clear-confirmation" "settings ready"; do
  read -r screen state <<< "$scenario"
  "$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-window --e2e-screen "$screen" --e2e-state "$state" --e2e-auto-quit
done
