#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

scripts/build-app.sh debug >/dev/null
for scenario in "menu ready" "menu recording" "menu failed" "transcript ready" "transcript recording" "settings ready"; do
  read -r screen state <<< "$scenario"
  "$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-window --e2e-screen "$screen" --e2e-state "$state" --e2e-auto-quit
done
