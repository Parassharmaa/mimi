#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

scripts/build-app.sh debug >/dev/null
"$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-window --e2e-auto-quit
