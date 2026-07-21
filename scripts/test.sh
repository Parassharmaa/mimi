#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

swift build --product MimiSelfTest
swift build --product Mimi
"$ROOT/.build/debug/MimiSelfTest"
swift run MimiE2E
swift run MimiSessionE2E
scripts/run-ui-smoke.sh
"$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-main-window-lifecycle
