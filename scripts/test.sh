#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

swift build --product MimiSelfTest
"$ROOT/.build/debug/MimiSelfTest"
swift run MimiE2E
