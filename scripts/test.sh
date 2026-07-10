#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

swift build --product MimiSelfTest
swift build --product Mimi
swift build --product MimiNemotronE2E
swift build --product MimiMLXRuntimeE2E
"$ROOT/.build/debug/MimiSelfTest"
swift run MimiE2E
swift run MimiSessionE2E
