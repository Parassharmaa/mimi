#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

swift build --product MimiSelfTest
swift build --product Mimi
python3 scripts/translation/verify_shipped_translation_pack.py \
  --model-root App/Resources/TranslationModels \
  --license-root App/Resources/TranslationLicenses
"$ROOT/.build/debug/Mimi" \
  --validate-translation-mlx "$ROOT/App/Resources/TranslationModels"
"$ROOT/.build/debug/MimiSelfTest"
swift run MimiE2E
swift run MimiSessionE2E
scripts/run-ui-smoke.sh
"$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-main-window-lifecycle
