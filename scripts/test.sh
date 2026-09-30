#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

zsh scripts/test-history-safety.sh
zsh scripts/test-onboarding-policy.sh

swift build --disable-index-store --product MimiSelfTest
swift build --disable-index-store --product Mimi
"$ROOT/.build/debug/Mimi" --verify-voice-typing-destination
PERSISTENCE_REPORT="$(mktemp -t mimi-persistence-safety).json"
"$ROOT/.build/debug/Mimi" --verify-transcript-persistence-safety "$PERSISTENCE_REPORT"
LIFECYCLE_REPORT="$(mktemp -t mimi-voice-lifecycle).json"
"$ROOT/.build/debug/Mimi" --verify-voice-typing-lifecycle "$LIFECYCLE_REPORT"
VOICE_TYPING_REPORT="$(mktemp -t mimi-voice-typing-model-selection).json"
"$ROOT/.build/debug/Mimi" \
  --verify-voice-typing-model-selection "$VOICE_TYPING_REPORT"
EXCLUSIVITY_REPORT="$(mktemp -t mimi-speech-exclusivity).json"
"$ROOT/.build/debug/Mimi" --verify-speech-exclusivity "$EXCLUSIVITY_REPORT"
python3 scripts/speech/test_speech_benchmark_tools.py
python3 scripts/translation/test_app_payload_budget.py
python3 scripts/speech/verify_paced_speech_evidence.py
python3 scripts/speech/verify_adaptive_segmentation_evidence.py
python3 scripts/translation/verify_shipped_translation_pack.py \
  --model-root App/Resources/TranslationModels \
  --license-root App/Resources/TranslationLicenses
"$ROOT/.build/debug/Mimi" \
  --validate-translation-mlx "$ROOT/App/Resources/TranslationModels"
"$ROOT/.build/debug/MimiSelfTest"
swift run --disable-index-store MimiE2E
swift run --disable-index-store MimiSessionE2E
scripts/run-ui-smoke.sh
"$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-main-window-lifecycle
