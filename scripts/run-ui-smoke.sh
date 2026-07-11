#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
cd "$ROOT"

scripts/build-app.sh debug >/dev/null
for scenario in "menu ready" "menu recording" "menu backpressure" "menu failed" "menu clear-confirmation" "menu follow-latest-paused" "transcript ready" "transcript recording" "transcript backpressure" "transcript clear-confirmation" "transcript follow-latest-paused" "transcript incremental-translation" "transcript translation-stream" "onboarding welcome" "onboarding ready" "onboarding voice-enabled" "captions ready" "captions caption-stream" "voice-typing ready" "settings-models ready" "settings-capture ready" "settings-privacy ready" "settings-captions ready" "settings-voice ready" "settings-voice voice-enabled"; do
  read -r screen state <<< "$scenario"
  "$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-window --e2e-screen "$screen" --e2e-state "$state" --e2e-auto-quit
done

for appearance in light dark; do
  for screen in menu transcript onboarding captions voice-typing settings-models settings-capture settings-privacy settings-captions settings-voice; do
    "$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" --e2e-window --e2e-screen "$screen" --e2e-state ready --e2e-appearance "$appearance" --e2e-auto-quit
  done
done
