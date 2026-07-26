#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
CONFIGURATION="${1:-debug}"

MIMI_TRANSLATION_CHANNEL=development \
MIMI_SPEECH_CHANNEL=development \
  "$ROOT/scripts/build-app.sh" "$CONFIGURATION"
