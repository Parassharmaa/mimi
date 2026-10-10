#!/bin/zsh
set -euo pipefail

MIMI_ROOT="${0:A:h:h}"
MIMI_TEST_BUILD="$(mktemp -d -t mimi-transcript-test)"
trap 'rm -rf "$MIMI_TEST_BUILD"' EXIT
swiftc -parse-as-library \
  "$MIMI_ROOT/Sources/Mimi/SelectableTranscriptText.swift" \
  "$MIMI_ROOT/Sources/Mimi/TranscriptClipboard.swift" \
  "$MIMI_ROOT/Tools/MimiTranscriptSelfTest/main.swift" \
  -o "$MIMI_TEST_BUILD/MimiTranscriptSelfTest"
"$MIMI_TEST_BUILD/MimiTranscriptSelfTest"
