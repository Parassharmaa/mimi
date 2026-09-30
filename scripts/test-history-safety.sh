#!/bin/zsh
set -euo pipefail

MIMI_ROOT="${0:A:h:h}"
MIMI_HISTORY_BUILD="$(mktemp -d -t mimi-history-build)"

swiftc -parse-as-library -emit-module -emit-library -module-name MimiCore \
  "$MIMI_ROOT"/Sources/MimiCore/*.swift \
  -emit-module-path "$MIMI_HISTORY_BUILD/MimiCore.swiftmodule" \
  -o "$MIMI_HISTORY_BUILD/libMimiCore.dylib"
swiftc -parse-as-library -I "$MIMI_HISTORY_BUILD" -L "$MIMI_HISTORY_BUILD" -lMimiCore \
  -Xlinker -rpath -Xlinker "$MIMI_HISTORY_BUILD" \
  "$MIMI_ROOT/Sources/Mimi/LocalStorage.swift" \
  "$MIMI_ROOT/Sources/Mimi/TranscriptHistoryStore.swift" \
  "$MIMI_ROOT/Tools/MimiHistorySelfTest/main.swift" \
  -o "$MIMI_HISTORY_BUILD/MimiHistorySelfTest"
"$MIMI_HISTORY_BUILD/MimiHistorySelfTest"
