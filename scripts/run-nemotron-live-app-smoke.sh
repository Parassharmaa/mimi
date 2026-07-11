#!/bin/zsh
set -euo pipefail

# Runs Mimi's real app-owned Nemotron live path against already-downloaded
# local weights. It generates deterministic 48 kHz silence in memory: no
# microphone prompt, no model download, and no source-audio file.
#
# Example:
#   MIMI_NEMOTRON_MODEL_DIR=/path/to/model \
#   MIMI_MLX_METALLIB=/path/to/mlx.metallib \
#   scripts/run-nemotron-live-app-smoke.sh

ROOT="${0:A:h:h}"
MODEL_DIRECTORY="${MIMI_NEMOTRON_MODEL_DIR:?Set MIMI_NEMOTRON_MODEL_DIR to an already-downloaded MLX Nemotron model.}"
METAL_LIBRARY="${MIMI_MLX_METALLIB:?Set MIMI_MLX_METALLIB to a matching MLX Metal shader.}"

[[ -f "$MODEL_DIRECTORY/config.json" ]]
[[ -f "$MODEL_DIRECTORY/model.safetensors" ]]
[[ -s "$METAL_LIBRARY" ]]

cd "$ROOT"
swift build --product Mimi
binary="$ROOT/.build/arm64-apple-macosx/debug/Mimi"
if [[ ! -x "$binary" ]]; then
  binary="$ROOT/.build/debug/Mimi"
fi
[[ -x "$binary" ]]
# This direct executable is intentionally not the sandboxed release .app, so
# it can read the caller's explicit fixture directory. It still executes the
# real Mimi SwiftUI/MLX live engine and its AVAudioConverter path.
cp "$METAL_LIBRARY" "${binary:h}/mlx.metallib"
for language in en ja-JP; do
  output=$(MIMI_NEMOTRON_MODEL_DIR="$MODEL_DIRECTORY" "$binary" --e2e-nemotron-live-smoke --e2e-language "$language" 2>&1)
  print -r -- "$output"
  [[ "$output" == *"Mimi Nemotron"*"live app smoke passed:"* ]]
done
