#!/bin/zsh
set -euo pipefail

# Runs real native Swift/MLX inference against pre-existing local weights.
# This deliberately does not download anything. Supply the revision-pinned
# model directory and a matching Metal shader explicitly, for example:
#   MIMI_NEMOTRON_MODEL_DIR=/path/to/model \
#   MIMI_MLX_METALLIB=/path/to/mlx.metallib scripts/run-nemotron-fixture-smoke.sh

ROOT="${0:A:h:h}"
MODEL_DIRECTORY="${MIMI_NEMOTRON_MODEL_DIR:?Set MIMI_NEMOTRON_MODEL_DIR to an already-downloaded MLX Nemotron model.}"
METAL_LIBRARY="${MIMI_MLX_METALLIB:?Set MIMI_MLX_METALLIB to a matching MLX Metal shader.}"

[[ -f "$MODEL_DIRECTORY/config.json" ]]
[[ -f "$MODEL_DIRECTORY/model.safetensors" ]]
[[ -s "$METAL_LIBRARY" ]]

cd "$ROOT"
swift build --product MimiNemotronE2E

binary="$ROOT/.build/arm64-apple-macosx/debug/MimiNemotronE2E"
if [[ ! -x "$binary" ]]; then
  binary="$ROOT/.build/debug/MimiNemotronE2E"
fi
[[ -x "$binary" ]]
cp "$METAL_LIBRARY" "${binary:h}/mlx.metallib"

temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/mimi-nemotron-fixture.XXXXXX")"
trap 'rm -rf "$temporary_directory"' EXIT

say -o "$temporary_directory/english.aiff" "Mimi performs English and Japanese transcription locally."
afconvert -f WAVE -d LEI16 "$temporary_directory/english.aiff" "$temporary_directory/english.wav"
"$binary" "$MODEL_DIRECTORY" "$temporary_directory/english.wav" en-US "English and Japanese transcription"
"$binary" --stream "$MODEL_DIRECTORY" "$temporary_directory/english.wav" en-US "English and Japanese transcription"

say -v Kyoko -o "$temporary_directory/japanese.aiff" "ミミはこのマックで日本語の文字起こしをします。"
afconvert -f WAVE -d LEI16 "$temporary_directory/japanese.aiff" "$temporary_directory/japanese.wav"
"$binary" "$MODEL_DIRECTORY" "$temporary_directory/japanese.wav" ja-JP "日本語"
"$binary" --stream "$MODEL_DIRECTORY" "$temporary_directory/japanese.wav" ja-JP "日本語"
