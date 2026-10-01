#!/bin/zsh
set -euo pipefail
ROOT="${0:A:h:h:h}"
MODEL_ROOT="${1:?usage: run-live-translation-benchmark.sh <model-root> [output-directory]}"
OUTPUT="${2:-$ROOT/.build/live-translation-benchmark}"
mkdir -p "$OUTPUT"
cd "$ROOT"
swift build --disable-index-store --product Mimi
if [[ ! -s "$ROOT/.build/debug/mlx.metallib" ]]; then
  "$ROOT/scripts/prepare-mlx-metallib.sh" "$ROOT/.build/debug" debug required
fi
"$ROOT/.build/debug/Mimi" --benchmark-live-translation-baseline "$OUTPUT/baseline.json" --model-root "$MODEL_ROOT"
"$ROOT/.build/debug/Mimi" --benchmark-instant-translation "$OUTPUT/instant.json" --model-root "$MODEL_ROOT" --baseline "$OUTPUT/baseline.json"
"$ROOT/.build/debug/Mimi" --benchmark-translation-prewarm "$OUTPUT/prewarm.json" --model-root "$MODEL_ROOT"
python3 "$ROOT/scripts/translation/summarize_live_translation.py" "$OUTPUT/baseline.json" "$OUTPUT/instant.json"
