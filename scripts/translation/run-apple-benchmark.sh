#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h}"
SUITE="${1:-$ROOT/Research/translation/benchmark/canary.jsonl}"
OUTPUT="${2:-$ROOT/Research/translation/results/apple-canary.json}"
WARM_RUNS="${3:-3}"

mkdir -p "${OUTPUT:h}"
"$ROOT/scripts/build-app.sh" debug >/dev/null
"$ROOT/.build/Mimi.app/Contents/MacOS/Mimi" \
  --benchmark-translation-apple "$SUITE" \
  --benchmark-translation-apple-warm-runs "$WARM_RUNS" \
  --output "$OUTPUT"

echo "$OUTPUT"
