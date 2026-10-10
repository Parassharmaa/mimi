#!/bin/zsh
set -euo pipefail
MIMI_ROOT="${0:A:h:h}"
if (( $# < 2 )); then
  print -u2 "Usage: scripts/run-realtime-benchmark.sh <frozen-manifest.jsonl> <new-output-directory> [benchmark options]"
  print -u2 "See docs/BENCHMARKS.md for fixture preparation, models, and reporting."
  exit 2
fi
exec python3 "$MIMI_ROOT/scripts/benchmarks/run.py" "$@"
