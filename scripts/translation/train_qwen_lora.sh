#!/bin/zsh
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

data_directory="${1:-Research/translation/work/tatoeba}"
adapter_directory="${2:-Research/translation/models/qwen3-0.6b-tatoeba-lora}"

uv run --python 3.12 --with mlx-lm==0.31.3 mlx_lm.lora \
  --model mlx-community/Qwen3-0.6B-4bit \
  --train \
  --data "$data_directory" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers 16 \
  --batch-size 2 \
  --iters 2000 \
  --val-batches 50 \
  --learning-rate 0.00002 \
  --steps-per-report 50 \
  --steps-per-eval 250 \
  --adapter-path "$adapter_directory" \
  --save-every 500 \
  --max-seq-length 256 \
  --seed 20260717
