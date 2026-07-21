#!/bin/zsh
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

model_directory="${1:-Research/translation/models/smollm2-135m-instruct-4bit}"
data_directory="${2:-Research/translation/work/kftt}"
adapter_directory="${3:-Research/translation/models/smollm2-135m-kftt-lora}"

uv run --python 3.12 --with mlx-lm==0.31.3 mlx_lm.lora \
  --model "$model_directory" \
  --train \
  --data "$data_directory" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers 30 \
  --batch-size 4 \
  --iters 4000 \
  --val-batches 50 \
  --learning-rate 0.0001 \
  --steps-per-report 100 \
  --steps-per-eval 500 \
  --adapter-path "$adapter_directory" \
  --save-every 1000 \
  --max-seq-length 256 \
  --seed 20260717
