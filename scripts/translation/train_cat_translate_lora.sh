#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h}"
MODEL="${CAT_TRANSLATE_MODEL:-$ROOT/Research/translation/models/hf-cache/models--hotchpotch--CAT-Translate-0.8b-mlx-q4/snapshots/84cbdd97cf628fa98fcd5a757d2599ebee765cd7}"
DATA="${CAT_TRANSLATE_DATA:-$ROOT/Research/translation/work/cat-translate-licensed-unified-v1}"
ADAPTER="${CAT_TRANSLATE_ADAPTER:-$ROOT/Research/translation/models/cat-translate-0.8b-licensed-unified-lora-v1}"
ITERATIONS="${CAT_TRANSLATE_ITERATIONS:-200}"
LEARNING_RATE="${CAT_TRANSLATE_LEARNING_RATE:-0.000005}"
GRAD_ACCUMULATION="${CAT_TRANSLATE_GRAD_ACCUMULATION:-4}"

cd "$ROOT"
uv run --python 3.12 --with mlx-lm==0.31.3 mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --fine-tune-type lora \
  --mask-prompt \
  --num-layers 16 \
  --batch-size 1 \
  --grad-accumulation-steps "$GRAD_ACCUMULATION" \
  --iters "$ITERATIONS" \
  --learning-rate "$LEARNING_RATE" \
  --max-seq-length 384 \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --val-batches 25 \
  --save-every 50 \
  --adapter-path "$ADAPTER" \
  --grad-checkpoint \
  --seed 20260718
