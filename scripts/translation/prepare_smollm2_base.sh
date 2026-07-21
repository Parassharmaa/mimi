#!/bin/zsh
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

source_revision="422de227b90002f443a21a58b1087f6ee7632731"
cache_root="Research/translation/models/hf-cache"
source_snapshot="$cache_root/models--mlx-community--SmolLM2-135M-Instruct/snapshots/$source_revision"
output_directory="${1:-Research/translation/models/smollm2-135m-instruct-4bit}"

uv run --python 3.12 --with huggingface-hub python - "$cache_root" "$source_revision" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    "mlx-community/SmolLM2-135M-Instruct",
    revision=sys.argv[2],
    cache_dir=sys.argv[1],
)
PY

uv run --python 3.12 --with mlx-lm==0.31.3 mlx_lm.convert \
  --hf-path "$source_snapshot" \
  --mlx-path "$output_directory" \
  --quantize \
  --q-bits 4 \
  --q-group-size 64

find "$output_directory" -type f -print0 | xargs -0 stat -f '%z' | awk \
  '{ total += $1 } END { print total " bytes" }'
