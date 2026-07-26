#!/bin/zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
  print -u2 "usage: $0 OUTPUT_DIRECTORY"
  exit 2
fi

script_directory=${0:A:h}
repository_root=${script_directory:h:h}
cd "$repository_root"

python3 scripts/translation/package_elanmt_mlx_experts.py \
  App/Resources/TranslationModels \
  Research/translation/models/elanmt-release-clean-full-depth-en-ja-v1-avg3-mlx-4bit \
  Research/translation/routers/guarded-expert-cascade-v19-en-ja.json \
  Research/translation/models/elanmt-release-clean-legal-specialist-ja-en-v1-mlx-4bit \
  Research/translation/routers/guarded-expert-cascade-v19-ja-en.json \
  "$1"
