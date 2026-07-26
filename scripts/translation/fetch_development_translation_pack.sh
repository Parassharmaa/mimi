#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h}"
REVISION="e6a02099fc62d9f351ffab1fa1efbd46b2c512a0"
REPOSITORY="blazeofchi/mimi-en-ja-mlx-development-v1"
OUTPUT="${1:-$ROOT/.build/translation-models/$REPOSITORY-$REVISION}"
MODEL_ROOT="$OUTPUT/model"
NOTICE_ROOT="$OUTPUT/notices"
BASE_URL="https://huggingface.co/$REPOSITORY/resolve/$REVISION"

if [[ -d "$MODEL_ROOT" ]]; then
  python3 "$ROOT/scripts/translation/verify_development_translation_pack.py" \
    "$MODEL_ROOT"
  echo "$OUTPUT"
  exit 0
fi
if [[ -e "$OUTPUT" ]]; then
  print -u2 "refusing to overwrite incomplete development model cache: $OUTPUT"
  exit 1
fi

mkdir -p "${OUTPUT:h}"
TEMP_ROOT="$(mktemp -d "${OUTPUT:h}/mimi-translation-development.XXXXXX")"
trap 'rm -rf "$TEMP_ROOT"' EXIT
mkdir -p "$TEMP_ROOT/model/en-ja" "$TEMP_ROOT/model/ja-en" "$TEMP_ROOT/notices"

MODEL_FILES=(
  manifest.json
  en-ja/manifest.json
  en-ja/model.safetensors
  en-ja/tokenizer.json
  en-ja/tokenizer_config.json
  ja-en/manifest.json
  ja-en/model.safetensors
  ja-en/tokenizer.json
  ja-en/tokenizer_config.json
)
for relative in "${MODEL_FILES[@]}"; do
  curl --fail --location --retry 3 --silent --show-error \
    "$BASE_URL/$relative" \
    --output "$TEMP_ROOT/model/$relative"
done

NOTICE_FILES=(ATTRIBUTIONS.md LICENSE README.md model-equivalence.json tatoeba-attributions.jsonl.gz)
for relative in "${NOTICE_FILES[@]}"; do
  curl --fail --location --retry 3 --silent --show-error \
    "$BASE_URL/$relative" \
    --output "$TEMP_ROOT/notices/$relative"
done

python3 "$ROOT/scripts/translation/verify_development_translation_pack.py" \
  "$TEMP_ROOT/model"

EXPECTED_NOTICE_RECORDS=(
  "ATTRIBUTIONS.md:c1002013efb88598ff7260bd6e2af73c306afbd0d8bf3ddfaa21885a8cfdf142"
  "LICENSE:28a9529c7d0bb4dc51f4bf5c116a3d16ef247a052f7591466768ddf563fd1cf5"
  "README.md:6e0d4f644da2b48d998fbd627f0f58c6877dd7c1859963608d14b2173d8b633d"
  "model-equivalence.json:e996706e826b275d8f9d2ca79117894190ca35fb6bca898f12e575fe9054c600"
  "tatoeba-attributions.jsonl.gz:5036ea849a18729711f71be2628f14df065a0ce152aeaa26ff80b4f6ee6eec18"
)
for record in "${EXPECTED_NOTICE_RECORDS[@]}"; do
  relative="${record%%:*}"
  expected="${record#*:}"
  actual="$(shasum -a 256 "$TEMP_ROOT/notices/$relative" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    print -u2 "development notice verification failed: $relative"
    exit 1
  fi
done

mv "$TEMP_ROOT" "$OUTPUT"
trap - EXIT
echo "$OUTPUT"
