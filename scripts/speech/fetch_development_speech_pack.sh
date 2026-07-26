#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h}"
REVISION="321a6ead9f6e0646bc8188a54d2a470e275c6b76"
REPOSITORY="mlx-community/whisper-large-v3-turbo-asr-4bit"
OUTPUT="${1:-$ROOT/.build/speech-models/$REPOSITORY-$REVISION}"
MODEL_ROOT="$OUTPUT/model"
NOTICE_ROOT="$OUTPUT/notices"
BASE_URL="https://huggingface.co/$REPOSITORY/resolve/$REVISION"

if [[ -d "$MODEL_ROOT" ]]; then
  python3 "$ROOT/scripts/speech/verify_development_speech_pack.py" "$MODEL_ROOT"
  mkdir -p "$NOTICE_ROOT"
  cp "$ROOT/App/Resources/SpeechLicenses/OPENAI-WHISPER-MIT.txt" "$NOTICE_ROOT/"
  cp "$ROOT/App/Resources/SpeechLicenses/PROVENANCE.md" "$NOTICE_ROOT/"
  echo "$OUTPUT"
  exit 0
fi
if [[ -e "$OUTPUT" ]]; then
  print -u2 "refusing to overwrite incomplete development speech cache: $OUTPUT"
  exit 1
fi

mkdir -p "${OUTPUT:h}"
TEMP_ROOT="$(mktemp -d "${OUTPUT:h}/mimi-speech-development.XXXXXX")"
trap 'rm -rf "$TEMP_ROOT"' EXIT
mkdir -p "$TEMP_ROOT/model" "$TEMP_ROOT/notices"

MODEL_FILES=(
  README.md
  added_tokens.json
  config.json
  generation_config.json
  merges.txt
  model.safetensors
  model.safetensors.index.json
  normalizer.json
  preprocessor_config.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
)
for relative in "${MODEL_FILES[@]}"; do
  curl --fail --location --retry 3 --silent --show-error \
    "$BASE_URL/$relative" \
    --output "$TEMP_ROOT/model/$relative"
done

cp "$ROOT/App/Resources/SpeechLicenses/OPENAI-WHISPER-MIT.txt" "$TEMP_ROOT/notices/"
cp "$ROOT/App/Resources/SpeechLicenses/PROVENANCE.md" "$TEMP_ROOT/notices/"
python3 "$ROOT/scripts/speech/verify_development_speech_pack.py" "$TEMP_ROOT/model"

mv "$TEMP_ROOT" "$OUTPUT"
trap - EXIT
echo "$OUTPUT"
