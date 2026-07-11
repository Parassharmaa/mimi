#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
APP="$ROOT/.build/Mimi.app/Contents/MacOS/Mimi"
CONTAINER_FIXTURES="$HOME/Library/Containers/dev.paras.mimi/Data/tmp/realtime-benchmark-suite"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS="$ROOT/.build/realtime-benchmark/$RUN_ID"

if [[ ! -x "$APP" ]]; then
  echo "Build .build/Mimi.app before running the realtime benchmark."
  exit 1
fi

mkdir -p "$CONTAINER_FIXTURES" "$RESULTS"
rm -f "$CONTAINER_FIXTURES/english.aiff" "$CONTAINER_FIXTURES/english.wav"
rm -f "$CONTAINER_FIXTURES/japanese.aiff" "$CONTAINER_FIXTURES/japanese.wav"

ENGLISH_REFERENCE="Mimi keeps English and Japanese transcription on this Mac."
JAPANESE_REFERENCE="ミミはこのマックで日本語の文字起こしをします。"

say -v Samantha -o "$CONTAINER_FIXTURES/english.aiff" "$ENGLISH_REFERENCE"
afconvert -f WAVE -d LEI16 "$CONTAINER_FIXTURES/english.aiff" "$CONTAINER_FIXTURES/english.wav"
say -v Kyoko -o "$CONTAINER_FIXTURES/japanese.aiff" "$JAPANESE_REFERENCE"
afconvert -f WAVE -d LEI16 "$CONTAINER_FIXTURES/japanese.aiff" "$CONTAINER_FIXTURES/japanese.wav"

run_case() {
  local label="$1"
  local engine="$2"
  local audio="$3"
  local language="$4"
  local reference="$5"
  shift 5

  local raw_output
  raw_output="$($APP \
    --benchmark-realtime "$engine" \
    --audio "$audio" \
    --language "$language" \
    --reference "$reference" \
    "$@")"
  local json
  json="$(print -r -- "$raw_output" | awk 'BEGIN { capture = 0 } /^\{/ { capture = 1 } capture { print }')"
  if [[ -z "$json" ]]; then
    print -r -- "$raw_output"
    echo "Benchmark $label did not produce JSON."
    return 1
  fi
  print -r -- "$json" | tee "$RESULTS/$label.json"
}

for language in en ja; do
  if [[ "$language" == "en" ]]; then
    audio="$CONTAINER_FIXTURES/english.wav"
    reference="$ENGLISH_REFERENCE"
  else
    audio="$CONTAINER_FIXTURES/japanese.wav"
    reference="$JAPANESE_REFERENCE"
  fi

  run_case "apple-accurate-$language" apple-accurate "$audio" "$language" "$reference"
  run_case "apple-progressive-$language" apple-progressive "$audio" "$language" "$reference"
  run_case "whisper-rolling-1s-$language" whisper "$audio" "$language" "$reference" --step 1
  run_case "qwen-dual-pass-$language" qwen "$audio" "$language" "$reference"
done

echo "$RESULTS"
