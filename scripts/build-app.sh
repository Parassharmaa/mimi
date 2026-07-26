#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
CONFIGURATION="${1:-debug}"
APP="$ROOT/.build/Mimi.app"
TRANSLATION_CHANNEL="${MIMI_TRANSLATION_CHANNEL:-development}"
SPEECH_CHANNEL="${MIMI_SPEECH_CHANNEL:-stable}"

case "$TRANSLATION_CHANNEL" in
  development)
    MODEL_CACHE="$ROOT/.build/translation-models/blazeofchi/mimi-en-ja-mlx-development-v1-e6a02099fc62d9f351ffab1fa1efbd46b2c512a0"
    "$ROOT/scripts/translation/fetch_development_translation_pack.sh" "$MODEL_CACHE"
    MODEL_RESOURCES="$MODEL_CACHE/model"
    LICENSE_RESOURCES="$MODEL_CACHE/notices"
    ;;
  stable)
    MODEL_RESOURCES="$ROOT/App/Resources/TranslationModels"
    LICENSE_RESOURCES="$ROOT/App/Resources/TranslationLicenses"
    ;;
  *)
    print -u2 "unsupported MIMI_TRANSLATION_CHANNEL: $TRANSLATION_CHANNEL"
    exit 2
    ;;
esac

case "$SPEECH_CHANNEL" in
  development)
    SPEECH_CACHE="$ROOT/.build/speech-models/mlx-community/whisper-large-v3-turbo-asr-4bit-321a6ead9f6e0646bc8188a54d2a470e275c6b76"
    "$ROOT/scripts/speech/fetch_development_speech_pack.sh" "$SPEECH_CACHE"
    SPEECH_MODEL_RESOURCES="$SPEECH_CACHE/model"
    SPEECH_LICENSE_RESOURCES="$SPEECH_CACHE/notices"
    ;;
  stable)
    SPEECH_MODEL_RESOURCES=""
    SPEECH_LICENSE_RESOURCES="$ROOT/App/Resources/SpeechLicenses"
    ;;
  *)
    print -u2 "unsupported MIMI_SPEECH_CHANNEL: $SPEECH_CHANNEL"
    exit 2
    ;;
esac

cd "$ROOT"
swift build -c "$CONFIGURATION" --product Mimi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ROOT/.build/$CONFIGURATION/Mimi" "$APP/Contents/MacOS/Mimi"
cp "$ROOT/App/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/App/Resources/Mimi.icns" "$APP/Contents/Resources/Mimi.icns"
if [[ "$TRANSLATION_CHANNEL" == "development" ]]; then
  python3 "$ROOT/scripts/translation/verify_development_translation_pack.py" \
    "$MODEL_RESOURCES"
else
  python3 "$ROOT/scripts/translation/verify_shipped_translation_pack.py" \
    --model-root "$MODEL_RESOURCES" \
    --license-root "$LICENSE_RESOURCES"
fi
cp -R "$MODEL_RESOURCES" "$APP/Contents/Resources/TranslationModels"
cp -R "$LICENSE_RESOURCES" "$APP/Contents/Resources/TranslationLicenses"
if [[ "$SPEECH_CHANNEL" == "development" ]]; then
  python3 "$ROOT/scripts/speech/verify_development_speech_pack.py" \
    "$SPEECH_MODEL_RESOURCES"
  mkdir -p "$APP/Contents/Resources/SpeechModels"
  cp -R "$SPEECH_MODEL_RESOURCES" \
    "$APP/Contents/Resources/SpeechModels/mimi-whisper-large-v3-turbo-q4"
fi
cp -R "$SPEECH_LICENSE_RESOURCES" "$APP/Contents/Resources/SpeechLicenses"
cmp "$ROOT/App/Resources/SpeechLicenses/OPENAI-WHISPER-MIT.txt" \
  "$APP/Contents/Resources/SpeechLicenses/OPENAI-WHISPER-MIT.txt"
cmp "$ROOT/App/Resources/SpeechLicenses/PROVENANCE.md" \
  "$APP/Contents/Resources/SpeechLicenses/PROVENANCE.md"
"$ROOT/scripts/prepare-mlx-metallib.sh" "$APP/Contents/MacOS" "$CONFIGURATION" required
if [[ "$SPEECH_CHANNEL" == "development" ]]; then
  python3 "$ROOT/scripts/speech/verify_development_speech_pack.py" \
    "$APP/Contents/Resources/SpeechModels/mimi-whisper-large-v3-turbo-q4"
fi
if [[ "$TRANSLATION_CHANNEL" == "development" ]]; then
  python3 "$ROOT/scripts/translation/verify_development_translation_pack.py" \
    "$APP/Contents/Resources/TranslationModels"
  "$APP/Contents/MacOS/Mimi" \
    --validate-translation-mlx "$APP/Contents/Resources/TranslationModels"
else
  python3 "$ROOT/scripts/translation/verify_shipped_translation_pack.py" --app "$APP"
fi
codesign --force --deep --sign - --entitlements "$ROOT/App/Mimi.entitlements" "$APP"

echo "$APP"
