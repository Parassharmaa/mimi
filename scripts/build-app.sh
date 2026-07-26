#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
CONFIGURATION="${1:-debug}"
APP="$ROOT/.build/Mimi.app"
TRANSLATION_CHANNEL="${MIMI_TRANSLATION_CHANNEL:-development}"

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
"$ROOT/scripts/prepare-mlx-metallib.sh" "$APP/Contents/MacOS" "$CONFIGURATION" required
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
