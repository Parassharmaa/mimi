#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
CONFIGURATION="${1:-debug}"
APP="$ROOT/.build/Mimi-development.app"
MODEL_CACHE="$ROOT/.build/translation-models/blazeofchi/mimi-en-ja-mlx-development-v1-e6a02099fc62d9f351ffab1fa1efbd46b2c512a0"
MODEL_ROOT="$MODEL_CACHE/model"
NOTICE_ROOT="$MODEL_CACHE/notices"

cd "$ROOT"
"$ROOT/scripts/translation/fetch_development_translation_pack.sh" "$MODEL_CACHE"
swift build -c "$CONFIGURATION" --product Mimi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ROOT/.build/$CONFIGURATION/Mimi" "$APP/Contents/MacOS/Mimi"
cp "$ROOT/App/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/App/Resources/Mimi.icns" "$APP/Contents/Resources/Mimi.icns"
cp -R "$MODEL_ROOT" "$APP/Contents/Resources/TranslationModels"
cp -R "$NOTICE_ROOT" "$APP/Contents/Resources/TranslationDevelopmentLicenses"

/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName Mimi Development" \
  "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName Mimi Development" \
  "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier dev.paras.mimi.development" \
  "$APP/Contents/Info.plist"

"$ROOT/scripts/prepare-mlx-metallib.sh" \
  "$APP/Contents/MacOS" "$CONFIGURATION" required
python3 "$ROOT/scripts/translation/verify_development_translation_pack.py" \
  "$APP/Contents/Resources/TranslationModels"
"$APP/Contents/MacOS/Mimi" \
  --validate-translation-mlx "$APP/Contents/Resources/TranslationModels"
codesign --force --deep --sign - --entitlements "$ROOT/App/Mimi.entitlements" "$APP"
codesign --verify --deep --strict "$APP"
plutil -lint "$APP/Contents/Info.plist"

echo "$APP"
