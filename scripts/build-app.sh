#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
CONFIGURATION="${1:-debug}"
APP="$ROOT/.build/Mimi.app"
MODEL_RESOURCES="$ROOT/App/Resources/TranslationModels"
LICENSE_RESOURCES="$ROOT/App/Resources/TranslationLicenses"

cd "$ROOT"
swift build -c "$CONFIGURATION" --product Mimi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ROOT/.build/$CONFIGURATION/Mimi" "$APP/Contents/MacOS/Mimi"
cp "$ROOT/App/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/App/Resources/Mimi.icns" "$APP/Contents/Resources/Mimi.icns"
python3 "$ROOT/scripts/translation/verify_shipped_translation_pack.py" \
  --model-root "$MODEL_RESOURCES" \
  --license-root "$LICENSE_RESOURCES"
cp -R "$MODEL_RESOURCES" "$APP/Contents/Resources/TranslationModels"
cp -R "$LICENSE_RESOURCES" "$APP/Contents/Resources/TranslationLicenses"
"$ROOT/scripts/prepare-mlx-metallib.sh" "$APP/Contents/MacOS" "$CONFIGURATION" required
python3 "$ROOT/scripts/translation/verify_shipped_translation_pack.py" --app "$APP"
codesign --force --deep --sign - --entitlements "$ROOT/App/Mimi.entitlements" "$APP"

echo "$APP"
