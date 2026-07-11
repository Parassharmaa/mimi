#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
CONFIGURATION="${1:-debug}"
APP="$ROOT/.build/Mimi.app"

cd "$ROOT"
swift build -c "$CONFIGURATION" --product Mimi

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$ROOT/.build/$CONFIGURATION/Mimi" "$APP/Contents/MacOS/Mimi"
cp "$ROOT/App/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/App/Resources/Mimi.icns" "$APP/Contents/Resources/Mimi.icns"
codesign --force --deep --sign - --entitlements "$ROOT/App/Mimi.entitlements" "$APP"

echo "$APP"
