#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
ARM_BUILD="$ROOT/.build/package-arm64"
INTEL_BUILD="$ROOT/.build/package-x86_64"
APP="$ROOT/.build/Mimi.app"
DIST="$ROOT/.build/dist"
ARCHIVE="$DIST/Mimi-macOS.zip"
SIGNING_IDENTITY="${MIMI_CODESIGN_IDENTITY:--}"

cd "$ROOT"

swift build -c release --product Mimi --arch arm64 --build-path "$ARM_BUILD"
swift build -c release --product Mimi --arch x86_64 --build-path "$INTEL_BUILD"

rm -rf "$APP" "$DIST"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$DIST"

lipo -create \
  "$ARM_BUILD/release/Mimi" \
  "$INTEL_BUILD/release/Mimi" \
  -output "$APP/Contents/MacOS/Mimi"
cp "$ROOT/App/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/App/Resources/Mimi.icns" "$APP/Contents/Resources/Mimi.icns"

ARCHS="$(lipo -archs "$APP/Contents/MacOS/Mimi")"
[[ "$ARCHS" == *arm64* && "$ARCHS" == *x86_64* ]]

SIGNING_ARGUMENTS=(
  --force
  --deep
  --options runtime
  --entitlements "$ROOT/App/Mimi.entitlements"
)
if [[ "$SIGNING_IDENTITY" != "-" ]]; then
  SIGNING_ARGUMENTS+=(--timestamp)
fi
codesign "${SIGNING_ARGUMENTS[@]}" --sign "$SIGNING_IDENTITY" "$APP"
codesign --verify --deep --strict "$APP"
plutil -lint "$APP/Contents/Info.plist"

ditto -c -k --sequesterRsrc --keepParent "$APP" "$ARCHIVE"
(
  cd "$DIST"
  shasum -a 256 "${ARCHIVE:t}" > "${ARCHIVE:t}.sha256"
)

echo "$ARCHIVE"
