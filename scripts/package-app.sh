#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
ARM_BUILD="$ROOT/.build/package-arm64"
INTEL_BUILD="$ROOT/.build/package-x86_64"
APP="$ROOT/.build/Mimi.app"
DIST="$ROOT/.build/dist"
ARCHIVE="$DIST/Mimi-macOS.zip"
SIGNING_IDENTITY="${MIMI_CODESIGN_IDENTITY:--}"
MODEL_RESOURCES="$ROOT/App/Resources/TranslationModels"
LICENSE_RESOURCES="$ROOT/App/Resources/TranslationLicenses"
SPEECH_LICENSE_RESOURCES="$ROOT/App/Resources/SpeechLicenses"
PHONON_CACHE="$ROOT/.build/phonon2-model"
"$ROOT/scripts/speech/fetch_phonon2_pack.sh" "$PHONON_CACHE"

cd "$ROOT"

swift build --disable-index-store -c release --product Mimi --arch arm64 --build-path "$ARM_BUILD"
swift build --disable-index-store -c release --product Mimi --arch x86_64 --build-path "$INTEL_BUILD"

rm -rf "$APP" "$DIST"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$DIST"

lipo -create \
  "$ARM_BUILD/release/Mimi" \
  "$INTEL_BUILD/release/Mimi" \
  -output "$APP/Contents/MacOS/Mimi"
cp "$ROOT/App/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/App/Resources/Mimi.icns" "$APP/Contents/Resources/Mimi.icns"
python3 "$ROOT/scripts/translation/verify_shipped_translation_pack.py" \
  --model-root "$MODEL_RESOURCES" \
  --license-root "$LICENSE_RESOURCES"
cp -R "$MODEL_RESOURCES" "$APP/Contents/Resources/TranslationModels"
cp -R "$LICENSE_RESOURCES" "$APP/Contents/Resources/TranslationLicenses"
[[ -s "$SPEECH_LICENSE_RESOURCES/OPENAI-WHISPER-MIT.txt" ]]
[[ -s "$SPEECH_LICENSE_RESOURCES/PROVENANCE.md" ]]
cp -R "$SPEECH_LICENSE_RESOURCES" "$APP/Contents/Resources/SpeechLicenses"
mkdir -p "$APP/Contents/Resources/SpeechModels/mimi-phonon2"
cp -cR "$PHONON_CACHE/model/." "$APP/Contents/Resources/SpeechModels/mimi-phonon2/" 2>/dev/null \
  || cp -R "$PHONON_CACHE/model/." "$APP/Contents/Resources/SpeechModels/mimi-phonon2/"
cp -R "$PHONON_CACHE/notices" "$APP/Contents/Resources/SpeechLicenses/Phonon2"
cp "$ROOT/App/Resources/SpeechLicenses/PHONON2-CONVERSION.md" \
  "$APP/Contents/Resources/SpeechLicenses/Phonon2/MIMI-CONVERSION.md"
python3 "$ROOT/scripts/speech/verify_phonon2_pack.py" "$APP/Contents/Resources/SpeechModels/mimi-phonon2"
cmp "$SPEECH_LICENSE_RESOURCES/OPENAI-WHISPER-MIT.txt" \
  "$APP/Contents/Resources/SpeechLicenses/OPENAI-WHISPER-MIT.txt"
cmp "$SPEECH_LICENSE_RESOURCES/PROVENANCE.md" \
  "$APP/Contents/Resources/SpeechLicenses/PROVENANCE.md"
"$ROOT/scripts/prepare-mlx-metallib.sh" "$APP/Contents/MacOS" release required
python3 "$ROOT/scripts/translation/verify_shipped_translation_pack.py" --app "$APP"

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
python3 "$ROOT/scripts/verify_microphone_entitlement.py" "$APP"
plutil -lint "$APP/Contents/Info.plist"

ditto -c -k --sequesterRsrc --keepParent "$APP" "$ARCHIVE"
(
  cd "$DIST"
  shasum -a 256 "${ARCHIVE:t}" > "${ARCHIVE:t}.sha256"
)

echo "$ARCHIVE"
