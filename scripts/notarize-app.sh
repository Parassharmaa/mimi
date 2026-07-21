#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
APP="$ROOT/.build/Mimi.app"
DIST="$ROOT/.build/dist"
ARCHIVE="${1:-$DIST/Mimi-macOS.zip}"

[[ -d "$APP" ]]
[[ -f "$ARCHIVE" ]]

SIGNATURE_DETAILS="$(codesign --display --verbose=4 "$APP" 2>&1)"
if [[ "$SIGNATURE_DETAILS" != *"Authority=Developer ID Application:"* ]]; then
  print -u2 "Mimi must be signed with a Developer ID Application certificate before notarization."
  exit 1
fi
if [[ "$SIGNATURE_DETAILS" != *"flags="*"runtime"* ]]; then
  print -u2 "Mimi must enable the hardened runtime before notarization."
  exit 1
fi

AUTHENTICATION=()
if [[ -n "${MIMI_NOTARY_KEYCHAIN_PROFILE:-}" ]]; then
  AUTHENTICATION+=(--keychain-profile "$MIMI_NOTARY_KEYCHAIN_PROFILE")
elif [[ -n "${MIMI_NOTARY_KEY_PATH:-}" && -n "${MIMI_NOTARY_KEY_ID:-}" && -n "${MIMI_NOTARY_ISSUER:-}" ]]; then
  AUTHENTICATION+=(
    --key "$MIMI_NOTARY_KEY_PATH"
    --key-id "$MIMI_NOTARY_KEY_ID"
    --issuer "$MIMI_NOTARY_ISSUER"
  )
else
  print -u2 "Provide MIMI_NOTARY_KEYCHAIN_PROFILE or the three MIMI_NOTARY_KEY_* values."
  exit 1
fi

codesign --verify --deep --strict --verbose=2 "$APP"
xcrun notarytool submit "$ARCHIVE" "${AUTHENTICATION[@]}" --wait
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"
spctl --assess --type execute --verbose=4 "$APP"
python3 "$ROOT/scripts/translation/verify_shipped_translation_pack.py" --app "$APP"

# Stapling changes the app bundle, so rebuild the downloadable archive and
# checksum only after the ticket is attached.
rm -f "$ARCHIVE" "$ARCHIVE.sha256"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ARCHIVE"
(
  cd "${ARCHIVE:h}"
  shasum -a 256 "${ARCHIVE:t}" > "${ARCHIVE:t}.sha256"
)

echo "$ARCHIVE"
