#!/bin/zsh
set -euo pipefail

# MLX Swift's command-line SwiftPM build cannot produce Metal shaders. Build
# the matching shader through its checked-out Xcode project, then put it where
# the statically linked MLX runtime looks first: beside Mimi's executable. The
# shader is used by Mimi's bundled translation model and optional ASR runtimes.
# A local developer may supply an already-built matching library explicitly;
# release packaging invokes this helper with `required`.

ROOT="${0:A:h:h}"
DESTINATION="${1:?usage: prepare-mlx-metallib.sh <resources-dir> <debug|release> <optional|required>}"
CONFIGURATION="${2:-debug}"
REQUIREMENT="${3:-optional}"

case "$CONFIGURATION" in
  debug) XCODE_CONFIGURATION="Debug" ;;
  release) XCODE_CONFIGURATION="Release" ;;
  *)
    print -u2 "Unsupported build configuration: $CONFIGURATION"
    exit 64
    ;;
esac

case "$REQUIREMENT" in
  optional|required) ;;
  *)
    print -u2 "Requirement must be optional or required, got: $REQUIREMENT"
    exit 64
    ;;
esac

source_library=""
if [[ -n "${MIMI_MLX_METALLIB:-}" ]]; then
  source_library="$MIMI_MLX_METALLIB"
elif xcrun --find xcodebuild >/dev/null 2>&1 && xcrun --find metallib >/dev/null 2>&1; then
  mlx_checkout=""
  for candidate in \
    "$ROOT/.build/checkouts/mlx-swift" \
    "$ROOT/.build/package-arm64/checkouts/mlx-swift" \
    "$ROOT/.build/package-x86_64/checkouts/mlx-swift"; do
    if [[ -d "$candidate/xcode/MLX.xcodeproj" ]]; then
      mlx_checkout="$candidate"
      break
    fi
  done
  if [[ -z "$mlx_checkout" ]]; then
    print -u2 "MLX Swift checkout is missing; resolve Swift packages before packaging Mimi."
    exit 1
  fi

  derived_data="$ROOT/.build/mlx-metal-$CONFIGURATION"
  xcodebuild \
    -project "$mlx_checkout/xcode/MLX.xcodeproj" \
    -scheme MLX \
    -configuration "$XCODE_CONFIGURATION" \
    -sdk macosx \
    -derivedDataPath "$derived_data" \
    build CODE_SIGNING_ALLOWED=NO

  source_library="$(find "$derived_data/Build/Products/$XCODE_CONFIGURATION" -type f -name default.metallib -print -quit)"
else
  print -u2 "Full Xcode is not selected, so MLX's Metal shader cannot be built."
fi

if [[ -z "$source_library" || ! -s "$source_library" ]]; then
  if [[ "$REQUIREMENT" == "required" ]]; then
    print -u2 "Mimi cannot package its local MLX translator without a matching Metal shader. Install/select full Xcode or set MIMI_MLX_METALLIB explicitly."
    exit 1
  fi
  print -u2 "Warning: building Mimi without optional MLX runtimes. Apple Speech and Whisper remain available."
  exit 0
fi

mkdir -p "$DESTINATION"
cp "$source_library" "$DESTINATION/mlx.metallib"
print "$DESTINATION/mlx.metallib"
