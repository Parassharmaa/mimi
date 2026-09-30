#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h}"
CHECK_DIRECTORY="$(mktemp -d -t mimi-onboarding-policy)"
trap 'rm -r -- "$CHECK_DIRECTORY"' EXIT

swiftc -swift-version 6 -emit-library -emit-module -module-name MimiCore \
  "$ROOT/Sources/MimiCore/Domain.swift" \
  -emit-module-path "$CHECK_DIRECTORY/MimiCore.swiftmodule" \
  -o "$CHECK_DIRECTORY/libMimiCore.dylib"
swiftc -swift-version 6 -I "$CHECK_DIRECTORY" -L "$CHECK_DIRECTORY" -lMimiCore \
  "$ROOT/Sources/Mimi/OnboardingRequirements.swift" \
  "$ROOT/Sources/Mimi/UserPreferences.swift" \
  "$ROOT/Tools/MimiUXChecks/main.swift" \
  -Xlinker -rpath -Xlinker "$CHECK_DIRECTORY" \
  -o "$CHECK_DIRECTORY/onboarding-checks"
"$CHECK_DIRECTORY/onboarding-checks"
