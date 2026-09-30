#!/bin/zsh
set -euo pipefail
ROOT="${0:A:h:h}"
APP="${1:-$ROOT/.build/Mimi.app/Contents/MacOS/Mimi}"
[[ -x "$APP" ]] || { print -u2 'Build Mimi before running the UX matrix.'; exit 1; }
cases=0
for language in english japanese; do
  for appearance in light dark; do
    for mode in normal accessible; do
      preview_flags=()
      if [[ "$mode" == accessible ]]; then
        preview_flags=(--e2e-reduce-motion --e2e-reduce-transparency --e2e-increase-contrast)
      fi
      for screen in menu transcript onboarding captions voice-typing settings-voice; do
        "$APP" --e2e-window --e2e-screen "$screen" --e2e-state ready \
          --e2e-language "$language" --e2e-appearance "$appearance" \
          "${preview_flags[@]}" --e2e-auto-quit
        cases=$((cases + 1))
      done
    done
  done
done
for state in history empty; do
  "$APP" --e2e-window --e2e-screen transcript --e2e-state "$state" --e2e-auto-quit
  cases=$((cases + 1))
done
"$APP" --e2e-window --e2e-screen onboarding --e2e-state local-setup --e2e-engine phonon --e2e-auto-quit
print "PASS: $((cases + 1)) UX rendering cases. Accessibility overrides exercise app fallbacks, not changed macOS preferences or spoken VoiceOver."
