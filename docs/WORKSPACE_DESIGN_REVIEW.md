# Workspace design preview

This feedback branch starts after UX PR16 merged. It is not a release-qualified
build and does not replace the installed app. Wait for user feedback before
opening a delivery PR or merging.

## Direction

The [official Codex interface](https://learn.chatgpt.com/images/codex/app/codex-app-basic-light.webp)
and [model control documentation](https://learn.chatgpt.com/docs/models) informed
the separation of navigation, content, and controls near the primary action.
Mimi keeps native macOS controls and readable transcript surfaces.

- Only session history scrolls in the sidebar. New session, search, Voice Type
  and Settings remain visible.
- A fixed recording bar groups input, language, model and Record.
- Transcript and Bilingual are explicit workspace views.
- All speech models fit in a popover with language support and size visible.
- Settings use a persistent section list. Onboarding has named steps and fixed
  Back/Continue actions. The dictation overlay explains its current phase.

## Shared components

`Sources/Mimi/MimiDesign.swift` owns metrics, cards, accessible Glass fallbacks,
quiet-button interaction states, model-option rows and transcript-pane headers.
`Sources/Mimi/WorkspaceControls.swift` owns shared input, language and model
controls, plus the recording bar. Views reuse these rather than duplicating
model choices or pointer treatment.

Quiet buttons have a rounded neutral hover background and a subtle pointer
press scale. Hover transitions take 120 ms. Reduced Motion removes movement
and animation; increased contrast strengthens the hover background. Disabled
controls stay dimmed and never display an active hover background. Native
prominent buttons, menus, pickers, Forms and list selection retain macOS behavior.

## Evidence

Screenshots use synthetic content and isolated preferences. Capture them with:

```sh
swiftc -parse-as-library scripts/ux_accessibility_audit.swift -o .build/mimi-ax-audit
swiftc -parse-as-library scripts/capture-workspace-review.swift -o .build/capture-workspace-review
.build/capture-workspace-review "$PWD/.build/Mimi.app/Contents/MacOS/Mimi" /tmp/mimi-workspace-review
```

The capture includes English/Japanese workspaces, light/dark appearances,
72 history entries, menu controls, settings, onboarding, dictation, captions,
model selection and a real pointer hover. Each captured process is owned by the
helper and terminated afterward. No user transcripts are read.

Development packaging and model-integrity checks passed. Core policy checks,
history/persistence, dictation lifecycle, model selection, speech exclusivity,
self-test, MimiE2E and MimiSessionE2E passed. The rendering matrix exercises
51 combinations of language, appearance and accessibility overrides; it is not
a spoken VoiceOver audit or a substitute for functional tests.

## Gates still open

- The full test script stops at the source-bound adaptive-speech gate because
  the previous evidence and generated executable identity belong to PR16.
  Regenerate and rerun source-bound evidence after the design is settled. Do
  not relabel previous reports or change thresholds.
- A fresh live field test stopped on its first selection case with the text/
  cursor ownership guard. Its synthetic field did not receive the expected
  Unicode replacement. Reproduce and diagnose this before delivery; do not
  claim the ten-case live field suite passed for this branch.
- Claude independent review remains unavailable until its login is renewed.
- User feedback, exact-head CI, final independent review and merge checks are
  required for delivery. No release or installation is part of this preview.
