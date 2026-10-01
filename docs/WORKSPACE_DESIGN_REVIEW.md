# Workspace design preview

This feedback branch starts after UX PR16 merged. It is not a release-qualified
build and does not replace the installed app. PR17 carries the current iteration.
The user authorized merging only after qualification and independent review pass.

## Direction

The [official Codex interface](https://learn.chatgpt.com/images/codex/app/codex-app-basic-light.webp)
and [model control documentation](https://learn.chatgpt.com/docs/models) informed
the separation of navigation, content, and controls near the primary action.
Mimi keeps native macOS controls and readable transcript surfaces.

- One Sessions list scrolls in the sidebar. Its plus action, search, Voice Type
  and Settings remain visible. The recording session has an activity indicator.
- A fixed recording bar groups input, language, model and Record.
- Transcript and Bilingual are explicit workspace views.
- All speech models fit in a popover with language support and size visible.
- Spoken language uses one flat popover with no submenu.
- Record appends to the open session. Plus creates a separate empty draft, and
  Record from the empty state creates the first session.
- Settings use a persistent section list. Onboarding has named steps and fixed
  Back/Continue actions. The dictation overlay explains its current phase.

## Shared components

`Sources/Mimi/MimiDesign.swift` owns metrics, cards, accessible Glass fallbacks,
quiet-button interaction states, model-option rows and transcript-pane headers.
`Sources/Mimi/WorkspaceControls.swift` owns shared input, language and model
controls, plus the recording bar. Views reuse these rather than duplicating
model choices or pointer treatment.

The working transcript carries an optional stable session identity with its
original start date and source. Existing raw files still decode without this
field. Resuming copies the saved document without regenerating segment IDs,
languages or dates. History is upserted by session UUID, never by matching text.
Working text is persisted before switching owners; failed writes retain the
previous working value. Sidebar browsing stays separate from the recording
destination. Both Start and Stop reserve a transition synchronously.

Quiet buttons have a rounded neutral hover background and a subtle pointer
press scale. Hover transitions take 120 ms. Reduced Motion removes movement
and animation; increased contrast strengthens the hover background. Disabled
controls stay dimmed and never display an active hover background. Native
prominent buttons, menus, pickers, Forms and list selection retain macOS behavior.
The shared quiet style is the default for secondary actions in all app surfaces;
explicit prominent buttons keep native primary-action styling. Destructive
actions retain their semantic red label.

Popovers retain macOS-native shadow, outline and arrow treatment. Screenshot
capture must not use `screencapture -o`, which strips these and makes the popup
look flat. Settings navigation uses the same window surface as its content,
without an extra sidebar tint. Mimi Speech is labelled "Custom model" in the
chooser; upstream model attribution and distribution licenses remain intact.

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
Global shortcut registration is disabled at the registration boundary in
verification processes, including enabled Voice Type previews. Disabled
registration is not displayed as a shortcut collision.
The enabled Voice Type capture asserts that no collision warning appears.

Live field tests explicitly activate their owned fixture before starting Mimi.
Focus/caret/edit adversaries are triggered when the first partial appears,
instead of racing against process-startup timers. All ten cases passed with
unchanged whole-field and second-field-isolation assertions. Evidence is local
in the temporary fixture directory printed by the test script.

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
- Claude independent review remains unavailable until its login is renewed.
- User feedback, exact-head CI, final independent review and merge checks are
  required for delivery. No release or installation is part of this preview.
