# Mimi UX overhaul

## Outcome and scope

Make recording, translation, captions and Voice Type discoverable, usable by
keyboard and assistive technology, and consistent with Apple's macOS design.
Use native Liquid Glass for navigation and controls on macOS 26; keep readable
content surfaces and solid/material fallbacks on macOS 15 and accessibility
display modes. Preserve existing recordings, preferences and model choices.

The initial scope is eight surfaces and approximately 50 user interactions:
onboarding, menu-bar controls, transcript, history, translation, captions,
Voice Type and settings. Findings may change the count, but must include a
reproduction or source-level explanation before they become implementation work.

This is separate from the Phonon integration. Branch
`codex/mimi-liquid-glass-ux` starts from the Phonon feature commit so the UI can
exercise all model choices. Parent PR15 must qualify before the UX PR can merge.

## Definition of done

- Every documented critical/high-priority functional or accessibility finding
  has a verified fix, or a specific reason it is outside this change.
- The tested recording, translation, history/export and dictation flows work
  with the real app, including cancellation and permission/model failures.
- Controls expose meaningful names, roles and values; keyboard actions,
  focus order and cancellation work without a pointer.
- Light/dark, English/Japanese, small/large windows, increased contrast,
  reduced motion and reduced transparency have no clipped essential controls
  or unreadable content on the tested Mac.
- Native Glass is gated to supported macOS versions and applied to functional
  chrome instead of obscuring transcripts.
- Before/after evidence uses synthetic content. Private live screenshots and
  the user's saved transcripts are never included in public artifacts.
- Local gates and live PR checks pass at the final head, followed by an
  independent review. Merge authorization is already provided by the user.

## Work sequence

1. Read the design and verification principles. Complete.
2. Qualify Phonon PR15. Resolve evidence versioning and investigate its latency
   gate without weakening tests or changing historical results.
3. Capture the existing UI and build a reproducible interaction/accessibility
   audit. Treat rendering-only smoke tests as incomplete functional evidence.
4. Record findings in UX_AUDIT.md with severity, reproduction, expected
   behavior, affected files and a test.
5. Fix functional/data-handling problems first, one verified commit at a time.
6. Apply the native navigation, toolbar, materials and typography design.
   Reuse platform conventions and existing native split-view foundations.
7. Fix keyboard, focus, labels, localization and accessibility display modes.
8. Exercise the complete flow matrix and compare synthetic before/after
   screenshots. Recheck model/capture ownership and persistence behavior.
9. Open the separate UX PR, run independent code/UX review, address findings,
   and merge only the verified sequence of PRs.

## Evidence and decisions

`docs/ux-decisions.tsv` is the canonical decision trail. Test scripts, synthetic
screenshots and results are evidence; agent summaries and a successful compile
are not substitutes for observing the product.

Live observations of installed Mimi are kept locally in
`/Users/paras/Documents/Codex/2026-10-01/mimi-ux-audit`. The initial screenshot
contains private saved sessions and is intentionally excluded from the repo.
