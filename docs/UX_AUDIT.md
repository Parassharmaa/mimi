# Mimi UX audit

Baseline: Phonon development commit 84cd1c4, macOS 26.5.1, Apple M3 Pro.
The installed app is observable through macOS Accessibility and window
screenshots. Initial live content is private; published evidence will use
synthetic sessions.

Findings start as hypotheses until reproduced or confirmed from the full
implementation. The audit covers usability, functional behavior and native
accessibility semantics. Web ARIA rules are not applied literally to SwiftUI.

| ID | Priority | Finding | Evidence/status | Verification |
| --- | --- | --- | --- | --- |
| UX-001 | High | Historical session headers use current input/model/language settings rather than distinguishing saved content from current capture configuration. | Live screenshot and TranscriptWindow.sessionStrip; needs synthetic reproduction. | Select a saved Japanese session while current input is English; verify accurate context and recording destination. |
| UX-002 | High | History deletion ignores storage errors and removes the item in memory before confirming durable persistence. | AppStore.clearTranscript uses try? historyStore.save. | Inject a storage failure; item must remain recoverable and error must be visible. |
| UX-003 | Medium | Current transcript is a button rather than a selected source-list row, so selection/wayfinding differs from saved sessions. | TranscriptHistorySidebar; verify keyboard and VoiceOver behavior. | Navigate current/history by arrow keys and inspect selected state. |
| UX-004 | Medium | Long transcript has no visible search or export action in the main toolbar. | Live screenshot and transcriptToolbar; discover menu alternatives before fixing. | Find text and export the selected session using pointer and keyboard. |
| UX-005 | Medium | Settings, Voice Type and source/model preparation need clearer entry points from the transcript workflow. | Live screenshot; requires full navigation audit. | Complete first-use setup from main window without relying on prior product knowledge. |
| UX-006 | High | Voice Type inserts before selected text and duplicates it on cancellation. | Reproduced in a synthetic AppKit field: `hello world` became `hello worldworld`; original smoke incorrectly passed. | Full-field replacement/rollback checks, including Japanese and emoji selection. |
| UX-007 | High | Moving to another field in the same app can direct global key events to the wrong destination. | Confirmed from complete insertion implementation. | Two-field fixture changes focus between partial results; second field must remain unchanged. |
| UX-008 | High | Editing text or moving the cursor inside the same field can cause subsequent partials or rollback to overwrite user edits. | Independent source review. | Intervening edit/caret fixtures must refuse further mutation and preserve edits. |
| UX-009 | High | Cancelled asynchronous startup/finalization can release shared speech ownership too early or lose the rollback target. | Independent source review; production task-owner suspension test passes after fix. | Five lifecycle invariants plus actual dictation cancellation path. |
| UX-010 | High | Archival failure is reported but New Session/Start still clears the current transcript. | Confirmed from full caller/callee flow; fixture integration verifier added. | Eight app persistence invariants, including archive/start/clear failures. |
| UX-011 | High | Malformed or unreadable history is treated as empty and can be overwritten. | Failing-before store regression reproduced; actual store test now passes. | Exact byte preservation, missing/valid/malformed/repaired/unreadable cases. |
| UX-012 | High | Menu preview displays current text while Copy/Delete targets the selected historical session. | Confirmed source mismatch. | Historical-session fixture plus live copy/delete actions. |
| UX-013 | High | Deletion confirmation targets the selection/current content at confirmation time rather than the original target. | Confirmed source flow and independent review of current-session race. | Capture history UUID/current-document snapshot; reject changes and active recording. |
| UX-014 | High | Offline onboarding requires Apple speech/translation downloads even when bundled local models are selected. | Confirmed source flow; 36 provider/permission assertions pass after fix. | Live first-run selection of bundled Whisper and Phonon without Apple preparation. |
| UX-015 | Medium | Permission UI uses a green check for permissions that were never granted but are unnecessary for the selected source. | Confirmed source semantics; actual policy helper tests pass. | Distinct Granted/Not required/Request/Denied states. |
| UX-016 | Medium | Voice Type preparation cannot be cancelled with Escape, and all active phases are announced as listening. | Confirmed source. | Escape during setup, startup/listening/finishing AX labels. |
| UX-017 | Medium | Settings access status can remain stale after macOS permission changes, and setup links open the wrong tab. | Confirmed source. | App/window activation refresh and native Settings deeplinks. |
| UX-018 | Medium | Voice Type error disappears after four seconds with no persistent recovery context. | Confirmed source. | Persistent latest issue in Voice Type settings. |
| UX-019 | Medium | Caption repositioning animates despite Reduce Motion; setup progress lacks a spoken step count. | Confirmed source; code checks only after fix. | Reduced-motion fixture and AX step label. |
| UX-020 | Medium | Several user-visible controls and privacy descriptions are English-only or describe Apple despite selecting local engines. | Confirmed source. | Japanese interface matrix and selected-provider privacy inspection. |
| UX-021 | High | Terminal keyboard fallback can delete existing shell text when the cursor or prompt changes. | Actual LiveTextEdit simulation reproduced deletion; AX history does not expose the writable prompt cursor. | Refuse Terminal at capture before model/microphone setup; remove unverified Backspace fallback and disclose supported-field scope. |
| UX-022 | High | First-use setup can prepare Phonon/Apple but enable Voice Type with an unavailable Whisper default. | Stable bundle and complete setup source flow confirm mismatch. | 210 real preference/policy assertions; default follows prepared engine only when no saved model choice exists. |

## Implementation and evidence status

Functional/data safety changes precede visual changes. Commits `071af3a`,
`207e293` and `208194a` cover persistence, provider-aware setup and dictation
safety. Remaining interface changes are being compiled and tested before the
separate UX PR is opened. Parsing does not establish runtime correctness.

Verified so far:

- Baseline selected-text duplication in a real synthetic AppKit text field.
- Actual history store regression, including exact damaged-file preservation.
- 210 provider, permission and persisted preference-policy assertions.
- Five asynchronous ownership invariants using the production task owner.
- Native Settings coordinator SDK typecheck and edited-source parse checks.

The full application builds. Eight app persistence invariants, five lifecycle
ownership invariants and eight real text-field checks have passed. Live history
Copy, named Delete/Cancel, Export/Cancel, Japanese search/no-match state, Voice
Type Settings deeplink, microphone callbacks and native Start/Stop have passed.
The original 47-scenario state/light/dark rendering matrix passes. Tests using
physical focus run serially, and the field harness pins its fixture PID.

Pending: final-head repeat after the last safety changes, actual file export,
extended Japanese/accessibility-display matrix, fresh source-bound speech
controls and exact-head CI. AX semantics are inspected; a complete spoken
VoiceOver audit is not claimed. Native macOS 15 compatibility is compile-gated,
not verified on a second OS installation. Claude review could not run because
its login expired; independent code review found and resolved the ownership
and Terminal defects.

Private installed-app screenshots remain local. Public evidence uses only
synthetic English/Japanese content. Developer fixtures use transient transcript
and history storage and do not register the user's dictation shortcut.

## Flow matrix

| Flow | Baseline | Fixed | Evidence |
| --- | --- | --- | --- |
| First launch and setup | Not tested | Not tested | Pending synthetic harness |
| Microphone recording/start/stop/cancel | Prior pipeline checks only | Not tested | Pending real UI actions |
| Output/app/display capture and permission recovery | Not tested | Not tested | Pending |
| English/Japanese/model selection | Prior code checks only | Not tested | Pending live UI |
| History selection/copy/delete/recovery | Source observations | Not tested | Pending synthetic sessions |
| Transcript search/export | Source observations | Not tested | Pending |
| Translation/empty/partial/failure states | Rendering smoke only | Not tested | Pending |
| Floating captions/settings/window lifecycle | Rendering/lifecycle smoke only | Not tested | Pending |
| Voice Type/shortcut/permissions/secure field/cancel | Prior insertion and ownership checks | Not tested | Pending full UI path |
| Keyboard/VoiceOver/full keyboard access | Not tested | Not tested | Pending AX inspection |
| Light/dark/contrast/transparency/motion/localization | Light/dark rendering smoke | Not tested | Pending matrix |
