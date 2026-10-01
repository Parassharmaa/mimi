# Translation while speech is still changing

The bundled Mimi translator now starts on the first temporary ASR hypothesis.
The Bilingual pane shows its replaceable translation in secondary text below
completed translations. Floating captions use the same worker implementation.
A new hypothesis replaces pending work without cancelling active inference.
This removes the caption's 140 ms debounce and the pane's wait for finalized
speech.

A matching final transcript reuses its immediately preceding successful
translation. Corrected final text is translated again. Session, language, and
utterance boundaries reject stale results. Failed partials retain the last good
translation within that utterance. Live updates and untranslated finalized
segments alternate so neither can starve the other. Each view owns a worker;
MLX inference runs through the shared translation actor. Translations remain
in memory, with the existing persistence and model safety rules unchanged.

The app also prepares both translation directions at normal AppStore startup.
This moves model loading and first Metal kernel evaluation ahead of recording.
Recording immediately after launch can still overlap preparation. Apple
Translation's existing fallback lanes remain finalized-only in the pane.

## Measurement on October 2, 2026 JST

These are controlled transcript replays on Mac15,6, macOS 26.5.1 (Build 25F80), using the pinned
73.4 MB development model. The model manifest SHA-256 is
`824214ae6434ef0abdf06e7b208077ff3014d372144ab905b64127bbe7c1b8f7`.
Both policies use the same full integrity-checked MLX
translator, unchanged weights and greedy decoder.

The metric ends when translated text is ready for publication. It excludes
speech recognition, audio delivery, SwiftUI rendering and compositor paint.
Numbers below describe warmed translation, not a cold application launch.

| Policy | Successful attempts | Median | p95 |
| --- | ---: | ---: | ---: |
| Initial final-only pane replay | 21/24 | 1,304.4 ms | 1,339.0 ms |
| Initial caption replay | 24/24 | 169.6 ms | 197.9 ms |
| Caption delay, in paired comparison | 24/24 | 178.6 ms | 197.1 ms |
| Immediate dispatch, paired comparison | 24/24 | 25.0 ms | 56.4 ms |
| Production worker, first publication | 24/24 | 21.5 ms | 33.0 ms |

Immediate dispatch reduces the paired median by 86%. The mean paired reduction
is 149.5 ms. An exploratory bootstrap over the eight fixtures, retaining all
three repetitions together, gives a 95% interval of 145.5–153.1 ms for that
mean reduction. Tail percentiles are exploratory with this small sample.

The final-only replay emits its final transcript 1,200 ms after its first
partial. Its measured delay includes that specified wait; it is not a measured
average speech endpoint delay. The initial baseline was collected before the
worker change. The subsequent immediate/delayed comparison counterbalances
policy order on the same short partials, reducing sensitivity to run order.

There are eight owned utterances, four in each direction, with three dependent
repetitions per utterance. Worker replays emit growing prefixes at 0, 220 and
440 ms, and a final at 1,200 ms. The independent unit is the utterance. Every
attempt, output, failure, and publication is retained in the reports. The final
drain cap is five seconds; a missing final fails parity. No per-call timeout or
retry was used in the original collection. This is an exploratory latency
screen, not a translation-quality or long-meeting evaluation. Input/output
model token counts and GPU cache telemetry were not collected. All inference
is local and incurred no external API spend.

All seven utterances with successful baseline finals produce exactly the same
final translation in all three repeats. The eighth sentence, "I would like to
book a room for two nights.", fails the existing protected-number check in
both policies. Across the 75 production inference calls, nine fail: that travel
sentence and its growing prefix account for six, and an incomplete Japanese
microphone phrase accounts for three. Failures are retained rather than
counted as successful low-latency translations. All eight utterances produce
an initial partial translation. Incomplete translations can change substantially
as more source text arrives; their meaning was not human-scored in this screen.

Cold first full calls measured 1,008/310 ms in the initial baseline process and
369/312 ms in the subsequent process for EN→JA/JA→EN. Filesystem and driver
caches were not cleared, so those numbers cannot establish a cold-load speedup.
A separate fresh-process startup preparation took 605.5 ms for both directions;
first short partials after preparation took 22.3 and 28.7 ms. This shifts work
to startup rather than making model loading itself faster.

## Release pack compatibility

The release archive keeps the stable translation pack. The same eight-utterance,
three-repeat worker replay was also run against `App/Resources/TranslationModels`.
All 24 replays produced a first partial translation; 21 final outputs exactly
matched the stable final-only baseline, and three retained its existing travel
sentence safety rejection. The incomplete microphone prefix was also rejected,
with the previous successful translation retained. Compiler activity could
have overlapped this compatibility run, so its timings are not used for the
primary speed comparison above.

- [Stable pack baseline](benchmarks/live-translation-stable-baseline.json)
- [Stable pack worker replays](benchmarks/live-translation-stable-instant.json)

## Speech qualification for this release

The existing speech gate binds its reports to all app source files, so this
translation change also required a fresh speech qualification. Two paced Debug
attempts failed the existing runtime gates. Japanese lost 14,400 and 8,000 queued
samples and finalized in 2.548 and 2.564 seconds. English lost no audio but
finalized in 1.360 and 1.325 seconds. Both paused Debug outputs exactly match
the previous reports. The speech engines, session/capture implementation, and
core transcript code are unchanged by this PR.

The selected qualification uses a separately compiled production Release
executable, the configuration used by the downloadable archive. It has the
same source identity and uses the same pinned speech weights, profiles and
queue capacity. The existing recognition, zero-drop and 1.1-second stop gates
remain unchanged. Local qualification reuses the matching MLX 0.31.4 Metal
library because the local Metal compiler is unavailable; GitHub packaging
builds its own library. The configuration change was chosen after the Debug
failures, so this is release validation rather than an unbiased comparison of
compiler modes. All Debug attempts are retained under
`Research/speech/work/long-form-{en,ja}-v1/*instant-translation*debug*.json` and
`*instant-translation-attempt1-v4.json`.

The four Release cases pass. Japanese gapless CER is 6.03% and paused CER is
6.57%; English gapless WER is 12.36% and paused WER is 5.90%. Both paced cases
lose zero audio. Stop finalization is 0.821 seconds in Japanese and 0.724 seconds
in English. These are single fixed-case timings, not independent tail estimates.
Both paused outputs exactly match the original baseline.

The selected reports are `*instant-translation-v4.json`; their executable
SHA-256 must match the [recorded Release protocol](benchmarks/live-translation-release-speech-protocol.json).
The protocol was recorded after the first Release replay started and before
any Release result was available. CI verifies the reports' source identity and
gates; it does not replay the long audio on the hosted runner.

## Verification and reproduction

The core self-test, session E2E, local fallback contract and live-worker
verification pass. The worker checks cover continued publication under rapid
revisions, newest pending work, unchanged text, failed requests, promotion to
finals, corrected finals, stale results after a reset/session/language change,
and fair progress of history and live speech.

A native light-appearance window was inspected with zero finalized segments.
It showed the temporary source "Can we move the meeting to tomorrow morning?"
and the secondary translation "明日の朝、会議を移動できますか。" in the
Bilingual pane. The fixture uses temporary in-memory session storage and does
not capture microphone audio. This confirms the UI wiring, not paint latency.

```sh
scripts/translation/run-live-translation-benchmark.sh \
  .build/translation-models/blazeofchi/mimi-en-ja-mlx-development-v1-e6a02099fc62d9f351ffab1fa1efbd46b2c512a0/model

.build/debug/Mimi --verify-live-translation .build/live-translation-verification.json

.build/Mimi.app/Contents/MacOS/Mimi --e2e-window --e2e-screen transcript \
  --e2e-state provisional-translation --e2e-live-translation
```

The benchmark runner writes to `.build/live-translation-benchmark` by default.
Retained evidence from this run:

- [Initial baseline](benchmarks/live-translation-baseline.json)
- [Paired comparison and production replays](benchmarks/live-translation-instant.json)
- [Startup preparation](benchmarks/live-translation-prewarm.json)
- [Worker verification](benchmarks/live-translation-verification.json)
