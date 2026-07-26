# Mimi V19 guarded local expert cascade

Date: 2026-07-26

Status: stored-output public metrics passed, but the exact Swift runtime screen
is rejected. This does not authorize replacing Mimi's bundled resources,
protected evaluation, release, or publication.

## Outcome

V18 showed that a single 54.46 MB projected shared Marian checkpoint can learn
both directions, but its final JA→EN chrF++ was only 19.85 against the frozen
50.0 floor. V19 therefore keeps the stronger public q4 direction specialists
and uses Mimi's current local pair as a deterministic safety fallback.

The exact developer pack is 146,816,974 bytes. It remains below the 150 MB
ideal and uses no Apple or cloud translation:

```text
source segment
      |
      v
stronger public q4 direction expert
      |
      +-- normal output ------------------------------> return expert
      |
      +-- empty or 3x repeated token phrase
                           |
                           v
                 current bundled local q4 direction
                           |
                           +-- safe termination ------> return fallback
                           |
                           +-- empty or repeated ------> fail closed
```

## Public 200-case screen

The screen composes the exact stored q4 outputs from the candidate and bundled
reports. It uses the same 100 cases per direction and 400 total segments,
including 40 six-segment documents.

| Direction | System | chrF++ | BLEU |
|---|---|---:|---:|
| EN→JA | Bundled Mimi | 28.1283 | 9.1427 |
| EN→JA | Stronger pair | 28.4063 | 9.6287 |
| EN→JA | V19 guarded cascade | **28.5543** | **9.6314** |
| JA→EN | Bundled Mimi | 50.0632 | 25.7699 |
| JA→EN | Stronger pair | 55.1903 | 30.6207 |
| JA→EN | V19 guarded cascade | **55.1903** | **30.6207** |

The guard selected the expert for 398/400 segments and the bundled local
fallback for two EN→JA segments:

- `development-accuracy-v1:document:jlt:law-3518:en-ja:segment-05`
- `development-accuracy-v1:sentence:jlt:law-3215:tu-102:en-ja:segment-01`

Neither fallback output was empty or repetitive. The document fallback removes
the known casino-law nontermination. It also slightly improves aggregate
EN→JA chrF++ and BLEU; JA→EN is unchanged from the stronger specialist.

## Pinned COMET-22

The exact composed report was scored with the existing pinned Apache-2.0
`Unbabel/wmt22-comet-da` revision and compared case-by-case with both parents
using 10,000 paired bootstrap samples.

| Direction | Bundled Mimi | Stronger pair | V19 cascade | V19 vs stronger pair |
|---|---:|---:|---:|---:|
| EN→JA | 0.8486 | 0.8669 | **0.8702** | +0.00328 `[+0.00000, +0.00888]` |
| JA→EN | 0.7825 | 0.8192 | **0.8192** | +0.00000 `[0, 0]` |

V19 improves over bundled Mimi by +0.02154 EN→JA, with paired 95% interval
`[+0.00404, +0.04262]`, and +0.03667 JA→EN, interval
`[+0.01889, +0.05758]`. The six EN→JA long legal documents improve +0.03871
COMET over the stronger pair; JA→EN long legal is unchanged. This passes the
public COMET non-inferiority gate without weakening either direction.

This is a stored-output result, not the final app-path result. The stored
composition modeled the preregistered empty/repetition fallback but did not
model the runtime's existing critical-token and plausibility guards.

## Actual Swift/MLX evidence

The runtime change is developer-only and does not alter app resources.

- `swift build --product Mimi`: passed.
- Pure runtime cache/critical-token/repetition contract: passed.
- Normal EN→JA `"Start recording"`: selected expert and returned `録音開始`.
- Normal JA→EN `窓辺に向かった。`: selected expert and returned
  `I headed for the window.`
- Exact casino-law failure segment: the expert loop was detected and the
  current bundled local generalist returned a finite translation.
- A new exact Swift cascade benchmark records per-case routing, deterministic
  warm outputs, failures, fallback IDs, latency, bundle bytes, and process peak
  RSS.
- Its 12-case canary initially found two valid calendar translations rejected
  by the critical-number guard. Lexical English month names are now normalized
  to their numeric month only when adjacent to a day. Both valid date
  translations pass; two wrong-month negative controls still fail.
- The corrected 12-case canary passes with 12 expert outputs, zero failures,
  the 146,816,974-byte pack, and 424,853,504 bytes peak process RSS. Its timing
  is not a promotion result because the V18 Metal evaluator was concurrent.
- The complete developer pack passed Mimi's authenticated model-pack validator.

### Full 400-segment runtime screen

The exact Swift cascade rejects V19:

| Direction | Segments | Expert | Local fallback | Failed closed |
|---|---:|---:|---:|---:|
| EN→JA | 200 | 166 | 5 | **29** |
| JA→EN | 200 | 154 | 4 | **42** |
| Total | 400 | 320 | 9 | **71** |

All 71 failures are critical-token mismatches after both compact local parents
were tried. The largest slice is JA→EN long legal (15/36), followed by EN→JA
long news (10/48). Several are genuine severe number failures—for example,
`300 miles (480 km)` becomes `3 km`, article 74 becomes 71, or list item 226
becomes 222. Others are valid bilingual surface changes that the strict guard
cannot currently prove, such as English number words becoming Japanese digits.

Therefore the real runtime's 320/9/71 routing supersedes the simulated 398/2
routing for every integration decision. The stored COMET result remains
reproducible evidence about those exact stored hypotheses, but it cannot be
claimed as app-path quality.

### Larger independent fallback probe

The 448,893,412-byte FP16 HPLT v2 pair was tested as an independent third
fallback. It strictly recovers only 14/71 failures and leaves 57 unresolved.
The mean sentence chrF++ of those 14 is 22.28, and strict surface matching
admits at least one lexical-number omission (`one month` disappears). The
strong expert pair plus HPLT would be 522,319,220 bytes before app overhead.
HPLT is rejected before MLX conversion or quantization.

## Remaining gates

V19 is rejected for shipping in its current form:

1. Train or select a stronger number- and legal-structure-preserving fallback,
   or validate source-normalized constrained numeric decoding on adversarial
   and held-out suites.
2. Keep the strict app-path guard; do not promote the offline typed numeric
   ablation because it has reference-disagreement accepts.
3. Rerun the complete public Swift runtime screen before opening protected
   evidence or measuring isolated latency.
4. Resolve inherited attribution/share-alike status only after runtime safety
   passes.

The exact contract is
`Research/translation/guarded-expert-cascade-v19-contract-2026-07-26.json`;
the machine-readable public result is
`Research/translation/guarded-expert-cascade-v19-preliminary-result-2026-07-26.json`,
and the pinned learned-metric result is
`Research/translation/guarded-expert-cascade-v19-comet-result-2026-07-26.json`.
The superseding app-path decision is
`Research/translation/guarded-expert-cascade-v19-swift-runtime-result-2026-07-26.json`;
the rejected large-fallback probe is
`Research/translation/guarded-expert-cascade-v19-hplt-fallback-result-2026-07-26.json`.
Rebuild the developer pack without changing app resources:

```sh
scripts/translation/package_guarded_expert_cascade_v19.sh \
  Research/translation/work/guarded-expert-cascade-v19/bundle
```
