# Mimi V19 guarded local expert cascade

Date: 2026-07-26

Status: public development screen passed. This does not authorize replacing
Mimi's bundled resources, protected evaluation, release, or publication.

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

## Actual Swift/MLX evidence

The runtime change is developer-only and does not alter app resources.

- `swift build --product Mimi`: passed.
- Pure runtime cache/critical-token/repetition contract: passed.
- Normal EN→JA `"Start recording"`: selected expert and returned `録音開始`.
- Normal JA→EN `窓辺に向かった。`: selected expert and returned
  `I headed for the window.`
- Exact casino-law failure segment: the expert loop was detected and the
  current bundled local generalist returned a finite translation.
- The complete developer pack passed Mimi's authenticated model-pack validator.

## Remaining gates

V19 is not ready to ship yet:

1. Run COMET-22 on the exact composed report and compare it with both parents.
2. Run the complete long-document, critical-meaning, negation, typed-token, and
   protected suites without changing the guard.
3. Measure warm p50/p95, fallback-tail latency, preparation time, and peak RSS
   using the exact Swift cascade pack.
4. Resolve the pack's inherited attribution/share-alike distribution status
   against the already-public candidate release sidecars.
5. Only after all gates pass, copy the authenticated pack into
   `App/Resources/TranslationModels`, rebuild the app, and run release parity.

The exact contract is
`Research/translation/guarded-expert-cascade-v19-contract-2026-07-26.json`;
the machine-readable public result is
`Research/translation/guarded-expert-cascade-v19-preliminary-result-2026-07-26.json`.
Rebuild the developer pack without changing app resources:

```sh
scripts/translation/package_guarded_expert_cascade_v19.sh \
  Research/translation/work/guarded-expert-cascade-v19/bundle
```
