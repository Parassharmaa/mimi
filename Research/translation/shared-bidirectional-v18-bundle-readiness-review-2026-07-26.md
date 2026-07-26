# V18 shared-model bundle readiness review

Date: 2026-07-26

Status: **development design only; no app integration, model promotion, or
release authorization**

## Verdict

The V18 shared Marian candidate cannot be bundled or loaded by the current
native runtime. Mimi's shipped two-model q4 pack remains the production
translator and must stay byte-pinned until a trained full-precision checkpoint,
its exact q4 conversion, and the complete promotion surface pass.

An independent read-only review confirmed that the current shipped pack,
license verifier, native pack validation, cache/fallback CLI contracts, pair
packaging tests, and release-artifact staging tests pass. It made no repository
changes.

## Why the current runtime cannot load V18

The candidate intentionally uses one physical model for both EN→JA and JA→EN:

| Property | V18 requirement | Current Swift assumption |
| --- | ---: | ---: |
| Physical models | 1 | 2 directional engines |
| Encoder / decoder layers | 6 / 6 | 6 / 6 |
| Model width / heads | 512 / 8 | 512 / 8 |
| Encoder / decoder FFN | 4,608 / 4,608 | 2,048 / 2,048 |
| Vocabulary | 32,001 | 32,001 |
| Direction control | `<2ja> ` / `<2en> ` source prefix | raw source text |
| Projected q4 pack | 54.46 MB | not yet materialized |

`MarianMLXTranslationModel.swift` constructs fixed 2,048-wide feed-forward
layers before exact weight verification, so FFN-4,608 weights fail to load.
`ExperimentalMLXTranslationEngine.swift` caches neural runtimes by source
language. Pointing both logical directions at one directory would therefore
load and retain the same physical model twice. Native tokenization also omits
the direction prefix that Python MLX applies.

The existing pair and MoE root formats require two logical generalists and
direction-specific child manifests. Reusing either schema would hide the
one-model invariant and risk duplicating weights in the bundle.

## Conditional shared-pack contract

Only after a V18 full-precision and exact-q4 quality win, introduce a distinct
`mimi-mlx-marian-shared-v1` development format with these fail-closed
properties:

1. exactly one safe relative engine path and one physical
   `model.safetensors`;
2. exactly two logical directions, both mapped to that engine;
3. child direction `bidirectional`;
4. exact authenticated prefixes:
   `{"en-ja": "<2ja> ", "ja-en": "<2en> "}`;
5. exact Marian architecture: width 512, eight encoder and decoder heads, six
   encoder and decoder layers, FFN width 4,608, vocabulary 32,001, and the
   frozen special-token IDs;
6. exact activation, tied/shared embedding, embedding scaling, static
   positions, decoder-start, EOS, and pad behavior;
7. exact tokenizer inventory and known prefix encodings;
8. exact exhaustive file sizes and SHA-256 values with no undeclared files;
9. q4/group-64 quantization; and
10. `doesNotAuthorizeAppIntegration: true` until a separate release contract
    authenticates all promotion evidence.

Legacy `mimi-mlx-marian-pair-v1` manifests should retain their current
behavior. Omitted architecture fields may map only to the exact legacy
FFN-2,048 configuration, never to arbitrary inferred shapes.

## Minimal native implementation after a q4 win

1. Make encoder and decoder FFN widths constructor parameters, while continuing
   to reject any change to V18's other frozen architecture fields.
2. Resolve the physical engine and direction prefix from the authenticated root
   manifest.
3. Prepend the prefix before tokenization and expose input token IDs to parity
   verification.
4. Cache runtimes by root revision plus physical engine path. Alternating
   directions must reuse one V18 runtime; the legacy pair must continue to use
   two.
5. Route parity and smoke commands through root-manifest resolution rather than
   appending a direction directory directly.
6. Initially accept the shared format only through the explicit developer model
   directory. Bundled discovery must reject a development-only shared pack.
7. Preserve `App/Resources/TranslationModels`,
   `App/Resources/TranslationLicenses`, and
   `verify_shipped_translation_pack.py` byte-for-byte until promotion.

Mimi currently fails a selected local candidate closed without falling through
to Apple Translation. This review preserves that product behavior. Automatic
shared-model-to-incumbent-model failover would be a separate feature requiring
a primary/fallback pack schema, both physical payloads, and explicit UI/runtime
tests; it is not part of the minimal shared-model loader.

## Dedicated packaging boundary

Do not weaken `package_elanmt_mlx.py`, which correctly represents the incumbent
two-engine pack. Add a dedicated shared packager only after q4 qualification.
It must validate the exact architecture and prefix map above, prove tokenizer
compatibility, emit one model file, rewrite an exhaustive child inventory, and
retain the development-only authorization bit.

Normal release staging and distribution must require a hash-bound release
contract for the shared format. Missing release artifacts must fail closed.
The current pair/MoE release-contract builder cannot be extended mechanically
because it assumes two generalist lineages and routed experts.

## Required tests

- Legacy manifests without architecture fields resolve only to the incumbent
  6e/6d FFN-2,048 shape and still translate both directions.
- The shared schema rejects missing or altered prefixes, wrong dimensions,
  wrong layers, wrong token IDs, unsafe paths, extra directions, hash
  mismatches, undeclared files, and bundled development-only activation.
- Python and Swift input-token and output-token parity is exact for both
  directions with cached and full-prefix decoding.
- Negative parity fixtures prove that an omitted or reversed prefix fails.
- Alternating directions retains one shared runtime; the legacy pair retains
  two; a root revision change invalidates both cache shapes.
- The real q4 checkpoint loads with exact weight verification and passes warm
  latency, peak RSS, and bundle-size limits while resident only once.
- The shared pack contains one model file; blocked or absent release contracts
  fail normal staging and distribution.
- Current shipped build, packaging, model hash, byte count, source revisions,
  notices, and license contract remain pinned.
- Promotion accepts the actual Swift parity engine identifiers; the current
  evaluator's accepted identifier must be reconciled with the cached and
  full-prefix identifiers emitted by Swift.

The shared schema and package tests must run in normal required CI, not only in
the larger research test script.

## Promotion order

The implementation remains gated in this order:

1. finish the immutable 1,000-update V18 training run;
2. select without protected or final held-out data;
3. reject any directional, domain, critical-meaning, or generation regression;
4. convert the surviving checkpoint to exact MLX q4/group-64;
5. run chrF++, BLEU, COMET, dual-family semantic judgment, long-document, and
   critical-token evaluations;
6. measure Apple-Silicon p50/p95 latency, peak RSS, and actual pack bytes;
7. prove Python/Swift input-token and output-token parity through the
   development-only shared format;
8. complete structured data/model attribution and distribution review;
9. freeze a hash-bound release contract; and only then
10. replace Mimi's bundled resources in a separate atomic change.

