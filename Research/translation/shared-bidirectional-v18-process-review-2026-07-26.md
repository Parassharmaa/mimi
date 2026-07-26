# V18 independent code and process review

Date: 2026-07-26

Status: **pre-restart findings addressed in code/contract; replacement run not
yet started**

This review covers the shared 92.04M-parameter Marian capacity control, its
licensed-human mixture, the function-preserving FFN widening, the phase-one
trainer, and the evidence boundary before any q4, app, or publication work.
The review was read-only and independent of implementation.

## Verdict

Do not accept either abandoned launch as V18 experimental evidence. Both
surviving outputs are the exact step-zero initializer. The second launch entered
step-250 evaluation but returned, printed, and serialized no trained metric or
checkpoint before it was stopped.

One replacement 1,000-update run is defensible only after the frozen contract
authenticates all corrective controls below.

| Review finding | Severity | Resolution before replacement run |
| --- | --- | --- |
| Scheduled checkpoints were only saved when they became best | P0 | Persist immutable model/tokenizer directories at 250/500/750/1,000 regardless of score; best becomes a manifest pointer |
| No optimizer/RNG/sampler recovery state | P0 | Keep one rolling, authenticated full resume state for the newest immutable checkpoint; retain one because local disk has only about 6.7 GB free |
| Two updates would contain 30 rather than 32 examples | P1 | Freeze `drop_last=true`; 12,066 rows become 3,016 full microbatches, exactly divisible by accumulation 8 |
| Objective logging represented only the last microbatch | P1 | Record the mean over all eight microbatches and the mean over each 250-update interval |
| Random selection was heavily KFTT-skewed | P1 | Freeze an explicit 512-case direction/domain/length-stratified selector and retain every scheduled checkpoint for external regression evaluation |
| Cache-only validation change lacked parity evidence | P1 | Run the full frozen 512-case selector cached and uncached; require exact tokens, loss, and chrF++ |
| Trainer recorded rather than enforced the contract | P2 | Require the contract and fail closed on exact tool, data, selection, model, recipe, and runtime hashes |
| Dataset release eligibility was too broad | P1 for distribution | Mark distribution ineligible until structured URL/date/transformation/content-hash and offline attribution review is complete |
| Real widening parity covered only six EN-JA cases for 20 steps | P2 for release | Keep the transform as training initialization evidence only; require a tracked full-to-EOS harness before release |

## Positive independent checks

- The FFN widening transform is mathematically correct. Copied `fc1`
  features sit behind exact-zero new `fc2` columns.
- The widened checkpoint contains exactly 92,043,009 float32 parameters and
  occupies 368,201,220 bytes.
- Direction-specific teacher routing, masked KL, temperature scaling, and
  projection-free encoder-state MSE are mathematically sound.
- The materialized mixture has exactly 6,033 rows per direction.
- Independent scans found zero exact canonical train/validation overlap,
  including reversed directions; zero exact English-side, Japanese-side, or
  source-ID overlap; and zero character-5 Jaccard overlap above 0.8.

## Frozen replacement selector

The tracked selector contains exactly 256 cases per direction:

| Direction | UI | Conversation | News | Wikipedia short-pair | Wikipedia long-pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN→JA | 13 | 128 | 0 | 91 | 24 |
| JA→EN | 13 | 64 | 64 | 91 | 24 |

The source validation pool contains no EN→JA news row, no legal row, and no
JA→EN source of at least 100 Unicode characters. This selector is therefore
checkpoint-selection evidence, not sufficient domain coverage. Every immutable
checkpoint must still face the separate legal, safety, and true long-input
suites before q4 selection.

## Cache parity result

The exact widened initializer passed the full 512-case check:

- generated token IDs: identical;
- loss delta: `0.0`;
- EN→JA chrF++ delta: `0.0`;
- JA→EN chrF++ delta: `0.0`;
- macro chrF++ delta: `0.0`.

The balanced step-zero baseline is EN→JA `30.6907`, JA→EN `1.1677`, and macro
`15.9292`. These are internal-selection values and are not directly comparable
with the public 200-case bundled/candidate chart.

## Remaining boundary

Passing phase one cannot authorize integration. The projected q4 package is
54.46 MB, but actual q4 quality, warm p95 below 250 ms, peak RSS below 500 MB,
Swift/Python token parity, native variable-FFN support, held-out 400+400
superiority, long-document safety, complete attribution, and a non-Apple
failure path all remain mandatory.
