# V14 rollout-conditioned repair plan

Date: 2026-07-26

Status: **preregistered; no gradient update has run**

## Why this arm

V12 clears its aggregate quality gates but fails after conditioning on its own
generated prefixes. Its teacher-forced chosen-token preference is already
97.7%, yet it can enter an `Article (25)` loop. V13 then confirms that the
loop is real while showing that several lexical detector categories are
misnamed. V14 therefore changes the training context and future evaluator,
not model capacity.

V14 starts from the rejected V12 step-50 checkpoint only as a research
initialization. The frozen safe parent remains the evaluation baseline and
retention teacher. V12's rejection is final.

## Licensed data and fresh validation

The positive corpus contains the same 7,104 authenticated, licensed-human
training sources used by V12. No V12 validation row becomes training data.
The new validation suite contains 768 Japanese-law test rows:

| Fresh V14 validation stratum | Rows |
| --- | ---: |
| negation | 96 |
| critical structure | 128 |
| repetition risk | 1 |
| terminology risk | 96 |
| omission risk | 16 |
| long legal | 192 |
| general legal | 239 |

Every validation source is absent from all V10 and V12 train/validation
splits. Both sides of every selected row are screened against the ten
protected suites using the established Unicode-normalized character-5-gram
policy. The output has zero protected hits and zero train/validation source
overlap. Every positive target is a distributable licensed-human reference;
no model rollout is a positive target.

The dataset manifest hashes to
`1cd2e3629513f4662c6c9ffd6854d463bd638f08c8001bdb73027db0dc03d245`.

## Free-running evidence

Before freezing training, V12 step 50 greedily translated all 7,104 training
sources with the same 192-token limit used for evaluation. The miner records
detector disagreements as discovery signals, not semantic ground truth:

| Rollout signal | Rows |
| --- | ---: |
| exact critical-token disagreement | 2,925 |
| typed critical-token disagreement | 2,969 |
| negation-detector disagreement | 1,088 |
| repetition/generation failure | 74 |

All 74 generation failures contain a real third contiguous repetition of a
3–16-token phrase. The hard set retains all generation failures, then fills a
stable SHA-256-ranked 2,048-row budget with other structure disagreements.
Each recovery row contains the actual free-running decoder prefix, the next
repeated token as negative evidence, and EOS as the only recovery target.
Rollout translations are never used as positive translations.

The rollout manifest hashes to
`f93ecd7d724e37f468321cca8fbf3e9ac472ee290bb2f979daa380cb5dddd4e4`.

## Frozen one-arm recipe

Only checkpoints 25 and 50 may be evaluated:

- learning rate 2e-7, effective batch 32, five warmup updates;
- ordinary cross-entropy on licensed references from the 2,048 hard rows;
- 20% one-step scheduled replacement from the model's teacher-forced
  predictions, weighted 0.25;
- token-local unlikelihood at each real repeated-token recovery state,
  weighted 0.25;
- EOS-over-repeated-token margin ranking, weighted 0.50 with margin 1.0;
- frozen-safe-parent teacher-forced KL at 0.10 and parameter L2 at 1e-5;
- no architecture, decoder, MoE, quantization, or bundle change.

The contract binds the corpus, rollout evidence, V12 step-50 initialization,
safe parent, recipe, evaluator, and every implementation file. Its SHA-256 is
`63ae84a0661b3b84aba62232b2c0115fe2ee61b23a07578c6e18717e4f9b4618`.
Post-result hyperparameter changes are forbidden.

## Gates and calibrated semantics

A checkpoint may advance to semantic auditing only if, versus the frozen safe
parent on the fresh 768 rows, it achieves:

- at least +0.25 corpus and mean-sentence chrF++;
- at least +0.20 long-legal chrF++;
- no stratum below -0.50;
- paired 90% chrF++ interval lower bound at least -0.25;
- non-increasing repeated-token probability;
- at least 0.90 EOS-over-repeat preference accuracy;
- zero new repetition/generation-limit failures.

New exact, typed, or negation-detector disagreements do not automatically
become semantic approvals or rejections. A pre-semantic winner must send every
new detector case to identical blinded exact `claude-sonnet-5` and
`claude-opus-5` assessment. Any critical-error disagreement fails closed, and
zero new fail-closed semantic errors relative to the safe parent are required.
No human reviewer is required and no reasoning trace may be stored.

Even a pre-semantic pass does not authorize q4, COMET, protected evaluation,
bundling, app changes, release, or upload. Those stages remain sequentially
gated after dual semantic consensus.
