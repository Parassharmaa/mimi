# V15 constrained-recovery preregistration

Date: 2026-07-26
Status: **one arm frozen; training authorized; no promotion stage authorized**

V15 tests whether a better-targeted objective can keep V12's legal-domain
quality gain while repairing V14's omission and rollout-recovery failures. It
does not change architecture or runtime: the student remains the compact
6-encoder/6-decoder Marian model, starts from rejected V12 step 50 as a
research initialization, and uses the distributable safe parent as retention
teacher and evaluation baseline.

The immutable contract is
`canonical-constrained-recovery-v15-contract-2026-07-26.json`, SHA-256
`f342d8bf027f88143159c1b0ae2d5da3fb5ccad3cabb9aeb73e6d3175699549a`.
It binds the datasets, contrast examples, checkpoints, implementation,
hyperparameters, randomness, evaluation suites, and gates before the first
gradient update.

## Frozen data

All 7,104 positive training targets are authenticated licensed-human
references. The fresh validation set contains 768 unique sources, excludes
every V10, V12, and V14 train or validation source, and screens both source and
target against ten protected suites.

| Fresh V15 validation stratum | Rows |
| --- | ---: |
| Critical | 128 |
| General | 266 |
| Long | 192 |
| Negation | 96 |
| Omission risk | 6 |
| Terminology risk | 80 |

The separate training-only contrast set contains:

- 1,024 first token-level divergences between a V12 free-running rollout and
  its licensed reference;
- 1,024 licensed-reference counterfactual prefixes that would otherwise begin
  a third contiguous phrase copy;
- 2,048 deterministic licensed-reference span-deletion contrasts, with 949
  long and 431 omission-risk examples.

Generated strings provide context or rejected-token evidence only. They are
never positive targets. No LLM-authored translation or private reasoning trace
is used.

## Frozen objective

One 50-update arm uses batch size 4, gradient accumulation 8, learning rate
`3e-7`, checkpoints at steps 25 and 50, and these fixed components:

1. licensed-reference token cross-entropy;
2. correct licensed continuation over the rejected token under a perturbed
   prefix, weight `0.25`;
3. the same licensed continuation under the clean prefix over its probability
   under the perturbed prefix, weight `0.10`;
4. first omitted licensed token over the post-deletion skip token, weight
   `0.50`;
5. safe-parent teacher-forced KL, weight `0.10`;
6. parameter L2 to the safe parent, weight `1e-5`.

There is no scheduled sampling, unconditional EOS recovery, capacity change,
MoE, generated positive target, or post-result hyperparameter edit.

## Frozen selection gates

The safe parent is generated once on three bound suites: fresh V15 (768), V12
regression (1,536), and V14 regression (768). Each candidate must pass every
gate:

- fresh corpus and mean-sentence chrF++ deltas at least `+0.25`;
- fresh long delta at least `+0.20`, worst stratum at least `-0.50`, and
  paired 90% bootstrap lower bound at least `-0.25`;
- V12 and V14 mean deltas at least `+0.20` and worst-stratum deltas at least
  `-0.50`;
- V14 omission-risk delta at least `-0.50`;
- recovery and omission preference accuracy each improve at least `+0.05`
  from the frozen step-zero diagnostic;
- clean-over-recovery accuracy changes by at least `-0.02`, rejected-token
  probability does not increase, and no suite gains a new
  repetition/generation-limit failure.

BLEU is reported alongside chrF++, but chrF++ and the deterministic checks are
the pre-semantic selection measures. New exact, typed, or negation detector
cases form a review queue rather than an automatic semantic verdict.

Only a checkpoint passing all pre-semantic gates may receive exact,
independent Claude Sonnet 5 and Claude Opus 5 review. Exact q4 conversion,
COMET, protected evaluation, MLX latency/RSS, bundle work, app changes,
release, and public upload remain forbidden until the preceding gate
explicitly authorizes them. The shipped translator remains unchanged.
