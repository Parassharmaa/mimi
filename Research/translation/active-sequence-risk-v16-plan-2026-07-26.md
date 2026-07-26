# V16 active sequence-risk preregistration

Date: 2026-07-26

Status: **one bounded arm frozen before training**

## Motivation

The safe-parent diagnostic finds 677 active full-sequence risks, including 169
cases where a deterministic omission or repetition negative is preferred to
the complete licensed reference. MLE gradients align with both safety
objectives, while omission and repetition gradients conflict in all four
disjoint diagnostic batches.

V16 therefore starts from the distributable safe parent and applies symmetric
PCGrad only between the two safety gradients. It does not project MLE, inherit
rejected V12/V15 weights, add parameters, add an inference pass, or use a
generated positive target.

## Frozen evidence

- diagnostic result SHA-256:
  `0272e49d4a6ebd9d87df8b51099beb354510c26f277b4b29b29d9ab98d98978d`;
- dataset manifest SHA-256:
  `21a3c7e23c190b28bb6d2ded323f3bd1bbde93e8fda44216d44c5be466b25902`;
- active-pair SHA-256:
  `ce93785835a7a0c37b6b3661479aef4cc7059c0b1e60f2245c7dfe89a5598e50`;
- fresh-validation SHA-256:
  `511b9d4c52d28307f9ba0be33007e89d62224e8efc61c9626abaff20b3f3639b`;
- contract SHA-256:
  `4a3c8ee0fa08a97bf9707501cb6c1b4d36d11b70bec58dfe2c16a64b720ff6c9`.

The training corpus contains the same 7,104 authenticated licensed-human
references as V15. The active set contains 228 omission and 449 repetition
comparisons. Negative text is deterministic corruption and is never a
positive target. The fresh 768-row legal suite excludes every V10, V12, V14,
and V15 train/validation source and is screened on both sides against the ten
protected suites.

The legal reservoir has no remaining fresh omission or repetition rows and
only seven terminology rows. The new suite therefore contains 128 critical,
345 general, 192 long, 96 negation, and seven terminology cases. The V12,
V14, and V15 suites remain hard regression gates, including their historical
omission/repetition strata.

## Frozen training arm

- architecture: unchanged 6-encoder/6-decoder Marian;
- initialization and retention teacher: distributable safe parent;
- 50 updates, checkpoints only at steps 25 and 50;
- batch size four for MLE and four per safety role;
- AdamW, learning rate `3e-7`, five warmup steps, weight decay `0.01`;
- full-reference-over-full-negative hinge margin `0.25`;
- omission and repetition weights `0.35` each;
- safe-parent token KL `0.10`, parameter L2 `1e-5`;
- global gradient clipping at `1.0`.

For original safety gradients `g_o` and `g_r`, projection occurs only when
their dot product is negative:

```text
g_o' = g_o - dot(g_o, g_r) / ||g_r||² * g_r
g_r' = g_r - dot(g_r, g_o) / ||g_o||² * g_o
g     = g_MLE + 0.35 * g_o' + 0.35 * g_r'
```

Both projections use the original gradients. MLE, safe-parent KL, and L2 are
never projected.

## Commands

Training:

```sh
PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with torch --with transformers==4.57.6 --with sentencepiece \
  --with sacremoses --with sacrebleu==2.6.0 --with numpy \
  scripts/translation/train_active_sequence_risk_v16.py \
  Research/translation/active-sequence-risk-v16-contract-2026-07-26.json \
  Research/translation/models/elanmt-active-sequence-risk-v16-ja-en-checkpoints
```

Pre-semantic evaluation:

```sh
PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with torch --with transformers==4.57.6 --with sentencepiece \
  --with sacremoses --with sacrebleu==2.6.0 --with numpy \
  scripts/translation/evaluate_active_sequence_risk_v16.py \
  Research/translation/active-sequence-risk-v16-contract-2026-07-26.json \
  Research/translation/active-sequence-risk-v16-presemantic-result-2026-07-26.json \
  --candidate 25 \
    Research/translation/models/elanmt-active-sequence-risk-v16-ja-en-checkpoints/step-0000025 \
  --candidate 50 \
    Research/translation/models/elanmt-active-sequence-risk-v16-ja-en-checkpoints/step-0000050
```

## Stop boundary

A checkpoint must clear the fresh V16 suite, all V12/V14/V15 regression
barriers, both active-risk roles, and zero-new-generation-failure gates.
Detector disagreements then require exact independent Sonnet 5 and Opus 5
review.

No semantic judge, q4 conversion, COMET, protected evaluation, runtime
comparison, bundle replacement, app change, release, or public upload is
authorized by this preregistration. If neither checkpoint clears every
pre-semantic gate, V16 stops.
