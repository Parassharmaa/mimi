# V16 active sequence-risk diagnostic

Date: 2026-07-26

Status: **diagnostic complete; training remains unauthorized**

## Question

V15 showed that token-local recovery constraints were usually inactive and did
not generalize to the held-out omission slice. This diagnostic asks whether the
distributable safe parent has enough difficult full-sequence omission and
repetition examples to support a new arm, and whether their gradients require
conflict-aware optimization.

No optimizer step was executed and no model checkpoint was written.

## Frozen inputs and outputs

The diagnostic uses 1,024 training-only rows from the authenticated V15
training corpus. Validation and protected rows are excluded. Every positive is
an openly licensed human reference; deterministic corruptions are used only as
negative examples.

The tracked result is
`active-sequence-risk-v16-diagnostic-result-2026-07-26.json`, SHA-256
`0272e49d4a6ebd9d87df8b51099beb354510c26f277b4b29b29d9ab98d98978d`.
It binds:

- implementation SHA-256
  `9dc0ef32b6f80ea58a82542c4f288cb33c5f82be19c464812cd9930a9602ba67`;
- 2,028 scored comparisons, SHA-256
  `2de01459ab1b978f870b5bd69c665586a4de7531aa5a375af61b3f318f40da97`;
- 677 active comparisons, SHA-256
  `ce93785835a7a0c37b6b3661479aef4cc7059c0b1e60f2245c7dfe89a5598e50`.

The large scored and active JSONL files remain in the ignored research work
area. Their hashes and row counts are sealed in the tracked result.

## Active-risk evidence

With a frozen length-normalized reference-minus-negative margin threshold of
`0.25`, 677 comparisons are active:

| Risk | Candidates | Active | Active rate | Minimum margin | Median margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Omission | 1,004 | 228 | 22.71% | -1.083 | 0.570 |
| Repetition | 1,024 | 449 | 43.85% | -0.539 | 0.279 |

The active pool contains 437 long-legal, 181 omission-risk, and 59
repetition-risk comparisons. In 169 active comparisons the margin is below
zero, meaning the safe parent assigns a higher length-normalized probability
to the structurally wrong negative than to the complete licensed reference.
This is sufficient hard-example density to justify preregistering one bounded
training arm.

## Gradient-conflict evidence

Exact full-model gradients were measured on four disjoint, stratum-balanced
16-row batches for each role:

| Pair | Mean cosine | Range | Negative fraction |
| --- | ---: | ---: | ---: |
| MLE vs omission | +0.142 | +0.113 to +0.184 | 0/4 |
| MLE vs repetition | +0.159 | +0.063 to +0.238 | 0/4 |
| Omission vs repetition | -0.119 | -0.161 to -0.095 | 4/4 |

All objective gradient norms are finite and non-zero. Ordinary
licensed-reference MLE is aligned with both safety objectives, so projecting
it would discard useful signal. Omission and repetition gradients conflict in
every replicate.

## Decision

The evidence selects deterministic symmetric PCGrad **only between the
omission and repetition sequence-ranking gradients**. A future frozen V16
contract may combine the unprojected MLE gradient with both projected safety
gradients and a safe-parent retention term. Projection must use the two
original safety gradients, fixed weights, and a deterministic update order.

This report does not itself authorize training. The next atomic milestone is a
contract that binds the active dataset, fresh source-disjoint validation,
historical V12/V14/V15 regression gates, update count, weights, PCGrad formula,
and stopping conditions before any gradient update.

No semantic judge, q4 conversion, COMET run, protected evaluation, app change,
bundle replacement, release, or public upload is authorized. The shipped
translator remains unchanged.
