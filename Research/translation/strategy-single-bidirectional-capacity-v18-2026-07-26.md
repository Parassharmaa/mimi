# V18 architecture scope: one shared 90M-class EN↔JA Marian student

Date: 2026-07-26

Status: **architecture-only scope; no training, promotion, app, or release authorization**

## Decision

The first single-physical-model capacity control should be a fully shared
Marian encoder-decoder with the authenticated 32,001-token vocabulary,
512-wide hidden states, six encoder and six decoder layers, and both FFNs
widened from 2,048 to 4,608. It has exactly 92,043,009 dense parameters.
The pinned MLX q4/group-64 storage model projects a 52.06 MB weight file and
about a 54.46 MB one-model pack.

This supersedes a deep-encoder/shallow-decoder model as the *first* shared
capacity control. It does not reject the deep architecture. It follows from
Mimi's measured evidence:

- direct 6→5, 6→4, and 6→2 decoder pruning all lost too much quality;
- a purpose-trained 6e/4d student recovered to 27.18 development chrF++ but
  still lost 4-bit quality;
- appended post-norm encoder blocks were not function-preserving;
- widening Marian FFNs is mathematically output-preserving when copied active
  features sit behind zero-initialized new `fc2` columns; and
- the earlier shared 60.6M bidirectional pilot learned the missing direction
  but had insufficient training/capacity and severe negative transfer.

The wide 6e/6d control therefore isolates whether approximately 31.5M new FFN
parameters can hold two directional functions without changing the topology
that already works in MLX. It is not a model-quality claim.

## Reproducible architecture comparison

Run:

```sh
python3 scripts/translation/analyze_bidirectional_student_architecture.py --pretty
python3 scripts/translation/test_bidirectional_student_architecture.py
python3 scripts/translation/test_widen_marian_checkpoint_ffn.py
```

| Fully shared shape | Dense parameters | Projected q4 model | Projected one-model pack | Approximate MAC ratio short / sentence / long |
| --- | ---: | ---: | ---: | ---: |
| Current 6e/6d, FFN 2,048 | 60.56M | 34.30 MB | 36.70 MB | 1.00 / 1.00 / 1.00 |
| **Wide 6e/6d, FFN 4,608** | **92.04M** | **52.06 MB** | **54.46 MB** | **1.71 / 1.70 / 1.68** |
| Deep 18e/4d, FFN 2,048 | 89.98M | 50.96 MB | 53.36 MB | 1.67 / 2.00 / 2.18 |
| Deep 24e/4d, FFN 2,048 | 108.89M | 61.68 MB | 64.08 MB | 2.09 / 2.57 / 2.82 |
| Deep 30e/4d, FFN 2,048 | 127.80M | 72.39 MB | 74.79 MB | 2.52 / 3.15 / 3.47 |

The q4 estimate reproduces the authenticated incumbent child within 5,000
bytes. It accounts for packed four-bit matrices, float16 group scales/biases,
float16 vector parameters, and calibrated safetensors metadata. It remains a
projection until a real checkpoint is converted.

The compute column counts leading cached-greedy matrix and attention
multiply-accumulates at source/target token shapes 16/16, 64/32, and 192/64.
It is not a latency claim. The wide option has the most stable compute ratio
across long inputs and retains the current six-layer cached decoder.

## Conditional training design

No training contract is frozen here. If a later immutable contract authorizes
this architecture, the single arm should:

1. start from the stronger directional parent selected without protected data;
2. expand every encoder and decoder FFN to 4,608 with a zero-output-column,
   mathematically preserving transform, then verify floating-logit tolerance
   and exact greedy-token parity;
3. use balanced licensed EN→JA and JA→EN sources with existing `<2ja>` and
   `<2en>` source prefixes;
4. distill final sequences and token distributions from the two authenticated
   directional teachers, routing each row only to its direction's teacher;
5. align student and teacher encoder states directly at width 512, avoiding a
   learned projection whose capacity could hide poor alignment;
6. mix licensed human-reference MLE and explicit retention to both frozen
   teachers;
7. select on a fresh direction-balanced development surface containing short
   speech, news, Wikipedia, legal text, and long segments;
8. reject any checkpoint with a direction, domain, critical-meaning, or
   generation regression before q4 conversion; and
9. require exact q4, COMET, chrF++, BLEU, dual-family semantic judgment,
   latency, RSS, bundle, Swift parity, and distribution evidence before any
   app or Hugging Face action.

Final translations and compact score/error labels are sufficient. No teacher
or judge reasoning trace is requested or retained.

`widen_marian_checkpoint_ffn.py` implements the authenticated transform for
both stacks. Its contract test proves the original and widened FFN outputs
match within a pinned `1e-6` floating tolerance for every fixture layer while
all non-FFN weights remain unchanged. A real checkpoint must additionally pass
exact greedy-token parity because wider matrix kernels can change floating
summation order. The emitted manifest still forbids training and promotion; a
later immutable experiment contract must bind the real source model and
widened artifact before any update.

The authenticated EN→JA parent has now passed that real-checkpoint smoke. The
6e/6d FFN-4,608 transform produced a reproducible 368,201,220-byte temporary
full-precision checkpoint. On Apple M3 Pro MPS, all six EN→JA canary cases
matched the parent for 20 generated steps: token sequences and finite-logit
masks were identical, and the maximum absolute finite-logit delta was `0.0`.
The temporary checkpoint was deleted after hashing because it is reproducible
and does not authorize training or distribution. The tracked evidence is
`shared-bidirectional-v18-wide-parent-parity-smoke-2026-07-26.json`.

The release-eligible data pass then removes every source row explicitly marked
training-only or promotion-ineligible and re-screens both sides against ten
historical/protected suites. It retains 12,066 balanced licensed-human training
rows and 2,370 unrepeated validation rows. A real one-update MPS smoke confirms
that the widened student can jointly optimize human-reference cross-entropy,
direction-specific token KL, and projection-free encoder-state alignment
against the two stronger directional teachers. Both temporary 368 MB
checkpoints were deleted after hashing. The compact evidence is in
`shared-bidirectional-v18-dataset-result-2026-07-26.json` and
`shared-bidirectional-v18-training-path-smoke-2026-07-26.json`.

The immutable phase-one contract now authorizes exactly one 1,000-update dense
capacity control, with checkpoints at 250/500/750/1,000. It does not admit V17
generated candidates, teacher reasoning, or source-only teacher sequences as
positive targets. Full-precision, q4, safety, long-document, learned-metric,
latency, RSS, Swift-parity, held-out, and licensing gates remain independent;
phase-one training alone cannot authorize app work, promotion, or upload. See
`shared-bidirectional-v18-phase1-contract-2026-07-26.json`.

## Why not MoE first

This control spends the same total parameter budget densely and activates all
capacity for both directions. A routed or direction-specific expert model is
only informative after this control establishes whether shared capacity is the
limitation. The existing routed systems already show complementary public
development outputs, but not a held-out semantic or release win.

If the dense shared student passes quality but misses latency, a later
capacity-matched low-rank or direction-routed expert ablation may reduce active
compute. If it fails quality through directional interference, that is evidence
for selective experts. Neither conclusion may be assumed in advance.

## Runtime boundary

Python MLX already supports variable FFN widths, but the native Swift Marian
loader currently fixes six layers and FFN width 2,048. Do not widen the Swift
loader yet. First prove a full-precision and exact-q4 model win in the
research lane; only then implement manifest-driven widths and exact
Swift/Python token parity.

The current Mimi translator, bundle, routing, and fallback remain unchanged.
