# Accuracy-first EN-JA distillation strategy

Date: 2026-07-21

Status: experiment design, not a promotion or distribution decision

## Decision

The next training lane should be **lexically constrained, final-output-only
sequence distillation into the existing compact Marian student**, followed by a
small frozen feasibility screen of the MIT-licensed M2M-100 418M checkpoint as
a single-model bidirectional alternative. Do not train on teacher reasoning
traces. Do not replace human references wholesale with teacher outputs. Do not
port M2M-100 to Swift/MLX unless it first clears a cheap quality screen.

This ordering targets Mimi's measured problem: the current 140,875,791-byte
routed pack is already fast and compact, but it propagates important-word,
negation, number, and entity errors. A larger architecture is useful only if it
improves those failures on untouched data.

## What the literature changes

1. **Sequence targets are the high-value compression signal.** Kim and Rush's
   sequence-level knowledge-distillation work reports that a compact student
   can learn the teacher's mode and decode greedily, with a large speedup over
   the teacher and much smaller quality loss than an undistilled student. That
   supports final translation targets, not natural-language rationales.
   Source: [Kim and Rush, EMNLP 2016](https://aclanthology.org/D16-1139/).

2. **Blind teacher imitation propagates exactly the errors Mimi is rejecting.**
   Mino et al. identify degradation from teacher errors on words important to
   sentence meaning and improve English-Japanese translation by giving the
   teacher privileged lexical constraints selected for importance and
   fallibility. Their gains hold without ensemble or beam-search decoding.
   Source: [Mino et al., 2022](https://doi.org/10.5715/jnlp.29.1082).

3. **Top-1 divergences deserve more weight than the teacher's whole softmax.**
   Zhang et al. find that much of useful distillation knowledge is concentrated
   in teacher top-1 predictions and propose a ranking loss plus iterative
   distillation on examples without gold targets. This suggests weighting the
   small set of teacher/gold divergence positions rather than paying the cost
   of storing full teacher distributions.
   Source: [Zhang et al., ACL 2023](https://aclanthology.org/2023.acl-long.448/).

4. **A single bidirectional baseline is technically plausible but unproven for
   Mimi.** M2M-100 is one encoder-decoder supporting direct translation between
   100 languages, including English and Japanese. The official 418M model
   metadata declares MIT, at revision
   `55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`; its published PyTorch checkpoint
   is 1,935,796,948 bytes. Ideal four-bit parameter storage is about 209 MB
   before quantization scales, tokenizer files, and any unquantized tensors, so
   a sub-500-MB MLX package is plausible but must be measured.
   Sources: [official model card](https://huggingface.co/facebook/m2m100_418M),
   [official model metadata](https://huggingface.co/api/models/facebook/m2m100_418M),
   and [M2M-100 paper](https://arxiv.org/abs/2010.11125).

## Teacher-data contract

The teacher receives only a source sentence, direction, domain, and an optional
machine-readable constraint list. It returns only one or more final translation
candidates. Prompts must explicitly prohibit explanations, analysis, hidden
reasoning, confidence prose, or commentary. Candidate JSON contains the final
translation plus request/model identifiers; it contains no chain of thought.

For licensed human parallel rows, constraints may be derived from the gold
target as privileged training information:

- numbers, units, dates, times, currencies, percentages, identifiers, URLs,
  placeholders, code spans, and named entities;
- negation/polarity markers and legally operative terms;
- low-frequency content words on which the incumbent or teacher has previously
  failed.

For source-only rows, constraints are limited to deterministically extractable
source structures and a reviewed bilingual terminology table. No benchmark
reference, Apple output, or held-out target may enter teacher generation.

Every candidate remains rejected unless all of these independent automated
checks pass:

1. exact structural preservation for numbers, entities, placeholders, code,
   and typed critical tokens;
2. lexical-constraint coverage with direction-aware inflection rules;
3. language/script and non-copy checks;
4. agreement with the licensed human target where one exists, using pinned
   chrF++ and a pinned learned metric;
5. round-trip or bilingual-judge consensus from a model that did not generate
   the candidate;
6. duplicate, source-overlap, and semantic-contamination screening against all
   frozen evaluation sets;
7. provenance and license records that keep failed candidates auditable but out
   of training.

There is no human-review requirement, but there is also no automatic right to
admit a teacher output merely because the teacher is larger.

## Student objective

Use the current Marian tokenizer and architecture first so the new training
signal can be evaluated without adding a runtime confound.

- Keep the original licensed human target as the anchor.
- Add an accepted teacher target as a second sequence target rather than
  replacing the human target.
- Up-weight teacher top-1 divergence positions only when the teacher candidate
  passes the reference and structure gates.
- Add an auxiliary constraint-coverage loss over important target tokens.
- Add contrastive preference pairs for incumbent critical failures: accepted
  constrained candidate over structure-breaking candidate.
- Oversample critical-token, negation, conversational, and legal slices without
  changing the frozen evaluation distribution.
- Train direction-specific students first. Merge into one bidirectional student
  only after each direction independently clears the corresponding directional
  baseline.

The first ablation must compare four arms with the same seed and update budget:

1. human references only;
2. human plus ordinary accepted sequence targets;
3. human plus constrained sequence targets;
4. arm 3 plus divergence-weighted top-1 and contrastive critical-error loss.

## M2M-100 feasibility gate

Do not begin with a full MLX port. First run the pinned upstream checkpoint on a
frozen, contamination-clean 40+40 screen sampled before inference, stratified
across conversation, general, formal/news, legal, and critical-structure cases.
Record hypotheses, token IDs, chrF++, sacreBLEU, the pinned learned metric,
critical failures, PyTorch latency, and peak memory.

Proceed to 400+400 and an MLX port only if both directions satisfy all of:

- no worse corpus chrF++ than the compact Marian generalist by more than 0.25;
- fewer strict critical failures in each direction;
- no domain slice regression larger than 1.0 chrF++;
- a credible four-bit package estimate below 500,000,000 bytes;
- confirmed MIT model and tokenizer licensing at the pinned revision.

If the small screen fails, retain M2M-100 as a rejected baseline and spend the
compute on constrained Marian distillation instead.

### Observed feasibility result

The frozen 40+40 run rejects M2M-100 before quantization or MLX work. Against
the exact portable Marian pack, corpus chrF++ changes 28.4909→18.6602 EN→JA
and 51.9351→40.8343 JA→EN. Seven of eight domain slices miss the allowed
one-point regression bound. Strict critical-token failures are 7/40 and 9/40,
not fewer than Marian's 7 and 8. Authenticated p95 latency changes 63.8→744.1
ms and 80.6→1049.0 ms. The upstream snapshot is 1,941,931,012 bytes and peaks
at 3,774,595,072 resident bytes. The report and score hashes are
`4d98c098383573b932a0dfbbe86b8b8597962c76dae92f21d1f9f085a0095e38`
and
`1315bd9279297435034d0ffcea0d1b8ae7904b8571e0d79ed69d937a456a8b85`.
The next compute therefore goes to constrained Marian distillation.

## Promotion evaluation

Training and development selection may use training/validation material only.
The final claim uses the already frozen contamination-screened benchmark of at
least 400 EN-JA and 400 JA-EN cases. Promotion requires, per direction:

- positive paired 95% lower confidence bounds against the incumbent;
- strong absolute chrF++, sacreBLEU, and pinned learned-metric scores;
- zero critical number, entity, negation, omission, placeholder, or code-switch
  failures;
- native Swift/MLX route and token parity;
- real-time latency, peak RSS, energy where practical, and exact bundle bytes;
- a complete distributable license and attribution payload.

Apple Translation is not a runtime dependency or fallback. The existing Mimi
default remains unchanged until the new local candidate passes every gate.

## Immediate execution order

1. Freeze the M2M-100 40+40 source-only screen and its preregistered thresholds.
2. Run the upstream MIT checkpoint once; reject it early if it misses the gate.
3. When GPT-5.6 credentials are available, generate only final constrained
   candidates for training rows and the separately sealed 400+400 reference
   lane; never request or store reasoning traces.
4. Build the four-arm constrained-distillation ablation with immutable accepted
   and rejected candidate manifests.
5. Evaluate on validation, select once, then run the frozen final benchmark.
6. Convert and integrate only a candidate that passes quality, safety,
   performance, size, licensing, and policy gates.
