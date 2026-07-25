# V16 strategy review: active sequence risk from the safe parent

Date: 2026-07-26

Status: **literature review and diagnostic recommendation; no V16 training
contract exists yet**

## Why V15 failed

V15 improves aggregate quality, but its new objectives barely move:

- recovery preference starts at 72.314% and changes by at most +0.049
  percentage points;
- omission preference starts at 95.508% and changes by +0.049 percentage
  points;
- initial mean recovery and omission margins are already 5.28 and 10.13;
- fresh omission risk improves, but the independent V14 omission slice loses
  1.344–1.505 chrF++ points;
- both checkpoints create the same new generation failures on fresh V15 and
  V12.

The problem is not simply a loss weight that is too small. Most V15
token-level examples are already easy and outside the hinge, while the
research initialization already carries V12's known failure. Increasing a
mostly inactive loss would be a post-hoc recipe change and would not repair the
distribution mismatch.

## What the literature adds

Yang et al. formulate omission repair at the **sequence** level: the model
should assign higher probability to the complete ground-truth translation
than to a version constructed by deleting words. That is closer to Mimi's
actual failure than V15's single next-token comparison:
[Reducing Word Omission Errors in Neural Machine
Translation](https://aclanthology.org/P19-1623/).

Minimum Risk Training optimizes expected sentence-level translation loss over
model candidates rather than only teacher-forced next-token likelihood:
[Minimum Risk Training for Neural Machine
Translation](https://aclanthology.org/P16-1159/). Wang and Sennrich connect
this family to reduced exposure-driven hallucination under domain shift:
[On Exposure Bias, Hallucination and Domain Shift in Neural Machine
Translation](https://aclanthology.org/2020.acl-main.326/). Document-level MRT
has also shown value in a narrow biomedical domain where imperfect pairs and
source neglect are prominent:
[Addressing Exposure Bias With Document Minimum Risk
Training](https://aclanthology.org/2020.wmt-1.94/).

This does not authorize direct BLEU/COMET optimization. Mimi's prior evidence
shows that aggregate metrics can improve while legal content is omitted.
Instead, the sequence score should compare a licensed reference with a
specific structurally wrong negative, and evaluation must remain external.

Hard-negative theory gives a second relevant principle: negatives sampled near
the model distribution can provide a less biased contrastive gradient than
arbitrary easy negatives:
[Understanding Hard Negatives in Noise Contrastive
Estimation](https://aclanthology.org/2021.naacl-main.86/). This paper studies
contrastive estimation rather than NMT, so its conclusion is a design clue,
not evidence that the proposed V16 method will work.

The remaining uncertainty is objective interference. PCGrad projects away
components of task gradients whose cosine similarity is negative:
[Gradient Surgery for Multi-Task
Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).
Conflict-Averse Gradient Descent instead seeks good average descent while
regularizing the worst local task improvement:
[Conflict-Averse Gradient Descent](https://openreview.net/forum?id=_61Qh8tULj_).
Neither paper establishes a translation-specific benefit here. Mimi should
measure gradient norms and cosines first, then preregister one optimizer rather
than selecting between them after seeing translation scores.

## Recommended diagnostic

Before another gradient update, build one authenticated diagnostic from
training-only rows:

1. Start from the distributable safe parent, not rejected V12 or V15 weights.
2. Score the length-normalized teacher-forced log probability of each complete
   licensed reference.
3. Construct full-sequence deletion negatives from the same licensed
   reference, including parentheticals, enumerated clauses, quantities,
   identifiers, named entities, and low-frequency content spans.
4. Add actual safe-parent free-running outputs only when a deterministic
   structure or generation detector identifies an omission, truncation,
   repetition, or critical-token mismatch.
5. Keep only active or near-active comparisons where the negative is preferred
   or the reference margin is below a frozen threshold. Generated output
   remains negative evidence only.
6. On a fixed stratified batch, measure MLE, omission-ranking,
   rollout-ranking, and safe-parent-retention gradient norms and pairwise
   cosine similarities.
7. Report how many examples remain, their licenses/provenance, margin
   distribution, detector composition, and gradient-conflict matrix.

No validation or protected source may become training data. V12, V14, and V15
validation suites remain regression-only.

## Candidate V16 arm, conditional on the diagnostic

If the diagnostic finds enough active licensed examples and usable gradients,
freeze one new contract:

- the same compact 6-encoder/6-decoder Marian architecture;
- initialization and retention teacher both equal to the safe parent;
- clean licensed-reference MLE;
- length-normalized full-reference-over-full-negative sequence ranking on
  active omission and rollout negatives;
- one preregistered gradient rule chosen from ordinary weighted descent or
  PCGrad/CAGrad based only on the pre-training conflict diagnostic;
- no scheduled sampling, free-form synthetic positive, reasoning trace, new
  parameters, MoE, or inference-time pass;
- checkpoints at fixed steps only, with no sweep.

The new selection surface should include a fresh source-disjoint V16 suite and
all V12, V14, and V15 suites as regression barriers. The fresh corpus has no
remaining untouched omission-risk reservoir, so historical omission suites
must remain explicit hard gates rather than being replaced by a tiny new
slice.

## Go/no-go criteria

Do not preregister training unless:

- at least several hundred recovery and omission comparisons are active or
  near-active under the safe parent;
- positive targets are exclusively licensed human references;
- no V12/V14/V15/protected row is used for training;
- the sequence score is length-normalized and tested against trivial
  short-negative preference;
- gradient norms are finite and no objective is effectively zero;
- the chosen gradient rule is fixed from the diagnostic, not from candidate
  translation results.

V16 remains a same-size model experiment. MoE, a larger model, and runtime
acceleration remain deferred until the existing compact model has a candidate
that clears semantic safety. V15 failed because its intended gradients were
mostly inactive, not because the 150 MB target was exhausted.
