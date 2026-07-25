# V15 strategy review: constrained recovery without omission

Date: 2026-07-26

Status: **literature review and experiment recommendation; no V15 contract or
training is authorized by this document**

## Problem exposed by V14

V14 provides unusually specific evidence:

- actual free-running bad prefixes are useful training signals;
- repeated-token probability moves in the intended direction;
- unconditional EOS is not a sufficient recovery target;
- the best checkpoint improves aggregate and long-legal chrF++ and removes two
  loops, but loses 1.142 points on omission risk;
- extending the same recipe creates new loops and reduces aggregate quality.

The next method must therefore solve two coupled problems: recovery from a
model-generated prefix and preservation of all licensed-reference content. A
larger dense model or MoE adds capacity but does not directly impose either
property.

## What the literature says

### Exposure bias is real but recovery can itself become unsafe

Wang and Sennrich connect exposure bias to hallucination under domain shift and
show that sequence-level minimum-risk training can improve robustness:
[On Exposure Bias, Hallucination and Domain Shift in Neural Machine
Translation](https://aclanthology.org/2020.acl-main.326/).

He et al. identify the exact failure mode relevant to V14: ordinary recovery
training can make a sequence containing an error more probable than the clean
reference path. Their token-level contrastive method coordinates three
objectives:

1. prefer the licensed next token to a wrong token under the clean prefix;
2. recover the licensed next token after a perturbed prefix;
3. keep the licensed next token under the clean prefix more probable than the
   same token under the perturbed prefix.

That third constraint prevents error-conditioned recovery from dominating the
ground-truth sequence:
[Recovery Should Never Deviate from Ground Truth](https://aclanthology.org/2024.eamt-1.10/).

This is a better fit than V14's EOS-over-repeat ranking. EOS is not universally
correct for an incomplete legal sentence; a licensed continuation is.

### Omission needs its own contrastive signal

Yang et al. construct negative translations by deleting words and train the
model to score the complete ground truth above those omissions. They report
reduced omission errors across three translation directions:
[Reducing Word Omission Errors in Neural Machine
Translation](https://aclanthology.org/P19-1623/).

Berger et al. align a system hypothesis with its reference and use the
differences as token-level positive and negative markings. Their method needs
one translation pass over training data and leaves inference unchanged:
[Enhancing Supervised Learning with Contrastive Markings in Neural Machine
Translation Training](https://aclanthology.org/2023.eamt-1.8/).

These methods suggest that V15 should use both automatically deleted
licensed-reference spans and aligned free-running errors. Generated strings
remain negative/context evidence; only licensed references are positive
targets.

### Repetition penalties help, but broad penalties are risky for legal text

Unlikelihood training lowers the probability of repetitive sequences while
retaining ordinary likelihood quality:
[Neural Text Generation with Unlikelihood
Training](https://arxiv.org/abs/1908.04319).

The WMT 2024 SYSTRAN system modifies the training distribution to discourage
unwanted repetition with no extra inference work:
[SYSTRAN at the WMT24 Non-Repetitive Translation
Task](https://aclanthology.org/2024.wmt-1.108/).

Penalty decoding and local-temperature beam search also reduce
self-reinforcing repetition, but their published evaluations target
open-ended generation rather than high-fidelity legal translation:
[Penalty Decoding](https://aclanthology.org/2023.emnlp-main.78/) and
[Local Temperature Beam Search](https://aclanthology.org/2023.findings-acl.628/).

A general no-repeat trigram rule is too blunt for statutes, which legitimately
repeat citations and formulaic phrases. Mimi's narrower observed condition is
safer: reject only a token that would complete a third contiguous copy of a
3–16-token phrase. That policy should be measured as a separate deterministic
diagnostic, not used to excuse training failures.

### Metrics and speed remain later gates

Optimizing a learned metric directly is unsafe: COMET-based minimum-Bayes-risk
decoding can prefer outputs with number and named-entity discrepancies:
[Identifying Weaknesses in Machine Translation Metrics Through Minimum Bayes
Risk Decoding](https://aclanthology.org/2022.aacl-main.83/).
Mimi should continue combining BLEU, chrF++, COMET, deterministic structure
checks, and blinded semantic assessment.

For a candidate that first clears quality and safety, shallow decoders plus
vocabulary filtering have demonstrated nearly 2x multilingual NMT inference
speed without measured quality loss:
[Efficient Inference for Multilingual Neural Machine
Translation](https://aclanthology.org/2021.emnlp-main.674/).
Parallel fixed-point decoding reports up to 38% speedup without retraining:
[Accelerating Transformer Inference for Translation via Parallel
Decoding](https://aclanthology.org/2023.acl-long.689/).
Those are follow-up runtime experiments, not substitutes for a safe model.

## Recommended V15 arm

V15 should remain the same compact 6-encoder/6-decoder Marian architecture and
start from the rejected V12 step-50 research initialization, with the frozen
safe parent as retention teacher and evaluation baseline. It should use one
preregistered arm, not a hyperparameter sweep:

1. **Clean MLE:** licensed-reference cross-entropy.
2. **Aligned recovery:** align each free-running training hypothesis to its
   licensed reference. At high-confidence single-span errors, condition on the
   model-like prefix and rank the licensed continuation above the generated
   continuation.
3. **Constrained recovery:** rank the licensed continuation under its clean
   prefix above the same continuation under the perturbed prefix, following
   the three-objective contrastive formulation.
4. **Omission contrast:** create deterministic negatives by deleting complete
   parenthetical exceptions, enumerated items, named entities, amounts, and
   low-frequency content spans from licensed targets. Rank the untouched
   licensed sequence above each deletion.
5. **Repetition contrast:** construct reference-anchored repeated-prefix
   contexts only when the correct next licensed token is unambiguous. Rank
   that continuation above another repetition. Do not use EOS unless the
   licensed sequence actually ends there.
6. **Retention:** preserve frozen-parent KL and L2, with no new parameters.

No generated or LLM-authored string becomes a positive target. No reasoning
trace is requested or stored.

## Required evaluation design

The future contract should bind all code, data, weights, thresholds, and
randomness before the first gradient update. It should include:

- a newly sampled V15 source-disjoint legal suite not used by V10–V14;
- V12 and V14 suites as non-selective regression suites, because their known
  failures cannot be forgotten;
- explicit omission, critical-structure, negation, terminology, long-legal,
  and repetition strata;
- corpus BLEU, corpus and sentence chrF++, paired bootstrap intervals, and
  deterministic structure/generation checks;
- exact Sonnet 5 and Opus 5 review of every new detector disagreement, with
  fail-closed consensus and no reasoning traces;
- only after internal semantic passage: exact q4, all protected suites,
  pinned COMET, Apple-Silicon latency/RSS, and final bundle size.

The decoder guard should be evaluated on the safe parent and the candidate as
a two-mode diagnostic. The unguarded candidate must still pass every
translation-quality and semantic gate. Guarded output may advance only if it
introduces zero semantic regressions, does not truncate content, and adds
negligible p95 latency.

## Go/no-go intuition

This arm has a better causal match than MoE:

- V10–V14 already show that the current parameter count can gain quality;
- failure is concentrated in structural swaps, omissions, and
  self-reinforcing decoding;
- the proposed losses target those errors without adding model bytes or
  runtime passes;
- known V12/V14 suites remain explicit regression barriers.

If this arm cannot clear the internal gates, the next architecture decision
should be a new deep-encoder/shallow-decoder student distilled from a stronger
open teacher, not an MoE assembled from specialists that already fail the same
safety tests.
