# V12 constraint-aware safety-repair report

Date: 2026-07-26

Status: **rejected at the protected-independent internal gate**

## Outcome

V12 produced a statistically clear JA→EN quality improvement over the safe
parent, reduced every absolute safety-failure count, and improved the
deterministic negative-space objective. It nevertheless introduced new
case-level failures in every registered safety category, including one genuine
nonterminating repetition loop. The preregistered zero-new-failure rule
therefore rejects both checkpoints.

No checkpoint was quantized. No protected output, COMET score, or LLM judgment
was used for selection. No model was bundled, integrated, released, or
uploaded. The shipped translator and its fallback behavior are unchanged.

| Fresh 1,536-case metric | Safe parent | Step 50 | Delta | Step 100 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| corpus chrF++ | 47.6248 | 47.9983 | **+0.3735** | 47.9943 | **+0.3695** |
| corpus BLEU | 23.1026 | 23.7353 | **+0.6327** | 23.6867 | **+0.5840** |
| mean sentence chrF++ | 48.3823 | 48.7059 | **+0.3236** | 48.7099 | **+0.3275** |
| paired 90% interval | — | — | **+0.1951 to +0.4443** | — | **+0.1953 to +0.4645** |
| fresh general chrF++ | — | — | +0.0555 | — | +0.1639 |
| long legal chrF++ | — | — | +0.3371 | — | +0.2503 |
| worst stratum | — | — | -0.1408 | — | -0.1408 |
| negative-token probability | — | — | -0.000424 | — | -0.000565 |

Every registered non-safety gate passes at both checkpoints. Safety alone fixes
the rejection:

| Failure audit | Safe parent | Step 50 absolute | Step 50 resolved / new | Step 100 absolute | Step 100 resolved / new |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact critical structure | 571 | 561 | 23 / **13** | 561 | 25 / **15** |
| typed critical structure | 690 | 654 | 43 / **7** | 651 | 47 / **8** |
| negation policy | 193 | 191 | 7 / **5** | 191 | 7 / **5** |
| repetition / generation | 14 | 10 | 5 / **1** | 9 | 6 / **1** |

“Resolved” and “new” are relative to the safe parent and can coexist: the
candidate fixes more cases than it breaks, but the contract does not permit
trading one critical case for another.

## Data and contamination control

The V12 builder removes every GPT-5.6 teacher-repeat row from V10 and retains
7,104 unique licensed-human training rows. It creates a new 1,536-row
validation suite from previously unused test-split material:

| Validation stratum | Rows |
| --- | ---: |
| legal negation | 160 |
| legal critical tokens | 160 |
| legal repetition risk | 8 |
| legal terminology risk | 128 |
| legal omission risk | 64 |
| long legal | 240 |
| general legal | 328 |
| ALT | 128 |
| KFTT | 216 |
| Tatoeba | 104 |

The general quota had to be reduced from the initial 512-row intent to 448.
Most public test rows were already present in the ten protected suites: only
132 ALT, 219 KFTT, and 106 Tatoeba JA→EN rows survived the complete
source-disjointness and protected-overlap feasibility scan. The final
128/216/104 allocation stays within that clean inventory; the remaining 64
rows move to the fresh legal-general stratum. This happened before training
and before any model output was inspected.

The builder:

- excludes every normalized V10 train and validation source from V12
  validation;
- screens both source and target using Unicode NFKC, case-folding,
  whitespace removal, character 5-grams, and a strict greater-than-0.8
  Jaccard near-overlap rule;
- screens against canary, public-stress v1-v3, both legal-safety suites,
  M2M-100 feasibility, development documents and segments, and automated-claim
  sources;
- requires source license, source provenance, and attribution on every row;
- admits only CC-BY-2.0-FR, CC-BY-4.0, CC-BY-SA-3.0,
  PDL-1.0-compatible-CC-BY-4.0, or project-owned material;
- contains no free-form synthetic translation and no private reasoning trace.

The authenticated positive manifest hashes to
`6ef8e99fe52ce30613119e81c1b0d788655d73c46ca7852d42a302e039b9c3ac`.

The deterministic negative builder selects 6,000 training and 512 validation
positives and creates up to four target corruptions per positive:
number/unit substitution, negation reversal, head/tail omission, and
full/tail duplication. Rejected strings are negative evidence only and are
never translation targets. The result contains 23,924 training pairs and
2,019 validation pairs.

## Frozen model and objective

V12 starts from V11's full-precision 0.375 interpolation checkpoint and keeps
the safe JA→EN parent frozen as a retention teacher. It does not change the
Marian architecture or greedy decoder.

The one frozen arm uses:

- 100 updates, with selectable checkpoints only at steps 50 and 100;
- learning rate 5e-7, effective batch 32, and 192 source/target tokens;
- ordinary cross-entropy on the licensed-human chosen translation;
- severity-weighted `-log(1-p(rejected))` at the first divergent target token;
- a chosen-over-rejected hinge-ranking loss at that same token;
- teacher-forced token KL and parameter L2 to the frozen safe parent.

The contract was written before the first gradient update and hashes to
`a43ff4695b5838fb100bf1a6615601a4486d44df73f2b66b5a6da55a8275d99c`.
It forbids post-result hyperparameter changes and requires zero new exact,
typed, negation, or repetition/generation failures.

## Counterexample audit

The post-result diagnostic regenerates all 27 distinct cases that contribute
to a registered new failure at either checkpoint. It does not change the
selection decision.

The genuine blocker is an autoregressive repetition loop:

> Source: `２ 指定完成検査機関は、完成検査を行うときは、第五十八条の二十第一号に規定する…`
>
> Safe parent: “…Article 58, paragraph 201…Article (2)…”
>
> Step 50: “…Article (25) of the Article (25) of the Article (25)…” until the
> generation limit.

This case explains why target corruption under the *correct reference prefix*
is insufficient. At the last observed training batch, chosen-token preference
was already 100% and ranking loss was zero. Across negative validation,
chosen-token preference was 97.7%. The model knows the chosen token under
teacher forcing but can still enter an unseen bad state after conditioning on
its own generated prefix.

The audit also exposes measurement-taxonomy problems that must be fixed
without retroactively rescuing V12:

- The five “new negation” cases are detector disagreements rather than five
  polarity reversals. Three treat legal `No.` as English “no”; two flag
  semantically appropriate “not more than” renderings of Japanese upper-bound
  language. One of those translations still has a wrong amount, so it remains
  unsafe for a different reason.
- Several exact failures are equivalent legal-number surfaces such as
  `(ii)` becoming `(2)` or `Chapter VI` becoming `Chapter 6`.
- Other typed failures are real, including article/paragraph fusion,
  repeated paragraph numbers, and leaving the Japanese item marker `十一`
  untranslated.

The zero-new-failure contract remains correctly failed because at least the
repetition loop and several number/article corruptions are genuine. The
diagnostic instead says the next evaluator must separate semantic critical
errors from representational differences and detector artifacts.

## Literature update and changed intuition

The V12 design followed four useful results from prior work:

1. Training NMT to consume terminology constraints can preserve unconstrained
   decoding speed and be more robust than constrained beam search
   ([Dinu et al., ACL 2019](https://aclanthology.org/P19-1294/)).
2. Input augmentation and constrained decoding have been demonstrated for
   English-Japanese translation
   ([Chousa and Morishita, WAT 2021](https://aclanthology.org/2021.wat-1.3/)).
3. Token-ranking distillation can emphasize the teacher's most useful top-1
   information
   ([Zhang et al., ACL 2023](https://aclanthology.org/2023.acl-long.448/)).
4. Training-time repetition penalties can reduce repeated MT output without
   adding inference overhead
   ([Avila and Crego, WMT 2024](https://aclanthology.org/2024.wmt-1.108/)).

V12 narrows how those ideas should be applied in Mimi. Token-local
unlikelihood and ranking under a gold prefix improve aggregate behavior, but
they do not directly train recovery from self-generated prefixes. That is the
classic train/inference mismatch described by
[scheduled sampling](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html).
Sequence- or token-level
[unlikelihood training](https://iclr.cc/virtual_2020/poster_SJeYe0NtvH.html)
is still relevant, but the negative context must come from model rollouts or
another realistic decoding state rather than only a mechanically corrupted
reference.

The updated intuition is:

- do not add MoE capacity yet; V12 already passes every quality gate, so
  capacity is not the bottleneck;
- do not spend latency on beam search or dual-model routing;
- do not run another teacher-forced first-divergence arm;
- first calibrate the critical-error ontology on the 27 disagreement cases,
  preserving the genuine loop and article/number errors;
- then preregister one rollout-conditioned repair arm that trains on
  model-generated bad prefixes, explicit EOS/repetition recovery, and
  source-to-target legal-number constraints while retaining the frozen-parent
  KL/L2 boundary;
- evaluate it on another source-disjoint internal suite before q4 or protected
  testing.

## Reproducible evidence

- `canonical-safety-repair-v12-contract-2026-07-26.json`
- `canonical-safety-repair-v12-result-2026-07-26.json`
  (`0db46aee124cb3d9e61898630349e61cfa2a1fcecc5d04f591d3f5c5a2a9dcd1`)
- `canonical-safety-repair-v12-diagnostic-2026-07-26.json`
  (`4c3a651f5a14ade02da41c64a077401d9627c0645cc8a33a8b9c1e329ce83c10`)

The result keeps exact-q4 conversion, protected evaluation, bundle
replacement, app changes, promotion, and public Hugging Face upload
unauthorized.
