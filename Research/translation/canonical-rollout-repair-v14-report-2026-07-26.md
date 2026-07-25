# V14 rollout-conditioned repair result

Date: 2026-07-26

Status: **rejected before semantic judging**

## Decision

Neither registered checkpoint passes the frozen pre-semantic gate. Stop V14.
Do not run Sonnet 5 or Opus 5 judging, q4 conversion, protected evaluation,
COMET, runtime comparison, bundling, release, or upload for either checkpoint.
The shipped translator and frozen safe parent remain unchanged.

The machine-readable result is
`canonical-rollout-repair-v14-presemantic-result-2026-07-26.json`, SHA-256
`79ba77a4e540cedd91f1ae30b42b6b29ff0c1b85bb20136b26fdc9042e33bfb9`.

## What ran

The preregistered arm ran exactly 50 updates from rejected V12 step 50. It
combined licensed-reference cross-entropy, one-step scheduled replacement,
unlikelihood on the repeated next token, EOS-over-repeat ranking, frozen safe
parent KL, and parameter L2. Only steps 25 and 50 were retained and evaluated.

Each full-precision checkpoint is 242,249,092 bytes. They remain ignored local
research artifacts rather than distributable model candidates:

| Checkpoint | Model SHA-256 | Training-manifest SHA-256 |
| --- | --- | --- |
| step 25 | `b76b327be5ddc2aa800c6212940622b22bdaad9cbce2b35f9c67d29ef599ed50` | `bca9e4c6057e02707a43d527a07ea605484c7f5440db631b6d124f99d90fe2b5` |
| step 50 | `bae926a8c268186143bd6521d45a7f903051780faa9bd4e9d27d4fb2b1e4cd8a` | `be0c8027c89df109188971e82bf0aa41e30ab13732620ad25dc7bf81247859c4` |

## Fresh-suite result

All values below come from the frozen 768-row legal test suite. BLEU and
chrF++ are corpus scores; “mean” is mean sentence chrF++. Detector counts are
diagnostic rather than semantic error counts.

| System | BLEU | chrF++ | Mean chrF++ | Long mean | Omission mean | Generation failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| safe parent | 21.295 | 45.821 | 44.770 | 45.215 | 45.631 | 9 |
| V14 step 25 | 22.111 | 46.253 | 45.240 | 45.611 | 44.489 | 7 |
| V14 step 50 | 22.033 | 46.121 | 45.144 | 45.560 | 44.489 | 9 |

Step 25 clears the aggregate corpus, mean-sentence, long-legal, bootstrap, and
no-new-generation gates. It still fails:

- worst-stratum delta: -1.142 chrF++ on omission risk, below the -0.50 floor;
- EOS-over-repeat preference: 0.0, below the required 0.90.

Step 50 fails those two gates and creates two new generation failures. Its
paired 90% bootstrap lower bound remains positive (+0.149), but this cannot
override a hard safety or generation gate.

| Recovery diagnostic | Initial V12 step 50 | V14 step 25 | V14 step 50 |
| --- | ---: | ---: | ---: |
| repeated-token probability | 0.595975 | 0.573478 | 0.564061 |
| EOS minus repeat-token margin | -7.9486 | -7.5722 | -7.4213 |
| EOS preference accuracy | 0.0% | 0.0% | 0.0% |

The objective moves both continuous recovery measures in the intended
direction, but not nearly enough to change any of the 74 recovery decisions.
More updates also trade away quality and introduce new loops. This rules out
simply extending the same low-rate recipe.

## Behavioral examples

V14 can make large real improvements. For example, it changes the looping
baseline:

> `(iv) measurements are performed (2) to to to ...`

to:

> `(4) Measurements are performed (2) to ) three times, and the mean is the
> measurement.`

It also restores a missing negation:

> `(6) The provisions of each of the preceding paragraphs do not apply to
> creditors ...`

But the regressions are disqualifying. One omission-risk item collapses the
exception detail to:

> `(i) not to use steel, polished powder, etc.)`

and step 50 creates a new nonterminating legal-citation loop:

> `Article 68, paragraph 73, paragraph 73, paragraph 73, ...`

These examples explain why aggregate overlap gains cannot promote the arm.

## Interpretation

V14 validates part of the exposure-bias hypothesis: training on actual bad
prefixes reduces the rejected-token probability and removes two existing
fresh-suite generation failures at step 25 without creating a new one.
However, forcing EOS as the only recovery action is too weak at this learning
rate and is not a generally correct continuation policy for incomplete legal
sentences. The arm also inherits the V12 initialization's quality gains rather
than improving on that initialization.

The next experiment must be a new preregistered arm. It should not extend V14.
The strongest next direction is a bounded decoding intervention with
backtracking or alternate-token selection when a contiguous repetition loop is
detected, evaluated first as a zero-bundle-byte runtime policy. A training arm
would need sequence-level recovery continuations anchored to licensed targets,
not EOS-only recovery. Either path must preserve omission-risk content and
retain the same held-out, semantic, protected, size, memory, and latency gates.
