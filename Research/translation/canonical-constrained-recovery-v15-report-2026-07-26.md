# V15 constrained-recovery result

Date: 2026-07-26

## Decision

**Reject both checkpoints at the frozen pre-semantic gate. Stop V15.**

The complete result is
`canonical-constrained-recovery-v15-presemantic-result-2026-07-26.json`,
SHA-256
`0f324061b3a8b4da8ac86844b433b7163559bb9ac8e79bd8f4ab792a70586d8f`.
Neither checkpoint is selected for semantic review. Sonnet 5, Opus 5, exact
q4, COMET, protected evaluation, runtime/bundle tests, app changes, release,
and upload were not run and remain unauthorized.

The app and its shipped translator are unchanged.

## Reproducibility

The immutable contract SHA-256 is
`f342d8bf027f88143159c1b0ae2d5da3fb5ccad3cabb9aeb73e6d3175699549a`.
Training ran on Apple M3 Pro MPS with PyTorch 2.13.0 and Transformers 4.57.6.
It executed the registered 50 updates and saved only steps 25 and 50.

| Checkpoint | Full-precision model SHA-256 | Training manifest SHA-256 |
| --- | --- | --- |
| Step 25 | `50cdcdda33cdc5e1d4b365605e996afa3ed62eb4ad4e25747d36f47270ac197f` | `b549825e8b7cf064c37bf852a4d9737430f3ff52be2403a5bc9e1addbeaec909` |
| Step 50 | `8041917c09e080313c05b87429459ea0e5050daca77027fb9631d8cd8e399df3` | `445f619e925be116f3647b13f97db6280100668ed907605c632a737d34414042` |

These 242,249,092-byte checkpoints are ignored local research artifacts.
They are not release or Hugging Face artifacts.

## Translation quality

All values below are unquantized greedy decoding against the same frozen safe
parent. Each suite was generated independently for the parent and both
checkpoints.

### Fresh V15 selection suite (768)

| Model | BLEU | Corpus chrF++ | Mean sentence chrF++ | Long mean chrF++ | Omission-risk mean chrF++ | Generation failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Safe parent | 21.669 | 46.226 | 45.232 | 46.566 | 40.940 | 4 |
| Step 25 | 22.222 | 46.559 | 45.577 | 46.767 | 43.566 | 5 |
| Step 50 | 22.212 | 46.549 | 45.567 | 46.743 | 43.560 | 5 |

Step 25 gains `+0.333` corpus and `+0.345` mean sentence chrF++, with a
paired 90% interval of `+0.100…+0.599`. Step 50 gains `+0.323` and `+0.335`,
with `+0.087…+0.595`. The aggregate gains are real on this suite. Both
checkpoints nevertheless introduce the same new generation failure:
`v15-valid:negation:jlt:law-3215:tu-2957:ja-en`.

### V12 regression suite (1,536)

| Model | BLEU | Corpus chrF++ | Mean sentence chrF++ | Generation failures |
| --- | ---: | ---: | ---: | ---: |
| Safe parent | 23.103 | 47.625 | 48.382 | 14 |
| Step 25 | 23.692 | 47.968 | 48.685 | 15 |
| Step 50 | 23.706 | 47.970 | 48.707 | 15 |

Both checkpoints pass the aggregate and worst-stratum retention requirements,
but introduce the same new generation failure:
`v12-valid:omission-risk:jlt:law-4754:tu-836:ja-en`.

### V14 regression suite (768)

| Model | BLEU | Corpus chrF++ | Mean sentence chrF++ | Omission-risk mean chrF++ | Generation failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Safe parent | 21.295 | 45.821 | 44.770 | 45.631 | 9 |
| Step 25 | 22.244 | 46.329 | 45.293 | 44.287 | 9 |
| Step 50 | 22.234 | 46.323 | 45.280 | 44.126 | 9 |

This is the decisive generalization failure. The new omission objective raises
fresh V15 omission risk by about 2.62 points, yet lowers the frozen V14
omission stratum by `-1.344` at step 25 and `-1.505` at step 50. The gate
allows at most `-0.50`. Aggregate gains therefore mask a worse known safety
slice.

## Contrast diagnostics

| Diagnostic | Step 0 | Step 25 | Step 50 | Required change |
| --- | ---: | ---: | ---: | ---: |
| Recovery preference accuracy | 0.72314 | 0.72314 | 0.72363 | at least +0.05 |
| Omission preference accuracy | 0.95508 | 0.95557 | 0.95557 | at least +0.05 |
| Clean-over-recovery accuracy | 0.76416 | 0.76465 | 0.76416 | at least -0.02 |
| Mean rejected-token probability | 0.010150 | 0.010074 | 0.010041 | no increase |

The retention diagnostics pass, but the two intended behavior changes are
effectively zero. The training set is mostly already outside the active hinge:
at initialization, recovery accuracy is 72.3%, omission accuracy is 95.5%,
mean recovery margin is 5.28, and mean omission margin is 10.13. The registered
small-margin objective therefore supplies little gradient on most examples.

## Failed gates

Step 25 fails six frozen gates:

- V14 worst stratum and omission risk (`-1.344`, minimum `-0.50`);
- recovery preference improvement (`0.000`, minimum `+0.05`);
- omission preference improvement (`+0.00049`, minimum `+0.05`);
- one new generation failure on fresh V15;
- one new generation failure on V12 regression.

Step 50 fails the same six plus fresh long legal (`+0.177`, minimum `+0.20`);
its V14 omission delta worsens to `-1.505`.

Each checkpoint also creates 63 exact/typed/negation detector-disagreement
cases across the three suites. Those cases would require semantic
adjudication only if every earlier gate passed. They were not sent to an LLM
judge.

## What changed and what did not

The model does repair some obvious loops. On
`v15-valid:omission-risk:jlt:law-3215:tu-3700:ja-en`, the safe parent repeats
“paragraph 68” until the generation limit, while the candidate produces a
finite sentence and gains about 21.8 sentence chrF++ points. This confirms
that the initialization and objective can alter exposure-driven behavior.

The changes are not reliably safe. On
`v15-valid:long:jlt:law-4746:tu-157:ja-en`, the candidate drops the Ministry
of Economy, Trade and Industry from a legal provision and loses 33.4 sentence
chrF++ points. On
`v14-valid:terminology-risk:jlt:law-3215:tu-3883:ja-en`, it truncates the
defined accreditation phrase and loses 35.0 points.

## Next experiment

Do not continue V15 with a post-hoc larger weight. A new contract should:

1. mine only active or near-active sequence-level omission and recovery
   violations instead of spending most contrast batches on already-easy
   token comparisons;
2. score the complete licensed reference against an explicit
   licensed-reference deletion/repetition negative, so later continuation
   cannot erase the omission signal;
3. balance fresh, V12, and V14 hard examples in the training contrast set,
   rather than relying on one newly sampled omission slice;
4. start from the safe parent or a freshly interpolated safe point and require
   zero new generation failures during development;
5. measure gradient contribution per objective before freezing the next arm.

V15 does not justify MoE or a larger bundle. The registered objective barely
moved its target diagnostics; that is an optimization/data-hardness failure,
not evidence that the compact model lacks capacity.
