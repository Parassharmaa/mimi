# Canonical-target distillation experiment — 2026-07-25

## Decision

Stop the 250-step canonical-target recipe. Keep Mimi's current translator
unchanged.

The experiment found a real EN→JA quality gain after exact MLX q4 conversion,
but the required bidirectional system failed JA→EN quality, deterministic
critical-safety, generation-stability, and distribution-provenance gates.
Training may not continue to 1,000 steps, neither model may replace the current
Mimi engines, and these experiment artifacts are not authorized for public
upload.

The machine result is
`canonical-target-student-v3-result-2026-07-25.json` (SHA-256
`5469c71b48e02da6688ad51250f3286c48f7596249fcceb0de017dd640b54587`).
The preregistered contract is
`canonical-target-student-v3-contract-2026-07-25.json` (SHA-256
`49f52ca20f9e12a66bc0a61a179dcf85b85387aed5facdd49fbbf3b55f2d7351`).

## Keyless teacher execution

GPT-5.6 Sol teacher translations were generated through the Codex CLI's cached
ChatGPT authentication. No OpenAI API key was read, stored, forwarded, or used.
Requests were source-only and stored final translations plus compact risk tags,
never private reasoning traces.

The final scale run generated one canonical target for each of 400 fresh,
balanced sources: 200 EN→JA and 200 JA→EN across conversation, news, law,
Wikipedia, hard professional text, long documents, and Mimi UI. Ten resumable
40-row shards completed with 231,662 reported Codex tokens.

## Why the candidate design changed

The first 87-source experiment asked one teacher for three stylistic
paraphrases and required two independent judges to agree on one unique best
teacher candidate. Zero of 67 deterministically admitted rows passed. This was
an identifiability failure, not evidence that every translation was bad:
multiple valid paraphrases often split the vote.

The replacement design selected exactly one canonical teacher target before
judgment. Two distinct judge families then evaluated that fixed target
absolutely:

- adequacy exactly 4/4;
- fluency at least 3/4;
- terminology at least 3/4;
- protected tokens preserved;
- no critical error;
- no error tags.

The licensed human reference and current Mimi output remained anonymous
comparison candidates, but the gate no longer required the canonical target to
be uniquely preferable to every acceptable paraphrase. This is a new,
preregistered experiment; the earlier results were not retroactively approved.

The 40-source validation v2 admitted 22 rows after deterministic filtering and
approved 17: 11 EN→JA and 6 JA→EN. It passed the registered 12-total,
5-per-direction scaling gate. The fresh 400-source v3 admitted 223 rows and
approved 180: 97 EN→JA and 83 JA→EN. It passed the harder 120-total,
50-per-direction gate.

The independent judges were Claude Fable 5 and the pinned Apache-2.0
Qwen3-8B 4-bit MLX model. Their judgments and the deterministic admission gate
were hash-bound. A redundant Qwen source-ID typo was normalized only when its
candidate coverage matched exactly; no score, tag, or judgment was repaired.

## Training data

Each direction used an exact 25% reviewed-synthetic mixture:

| Direction | Canonical targets | Same-source human anchors | General human replay | Train | Human-only validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| EN→JA | 97 | 97 | 194 | 388 | 1,285 |
| JA→EN | 83 | 83 | 166 | 332 | 1,285 |

All rows retain source/target license and provenance. Sources cover
CC-BY-2.0-FR, CC-BY-4.0, CC-BY-SA-3.0, Japan Public Data License-compatible
terms, and project-owned UI pairs. Approved-source replay was excluded,
train/validation exact overlap is false, and nine protected suites were
screened with the registered character-ngram policy.

The release converter found a schema integration defect: the mixed dataset
manifest preserves per-row licenses and aggregate counts but does not expose
the legacy top-level `effective_licenses` map consumed by release packaging.
Both q4 manifests therefore truthfully report
`provenance-incomplete-not-approved-for-distribution`. This is a packaging-gate
failure even though the underlying rows are licensed.

## Student and training

The student remained the existing pair of intact Marian 6-encoder /
6-decoder models:

- EN→JA parent weights:
  `d080ce30490e878ab745206220c97bf18d4b2662a35e5fff0bdd2157a7c4dc4b`;
- JA→EN parent weights:
  `8e7f7eff76d74b343884fe9a170b6dbad55d42f20ac5f526b6e8ec71e6c94f71`.

Each direction ran 250 steps at learning rate `2e-6`, batch 8, gradient
accumulation 4, effective batch 32, warmup 25, maximum 192 source and target
tokens, frozen-parent KL weight `0.10` on licensed human rows, and L2-to-parent
weight `0.01`. No reasoning traces were used.

Full-precision human-only validation moved:

| Direction | Step 0 chrF++ | Step 250 chrF++ | Delta |
| --- | ---: | ---: | ---: |
| EN→JA | 31.0157 | 31.0949 | +0.0792 |
| JA→EN | 52.6017 | 52.5846 | −0.0171 |

The first JA→EN run correctly retained step 0 as its best checkpoint, which
also meant the actual step-250 weights were not saved. A deterministic replay
with checkpoint retention reproduced chrF++ and every slice exactly; loss
differed by only `8e-9`. The retained step-250 weights hash to
`812d0c7850f90e60e5f100277189a4ee6531e805ed76687aaced33ca46a440af`.

Both actual step-250 checkpoints were converted with MLX 0.30.6 to exact affine
q4/group-64 packs. Each directory is 39,139,880 bytes; the pair is 78,279,760
bytes.

## Exact-q4 benchmark

The candidate and current Mimi were evaluated with the same MLX runtime,
tokenizer versions, cached greedy decoding, preallocated 192-token KV blocks,
precomputed 192×512 position table, maximum 192 output tokens, and one warm
run per case.

### Composed documents

| Metric | EN→JA current | EN→JA candidate | JA→EN current | JA→EN candidate |
| --- | ---: | ---: | ---: | ---: |
| chrF++ | 28.4063 | 29.2845 | 55.1903 | 54.8708 |
| sacreBLEU intl | 9.6287 | 9.8116 | 30.6207 | 30.2131 |
| COMET-22 | 0.86688 | 0.87526 | 0.81921 | 0.82001 |

EN→JA mean paired sentence chrF++ improved `+0.7510`, with registered 90%
bootstrap interval `[+0.2050, +1.3582]`. COMET improved `+0.00837`, interval
`[+0.00389, +0.01343]`. It therefore passed two independent point-improvement
signals and both non-inferiority bounds.

JA→EN mean paired sentence chrF++ regressed `−0.3689`, interval
`[−1.0499, +0.3048]`; corpus BLEU regressed `−0.4075`. COMET moved only
`+0.00079`, below the registered `+0.002` improvement signal, although its
interval remained within the COMET non-inferiority floor. JA→EN passed zero of
the required two improvement signals. A later quality judge could contribute
at most one remaining signal, so it could not change the continuation decision
and was not run.

EN→JA gains concentrate in ministry legal, long legal, and news content.
JA→EN regresses in conversation, ministry legal, and long-document news while
improving some Wikipedia slices. This argues for direction-specific data and
regularization, not a symmetric continuation.

### Runtime and size

| Gate | EN→JA | JA→EN | Requirement | Result |
| --- | ---: | ---: | ---: | --- |
| Warm segment p95 | 80.5 ms | 90.4 ms | ≤175 ms each | pass |
| Peak process RSS | \- | 222.9 MB pair maximum | ≤250 MB | pass |
| Two-direction bundle | \- | 78.3 MB | preferred ≤150 MB; hard <500 MB | pass |

Direct long-unit p95 is higher because it measures whole documents rather than
interactive segments; it remains diagnostic. The registered real-time gate is
segment p95.

## Safety and stability

Aggregate failure totals alone are misleading because a candidate can fix one
case while breaking a different one. The gate therefore compares case-ID set
difference.

| Check | Current failures | Candidate failures | New candidate cases | Gate |
| --- | ---: | ---: | ---: | --- |
| Exact critical tokens | 62 | 60 | 2 | fail |
| Typed number/date/unit/token policy | 64 | 63 | 4 | fail |
| Negation | 32 | 30 | 2 | fail |
| Union | 96 | 96 | 2 | fail |
| Repetition/generation-limit | 6 | 4 | 2 | fail |

Every registered allowance is zero. The candidate therefore fails even though
some aggregate totals decrease.

## What this teaches us

1. One canonical final translation plus absolute independent judging is
   substantially more data-efficient than ranking three near-equivalent
   paraphrases.
2. The teacher data contains useful EN→JA legal/news signal: both chrF++ and
   COMET improve after q4.
3. Applying the same small 25% mixture to both directions is not justified.
   JA→EN needs a different selection/data ratio or stronger preservation.
4. Global KL/L2 regularization does not prevent case-level critical regressions.
   Critical examples need explicit counterfactual replay or a constrained
   decoding/verification layer.
5. A Mixture-of-Experts expansion is premature. The present blocker is
   directional data and safety behavior, not capacity, latency, or bundle size.
6. Future dataset builders must emit both per-row provenance and the release
   packager's authenticated `effective_licenses`/attribution schema before
   training begins.

## Registered next experiment

Do not continue these checkpoints. A new contract should test the evidence-backed
asymmetry:

- EN→JA: a smaller follow-up around the successful legal/news canonical rows,
  with explicit counterfactual critical-token and negation replay;
- JA→EN: return to the current parent and run a human-heavy or much lower
  synthetic-ratio ablation, stratified by conversation and legal content;
- both directions: make the dataset manifest release-compatible before
  training, retain the exact final checkpoint even when step 0 wins, and
  preserve the same q4 held-out quality/safety/runtime/size gates.

Only after a direction-specific recipe passes should routed experts or a
single shared-encoder/bidirectional student be revisited.

## Post-hoc routed-expert diagnostic

After the registered stop, one explicitly exploratory simulation tested whether
the positive EN→JA model could be retained as a source-only safety-routed
expert while JA→EN stayed exactly current. This was designed after seeing the
development results, so it is not claim-eligible and was not run on the
protected 2,800-case stress suite.

The preflight routes EN→JA sources with exact critical tokens or deterministic
negation markers to current Mimi. Other sources attempt the expert, then fall
back when source/output-only exact or typed critical tokens, negation parity,
generation length, repetition, empty-output, or runtime guards fail. It never
uses a reference or metric at runtime. JA→EN always uses current Mimi.

The simulation routes 90/200 EN→JA segments to the expert, keeps every JA→EN
case current, introduces zero new deterministic exact/typed/negation/union or
direct-generation failures, and totals 117,420,087 model bytes. EN→JA
composed-document mean paired chrF++ improves only +0.2934 with registered 90%
interval `[−0.0334, +0.7166]`. COMET-22 improves +0.00196 with interval
`[−0.00031, +0.00463]`, narrowly below the +0.002 signal. Simulated selected
latencies exclude attempted-expert plus fallback time and simultaneous
residency, so they are not runtime evidence.

This preserves the safety lesson but does not justify a routed release or more
post-hoc threshold tuning. A fresh direction-specific training experiment is
the next valid step.
