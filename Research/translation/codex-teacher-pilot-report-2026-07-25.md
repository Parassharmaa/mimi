# Codex teacher pilot for Mimi translation

Date: 2026-07-25

## Outcome

Mimi can generate English↔Japanese distillation candidates through the locally
authenticated Codex CLI without an OpenAI API key. The transport uses the
cached ChatGPT login, `gpt-5.6-sol`, reasoning effort `none`, strict Structured
Outputs, an ephemeral session, a read-only sandbox, and no user configuration
or repository rules. Only the already sealed source fields are sent:
`source_id`, source language, target language, domain, and source text.
Licensed references and student outputs are absent.

This is a teacher-data transport result, not a model promotion result. After a
reference-validated deterministic re-filter and two blinded independent judge
paths, the unchanged consensus gate approved **0/67** sources. No candidate
from this pilot is approved for training, no student has been trained from it,
and Mimi's shipped translator is unchanged. The remaining 335 teacher shards
are stopped.

## Reproducible artifacts

- Frozen request corpus:
  `Research/translation/work/gpt56-final-translation-pilot-v1.requests.jsonl`
- Request SHA-256:
  `17eede0183f2863190533867282a75de6d11179e79e066a31b260d60a787e3b7`
- Source-only rows: 16,000
- Codex run manifest:
  `Research/translation/work/gpt56-final-translation-pilot-v1.codex-teacher-v1/manifest.json`
- Deterministic shard contract: at most 48 rows and 12,000 source characters
- Frozen shard count: 337
- Runner: `scripts/translation/run_codex_teacher.py`
- Training-reference diagnostic:
  `scripts/translation/evaluate_codex_teacher_outputs.py`
- Offline transport contract:
  `scripts/translation/test_codex_teacher_pipeline.py`

Generated work artifacts remain ignored and local. They must not be published
as an approved dataset until the existing licensing, contamination,
deterministic-safety, and independent-consensus gates succeed.

## Safety and provenance contract

The runner:

1. revalidates the existing Batch request contract;
2. rejects any teacher input with fields beyond the five source-only fields;
3. binds the run manifest to the request SHA-256, prompt SHA-256, schema
   SHA-256, exact IDs, and deterministic shard payload hashes;
4. invokes `codex exec` with cached ChatGPT authentication, no API key,
   `--ephemeral`, `--sandbox read-only`, `--ignore-user-config`, and
   `--ignore-rules`;
5. stores final structured translations only, never reasoning traces;
6. validates exact result cardinality, IDs, candidate styles, risk tags, and
   schema before writing a shard;
7. records per-shard Codex version, prompt/result hashes, timestamps, and
   reported token use; and
8. emits Batch-compatible JSONL for the existing fail-closed filtering and
   independent-review pipeline.

No Zero Data Retention or equivalent retention property is claimed for this
transport. The local run metadata records only what Mimi can verify.

## Deterministic critical-token audit

The original exact-surface filter admitted 52/87 sources. Its rejections mixed
real failures with safe cross-script forms such as source `40` becoming
Japanese `40か国`. The replacement remains fail-closed:

- URLs, placeholders, printf tokens, and markup must still match exactly;
- source-only rows retain exact critical-surface matching;
- a referenced row may use the human target as a typed bilingual witness only
  when the source/reference protected-token contract itself is valid; and
- the reference authorizes only the exact target-side critical structure, not
  general semantic acceptance.

Adversarial tests cover wrong months, wrong URLs, Japanese digit adjacency,
percentage suffixes, temporal forms, and typed number forms. The re-filter
admitted 67/87 sources: 44 EN→JA and 23 JA→EN. Its 265 anonymous candidates
contain 198 teacher candidates, 64 licensed-reference candidates, and three
teacher/reference equivalents. Critical admission modes are 147 source-strict,
53 reference-target-strict, and one reference-typed teacher candidate. The
queue SHA-256 is
`607994bd8feea41e622ce623bd13c9d55a28ebcaa0ecafe36e903b37f9d5bd06`.

## Measured pilots

An early one-row probe accidentally included a seed object containing its
licensed reference. It is retained only as an execution-path test and excluded
from every quality result below.

An eight-row blind source-only probe verified both directions, short inputs,
and long documents under the strict batch schema. It reported 22,653 tokens.
On the seven rows with licensed references, natural/meaning-conservative
chrF++ was about 30 for EN→JA and 64–65 for JA→EN. This sample is too small for
decision-making.

### Frozen shard 00000: EN→JA long-document stress

- Rows: 39, all EN→JA long-document news
- Source characters: 11,728
- Runtime: 426.46 seconds
- Reported tokens: 15,850
- Structured-output completion: 39/39
- Deterministic admission after the adversarial reference-validated audit:
  26/39 sources (66.67%)
- Rejections: eight critical-token mismatches and five length-ratio failures

| Candidate | chrF++ | BLEU |
|---|---:|---:|
| Natural spoken | 28.41 | 8.89 |
| Concise caption | 18.69 | 6.47 |
| Meaning conservative | 31.86 | 9.06 |
| Reference-leaking per-row oracle upper bound | 32.43 | 9.30 |

The reference-leaking oracle is diagnostic only and cannot be used to select
training candidates.

### Frozen shard 00169: mixed-direction boundary

- Rows: 48
- Composition: 23 EN→JA Mimi UI rows and 25 JA→EN long-document news rows
- Source characters: 4,049
- Runtime: 245.92 seconds
- Reported tokens: 37,369
- Structured-output completion: 48/48
- Deterministic admission after the adversarial reference-validated audit:
  41/48 sources (85.42%)
- Admitted queue: 161 candidates
- Blinded licensed-reference candidates: 38
- Exact teacher/reference equivalents: 3
- Rejections: two critical-token mismatches, four duplicate-candidate failures,
  and one length-ratio failure

| Direction | Candidate | chrF++ | BLEU |
|---|---|---:|---:|
| EN→JA | Natural spoken | 46.89 | 15.34 |
| EN→JA | Concise caption | 39.39 | 20.32 |
| EN→JA | Meaning conservative | 40.76 | 2.90 |
| EN→JA | Reference-leaking oracle upper bound | 53.53 | 23.74 |
| JA→EN | Natural spoken | 63.08 | 39.25 |
| JA→EN | Concise caption | 46.37 | 18.20 |
| JA→EN | Meaning conservative | 63.71 | 38.81 |
| JA→EN | Reference-leaking oracle upper bound | 66.07 | 42.64 |

These references belong to training-data diagnostics. They are not the held-out
development/test benchmark and cannot support an accuracy-improvement claim.

## Independent blinded consensus

Both judges received the same randomized candidate IDs, source text, and
translations. Neither received candidate origin, reference provenance, metric
scores, or teacher metadata. Neither path stored reasoning traces.

Judge A was Claude Fable 5 through the locally authenticated Claude CLI. It
completed 67/67 sources, reported $2.722298, uniquely selected a threshold-
eligible best candidate for 10 sources, and abstained on a top-score tie for
57. Its collected-output and normalized-judgment SHA-256 values are
`3013b4d4b672a80b0c4c8a4c6e9dfa3acb28ffea453e6888134c1eca612c9e92`
and
`677b086a583cbc9843ab2060a37572e9f8f22895e57c412852f7b13072e7d5e2`.

Judge B was the pinned Apache-2.0
`mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192`
running locally with MLX. It compared all candidates side by side, completed
67/67 sources, uniquely selected 12, abstained on 53 ties, and found no
error-free threshold candidate for two. The judge is evaluation infrastructure,
not a shippable translation candidate: its snapshot is 4,623,784,971 bytes and
the run peaked at 4,963,336,192 resident bytes. Output and normalized-judgment
SHA-256 values are
`d5609e755221c646f6834ef610976f9cce2b6a75c9359f5e1ad920d13e26c5b8`
and
`fa637551167eb9e963f0e68f285b2e8a16de67861daedbb3cd24e5c4eb85a651`.

The gate requires adequacy at least 4, fluency and terminology at least 3, no
error tags, no critical error, protected-token preservation, one unique best
per judge, the same candidate from both judges, and a teacher—not reference—
winner. Five sources had a unique selection from both judges, but all five
selections disagreed. The gate therefore approved 0 and rejected 67. The empty
approved artifact has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
the rejection artifact hashes to
`495847af662b75cfd62acd4e60ed334fd98b43ee7eb57e04b16469786ab7de7d`.
The machine-readable decision is
`Research/translation/codex-teacher-pilot-consensus-2026-07-25.json`.

## Interpretation

Codex CLI is viable as a high-quality teacher transport, and batching
substantially amortizes its fixed context overhead. The two production shards
also show that reported token use is not predicted by source characters alone:
candidate output length, language direction, and schema generation matter.
Using the observed 4.1–7.1 minute range, a naive sequential run of all 337
shards would take roughly 23–40 hours. The observed token envelope extrapolates
very loosely to about 5.3–12.6 million reported tokens. Neither estimate is a
quota or cost guarantee.

Forty-eight-result shards work, but smaller 24–32 item shards would improve
failure recovery and interactive latency. Changing shard size requires a new
manifest; completed results can be imported by ID only after an explicit,
hash-checked migration tool exists.

The critical-token audit is complete, but independent consensus did not pass.
The result does not show that every translation is poor: both judges frequently
regarded two or more candidates as equally strong. It does show that the
three-style candidate design cannot produce an auditable unique target under
the registered gate. Treating ties as wins, lowering score thresholds, or
choosing between styles after seeing judge results would weaken the experiment
and is prohibited.

## Stop/go gate before scaling

Do not run the remaining 335 shards. The deterministic audit passed, but the
two-judge gate produced no approved dataset, so the registered 250-step
continuation has no valid input.

The next falsifiable data-design pilot should remove vote splitting without
weakening review: preregister one canonical teacher translation per source and
compare it blindly with the licensed human reference and the current Mimi
baseline. Do not use these judge results to select one of the three existing
styles. Run only a small, direction-balanced sample first. Scale or train only
if that new design yields a sufficiently sized, direction-balanced approved
set under the same deterministic and two-family consensus thresholds.

This pilot does not authorize app integration, a default-model change, public
dataset publication, or a release. Because the approved dataset is empty,
nothing from this pilot is eligible for public Hugging Face publication.

The proposed canonical-target follow-up has since been completed under a fresh
contract; no result from this three-style pilot was retroactively approved.
See `canonical-target-distillation-report-2026-07-25.md`. The canonical design
approved useful data and improved EN→JA, but the exact-q4 bidirectional student
still failed JA→EN, safety, generation-stability, and packaging-provenance
gates, so Mimi remains unchanged.

## Commands

```sh
python3 scripts/translation/run_codex_teacher.py prepare \
  Research/translation/work/gpt56-final-translation-pilot-v1.requests.jsonl \
  Research/translation/work/gpt56-final-translation-pilot-v1.codex-teacher-v1 \
  --maximum-items 48 \
  --maximum-source-characters 12000

python3 scripts/translation/run_codex_teacher.py run \
  Research/translation/work/gpt56-final-translation-pilot-v1.requests.jsonl \
  Research/translation/work/gpt56-final-translation-pilot-v1.codex-teacher-v1 \
  --start-shard 169 \
  --maximum-shards 1

python3 scripts/translation/run_codex_teacher.py status \
  Research/translation/work/gpt56-final-translation-pilot-v1.requests.jsonl \
  Research/translation/work/gpt56-final-translation-pilot-v1.codex-teacher-v1

# Collection intentionally fails until every frozen shard is complete.
python3 scripts/translation/run_codex_teacher.py collect \
  Research/translation/work/gpt56-final-translation-pilot-v1.requests.jsonl \
  Research/translation/work/gpt56-final-translation-pilot-v1.codex-teacher-v1 \
  Research/translation/work/gpt56-final-translation-pilot-v1.output.jsonl
```
