# Claim-ready benchmark workflow

Mimi may not claim that a local model beats Apple from the checked-in canary.
The legacy strongest-evidence suite requires exactly 400 human-authored,
non-synthetic cases in each direction with the exact domain allocation in
`manifest.json`. Each case needs two independently reviewed references and a
distinct adjudicator. Keep the draft, packets, identities, and responses outside
any training path.

The user has authorized a no-human-review path. That does not silently weaken
this contract. A separate preregistered policy now lives in
`automated-claim-v1.manifest.json`, with executable validation in
`validate_automated_benchmark_suite.py` and promotion evaluation in
`evaluate_automated_translation_promotion.py`. It requires exactly 400 cases
per direction; frozen source/reference hashes; complete authenticated
project-controlled exposure plus explicit upstream revision bounds;
source/reference generators isolated from training teachers;
distinct-model-family bilingual consensus; deterministic and semantic
contamination evidence; positive paired lower bounds against the frozen best
prior local model; strong absolute quality; a 250 ms warm-p95 ceiling; and a
union-veto of zero critical errors. Apple is diagnostic-only in this lane.

The policy is implemented and tested. Its source side is now frozen in
`automated-claim-v1.sources.jsonl`: 800 unique deterministic project-owned
scenarios, exactly 400 per direction, SHA-256
`f039ce456c55f051e8bbcc13ed9bc8270a722819308e008b39da7f30327ec16c`.
Every row remains source-only and `claimEligible: false`. A current-release
lineage scan checked 1,358,264 exposed texts across 15 authenticated
training/validation/memory/public files with exact matching and normalized
character-5-gram Jaccard capped at 0.65; it passed. The stronger schema-v2
exposure freeze now binds 17 text assets, 14 evidence assets, and 1,411,076 raw
strings. It explicitly records `upstreamExactRowsComplete: false`; the opaque
ElanMT pretraining rows are bounded only by two pinned May 2024 model revisions
that predate the private July 2026 sources. An exhaustive MiniLM semantic scan
then compared all 800 sources with 599,317 normalized-unique controlled strings:
zero exceeded 0.82, with a maximum of 0.798585. References and independent judge
reports do not yet exist, so every row remains `claimEligible: false`.

The final-output-only reference lane is pinned to `gpt-5.6-sol` for three
candidates, `gpt-4o-2024-08-06` for judge A, and `gpt-4.1-2025-04-14` for judge
B. They are three distinct model families and no training teacher exists in
this release lineage. The exact v2 generator file is sealed at SHA-256
`d6f85b9af0f10767067c66d1332a6334233044ee28674694612ea2614db6822a`,
uses `store: false`, strict Structured Outputs, `reasoning.effort: none`, and a
1,024-token final-output allowance. It was submitted on 2026-07-21.

The earlier high-reasoning file at SHA-256
`29b877556aaa037b9e353408de32ff651e4b85d2bbb417b00cb3ac4b1efe4d74`
is quarantined and ineligible. Its API batch completed at the transport layer,
but 10/800 response bodies were incomplete at the 768-token limit and 300/800
contained one or more encrypted reasoning items. The collector rejected the first case and
no candidate was admitted. [Official GPT-5.6 guidance](https://developers.openai.com/api/docs/guides/latest-model#update-api-and-model-parameters)
documents that encrypted reasoning is returned by default for `store: false`
when reasoning is used, so v2 disables reasoning rather than weakening the
no-trace collector.

## Reviewer-free automated lane

The automated suite must use newly sealed project-owned/CC0 source material,
not reverse, paraphrase, or resample KFTT, ALT, Tatoeba, BTEC, Japanese Law,
FLORES, WMT, Mimi UI text, existing prompts, or any training/model-selection
asset. The exact per-direction allocation is 120 meeting/live-speech, 80
everyday conversation, 60 macOS/technical UI, 60 numbers/dates/entities, 60
politeness/ambiguity/omission, and 20 code-switching cases.

The exact deterministic source render is reproducible and refuses to overwrite
an existing freeze:

```sh
python3 scripts/translation/prepare_automated_claim_sources.py \
  Research/translation/benchmark/automated-claim-v1.sources.jsonl \
  --manifest-output \
  Research/translation/benchmark/automated-claim-v1.sources.manifest.json

PYTHONPATH=scripts/translation python3 \
  scripts/translation/audit_automated_claim_source_contamination.py \
  Research/translation/benchmark/automated-claim-v1.sources.jsonl \
  Research/translation/release/elanmt-release-clean-human-routed-moe-v2-memory-v2/release-contract.json \
  Research/translation/results/automated-claim-v1-source-release-lineage-contamination.json \
  --protected-jsonl Research/translation/benchmark/canary.jsonl \
  --protected-jsonl Research/translation/benchmark/public-stress-v1.jsonl \
  --protected-jsonl Research/translation/benchmark/public-stress-v2.jsonl \
  --protected-jsonl Research/translation/benchmark/public-stress-v3.jsonl

# The complete command, including evidence paths, is reproducibly exercised by
# test_automated_claim_exposure_manifest.py. The real frozen output is:
PYTHONPATH=scripts/translation python3 -c \
  'import json; from pathlib import Path; from validate_automated_benchmark_suite import validate_exposure_manifest; p=Path("Research/translation/benchmark/automated-claim-v1.exposure.manifest.json"); m=json.loads(Path("Research/translation/benchmark/automated-claim-v1.manifest.json").read_text()); validate_exposure_manifest(p,m)'

uv run --python 3.12 --with torch --with transformers==4.57.6 \
  --with sentencepiece --with numpy \
  env PYTHONPATH=scripts/translation python3 \
  scripts/translation/scan_automated_claim_semantic_contamination.py \
  Research/translation/benchmark/automated-claim-v1.sources.jsonl \
  Research/translation/benchmark/automated-claim-v1.manifest.json \
  Research/translation/benchmark/automated-claim-v1.exposure.manifest.json \
  Research/translation/results/automated-claim-v1-source-semantic-contamination.json \
  --cache-directory Research/translation/models/hf-cache --device mps

python3 scripts/translation/prepare_automated_claim_reference_batch.py \
  Research/translation/benchmark/automated-claim-v1.sources.jsonl \
  Research/translation/benchmark/automated-reference-model-plan-v1.json \
  Research/translation/benchmark/automated-claim-v1.reference-generator.prompt.txt \
  Research/translation/work/automated-claim-v1/reference-generator-final-only-v2.requests.jsonl

python3 scripts/translation/run_synthetic_batch.py validate \
  Research/translation/work/automated-claim-v1/reference-generator-final-only-v2.requests.jsonl
```

After the credentialed generator Batch completes, collect only final Structured
Output candidates, create two independently shuffled blinded judge files, and
collect their reports. Each collector requires exact case coverage, exact model
revision, source/request/response hashes, `store: false`, and no visible or
encrypted reasoning trace. The assembler writes no suite if even one case has
fewer than two candidates accepted perfectly by both judges.

```sh
PYTHONPATH=scripts/translation python3 \
  scripts/translation/collect_automated_claim_reference_candidates.py \
  Research/translation/benchmark/automated-claim-v1.sources.jsonl \
  Research/translation/work/automated-claim-v1/reference-generator-final-only-v2.requests.jsonl \
  Research/translation/benchmark/automated-reference-model-plan-v1.json \
  PRIVATE_PATH/reference-generator.responses.jsonl \
  PRIVATE_PATH/reference-generator.report.json \
  PRIVATE_PATH/reference-candidates.jsonl

for ROLE in reference-judge-a reference-judge-b; do
  PYTHONPATH=scripts/translation python3 \
    scripts/translation/prepare_automated_claim_reference_judge_batch.py \
    Research/translation/benchmark/automated-claim-v1.sources.jsonl \
    PRIVATE_PATH/reference-generator.report.json \
    Research/translation/benchmark/automated-reference-model-plan-v1.json \
    Research/translation/benchmark/automated-claim-v1.reference-judge.prompt.txt \
    "$ROLE" PRIVATE_PATH/$ROLE.requests.jsonl

  python3 scripts/translation/run_synthetic_batch.py validate \
    PRIVATE_PATH/$ROLE.requests.jsonl

  # Submit and collect the sealed request with run_synthetic_batch.py, then:
  PYTHONPATH=scripts/translation python3 \
    scripts/translation/collect_automated_claim_reference_judgments.py \
    Research/translation/benchmark/automated-claim-v1.sources.jsonl \
    PRIVATE_PATH/reference-generator.report.json \
    PRIVATE_PATH/$ROLE.requests.jsonl \
    Research/translation/benchmark/automated-reference-model-plan-v1.json \
    "$ROLE" PRIVATE_PATH/$ROLE.responses.jsonl PRIVATE_PATH/$ROLE.report.json
done

PYTHONPATH=scripts/translation python3 \
  scripts/translation/assemble_automated_claim_reference_suite.py \
  Research/translation/benchmark/automated-claim-v1.sources.jsonl \
  PRIVATE_PATH/reference-generator.report.json \
  PRIVATE_PATH/reference-judge-a.report.json \
  PRIVATE_PATH/reference-judge-b.report.json \
  PRIVATE_PATH/automated-heldout.jsonl PRIVATE_PATH/reference-decisions.json

PYTHONPATH=scripts/translation python3 \
  scripts/translation/audit_automated_claim_reference_structures.py \
  PRIVATE_PATH/automated-heldout.jsonl \
  PRIVATE_PATH/reference-judge-a.report.json \
  PRIVATE_PATH/reference-judge-b.report.json \
  PRIVATE_PATH/reference-structural-audit.json
```

For every case, an isolated reference generator emits at least three final
translation candidates. Two judges from distinct model families, both separate
from the generator and every training teacher, must independently accept the
same two references with maximum adequacy, fluency, and terminology scores,
unchanged protected tokens, no error tags, and no critical flag. Request,
response, prompt, model-revision, source, and reference hashes are mandatory;
reasoning traces are never stored.

Before validation, build a complete project-controlled exposure manifest that
covers training, development, teacher inputs/outputs, routing, model selection,
and exact memory. Opaque upstream rows must never be called complete: use pinned
revision temporal evidence or provide the actual rows. Every controlled text
asset must have a hash-bound JSONL extraction. The validator
checks all extracted text for exact/document and normalized character-5-gram
overlap and requires a separately pinned multilingual semantic-neighbor report.
Missing scope, changed input, incomplete coverage, judge disagreement, or one
critical flag fails the whole freeze.

```sh
PYTHONPATH=scripts/translation python3 \
  scripts/translation/validate_automated_benchmark_suite.py \
  PRIVATE_PATH/automated-heldout.jsonl \
  Research/translation/benchmark/automated-claim-v1.manifest.json \
  PRIVATE_PATH/reference-generator.json \
  PRIVATE_PATH/reference-judge-a.json \
  PRIVATE_PATH/reference-judge-b.json \
  PRIVATE_PATH/reference-structural-audit.json \
  PRIVATE_PATH/complete-exposure-manifest.json \
  PRIVATE_PATH/semantic-contamination.json \
  --output PRIVATE_PATH/automated-heldout-validation.json
```

After freezing, evaluate the exact local candidate and frozen best prior local
model. Two separately pinned judge families blindly score candidate/baseline
outputs. A deterministic critical report is unioned with both judges, so
consensus can never outvote one critical error. Exact Swift/MLX parity, current
archive integrity, the non-Apple failure path, hard 500 MB size, 150 MB
preferred size, peak memory, and absolute latency remain executable gates.

```sh
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/evaluate_automated_translation_promotion.py \
  PRIVATE_PATH/automated-heldout.jsonl \
  Research/translation/benchmark/automated-claim-v1.manifest.json \
  PRIVATE_PATH/automated-heldout-validation.json \
  Research/translation/results/candidate-heldout.json \
  Research/translation/results/prior-local-heldout.json \
  Research/translation/results/candidate-heldout-comet.json \
  Research/translation/results/prior-local-heldout-comet.json \
  PRIVATE_PATH/pairwise-judge-a.json \
  PRIVATE_PATH/pairwise-judge-b.json \
  Research/translation/results/candidate-critical-audit.json \
  Research/translation/results/non-apple-failure-path.json \
  Research/translation/results/candidate-heldout-parity.json \
  Research/translation/results/candidate-distribution.json \
  Research/translation/results/automated-promotion.json
```

`test_automated_claim_contract.py` and
`test_automated_promotion_contract.py` exercise both positive paths and
fail-closed cases including a wrong direction count, changed candidate hash,
judge-family collision, a judge critical flag, direct train/test contamination,
and a non-positive automated pairwise confidence bound.

## 1. Review and freeze references

Create the quota-exact incomplete authoring file outside every training path:

```sh
python3 scripts/translation/prepare_benchmark_authoring_template.py \
  Research/translation/benchmark/manifest.json \
  PRIVATE_PATH/heldout-draft.jsonl
```

The generated 800 rows are deliberately blank, non-claimable, and not marked
human-authored. Humans must fill every field listed in the companion manifest;
the review tools reject the untouched template.

Draft rows use the benchmark case shape plus `documentID`,
`sourceAuthorID`, one distinct `referenceAuthorIDs` entry per reference,
`sourceGeneratedByAI: false`, `referenceGeneratedByAI: false`,
`claimEligible: false`, and at least two independently authored references.
The response templates deliberately initialize the `human`,
`bilingualQualified`, `independent`, and `noAIAssistance` attestations to
`false`. Each reviewer and the adjudicator must personally change all four to
`true`; the tools fail closed rather than inferring qualifications from an ID.
Source author, reference authors, both reviewers, and the adjudicator must all
be distinct for each case. Author identities are omitted from blind packets but
bound into the finalized per-case SHA-256.

```sh
python3 scripts/translation/prepare_benchmark_reference_review.py \
  PRIVATE_PATH/heldout-draft.jsonl PRIVATE_PATH/reference-review \
  --reviewer REVIEWER_A --reviewer REVIEWER_B

# Reviewers complete their own response file without seeing the other packet.
python3 scripts/translation/prepare_benchmark_adjudication.py \
  PRIVATE_PATH/heldout-draft.jsonl \
  PRIVATE_PATH/reference-review/REVIEWER_A.responses.jsonl \
  PRIVATE_PATH/reference-review/REVIEWER_B.responses.jsonl \
  PRIVATE_PATH/adjudication --adjudicator ADJUDICATOR

# The adjudicator completes ADJUDICATOR.responses.jsonl.
python3 scripts/translation/finalize_benchmark_suite.py \
  PRIVATE_PATH/heldout-draft.jsonl \
  PRIVATE_PATH/reference-review/REVIEWER_A.responses.jsonl \
  PRIVATE_PATH/reference-review/REVIEWER_B.responses.jsonl \
  PRIVATE_PATH/adjudication/ADJUDICATOR.responses.jsonl \
  Research/translation/benchmark/heldout.jsonl \
  PRIVATE_PATH/heldout-review-records.jsonl \
  PRIVATE_PATH/rejected-heldout.jsonl
```

Any critical reviewer flag rejects that version of the case. Fix its source or
references and send the revision through a fresh two-reviewer cycle; the
adjudicator cannot silently edit reviewed text.

Run `validate_benchmark_suite.py` against every final train/dev JSONL before
generating the teacher batch and again before evaluation. Its validation JSON
binds the suite and review records by SHA-256 and is required by the promotion
evaluator. Validation requires exactly 400 cases in each direction and rejects
both text-level near matches and any shared `documentID` between the held-out
suite and a supplied training file.

## 2. Blindly compare engines

Run Apple and the candidate against the exact frozen suite on the same machine
and OS. Score both exact reports with the pinned Apache-2.0 COMET-22 model; the
runner uses float32 and averages scores across each case's references:

```sh
uv run --python 3.12 --with unbabel-comet==2.2.7 --with setuptools==80.9.0 \
  scripts/translation/score_comet.py \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/results/candidate-heldout.json \
  Research/translation/results/candidate-heldout-comet.json

uv run --python 3.12 --with unbabel-comet==2.2.7 --with setuptools==80.9.0 \
  scripts/translation/score_comet.py \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/results/apple-heldout.json \
  Research/translation/results/apple-heldout-comet.json
```

When two already-captured public reports use different deterministic samples,
do not compare unmatched aggregates. `align_translation_report_intersection.py`
authenticates both input files, aligns the complete immutable source/reference
intersection, assigns shared deterministic IDs, forces every row to
`claimEligible: false`, and fails unless both directions meet the declared
minimum. Its output is diagnostic only; it never substitutes for `heldout.jsonl`.

Then create two independently randomized human packets:

```sh
python3 scripts/translation/prepare_engine_comparison_packets.py \
  Research/translation/results/candidate-heldout.json \
  Research/translation/results/apple-heldout.json \
  PRIVATE_PATH/engine-comparison \
  --reviewer HUMAN_A --reviewer HUMAN_B
```

Do not expose `sealed-assignments.jsonl` until both response files are frozen.
Each reviewer scores adequacy 0–4, fluency 0–4, and terminology/register 0–2,
and independently flags critical meaning errors. References and engine identity
are omitted from the packet; reviewers must be bilingual.

Generate Mimi's executable fallback evidence and evaluate all gates:

```sh
.build/Mimi.app/Contents/MacOS/Mimi \
  --verify-translation-fallback \
  Research/translation/work/fallback-verification.json

.build/Mimi.app/Contents/MacOS/Mimi \
  --verify-translation-mlx-parity \
  Research/translation/results/candidate-heldout-parity.json \
  --model-root .build/Mimi.app/Contents/Resources/TranslationModels \
  --suite Research/translation/benchmark/heldout.jsonl \
  --python-report Research/translation/results/candidate-heldout.json

python3 scripts/translation/verify_translation_distribution.py \
  .build/dist/Mimi-macOS.zip \
  PRIVATE_PATH/candidate-4bit-pair \
  PRIVATE_PATH/pinned-mlx-0.30.6/mlx.metallib \
  Research/translation/results/candidate-distribution.json \
  --maximum-archive-bytes 150000000

# Routed MoE candidates must also authenticate the staged in-app notices:
#   --release-artifacts \
#     .build/Mimi.app/Contents/Resources/TranslationLicenses
# An explicitly blocked local-development archive additionally requires:
#   --allow-blocked-development
# and can only report status `passed-development-only`.

uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/evaluate_translation_promotion.py \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/benchmark/manifest.json \
  Research/translation/work/heldout-validation.json \
  Research/translation/results/candidate-heldout.json \
  Research/translation/results/apple-heldout.json \
  Research/translation/results/candidate-heldout-comet.json \
  Research/translation/results/apple-heldout-comet.json \
  PRIVATE_PATH/engine-comparison/sealed-assignments.jsonl \
  PRIVATE_PATH/engine-comparison/HUMAN_A.responses.jsonl \
  PRIVATE_PATH/engine-comparison/HUMAN_B.responses.jsonl \
  Research/translation/work/fallback-verification.json \
  Research/translation/results/candidate-heldout-parity.json \
  Research/translation/results/candidate-distribution.json \
  Research/translation/results/promotion.json
```

The evaluator exits zero only when both directions independently pass the
paired chrF++ and human-score lower bounds, critical-error, warm-p95, sample
count, memory, model-size, exact Swift/Python output parity, current combined
archive size/integrity, suite-integrity, hardware, and non-Apple fail-closed gates. The
combined signed universal archive—not only the model directory—must be at most
150,000,000 bytes. It also reports every domain separately. A rejected report
exits with status 2.
