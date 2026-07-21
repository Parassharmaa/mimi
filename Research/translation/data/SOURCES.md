# Data source and license register

Every training row must retain `source_id`, source URL, license, attribution,
acquisition date, transformation history, and a normalized-content hash. A
release manifest must list row counts by source and license.

The current release-contract builder authenticates the exact model lineage,
training manifests, dataset hashes/counts, and retained per-row notices. It
fails closed on missing third-party attribution and separately counts rows
marked `promotion_eligible: false` or `training_only: true`. That audit rejected
the earlier routed-v3 pack because its EN→JA interpolation contained 256 such
automated-consensus rows. The replacement human-only routed-v2 trace covers 10
dataset files and reports zero promotion-excluded rows, but distribution remains
blocked pending final license and macOS app-distribution review.

## Allowed for the first experiment

### Mimi-authored source seeds and translations

- License: CC0-1.0 dedication in this research directory.
- Use: domain-balanced prompts, synthetic source seeds, canary plumbing cases.
- Control: anything created for the protected benchmark is stored separately and
  is never sent to GPT, a baseline model, or the training-data builder.

### Mimi shipping UI parallel copy

- License: project-owned; extracted from paired English/Japanese literals in
  Mimi's Swift sources.
- Use: high-precision product/UI domain adaptation and reviewed-development
  examples in both directions.
- Control: `prepare_mimi_ui_parallel.py` accepts only literal bilingual pairs,
  rejects interpolation, duplicates, language/length failures, and protected-
  benchmark near matches, and groups the split by source file. The current
  corrected extraction has 76 unambiguous pairs: 63 train and 13 validation.
  Four one-to-many source mappings are rejected in both directions.
- Quality note: this is copy already shipping in Mimi, not a GPT translation.
  Preserve repository revision and source line in provenance and spot-review
  the extracted corpus before a release-training run.

### Tatoeba text pairs via ManyThings

- License: per-row CC-BY 2.0 France attribution supplied in the third TSV column.
- Source: <https://www.manythings.org/anki/> and
  <https://tatoeba.org/en/terms_of_use>.
- Use: conversational training ablation after reciprocal and ambiguity checks,
  protected-suite screening, and moderate agreement from both pinned students.
- Conditions: preserve both sentence IDs and contributors in the row provenance
  and release attribution file. Do not use audio or rows with a different
  per-contribution license. ElanMT documents Tatoeba in its pretraining, so its
  agreement score is a noise filter rather than independent quality evidence;
  this corpus is never eligible for Mimi's benchmark. The current deterministic
  pass scored 3,900 reciprocal candidates and retained 1,537 pairs. It rejected
  1,484 one-to-many mappings before scoring and 2,363 low-agreement pairs.

### GPT-5.6 candidate translations

- Source: OpenAI Responses API, `gpt-5.6-sol`, `store: false`.
- Use: candidate translations only, never benchmark references.
- Conditions: record model ID returned by the API, system fingerprint when
  present, prompt hash, request ID/custom ID, source-row license, and timestamp.
  Confirm the account's then-current OpenAI terms and output rights before a
  release. API output still requires deterministic filtering and independent
  bilingual review; model self-review is not independent acceptance.
- Official implementation references:
  <https://developers.openai.com/api/docs/guides/latest-model.md>,
  <https://developers.openai.com/api/docs/guides/structured-outputs>,
  <https://developers.openai.com/api/docs/guides/batch>, and
  <https://developers.openai.com/api/docs/guides/your-data>.

### Kyoto Free Translation Task (KFTT)

- License: CC-BY-SA 3.0.
- Source: <https://www.phontron.com/kftt/>; pinned archive SHA-256
  `fcfcaa670d6d59aa691b0e909c0d7c393852dd2fb1d6310fda9b3282dc6d1638`.
- Strength: Japanese Wikipedia sentences translated and checked by professional
  translators, with reproducible train/tune/test splits.
- Use: primary high-quality student-training source in both directions.
- Current distillation use: deterministically sample a pool, translate it with
  the exact 4-bit student, and prioritize weak-but-aligned rows for teacher
  candidates and bilingual review. The professional reference and student
  hypothesis stay local; only the licensed source is exposed to GPT.
- Conditions: preserve the NICT attribution verbatim and distribute derivative
  weights under a release policy compatible with CC-BY-SA 3.0. Shipping remains
  gated on documenting that policy; the corpus is commercially usable but not
  permissively relicensable.

### NICT Asian Language Treebank parallel corpus

- License: CC BY 4.0 for NICT's translations; the English Wikinews source text
  is CC BY 2.5.
- Source: <https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/>; pinned
  archive SHA-256
  `05f7b31b517d4c4e074bb7fb57277758c0e3e15d1ad9cfc5727e9bce79b07bbd`.
- Strength: 20,106 aligned English/Japanese Wikinews rows translated by NICT.
  After empty/alignment, language, length, duplicate, and provisional protected-
  suite filtering, Mimi retains 16,011 short unique pairs.
- Use: capped human-parallel replay in both directions. The dataset builder
  defaults to at most 2,000 train and 200 validation rows per direction so news
  text cannot swamp the reviewed live-speech domain data.
- Conditions: preserve NICT and Wikinews attribution, source URLs, sentence IDs,
  archive/member hashes, and the requested Riza et al. citation. Re-screen
  against the final held-out set before training.

### NICT English BTEC 20K source utterances

- License: CC BY 4.0.
- Source: <https://att-astrec.nict.go.jp/en/product/>; pinned archive SHA-256
  `9c0ffaf912cb02eacdff0f3882a2bbcb53a7996af8b2299b6a13b9745c4cb955`.
- Strength: short Basic Travel Expression Corpus utterances matching spoken,
  service, navigation, and travel use. The prepared rehearsal has 22,012 unique
  clean English segments and hash-samples 300.
- Use: source-only EN→JA GPT teacher seeds. This release contains no Japanese
  side, so it is never labeled parallel gold; a target enters training only
after deterministic checks and either bilingual selection or the strict
promotion-ineligible automated-consensus lane documented in the root README.

### Japanese Law Translation Database System

- License: Public Data License 1.0 (PDL 1.0), which explicitly permits
  commercial use, copying, public transmission, translation, and modification
  with source attribution. PDL 1.0 declares compatibility with CC BY 4.0.
- Official terms: <https://www.japaneselawtranslation.go.jp/en/index/terms> and
  <https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0>.
- Source: finalized sentence-aligned TMX exports from
  <https://www.japaneselawtranslation.go.jp/en/laws>. Tentative translations
  are excluded at search time.
- Acquisition: 1,001 unique finalized law records were discovered on 2026-07-18.
  The resumable crawl authenticated 951 valid TMX files totaling 242,830,013
  bytes and 395,656 translation units; 18 records exposed no TMX and 32
  malformed XML exports were hash-recorded and quarantined. The completed
  inventory SHA-256 is
  `8ff525fa6e095477918b44772d47ca534710e0f5f3023fc95459dd115b59f730`.
- Preparation: strict language, 240/160-character, duplicate, and protected-
  suite filters retain 229,139 unique pairs. Law-grouped splitting emits
  204,407 train, 7,032 validation, and 17,700 test pairs, represented in both
  directions. The manifest SHA-256 is
  `0c8ee33203c4efd3af7711dfd716d3a889a37515b74a57a02a8b53806a2234ce`.
- Use: cap the legal component to 10,000 unique pairs per direction in the first
  full-depth adaptation. This is precise formal-domain data, not conversational
  speech, and must not dominate the mixture. The test documents contribute only
  to the non-claimable public stress suite.
- Exact-memory development use: the train split alone yields 6,179 short
  repeated-source entries after requiring at least two distinct laws, discarding
  conflicting within-document observations, selecting an observed human target
  medoid, and enforcing exact critical-token equality. The runtime SHA-256 is
  `fe7a81f12ebbf1bfd140f6400c37980ae6026013f8b9dc92cf4a6080b78b8787`;
  its full compressed evidence audit SHA-256 is
  `b2e7af596be54c5e0c920405bed2b05a734363c9d52ea3aa8cf80215cc6054d5`.
  Validation and public test references never enter the memory. Retrieval
  results are reported separately from neural generalization.
- Conditions: ship the Ministry of Justice source citation, PDL link, and a
  statement that Mimi filtered, normalized, and converted the source content;
  never imply government endorsement. Preserve law ID, TMX unit ID, source URL,
  normalized hashes, TMX hash, status, and acquisition inventory.
- Promotion status: the memory's source rows remain explicitly
  `training_only: true` and `promotion_eligible: false`. The development release
  contract preserves this as a blocker rather than treating the PDL license as
  automatic app-integration approval.

### Local diversity teachers and automated filters

The July local-reference experiment freezes 1,785 professional KFTT pairs
(892 EN→JA and 893 JA→EN) after excluding every exact current base-training
source, 141 already-used student sources, 300 missing references, both protected
benchmark suites, and near protected 5-gram matches. The frozen suite is
`Research/translation/work/local-reference-teacher-v2/suite.jsonl`, SHA-256
`b87f3ea53699a35ba816723bdfcb05e907ca31676e0d1576726251dbf1c2eca8`.
All rows retain KFTT's CC-BY-SA-3.0 attribution. They are training-only and
never become promotion evidence.

The exact preferred-v3 students generate a frozen baseline on those sources.
Qwen then receives only source text and direction—never the licensed reference
or student hypothesis—and returns a final translation with no requested or
stored reasoning. Independent chrF++, pinned COMET-22, numbers/URLs/markup,
script, repetition, length, source-copy, and protected-suite gates retain 67
rows (27 EN→JA, 40 JA→EN) at SHA-256
`2f0724bffb2609c239e8df56940cedcfaaaa7bdaa68f1d13c4c8d21a2a347e5e`.
The Qwen targets are provisional sequence-distillation data only. Matched
controls train on the same sources with the professional KFTT reference, which
is necessary to distinguish teacher value from the value of adding a novel
licensed pair.

The balanced follow-up combines only reference-backed train rows from KFTT,
NICT ALT, and Tatoeba/ManyThings. Its full-corpus inventory is
`Research/translation/work/balanced-reference-teacher-v3/inventory.json`,
SHA-256 `0ec59a9371a929eedcb87d02492c4b7d38c99de2bd1a17926e3afb2bab8c8a25`.
After removing active student data, the prior teacher suite, ambiguous Tatoeba
IDs, normalized duplicates, and protected exact/5-gram matches, the exact
preferred-v3 student selects 400 uncertainty/diversity examples from a
deterministic 600-row pool in each corpus and direction. The 2,400 selected
seed rows have SHA-256
`0592d1dfc3ef71a81a3bbed8dbffdc42d7231b457c8243fd8d6604d06b028046`;
the finalized reference-hidden suite has SHA-256
`98da175c5a7d937afd280fec0db23757702c74dc8dd64f43e1eb3b2cd48d1198`.
It retains 800 rows under each corpus license and remains training-only.

Qwen3-8B generated a final translation for every frozen source without seeing
the licensed reference, student output, or hidden reasoning. Strict chrF++,
COMET-22, structure, and positive-student-delta gates found 281 potential
targets, but the professional-Wikipedia cells retained only 9 EN→JA and 7
JA→EN against a predeclared minimum of ten. The entire synthetic round is
rejected and no accepted-target JSONL exists. Its authenticated floor-failure
report is
`Research/translation/work/local-qwen-balanced-reference-filtered-v3.jsonl.floor-failure.json`,
SHA-256
`61a94ade4675cf9ebad7096dc570b3f57e83c8c9bf591ce48d7559adcc5d0e17`.

The matched control uses the original licensed KFTT, ALT, and Tatoeba references
for the same 2,400 selected sources. Those rows retain their original
CC-BY-SA-3.0, CC-BY-4.0, and CC-BY-2.0-FR provenance and attribution in the
derived dataset manifests. The resulting model artifacts remain explicitly
distribution-blocked until the combined share-alike and attribution release
policy is reviewed; the experiment does not turn the rows into product
evaluation evidence.

- CAT-Translate-0.8B: MIT-licensed candidate teacher, pinned through
  `hotchpotch/CAT-Translate-0.8b-mlx-q4` revision
  `84cbdd97cf628fa98fcd5a757d2599ebee765cd7`. Research cache only; never bundled.
- HPLT v2 English↔Japanese: CC-BY-4.0 independent forward/reverse filter. Its
  hypotheses are not automatically selected as targets and its weights are not
  bundled.
- `cross-encoder/nli-deberta-v3-small`: Apache-2.0 English roundtrip filter at
  revision `84ccdcb62589067b29b930cff8e362e75ba0dd15`.
- `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`: MIT-licensed optional
  Japanese/English roundtrip filter pinned at
  `acf08db83390e23428c560cb578a865b39196993` for the symmetric next round.
- `mlx-community/Qwen3-8B-4bit`: Apache-2.0 final bilingual training-data judge
  at revision `545dc4251c05440727734bcd94334791f6ab0192`. The 4.62 GB snapshot and
  prompt cache are research-only and must never enter the app bundle.

All automated selections retain the source license/provenance and are marked
`promotion_eligible: false`. The models return translations or compact
judgments only; no chain-of-thought is requested, persisted, or used as a
student target.

The current release-clean routed candidate intentionally excludes every such
automated target. Its two generalists and two experts trace only to authenticated
human-parallel/project-owned rows; the generated release contract reports zero
promotion-excluded rows. This lineage fact does not by itself approve shipping:
the effective CC BY/CC BY-SA/Japan Public Data License obligations and final app
distribution review still apply.

### ElanMT-BT student checkpoints

- License: CC-BY-SA 4.0.
- Source: <https://huggingface.co/Mitsua/elan-mt-bt-en-ja> revision
  `02c48e7031386cd2d41974b0ff1aaf52f010c5fa` and
  <https://huggingface.co/Mitsua/elan-mt-bt-ja-en> revision
  `539f80eb05306e27a166b45e4264c7fa2eb4de97`.
- Strength: compact Marian encoder-decoder models already trained on KFTT and
  other openly licensed corpora, with documented exclusion of web-crawled and
  unauthorized copyrighted training text.
- Conditions: retain ELAN MITSUA Project / Abstract Engine attribution, model
  card and license links, source revisions and checksums, and distribute the
  quantized/adapted weights under a compatible share-alike policy.

### COMET-22 evaluation model

- License: Apache-2.0; evaluation only, never bundled into Mimi or used as a
  translation teacher.
- Source: `Unbabel/wmt22-comet-da` revision
  `371e9839ca4e213dde891b066cf3080f75ec7e72` with
  `unbabel-comet==2.2.7` and its required legacy-compatibility dependency
  `setuptools==80.9.0`.
- Use: reference-based learned-metric comparison of the frozen candidate and
  Apple reports, in float32 with per-case mean aggregation across references.
  The report records the model revision, package version, precision, signature,
  suite hash, and engine-report hash.
- Exclusion: XCOMET-XL/XXL model weights are CC BY-NC-SA 4.0 and are not used in
  this commercial-product evaluation lane.

### Automated claim-suite source draft

- License: project-owned; source text rendered deterministically from original
  Mimi product-scenario templates without a model call.
- Artifact: `Research/translation/benchmark/automated-claim-v1.sources.jsonl`,
  800 unique source-only rows, SHA-256
  `f039ce456c55f051e8bbcc13ed9bc8270a722819308e008b39da7f30327ec16c`.
- Balance: exactly 400 EN→JA and 400 JA→EN, each with 120 meeting/live-speech,
  80 everyday, 60 macOS/UI, 60 number/date/entity, 60
  politeness/ambiguity/omission, and 20 code-switching cases.
- Contamination evidence: 1,358,264 texts from 15 authenticated current-release
  training, validation, exact-memory, canary, and public-development inputs
  passed exact and NFKC-normalized character-5-gram screening at maximum
  Jaccard 0.65.
- Status: source freeze only, `claimEligible: false`. Independent references,
  two judge families, a complete exposure manifest, semantic-neighbor scan,
  and structural audit are mandatory before validation. If GPT-5.6 trains the
  student, GPT-5.6 may not generate these benchmark references.

## Conditional / legal-review queue

### CAT-Translate-Dataset

- Dataset: `cyberagent/CAT-Translate-Dataset`, pinned repository revision
  `f72dd5d8e1598f09b007f54d6847c7d14e75f6aa`.
- Public terms: ODC-BY-1.0 plus Common Crawl Terms of Use. Access is gated on
  accepting conditions and sharing Hugging Face contact information; the
  published files total 7.11 GB.
- Contents: synthetic English/Japanese pairs generated mostly by
  Apache-2.0 `gpt-oss-20b`, with `gpt-oss-120b` for harder quality domains.
  The public card identifies FineWeb/FineWeb-Edu/FineWeb-2 Japanese,
  Laboro-ParaCorpus, HPLT, and CCMatrix only at corpus level. It says the
  `corpus` subset was not used to train CAT; `fineweb` and `abstracts` were.
- Audit result: do not import the released rows into Mimi training. Every
  unauthenticated dataset-server schema/preview/parquet endpoint returned 401,
  and the public metadata does not establish per-row source URL, source
  license, generator revision, or filtering lineage. Accepting the gate is an
  external account/terms action and would not repair that release-provenance
  gap.
- Reusable finding: adopt the documented diversity-first then quality-first
  curriculum, MinHash/length/language filters, and explicit lexical, format,
  and length anti-hallucination objectives using Mimi's own admissible rows.
  CAT-Translate-0.8B may remain a local diversity teacher; the corpus may not
  enter distributable weights without a complete row-level lineage audit.

### Hy-MT2-1.8B 1.25-bit

- Upstream: `tencent/Hy-MT2-1.8B`, Apache-2.0, with one physical model for
  both directions. The official Sherry/STQ 1.25-bit GGUF repository revision
  is `9df5c824a00a744fb0512a29c640466f4d97dfb0`; its 461,860,800-byte file
  hashes to
  `cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93`.
- Upstream packaging defect: the official GGUF still uses an earlier STQ type
  layout and fails closed in the referenced open llama.cpp kernel branch at
  tensor `blk.0.attn_k_norm.weight`. Do not ship that file/runtime pair.
- MLX research control: `kuotient/Hy-MT2-1.8B-1.25Bit-MLX` revision
  `03d1df683157fde0a4ec80636e749867d0c13a5e`, also labeled Apache-2.0,
  losslessly preserves the official sparse-ternary linear tensors and adds a
  custom Metal kernel. The authenticated local snapshot is 464,192,044 bytes;
  the benchmark counts 464,213,662 bytes after generated local metadata.
- Status: rejected before Swift/runtime integration. Canary chrF++ looked
  promising at 35.62/60.70, but the matched 800-case stress run fell to
  22.64/44.09 versus the incumbent's 33.23/54.00, with 127 exact
  critical-token failures. The custom Python/Metal runtime is also not
  automatically MLX Swift compatible.
  The Hy-MT2 report describes roughly one trillion multilingual monolingual and
  parallel mid-training tokens plus family-specific translation/business/
  instruction data, but does not enumerate the source corpora or their licenses.
  Apache-2.0 weight permission therefore does not satisfy Mimi's stricter
  training-data lineage gate. Do not ship or use it as a release teacher unless
  Tencent publishes an auditable commercial-rights inventory.

### LMT-60-0.6B

- Model: `NiuTrans/LMT-60-0.6B`, Apache-2.0, pinned revision
  `dd189845cdc73346cef33c7a94f4b8bd8efdd4eb`. The authenticated BF16 weight is
  1,503,300,328 bytes with SHA-256
  `c48c3b8d7b04d3c6e56452fa51ddefc03921043b0f70641cb80c5d3ac71b73e6`.
- MLX research control: exact `mlx==0.30.6` / `mlx-lm==0.30.6` affine
  4-bit/group-64 conversion, 346,929,488 bytes total. Its 335,450,548-byte
  weight hashes to
  `8a2437b9f22eb7217be5f51bf5e10a7c36d5cc610f7f7f078740dd3e670af65f`.
- Technical status: rejected. Canary chrF++ is 31.38/54.15, while the matched
  800-case stress run reaches only 17.92/40.42 versus 33.23/54.00 for the
  incumbent and produces 141 exact critical-token failures. Do not spend
  release effort on beam search, COMET, Swift parity, adapters, or integration.
- Training-data status: the report describes 90B mixed monolingual/bilingual
  continued-pretraining tokens and about 567k SFT pairs. Named corpus families
  include CulturaX, MADLAD, FineWeb2, OPUS, C4, OSCAR, WanJuanSiLu, in-house
  material, and billions of synthetic pairs. It does not publish row-level
  source/license/generator lineage. The linked `NiuTrans/LMT-60-sft-data`
  repository at revision `47914a5aac70e3e930aa8e7e8dae2969219319c3` has no
  license tag. Its 4,371,634-byte EN–JA shard hashes to
  `5dd641719d2ec2f8727452e744a65226ac238e3384df2add79ff04de506c89fa`
  and contains 13,169 pairs entirely labeled FLORES-200, NTREX-128, IWSLT
  2017/2022, or WMT news 2020/2021/2022. Do not admit these evaluation-family
  rows into Mimi training, treat public-benchmark scores as independent, ship
  the weights, or use them as a release teacher without a complete rights and
  split-contamination inventory.
- Sources: <https://huggingface.co/NiuTrans/LMT-60-0.6B>,
  <https://huggingface.co/datasets/NiuTrans/LMT-60-sft-data>, and
  <https://arxiv.org/abs/2511.07003>.

### QuickMT

- Models: `quickmt/quickmt-en-ja` revision
  `c09e98b8438a239a8210060114cea19c426c0559` and
  `quickmt/quickmt-ja-en` revision
  `e9ae594ff322d95254d730867c6166b25fd2c704`; weight cards declare
  CC-BY-4.0.
- Technical result: official CTranslate2 int8 exports total 813,661,710 bytes.
  The 8-encoder/2-decoder EN→JA and 12-encoder/2-decoder JA→EN pair scores
  26.54/57.98 chrF++ on Mimi's canary at about 0.076/0.077-second warm p95,
  but peaks at 1,776,779,264 process-resident bytes. EN→JA fails the compact
  incumbent gate; retain the encoder-heavy/shallow-decoder shape as a training
  blueprint only.
- Training-data exclusion: `quickmt/quickmt-train.ja-en` revision
  `d4c8cfe34aebfe82dd15ea69dd384d30b8d57f98` declares 63,285,158 rows but no
  dataset license. Its card mixes WMT/newstest, KFTT train/test/dev/tune,
  JParaCrawl, JESC, OpenSubtitles, NLLB, CCAligned, CCMatrix, WikiMatrix,
  Tatoeba, and Bible sources, plus MADLAD and NewsCrawl backtranslations,
  without row-level source license or generator identity. Do not train on it,
  use it as a teacher, treat its public benchmark scores as independent, or
  ship derivative weights without a complete rights/contamination inventory.
- Sources: <https://huggingface.co/quickmt/quickmt-en-ja>,
  <https://huggingface.co/quickmt/quickmt-ja-en>, and
  <https://huggingface.co/datasets/quickmt/quickmt-train.ja-en>.

### Patchouli JA–EN / QuickMT JA→EN v2

- Dataset revision: `Yokii2/patchouli-jaen` at
  `e304a3b0b860745084976b2a50eef7b622742932`; the card reports 61,977,073
  rows and says generation is only 13.52% complete.
- Source/teacher lineage: Japanese source strings come from
  `range3/cc100-ja`, whose card declares the license unknown. Nearly all English
  targets are attributed only to “Mistral Small 4”; no exact teacher revision,
  prompt, filtering, output-rights analysis, or reproducible row selection is
  published. The card mentions chain-of-thought generation, although only final
  strings appear in the released schema; Mimi does not import either.
- Derived model: `Yokii2/quickmt-ja-en-v2` revision
  `3eeaf22dcd0a0ee7331aa4fa824d0cf33e6ea088` says it fine-tuned on about 31M
  rows but does not bind their IDs or hashes. Its CC-BY-4.0 repository label
  does not repair the underlying data-lineage gaps.
- Decision: exclude the dataset, outputs, and weights from distributable Mimi
  training and teaching. A research-only CPU benchmark would not change that
  decision and is lower value than the already rejected QuickMT control.
- Sources: <https://huggingface.co/datasets/Yokii2/patchouli-jaen>,
  <https://huggingface.co/datasets/range3/cc100-ja>, and
  <https://huggingface.co/Yokii2/quickmt-ja-en-v2>.

### LFM2-350M-ENJP-MT

- Model: `LiquidAI/LFM2-350M-ENJP-MT`, pinned revision
  `80367784d525777ad7565b24534ba5810eeac59f`.
- Technical fit: one bidirectional model with a published 381.6 MB MLX 8-bit
  conversion and direct LFM2 support in MLX Swift LM.
- Exclusion: LFM Open License 1.0 does not license commercial use by a legal
  entity at or above USD 10M annual revenue. Keep outside Mimi unless a separate
  commercial agreement removes that restriction.
- Source: <https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT>.

### Translate-15L

- Model: `WhirlwindAI/Translate-15L`, Apache-2.0, pinned revision
  `ce860c33668440b031e30f50cc31377c6b6fac59`.
- Authenticated snapshot: 244,616,918 bytes locally; 60.5M-parameter T5-small.
- Result: rejected before MLX porting. Greedy canary scores 0.00/1.37 chrF++;
  documented beam 4 scores 0.00/4.11 and produces empty or repeated-punctuation
  outputs. Beam score SHA-256:
  `813801ccd9fd7f93e7db888e509621dc8085af5cc78c67c74d7503e76ca0d0db`.
- Source: <https://huggingface.co/WhirlwindAI/Translate-15L>.

### ParaNatCom

- License: CC BY 4.0; pinned archive SHA-256
  `73d1510f7f04872605a57269615b8db4cc65d07ec1be524d6ebc1f7f94a35785`.
- Source: <https://www2.nict.go.jp/astrec-att/member/mutiyama/paranatcom/index.html>.
- Strength: Nature Communications abstracts translated by three professional
  agencies.
- Hold for a later terminology ablation: the release is abstract-level rather
  than sentence-aligned and is scientific, so it must not be naively split or
  allowed to dilute the live-caption corpus.

### FLORES-200 / open FLORES

- Use: external metric smoke only.
- Hold: common public benchmark data is likely to overlap pretrained-model
  training or model selection. It cannot be Mimi's held-out promotion evidence.
  Follow the current upstream dataset and license rather than an old mirror.

## Excluded from distributable weights

### JParaCrawl

The publisher restricts the corpus, derived data, and trained translators to
research and asks commercial users to contact NTT. This conflicts with shipping
Mimi, so none of it may enter the train/dev corpus.

Source: <https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/>.

### JESC / OpenSubtitles-derived corpora

The corpus is useful conversationally, but the checked source did not provide a
clear, row-level commercial redistribution chain for the underlying subtitles.
Exclude it unless counsel or the publisher supplies a suitable license record.

Source: <https://nlp.stanford.edu/projects/jesc/>.

### Business Scene Dialogue (BSD)

The professionally constructed business-conversation corpus is attractive for
this task, but its CC BY-NC-SA license is incompatible with shipping Mimi. Do
not train a distributable student on it.

Source: <https://github.com/tsuruoka-lab/BSD>.

### NLLB-200 weights and outputs as a teacher

The 600M checkpoint is CC-BY-NC-4.0 and its card says it is a research model not
released for production deployment. Do not ship, fine-tune, or use it to create
the commercial training set.
