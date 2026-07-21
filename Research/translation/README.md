# Mimi local translation research

This directory is the reproducible research lane for Mimi's local
English↔Japanese translator. Apple Translation is a diagnostic comparison, not
the intended product backend: the final system must use a validated local model
as primary and define a non-Apple failure path. The current worktree temporarily
preserves the existing Apple behavior while no local candidate has passed the
absolute held-out accuracy and critical-error gates.

The preferred model target remains 150,000,000 bytes or less, but accuracy may
justify a larger dense, sparse-expert, or cascaded bundle up to a hard
500,000,000-byte ceiling. Real-time latency, distributable licensing, and
Apple-Silicon/MLX execution remain hard requirements.

The current accuracy-first training decision is documented in
[`strategy-lexically-constrained-distillation-2026-07-21.md`](strategy-lexically-constrained-distillation-2026-07-21.md):
final-output-only constrained sequence distillation into Marian first, with a
cheap frozen M2M-100 418M single-model feasibility gate before any MLX port.

The reviewer-free promotion route is now preregistered separately from the
legacy human contract. `benchmark/automated-claim-v1.manifest.json` fixes 400
new sealed product-domain cases per direction, strong absolute chrF++/COMET
floors, positive paired confidence bounds against the best prior local model,
two distinct automated judge families, complete exposure and semantic
contamination evidence, a zero-critical-error union veto, 250 ms warm p95, and
the 500 MB hard cap. Its validator and evaluator have fail-closed contract
tests. The source side is now frozen at 800 unique project-owned cases (400 per
direction), SHA-256
`f039ce456c55f051e8bbcc13ed9bc8270a722819308e008b39da7f30327ec16c`.
An exact and normalized 5-gram scan passed against 1,358,264 exposed texts in
the current release lineage and public development suites. The stronger
schema-v2 exposure manifest now binds 17 text assets, 14 evidence assets, and
1,411,076 raw strings while explicitly refusing to claim exact access to opaque
upstream pretraining rows. An exhaustive pinned multilingual MiniLM scan reduced
those to 599,317 normalized-unique strings and found zero source cases above the
0.82 threshold (maximum 0.798585). This is still not a quality claim: references
and independent judge evidence are pending. The exact sealed generator batch
was submitted on 2026-07-21 and is being collected only into the git-ignored
private work area.

The reviewer-free reference generator is sealed as 800 `gpt-5.6-sol`
Responses Batch requests using strict Structured Outputs, `store: false`,
`reasoning.effort: none`, and a 1,024-token final-output allowance. The v2
SHA-256 is
`d6f85b9af0f10767067c66d1332a6334233044ee28674694612ea2614db6822a`.
Two later blinded review passes are pinned to `gpt-4o-2024-08-06` and
`gpt-4.1-2025-04-14`, distinct from the generator family and from each other.
The offline lane now also validates and hash-binds raw generator responses,
builds independently shuffled blinded requests for both judges, validates their
raw responses, selects exactly two unanimously perfect references by a
deterministic diversity tie-break, and audits numeric/ID/URL/placeholder/markup/
code-switch preservation. Named-entity, negation, and omission acceptance is
never inferred from brittle bilingual regexes; it remains bound to both judges'
perfect, error-free assessments. An end-to-end adversarial fixture rejects
visible reasoning summaries and any one-judge veto. The exact generator batch
is now submitted; promotion remains blocked until it completes, every response
passes the collector, both independently shuffled judge batches complete, and
the two-judge unanimity contract assembles all 800 cases without a veto. An
earlier high-reasoning submission is quarantined: 10 response bodies exhausted
their token limit and 300 contained encrypted reasoning items, so the
fail-closed collector admitted zero candidates from it.

The evidence-backed next strategy is documented in
[`literature-review-2026-07.md`](literature-review-2026-07.md): diverse
quality-aware sequence distillation, uncertainty-plus-diversity source
selection, frozen-base regularization, a mixed-domain curriculum, and
checkpoint averaging, with preference optimization only after a supervised win.

## Current decision

The preferred-v3 student bundle is two direction-selected ElanMT-BT Marian
students exposed through one bidirectional Mimi interface. EN→JA is a 15%
linear blend of the strict local-teacher checkpoint into the conversational
control; JA→EN remains the macro-domain-selected, three-checkpoint averaged
licensed-unified regularized student from preferred-v2. Both checkpoints are
CC-BY-SA-4.0 and their
model cards document training exclusively from openly licensed corpora,
including professionally translated KFTT. The direct MLX port preserves the
full 32,001-token vocabulary and matches the verified PyTorch architecture.
The preferred-v3 4-bit bundle is 73,402,252 bytes (70.0 MiB), comfortably below
the preferred 150 MB target and hard 500 MB ceiling.

Preferred-v3 preserves all preferred-v2 canary hypotheses and scores 31.31
chrF++ English to Japanese and 56.52 Japanese to English, versus Apple's 37.91
and 61.79 on that non-claimable 12-case set. On 2,400 non-claimable public-v2
cases it scores 30.17/55.95; EN→JA conversation improves over v2 by +0.17 mean
sentence chrF++ (95% paired interval +0.04…+0.35), while the all-domain change
is inconclusive (+0.07, -0.02…+0.15). Cached Python p95 is 0.051/0.059 seconds,
peak RSS is 212,746,240 bytes, and exact Swift/Python token parity passes
2,400/2,400. Against Apple on the identical 2,400 public sources, chrF++ favors
the local pair overall but pinned COMET-22 rejects that apparent win: EN→JA is
lower by -0.00537 (95% paired interval -0.01064…-0.00016) and JA→EN is
inconclusive (+0.00007, -0.00481…+0.00508). News loses significantly on both
metrics in both directions. Quality remains the only failing gate. It is not
promoted.

The public stress surface is now expanded from 2,400 to 2,800 cases (1,400 per
direction) by adding 400 finalized Japanese Law Translation cases. It remains
public development evidence, not the independently sourced claim suite. On this public-v3
suite preferred-v3 scores 28.30/51.86 chrF++ for EN→JA/JA→EN. A release-clean
full-depth continuation over 123,050 screened rows improves EN→JA by +0.639
mean sentence chrF++ (95% paired bootstrap +0.257…+1.026), including +2.142
on legal and +1.363 on Wikipedia, but regresses conversation and the protected
canary after 4-bit conversion. It is therefore rejected as a global
replacement.

A release-lineage audit found that preferred-v3's EN→JA interpolation includes
256 rows explicitly marked `promotion_eligible: false` and `training_only:
true`. The hash-bound v3 release contract therefore now reports
`blocked-promotion-ineligible-training-data`; it is a development artifact, not
a distributable candidate. Its mean sentence chrF++ gain over the authenticated
human-only parent is only +0.062 on 1,400 public-v3 EN→JA cases, with a 95%
paired interval crossing zero (-0.013…+0.146). The clear +0.191 conversation
gain does not justify carrying promotion-excluded lineage, especially because
the legal slice regresses significantly.

Recalibrating the frozen source-only ridge router against that human-only
generalist makes the formal expert slightly stronger. On the same document/law-
grouped development test split it routes 160/386 cases and improves mean
sentence chrF++ by +0.432 (95% paired bootstrap +0.153…+0.733). It still routes
no conversation cases; the product canary routes two non-negative cases for a
+0.267 mean gain. This is promising development evidence only: the expert is
not wired into Mimi and cannot advance until the private held-out, critical-
error, Swift-parity, routed latency/RSS, and exact-distribution gates all pass.

The matched high-dose JA→EN legal specialist is stronger but likewise unsafe
as a global replacement. On public-v3 it gains +1.218 mean sentence chrF++
(95% paired bootstrap +0.652…+1.774), led by +9.867 on legal
(+8.128…+11.718), while conversation and Wikipedia remain inconclusive and the
canary meeting slice regresses. Its legal-domain router isolates the useful
region: on a separate law/document-grouped test split it routes 35/359 cases
(34 legal, one Wikipedia), routes no conversation, news, or canary cases, and
gains +0.918 mean sentence chrF++ (+0.427…+1.569).

The current development candidate is
`elanmt-release-clean-human-routed-moe-v3-memory-v2-mlx-4bit-shared-tokenizer-pack`: the
human-only EN→JA generalist, the unchanged authenticated JA→EN generalist, two
human-data experts, and a compact exact-source memory built only from repeated
finalized Japanese Law Translation pairs. Its four engine tokenizers were
byte-identical, so pack format v2 authenticates one root copy while preserving
each direction-specific tokenizer config. The pack is 141,492,266 bytes. Its
hash-bound release evidence is 892,561 bytes, including the complete compressed
memory audit, for 142,384,827 bytes combined—7,615,173 bytes below the preferred
150,000,000-byte ceiling. The pack manifest SHA-256 is
`deda4fe0d6c9ca3fd069ca99f7c45a42b5bab1fcfda3ff861cb3a0bdee40c2ee`;
the release-contract SHA-256 is
`8c4baec93d53f499914201f3a42179eb7a6071490e86e5cda2fd952946db4d45`.
The exact pack size is 141,492,266 bytes; the 3,702-byte increase over v2 is
authenticated conversion/lineage metadata only. Re-conversion under MLX 0.30.6
reproduced all four weight hashes, and all twelve non-manifest payload files are
byte-identical. The contract now computes `provenanceComplete=true`.
That does not make it promotable: recursive lineage exposes 256 provisional
local-teacher consensus ancestor rows marked training-only and
promotion-ineligible, and the exact memory carries the same policy block. The
sealed 400+400 reference evaluation and final license/app reviews are also
pending.

The memory-free lineage-clean successor is
`elanmt-release-clean-human-only-routed-moe-v4-mlx-4bit-shared-tokenizer-pack`.
It replaces the formal EN→JA expert with a matched human-only retrain and is
140,875,806 bytes. On an otherwise identical neural-only 2,800-case control it
is quality-neutral overall (+0.0083 EN→JA, exactly 0.0 JA→EN) and improves the
EN→JA legal slice by +0.0831 with a positive paired lower bound. It also creates
one additional strict critical-token failure, lowering runtime acceptance from
2,296 to 2,295. The candidate therefore remains research-only. Its manifest
SHA-256 is
`8fd2dd3ecf39ab86ff535e8a2f77576390898fb92a7600a923efd90dbed8704e`;
the fail-closed release contract is
`b56eb15418d8661626fdfc428611783671eabf2c24ba88456e2d23165fa0af0b`.

The v4 distribution review now records the exact rights layers and macOS
channel decision without granting release authorization. Direct signed and
notarized distribution is conditionally feasible only after the model files
are separately scoped under CC BY-SA 4.0, the complete offline license and
attribution payload is bundled, the completed portable inventory is used, and
every quality/policy gate passes. Mac App Store
distribution remains blocked pending qualified review or written permission
because the exact EULA and technological-measure interaction with ShareAlike
has not been resolved. The human-readable review SHA-256 is
`0c5f390d55e935b266f7110ac2fd2927d74c2d93966db00cad34c1ce126e8ca5`;
the machine-readable review SHA-256 is
`6b838a6479b4d9022124f39d258fe0203d17fc825bc617aece9ffbe06451af25`.
The resulting path-clean clone is 140,875,791 bytes; its pack manifest,
portable release contract, and inventory hashes are
`71f330302559a0948c1b35f6def2d6107b4b57728cf53e1e99e0279141d84e79`,
`3d8ccfdfb95d2365de21a0912bd2c370c8ebf6061c903d758798720a8dc0a8f6`,
and
`fb62b633e56c5c2fc5aa983e71f8475db6da2588c99479fae6031dcd5b8f01e7`.
Every weight, tokenizer, router, and tokenizer-configuration payload is
unchanged; only metadata manifests differ. Its staged authorization remains
`blocked-development-only`.
An exact MLX replay matches all 12/12 canary routes, hypotheses, output token
IDs, safety decisions, and failure reasons, with comparison SHA-256
`885502fa4d31b595ccc76b74993ff24ad3e4d40a88d6f5965c6172c4544d1b58`.
The native Swift loader validates the portable pack and generalist/expert cold
smokes pass in both directions. The original two canary critical-token failures
also remain, so this proves metadata equivalence rather than quality.

The successor
`elanmt-release-clean-human-only-routed-moe-v4-portable-licensed-audit-v3`
closes the offline-public-license-file blocker without changing one model byte
or granting release. It freezes five official Creative Commons legal codes,
the official Japanese PDL 1.0 PDF, its English reference page, and the Japanese
Law Translation terms page. The eight source documents total 413,215 bytes;
the hash-bound manifest brings the offline license directory to 418,162 bytes.
The contract, portable inventory, and license-manifest SHA-256 values are
`1da6991a7f6640b749301b04d73115c57637d1cbd4319bd275f093f46ab348f1`,
`55d33deb8f6302152c934b601e3e55c7243cddc592bbd7c4a52f8b39befa9d04`,
and
`54c9334674e45dd3ba3ea55845fc0b1801a23dbe6b62a19b77eeb22eb4728bac`.
The Japanese Law Translation snapshot deterministically redacts its sole
per-request anti-CSRF token; the transform and normalized hash are explicit in
the manifest. Development staging authenticates all nested files and remains
`blocked-development-only`. Dataset-policy, sealed-quality,
license-compatibility, and app-distribution blockers all remain.

An independent validation-only law-group screen now covers 400 human legal
units in both directions (800 cases total), selected without model outputs from
JLT validation laws disjoint from training and public test laws. It is not claim
or promotion evidence. A preregistered EN→JA screen rejects all four clean
checkpoints at steps 250/500/750/1,000. Step 1,000 is closest: its paired legal
sentence-chrF++ delta is -0.0685 with a 95% bootstrap interval of
-0.2542…+0.1110; it matches the old neural expert's 208 exact critical-token
and 101 negation mismatches and retains general development quality, but misses
the required -0.1 lower-bound floor. The exact selection artifact hashes to
`aeea1867d0371499643db7e5b7ee7a0a65e6b55131dd533e6bda2ff5142b84af`.
No checkpoint is promoted or substituted into the pack.

A conservative continuation subsequently produces a real validation-law
quality win, but every raw checkpoint regresses the frozen negation and/or exact
critical-token counts. A reference-free two-checkpoint structure fallback
retains that win on validation and is therefore tested once on a separately
frozen 400-case EN→JA test-law slice. The independent result improves corpus
chrF++ 19.2917→19.9002 and critical/negation counts 180/76→161/74, with 22
fallbacks, 62.57 ms warm p95, and a conservative 180,019,246-byte bundle.
However, its paired quality interval is +0.5227 with 95% -0.0816…+1.0782, so
the preregistered positive-lower-bound gate rejects it. The exact result hashes
to `a713673b3d1ae9454bf4a4b7b22d87df9871342d0380820a7ec05654d103b3f3`.
It is not packaged, integrated, or promoted.

The conservative peak from the fresh v2 four-model residency run is
421,003,264 bytes RSS; a same-runtime repeat was 401,195,008 bytes, so the
larger observation is retained.
Dependency-free routing has exact scikit-learn route parity; canary p95 is 0.101
ms EN→JA and 0.052 ms JA→EN. The new native Swift router matches Python on all
2,800 public-v3 cases, including 2,800/2,800 identical route decisions and a
maximum absolute score delta of 5.11e-15. The exact quantized expert pair also
passes 12/12 Swift/Python output-token parity under cached decoding; the
unchanged human-only generalist pair retains its prior 12/12 parity result.
The developer-gated Swift loader now validates both pair and memory-bearing MoE
manifests, including v1 per-engine and v2 shared-tokenizer layouts, every
memory entry's schema, provenance hashes, length,
target script, and exact critical-token equality. It has passed cold end-to-end
selection/decoding smokes for all four neural roles plus a JA→EN memory hit.
The v1 and v2 packs match exactly on those five paths, including 28/28 EN→JA
and 9/9 JA→EN output token IDs; the self-contained parity report SHA-256 is
`5eb077873d26c9e9339aaeafeb2366aea7ab0eebd5d5eacbc824afa830fa5090`.
Swift matches Python on 2,800/2,800 public memory decisions and hypotheses.
None of this adds a user-facing switch or changes the normal engine.

The complete source-routed public-v3 simulation selects the EN→JA expert for
588/1,400 cases before 11 critical-token fallbacks and the JA→EN expert for
196/1,400 before 15 fallbacks. Relative to the authenticated human-only
generalists, routed mean sentence chrF++ improves by +0.837 (95% paired
bootstrap +0.613…+1.089) EN→JA and +1.295 (+1.003…+1.605) JA→EN. Corpus chrF++
is 29.18/53.44, while conversation is exactly unchanged in both directions.
These are positive public development results, not promotion evidence.

The exact-token guard is deliberately conservative: it requires NFKC-equivalent
URLs, placeholders, markup, and digit tokens, plus the separately gated single-
percentage equivalence described below. It safely falls back
from an expert in 26 cases. A deeper final-output audit finds 498/2,800 strict
mismatches: 237 EN→JA and 261 JA→EN, comprising 197 expert-selected and 301
generalist-derived cases. The native guard now covers every path and demonstrably
fails closed in both directions. Some flags are legitimate number-word or era-
year transformations, so 498 is not a semantic error rate; it is a product
failure-rate blocker until a typed structure policy is pre-registered. The
broader source-only number/negation/structure heuristic flags 707 routed outputs
and cannot substitute for a bilingual or independently automated meaning audit.

Replacing source numbers and URLs with generic `[NUMn]`/`[URLn]` labels before
inference is rejected. Although 711/2,800 sources contain replacements, the
generalists lose or duplicate labels in 328 cases and the experts in 329. After
restoration, quality falls by -2.004 mean sentence chrF++ EN→JA (95% interval
-2.496…-1.551) and -3.067 JA→EN (-3.654…-2.521) versus untranslated inputs.
The placeholder preprocessing is therefore not ported to Swift or packaged.

The exact routed pack now also has a reproducible Apple diagnostic. Because
public-v2 and public-v3 sampled different rows, the alignment tool authenticates
both reports and selects their complete content-identical intersection: 647
cases per direction, all explicitly post-hoc and claim-ineligible. Routed
chrF++ leads Apple by +7.818 EN→JA (95% paired interval +6.050…+9.584) and
+2.456 JA→EN (+0.954…+3.990). Local p95 is 56/65 ms versus Apple's 2.17/2.27
seconds, about 39×/35× lower.

Pinned COMET-22 does not confirm an overall accuracy win. The routed deltas are
-0.00265 EN→JA (-0.00855…+0.00324) and -0.00371 JA→EN
(-0.01017…+0.00290), both inconclusive. News significantly favors Apple in both
directions (-0.02584/-0.01929), and JA→EN conversation also favors Apple
(-0.01207); Wikipedia favors the local model. This domain reversal is a hard
quality blocker and demonstrates why chrF++ alone cannot authorize promotion.

```sh
python3 scripts/translation/align_translation_report_intersection.py \
  Research/translation/results/release-clean-human-routed-moe-v2-public-stress-v3.json \
  Research/translation/results/apple-public-stress-v2.json \
  Research/translation/results/release-clean-human-routed-moe-v2-public-v2-v3-overlap.json \
  Research/translation/results/apple-public-v2-v3-overlap.json \
  --minimum-per-direction 400 \
  --suite-output Research/translation/benchmark/public-v2-v3-diagnostic-overlap.jsonl
```

The exact memory repairs that concrete neural failure without using test
references: repeated train documents select the observed human medoid
“(On-site Inspections)” for normalized `（立入調査等）`. It contains 6,179 entries
(676 EN→JA, 5,503 JA→EN), requires the exact source in at least two independent
laws, discards conflicting within-document observations, and rejects every
critical-token mismatch. On untouched validation matches, memory versus routed
neural output gains +13.591 mean sentence chrF++ on 9 EN→JA cases (95% interval
+0.573…+24.174) and +23.962 on 213 JA→EN cases
(+21.034…+26.957). Public-v3 has only seven hits; all improve, yielding +0.035
EN→JA overall (interval lower bound 0) and +0.167 JA→EN (+0.044…+0.319).
These are separately labeled retrieval results, never neural generalization.

The memory source rows are human and PDL-licensed, but they are explicitly
marked `training_only: true` and `promotion_eligible: false`. The new release
contract therefore honestly records
`blocked-training-only-runtime-memory-and-final-review`; it does not reinterpret
those flags to force a release. The pack still records
`doesNotAuthorizeAppIntegration: true`. The independently sourced automated
400-distinct-case-per-direction claim suite, typed critical-error policy, final
license review, and non-Apple production failure path remain open. The sealed
GPT-5.6 reference batch is submitted, but its output and both independent judge
passes remain pending.
Mimi's current default is unchanged.

The exact critical-token tokenizer now accepts sentence-final numbers such as
`12.` without treating the punctuation as a partial decimal, while preserving
multi-dot versions and validating comma grouping. That removes 22 audit-only
false failures. The remaining broad word/kanji/era typed relaxation is rejected
because 27/167 accepted public cases disagree with the human-reference typed
signature. Only one narrow rule is enabled in the native guard: exactly one
explicit digit percentage may change between `percent`/`per cent`/`パーセント`
and `%` when its value, all other numbers, URLs, placeholders, printf tokens,
and markup are exact. It passes one public case with zero disagreements plus
the adversarial suite; it does not authorize model promotion.

Runtime alternatives did not create a free win. Exact full-sentence shallow
draft verification accepted only 18.6% of sampled EN→JA drafts, no usable
JA→EN drafts, and made end-to-end inference roughly four times slower. A matched
precision/group sweep also found the current 4-bit group-64 weights more
accurate than 4-bit groups 32/128 and 6-/8-bit conversions. These negative
results close whole-sentence speculative verification and precision increases
for the current Marian pair; decoder K/V caching remains the measured fast
path.

The v2 JA→EN child comes from a contamination-screened 6,035-row licensed
mixture of capped ALT news, KFTT replay, reciprocal-agreement-filtered Tatoeba,
and Mimi UI pairs. Training used frozen preferred-student KL/L2 preservation;
selection used an unweighted ALT/Tatoeba/UI macro score plus a 0.5 chrF++ KFTT
retention gate; steps 150/225/300 were averaged before exact 4-bit conversion.
It preserves all six JA→EN canary hypotheses and improves the 400-case public
JA→EN stress slice by +0.67 mean sentence chrF++ over v1 (95% paired interval
+0.08…+1.35). The analogous EN→JA child is rejected: its +0.37 public-stress
delta is inconclusive (-0.11…+0.87) and its canary regresses 31.31→29.82.
The hybrid therefore keeps v1 EN→JA and changes only JA→EN.

A deployment sweep confirmed that this apparently aggressive 4-bit/group-64
greedy configuration is also the strongest measured variant. Four-bit groups
32 and 128, 6-bit and 8-bit weights, and beam sizes 2 and 4 all regressed chrF++
in both directions. Every quantized variant passed exact 12/12 Swift/Python
output parity; rejected beam decoders were not added to Swift.

A licensed-data regularization run then applied frozen-base KL, L2-to-base, a
domain curriculum, and deterministic checkpoint averaging to the conversational
EN→JA control. Full-precision development chrF++ improved 30.351→30.895, but
both the best single checkpoint and the average of steps 150/200/250 scored
30.81 after 4-bit shipping conversion. That is below the current 31.31 EN→JA;
both variants are rejected and the preferred bundle is unchanged.

Two exact-MLX quantization-aware continuations were also tested. The training
path reproduces the pinned MLX 0.30.6 affine group-64 quantizer with float16
source/storage semantics and a straight-through gradient, then still uses the
authoritative converter. Quantized development improved slightly in both arms,
but the regularized-parent and shipping-best-parent canaries scored only 30.81
and 30.88 EN→JA. Both are rejected; the mechanism remains available for future
reviewed teacher targets and the preferred bundle is unchanged.

A no-GPT hard-reference control then fine-tuned both students on 665 mined
professional/project-owned pairs plus 1,800 KFTT replay rows per direction. The
fused-kernel EN→JA canary regressed 29.33→27.78 and is rejected. JA→EN improved
55.92→56.52;
paired with the unchanged EN→JA base, the pack is 73,403,570 bytes. This remains
a development-only starting point because the canary is not claim-eligible and
JA→EN still trails Apple by 5.27 corpus chrF++.

A second no-GPT licensed-parallel control added all 63 unambiguous Mimi UI
training pairs and a deterministic cap of 2,000 NICT ALT pairs to 4,126 KFTT
replay rows per direction. After 150 steps, EN→JA development chrF++ improved
28.575→29.117 but its 4-bit canary regressed to 27.59. JA→EN development
improved 49.664→51.063; fused-kernel canary scores are 27.59/56.50, still below Apple.
This confirms that more human-translated news data is not automatically better
for Mimi's live-speech domain; the exact pair remains 73,403,570 bytes and is
not promoted.

The best EN→JA development control replaces ALT with a reciprocal conversational
Tatoeba subset. The gate rejected every one-to-many source mapping, rescanned
the protected suite, and required moderate agreement from both pinned students;
because ElanMT documents Tatoeba exposure, this is only a training-noise filter,
never evaluation evidence. With 1,386 screened Tatoeba pairs, 63 Mimi UI pairs,
and 2,898 KFTT replay rows per direction, fused-kernel reruns reached 31.31
EN→JA and 55.92 JA→EN. Direction-specific selection therefore combines that
EN→JA checkpoint with the hard-reference JA→EN checkpoint, producing the
31.31/56.52 preferred-v1 pair. The licensed-unified experiment above later
replaced only its JA→EN child in preferred-v2. It still trails Apple and is not
promoted.

Qwen3-0.6B 4-bit remains a larger quality reference. Its 351 MB base misses the
new size target and cannot be the shipping candidate.

CyberAgent's June 2026 MIT-licensed CAT-Translate-0.8B was also evaluated as a
single-model alternative. Its pinned MLX q4 snapshot is 453,006,430 bytes and
fits the hard 500 MB ceiling, but 800-case chrF++ (24.90/42.47), COMET-22
(0.8618/0.8034), 0.369/0.442-second p95, and roughly 874 MB RSS all lose to the
73.4 MB preferred pair overall. Multi-domain and conversational/UI QLoRA sweeps
did not produce a two-direction or all-domain win. CAT is not integrated; it is
retained only as a potential distributable diversity teacher.

The July compact-model refresh added two no-go results. LFM2-350M-ENJP-MT has
an attractive 381.6 MB MLX 8-bit bidirectional pack, but LFM 1.0 excludes
commercial use by legal entities at or above its USD 10M annual-revenue
threshold, so it fails the unconditional shipping-license gate. Apache-2.0
Translate-15L fits at 244.6 MB FP32, but exact canary screening is catastrophic:
0.00/1.37 chrF++ under greedy and 0.00/4.11 under its documented beam-4
decoding, with empty/repeated-punctuation output and 2.60-second JA→EN p95.
Neither advances to porting, fine-tuning, or integration.

The same refresh authenticated Tencent's Apache-2.0 Hy-MT2-1.8B 1.25-bit
checkpoint as a one-model hard-ceiling control. The official GGUF is 461.9 MB
but fails to load against its still-unmerged STQ kernel because the published
file uses an older tensor layout/type ID. A 464.2 MB community MLX conversion
with a custom Metal kernel does run: canary chrF++ reaches 35.62/60.70, but the
matched 800-case stress result collapses to 22.64/44.09 versus preferred-v2's
33.23/54.00, with 127 exact critical-token failures and 1.026/0.850-second p95.
It is rejected before COMET or Swift work. Its paper also omits a licensable
inventory for the roughly one-trillion-token training mixture, independently
failing Mimi's release-lineage requirement.

NiuTrans's Apache-2.0 `LMT-60-0.6B` was then tested as a more conventional
single-model MLX/Swift control. The pinned Qwen3-based checkpoint was converted
with MLX 0.30.6 to an authenticated 346,929,488-byte affine q4/group-64 pack.
Its 31.38/54.15 canary does not survive broader coverage: the matched 800-case
stress result is 17.92/40.42 versus preferred-v2's 33.23/54.00, with paired
95% intervals entirely below zero, 141 exact critical-token failures,
0.491/0.397-second p95, and 745.6 MB peak RSS. It is rejected before COMET,
Swift parity, fine-tuning, or integration. The paper's 90B-token source mixture
also lacks row-level license/generator lineage, and its released SFT dataset has
no license tag. The pinned EN–JA SFT shard contains 13,169 pairs entirely from
FLORES-200, NTREX-128, IWSLT 2017/2022, and WMT news 2020/2021/2022. That does
not prove exact test-split overlap, but it makes public-benchmark interpretation
contamination-sensitive and is not admissible independent training/evaluation
evidence. The weight's Apache label therefore does not independently clear the
release-data gate.

CAT's newly released training corpus is likewise excluded. Its public card is
ODC-BY-1.0 plus Common Crawl terms and identifies corpus-level sources and
Apache-2.0 `gpt-oss` generators, but file access is gated and the public
metadata does not bind each row to its exact source license, URL, generator
revision, and filtering decision. Mimi reuses the documented diversity-first,
quality-second curriculum design without importing the 7.11 GB corpus.

A reviewer-free, training-only teacher pilot is now complete. From 2,000
CC-BY-4.0 English BTEC sources, surface agreement among preferred-v2, CAT, and
HPLT retained 542 rows. Three independent reverse engines plus strict mutual
entailment retained 283, and a calibrated Apache-2.0 Qwen3-8B bilingual judge
accepted 256. The judge emits only compact scores/tags, never reasoning traces;
all accepted rows remain permanently promotion-ineligible. Shared 614-token
prompt-prefix caching plus batch-16 generation preserved 16/16 judgments in a
parity smoke while reducing that smoke from about 40 to 13 seconds.

The next distillation comparison is now literature-grounded as a staged
teacher curriculum. An architecture-matched Marian junior stage must first
preserve the frozen development surface; a stronger, independently filtered
GPT final-translation stage then follows. A matched direct-senior control uses
the same accepted rows and total update budget. This adapts Evolving Knowledge
Distillation's capacity-gap result without importing its benchmark data or
requiring teacher reasoning traces. The three larger local candidates screened
here—CAT, Hy-MT2, and LMT-60—do not qualify as senior teachers because all lose
the broad stress control.

The first regularized EN→JA student improved full-precision development
30.579→30.847 and public-v2 30.13→30.25, but its exact 4-bit canary regressed
31.31→30.81. A stronger 2→4 teacher-weight curriculum reached 30.853
full-precision and 30.30 public-v2, but repeated the same canary regression.
A predeclared interpolation sweep found the 15% adapted / 85% parent blend:
it restores every canary hypothesis and yields the small conversation gain
above. That blend becomes preferred-v3 for developer research only; neither
the raw students nor the blend satisfy product promotion.

A stricter professional-reference teacher/control experiment is also complete.
It froze 1,785 novel CC-BY-SA-3.0 KFTT pairs after base-training and protected-
suite exclusion, generated Qwen translations from source text alone, and kept
only 67 rows after pinned COMET-22, chrF++, structural, and baseline-delta
gates. Qwen generation used batch 16 plus a shared prompt-prefix cache, took
1,551.7 seconds for all 1,785 rows, and peaked at 4.98 GB RSS; no references,
student hypotheses, or reasoning traces were exposed to the teacher.

Matched low-dose students used the same sources with either Qwen targets or the
professional KFTT references. EN→JA selected step 75 in both arms: 30.908
full-precision development chrF++ for Qwen and 30.872 for the human-reference
control, versus 30.595 at step 0. After provenance-authenticated rebuilding,
both exact 4-bit packs produce the same canary hypotheses and score 29.996
EN→JA, below preferred-v3's 31.305 by -1.357 mean sentence chrF++ (95% paired
bootstrap -3.119…-0.030). JA→EN selects the untouched step-0 parent in both
arms; every trained checkpoint regresses. The 73,410,348-byte Qwen and
73,410,410-byte control packs are rejected without a public-v2 run, and
preferred-v3 remains unchanged. The earlier unauthenticated 30.808 Qwen score
is superseded: it did not bind the input report hashes or model revisions, while
the rebuilt Qwen result repeats exactly.

The evidence supports larger, more domain-diverse accepted sets rather than
more loss weight or hidden teacher reasoning.

The next round therefore freezes a balanced 2,400-row training-only suite:
400 conversation, 400 human-translated news, and 400 professional Wikipedia
cases per direction. A full licensed-pool inventory first excludes the active
train/validation sources, the prior 1,785-row teacher suite, ambiguous Tatoeba
source IDs, normalized duplicates, and exact or greater-than-0.8 character
5-gram overlap with the canary and public-v2 suites. The smallest surviving
cell still has 8,411 candidates. The exact preferred-v3 pack then scores a
deterministic 600-row pool per cell and selects 400 using uncertainty thirds
plus greedy encoder-cosine k-center diversity. The frozen suite SHA-256 is
`98da175c5a7d937afd280fec0db23757702c74dc8dd64f43e1eb3b2cd48d1198`.
Before teacher results are visible, the automated admission policy requires at
least ten strict chrF++/COMET improvements in every domain/direction cell or
rejects the entire round. This suite is training evidence only and cannot be
used to claim a model improvement.

The balanced Qwen3-8B pass is now complete. It generated all 2,400 source-only
translations in 1,598.2 seconds with 4,973,641,728 bytes peak RSS; references,
student hypotheses, and reasoning traces were never exposed. The strict gates
found 281 potential targets, but the Wikipedia cells retained only 9 EN→JA and
7 JA→EN rows against the predeclared minimum of 10. The whole synthetic round
is therefore rejected and no training JSONL was emitted. The authenticated
failure report has SHA-256
`61a94ade4675cf9ebad7096dc570b3f57e83c8c9bf591ce48d7559adcc5d0e17`.

A bounded recovery kept the frozen sources, metrics, thresholds, and ten-row
cell floor. It regenerated only the deficient Wikipedia slice, retained the
original candidate unless an alternative passed every gate, and never selected
on the canary or public-v2 outputs. The protected-token validator now
canonicalizes English month names and Japanese numeric months (for example,
`August` and `8月`) while still requiring every other number, percentage, URL,
placeholder, and markup token exactly; the manifest pins this as policy v2.
The recovered set admits 290 training-only targets: 36/87 conversation, 50/94
news, and 10/13 Wikipedia rows for EN→JA/JA→EN. Its JSONL SHA-256 is
`f28b51052655d6fd4958fdaaadb561872de86cace2fcc03fa9893edff9f42382`.

Matched 200-step regularized students then provide a negative result. EN→JA
development improves 30.595→30.819 at step 50, but only a 1% global blend into
preferred-v3 preserves every canary output; 2–10% blends change protected
outputs. Against a freshly regenerated preferred-v3 public-v2 control under the
pinned MLX 0.30.6 shipping contract, that 1% arm changes 28/1,200 EN→JA token
sequences and regresses by -0.079 mean sentence chrF++ (95% paired interval
-0.147…-0.020). JA→EN selects the untouched step-0 parent: steps 50/100/150/200
score 52.157/52.100/52.108/52.106 development chrF++, all below 52.180 at
step 0. Neither direction advances to promotion, and preferred-v3, the app
default, and the Apple fallback remain unchanged.

The matched all-human control trained on the same 2,400 selected sources using
their licensed references. EN→JA development improved 30.595→30.871 at step
100, while JA→EN selected the untouched step-0 parent. The resulting
73,411,676-byte exact-4-bit pair gains +0.381 mean sentence chrF++ over
preferred-v3 on public-v2 (95% paired interval +0.153…+0.623), led by the
Wikipedia slice, but regresses the canary by -0.476 and changes a protected
number/entity output. Blending 5%, 10%, 15%, or 25% of the adapted EN→JA
checkpoint repeats the same protected regression; 50% is worse. Every arm is
rejected, preferred-v3 remains unchanged, and no app default or fallback was
modified.

A component/layer task-vector sweep cannot separate the public gain either.
Encoder-only, decoder-only, tied-embedding/output-only, and transformer-only
merges all regress the canary. Replacing each of the twelve encoder/decoder
layers individually also changes at least one protected output: eleven repeat
the main number/entity loss, while decoder layer 5 preserves that row but
regresses the macOS/UI row by -0.177 sentence chrF++. This closes selective
checkpoint merging; further canary-guided masking would overfit the regression
set rather than improve the translator.

The requested one-model SmolLM2 route was tested first. Its English-centric
tokenizer and frozen embeddings capped KFTT LoRA at 4.79/27.52 chrF++; it is
rejected despite fitting in 89.2 MB. The two-student ElanMT design is both much
smaller after quantization and substantially better. NLLB-200 and JParaCrawl
remain excluded because their licenses prohibit the intended product use.

## Reproduce the evidence

The checked-in canary only tests plumbing. Its references are not independent
claim evidence, and every row has `claimEligible: false`.

```sh
scripts/translation/test.sh

# Also perform the real one-update MPS training smokes when the pinned local
# full-precision Marian checkpoints are available:
MIMI_TRANSLATION_TRAINING_SMOKE=1 scripts/translation/test.sh

scripts/build-app.sh debug
scripts/translation/run-apple-benchmark.sh \
  Research/translation/benchmark/canary.jsonl \
  Research/translation/results/apple-canary.json

uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/score_translation.py \
  Research/translation/results/apple-canary.json
```

The promotion suite is intentionally not generated by GPT and is not checked
into a training path. The checked-in legacy claim contract requires two blind
bilingual reviews and adjudication. The user has authorized reviewer-free
development, so experiments do not wait on that process; however, no row may be
set `claimEligible: true` until a replacement independently sourced automated
product-domain contract is implemented and frozen. Enforce the direction/domain
quotas and scan every final training JSONL with:

```sh
python3 scripts/translation/validate_benchmark_suite.py \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/benchmark/manifest.json \
  PRIVATE_PATH/heldout-review-records.jsonl \
  --training-jsonl Research/translation/work/distillation-en-ja/train.jsonl \
  --training-jsonl Research/translation/work/distillation-ja-en/train.jsonl \
  --output Research/translation/work/heldout-validation.json
```

The full author → two blind reference reviewers → adjudicator → two blind
engine reviewers → promotion-evaluator workflow is documented in
`benchmark/README.md`. `evaluate_translation_promotion.py` exits zero only when
every metric, human, performance, integrity, and fallback gate passes in both
directions; descriptive `score_translation.py` output cannot promote a model.
Score reports are nevertheless evidence artifacts: schema v2 authenticates the
candidate and comparator report SHA-256 values, their model revisions, and a
canonical digest of the aligned suite content. It also records SacreBLEU 2.6.0
metric signatures and the paired-bootstrap seed/count. Use `--compare-report`
for any model/control/incumbent comparison; reserve `--compare-apple` for a
report actually produced by Apple Translation.

## Data and training flow

1. Prepare professionally translated KFTT, conservatively screened reciprocal
   Tatoeba, and Mimi's paired project-owned English/Japanese shipping copy. The corrected UI
   extraction yields 76 unambiguous pairs (63 train, 13 source-file-grouped
   validation); four one-to-many source mappings are rejected rather than
   giving the student conflicting targets. NICT ALT remains a capped secondary
   ablation because its news-domain control regressed conversational EN→JA.
2. After freezing the final held-out suite, run
   `prepare_distillation_seeds.py` against the exact 4-bit student. It mines
   licensed KFTT rows using sequence uncertainty strata plus greedy cosine
   coverage of the student's mean-pooled encoder states, after excluding badly
   aligned references. The legacy lowest-chrF selector remains available only
   for reproduction. Professional references, embeddings, and student
   hypotheses remain local; only source text is sent to the teacher. Add a
   small hash-sampled set of
   English BTEC travel/service utterances as source-only EN→JA teacher seeds;
   BTEC is not mislabeled as parallel gold.
3. Generate three candidate translations plus compact structured translation
   facts with a high-reasoning GPT-5.6 teacher. First,
   `merge_distillation_seeds.py` fails on duplicate normalized sources; then
   `prepare_synthetic_batch.py` seals the request. The tools refuse benchmark and held-out rows
   and never requests or stores chain-of-thought. Validate the sealed JSONL
   offline, then use `run_synthetic_batch.py` for an explicitly hash-confirmed
   upload/submission, status checks, and complete-ID-verified collection.
4. Run `filter_synthetic_batch.py` for language, placeholder, number, length,
   exact-duplicate, and near-duplicate checks against the protected benchmark.
5. Use `prepare_distillation_judge_batch.py` with a model distinct from the
   teacher, submit it through the same hash-confirmed runner, and validate it
   with `prioritize_distillation_judgments.py`. For promotion-eligible data, the
   scores only order human work. For the explicitly provisional reviewer-free
   SFT lane, repeat this with two distinct judge models and run
   `approve_automated_consensus.py`; both judges must uniquely select the same
   error-free candidate, and the result remains ineligible for DQO/promotion.
6. For the optional human-evidence lane, run `approve_bilingual_selections.py`.
   The same candidate must be selected
   independently by both reviewers; disagreements require selection or
   reject-all by a distinct third adjudicator. Reviewers may also independently
   approve the same noncanonical, meaning-equivalent but genuinely distinct
   alternative. A one-sided alternative is discarded. `approve_synthetic_reviews.py`
   remains only as a lower-level candidate-review contract test.
   Reviewer-free training instead uses the exact automated consensus artifact
   from step 5 and passes `--allow-automated-consensus`; those rows remain
   promotion-ineligible and cannot feed DQO.
7. Run `build_distillation_dataset.py` separately for each direction. It makes
   a deterministic reviewed train/dev split, samples professional KFTT replay,
   admits multiple prepared parallel corpora with per-corpus caps, and repeats
   the protected-suite contamination scan. Arm B uses canonical targets; arm C
   passes `--reviewed-target-mode sample-approved-diverse`, keeps one row per
   source, samples one of at most two approved targets per epoch, and always
   evaluates against the canonical target.
8. Fine-tune all ElanMT student weights with
   `train_marian_distillation.py`. The DQRD arm adds KL from a frozen copy of
   the pinned base on KFTT replay, L2-to-base, and a linear domain-loss
   curriculum while keeping replay weight at 1.0. The one-update MPS smoke is
   `smoke_train_marian_distillation.py` and covers the regularized objective.
9. Save every reviewed-dev evaluation checkpoint and run
   `average_marian_checkpoints.py`. It chooses the best three adjacent
   checkpoints by mean reviewed-dev chrF++ (then loss) and writes an exact
   input/output hash manifest. Quantize only this averaged checkpoint.
10. If and only if `evaluate_supervised_win.py` emits an approved, hash-bound
    report, build conservative two-reviewer preference pairs and run the short
    DQO ablation. The algebra/data contract is covered by
    `test_dqo_pipeline.py`; `smoke_train_marian_dqo.py` performs a real
    one-update MPS optimizer smoke against the local Marian checkpoint.
11. Evaluate the base model, distilled/DQO checkpoint, and Apple on the same frozen
   held-out suite.

`build_hard_reference_ablation.py` is the no-GPT control for this experiment.
It uses the licensed human reference attached to each mined row plus KFTT replay
and Mimi validation copy. Its mixed result—EN→JA regressed while JA→EN improved—
is why the teacher experiment must be compared with both the original base and
this control rather than credited from development loss alone.

The pinned compact-model conversion and benchmark commands are:

```sh
uv run --python 3.12 --with mlx --with tokenizers --with protobuf \
  --with sentencepiece scripts/translation/prepare_elanmt_mlx.py \
  PATH_TO_PINNED_ELANMT_EN_JA Research/translation/models/elanmt-en-ja-mlx \
  --repository Mitsua/elan-mt-bt-en-ja \
  --revision 02c48e7031386cd2d41974b0ff1aaf52f010c5fa --direction en-ja

python3 scripts/translation/package_elanmt_mlx.py \
  Research/translation/models/elanmt-en-ja-mlx \
  Research/translation/models/elanmt-ja-en-mlx \
  Research/translation/models/elanmt-bt-mlx-4bit-pack

uv run --python 3.12 --with mlx --with transformers==4.40.2 \
  --with sentencepiece --with sacremoses \
  scripts/translation/run_mlx_marian_benchmark.py \
  Research/translation/benchmark/canary.jsonl \
  Research/translation/results/elanmt-canary.json \
  --en-ja-model Research/translation/models/elanmt-en-ja-mlx \
  --ja-en-model Research/translation/models/elanmt-ja-en-mlx
```

The source-only teacher and independent review lane is reproducible as follows.
The generated work artifacts from the canary-only run are provisional: freeze
`heldout.jsonl`, then regenerate them before submitting any API batch.

```sh
python3 scripts/translation/prepare_mimi_ui_parallel.py \
  Sources/Mimi Research/translation/benchmark/heldout.jsonl \
  Research/translation/work/mimi-ui-parallel

python3 scripts/translation/prepare_alt.py \
  Research/translation/work/source-archives/ALT-Parallel-Corpus-20191206.zip \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/work/alt

python3 scripts/translation/prepare_btec_teacher_seeds.py \
  Research/translation/work/source-archives/enBTEC20K.zip \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/work/btec-teacher-seeds.jsonl

uv run --python 3.12 --with mlx --with transformers==4.40.2 \
  --with tokenizers --with sacrebleu==2.6.0 \
  scripts/translation/prepare_tatoeba_parallel.py \
  Research/translation/work/tatoeba \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/work/tatoeba-agreement \
  --en-ja-model Research/translation/models/elanmt-bt-mlx-4bit-pack/en-ja \
  --ja-en-model Research/translation/models/elanmt-bt-mlx-4bit-pack/ja-en

uv run --python 3.12 --with mlx --with transformers==4.40.2 \
  --with sentencepiece --with sacremoses --with sacrebleu==2.6.0 \
  scripts/translation/prepare_distillation_seeds.py \
  Research/translation/work/kftt/train.jsonl \
  Research/translation/work/mimi-ui-parallel/train.jsonl \
  Research/translation/models/elanmt-dqrd-preferred-v2-mlx-4bit-pack \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/work/dqrd-distillation-seeds.jsonl \
  --selection-strategy uncertainty-diversity \
  --seed mimi-dqrd-distillation-v1

python3 scripts/translation/merge_distillation_seeds.py \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/work/distillation-seeds-merged.jsonl \
  Research/translation/work/dqrd-distillation-seeds.jsonl \
  Research/translation/work/btec-teacher-seeds.jsonl

python3 scripts/translation/prepare_synthetic_batch.py \
  Research/translation/work/distillation-seeds-merged.jsonl \
  Research/translation/work/gpt-5.6-distillation-requests.jsonl

python3 scripts/translation/run_synthetic_batch.py validate \
  Research/translation/work/gpt-5.6-distillation-requests.jsonl

# Only after the final held-out suite is frozen and screened, paste the exact
# SHA-256 printed by validate. This is the first command that makes a request.
uv run --python 3.12 --with openai==2.46.0 \
  scripts/translation/run_synthetic_batch.py submit \
  Research/translation/work/gpt-5.6-distillation-requests.jsonl \
  Research/translation/work/gpt-5.6-distillation-batch-state.json \
  --confirm-input-sha256 PASTE_EXACT_VALIDATED_SHA256

uv run --python 3.12 --with openai==2.46.0 \
  scripts/translation/run_synthetic_batch.py status \
  Research/translation/work/gpt-5.6-distillation-batch-state.json

uv run --python 3.12 --with openai==2.46.0 \
  scripts/translation/run_synthetic_batch.py collect \
  Research/translation/work/gpt-5.6-distillation-batch-state.json \
  Research/translation/work/gpt-5.6-distillation-output.jsonl

python3 scripts/translation/filter_synthetic_batch.py \
  Research/translation/work/distillation-seeds-merged.jsonl \
  Research/translation/work/gpt-5.6-distillation-output.jsonl \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/work/review-queue.jsonl

# After deterministic filtering:
python3 scripts/translation/prepare_distillation_judge_batch.py \
  Research/translation/work/review-queue.jsonl \
  Research/translation/work/instant-judge-requests.jsonl \
  --model DISTINCT_FAST_OR_INSTANT_MODEL_ID

# Optional: validate/submit/status/collect instant-judge-requests.jsonl with
# run_synthetic_batch.py exactly as above, using a separate state file. Then:
python3 scripts/translation/prioritize_distillation_judgments.py \
  Research/translation/work/review-queue.jsonl \
  Research/translation/work/instant-judge-output.jsonl \
  Research/translation/work/instant-judge-priority.jsonl

# Reviewer-free provisional SFT only: independently prepare, run, and validate
# two judge batches as above, producing judge-a-priority.jsonl and
# judge-b-priority.jsonl. Then require exact two-model consensus:
python3 scripts/translation/approve_automated_consensus.py \
  Research/translation/work/review-queue.jsonl \
  Research/translation/work/judge-a-priority.jsonl \
  Research/translation/work/judge-b-priority.jsonl \
  Research/translation/work/automated-approved.jsonl \
  Research/translation/work/automated-rejected.jsonl

python3 scripts/translation/prepare_bilingual_review_packets.py \
  Research/translation/work/review-queue.jsonl \
  Research/translation/work/bilingual-review \
  --reviewer REVIEWER_A --reviewer REVIEWER_B \
  --priority Research/translation/work/instant-judge-priority.jsonl

python3 scripts/translation/approve_bilingual_selections.py \
  Research/translation/work/review-queue.jsonl \
  Research/translation/work/bilingual-review/REVIEWER_A.responses.jsonl \
  Research/translation/work/bilingual-review/REVIEWER_B.responses.jsonl \
  Research/translation/work/approved.jsonl \
  Research/translation/work/disagreements.jsonl \
  --adjudications Research/translation/work/adjudications.jsonl
```

After the API batch and bilingual review are complete, build and train one
direction as follows; repeat with the pinned Japanese-to-English checkpoint:

```sh
python3 scripts/translation/build_distillation_dataset.py \
  Research/translation/work/approved.jsonl \
  Research/translation/work/kftt \
  Research/translation/benchmark/heldout.jsonl \
  Research/translation/work/distillation-en-ja \
  --direction en-ja \
  --parallel-corpus-directory Research/translation/work/mimi-ui-parallel-v2 \
  --parallel-corpus-directory Research/translation/work/tatoeba-agreement-v1 \
  --maximum-parallel-train-per-corpus 1500 \
  --maximum-parallel-valid-per-corpus 150

# When using automated-approved.jsonl instead of human-approved rows, add:
#   --allow-automated-consensus

uv run --python 3.12 --with torch --with transformers==4.40.2 \
  --with sentencepiece --with sacremoses --with sacrebleu==2.6.0 --with numpy \
  scripts/translation/train_marian_distillation.py \
  Research/translation/work/distillation-en-ja \
  Research/translation/models/elanmt-en-ja-dqrd-b \
  --direction en-ja --repository Mitsua/elan-mt-bt-en-ja \
  --revision 02c48e7031386cd2d41974b0ff1aaf52f010c5fa \
  --frozen-base-kl-weight 0.25 \
  --l2-to-base-weight 0.0001 \
  --domain-loss-weight-start 0.25 \
  --domain-loss-weight-end 1.0 \
  --checkpoint-directory Research/translation/work/checkpoints/en-ja-dqrd-b

uv run --python 3.12 --with torch --with safetensors --with numpy \
  scripts/translation/average_marian_checkpoints.py \
  Research/translation/work/checkpoints/en-ja-dqrd-b \
  Research/translation/models/elanmt-en-ja-dqrd-b-averaged \
  --count 3 \
  --selection-origin reviewed-gpt-teacher \
  --retention-origin human-kftt-replay \
  --maximum-retention-regression 0.5
```

After the reviewed SFT arms have been trained, averaged, quantized, and measured,
the optional DQO arm remains locked until concrete supervised evidence passes:

```sh
python3 scripts/translation/build_dqo_preferences.py \
  PRIVATE_PATH/review-queue.jsonl \
  PRIVATE_PATH/approved-selections.jsonl \
  Research/translation/benchmark/heldout.jsonl \
  PRIVATE_PATH/dqo-preferences-en-ja \
  --direction en-ja

python3 scripts/translation/prepare_engine_comparison_packets.py \
  PRIVATE_PATH/sft-development-en-ja.json \
  PRIVATE_PATH/base-development-en-ja.json \
  PRIVATE_PATH/dev-review \
  --reviewer REVIEWER_A --reviewer REVIEWER_B \
  --baseline-key base

uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/evaluate_supervised_win.py \
  PRIVATE_PATH/reviewed-development-en-ja.jsonl \
  PRIVATE_PATH/sft-development-en-ja.json \
  PRIVATE_PATH/base-development-en-ja.json \
  PRIVATE_PATH/kftt-retention-en-ja.jsonl \
  PRIVATE_PATH/sft-retention-en-ja.json \
  PRIVATE_PATH/base-retention-en-ja.json \
  PRIVATE_PATH/dev-review/sealed-assignments.jsonl \
  PRIVATE_PATH/dev-review/reviewer-a.responses.jsonl \
  PRIVATE_PATH/dev-review/reviewer-b.responses.jsonl \
  PRIVATE_PATH/sft-4bit-pair \
  PRIVATE_PATH/sft-full-precision-en-ja \
  PRIVATE_PATH/supervised-win-en-ja.json \
  --direction en-ja

uv run --python 3.12 --with torch --with transformers==4.57.6 \
  --with sentencepiece --with sacremoses --with numpy \
  scripts/translation/train_marian_dqo.py \
  PRIVATE_PATH/dqo-preferences-en-ja \
  PRIVATE_PATH/sft-full-precision-en-ja \
  PRIVATE_PATH/supervised-win-en-ja.json \
  PRIVATE_PATH/dqo-en-ja \
  --direction en-ja --device mps
```

Repeat independently for JA→EN. `train_marian_dqo.py` rejects a failed gate,
changed checkpoint, adjudicated preference, automated preference, unreviewed
candidate, or source-level train/validation leakage. A DQO checkpoint is never
promoted by this command; it must be re-quantized and pass the same development,
retention, Swift/MLX, size, performance, blind-human, learned-metric, and Apple
held-out gates as every other candidate.

The benchmark runner pins the model revision, disables sampling, records model
bytes and peak process memory, and emits the same report shape as the Apple
harness. Each case has one first-pass translation for quality plus three warm
latency repetitions; scoring uses the warm repetitions for p50/p95. The
promotion evaluator additionally requires Apache-2.0 COMET-22 at the exact
manifest revision under `unbabel-comet==2.2.7`, float32 scoring, mean aggregation
across references, and a positive paired lower confidence bound in both
directions. XCOMET is not used because its published model license is
noncommercial.
generated training data, weights, and reports are ignored by Git.

The final canary command was run directly against the exact preferred-v3 pack.
It reported `modelBytes: 73402252` and model revision
`pair-manifest-sha256:362c17bafa48dcd50325e768747e945239d5d42d5b0a822ea6390b70527ac570`,
confirming that tokenizer and quality measurements correspond to the exact app
candidate rather than the larger conversion workspace.

For packaging, the matching MLX 0.30.6 prebuilt `mlx.metallib` is 128,008,745
bytes. A temporary signed universal app containing that shader and the exact
73.4 MB model pack occupied about 281.3 MB on disk. The current universal,
signed candidate archive embeds preferred-v2 and that exact shader, passes
byte-for-byte distribution verification, and is 124,847,075 bytes at SHA-256
`db4841203b312f5a6202d553fbdc9c00adbfae7541fdcfc52e4d7bc142f581df`.
That clears the strict 150,000,000-byte download target, not a 150 MB installed-
app target. Translation weights remain absent from Mimi's promoted public release.

The preferred-v3 candidate was independently packaged after the minimal-child-
manifest fix. Its signed universal archive is 124,846,501 bytes at SHA-256
`b7ab1de8b5af596a3d433559a625166b561acdb144f2c0ff1bceb0bc1674b598`.
Distribution verification binds all nine model-pack files, the 128,008,745-byte
MLX 0.30.6 shader, and root manifest
`362c17bafa48dcd50325e768747e945239d5d42d5b0a822ea6390b70527ac570`.
This is a developer candidate archive, not a promoted release.

## Developer-gated app integration

The local lane is deliberately unavailable through Mimi's settings. A developer
must provide a matching MLX Metal library beside the app executable and launch
with all required environment variables:

```sh
MIMI_EXPERIMENTAL_LOCAL_TRANSLATION=1 \
MIMI_TRANSLATION_MODEL_DIR="$PWD/Research/translation/models/elanmt-dqrd-preferred-v3-mlx-4bit-pack" \
  .build/Mimi.app/Contents/MacOS/Mimi

.build/Mimi.app/Contents/MacOS/Mimi \
  --validate-translation-mlx \
  Research/translation/models/elanmt-dqrd-preferred-v3-mlx-4bit-pack
```

Use `scripts/prepare-mlx-metallib.sh` when full Xcode is available, or provide
its documented `MIMI_MLX_METALLIB` override. An official Python MLX 0.30.6
prebuilt library was version-matched to the pinned Swift 0.30.6 runtime for the
local verification; never substitute a shader from a different MLX release.
The engine never downloads a model.
It accepts only an explicit local pair with `en-ja` and `ja-en` children,
validates formats, sizes, SHA-256 checksums, tokenizers, weights, and target
script. In the explicit experimental local lane, a failure preserves completed
local results, shows a retryable error, and never enables Apple. Floating partial captions
show the source text instead of calling Apple while the experimental lane is
enabled; the normal non-experimental default is unchanged. Swift compilation,
tokenizer parity, and
default-device inference now pass here with the exact Metal library. The first
EN→JA smoke process took 3.0 seconds including shader/model setup; a subsequent
JA→EN smoke process took 0.62 seconds with the system shader cache warm. The
in-app lane remains developer-only and still requires the matching library.

### Incremental-decoding acceleration

The shipping-shaped Swift runtime now caches every decoder layer's projected
self-attention keys/values and the encoder cross-attention keys/values. It
therefore evaluates only the new target token instead of recomputing the full
target prefix at every step. Full-prefix decoding remains callable through the
parity harness as the reference implementation.

On the release app binary, 30 warm repetitions for each of the 12 canary rows
(180 measurements per direction) reduced EN→JA p50/p95 from 50.9/54.1 ms to
29.4/32.4 ms and JA→EN from 49.0/61.9 ms to 27.7/34.2 ms. That is a 1.67× and
1.81× p95 speedup with no extra model bytes. Both modes preserve all 12 canary
hypotheses.

The 800-row public stress suite proves 800/800 exact generated-token parity
between Python MLX and Swift MLX for preferred-v2. Preferred-v3 independently
passes the expanded 2,400/2,400 audit from its exact 73,402,252-byte pack.
Eight v1 and sixteen v2 JA→EN rendered strings differ
only because the Swift and Transformers tokenizer decoders handle spaces before
punctuation differently; both cached and full-prefix Swift exhibit the same
eight formatting differences, and generated token IDs are identical. Scoring
the actual Swift strings changes JA→EN chrF++ only 53.9954→53.9897 and leaves
BLEU unchanged. The optimized path is therefore enabled inside the already
developer-gated local runtime.

Two further output-preserving acceleration ideas are now implemented as
measured experiments. A block-growing decoder self-K/V cache avoids a
concatenation on most token steps, but MLX's functional slice-update cost is
higher at this model's short target lengths. With 30 warm runs per canary item,
block sizes 16, 64, and 192 preserve all generated token IDs yet slow p50/p95
by about 3–11% relative to stable before/after concatenating-cache baselines.
The option remains available only through
`--preallocated-kv-cache-block-size`; it is rejected for Swift integration.
Precomputing the exact 192×512 sinusoidal position table also preserves every
canary token. The authenticated rerun gives only a 1–2% EN→JA improvement;
JA→EN p95 improves by under 1.3% while p50 falls between the two surrounding
baselines. This is too small relative to run drift to justify a Swift change,
so `--precomputed-position-table` remains a Python-only rejected experiment.
MLX 0.30.6 can compile the cached decoder step at fixed shapes, but shapeless
compilation fails shape inference in the quantized `AddMM` path. Because the
legacy self-cache grows every token, fixed-shape compilation would recompile at
each length. No compile flag was added; a valid compiled design would first
need a fixed-shape masked cache and must beat the rejected cache experiment.

A structural 6-encoder/2-decoder ablation tests the literature's stronger
latency lever. Directly retaining decoder layers 0 and 5 shrinks the two-way
4-bit pack to 54,333,716 bytes, but it is not a valid shortcut: repetitive
outputs hit the 192-token cap, canary chrF++ collapses to 0.144/0.075, and
matched cached-decoding p50 grows to 0.172/0.170 seconds despite fewer decoder
layers (30 warm runs, versus the incumbent's matching warm-30 KV report). A
bounded EN→JA recovery pilot then trains the 2-layer decoder for 300 steps on
4,346 licensed KFTT/Tatoeba/Mimi rows while matching the intact preferred
model's logits on every origin. It uses no reasoning traces and evaluates all
1,085 validation rows, but reaches only 4.981 development chrF++. Its exact
4-bit cached canary remains 1.986 and 0.162-second p50; the mixed two-way pack is
63,869,086 bytes. Both variants are rejected. The result narrows the next
architecture experiment: initialize and distill a shallow decoder throughout
training (or progressively drop layers), rather than amputating a converged
6-layer decoder.

That next architecture experiment is now measured at substantially higher
data scale. `build_shallow_student_dataset.py` authenticates 72,061 unique
EN→JA sources from KFTT, ALT, Tatoeba, and Mimi UI copy, excludes validation
and protected-suite overlap, and emits a deterministic 165,050-row repeated
training mixture plus 1,285 untouched validation rows. Its manifest SHA-256 is
`864c66d6c873db96547847cc668bed1a51347e9ee56cf8f4357d762404aa5f07`.
A source-only full-precision Marian teacher then generated canonical sequence
targets for 72,050 sources; eleven immediate-EOS outputs were rejected rather
than silently replaced by human references. The sequence-KD dataset manifest
is `8d8f4ecc44f1e1009ff3a3c25914a0d917fba3f09ef0f4c6b0f255422ab8cfac`,
records that references were not exposed, contains no reasoning traces, and
keeps the same human validation set.

At 1,000 matched updates, the 6-encoder/2-decoder human-reference arm reaches
12.288 development chrF++, while source-only sequence distillation reaches
12.823. The consistent roughly +0.5 gain validates canonical sequence KD but
does not rescue the two-layer decoder. A 6-encoder/4-decoder student is much
stronger: step 500 reaches 27.184 development chrF++ and wins the adjacent
checkpoint selection, while the 500/750/1,000 arithmetic average is lower on
the exact quantized canary. The step-500 mixed pair is 71,074,502 bytes and
reduces EN→JA warm p50/p95 from the incumbent's 29.30/31.86 ms to
22.64/25.86 ms. It nevertheless falls from 31.305 to 27.057 canary chrF++
with a -3.693 mean sentence delta (95% paired interval -11.789…+2.554), and
peak Python RSS rises from 205.8 MB to 229.9 MB. The averaged arm falls further
to 24.641. Both are rejected before public-v2 evaluation or Swift integration.
Their training manifests also remain explicitly blocked pending the documented
share-alike and attribution review, so neither is a distributable candidate.

Two deep-encoder/shallow-decoder follow-ups test whether one-time source-side
capacity can close that gap. Naively appending two zero-residual/unit-layer-norm
encoder blocks is not identity in Marian's post-norm architecture: development
starts at 0.487 and reaches only 4.592/13.708 at steps 250/500. The arm is
stopped there without quantization. Widening each of the six encoder FFNs from
2,048 to 4,096 is correctly output-preserving: copied active features sit
behind zero new `fc2` columns, and all six initial EN→JA canary outputs match
the 6/4 control. It reaches 26.567/27.135/27.208/27.245 development chrF++ at
steps 250/500/750/1,000, a small +0.061 over the normal-width best.

The gain does not survive the shipping representation. The variable-FFN MLX
loader and converter authenticate a 6-encoder/4-decoder, 4,096-wide encoder
pack, but exact 4-bit canary chrF++ falls to 23.592. Relative to preferred-v3,
the mean sentence delta is -7.751 with a wholly negative 95% interval
(-15.680…-0.898). The mixed pair is 78,177,088 bytes and remains faster at
24.93/28.73 ms EN→JA p50/p95 versus 29.75/32.14 ms, while RSS is effectively
tied at about 207.2 MB. This arm is also rejected before public-v2 or Swift;
its comparison report SHA-256 is
`3cf6d1955f91590be2c6ebbf3c7bd9e640051f729c72e05deae826dbf1da7423`.

The app-level runtime cache is separately improved. The experimental engine
now lazily retains one validated runtime per direction for the active model
configuration, so alternating English and Japanese finalized segments no
longer evict and reload the other roughly 36 MB specialist. Changing the
configured model directory clears both slots, and all checksum and cache
invalidation behavior remains unchanged. Candidate failure now stays on the
local lane, preserves successful results, and exposes a retry instead of
silently switching engines. Debug and
release verification cover both-direction retention and configuration
invalidation. The memory tradeoff must still be measured in the packaged app
before this becomes promotion evidence.

Previous-output verification for growing caption partials is also rejected in
its zero-bias form. The literature-matched variant verifies the previous target
in parallel, trims speculative self-attention caches at the first divergence,
and resumes exact greedy decoding. The first unrestricted canary run exposed a
parallel-versus-incremental MLX numerical divergence: one 128-token JA→EN draft
eventually changed target token 23. A conservative 64-token cap and exact
fallback for zero-token acceptance restore parity, but the 25/50/75/100%
source-prefix canary produces pathological 95–192-token partial outputs on
nearly every early update. It verifies no EN→JA draft tokens and no JA→EN
tokens, yielding 0.992× and 1.013× total speed ratios—noise, not acceleration.
The passing rejection report SHA-256 is
`b0f49fe43ec52e75e70f2e5d85260642722f4323f922cd81eb44eb3cbdb3101c`.
The ordinary cached path remains unchanged and preserves all twelve incumbent
canary token sequences after the experimental prefill methods were added.
Nothing is ported to Swift or enabled in Mimi; future partial translation first
needs prefix-trained EOS/length behavior and real 180 ms ASR traces.

```sh
.build/Mimi.app/Contents/MacOS/Mimi \
  --verify-translation-mlx-parity \
  Research/translation/results/dqrd-preferred-v2-kv-cache-swift-public-stress-v1-token-parity.json \
  --model-root Research/translation/models/elanmt-dqrd-preferred-v2-mlx-4bit-pack \
  --suite Research/translation/benchmark/public-stress-v1.jsonl \
  --python-report Research/translation/results/dqrd-preferred-v2-kv-cache-public-stress-v1.json
```

`--verify-translation-fallback OUTPUT.json` exercises the normal disabled
configuration, candidate-failure transition, candidate-output clearing,
invalid-pack rejection, and the separate Apple-only live-partial-caption path.
The final promotion evaluator requires this generated artifact.

The OpenAI Batch API stage is designed for `gpt-5.6-sol`, Structured Outputs,
`store: false`, stable custom IDs, and saved response metadata. The Batch API can
process `/v1/responses` JSONL asynchronously; output order is not guaranteed, so
all joins use `custom_id`. The preparation and validation commands are offline.
The separate runner validates both the source-only teacher and blinded judge
contracts. It refuses submission unless the operator pastes the exact
validated input SHA-256, persists file/batch IDs, refuses duplicate submission,
and proves that every request ID appears exactly once across the collected
success and error files. The documented commands pin the tested OpenAI Python
SDK 2.46.0, and the state records the actual SDK version. The runner never
reads or writes the API key itself.
The legacy balanced 2,400-source GPT-5.6-sol request file was generated at
6,131,829 bytes, SHA-256
`55ee19b9d617b11566d01e2ed83f825bb86fb3c0fa1c1465b414ac3a8dc273fd`.
It exposes no references or student hypotheses and uses `store: false`, but its
old high-reasoning policy is now superseded and the current runner deliberately
rejects it. Regeneration uses `reasoning.effort: none` because encrypted
reasoning items are incompatible with the final-output-only collector. The
training-teacher file has not been submitted; regenerate it only after the
benchmark freeze is complete.
See `distillation.md` for the teacher, independent judge, bilingual review, and
student-training design.

The corrected provisional canary-screened mining run produced 963 sources per
direction: 900 hard KFTT rows plus 63 unambiguous Mimi UI training pairs. A
separately prepared 300-row CC BY 4.0 BTEC source-only sample raises EN→JA to
1,263 sources while JA→EN remains 963. The legacy merged 2,226-request GPT-5.6
JSONL is 5,672,274 bytes at SHA-256
`7895c05a23ebd904bc2529e9c929d50d9679a9b6dbb107f9533194ff08dcd768`.
It has not been uploaded or submitted and is likewise superseded by the
final-output-only policy. Its deterministic pre-review split is
1,146/117 train/dev for EN→JA and 867/96 for JA→EN, allowing about 42.3%
uniformly distributed rejection in the limiting direction before the 500/50
minimum is missed. All artifacts remain provisional because the final held-out
set does not yet exist; regenerate them after it is frozen and screened.

## Promotion rule

The existing evaluator requires the manifest's minimum adjudicated sample count
in each direction, a positive 95% paired-bootstrap lower bound versus Apple for
chrF++, a positive human quality delta, zero new critical meaning or safety
errors, and the declared latency/memory/size caps. The revised product gate is
strictly stronger: Apple is only a diagnostic floor, so promotion also requires
predeclared absolute bilingual-quality and critical-error thresholds and a
non-Apple failure path. Until those additions and the held-out suite are ready,
the local lane remains developer-only and the current app behavior is unchanged.

## Public stress suite and single-model research

For a larger non-claimable robustness check, generate the deterministic public
stress suite and run the exact preferred pack:

```sh
python3 scripts/translation/prepare_public_stress_suite.py \
  Research/translation/work/kftt/test.jsonl \
  Research/translation/work/alt/test.jsonl \
  Research/translation/work/tatoeba/test.jsonl \
  Research/translation/benchmark/public-stress-v1.jsonl

uv run --python 3.12 --with mlx==0.30.6 --with transformers==4.57.6 \
  --with sentencepiece --with sacremoses \
  scripts/translation/run_mlx_marian_benchmark.py \
  Research/translation/benchmark/public-stress-v1.jsonl \
  Research/translation/results/dqrd-preferred-v2-public-stress-v1.json \
  --en-ja-model Research/translation/models/elanmt-dqrd-preferred-v2-mlx-4bit-pack/en-ja \
  --ja-en-model Research/translation/models/elanmt-dqrd-preferred-v2-mlx-4bit-pack/ja-en \
  --warm-runs 0
```

This suite has 400 cases per direction from licensed human corpora but is never
promotion evidence: it has one reference, public/pretraining overlap risk, and
the wrong domain distribution. The current preferred-v2 pair scores 33.24/53.95
chrF++ on it. The companion two-teacher single-model experiments produce a
39.14 MB physical model but remain far below the specialist pair in at least one
direction.

Run Apple once per stress case, then persist the paired direction/domain report:

```sh
scripts/translation/run-apple-benchmark.sh \
  Research/translation/benchmark/public-stress-v1.jsonl \
  Research/translation/results/apple-public-stress-v1.json 0

uv run --python 3.12 --with sacrebleu==2.6.0 scripts/translation/score_translation.py \
  Research/translation/results/dqrd-preferred-v2-public-stress-v1.json \
  --compare-apple Research/translation/results/apple-public-stress-v1.json \
  --output Research/translation/results/dqrd-preferred-v2-vs-apple-public-stress-v1-score.json
```

The zero means no latency-only warm repetitions; omitted defaults to three.
Preferred-v2 leads Apple overall by +8.22/+5.54 mean sentence chrF++, but loses
the ALT news slice by -3.35/-4.87 with both 95% intervals below zero.
That domain reversal is why the public aggregate is explicitly non-claimable.

An expanded `public-stress-v2` now increases development coverage to 2,400
cases: 1,200 per direction, balanced across 400 Tatoeba conversation, 400 ALT
news, and 400 KFTT Wikipedia pairs. It is still non-claimable for the same
single-reference, public-overlap, and domain-mismatch reasons. Preferred-v3
scores 30.17/55.95 chrF++; by domain it reaches 41.40/67.76 conversation,
32.64/56.57 news, and 29.69/51.42 Wikipedia. KV-cache p95 is 50.9/59.2 ms
over the larger and more balanced sample. Compared with preferred-v2, EN→JA's
conversation slice improves by +0.17 mean sentence chrF++ (+0.04…+0.35), while
the all-domain change remains inconclusive. The original 800 rows remain frozen
for exact Apple and cross-candidate comparisons; v2 is the broader robustness
surface for subsequent fine-tuning.

Apple Translation has now been captured on the same 2,400 source rows. Its
corpus chrF++ is 26.45/54.29 and p95 is 2.133/2.314 seconds. Preferred-v3's
paired mean sentence-chrF++ deltas are +7.59 (+6.22…+8.94) EN→JA and +3.32
(+2.20…+4.47) JA→EN, while local p95 is about 41.9x/39.1x lower. The aggregate
chrF++ result is not a promotion result: the ALT news deltas are significantly
negative at -1.25 (-2.25…-0.24) and -2.84 (-3.83…-1.87).

Pinned Apache-2.0 COMET-22 strengthens the rejection. Preferred-v3 versus
Apple is -0.00537 (-0.01064…-0.00016) EN→JA and statistically inconclusive at
+0.00007 (-0.00481…+0.00508) JA→EN. News is lower by -0.02902 and -0.02078
with both intervals wholly below zero; JA→EN conversation is also lower by
-0.01183 (-0.01966…-0.00428). Only the Wikipedia slice consistently favors the
local pair. `compare_learned_metric.py` validates identical suites and pinned
metric signatures, then persists deterministic paired-bootstrap intervals:

```sh
python3 scripts/translation/compare_learned_metric.py \
  Research/translation/results/dqrd-preferred-v3-kv-cache-public-stress-v2-comet22.json \
  Research/translation/results/apple-public-stress-v2-comet22.json \
  Research/translation/results/dqrd-preferred-v3-vs-apple-public-stress-v2-comet22-comparison.json
```

### Rejected: reverse-consistency expert reranking

The next zero-weight-byte ablation keeps the source router as the first stage,
generates both its generalist and expert hypotheses, translates both hypotheses
back through the opposite-direction generalist, and lets reverse chrF++ veto
the expert. A deterministic hash split calibrated the veto margin separately by
direction: 0.5 over 287 routed-expert EN→JA cases and 0.0 over 101 JA→EN cases.
The untouched public-stress-v3 test contains 1,389 cases (718 EN→JA and 671
JA→EN). The experiment is development evidence only.

The reranker is decisively worse than the existing routed release pack. Mean
sentence-chrF++ changes by -0.439 EN→JA (95% paired-bootstrap interval
-0.728…-0.198) and -0.725 JA→EN (-1.060…-0.450). JA→EN legal loses 5.425
points on average, while EN→JA legal and professional Wikipedia also have
wholly negative intervals. Its modeled sequential p95 latency is 212.4 ms and
119.7 ms. Those times sum the two forward candidates and two reverse passes;
end-to-end peak RSS for this multi-model selector was not measured and remains
`null` in the authoritative report.

Structural evidence independently rejects it. The selector has 241 exact
critical-token mismatches versus 227 for the aligned routed baseline, and the
typed policy accepts 27 unsafe choices versus 13. The authoritative reports use
the `roundtrip-expert-reranker-v1-*-final.json` suffix; the selector test report
SHA-256 is
`046a2cb442b893c57ac0bb8e74bd4f9030c6bf98beee3f0cff1a7e368757a957`.
No Swift code or product default changes. Reverse lexical consistency is not a
safe quality estimator for this EN↔JA router.

### Rejected: relative self-likelihood expert reranking

A second zero-weight-byte ablation tests a cheaper signal. On the exact 758
cases where the source router selected an expert, the packaged generalist and
expert both run cached greedy decoding and retain mean chosen-token NLL,
including EOS. The selector first applies the exact typed-structure veto, then
keeps the expert when `generalist NLL - expert NLL` reaches a calibrated margin.
All 1,516 regenerated candidate sequences exactly match the saved reports. A
separate hash split calibrates on 286/84 expert-routed EN→JA/JA→EN cases and
selects -0.15 in both directions.

The untouched 1,412-case test again rejects output-side reranking. Relative to
the aligned source-routed pack, mean sentence-chrF++ changes by -0.491 EN→JA
(95% interval -0.763…-0.246) and -0.939 JA→EN (-1.356…-0.575). JA→EN legal
loses 6.645 points (-9.284…-4.370); EN→JA news, legal, and Wikipedia also have
negative intervals. Modeled sequential p95 grows from 58.4 to 107.2 ms EN→JA
and from 66.0 to 91.6 ms JA→EN because expert-eligible inputs require both
forward candidates.

Structural safety also worsens: exact critical-token mismatches rise from 255
to 280, and unsafe typed acceptances rise from 11 to 28. End-to-end selector RSS
is unmeasured and remains `null`; the sequential evidence runner's process RSS
is not a product-residency measurement. The evidence SHA-256 is
`4cbdd6a8e9e15ce9347163e84a14cf6b78e5ae5efb1c7a25ea4c936ac1098c1c`,
and the paired score SHA-256 is
`82996ba10eaa64d1c66a81df9ed9af00ed7475496184e397a2f685e0bd1fa6ce`.
This method is closed without Swift integration or default changes.

### Rejected: weighted source-router classifier

A source-only `LinearSVC` control predicts whether the EN→JA expert's sentence
chrF++ delta is positive. Its training samples are weighted by
`sqrt(abs(delta) + 0.1)`, and its threshold and minimum source length are chosen
only on the grouped tune split subject to non-negative canary gain. This is a
portable linear shape, but it is not serialized unless it beats the incumbent.

The classifier routes no conversation cases and has a positive delta over the
generalist on its 386-case public test: +0.309 (+0.022…+0.611). That is the
wrong comparison for replacement. Against the current canary-constrained ridge
router on the identical cases, it loses -0.123 mean sentence chrF++ with a
wholly negative interval (-0.253…-0.008), adding 43 routes and removing 8.
The incumbent source router therefore remains unchanged. The reproducible
report is
`Research/translation/results/release-clean-human-full-depth-en-ja-weighted-classifier-router-v1.json`.

### Rejected: symmetric critical-token fallback

`run_mlx_marian_moe_benchmark.py` now evaluates the exact authenticated MoE
pack directly, including translation memory, source router, cached decoder,
critical-token fallback, output decoding, plausibility checks, warm latency,
and peak RSS. It deliberately accepts the frozen 800-case source-only suite:
references remain empty, every row remains non-claimable, and the report is
blocked on independent reference consensus.

The current one-way runtime accepts 414/800 sources and fails closed on 386
critical-token mismatches. Warm p95 is 112.6/58.0 ms EN→JA/JA→EN. A symmetric
ablation tries the opposite bundled role after any initial critical-token
failure. It accepts 419/800, but attempts a fallback 429 times, changes 215
hypotheses, and raises JA→EN p95 to 108.7 ms. Public-reference deltas are
positive but inconclusive; the frozen source-only rows cannot supply quality
evidence. The five structural recoveries are too sparse to justify a Swift
path, so the ablation remains research-only.

```sh
TOKENIZERS_PARALLELISM=false uv run --python 3.12 \
  --with mlx==0.30.6 --with transformers==4.40.2 --with tokenizers \
  --with sentencepiece --with sacremoses \
  scripts/translation/run_mlx_marian_moe_benchmark.py \
  Research/translation/models/elanmt-release-clean-human-routed-moe-v2-memory-v2-mlx-4bit-shared-tokenizer-pack \
  Research/translation/benchmark/automated-claim-v1.sources.jsonl \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-symmetric-critical-fallback-automated-source-v1.json \
  --warm-runs 3 --symmetric-critical-fallback

python3 scripts/translation/compare_source_only_moe_runtime.py \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-automated-source-v1.json \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-symmetric-critical-fallback-automated-source-v1.json \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-symmetric-critical-fallback-automated-source-v1-comparison.json
```

### Preservation-aware full-fine-tuning result

`build_critical_preservation_curriculum.py` derives a training-only curriculum
from already exposed licensed human pairs. It does not create synthetic text or
store reasoning traces. The EN→JA child is a statistically significant public
generalist win (+0.441 sentence chrF++, 95% +0.153…+0.738), but replacing the
generalist in the exact routed pack lowers frozen-source runtime acceptance
from 414/800 to 395/800. The 141,491,232-byte candidate pack is therefore
research-only and the incumbent app/default pack is unchanged.

The corrected JA→EN arm starts from the actual averaged shipping parent and
records its averaging manifest in `initial_checkpoint.lineage_manifests` and
`preservation_checkpoint.lineage_manifests`. Its public delta is inconclusive
and negative (-0.109, 95% -0.300…+0.061), so the current JA→EN model remains.
`run_mlx_marian_benchmark.py --direction en-ja|ja-en` can run bounded
single-direction checkpoint screens while still binding both declared model
manifests. Different-model source-only comparisons use
`compare_source_only_moe_candidates.py`; they never create a quality claim.

QuickMT adds a useful architecture result but not a reusable model/data result.
Its official 8e/2d EN→JA and 12e/2d JA→EN CTranslate2 int8 pair totals
813,661,710 bytes and scores 26.54/57.98 on the canary at roughly 76/77 ms warm
p95, with 1.78 GB peak RSS. EN→JA fails the incumbent gate. The 63.3M-row
training mixture has no dataset license or row-level rights/generator inventory
and mixes public benchmark splits, so QuickMT remains only evidence to train a
shallow decoder from initialization. A progressive 6→5 ElanMT pruning arm also
fails: after 250 recovery steps it reaches 27.066 licensed-development chrF++
versus 30.609 for the intact parent. Together with the earlier 6→4 and 6→2
arms, this closes post-hoc decoder deletion.

The next no-runtime-cost control is NSL-MT-inspired negative-space adaptation.
Mimi deterministically corrupts licensed targets with number, unit, placeholder,
URL, negation, omission, and duplication errors, but uses those strings only as
negative evidence. Correct references remain the sole positive targets. A
bounded token-local unlikelihood term penalizes the first divergent corrupted
token under the correct prefix, avoiding the reviewed paper's unbounded
whole-sequence shortcut. The pipeline is hash-bound, requires no human reviewer,
and contains no free-form synthetic translations or reasoning traces.

That control is now executed. Alpha 0.3 barely changes the generated-negative
metric. At alpha 3.0, EN→JA lowers mean bad-token probability by about 3.6%
relative and gains only +0.048 full-precision licensed-dev chrF++, then regresses
to 29.996 after exact q4 conversion. JA→EN lowers bad-token probability about
2.0% and gains +0.031 full-precision chrF++, but all six q4 canary token
sequences are exactly unchanged from the incumbent. The arm stops before
stress, learned metrics, Swift parity, or integration; Mimi's default remains
unchanged.

### Typed temporal and n-best rescue result

The frozen source-only failure taxonomy is reproducible and does not weaken the
runtime guard:

```sh
PYTHONPATH=scripts/translation python3 \
  scripts/translation/analyze_source_only_critical_failures.py \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-current-runtime-rerun-automated-source-v1.json \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-current-runtime-rerun-automated-source-v1-critical-taxonomy.json

uv run --python 3.12 --with mlx==0.30.6 \
  --with transformers==4.57.6 --with tokenizers \
  scripts/translation/evaluate_marian_typed_nbest_rescue.py \
  Research/translation/models/elanmt-release-clean-human-routed-moe-v2-memory-v2-mlx-4bit-shared-tokenizer-pack \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-current-runtime-rerun-automated-source-v1.json \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-typed-nbest-rescue-automated-source-v1.json \
  --beam-size 4 --max-tokens 192
```

The taxonomy attributes 290/386 strict failures to candidate date/time surface
normalization and 63 to concrete unsafe structural changes. The hardened arm
remains rejected: public reference coverage is too small, and beam-4 adds only
nine source-only typed candidates at about 0.46/0.34-second triggered p95.
Typed preservation alone does not protect negation or overall meaning. Do not
port it to Swift, enable constrained decoding, alter the release contract, or
change Mimi's default from this evidence.

### Exact target-vocabulary shortlist result

Build the tokenizer-only shortlist, run the paired canaries, and apply the
explicit stop gate with:

```sh
PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with transformers==4.57.6 --with tokenizers \
  scripts/translation/build_marian_target_shortlist.py \
  Research/translation/models/elanmt-release-clean-human-routed-moe-v2-memory-v2-mlx-4bit-shared-tokenizer-pack/shared/tokenizer.json \
  Research/translation/work/marian-tokenizer-target-shortlist-v1.json

# Run run_mlx_marian_moe_benchmark.py once without, then once with:
# --target-shortlist Research/translation/work/marian-tokenizer-target-shortlist-v1.json

python3 scripts/translation/compare_marian_target_shortlist.py \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-current-runtime-shortlist-control-canary.json \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-target-shortlist-canary.json \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-target-shortlist-canary-comparison.json
```

The candidate is exact across all 12 canary outputs and routes, but fails both
continuation conditions. Its best directional p95 gain is only 1.08% versus the
required 5%, while peak RSS rises 23,953,408 bytes (+8.38%). The comparison
report SHA-256 is
`c13035d0ed4a90673aa9b559e498f3414d13a61c93f994e1e258f8d31cf06134`.
Stop at canary: do not run the full source audit, port this path to Swift, or
change Mimi's current default.

### MLX 0.31.2 runtime-upgrade result

The exact canary was repeated as MLX 0.30.6 → 0.31.2 → 0.30.6 with 30 warm
runs per case. MLX 0.31.2 changes two EN→JA token sequences, regresses warm
p50/p95 against the following control in both directions, and increases peak RSS
by 2,965,504 bytes. The hash-bound comparison is
`94a0b03c8f305caa275cf6ce141679331d4e17dc0ed5877fe7515fe1c7682395`.
The runtime upgrade is rejected at canary; keep MLX 0.30.6 pinned for the current
Marian pack.

### Packed attention-projection result

The opt-in `--packed-attention-projections` runtime concatenates q4 self-QKV and
cross-KV projections without changing model files. Projection outputs and all 12
canary token sequences are exact. The isolated gate passes, but full translation
does not: all four direction/percentile latency deltas regress by 0.74–6.86%.
Peak RSS improves by 23,101,440 bytes, but the real-time continuation gate is
latency, not memory alone. The rejected comparison SHA-256 is
`6488e6fa1b45bd5049fd1601b6d3c1600ac36a7d8cdc522f72219e542ac384ac`.
Keep the path research-only and stop before full-suite or Swift work.

### Stable-block `mx.compile` result

Fixed-shape decoder residual/LayerNorm and FFN subgraphs are bit-exact when
compiled, but improve only 0.19% and 4.77% respectively. The isolated report is
`81f0c0d57a70b20b26030e551339cde28574d268b796ce6daa94389da4f35d5f`.
The 10% continuation floor is not met, so no cache-dependent compile path,
custom Metal fusion, full canary, Swift port, or default change follows.

### Same-depth SSRU speed-proxy result

Run the pre-training stop gate against the authenticated EN→JA generalist with:

```sh
PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with mlx==0.30.6 --with transformers==4.57.6 --with tokenizers \
  scripts/translation/benchmark_marian_ssru_proxy.py \
  Research/translation/models/elanmt-release-clean-human-routed-moe-v2-memory-v2-mlx-4bit-shared-tokenizer-pack/engines/generalist-en-ja/model.safetensors \
  Research/translation/results/release-clean-human-routed-moe-v2-memory-v2-ssru-layer-proxy.json
```

The proxy retains decoder depth, cross-attention, and FFNs while replacing only
self-attention with the paper's packed two-projection SSRU recurrence. Median
layer time improves 5.2267% across seven alternating 1,000-iteration blocks,
below the preregistered 10% continuation floor. The report SHA-256 is
`ce7f1d9257696150ad4706d4ab6d4ce5385ebcf62859ae4044cbc70675aa93f7`.
Stop before SSRU student training, q4 quality work, Swift porting, packaging, or
default changes. This does not reject a purpose-pretrained deep-encoder/one-
decoder architecture; it rejects spending the next training budget on the
same-depth mutation of Mimi's current incumbent.
