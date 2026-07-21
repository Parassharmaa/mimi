# Routed Marian release audit

This directory contains generated, hash-bound provenance and attribution
artifacts. It is a compliance-engineering record, not legal advice or approval
to distribute the model.

## Current candidate

There are now two deliberately separate development records:

- The best measured research incumbent remains
  `elanmt-release-clean-human-routed-moe-v3-memory-v2-mlx-4bit-shared-tokenizer-pack`.
  Its exact legal memory improves measured quality but is training-only and
  promotion-ineligible, and its EN→JA expert carries 256 provisional ancestor
  rows. It cannot ship.
- The lineage-clean, memory-free candidate is
  `elanmt-release-clean-human-only-routed-moe-v4-mlx-4bit-shared-tokenizer-pack`.
  It is 140,875,806 bytes (manifest SHA-256
  `8fd2dd3ecf39ab86ff535e8a2f77576390898fb92a7600a923efd90dbed8704e`)
  and has complete model/conversion provenance with zero promotion-excluded
  rows. Its release contract hashes to
  `b56eb15418d8661626fdfc428611783671eabf2c24ba88456e2d23165fa0af0b`.
  It remains blocked because four dataset policy manifests do not authorize
  promotion, final license/app review and sealed 400+400 evaluation are
  pending, and it adds one strict critical-token failure on public-v3.

Neither record is approved for staging, distribution, app integration, or a
default change.

The primary-source distribution review for the lineage-clean v4 pack is in
`elanmt-release-clean-human-only-routed-moe-v4-audit-v1/DISTRIBUTION-REVIEW-2026-07-21.md`;
its SHA-256 is
`0c5f390d55e935b266f7110ac2fd2927d74c2d93966db00cad34c1ce126e8ca5`.
The matching machine-readable rights/channel decision is
`distribution-review.json`, SHA-256
`6b838a6479b4d9022124f39d258fe0203d17fc825bc617aece9ffbe06451af25`.
It verifies the ALT parallel-text versus noncommercial treebank boundary,
preserves all corpus obligations, and distinguishes a conditionally feasible
direct notarized release from a Mac App Store release that remains blocked
pending review of EULA and technological-measure compatibility. It does not
change either release flag.

The metadata-only portable clone is 140,875,791 bytes (manifest SHA-256
`71f330302559a0948c1b35f6def2d6107b4b57728cf53e1e99e0279141d84e79`).
Its repository-relative release contract and inventory hash to
`3d8ccfdfb95d2365de21a0912bd2c370c8ebf6061c903d758798720a8dc0a8f6`
and
`fb62b633e56c5c2fc5aa983e71f8475db6da2588c99479fae6031dcd5b8f01e7`.
All model, tokenizer, router, and tokenizer-configuration payload hashes match
the source pack; only metadata manifests changed. Development staging retains
`blocked-development-only` authorization.
The exact source/portable canary comparison matches all 12 routes, hypotheses,
generated token IDs, guard/fallback decisions, and failure reasons (SHA-256
`885502fa4d31b595ccc76b74993ff24ad3e4d40a88d6f5965c6172c4544d1b58`).
The native Swift loader validates the pack, and cold role smokes pass for both
directions' generalist and expert engines.

The immutable portable-v2 record has a license-complete successor:
`elanmt-release-clean-human-only-routed-moe-v4-portable-licensed-audit-v3`.
It binds eight official source documents (413,215 bytes) and a deterministic
license manifest into the release contract. The complete license directory is
418,162 bytes. Its contract, portable inventory, and license-manifest SHA-256
values are
`1da6991a7f6640b749301b04d73115c57637d1cbd4319bd275f093f46ab348f1`,
`55d33deb8f6302152c934b601e3e55c7243cddc592bbd7c4a52f8b39befa9d04`,
and
`54c9334674e45dd3ba3ea55845fc0b1801a23dbe6b62a19b77eeb22eb4728bac`.
The official Japanese PDL PDF controls over the bundled English reference. The
Japanese Law Translation HTML snapshot replaces only its randomized
anti-CSRF-token value and records that transform in the manifest. This closes
the public-license-file blocker but not the trained-weight compatibility,
Mac App Store, dataset-policy, sealed-quality, or critical-failure gates.

Reproduce the successor audit and its explicitly blocked development staging:

```sh
python3 scripts/translation/freeze_translation_license_bundle.py \
  Research/translation/release/elanmt-release-clean-human-only-routed-moe-v4-portable-audit-v2 \
  Research/translation/release/elanmt-release-clean-human-only-routed-moe-v4-portable-licensed-audit-v3 \
  --repository-root . --freeze-date 2026-07-21

python3 scripts/translation/stage_translation_release_artifacts.py \
  Research/translation/models/elanmt-release-clean-human-only-routed-moe-v4-portable-mlx-4bit-shared-tokenizer-pack \
  Research/translation/release/elanmt-release-clean-human-only-routed-moe-v4-portable-licensed-audit-v3 \
  Research/translation/release/elanmt-release-clean-human-only-routed-moe-v4-portable-licensed-app-resources-development-v2 \
  --allow-blocked-development
```

Default staging still refuses this candidate. The development-only output
authenticates all 13 staged records and retains
`doesNotAuthorizeDistribution=true`, `doesNotAuthorizeAppIntegration=true`,
and `modelPromotionEligible=false`.

The current development candidate is
`elanmt-release-clean-human-routed-moe-v3-memory-v2-mlx-4bit-shared-tokenizer-pack`:

- model-pack bytes: `141492266`
- model-pack manifest SHA-256:
  `deda4fe0d6c9ca3fd069ca99f7c45a42b5bab1fcfda3ff861cb3a0bdee40c2ee`
- release-contract SHA-256:
  `8c4baec93d53f499914201f3a42179eb7a6071490e86e5cda2fd952946db4d45`
- attribution/provenance bytes: `892561`
- combined bytes: `142384827`
- bytes below the preferred ceiling: `7615173`
- neural weight-training promotion-excluded rows: `256`
- runtime-memory source rows marked training-only/promotion-ineligible:
  `408814`
- status: `blocked-training-only-runtime-memory-and-final-review`
- provenance complete: `true`

Public-v3 development replay improves mean sentence chrF++ over the exact
human-only generalists by +0.837 EN→JA and +1.295 JA→EN, with positive paired
95% lower bounds. A 6,179-entry human legal exact memory repairs the known
JA→EN heading `（立入調査等）` to “(On-site Inspections)”; its separate untouched-
validation retrieval gains are +13.591 EN→JA on 9 matches and +23.962 JA→EN on
213 matches. Public-v3 has only seven memory hits, all improved. Swift matches
Python on 2,800/2,800 memory lookup decisions and hypotheses.

This does not authorize release. The complete final-output exact critical-token
audit finds 498/2,800 mismatches after fixing sentence-final-number handling:
237 EN→JA and 261 JA→EN. A broad word/kanji/era typed relaxation is rejected;
it admits 27 cases whose independent human-reference signature disagrees. The
separately gated single-explicit-digit-percentage arm admits one reference-
validated case and zero disagreements, while every other typed arm stays off.
The native guard now applies to every neural path and passes executable failure
smokes in both directions. In the explicit local opt-in lane, runtime failure
preserves completed local results, exposes retry, and never enables Apple;
live partial captions show source text instead of calling Apple. With the opt-in
disabled, the existing app default is unchanged. The memory source rows are licensed human data, but
their existing training-only/promotion-ineligible flags are preserved as a
release blocker. Recursive lineage also finds 256 provisional local-teacher
consensus rows marked training-only and promotion-ineligible in an ancestor of
the EN→JA generalist; removing the exact memory alone therefore cannot make the
current neural pack promotable.

The authenticated Apple diagnostic intersection contains 647 cases per
direction. Although routed chrF++ has positive paired lower bounds and local
p95 is roughly 39×/35× lower, pinned COMET-22 is inconclusive overall and
significantly worse on news in both directions plus JA→EN conversation. This is
post-hoc, claim-ineligible evidence and leaves the quality block in force.

The memory-free routed-v2 and superseded routed-v3 audits are intentionally
retained. The former has release-clean neural lineage; the latter proves why its
automated-data interpolation cannot ship.

## Reproduce

Recreate the four MLX engines with the pinned converter and package them with
all four selected full-precision lineages:

```sh
MIMI_CONVERSION_WORK=$(mktemp -d /tmp/mimi-marian-conversion.XXXXXX)

PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with mlx==0.30.6 --with transformers==4.57.6 --with tokenizers \
  --with sentencepiece --with protobuf \
  python3 scripts/translation/prepare_elanmt_mlx.py \
  Research/translation/models/elanmt-conversational-control-en-ja \
  "$MIMI_CONVERSION_WORK/generalists/en-ja" \
  --repository Mitsua/elan-mt-bt-en-ja \
  --revision 02c48e7031386cd2d41974b0ff1aaf52f010c5fa \
  --direction en-ja --bits 4 --group-size 64

PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with mlx==0.30.6 --with transformers==4.57.6 --with tokenizers \
  --with sentencepiece --with protobuf \
  python3 scripts/translation/prepare_elanmt_mlx.py \
  Research/translation/models/elanmt-licensed-unified-regularized-ja-en-v1-avg3 \
  "$MIMI_CONVERSION_WORK/generalists/ja-en" \
  --repository Mitsua/elan-mt-bt-ja-en \
  --revision 539f80eb05306e27a166b45e4264c7fa2eb4de97 \
  --direction ja-en --bits 4 --group-size 64

PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with mlx==0.30.6 --with transformers==4.57.6 --with tokenizers \
  --with sentencepiece --with protobuf \
  python3 scripts/translation/prepare_elanmt_mlx.py \
  Research/translation/models/elanmt-release-clean-full-depth-en-ja-v1 \
  "$MIMI_CONVERSION_WORK/formal-en-ja" \
  --repository Mitsua/elan-mt-bt-en-ja \
  --revision 02c48e7031386cd2d41974b0ff1aaf52f010c5fa \
  --direction en-ja --bits 4 --group-size 64

PYTHONPATH=scripts/translation uv run --python 3.12 \
  --with mlx==0.30.6 --with transformers==4.57.6 --with tokenizers \
  --with sentencepiece --with protobuf \
  python3 scripts/translation/prepare_elanmt_mlx.py \
  Research/translation/models/elanmt-release-clean-legal-specialist-ja-en-v1 \
  "$MIMI_CONVERSION_WORK/legal-ja-en" \
  --repository Mitsua/elan-mt-bt-ja-en \
  --revision 539f80eb05306e27a166b45e4264c7fa2eb4de97 \
  --direction ja-en --bits 4 --group-size 64

python3 scripts/translation/package_elanmt_mlx_experts.py \
  "$MIMI_CONVERSION_WORK/generalists" \
  "$MIMI_CONVERSION_WORK/formal-en-ja" \
  Research/translation/models/release-clean-human-full-depth-en-ja-expert-router-v1.json \
  "$MIMI_CONVERSION_WORK/legal-ja-en" \
  Research/translation/models/release-clean-legal-specialist-ja-en-legal-router-v1.json \
  "$MIMI_CONVERSION_WORK/routed-v3" \
  --en-ja-generalist-lineage \
    Research/translation/models/elanmt-conversational-control-en-ja \
  --ja-en-generalist-lineage \
    Research/translation/models/elanmt-licensed-unified-regularized-ja-en-v1-avg3/mimi_checkpoint_averaging_manifest.json \
  --formal-en-ja-lineage \
    Research/translation/models/elanmt-release-clean-full-depth-en-ja-v1 \
  --legal-ja-en-lineage \
    Research/translation/models/elanmt-release-clean-legal-specialist-ja-en-v1
```

Build the deterministic human-only exact memory and clone-bind it into the
development pack:

```sh
uv run --python 3.12 --with sacrebleu==2.6.0 \
  scripts/translation/build_exact_translation_memory.py \
  Research/translation/work/japanese-law-translation-finalized-v1/train.jsonl \
  Research/translation/work/exact-translation-memory-v2/memory.json \
  Research/translation/work/exact-translation-memory-v2/audit.json.gz

python3 scripts/translation/package_marian_translation_memory.py \
  "$MIMI_CONVERSION_WORK/routed-v3" \
  Research/translation/work/exact-translation-memory-v2/memory.json \
  "$MIMI_CONVERSION_WORK/routed-v3-memory"

python3 scripts/translation/deduplicate_marian_moe_tokenizer.py \
  "$MIMI_CONVERSION_WORK/routed-v3-memory" \
  Research/translation/models/elanmt-release-clean-human-routed-moe-v3-memory-v2-mlx-4bit-shared-tokenizer-pack
```

Then generate the recursive contract and notices:

```sh
python3 scripts/translation/build_marian_release_contract.py \
  Research/translation/models/elanmt-release-clean-human-routed-moe-v3-memory-v2-mlx-4bit-shared-tokenizer-pack \
  Research/translation/models/elanmt-conversational-control-en-ja \
  Research/translation/models/elanmt-licensed-unified-regularized-ja-en-v1-avg3/mimi_checkpoint_averaging_manifest.json \
  Research/translation/release/elanmt-release-clean-human-routed-moe-v3-memory-v2-audit-v3 \
  --jlt-access-date 2026-07-18 \
  --translation-memory-audit \
    Research/translation/work/exact-translation-memory-v2/audit.json.gz \
  --translation-memory-training-data \
    Research/translation/work/japanese-law-translation-finalized-v1/train.jsonl \
  --formal-en-ja-lineage \
    Research/translation/models/elanmt-release-clean-full-depth-en-ja-v1 \
  --legal-ja-en-lineage \
    Research/translation/models/elanmt-release-clean-legal-specialist-ja-en-v1
```

The builder authenticates the pack file table, generalist transformations,
full-precision checkpoints, training manifests, initial/preservation
checkpoints, dataset files, row counts, and third-party attribution fields. It
also authenticates the memory runtime, complete compressed audit, all 408,814
source rows, their license/provenance fields, and their unchanged training-only
flags. It fails closed on a hash mismatch, missing attribution, unsupported
lineage, or missing memory evidence.

The release contract computes `provenanceComplete`; it never hardcodes it.
Selected expert checkpoints must be supplied explicitly and match both their
full-precision weight hashes and training-manifest hashes. Each engine must also
carry a hash-bound `conversion` record joining source weights, converted MLX
weights, and the exact converter tool. Older packs without those records remain
truthfully blocked even when all of their currently declared files authenticate.
The v3 pack contains all four records. Re-conversion under MLX 0.30.6 reproduced
every incumbent `model.safetensors` SHA-256 exactly; all twelve non-manifest
payload files are byte-identical to v2. Only authenticated metadata changed.
The rebuilt canary preserves 12/12 hypotheses, output token IDs, routes, guards,
and failure results; its report SHA-256 is
`abf72b7608e0bd06b8bfdc29fa1a7efa39e881764f0ac600fd03d6fbb8aebfd5`.

### Reproduce the lineage-clean v4 candidate

Retrain the EN→JA expert from the human-only conversational checkpoint with the
same checkpoint used for preservation:

```sh
TOKENIZERS_PARALLELISM=false PYTHONPATH=scripts/translation \
uv run --python 3.12 --with torch==2.13.0 \
  --with transformers==4.57.6 --with sacrebleu==2.6.0 \
  --with sentencepiece --with protobuf --with numpy \
  python3 scripts/translation/train_marian_distillation.py \
  Research/translation/work/release-clean-full-depth-en-ja-v1 \
  Research/translation/models/elanmt-release-clean-human-only-full-depth-en-ja-v2 \
  --direction en-ja --repository Mitsua/elan-mt-bt-en-ja \
  --revision 02c48e7031386cd2d41974b0ff1aaf52f010c5fa \
  --initial-checkpoint Research/translation/models/elanmt-conversational-control-en-ja \
  --preservation-checkpoint Research/translation/models/elanmt-conversational-control-en-ja \
  --device mps --seed 314159 --batch-size 8 --gradient-accumulation 2 \
  --max-steps 1000 --learning-rate 0.000002 --weight-decay 0.01 \
  --warmup-steps 50 --evaluation-steps 250 \
  --max-source-tokens 128 --max-target-tokens 128 \
  --frozen-base-kl-weight 0.25 --l2-to-base-weight 0.00001 \
  --domain-loss-weight-start 1.0 --domain-loss-weight-end 1.0 \
  --curriculum-ramp-steps 1000 \
  --preservation-origin human-kftt-replay \
  --preservation-origin mimi-shipped-ui-pair \
  --training-description \
    'supervised adaptation on licensed human-authored and project-owned parallel references; no synthetic targets; no reasoning traces'
```

Run the four pinned MLX conversion commands above, substituting
`elanmt-release-clean-human-only-full-depth-en-ja-v2` for the formal EN→JA
source. Package it with that same path as `--formal-en-ja-lineage`, omit the
translation-memory packaging step, and deduplicate directly to
`elanmt-release-clean-human-only-routed-moe-v4-mlx-4bit-shared-tokenizer-pack`.
Then build the contract without either translation-memory argument:

```sh
python3 scripts/translation/build_marian_release_contract.py \
  Research/translation/models/elanmt-release-clean-human-only-routed-moe-v4-mlx-4bit-shared-tokenizer-pack \
  Research/translation/models/elanmt-conversational-control-en-ja \
  Research/translation/models/elanmt-licensed-unified-regularized-ja-en-v1-avg3/mimi_checkpoint_averaging_manifest.json \
  Research/translation/release/elanmt-release-clean-human-only-routed-moe-v4-audit-v1 \
  --jlt-access-date 2026-07-18 \
  --formal-en-ja-lineage \
    Research/translation/models/elanmt-release-clean-human-only-full-depth-en-ja-v2 \
  --legal-ja-en-lineage \
    Research/translation/models/elanmt-release-clean-legal-specialist-ja-en-v1
```

## Distribution checklist

- ElanMT-derived weights: preserve creator attribution, indicate Mimi's
  modifications, provide the adapted material under CC BY-SA 4.0 or a permitted
  compatible license, include the license, and add no effective downstream
  restriction. See the
  [CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en).
- KFTT-derived material: retain the NICT notice and complete CC BY-SA 3.0
  share-alike review. See the
  [CC BY-SA 3.0 legal code](https://creativecommons.org/licenses/by-sa/3.0/legalcode.en).
- Tatoeba/ManyThings: ship the deterministic compressed sidecar containing the
  retained sentence IDs, contributor strings, source links, and per-row license
  notices.
- NICT ALT: retain NICT/Wikinews attribution and the requested paper citation.
- Japanese Law Translation Database: cite the source, state that Mimi edited
  the content, retain the unofficial/tentative-translation disclaimer, and
  include the access date. See the
  [database terms](https://www.japaneselawtranslation.go.jp/en/index/terms) and
  [Japan Public Data License 1.0](https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0).
- Mimi project-owned rows: retain the authenticated source revision and make
  sure the app's own licensing permits the intended redistribution.
- Final macOS packaging review: verify that App Store terms, code signing,
  archive layout, and any additional EULA do not impose an incompatible
  restriction on the share-alike material.

CC BY-SA uses attribution, indication-of-changes, share-alike, license-notice,
and no-additional-restrictions obligations rather than a GPL-style source-code
offer phrase. The final reviewer must still decide what form of adapted model
material and transformation information must accompany the app. The present
plan is conservative: ship the exact weights, manifests, notices, row-level
attribution sidecar, and this hash-bound transformation record together.

## Hash-bound app resources

The technical metadata gap is now fail-closed even though legal and quality
approval remain open. `stage_translation_release_artifacts.py` authenticates the exact
model-manifest digest and directory byte count against `release-contract.json`,
verifies every declared notice/sidecar, rejects symlinks and path traversal,
and writes a staged manifest. The incumbent app-resource artifact is:

- directory:
  `elanmt-release-clean-human-routed-moe-v3-memory-v2-app-resources-v4`
- staged-manifest SHA-256:
  `e78ff01d2a85a2c76e41580afd9ff7b27ad7584d1aff765afe54381676d359a5`
- authenticated model-manifest SHA-256:
  `deda4fe0d6c9ca3fd069ca99f7c45a42b5bab1fcfda3ff861cb3a0bdee40c2ee`
- authenticated release-contract SHA-256:
  `8c4baec93d53f499914201f3a42179eb7a6071490e86e5cda2fd952946db4d45`
- authorization: `blocked-development-only`
- experimental local only: `true`
- provenance complete: `true`

Experimental app builds now require both variables and stage the notices under
`Contents/Resources/TranslationLicenses`. The build scripts pass an explicit
`--allow-blocked-development` flag; the staged manifest remains labeled
`blocked-development-only` and `experimentalLocalOnly=true`:

```sh
MIMI_PACKAGE_EXPERIMENTAL_TRANSLATION=1 \
MIMI_TRANSLATION_MODEL_BUNDLE="$PWD/Research/translation/models/elanmt-release-clean-human-routed-moe-v3-memory-v2-mlx-4bit-shared-tokenizer-pack" \
MIMI_TRANSLATION_RELEASE_BUNDLE="$PWD/Research/translation/release/elanmt-release-clean-human-routed-moe-v3-memory-v2-audit-v3" \
scripts/package-app.sh
```

The archive verifier requires `--release-artifacts` for routed MoE packs and
authenticates the archived `TranslationModels` and `TranslationLicenses`
directories exactly. It refuses blocked release artifacts by default. A local
development audit may pass `--allow-blocked-development`, but its status is
`passed-development-only`, never `passed`. Pair-only benchmark fixtures retain
the legacy optional notice path.

The development-only path preserves rather than hides blocker state. The old
memory-bearing staged manifest carries promotion-ineligible memory and ancestor
rows. The lineage-clean portable v4 manifest removes those row-lineage and
absolute-path defects, but still carries dataset-policy, sealed-quality,
license-compatibility, and app-terms blockers. Default staging and archive
verification reject both states.

The upstream license path is technically coherent but remains a legal-review
item. KFTT explicitly describes its processed data as freely distributable
under CC BY-SA 3.0; Creative Commons lists later BY-SA versions as permitted
adapter licenses for BY-SA 3.0 material; Tatoeba requires sentence-author
attribution under CC BY 2.0 France, which the 9,305-record compressed sidecar
preserves. Creative Commons also states that its licenses permit AI-training
reuse when copyright permission is required, provided license conditions are
respected. None of those sources decides the jurisdiction-specific question of
whether the resulting weights are adapted material. The conservative package
therefore carries CC BY-SA 4.0 weights, all notices and row-level attribution,
the exact transformation lineage, and no release authorization.

No distribution or app integration is authorized until the contract status is
changed by a separately documented final review and every quality/runtime gate
passes.
