# Distribution review — lineage-clean routed Marian v4

This is a compliance-engineering review, not legal advice. It records the
minimum technical release payload and the legal questions that still require a
qualified reviewer or written licensor permission. It does not authorize
distribution, app integration, or a default change.

## Exact candidate and decision

- Pack: `elanmt-release-clean-human-only-routed-moe-v4-mlx-4bit-shared-tokenizer-pack`
- Pack bytes: `140875806`
- Pack manifest SHA-256:
  `8fd2dd3ecf39ab86ff535e8a2f77576390898fb92a7600a923efd90dbed8704e`
- Release-contract SHA-256:
  `b56eb15418d8661626fdfc428611783671eabf2c24ba88456e2d23165fa0af0b`
- Attribution notice SHA-256:
  `44afd50aaa9144181ae4d8bc7dbf86fc818731fd8a1835d0727d6a89403feb28`
- Tatoeba attribution sidecar SHA-256:
  `a82507020792da4cc6fdcdfec1ef2a37fbf44ba0527fc44ca8b3ca80ed6b0b0f`
- Status: `blocked`
- `modelPromotionEligible`: `false`
- `doesNotAuthorizeDistribution`: `true`
- `doesNotAuthorizeAppIntegration`: `true`

The pack is 9,124,194 bytes below the preferred 150,000,000-byte model-pack
ceiling. Provenance is complete and the pack contains no synthetic target or
reasoning-trace lineage, but those facts do not clear its quality, dataset
policy, or legal gates.

## Portable metadata successor

The model payload now also has a repository-relative metadata clone:

- Pack:
  `elanmt-release-clean-human-only-routed-moe-v4-portable-mlx-4bit-shared-tokenizer-pack`
- Bytes: `140875791`
- Pack manifest SHA-256:
  `71f330302559a0948c1b35f6def2d6107b4b57728cf53e1e99e0279141d84e79`
- Portable release-contract SHA-256:
  `3d8ccfdfb95d2365de21a0912bd2c370c8ebf6061c903d758798720a8dc0a8f6`
- Portable inventory SHA-256:
  `fb62b633e56c5c2fc5aa983e71f8475db6da2588c99479fae6031dcd5b8f01e7`

All four `model.safetensors` files, the shared tokenizer, both routers, and all
tokenizer configurations are byte-identical to the audited source pack. Only
the formal/legal engine manifests and root manifest changed. The portable
contract removes the `portable-release-inventory-pending` blocker but retains
every dataset-policy, sealed-quality, license-compatibility, and app-terms
blocker. Its staged audit is `blocked-development-only`, never distributable.

An exact 12-case source-versus-portable MLX comparison matches every route,
hypothesis, generated token ID, guard decision, fallback decision, and failure
reason; the comparison SHA-256 is
`885502fa4d31b595ccc76b74993ff24ad3e4d40a88d6f5965c6172c4544d1b58`.
Mimi's native Swift loader validates the portable pack and cold smokes pass for
the EN→JA generalist, EN→JA expert, JA→EN generalist, and JA→EN expert. This is
metadata-equivalence evidence only; the two known canary critical-token
failures remain and no quality or release gate is cleared.

## Rights matrix

### ElanMT upstream weights

Both pinned upstream model revisions are published as CC BY-SA 4.0 by the ELAN
MITSUA Project / Abstract Engine. Mimi modified the weights through supervised
fine-tuning, checkpoint averaging where declared, 4-bit quantization, and
packaging. If the adapted weights are shared, CC BY-SA 4.0 requires retained
creator and copyright information when supplied, a license reference, an
indication of modifications, and a permitted ShareAlike license for adapted
material. It also forbids additional downstream legal or effective
technological restrictions on the licensed material.

Engineering disposition: scope CC BY-SA 4.0 explicitly to the four weight
files and associated model manifests; retain the upstream revisions and
modification record; include the license text or canonical URL, disclaimer,
and non-endorsement statement. Do not put those files under Mimi's proprietary
app EULA.

Primary sources:

- https://huggingface.co/Mitsua/elan-mt-bt-en-ja
- https://huggingface.co/Mitsua/elan-mt-bt-ja-en
- https://creativecommons.org/licenses/by-sa/4.0/legalcode.en

### Kyoto Free Translation Task

KFTT states that the original and processed parallel data are freely
distributable under CC BY-SA 3.0. The retained release notice includes the NICT
attribution and KFTT source. CC BY-SA 3.0 requires attribution, identification
of changes for adaptations, and a permitted ShareAlike license when an
adaptation is distributed.

Engineering disposition: retain the complete NICT/KFTT notice and CC BY-SA 3.0
reference. The license does not contain a GPL-style source-code offer
requirement; the old internal phrase `source offer` must not be presented as a
confirmed CC obligation. Whether trained weights are an adaptation of the
training sentences is jurisdiction-specific and unresolved here. If counsel
concludes they are, approve an exact CC BY-SA 3.0/4.0 compatibility formulation
or obtain written permission from the licensor before release.

Primary sources:

- https://www.phontron.com/kftt/
- https://creativecommons.org/licenses/by-sa/3.0/legalcode.en

### Tatoeba sentences via ManyThings

Tatoeba's default textual-sentence license is CC BY 2.0 France. Tatoeba states
that reuse, modification, and distribution require the sentence author's
name, and the reuser is responsible for circulating each sentence with its
license. The authenticated sidecar contains 9,305 unique retained sentence-ID,
contributor, source-link, and license records.

Engineering disposition: ship `tatoeba-attributions.jsonl.gz` unchanged beside
the human-readable notice, link it from an in-app Legal/Acknowledgements view,
and verify complete row-to-sidecar coverage from the frozen source snapshot at
release time. Do not substitute corpus-level attribution for the per-row
records and do not import Tatoeba audio licensing into this review.

Primary source: https://tatoeba.org/en/terms_of_use

### NICT Asian Language Treebank parallel corpus

NICT publishes the ALT parallel-corpus translations as CC BY 4.0, states that
the English Wikinews source texts were CC BY 2.5, and requests citation of Riza
et al. (2016). The same page separately licenses the annotated English and
Japanese treebank downloads as CC BY-NC-SA 4.0. Mimi's authenticated archive
contains only the parallel sentence files `data_en.txt`, `data_ja.txt`, and
`URL.txt`; it does not use the annotated treebank ZIPs.

Engineering disposition: retain the NICT/Wikinews attribution, both license
layers, source URL, and requested paper citation. Enforce this parallel-text
boundary in future refreshes; importing the annotated treebanks would add a
noncommercial restriction and require a new review.

Primary source:
https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/

### Japanese Law Translation Database System

The site applies Japan's Public Data License 1.0 unless stated otherwise. PDL
1.0 permits commercial reuse and modification, requires source citation, and
requires edited content to name the editing entity and not appear to be
unaltered government content. The site additionally says the English
translations are unofficial reference material, may not all be finalized, and
carry no government guarantee of accuracy, reliability, or currency. The
frozen dataset used here excludes titles marked tentative, but that does not
remove the unofficial-text and no-warranty notices.

Engineering disposition: retain the Ministry of Justice source, access date
`2026-07-18`, PDL 1.0 URL, site terms URL, Mimi modification statement, and the
unofficial/reference/no-warranty disclaimer. Do not present model output as an
official government translation.

Primary sources:

- https://www.japaneselawtranslation.go.jp/en/index/terms
- https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0

### Mimi-authored rows

The project-owned UI pairs must remain linked to the authenticated repository
revision and source rows. Their use does not override any upstream model or
corpus obligation.

## Distribution-channel decision

### Direct signed and notarized macOS release

Status: `conditionally feasible, not approved`.

Before release, place the model files in a separately scoped CC BY-SA 4.0
component, bundle the complete license/notice payload, add an accessible
Legal/Acknowledgements view, exclude the model files from restrictive app-EULA
terms, and confirm that signing/notarization does not prevent recipients from
exercising the model license. A portable, repository-relative inventory must
replace the absolute worktree paths in the internal release contract.

### Mac App Store release

Status: `blocked pending qualified review or licensor permission`.

Apple's Standard EULA grants a nontransferable app license and restricts
redistribution, modification, and derivative works, while acknowledging that
open-source component terms may permit otherwise. CC BY-SA 4.0 prohibits
additional restrictions and effective technological measures that restrict
licensed rights. This review does not decide whether the App Store EULA and any
FairPlay protection can be scoped away from embedded model weights. Do not
submit this pack through the Mac App Store until the exact EULA, component
license scope, and technological-measure behavior are approved. A separately
available unencumbered copy should not be assumed to cure an incompatible
store-wrapped copy.

Primary sources:

- https://www.apple.com/legal/internet-services/itunes/dev/stdeula/
- https://developer.apple.com/app-store/review/guidelines/
- https://creativecommons.org/faq/#can-i-use-effective-technological-measures-such-as-drm-when-i-share-cc-licensed-material

## Required release payload

A public candidate must contain, and the portable inventory must hash-bind:

1. the exact 140,875,806-byte model pack;
2. `release-contract.json` with repository-relative paths only;
3. `ATTRIBUTIONS.md`;
4. `tatoeba-attributions.jsonl.gz`;
5. canonical or frozen copies of CC BY-SA 4.0, CC BY-SA 3.0, CC BY 4.0,
   CC BY 2.5, CC BY 2.0 France, PDL 1.0, and the Japanese Law Translation
   site-specific terms;
6. a model-license notice that explicitly scopes CC BY-SA 4.0 to the weight
   files and records Mimi's modifications;
7. an app Legal/Acknowledgements entry that opens the notices and row-level
   sidecar without a network connection; and
8. a distribution-channel declaration covering the EULA, signing,
   notarization, archive layout, and any DRM.

## Remaining blockers

- Four authenticated dataset manifests do not authorize promotion.
- The sealed independent GPT-5.6 400+400 evaluation has not run.
- The independent structure-fallback test missed its positive paired-quality
  lower-bound gate and was rejected.
- The trained-weight relationship to CC BY-SA training text is unresolved.
- The Mac App Store EULA/technological-measure interaction is unresolved.
- The public license-text bundle and repository-relative portable inventory do
  not both exist; the portable inventory is complete, while the public
  license-text bundle remains pending.
- The current quality evidence does not meet the zero-critical-failure rule.

Decision: keep `releaseAuthorization=blocked`, preserve the existing default,
and use this pack only as an authenticated local research candidate.
