# Incumbent translation-pack compliance audit — 2026-07-21

This is a compliance-engineering audit, not legal advice. It deliberately does
not decide whether machine-learning weights are an adaptation of training text
under any jurisdiction. That unresolved classification controls whether some
dataset share-alike terms attach to the trained weights, so the pack remains
blocked until qualified review or explicit licensor permission resolves it.

## Exact artifact audited

- Pack:
  `elanmt-release-clean-human-routed-moe-v2-memory-v2-mlx-4bit-shared-tokenizer-pack`
- Bytes: `141488564`
- Pack manifest SHA-256:
  `75ae72c87f0f3607571e6c5bbd9ccfd99d129d880ff6daf2e5559341248cd6db`
- Matching release-contract SHA-256:
  `b30a1cb5dccff8e750a84652e8b94a6d844ca1d66a9d47d443369d8ddd03647c`
- Staged app-resource manifest SHA-256:
  `c63d91def14e68b921665ef8a5e85ed206f82310ff34b52053d745407fd3d1ea`
- Contract status: `blocked-training-only-runtime-memory-and-final-review`
- Contract flags: `modelPromotionEligible=false` and
  `doesNotAuthorizeDistribution=true`

Every declared pack/release file and every referenced local dataset, manifest,
model, and memory artifact currently exists and matches its recorded hash. The
release bundle includes 9,305 unique Tatoeba attribution records. However, the
release contract omits both selected expert checkpoint lineages and the exact
full-precision-to-MLX conversion record while hardcoding
`provenanceComplete=true`. The hashes establish integrity for what is declared;
they do not establish complete transformation provenance, a legal conclusion,
or authorization to distribute.

## Engineering contradictions found

- The formal EN→JA and legal JA→EN engine records have `releaseLineage=null`.
  The selected full-precision expert checkpoints are independently present and
  match the pack's source-weight hashes, but their training manifests are absent
  from the release contract. The missing manifest SHA-256 values are
  `a34dbada004176c83fe9efb8ac0875bcc1166af393dfa084a448b2fcc018dd85`
  and `6d6aa771366cc1dedfbfa6d41ab1fe0c6867ae1b5872fa3eb4b7ae6e144de516`.
- The pack, contract, and staged resources use different single-string blocker
  statuses. These should become a unioned machine-readable `blockers[]` plus
  separate integrity, quality, and release-authorization fields.
- The existing release bundle does not contain full official license/terms
  snapshots. Its links and notices are evidence, but a conservative public
  bundle should add hash-bound CC BY-SA 4.0, CC BY-SA 3.0, CC BY 2.0 France,
  CC BY 4.0, CC BY 2.5, PDL 1.0, and Japanese Law Translation terms files.
- The machine-readable rights count collapses ALT to CC BY 4.0 even though the
  English Wikinews source layer is CC BY 2.5, and
  `PDL-1.0-compatible-CC-BY-4.0` is a project label rather than a standard
  license identifier. Both rights layers need an explicit rights matrix.
- The original stager and archive verifier authenticated hashes but did not
  enforce the no-distribution/no-integration flags. This worktree now rejects
  blocked staging and archive verification by default. The explicit
  `--allow-blocked-development` path produces only
  `releaseAuthorization=blocked-development-only`,
  `experimentalLocalOnly=true`, and `passed-development-only` verification.
  The regenerated staged-manifest SHA-256 is
  `bed736f3b6b6127d6275cd1a9baac4386afc445231fb904f3c7755ffe890854a`.

## Post-audit fail-closed remediation

The builder now accepts both selected expert checkpoints explicitly, verifies
their full-precision weights and training-manifest hashes, and recursively
follows training checkpoints, interpolation manifests, and checkpoint-averaging
manifests. The regenerated audit-v2 contract authenticates all four selected
engine lineages, eight training manifests, two transformation manifests, twelve
dataset files, and 9,305 Tatoeba attribution records. Its SHA-256 is
`ec7769e1b45252566f8515c8bfd94095f9ee3b5c3946d165c010c33f6a4aa2c9`.

That stronger traversal finds an additional blocker the earlier shallow audit
missed: 256 `strict-local-teacher-consensus-provisional` ancestor rows are
marked both `training_only=true` and `promotion_eligible=false`. It also finds
that none of the four engine manifests contains a hash-bound record joining the
selected full-precision input, converted MLX output, and exact converter tool.
The contract therefore computes `provenanceComplete=false`; it no longer
hardcodes a successful value.

Development staging can preserve this incomplete state only with the explicit
blocked-development flag. The v3 staged manifest has SHA-256
`4add7e677378965d7dc5e4398b5bb01c1acbbc2d7125485c97acf243f9b77601`,
sets `provenanceComplete=false`, lists the complete contract/staging blocker
union, and remains `blocked-development-only`. Normal staging and distribution
verification still fail closed.

## Deterministic conversion-lineage closure

The four selected full-precision checkpoints were re-converted with the pinned
MLX 0.30.6 path after adding an authenticated conversion record to every engine
manifest. Each record binds the source-weight SHA-256, converted-weight SHA-256,
4-bit/group-64/float16 policy, exact converter path and SHA-256, Python version,
and MLX/tokenizers/SentencePiece versions. All four resulting
`model.safetensors` files are byte-identical to the incumbent; all twelve
non-manifest pack files match v2 exactly.

The resulting v3 pack is 141,492,266 bytes and has manifest SHA-256
`deda4fe0d6c9ca3fd069ca99f7c45a42b5bab1fcfda3ff861cb3a0bdee40c2ee`.
Its audit-v3 release contract authenticates all four engine lineages and all
four conversions, computes `provenanceComplete=true`, and hashes to
`8c4baec93d53f499914201f3a42179eb7a6071490e86e5cda2fd952946db4d45`.
The v4 staged-development manifest hashes to
`e78ff01d2a85a2c76e41580afd9ff7b27ad7584d1aff765afe54381676d359a5`.

This closes the conversion-provenance defect without changing model behavior.
It does not clear the promotion-ineligible memory/ancestor rows, quality,
license-compatibility, portable-inventory, or app-distribution blockers.

## Primary-source obligations and current evidence

### ElanMT-derived weights — CC BY-SA 4.0

The upstream weights are labeled CC BY-SA 4.0. Creative Commons summarizes the
relevant conditions as attribution, a license link, indication of changes,
ShareAlike for adapted material, and no additional legal or technological
restrictions. The license permits commercial sharing and adaptation when its
conditions are met.

Current evidence:

- Exact upstream repositories, revisions, base-weight hashes, local
  transformations, and engine hashes are recorded.
- `ATTRIBUTIONS.md` names the ELAN MITSUA Project / Abstract Engine, identifies
  Mimi's modifications, disclaims endorsement, links the license, and proposes
  CC BY-SA 4.0 for the adapted weights.
- The pack still needs an explicit release-level model-license file or notice
  that unambiguously scopes CC BY-SA 4.0 to the redistributed weight files.
- Any Mimi EULA, archive wrapper, DRM, or store condition must be checked for
  an effective additional restriction on the licensed material.

Primary source: [CC BY-SA 4.0 deed and legal-code link](https://creativecommons.org/licenses/by-sa/4.0/deed.en).

### KFTT — CC BY-SA 3.0

The KFTT rows carry the NICT notice and CC BY-SA 3.0. Creative Commons' own
compatibility example says that when BY-SA 3.0 material is adapted and BY-SA
4.0 is applied to the adapter's contribution, both versions apply to the
adaptation to the extent their terms differ. Therefore this audit does not
assume that labeling the weights only CC BY-SA 4.0 automatically discharges the
3.0 obligations.

Current evidence:

- The required NICT/KFTT notice is present verbatim in the release notice.
- KFTT row counts, hashes, and lineage are authenticated.
- A final reviewer must decide whether trained weights are adapted material,
  and, if so, approve the exact dual-license/compatible-license formulation.
- CC BY-SA is not a GPL-style source-code license. The generated engine
  manifests' phrase “source offer” is a conservative internal review request,
  not a confirmed license-text requirement. Shipping exact weights and the
  transformation/provenance record is the current conservative plan.

Primary sources: [Creative Commons ShareAlike compatibility guidance](https://wiki.creativecommons.org/wiki/ShareAlike_compatibility) and [KFTT project page](https://www.phontron.com/kftt/).

### Tatoeba via ManyThings — per-sentence CC BY 2.0 France

Tatoeba states that text sentences use CC BY 2.0 France by default and that a
reuser must preserve the applicable license and name the author. Licenses are
per sentence and can change only prospectively, so authenticated row-level
metadata matters more than a corpus-level label.

Current evidence:

- `tatoeba-attributions.jsonl.gz` contains 9,305 unique retained sentence-ID,
  contributor, source, URL, and license records.
- The release builder fails closed when a retained Tatoeba row lacks a source
  ID or contributor identity.
- Before a real release, rerun the contract from the frozen source snapshot and
  verify that every Tatoeba-derived training row maps to the sidecar; do not use
  the Tatoeba audio licenses.

Primary source: [Tatoeba Terms of Use, sections 6.2–6.5](https://tatoeba.org/en/terms_of_use).

### NICT ALT parallel corpus and English Wikinews source

NICT says its ALT parallel-corpus translations are CC BY 4.0, the sampled
English Wikinews texts were CC BY 2.5, and users should cite the 2016 ALT paper.
The same page separately labels the English and Japanese annotated treebank
downloads CC BY-NC-SA 4.0. Mimi's licensed input is the parallel text, not those
annotated treebanks; this distinction must remain explicit to avoid importing a
noncommercial restriction accidentally.

Current evidence:

- The release notice records NICT, English Wikinews CC BY 2.5, the source URL,
  and the requested paper citation.
- The archive/member hashes bind `data_en.txt`, `data_ja.txt`, and `URL.txt` to
  the prepared parallel corpus.
- Do not substitute the separately licensed English/Japanese ALT treebank ZIPs
  without a new license audit.

Primary source: [NICT Asian Language Treebank project page](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/).

### Japanese Law Translation Database — PDL 1.0 plus site notice

PDL 1.0 permits commercial copying, transmission, translation, and other
modification. It requires source citation; edited content must identify the
editor and must not be presented as unedited government content. Users remain
responsible for third-party rights. The Japanese Law Translation site applies
PDL 1.0 unless stated otherwise and requires the unofficial, potentially
tentative translation and no-government-warranty notices.

Current evidence:

- The release notice names the Ministry of Justice source, PDL 1.0, access
  date, site terms, Mimi's editing, and the unofficial/tentative translation
  limitations.
- The exact-memory runtime and compressed audit are authenticated, but their
  408,814 source rows are explicitly `training_only=true` and
  `promotion_eligible=false`. That is an internal promotion blocker even if the
  underlying content is licensed.
- The memory must stay disabled for app integration until a new, predeclared
  evaluation authorizes it or the memory is removed from a release candidate.

Primary sources: [Japan Digital Agency PDL 1.0](https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0) and [Japanese Law Translation terms](https://www.japaneselawtranslation.go.jp/en/index/terms).

## Fail-closed release decision

The provenance and packaging mechanics are complete enough to reproduce and
audit the exact candidate. Distribution is still **not authorized** because:

1. The legal status of trained weights relative to CC-licensed training text is
   unresolved; the CC BY-SA 3.0/4.0 scope and license formulation therefore
   require qualified review or written licensor permission.
2. The exact translation memory is intentionally promotion-ineligible and may
   not be enabled or shipped as an app behavior from current evidence.
3. Quality promotion still lacks sealed 400+400 independently generated and
   filtered references, zero critical-error evidence, and the final learned-
   metric/domain gates.
4. The final distribution channel's EULA, signing, archive, and any store DRM
   must be reviewed against the no-additional-restrictions condition.
5. The staged `release-contract.json` still contains absolute local build paths.
   Its hashes are valid for internal reproduction and contamination audits, but
   a public app artifact must use a separately authenticated portable inventory
   of repository-relative paths and source URLs without user/worktree paths.

6. The public license bundle and machine-readable rights matrix must cover the
   exact weight, KFTT, Tatoeba, ALT/Wikinews, PDL, and project-owned layers.

Until all six close, keep `doesNotAuthorizeDistribution=true`, do not alter
Mimi's current default, and treat the pack only as an authenticated research
candidate.
