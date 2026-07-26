---
license: other
license_name: mixed-open-licenses
license_link: https://huggingface.co/datasets/blazeofchi/mimi-en-ja-development-v1/blob/main/LICENSES.md
language:
  - en
  - ja
task_categories:
  - translation
tags:
  - english
  - japanese
  - machine-translation
  - long-document
  - development-benchmark
pretty_name: Mimi EN-JA Development Accuracy v1
---

# Mimi EN↔JA development accuracy v1

This public dataset repository stores an openly licensed, deterministic,
non-claimable English↔Japanese development benchmark for compact local
translation models.

## Contents

- `development-accuracy-v1.jsonl`: 200 evaluation units.
- `development-accuracy-v1.segments.jsonl`: 400 segment-level inference calls.
- `development-accuracy-v1.manifest.json`: selection contract, hashes, counts,
  overlap caveats, and explicit confirmation that the sealed promotion suite was
  not touched.
- `development-accuracy-v1.results-summary.json`: aggregate model, metric,
  latency, blind-judge sensitivity, and no-promotion decision. It contains no
  raw Apple output or private judge mapping.

There are 100 units per direction: 80 sentences and 20 six-segment documents.
Domains cover conversation, human-translated news, professionally translated
Wikipedia, and ministry-published legal text. Document inference is
segment-then-join with no cross-segment context.

## License composition

The repository is marked `other` because rows retain their original mixed
licenses:

| Corpus | Cases | License |
|---|---:|---|
| Tatoeba via ManyThings | 40 | CC BY 2.0 France, per-row attribution |
| NICT ALT | 56 | CC BY 4.0; underlying English Wikinews attribution applies |
| KFTT | 52 | CC BY-SA 3.0 |
| Japanese Law Translation Database System | 52 | PDL 1.0-compatible terms |

Every row retains its source corpus, source ID, license, attribution, declared
test split, and review status. Users must preserve those notices and comply with
each source license; this card does not replace the original terms. See
`LICENSES.md` for the complete open-license notice and Mimi change statement.

## Intended use

- model development and error analysis;
- sentence and segmented-document robustness checks;
- multi-metric comparison using chrF++, BLEU, COMET, and blinded source-based
  judgments.

## Prohibited interpretation

This benchmark must not be used to claim product superiority:

- it has one reference per source segment;
- public content may overlap opaque model pretraining;
- composed documents lack independent document-level bilingual review;
- these rows may have influenced earlier research decisions;
- it is not the sealed Mimi 400+400 promotion suite.

No Apple outputs, private judge mappings, judge verdicts, secrets, or sealed
evaluation content are included in this repository.
