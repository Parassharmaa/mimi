# Mimi EN-JA model attributions

This notice applies to the Mimi EN-JA MLX development candidate. Mimi
fine-tuned, checkpoint-averaged where noted, quantized, and packaged the
identified upstream weights. No upstream author, corpus creator, contributor,
or public agency endorses Mimi or these adapted translations.

## Model license and upstream weights

The adapted weights and Mimi's model modifications are released under the
[Creative Commons Attribution-ShareAlike 4.0 International
license](https://creativecommons.org/licenses/by-sa/4.0/).

Both directional engines derive from ElanMT by the ELAN MITSUA Project /
Abstract Engine, also licensed CC BY-SA 4.0:

- EN-JA: `Mitsua/elan-mt-bt-en-ja` revision
  `02c48e7031386cd2d41974b0ff1aaf52f010c5fa`.
- JA-EN: `Mitsua/elan-mt-bt-ja-en` revision
  `539f80eb05306e27a166b45e4264c7fa2eb4de97`.

Changes made by Mimi include continued training on the openly licensed sources
below, EN-JA checkpoint averaging, MLX affine 4-bit quantization, and
bidirectional packaging. Exact transformation and weight hashes are recorded in
the repository manifests.

## Kyoto Free Translation Task

The data used in this model contains English content translated by the National
Institute of Information and Communications Technology (NICT) from Japanese
sentences on Wikipedia. Use of this data is licensed under the [Creative
Commons Attribution-ShareAlike 3.0
license](https://creativecommons.org/licenses/by-sa/3.0/). See the
[Kyoto Free Translation Task](https://www.phontron.com/kftt/) and the original
NICT WikiCorpus notice for details.

## NICT Asian Language Treebank

NICT Asian Language Treebank Parallel Corpus; NICT translations are licensed
CC BY 4.0 and the underlying English Wikinews source text is licensed CC BY
2.5. Cite Riza et al. (2016), “Introduction of the Asian Language Treebank.”
Source: [NICT ALT](https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/).

## Tatoeba via ManyThings

The retained Tatoeba sentence identifiers, contributor names, source links, and
CC BY 2.0 France notices for all 9,162 unique candidate-training attributions
are in `tatoeba-attributions.jsonl.gz`.

- Source: [ManyThings Japanese-English sentence
  pairs](https://www.manythings.org/anki/)
- Uncompressed SHA-256:
  `a57638f5c137719be16fc845943af9d8a1c417d861da6cdc3e8307118cf05c72`
- Compressed SHA-256:
  `5036ea849a18729711f71be2628f14df065a0ce152aeaa26ff80b4f6ee6eec18`

## Japanese Law Translation Database System

Mimi used finalized Japanese Law Translation Database System content published
by the Ministry of Justice, Japan, accessed 2026-07-18, under [Public Data
License 1.0](https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0).
Mimi filtered, normalized, selected, and converted the content into parallel
training rows.

Source: [Japanese Law Translation Database
System](https://www.japaneselawtranslation.go.jp/en/laws).

The English translations are not official texts; only the original Japanese
laws and regulations have legal effect. The translations are reference
material and remain subject to the database's [site-specific terms and
disclaimers](https://www.japaneselawtranslation.go.jp/en/index/terms).

## Mimi project-owned parallel copy

Small English/Japanese UI pairs authored for Mimi come from the
[Mimi repository](https://github.com/Parassharmaa/mimi) at revision
`bf55ada4b70136f881a30a020d2acb2d37816ace`. The repository is MIT-licensed.
