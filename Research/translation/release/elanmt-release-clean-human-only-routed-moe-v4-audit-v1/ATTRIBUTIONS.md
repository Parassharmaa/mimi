# Mimi local translation model attributions

This notice applies to the routed Marian MLX model pack whose manifest SHA-256 is
`8fd2dd3ecf39ab86ff535e8a2f77576390898fb92a7600a923efd90dbed8704e`. Mimi fine-tuned, interpolated, averaged, quantized, and
packaged the identified upstream weights; no upstream author or public agency
endorses Mimi or these adapted translations.

## ElanMT model weights

The four engines derive from ElanMT by the ELAN MITSUA Project / Abstract Engine,
licensed CC BY-SA 4.0. Upstream revisions and transformation hashes are recorded
in `release-contract.json`. The proposed license for the adapted model weights is
CC BY-SA 4.0; distribution remains blocked pending final compatibility and app-
distribution review.

## Kyoto Free Translation Task

The data used in this service contains English contents which is translated by
the National Institute of Information and Communications Technology (NICT) from
Japanese sentences on Wikipedia. Our use of this data is licensed by the Creative
Commons Attribution-Share-Alike License 3.0. Please refer to
http://creativecommons.org/licenses/by-sa/3.0/ or
http://alaginrc.nict.go.jp/WikiCorpus/ for details.

## NICT Asian Language Treebank

NICT Asian Language Treebank Parallel Corpus; NICT translations CC BY 4.0;
English Wikinews source text CC BY 2.5. Cite Riza et al. (2016), “Introduction of
the Asian Language Treebank.” Source: https://www2.nict.go.jp/astrec-att/member/mutiyama/ALT/

## Tatoeba via ManyThings

Retained sentence IDs, contributor names, source links, and CC BY 2.0 France
notices for 9,305 unique attributions are in
`tatoeba-attributions.jsonl.gz` (uncompressed content SHA-256
`d7b5d6e2988d2cea2f19da341e54f40c481ee320ce3b3958e40e92e703a19d63`). Source: https://www.manythings.org/anki/

## Japanese Law Translation Database System

Created by Mimi based on Japanese Law Translation Database System content
published by the Ministry of Justice, Japan, accessed 2026-07-18. PDL 1.0:
https://www.digital.go.jp/en/resources/open_data/public_data_license_v1.0
Source: https://www.japaneselawtranslation.go.jp/en/laws

Mimi filtered, normalized, selected, and converted the source content into
parallel training rows. The English translations are not official texts; only
the original Japanese laws and regulations have legal effect. The translations
are reference material, may include tentative versions, and carry the accuracy,
reliability, currency, and interpretation disclaimers in the database terms:
https://www.japaneselawtranslation.go.jp/en/index/terms


## Mimi project-owned parallel copy

Small English/Japanese UI pairs authored and shipped by Mimi are project-owned.
The authenticated source revision and source rows are recorded in
`release-contract.json`.
