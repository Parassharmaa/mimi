---
license: cc-by-sa-4.0
language:
  - en
  - ja
library_name: mlx
pipeline_tag: translation
tags:
  - marian
  - mlx
  - apple-silicon
  - english
  - japanese
  - development-only
---

# Mimi EN↔JA MLX development candidate v1

This public repository stores Mimi's openly licensed 73.4 MB two-direction
Apple-Silicon translation candidate. It is a development model release, not the
model currently integrated into the app.

## Architecture

The package contains two 4-bit Marian encoder-decoder specialists behind one
bidirectional interface:

- `en-ja`: six encoder layers, six decoder layers, width 512, eight attention
  heads, FFN width 2,048, vocabulary 32,001.
- `ja-en`: the same architecture.
- Quantization: MLX affine 4-bit, group size 64, float16 compute.
- Minimal package: 73,425,808 bytes.
- Peak benchmark RSS: 210,812,928 bytes on Apple M3 Pro.

## Checkpoints

- EN→JA: averaged steps 500/750/1000 of Mimi's release-clean full-depth
  continuation from `Mitsua/elan-mt-bt-en-ja` revision
  `02c48e7031386cd2d41974b0ff1aaf52f010c5fa`.
- JA→EN: step 750 of Mimi's release-clean legal-specialist continuation from
  `Mitsua/elan-mt-bt-ja-en` revision
  `539f80eb05306e27a166b45e4264c7fa2eb4de97`.

Training used licensed human-authored or project-owned parallel targets only.
It used no synthetic targets and no reasoning traces.

## Development benchmark

The 200-case public development suite has 100 cases per direction, including 40
six-segment documents total. It is not promotion evidence.

| Direction | chrF++ | BLEU | COMET-22 | Warm segment p95 |
|---|---:|---:|---:|---:|
| EN→JA | 28.41 | 9.63 | 0.8669 | 165.4 ms |
| JA→EN | 55.19 | 30.62 | 0.8192 | 159.5 ms |

Against Mimi's shipped pair, COMET-22 changes by +0.0183 EN→JA (95% paired
interval -0.00003 to +0.0398) and +0.0367 JA→EN (+0.0189 to +0.0576).

Two blinded GPT-5.6 variants disagree on the strength of the win. The sol
variant prefers the candidate 70–44 with 86 ties and counts 47 versus 63
critical errors. Terra reports 91–85 with 24 ties and counts 14 versus 13
critical errors. The result is promising but judge-sensitive.

## Limitations

- The development suite uses public test splits with one human reference per
  segment and possible opaque pretraining overlap.
- The model has known EN→JA long-legal repetition failures.
- A global JA→EN legal specialist can trade away conversational quality; a
  router is the safer next design.
- Machine translation may omit, reverse, or hallucinate important content.
- This package has not passed Mimi's sealed 400+400 promotion suite.

## Open license and attribution

The adapted weights and Mimi's model modifications are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and retain
attribution to the ELAN MITSUA Project / Abstract Engine. You may use, modify,
and redistribute them, including commercially, while following the attribution
and ShareAlike terms.

Training data includes CC BY 2.0 France, CC BY 4.0, CC BY-SA 3.0, Japan Public
Data License content, and project-owned pairs. The complete corpus notices and
change statement are in `ATTRIBUTIONS.md`; all 9,162 retained Tatoeba
contributor records are in `tatoeba-attributions.jsonl.gz`.

Open distribution is separate from Mimi's product-quality gate. Public release
does not mean this candidate is approved for automatic app integration.

## Safety

Do not use machine-translated legal, medical, financial, or safety-critical
text as authoritative. Mimi must preserve fail-closed model authentication and
must not silently fall back to an unvalidated backend.
