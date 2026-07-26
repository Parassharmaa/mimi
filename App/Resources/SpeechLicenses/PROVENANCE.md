# Mimi Speech development model

- Runtime name: Mimi Speech Preview
- Model artifact: `mlx-community/whisper-large-v3-turbo-asr-4bit`
- Pinned revision: `321a6ead9f6e0646bc8188a54d2a470e275c6b76`
- Conversion card's relative parent: `mlx-community/whisper-large-v3-turbo`
- Parent revision present when the Q4 artifact was published:
  `beea265c324f07ba1e347f3c8a97aec454056a86`
- Canonical OpenAI source revision:
  `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Source architecture: OpenAI Whisper Large V3 Turbo
- Conversion: MLX Audio 0.2.10, 4-bit quantization
- Canonical source model-card license metadata: MIT
- Evaluated payload: 468,150,715 bytes

The development pack is downloaded only from the pinned public revision. Mimi
checks the size and SHA-256 digest of every included file before packaging or
loading it. The exact hashes live in
`scripts/speech/verify_development_speech_pack.py`.

Canonical OpenAI source:
https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9

Conversion card's relative parent:
https://huggingface.co/mlx-community/whisper-large-v3-turbo/tree/beea265c324f07ba1e347f3c8a97aec454056a86

Pinned MLX conversion:
https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-4bit/tree/321a6ead9f6e0646bc8188a54d2a470e275c6b76

The Q4 conversion card identifies `./whisper-large-v3-turbo` as its parent but
does not record a parent commit or license field. Hugging Face history shows
that `beea265...` was the only parent revision current when the Q4 artifact was
published on 2026-01-13. This parent assignment is therefore a repository
history inference, not publisher-signed provenance. The intermediate card in
turn names `large-v3-turbo` without a working source link.

Mimi includes the MIT notice from the canonical OpenAI model card and records
the complete known chain above. Public release promotion remains blocked until
the Q4 publisher confirms the lineage or Mimi reproduces the conversion from
the pinned OpenAI source.
