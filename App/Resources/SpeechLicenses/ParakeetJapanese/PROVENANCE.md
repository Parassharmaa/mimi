# Japanese Parakeet encoder Q4 derivative

The original Japanese Hybrid FastConformer TDT-CTC 0.6B model is developed by
**NVIDIA**, trained on ReazonSpeech, and published under **CC-BY-4.0**:
https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja

The MLX conversion is by **mlx-community**:
https://huggingface.co/mlx-community/parakeet-tdt_ctc-0.6b-ja/tree/e3810190ff521dcd208bc444a71bf877f3864566

Mimi's exact input is the BF16 derivative by **mouri45**:
https://huggingface.co/mouri45/parakeet-tdt_ctc-0.6b-ja-bf16/tree/4f8f02b8473fa440313e02a089fc45b5dc496591

The BF16 publisher identifies the MLX-community model as its parent and records
that all floating tensors were cast to BF16. This is a publisher-described
lineage; Mimi does not claim byte equality with the canonical NVIDIA checkpoint.
Mimi verified the pinned BF16 weights against Hugging Face's LFS SHA-256:
`983b6338e3b323890dd13c71290dcfec2be4c85b4b0a5ead1d4a45f7892217ac`.

**Modification notice:** Mimi quantizes eligible encoder Linear/Embedding modules
to affine Q4 with group size 64, while preserving prediction/embedding/joint
modules and other residual tensors in BF16. Config vocabulary and architecture
are retained. This is a custom derivative, not an official NVIDIA Q4 release.
The derivative remains CC-BY-4.0. No additional restrictions are imposed.

- Weights: 481,725,023 bytes, SHA-256
  `84163fa7b961a579a63ae491ca04ebfac3078fa1d13bc1c5d08accd0a8e50c1c`.
- Config: 132,341 bytes, SHA-256
  `1d850ea4e1fffa94f43251707d260127b8489b73261e09bdc5f130ce5fccc9f2`.
- Conversion and runtime: MLX Swift 0.31.4 and Mimi's MLX Audio Swift fork
  `159c4d3c083de5881f9d0d91f81d37dce13f2763`.
- Conversion entry point: `--convert-parakeet-ja-q4` in the Mimi developer CLI.
  `MIMI_PARAKEET_Q4_ENCODER_ONLY=0` selects the rejected all-module ablation;
  the default selects the encoder-only derivative above.

The pinned input validation manifest, conversion output, registered comparisons,
and retained failures are in `Research/speech/fast-japanese-v1`. The older
681.5 MB/Q4 benchmark in the July speech review has no retained conversion recipe
or weight hash and was not reused as evidence for this derivative.

Mimi runs native Silero VAD through the Moonshine SDK in skip-transcription mode.
No Moonshine ASR model files are needed for Japanese Parakeet. SDK and embedded
VAD notices are in `../Moonshine`; its complete publisher terms are in
`../MOONSHINE-LICENSE.txt`. MLX runtime code uses its existing MIT notices.
