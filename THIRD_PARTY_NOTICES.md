# Third-party notices

Mimi bundles a compact English-Japanese translation model. Optional
transcription models are fetched only after the person using the app selects
one.

## WhisperKit and Whisper Large-v3

- The application links the [Argmax OSS Swift SDK / WhisperKit](https://github.com/argmaxinc/argmax-oss-swift), licensed under MIT.
- The optional `large-v3-v20240930_626MB` Core ML model is fetched by WhisperKit from its configured model source. Before redistributing a release that prebundles model weights, retain the upstream model and SDK notices and re-check the current model card/license.

## Apple Speech assets

Apple manages the on-device SpeechAnalyzer language assets. Mimi requests those
shared system downloads at runtime and does not redistribute them. Mimi's
shipping translation path does not use Apple Translation.

## Bundled ElanMT local translation

- The app links [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm) and
  [MLX Swift](https://github.com/ml-explore/mlx-swift), both under MIT, for local
  Apple Silicon inference.
- Mimi bundles 4-bit quantized copies of the directional
  [ElanMT English to Japanese](https://huggingface.co/Mitsua/elan-mt-bt-en-ja)
  and [ElanMT Japanese to English](https://huggingface.co/Mitsua/elan-mt-bt-ja-en)
  checkpoints by ELAN MITSUA Project / Abstract Engine. Both model cards declare
  CC BY-SA 4.0 and document training exclusively from openly licensed data.
  The exact attribution, modification notice, release contract, and full
  license text are shipped in `TranslationLicenses` inside the app bundle.
- Mimi converted the pinned upstream floating-point weights to MLX 4-bit affine
  quantization with group size 64 and float16 computation. The shipping pack
  was not fine-tuned by Mimi. It is separately licensed under CC BY-SA 4.0 and
  may be extracted, modified, and redistributed under those terms.

## Non-shipping local translation research

- Mimi's stronger routed Marian research pack is intentionally excluded from
  releases until its separate quality, dataset-policy, and distribution gates
  are cleared.
- The high-quality research corpus is KFTT 1.0, derived from English content
  professionally translated by NICT from Japanese Wikipedia and licensed
  CC-BY-SA-3.0. Required attribution: "The data used in this service contains
  English contents which is translated by the National Institute of
  Information and Communications Technology (NICT) from Japanese sentences on
  Wikipedia. Our use of this data is licensed by the Creative Commons
  Attribution-Share-Alike License 3.0. Please refer to
  http://creativecommons.org/licenses/by-sa/3.0/ or
  http://alaginrc.nict.go.jp/WikiCorpus/ for details."
- Do not distribute KFTT-derived weights until their share-alike release policy,
  attribution placement, and source offer have been reviewed and documented.
- The conversational research control also uses text pairs from Tatoeba via
  ManyThings under per-row CC-BY-2.0-France attribution. Preserve both sentence
  IDs, contributor names, source links, and license references from every row in
  the release manifest. These rows are training-only and are not used as Mimi's
  quality benchmark.
- NICT Asian Language Treebank translations are CC BY 4.0 and their English
  Wikinews source text is CC BY 2.5. Any derived release must preserve the NICT
  and Wikinews attribution, source URLs and requested Riza et al. citation. The
  current research incumbent uses ALT-derived training rows and remains blocked
  from distribution. Normal release packaging cannot admit that model pack.
- NICT's English BTEC 20K release is CC BY 4.0 and is used only as source text
  for proposed teacher translations, never as falsely labeled parallel gold.
- SmolLM2-135M-Instruct (Apache-2.0) was evaluated only as a failed research
  ablation and is not the proposed shipping model.

## Optional Nemotron MLX pack

- Mimi optionally downloads the pinned `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` conversion only after the person using the app chooses **Download Nemotron MLX**. The weights are not bundled in the repository or release archive, are stored in Mimi's app-managed Application Support cache, and can be removed from Settings.
- Mimi loads the optional pack directly in Swift through [MLX Swift](https://github.com/ml-explore/mlx-swift) (MIT) and [MLX Audio Swift](https://github.com/Blaizzy/mlx-audio-swift) (MIT). The download client is [swift-huggingface](https://github.com/huggingface/swift-huggingface) (Apache-2.0).
- Before distributing, mirroring, or prebundling these weights, review and retain the then-current terms/notices for both the [NVIDIA source model](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) and the [MLX community conversion](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit). This notice does not assert that either model's terms permit a particular redistribution.

## Deferred candidates

TranslateGemma is not included or downloaded by Mimi in this release. Any future integration must retain the model provider's then-current license and distribution notices.
