# Third-party notices

Mimi does not bundle model weights in its repository or release archive. Models are fetched only after the person using the app selects a model.

## WhisperKit and Whisper Large-v3

- The application links the [Argmax OSS Swift SDK / WhisperKit](https://github.com/argmaxinc/argmax-oss-swift), licensed under MIT.
- The optional `large-v3-v20240930_626MB` Core ML model is fetched by WhisperKit from its configured model source. Before redistributing a release that prebundles model weights, retain the upstream model and SDK notices and re-check the current model card/license.

## Apple Speech and Translation assets

Apple manages the on-device SpeechAnalyzer and Translation language assets. Mimi requests those shared system downloads at runtime and does not redistribute them.

## Optional Nemotron MLX pack

- Mimi optionally downloads the pinned `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` conversion only after the person using the app chooses **Download Nemotron MLX**. The weights are not bundled in the repository or release archive, are stored in Mimi's app-managed Application Support cache, and can be removed from Settings.
- Mimi loads the optional pack directly in Swift through [MLX Swift](https://github.com/ml-explore/mlx-swift) (MIT) and [MLX Audio Swift](https://github.com/Blaizzy/mlx-audio-swift) (MIT). The download client is [swift-huggingface](https://github.com/huggingface/swift-huggingface) (Apache-2.0).
- Before distributing, mirroring, or prebundling these weights, review and retain the then-current terms/notices for both the [NVIDIA source model](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) and the [MLX community conversion](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit). This notice does not assert that either model's terms permit a particular redistribution.

## Deferred candidates

TranslateGemma is not included or downloaded by Mimi in this release. Any future integration must retain the model provider's then-current license and distribution notices.
