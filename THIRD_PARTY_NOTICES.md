# Third-party notices

Mimi does not bundle model weights in its repository or release archive. Models are fetched only after the person using the app selects a model.

## WhisperKit and Whisper Large-v3

- The application links the [Argmax OSS Swift SDK / WhisperKit](https://github.com/argmaxinc/argmax-oss-swift), licensed under MIT.
- The optional `large-v3-v20240930_626MB` Core ML model is fetched by WhisperKit from its configured model source. Before redistributing a release that prebundles model weights, retain the upstream model and SDK notices and re-check the current model card/license.

## Apple Speech and Translation assets

Apple manages the on-device SpeechAnalyzer and Translation language assets. Mimi requests those shared system downloads at runtime and does not redistribute them.

## Experimental candidates

NVIDIA Nemotron 3.5 Streaming and TranslateGemma are not included or downloaded by Mimi in this release. Any future integration must retain the model provider's then-current license and distribution notices.
