# Mimi

**A local English/Japanese transcription utility for macOS.**

Mimi sits in the menu bar, transcribes locally, and makes its model choices visible. On macOS 26 it uses Apple’s on-device live transcription engine; when accuracy matters, it can download the multilingual Whisper Large-v3 Core ML model and perform a local accuracy pass after recording.

> `Mimi` (耳) means “ear” in Japanese.

## What works in this first build

- Native SwiftUI menu-bar control and a normal transcript window.
- Local microphone capture with a prominent active-recording state.
- English and Japanese source-language routing.
- Apple `SpeechAnalyzer` live ASR on macOS 26+, with system-managed local assets.
- Downloadable WhisperKit Large-v3 accuracy mode (about 626 MB) with forced `en` or `ja` decoding rather than auto-detect.
- On-device English ↔ Japanese text translation using Apple’s Translation framework.
- Transcript copy/clear actions and automatic deletion of temporary source audio after transcription.
- Deterministic English/Japanese E2E coverage and a GitHub Actions macOS packaging job.

## Model policy

Mimi is intentionally provider-pluggable:

| Model | Best use | Why it is here |
| --- | --- | --- |
| Apple Speech | Fast live captions on macOS 26 | Apple designed the new on-device engine for live, long-form, and meeting transcription. |
| Whisper Large-v3 (626 MB) / WhisperKit | Accuracy pass / macOS 15 fallback | Mature Apple-silicon/Core ML path for multilingual English and Japanese. |
| Nemotron 3.5 Streaming 0.6B | Experimental future live lane | Current EN/JA streaming candidate, but it needs a real native-Mac bake-off before it earns a default slot. |
| Qwen3-ASR | Future quality lane | Strong current open ASR family, but native Swift/Mac tooling is still less mature. |

The app will not quietly download model weights. The person using Mimi picks **Download Model** first. WhisperKit weights live under Mimi’s Application Support folder and can be removed from Settings; Apple assets remain system-managed.

## Audio sources

The first complete input is **Microphone**. Mimi's design also reserves separate lanes for **Selected App Audio** (Zoom, Chrome/Meet, etc.) and **All System Audio** using the first-party Core Audio tap APIs. Those two modes require the physical-Mac permission smoke suite before they are enabled in a release; Mimi does not falsely claim speaker/output capture while it is unverified.

For implementation details and the acceptance criteria, read [the v1 plan](docs/V1_PLAN.md).

## Requirements

- macOS 15 or later.
- Apple Silicon is recommended for WhisperKit performance.
- macOS 26 or later for the Apple live ASR engine.
- Xcode 16+ for the WhisperKit dependency in a normal Xcode project. This repo can be built from the installed Swift toolchain once dependencies resolve.

## Run locally

```sh
swift build
scripts/build-app.sh debug
open .build/Mimi.app
```

The first microphone action triggers the standard macOS permission flow. The first Apple/WhisperKit translation or ASR model use may ask to download the language/model pack.

## Verify

```sh
scripts/test.sh
```

This runs unit tests and a deterministic end-to-end English/Japanese pipeline. It does not auto-accept TCC prompts or transmit real audio. See the physical-Mac smoke checklist in [docs/V1_PLAN.md](docs/V1_PLAN.md).

Render the real native menu surface with deterministic data and auto-quit after the smoke assertion:

```sh
scripts/run-ui-smoke.sh
```

## Privacy

- No cloud ASR or cloud translation is part of Mimi.
- Temporary microphone audio is deleted after a transcription run completes.
- Mimi persists the latest finalized session locally under Application Support until you clear it.
- Apple documents that Translation content is processed on-device; it may collect non-content API/performance metadata such as language pair and app bundle ID.

## Distribution status

The CI archive is ad-hoc signed for local testing. It is not yet Developer-ID signed or notarized, so macOS may require Control-click → Open. Those release credentials are intentionally not embedded in this repository.

## References

- [Apple SpeechAnalyzer](https://developer.apple.com/videos/play/wwdc2025/277/)
- [Apple Translation](https://developer.apple.com/documentation/translation/translationsession)
- [Apple Core Audio taps](https://developer.apple.com/documentation/coreaudio/capturing-system-audio-with-core-audio-taps)
- [WhisperKit](https://github.com/argmaxinc/argmax-oss-swift)
- [NVIDIA Nemotron 3.5 Streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)

See [third-party notices](THIRD_PARTY_NOTICES.md) before packaging any model weights into a release.
