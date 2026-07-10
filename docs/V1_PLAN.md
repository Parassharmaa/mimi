# Mimi v1 plan

## Product promise

Mimi is a native, local-first macOS transcription utility for English and Japanese. It lives in the menu bar for fast control, keeps transcript text on the Mac, and lets a person choose the local ASR model that fits the moment: fastest live captions or a slower accuracy pass.

## Decisions made before implementation

| Decision | Choice | Reason |
| --- | --- | --- |
| App shape | SwiftUI `MenuBarExtra` plus a normal transcript window | Background control should be instant, but transcript reading/settings need a proper desktop surface. |
| Deployment target | macOS 15+ | Enables Apple Translation and Core Audio taps; macOS 26 unlocks the newer Apple live ASR engine. |
| Default ASR on macOS 26 | Apple `SpeechAnalyzer` / `SpeechTranscriber` | On-device, designed for live/meeting transcription, system-managed locale assets. |
| Accuracy ASR pack | WhisperKit / Whisper Large-v3 Core ML (626 MB) | Mature Swift integration, multilingual Japanese + English, explicit model download. |
| New streaming candidate | NVIDIA Nemotron 3.5 Streaming 0.6B | Promising EN/JA streaming option, but not promoted until a native-Mac accuracy/latency bake-off passes. |
| Translation | Apple Translation framework | EN↔JA is a local, on-demand language-pair download. Translate finalized segments only. |
| Meeting/speaker capture | Core Audio process taps | Captures a selected Zoom/Chrome process without requesting screen pixels; mic and app audio stay as separate lanes. |

## V1 user flow

1. Open Mimi from the menu bar.
2. Choose **English** or **日本語** and a local model.
3. Click **Download Model** explicitly; no large model is fetched on launch.
4. Click **Start**. The menu-bar glyph and red `LOCAL` indicator make the active recording state clear.
5. Read/copy the live transcript, then optionally translate finalized text into the other language.
6. Temporary source audio is deleted after the transcription engine completes; transcript text remains local.

## Capture lanes

Mimi never blindly mixes your microphone and meeting audio. They are independent lanes:

```text
Selected microphone ──┐
                     ├─> PCM normalization ─> chosen ASR ─> final segments ─> local translation
Selected app audio ───┘
```

- **Microphone** is the first complete lane and asks only for Microphone permission.
- **Selected App Audio** and **All System Audio** are designed around Core Audio taps, but remain outside this initial release until the local physical-Mac permission smoke test is complete. On a browser, this will be process/app level (for example, Chrome audio), not a specific Meet tab.
- ScreenCaptureKit is deliberately not the primary meeting-audio route. It remains a later option for a user-selected screen/window plus audio feature.

## Model-pack contract

All engines conform to the same functional contract: report volatile partial text, report final text, and expose a user-visible install/remove state.

| Pack | Role | Languages | Download | State |
| --- | --- | --- | --- | --- |
| Apple Speech | Native live lane on macOS 26+ | Runtime-probed English/Japanese assets | System managed | Default where available |
| Whisper Large-v3 (626 MB) | Local accuracy lane | English + Japanese | About 626 MB, selected by the user, stored in Mimi's app-managed model folder | Implemented and removable |
| Nemotron 3.5 Streaming | Lowest-latency third-party candidate | English + Japanese | Not exposed until benchmarked | Experimental/gated |
| TranslateGemma | Higher-quality local translation candidate | English + Japanese | Requires Gemma terms/notice | Deferred |

The rule is deliberate: no model is advertised as a quality default merely because it is new. A model graduates only after repeatable Mac measurements on English and Japanese meeting audio.

## Acceptance gates

### Automated in CI

- Dependency-free Swift self-tests for volatile/final transcript coalescing, Japanese rendering, and model routing (works with Command Line Tools, not just full Xcode).
- Deterministic E2E executable covering English and Japanese source sessions.
- A deterministic native menu-surface smoke launch that renders real SwiftUI sample data and exits automatically.
- Swift package build, universal app packaging, signing verification, and `Info.plist` validation.

### Physical-Mac smoke suite

- Microphone grant/deny/revoke and device route changes.
- Core Audio tap permission, selected Zoom/Chrome audio, sleep/wake, and app restart.
- Offline after Apple/WhisperKit model installation.
- Apple versus WhisperKit EN WER / JA CER, time-to-first-partial, finalization delay, real-time factor, memory, and thermal behavior.

## Explicit non-goals for the first PR

- Speaker diarization and mixing microphone + system audio.
- Background launch/login-item auto-enable.
- Cloud ASR or cloud translation.
- Persistent source-audio archive by default.
- Making the experimental Nemotron wrapper a default before it is tested on real Macs.

## Authoritative research links

- [Apple: Bring advanced speech-to-text to your app with SpeechAnalyzer](https://developer.apple.com/videos/play/wwdc2025/277/)
- [Apple: Capturing system audio with Core Audio taps](https://developer.apple.com/documentation/coreaudio/capturing-system-audio-with-core-audio-taps)
- [Apple: TranslationSession](https://developer.apple.com/documentation/translation/translationsession)
- [Apple: Designing for macOS](https://developer.apple.com/design/human-interface-guidelines/designing-for-macos/)
- [WhisperKit / Argmax OSS Swift](https://github.com/argmaxinc/argmax-oss-swift)
- [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
