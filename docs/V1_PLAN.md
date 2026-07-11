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
| Experimental MLX ASR packs | Qwen3-ASR 0.6B 4-bit (713 MB) and Nemotron 3.5 streaming 8-bit (756 MB) | Direct Swift/MLX integration across every capture lane; Qwen adds provisional, agreement-confirmed, and retrospective completed-window passes. |
| Translation | Apple Translation framework | EN↔JA is a local, on-demand language-pair download. Apple Speech drives a bounded rolling source/target view; other engines translate finalized text. |
| Meeting/speaker capture | ScreenCaptureKit content picker | The implemented audio-only lanes capture a person-selected app or display without registering a video output; microphone, app, and display audio stay separate. Core Audio process taps remain a later alternative. |

## V1 user flow

1. Open Mimi from the menu bar.
2. Choose **English** or **日本語** and a local model.
3. If using **Selected App Audio** or **Selected Display Audio**, click the matching **Choose … Audio** control and make an explicit choice in the macOS content picker.
4. Click **Download Model** explicitly; no large model is fetched on launch.
5. Click **Start**. The menu-bar glyph and red `LOCAL` indicator make the active recording state clear.
6. Read/copy the live transcript, then optionally translate the latest English/Japanese snapshot live into the other language.
7. Whisper's temporary source audio is deleted after its accuracy pass; Apple Speech and live MLX engines process bounded PCM in memory without writing a source-audio file. Transcript text remains local.

## Capture lanes

Mimi never blindly mixes your microphone and meeting audio. They are independent lanes:

```text
Selected microphone ──┐
Selected app audio ───┼─> PCM normalization ─> chosen ASR ─> final segments ─> local translation
Selected display audio┘
```

- **Microphone** uses the selected microphone input and asks only for Microphone permission.
- **Selected App Audio** presents the system ScreenCaptureKit picker in app-only mode. After an explicit selection, Mimi registers an audio output for that app's stream. On a browser this is app-level (for example, Chrome), not a specific Meet tab.
- **Selected Display Audio** presents the system picker in display-only mode. After an explicit display selection, Mimi captures the audio associated with that display. It is deliberately not labeled unrestricted “all system audio.”
- The implementation does not register a ScreenCaptureKit video output or build its own app/window picker. The system picker, its cancellation path, and the relevant macOS privacy flow remain visible to the person using Mimi.
- Each lane is selected and started independently. Mimi does not mix microphone and speaker-output audio by default.
- Core Audio process taps are still a planned alternative, not the capture mechanism claimed for this version.

## Model-pack contract

All engines conform to the same functional contract: report volatile partial text, report final text, and expose a user-visible install/remove state.

| Pack | Role | Languages | Download | State |
| --- | --- | --- | --- | --- |
| Apple Speech | Native live lane on macOS 26+ | Runtime-probed English/Japanese assets | System managed | Default where available |
| Whisper Large-v3 (626 MB) | Local accuracy lane | English + Japanese | About 626 MB, selected by the user, stored in Mimi's app-managed model folder | Implemented and removable |
| Nemotron 3.5 MLX (756 MB) | Experimental bounded live captions | English + Japanese | About 756 MB, selected by the user, revision-pinned in Mimi's app-managed cache | Streams 560 ms chunks, commits on a pause or hard 30-second cap, and is never the default without a Mac bake-off |
| Qwen3-ASR MLX (713 MB) | Experimental dual-pass live captions | English + Japanese | About 713 MB, selected by the user, revision-pinned in Mimi's app-managed cache | Native Swift/MLX provisional decoding, agreement confirmation, cached encoder windows, and retrospective completed-window finalization |
| TranslateGemma | Higher-quality local translation candidate | English + Japanese | Requires Gemma terms/notice | Deferred |

No optional model is auto-downloaded or bundled with Mimi. The rule is deliberate: no model is advertised as a quality default merely because it is new. A model graduates only after repeatable Mac measurements on English and Japanese meeting audio.

## Acceptance gates

### Automated in CI

- Dependency-free Swift self-tests for volatile/final transcript coalescing, Japanese rendering, and model routing (works with Command Line Tools, not just full Xcode).
- Deterministic E2E executable covering English and Japanese source sessions, model install/remove gates, temporary-audio cleanup, and screen-audio selection/cancellation lifecycle.
- Native MLX/Nemotron product compilation without fetching model weights.
- Qwen lifecycle, live partial/final routing, stale callback isolation, and install/remove coverage without fetching weights.
- A deterministic native menu-surface smoke launch that renders real SwiftUI sample data and exits automatically.
- Swift package build, universal app packaging, signing verification, and `Info.plist` validation.

### Physical-Mac smoke suite

- `scripts/run-microphone-smoke.sh`: microphone grant and one-second realtime
  callback check without retaining source audio. Also verify deny/revoke and
  device route changes manually.
- `scripts/run-apple-speech-smoke.sh` and `scripts/run-whisper-smoke.sh`:
  opt-in one-second local model checks after a person explicitly installs the
  corresponding asset. They never download models from a test run.
- ScreenCaptureKit app/display selection, picker cancellation, selected Zoom/Chrome audio, sleep/wake, app restart, and the relevant macOS privacy prompts.
- Offline transcription after Apple, WhisperKit, or Nemotron installation. The real MLX Nemotron fixture and direct-executable live PCM smoke checks are opt-in and must use an already-downloaded local model; they never download weights as part of a test.
- Apple versus WhisperKit versus Qwen versus Nemotron EN WER / JA CER, time-to-first-partial, first confirmation, hypothesis churn, finalization delay, real-time factor, memory, and thermal behavior. See [the repeatable realtime benchmark](REALTIME_BENCHMARK.md). Experimental MLX paths must be benchmarked on real Macs before either becomes a default.

## Explicit non-goals for the first PR

- Speaker diarization and mixing microphone + system audio.
- Background launch/login-item auto-enable.
- Cloud ASR or cloud translation.
- Persistent source-audio archive by default.
- Making the experimental native MLX Nemotron live path a default, or presenting its bounded windows as seamless long-meeting ASR, before it is tested on real Macs.

## Authoritative research links

- [Apple: Bring advanced speech-to-text to your app with SpeechAnalyzer](https://developer.apple.com/videos/play/wwdc2025/277/)
- [Apple: Capturing system audio with Core Audio taps](https://developer.apple.com/documentation/coreaudio/capturing-system-audio-with-core-audio-taps)
- [Apple: Capturing screen content with ScreenCaptureKit](https://developer.apple.com/documentation/screencapturekit/capturing-screen-content-in-macos)
- [Apple: TranslationSession](https://developer.apple.com/documentation/translation/translationsession)
- [Apple: Designing for macOS](https://developer.apple.com/design/human-interface-guidelines/designing-for-macos/)
- [WhisperKit / Argmax OSS Swift](https://github.com/argmaxinc/argmax-oss-swift)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [MLX Audio Swift](https://github.com/Blaizzy/mlx-audio-swift)
- [NVIDIA Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [MLX community Nemotron 8-bit conversion](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
