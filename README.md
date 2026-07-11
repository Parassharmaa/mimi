# Mimi

**A local English/Japanese transcription utility for macOS.**

Mimi sits in the menu bar, transcribes locally, and makes its model choices visible. On macOS 26 it uses Apple’s on-device live transcription engine; it can also download Whisper Large-v3 for a local post-stop accuracy pass or the optional MLX Nemotron pack for experimental bounded live captions.

> `Mimi` (耳) means “ear” in Japanese.

## What works in this first build

- Native SwiftUI menu-bar control and a normal transcript window.
- Local microphone capture with a prominent active-recording state.
- Audio-only capture of a person-selected app or display through the macOS
  ScreenCaptureKit content picker. These are separate lanes from microphone
  input; Mimi does not register a video output or silently mix the sources.
- English and Japanese source-language routing.
- Apple `SpeechAnalyzer` live ASR on macOS 26+, with system-managed local assets.
- Downloadable WhisperKit Large-v3 accuracy mode (about 626 MB) with forced `en` or `ja` decoding rather than auto-detect.
- Downloadable `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` mode
  (about 756 MB), loaded directly by native Swift/MLX code for experimental
  English/Japanese live captions from the selected microphone, app, or display
  audio lane. It finalizes a bounded local window at a pause or 30 seconds;
  that keeps memory predictable, but a forced speech-boundary reset can reduce
  accuracy versus Apple Speech or Whisper's post-stop pass.
- On-device English ↔ Japanese text translation using Apple’s Translation framework.
- Transcript copy/clear actions and automatic deletion of temporary source audio after Whisper's post-stop transcription; Apple Speech and live Nemotron process bounded PCM in memory without writing a source-audio file.
- Deterministic English/Japanese E2E coverage and a GitHub Actions macOS packaging job.

## Model policy

Mimi is intentionally provider-pluggable:

| Model | Best use | Why it is here |
| --- | --- | --- |
| Apple Speech | Fast live captions on macOS 26 | Apple designed the new on-device engine for live, long-form, and meeting transcription. |
| Whisper Large-v3 (626 MB) / WhisperKit | Accuracy pass / macOS 15 fallback | Mature Apple-silicon/Core ML path for multilingual English and Japanese. |
| Nemotron 3.5 MLX (756 MB) | Experimental bounded live captions | Native Swift/MLX path for English/Japanese selected-audio capture. It streams 560 ms chunks, finalizes at a pause or 30 seconds, and is not presented as seamless long-meeting ASR. |
| Qwen3-ASR | Future quality lane | Strong current open ASR family, but native Swift/Mac tooling is still less mature. |

The app will not quietly download model weights. Model setup lives in **Settings → Models**: Mimi checks the exact selected Apple language through `AssetInventory`, then shows **Checking**, **Needs download**, **Downloading**, **Ready**, or **Unavailable** instead of conflating macOS support with installed assets. Apple assets remain system-managed and language-specific; Whisper and Nemotron weights are app-managed, explicitly downloaded, and removable. Whisper shows truthful model-file progress, supports cancellation, and resumes partial downloads on retry. Neither optional model is bundled in the repository or release archive.

## Audio sources

**Microphone**, **Selected App Audio**, and **Selected Display Audio** are distinct inputs. For the two speaker-output lanes, Mimi opens macOS’s ScreenCaptureKit content picker: choose an app such as Zoom or Chrome for app audio, or choose the display carrying the speaker output for display-associated audio. The picker and macOS privacy flow remain in control; Mimi captures only the selected audio stream and does not capture screen pixels. A browser choice is app-level (for example, Chrome), not a specific Meet tab.

Core Audio process taps remain a future alternative. Mimi does not label the current display lane as unrestricted “all system audio,” and it never blindly mixes meeting audio with the microphone.

For implementation details and the acceptance criteria, read [the v1 plan](docs/V1_PLAN.md).

## Requirements

- macOS 15 or later.
- Apple Silicon is recommended for WhisperKit and required for the native MLX Nemotron runtime.
- macOS 26 or later for the Apple live ASR engine.
- Full Xcode is required to package the MLX Metal shader used by Nemotron. A Command Line Tools-only debug build still runs Apple Speech and Whisper, but truthfully disables the optional Nemotron lane.

## Run locally

```sh
swift build
scripts/build-app.sh debug
open .build/Mimi.app
```

The first microphone action triggers the standard macOS permission flow. Choosing app or display audio opens the system content picker and may require the relevant macOS privacy permission. Apple Speech never starts a hidden download: choose the selected English or Japanese system asset in **Settings → Models** and explicitly start setup. Whisper and Nemotron likewise download only after an explicit Mimi action.

## Verify

```sh
scripts/test.sh
```

This runs unit tests and a deterministic end-to-end English/Japanese pipeline. It does not auto-accept TCC prompts or transmit real audio. See the physical-Mac smoke checklist in [docs/V1_PLAN.md](docs/V1_PLAN.md).

Render the real native menu surface with deterministic data and auto-quit after the smoke assertion:

```sh
scripts/run-ui-smoke.sh
```

On a physical Mac with a microphone available, the following local-only smoke
test retains no audio and verifies that the realtime capture callback receives
buffers for one second. It is intentionally excluded from CI because macOS
permission prompts cannot be auto-accepted safely.

```sh
scripts/run-microphone-smoke.sh
```

After explicitly downloading a model in Mimi, these opt-in physical-Mac
checks run a one-second local start/stop session for the corresponding engine.
They never download a model implicitly and are intentionally excluded from CI.

```sh
scripts/run-apple-speech-smoke.sh
scripts/run-apple-speech-smoke.sh ja-JP
scripts/run-whisper-smoke.sh
scripts/run-whisper-smoke.sh ja-JP
scripts/run-nemotron-smoke.sh
scripts/run-nemotron-smoke.sh ja-JP
```

The native Nemotron product has an opt-in fixture smoke check for an
already-downloaded local model. It verifies both the offline comparison path
and the 560 ms bounded streaming path for English and Japanese; it is
intentionally not a CI download step because model weights are never fetched
implicitly.

```sh
MIMI_NEMOTRON_MODEL_DIR=/path/to/pinned-model \
MIMI_MLX_METALLIB=/path/to/mlx.metallib \
scripts/run-nemotron-fixture-smoke.sh
```

To exercise Mimi's actual 48 kHz PCM conversion, bounded queue, and live
stream lifecycle without opening a microphone, run the companion opt-in
smoke with the same already-downloaded model and shader:

```sh
MIMI_NEMOTRON_MODEL_DIR=/path/to/pinned-model \
MIMI_MLX_METALLIB=/path/to/mlx.metallib \
scripts/run-nemotron-live-app-smoke.sh
```

## Privacy

- No cloud ASR or cloud translation is part of Mimi.
- Whisper's temporary microphone, selected-app, and selected-display audio is deleted after its post-stop transcription completes. Apple Speech and live Nemotron process PCM only in memory; Nemotron bounds both its active window and waiting queue.
- Mimi persists the latest finalized session locally under Application Support until you clear it.
- Apple documents that Translation content is processed on-device; it may collect non-content API/performance metadata such as language pair and app bundle ID.

## Distribution status

The CI archive is ad-hoc signed for local testing. It is not yet Developer-ID signed or notarized, so macOS may require Control-click → Open. Those release credentials are intentionally not embedded in this repository.

## References

- [Apple SpeechAnalyzer](https://developer.apple.com/videos/play/wwdc2025/277/)
- [Apple Speech asset inventory](https://developer.apple.com/documentation/speech/assetinventory)
- [Apple Translation](https://developer.apple.com/documentation/translation/translationsession)
- [Apple Core Audio taps](https://developer.apple.com/documentation/coreaudio/capturing-system-audio-with-core-audio-taps)
- [Apple ScreenCaptureKit content capture](https://developer.apple.com/documentation/screencapturekit/capturing-screen-content-in-macos)
- [WhisperKit](https://github.com/argmaxinc/argmax-oss-swift)
- [MLX Swift](https://github.com/ml-explore/mlx-swift)
- [MLX Audio Swift](https://github.com/Blaizzy/mlx-audio-swift)
- [NVIDIA Nemotron 3.5 Streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [MLX community Nemotron 8-bit conversion](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit)

See [third-party notices](THIRD_PARTY_NOTICES.md) before packaging any model weights into a release.
