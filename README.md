# Mimi

**A local English/Japanese transcription utility for macOS.**

Mimi sits in the menu bar, transcribes locally, and makes its model choices visible. On macOS 26 it uses Apple’s progressive on-device transcription preset; it can also download Whisper Large-v3 for a local post-stop accuracy pass, or native MLX Qwen3-ASR and Nemotron packs for experimental live captions.

> `Mimi` (耳) means “ear” in Japanese.

## What works in this first build

- Native SwiftUI menu-bar control and a normal transcript window.
- Local microphone capture with a prominent active-recording state.
- Direct audio-only capture of everything playing through a selected physical
  or virtual output device, using a private, unmuted Core Audio process tap.
- Audio-only capture of a person-selected app or display through the macOS
  ScreenCaptureKit content picker. These are separate lanes from microphone
  input; Mimi does not register a video output or silently mix the sources.
- English and Japanese source-language routing.
- Apple `SpeechAnalyzer` progressive live ASR on macOS 26+, including fast volatile results while speech is arriving and system-managed local assets.
- Downloadable WhisperKit Large-v3 accuracy mode (about 626 MB) with forced `en` or `ja` decoding rather than auto-detect.
- Downloadable `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` mode
  (about 756 MB), loaded directly by native Swift/MLX code for experimental
  English/Japanese live captions from the selected microphone, app, or display
  audio lane. It finalizes a bounded local window at a pause or 30 seconds;
  that keeps memory predictable, but a forced speech-boundary reset can reduce
  accuracy versus Apple Speech or Whisper's post-stop pass.
- Downloadable `mlx-community/Qwen3-ASR-0.6B-4bit` mode (about 713 MB), loaded directly in Swift with MLX Audio. It combines frequent provisional decoding, agreement-based confirmation, cached encoder windows, and a retrospective finalization pass.
- Live on-device English ↔ Japanese text translation using Apple’s Translation framework. With Apple Speech selected, Mimi presents source and target panes together and translates a bounded 480-character rolling context on a steady cadence; other ASR engines translate finalized text. It uses the low-latency strategy on macOS 26.4+.
- Transcript copy/clear actions and automatic deletion of temporary source audio after Whisper's post-stop transcription; Apple Speech and live MLX engines process bounded PCM in memory without writing a source-audio file.
- Transcript and translation panes follow incoming text automatically. If a
  person scrolls up, Mimi preserves that reading position and shows a **New
  text ↓** control that returns to the bottom and resumes following.
- Deterministic English/Japanese E2E coverage and a GitHub Actions macOS packaging job.

## Model policy

Mimi is intentionally provider-pluggable:

| Model | Best use | Why it is here |
| --- | --- | --- |
| Apple Speech | Fast live captions on macOS 26 | Apple designed the new on-device engine for live, long-form, and meeting transcription. |
| Whisper Large-v3 (626 MB) / WhisperKit | Accuracy pass / macOS 15 fallback | Mature Apple-silicon/Core ML path for multilingual English and Japanese. |
| Nemotron 3.5 MLX (756 MB) | Experimental bounded live captions | Native Swift/MLX path for English/Japanese selected-audio capture. It streams 560 ms chunks, finalizes at a pause or 30 seconds, and is not presented as seamless long-meeting ASR. |
| Qwen3-ASR 0.6B 4-bit (713 MB) | Experimental dual-pass live captions | Native Swift/MLX streaming with provisional, agreement-confirmed, and completed-window correction passes. It is fast when warm, but remains opt-in while Japanese accuracy and partial-text churn trail Apple Speech. |

The app will not quietly download model weights. Model setup lives in **Settings → Models**: Mimi checks the exact selected Apple language through `AssetInventory`, then shows **Checking**, **Needs download**, **Downloading**, **Ready**, or **Unavailable** instead of conflating macOS support with installed assets. Apple assets remain system-managed and language-specific; Whisper, Qwen, and Nemotron weights are app-managed, explicitly downloaded, and removable. Model downloads expose progress and remain retryable. No optional model is bundled in the repository or release archive.

The repeatable local evaluation and current M3 Pro measurements are documented in [the realtime benchmark](docs/REALTIME_BENCHMARK.md). A model is promoted by measured latency, accuracy, stability, memory, and thermals—not by recency alone.

## Audio sources

**Microphone**, **Selected Audio Output**, **Selected App Audio**, and **Selected Display Audio** are distinct inputs. Selected Audio Output uses a private Core Audio tap for everything routed to one selected physical or virtual output, making it the direct choice for speaker or meeting audio. The app and display lanes instead open macOS’s ScreenCaptureKit content picker: choose an app such as Zoom or Chrome for app audio, or choose the display carrying the speaker output for display-associated audio. Mimi captures only audio and does not capture screen pixels. A browser choice is app-level (for example, Chrome), not a specific Meet tab.

Each lane is explicit and independent. Mimi never blindly mixes meeting audio with the microphone, and it does not label the display lane as unrestricted “all system audio.”

For implementation details and the acceptance criteria, read [the v1 plan](docs/V1_PLAN.md).

## Requirements

- macOS 15 or later.
- Apple Silicon is recommended for WhisperKit and required for the native MLX Qwen and Nemotron runtimes.
- macOS 26 or later for the Apple live ASR engine.
- Full Xcode is required to package the MLX Metal shader used by Qwen and Nemotron. A Command Line Tools-only debug build still runs Apple Speech and Whisper, but truthfully disables the optional MLX lanes.

## Run locally

```sh
swift build
scripts/build-app.sh debug
open .build/Mimi.app
```

The first microphone action triggers the standard macOS permission flow. Selected Audio Output may request System Audio Recording access; choosing app or display audio opens the system content picker and may require the relevant macOS privacy permission. Apple Speech never starts a hidden download: choose the selected English or Japanese system asset in **Settings → Models** and explicitly start setup. Whisper, Qwen, and Nemotron likewise download only after an explicit Mimi action.

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

The signed app also has an opt-in physical smoke for direct output capture. It
plays a local synthesized phrase, verifies PCM callbacks from the default
output tap, and retains no source audio:

```sh
scripts/build-app.sh debug
.build/Mimi.app/Contents/MacOS/Mimi --e2e-output-audio-smoke
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

Generate deterministic English/Japanese speech fixtures and compare the old
Apple accuracy preset, Apple progressive transcription, rolling Whisper, and
Qwen's MLX dual-pass session:

```sh
scripts/run-realtime-benchmark.sh
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
- Whisper's temporary microphone, selected-app, and selected-display audio is deleted after its post-stop transcription completes. Apple Speech, live Qwen, and live Nemotron process PCM only in memory; the MLX engines bound their active windows and waiting queues.
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
- [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)
- [MLX Qwen3-ASR 0.6B 4-bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit)
- [NVIDIA Nemotron 3.5 Streaming](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [MLX community Nemotron 8-bit conversion](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit)

See [third-party notices](THIRD_PARTY_NOTICES.md) before packaging any model weights into a release.
