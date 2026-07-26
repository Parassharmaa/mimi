# Mimi

Mimi is a private English and Japanese transcription app for macOS. It lives
in the menu bar, listens only to the source you choose, and keeps transcription
and translation on your Mac.

`Mimi` (耳) means “ear” in Japanese.

## A quick look

| Simple setup | Floating captions |
| --- | --- |
| ![Mimi bilingual onboarding](docs/images/mimi-onboarding.png) | ![Mimi floating original and translated captions](docs/images/mimi-captions.png) |

| Menu-bar controls | Transcript history and translation |
| --- | --- |
| ![Mimi menu-bar controls](docs/images/mimi-menu.png) | ![Mimi transcript window](docs/images/mimi-transcript.png) |

## What Mimi can do

- Transcribe a microphone, audio output, app, or display.
- Recognize English and Japanese automatically as speech arrives.
- Translate between English and Japanese locally.
- Float original text, translations, or both above other apps.
- Type into the selected field by speaking, with a global shortcut.
- Keep previous sessions so you can return to them later.
- Open automatically when you log in, if you choose.

Mimi uses Apple Speech by default for live transcription. The development
build also includes Mimi Speech Preview, a bilingual local MLX model.
Automatic language detection uses a small local helper that is downloaded only
when you choose Auto.
English↔Japanese translation uses the bundled 73.4 MB ElanMT Marian model
through MLX; translation text never leaves the Mac and does not use Apple
Translation.

## Local speech

![Apple SpeechAnalyzer and Mimi Speech Preview comparison](docs/images/speech-model-comparison.svg)

Mimi Speech Preview is one 468.15 MB 4-bit model for both English and Japanese.
On the fixed 24-clip screens, it scores 6.57% Japanese CER and 5.54% English
WER, versus 11.06% and 9.23% for Apple SpeechAnalyzer progressive. Its mean
compute RTF is 0.397 in Japanese and 0.405 in English.
The production-queue replay delivers audio on time with no drops and reaches
first text at 3.88 seconds in Japanese and 2.88 seconds in English. It retained
6.57% CER and 5.90% WER across six-minute and four-minute sessions with
one-second utterance boundaries. Finalization p95 is still 2.12 seconds in
Japanese and 1.73 seconds in English, and artificial gapless speech remains
weak. Apple remains the default. The preview currently requires manual English
or Japanese selection.

- [Pinned public model](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-4bit/tree/321a6ead9f6e0646bc8188a54d2a470e275c6b76)
- [Native MLX loader review](https://github.com/Blaizzy/mlx-audio-swift/pull/235)

## Local translation

![Mimi translation model comparison](docs/images/translation-model-comparison.svg)

Mimi's development build bundles a 73.4 MB bidirectional translator made from
two small Marian specialists, one for each direction. The exact model and
benchmark are public:

- [Model and weights on Hugging Face](https://huggingface.co/blazeofchi/mimi-en-ja-mlx-development-v1)
- [Benchmark data on Hugging Face](https://huggingface.co/datasets/blazeofchi/mimi-en-ja-development-v1)

The candidate starts from pinned ElanMT checkpoints and is continued on
licensed human translations and Mimi-owned pairs. It was not trained from
scratch and uses no synthetic targets or reasoning traces. The EN→JA model
averages three checkpoints; the JA→EN model uses the strongest legal-specialist
checkpoint. Both are quantized to 4-bit MLX weights.

On the public 200-case benchmark, Mimi scores 28.41 and 55.19 chrF++, 9.63 and
30.62 BLEU, and 0.8669 and 0.8192 COMET-22 for EN→JA and JA→EN. It stays below
155 ms segment p95 on the benchmark Mac. On the same 400 segment calls, Mimi is
33.1× and 16.5× faster at p95 than Apple Translation for EN→JA and JA→EN. The
signed release package keeps the previous stable model because a known long
legal document can trigger repetition and the public suite is not a promotion
test.

## Requirements

- macOS 15 or later.
- macOS 26 or later for Apple’s fastest live transcription.
- Apple Silicon is required for the bundled local translator and recommended
  for automatic language detection.

## Run it locally

```sh
swift build
scripts/build-app.sh debug
open .build/Mimi.app
```

macOS asks for microphone or system-audio access only when the selected source
needs it. Languages are prepared in **Settings → Languages**. To dictate into
another app, enable **Voice Type** during setup or in Settings, place the cursor
in a text field, and press the chosen shortcut. Words appear directly in the
field as you speak; press the shortcut again to stop.

## Test it

```sh
scripts/test.sh
```

This covers English and Japanese transcription, model setup, capture cleanup,
and the native interface in light and dark appearances.

For implementation details, benchmarks, and physical-Mac checks, see:

- [Version 1 plan](docs/V1_PLAN.md)
- [Realtime benchmark](docs/REALTIME_BENCHMARK.md)
- [Translation development report](Research/translation/development-accuracy-v1-report.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Release status

Mimi is under active development. Official tagged GitHub releases are built by
CI, signed with Developer ID, notarized by Apple, stapled, and verified with
Gatekeeper before the archive is published.
