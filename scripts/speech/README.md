# Mimi speech benchmark

Run from the repository root on Apple Silicon.

## Pinned inputs

* FLEURS revision: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`
* Whisper MLX model revision: `321a6ead9f6e0646bc8188a54d2a470e275c6b76`
* Python MLX Audio revision: `d28d68c6ac4e28f7d2d66007f640b06cf3fd8ceb`

The Whisper runner verifies SHA-256 hashes for the model weights, config, and
tokenizer before running. It also refuses an MLX Audio installation that does
not expose the pinned Git commit in `direct_url.json`.

## Materialize the public screens

```sh
uv run --with 'datasets[audio]==4.0.0' \
  scripts/speech/prepare_fleurs_ja_screen_v1.py \
  Research/speech/work/fleurs-ja-screen-v1

uv run --with 'datasets[audio]==4.0.0' \
  scripts/speech/prepare_fleurs_en_screen_v1.py \
  Research/speech/work/fleurs-en-screen-v1
```

## Run Whisper

Fetch the exact distributable pack without repository metadata, then run:

```sh
scripts/speech/fetch_development_speech_pack.sh \
  Research/speech/work/development-speech-pack
```

```sh
uv run \
  --with 'mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@d28d68c6ac4e28f7d2d66007f640b06cf3fd8ceb' \
  --with 'soundfile==0.13.1' \
  scripts/speech/run_whisper_mlx_benchmark.py \
  Research/speech/work/development-speech-pack/model \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/fleurs-ja-screen-v1/whisper-large-v3-turbo-q4.json \
  --language ja --metric cer

uv run \
  --with 'mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@d28d68c6ac4e28f7d2d66007f640b06cf3fd8ceb' \
  --with 'soundfile==0.13.1' \
  scripts/speech/run_whisper_mlx_benchmark.py \
  Research/speech/work/development-speech-pack/model \
  Research/speech/work/fleurs-en-screen-v1/manifest.jsonl \
  Research/speech/work/fleurs-en-screen-v1/whisper-large-v3-turbo-q4.json \
  --language en --metric wer
```

## Run Apple and compare

Build Mimi from the exact source revision under test. Apple's language assets
must already be installed.

```sh
swift build --product Mimi

python3 scripts/speech/run_apple_speech_benchmark.py \
  .build/debug/Mimi \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/fleurs-ja-screen-v1/apple-progressive.json \
  Research/speech/work/fleurs-ja-screen-v1/apple-staging \
  --language ja --metric cer

python3 scripts/speech/compare_asr_benchmarks.py \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/fleurs-ja-screen-v1/apple-progressive.json \
  Research/speech/work/fleurs-ja-screen-v1/whisper-large-v3-turbo-q4.json \
  Research/speech/work/fleurs-ja-screen-v1/apple-vs-whisper-large-v3-turbo-q4.json
```

Repeat the last two commands with the English directory, `--language en`, and
`--metric wer`. The comparison uses a deterministic 10,000-sample paired
clip bootstrap with seed `20260726`.

## Run Mimi Speech's bounded native live path

Place the matching `mlx.metallib` beside `.build/debug/Mimi`, then run:

```sh
python3 scripts/speech/run_mimi_whisper_live_benchmark.py \
  .build/debug/Mimi \
  Research/speech/work/development-speech-pack/model \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/fleurs-ja-screen-v1/mimi-whisper-native-live.json \
  --language ja --metric cer

python3 scripts/speech/run_mimi_whisper_live_benchmark.py \
  .build/debug/Mimi \
  Research/speech/work/development-speech-pack/model \
  Research/speech/work/fleurs-en-screen-v1/manifest.jsonl \
  Research/speech/work/fleurs-en-screen-v1/mimi-whisper-native-live.json \
  --language en --metric wer
```

This direct path feeds 100 ms PCM blocks into the actor runtime without sleeping.
It exercises the same adaptive VAD, rolling partial decoder, overlap stabilizer,
and final decoder used by the app, but intentionally bypasses the converter,
bounded queue, drain task, and MainActor scheduling. Its compute RTF is not
Apple's paced wall RTF. Time to first text includes the nominal audio arrival
time plus the corresponding local decode.

Run the complementary production-queue benchmark at wall-clock speed:

```sh
python3 scripts/speech/run_mimi_whisper_live_benchmark.py \
  .build/debug/Mimi \
  Research/speech/work/development-speech-pack/model \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/fleurs-ja-screen-v1/mimi-whisper-paced-queue.json \
  --language ja --metric cer --paced-queue
```

The paced mode uses the compiled language-aware product profile and the exact
`startLive` → `consumeLive` → bounded queue → actor → `stopLive` path. It
uses absolute audio deadlines, so converter work cannot progressively slow the
fixture below real time. It reports input-delivery RTF, maximum scheduling
lateness, queue capacity and peak depth, exact dropped samples and drop events,
backpressure notifications, first/final update latency, and time spent
finalizing after the input ends. Custom streaming-profile flags are rejected in
this mode so the evidence cannot silently diverge from the app.

Profile experiments must name both cadences explicitly:

```sh
python3 scripts/speech/run_mimi_whisper_live_benchmark.py \
  .build/debug/Mimi \
  Research/speech/work/development-speech-pack/model \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/profile.json \
  --language ja --metric cer \
  --initial-partial-stride 3 \
  --partial-stride 3 \
  --endpoint-silence 0.75
```

Both reports include final error rate, first text p50 and p95, hypothesis churn,
peak RSS, first-update prefix error, and first-update reference coverage. The
direct mode reports compute RTF. The paced mode reports wall-time RTF, queue
drops, backpressure, delivery timing, and finalization lag. Both record the
suite hash, selected case IDs, exact executable and model-weights hashes, feed
cadence, and effective profile. Every input audio hash is verified before
inference. `--limit` is available for a screening run, but a product decision
requires the complete registered suite.

Build a deterministic synthetic-noise condition without changing source text:

```sh
python3 scripts/speech/build_noisy_speech_fixture.py \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/noisy-ja-pink-15db-v2 \
  --noise-color pink --snr-db 15
```

The builder verifies every source hash, derives a per-case noise seed, preserves
the source level unless minimal clipping headroom is required, and records the
exact FFmpeg and FFprobe versions and executable hashes, gains, expected SNR,
output levels, and generated hashes. Repeated runs under that exact toolchain
produce byte-identical manifests and WAV files. This is a controlled
additive-noise diagnostic, not evidence about real rooms, microphones,
reverberation, or competing speakers.

Compare clean and derived conditions with paired case alignment:

```sh
python3 scripts/speech/compare_asr_benchmarks.py \
  Research/speech/work/noisy-ja-pink-15db-v2/manifest.jsonl \
  Research/speech/work/streaming-profile-sweep-v1/ja-product-paced-queue-8s-corrected-v2.json \
  Research/speech/work/noisy-ja-pink-15db-v2/mimi-paced-queue-8s-corrected-v2.json \
  Research/speech/work/noisy-ja-pink-15db-v2/clean-vs-noisy-8s-corrected-v2.json \
  --left-label "Mimi clean" \
  --right-label "Mimi pink noise 15 dB SNR"
```

Verify every selected report hash, same-executable direct/paced parity, noise
toolchain provenance, real-time delivery, and zero-loss queue telemetry:

```sh
python3 scripts/speech/verify_paced_speech_evidence.py
```

Build one long-form case from the same pinned and licensed clips:

```sh
python3 scripts/speech/build_long_form_fixture.py \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/long-form-ja-v1
```

Run the resulting `manifest.jsonl` with the same benchmark command. Generated
audio remains ignored; selected manifests and reports may be force-added as
reviewable evidence. The builder refuses missing or hash-mismatched registered
audio.
Add `--dynamic-normalize` to generate a second fixture with bounded dynamic
gain normalization. Report raw and normalized fixtures separately. The raw
concatenation is an abrupt multi-speaker gain-shift stress test, while the
normalized fixture isolates long-duration segmentation from that gain shift.
Add `--inter-clip-silence 1` to build a long-session fixture with explicit
utterance boundaries above Mimi's 750 ms endpoint threshold. Report this
separately from gapless concatenation, which is a continuous-speech stress
case.
