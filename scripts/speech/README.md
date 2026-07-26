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

Download the pinned public model into
`Research/speech/work/whisper-large-v3-turbo-asr-4bit`, then run:

```sh
uv run \
  --with 'mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@d28d68c6ac4e28f7d2d66007f640b06cf3fd8ceb' \
  --with 'soundfile==0.13.1' \
  scripts/speech/run_whisper_mlx_benchmark.py \
  Research/speech/work/whisper-large-v3-turbo-asr-4bit \
  Research/speech/work/fleurs-ja-screen-v1/manifest.jsonl \
  Research/speech/work/fleurs-ja-screen-v1/whisper-large-v3-turbo-q4.json \
  --language ja --metric cer

uv run \
  --with 'mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git@d28d68c6ac4e28f7d2d66007f640b06cf3fd8ceb' \
  --with 'soundfile==0.13.1' \
  scripts/speech/run_whisper_mlx_benchmark.py \
  Research/speech/work/whisper-large-v3-turbo-asr-4bit \
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
