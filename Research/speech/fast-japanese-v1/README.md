# Fast Japanese native ASR investigation

All inference is local on the M3 Pro. The registered FLEURS screen has 24
Japanese utterances; originals and references are unchanged and audio digests
are checked before inference. Sampling units are utterances. The eight timing
positions are 0,3,6,9,12,15,18,21. Each uses one new transcriber followed by two
recordings reusing that transcriber. Preparation and ASR publication are timed
separately; these are prerecorded replays, not a measured microphone or UI-paint
latency. English Phonon is a separate-corpus speed reference.

- `configs/protocol.json`: prospective gates, amendments, failed absolute gates,
  frozen final parameters, and repeat policy.
- `literature/00_shortlist.md`: primary sources and model/runtime pins.
- `scripts/00_download_moonshine.py`: publisher size/CRC validation and SHA-256
  recording for the native Japanese Small/Tiny assets.
- `src/moonshine_probe.cpp` and `scripts/01_screen_moonshine.py`: standalone
  native C API screen, including mixed PCM16/float32 original WAVs.
- `scripts/02_screen_mimi_native.py`: Mimi's production 50ms audio-queue replay.
  Finals are scored using NFKC/casefold with punctuation, separators, controls,
  and symbols removed. Every attempt, warning, timeout and original hash stays
  in the report.
- `scripts/03_prepacking_ablation.py`: research-only native utility replacement;
  no change to Mimi's linked official Swift runtime.

`native-direct-*` measures accelerated replay compute, not live wall latency.
`native-paced-*` measures 100ms paced delivery through the C harness.
`mimi-*-direct-*` and `mimi-*-paced-*` measure the application wrapper with 50ms
buffers delivered after their audio interval. Backend actor work excludes pacing;
preparation is separate. The C harness includes WAV preprocessing in compute.

The Tiny candidate fails the preliminary quality gate. CoreML fails setup due
to an unsupported zero-length cache dimension. The combined early VAD/single
thread/250ms pilot worsens compute and is not selected. Small v2's absolute
one-second first-text and 0.06 paced-RTF gates fail; preview integration does not
turn these into passes. Its final quality qualifies for an explicit alternative
while Mimi Speech remains the stronger Japanese accuracy reference.

The first application run retained native PCM output. It was interrupted when
the default was identified, kept as v1, and superseded by v2 with
`return_audio_data=false`. The interrupted run is not a completed timing screen.
The final eight-utterance repeats and long replay use v2. Source/executable hashes
in the application reports identify the exact measured implementation.

The full 24-clip application quality screen yields 10.05% CER. Standalone native
steady 1s yields 9.67%; the different buffer boundary and forced final flush mean
these must not be presented as identical pipelines. Historical Apple/Whisper
short-clip controls have matching registered source audio but were not freshly
rerun for this screen.

The selected application model is **Japanese Parakeet encoder-only Q4**, 482 MB.
Its BF16 source scores 8.89% CER, all-module Q4 scores 10.52%, and preserving the
prediction modules recovers 9.05%. This is a new verified derivative; the older
681.5 MB Parakeet row lacked a retained conversion recipe and was not used as
measured evidence. `04_prepare_parakeet.py` verifies the pinned input and exact
qualified output digests using the native Mimi converter. No training is used.

`05_summarize.py` reports descriptive quality and bootstraps the paired Japanese
first-text difference at the utterance level. English blank failures are retained
and its successful-only timings are explicitly conditional. The final prewarm
pipeline is v5; the older v4 cold long replay exposed a 1.9s queue backlog. Silent
preparation inference reduces that to 100ms with identical final text. The final
model-only release is
https://github.com/Parassharmaa/mimi/releases/tag/models-parakeet-ja-encoder-q4-v1.
