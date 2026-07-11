# Realtime transcription benchmark

## Purpose

Mimi evaluates local speech models as separate lanes instead of expecting one
model to optimize every tradeoff:

1. **Fast lane:** publish useful text while speech is still arriving.
2. **Confirmation lane:** stabilize recent words without waiting for Stop.
3. **Correction lane:** revisit a longer completed window for higher accuracy.
4. **Translation lane:** translate the newest stable transcript snapshot locally
   without blocking audio capture or ASR.

Apple progressive transcription currently supplies the production fast lane.
Qwen3-ASR's native MLX streaming session implements all three ASR passes in an
experimental model: frequent provisional decoding, agreement-based confirmation,
and completed-window finalization using cached encoder windows. Whisper remains
the post-stop accuracy option and a candidate for a future asynchronous
correction lane.

## Reproduce

On a physical Apple-silicon Mac with the model packs explicitly installed:

```sh
scripts/run-realtime-benchmark.sh
```

The script synthesizes fixed English and Japanese fixtures with macOS voices,
streams the audio at realtime speed, and writes one JSON report per engine and
language to `.build/realtime-benchmark/<timestamp>/`. Reports include:

- time to first text and first confirmed text;
- model-load and total wall time;
- decode time and real-time factor where available;
- update count and normalized hypothesis churn;
- English word error rate (WER) or Japanese character error rate (CER);
- final transcript and the initial update sequence.

These synthetic fixtures are a regression benchmark, not a claim about meeting
accuracy. Promotion also requires noisy, accented, code-switching, crosstalk,
long-duration, and thermal tests with audio the evaluator is permitted to use.

## Baseline: Apple M3 Pro, 36 GB

Measured 2026-07-11 with deterministic Samantha and Kyoko fixtures:

| Engine | English first text | English WER | Japanese first text | Japanese CER | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Apple accuracy preset | 3.74 s | 0.0% | 3.98 s | 22.7% | Accurate batch-like baseline, not suitable for live captions. |
| Apple progressive | 1.26 s | 0.0% | 1.25 s | 9.1% | Best current production balance. |
| Whisper rolling, 1 s prefixes | 2.80 s | 0.0% | 2.85 s | 27.3% | Real-time factor about 0.76; useful as a correction lane, not the fast lane. |
| Qwen3-ASR MLX dual pass | 0.22 s provisional / 1.47 s confirmed | 0.0% | 0.11 s provisional / 1.63 s confirmed | 13.6% | Very fast warm provisional output, but more hypothesis churn than Apple. |

Qwen's first provisional fragments can be unstable. The app therefore
coalesces display updates to a 160 ms cadence while retaining agreement-based
confirmation inside the model session. This is a UI stability policy, not a
change to the decoder's evidence.

## Current decision

- Keep **Apple progressive** as the default on macOS 26.
- Offer **Qwen3-ASR MLX** as an explicitly experimental dual-pass model.
- Keep **Whisper** as the mature post-stop accuracy path and research it as a
  background correction pass over bounded finalized windows.
- Keep **Nemotron** as an alternative experimental streaming model; do not call
  it unavailable when its pinned local pack and MLX runtime are installed.
- Run local translation independently on the latest transcript snapshot every
  700 ms while recording, using Apple's low-latency strategy on macOS 26.4+.

## Model-engineering iteration gates

Candidate Qwen, Nemotron, or Whisper changes must record the model revision,
precision, decoder settings, window/overlap sizes, confirmation policy, and Mac
hardware. They graduate only when they improve the target metric without an
unacceptable regression elsewhere:

- p50/p95 first-useful-text and confirmation latency;
- EN WER, JA CER, punctuation, numerals, and code-switch accuracy;
- hypothesis churn and retroactive edit distance;
- real-time factor, peak wired memory, sustained CPU/GPU use, and thermals;
- 30- and 120-minute stability, capture backpressure, cancellation, and sleep/wake;
- translation freshness, translation revision stability, and ASR isolation.

Likely MLX experiments are 4-bit versus 8-bit weights, decode interval,
agreement depth, encoder-window overlap, cached-window count, VAD boundaries,
and a separate correction model operating behind the live lane. The benchmark
must keep audio ingestion paced and bounded so a faster decoder cannot hide an
ever-growing queue.
