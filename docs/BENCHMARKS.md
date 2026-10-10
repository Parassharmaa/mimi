# Measuring transcription and translation

Mimi's common benchmark measures the production speech queues and translation
worker on a physical Mac. It has three lanes:

| Lane | Input → observed endpoint | Primary latency | Quality |
|---|---|---|---|
| `speech` | Paced audio → first nonempty transcript | `audioStartToFirstTextMs` | English WER, Japanese CER |
| `translation` | Reference text → finalized translation publication | `referenceSubmitToFinalTranslationMs` | chrF2 with exact signature |
| `pipeline` | Paced audio → first nonempty translation publication | `audioStartToFirstTranslationMs` | ASR WER/CER and translation chrF2 where references exist |

The pipeline exercises Apple Speech, Mimi Speech, Phonon English, and Parakeet
Japanese through their existing live engine methods. It uses `TranscriptDocument`
and `LocalTranslationStream`, including coalescing, failed partials, reused
finals, and safety rejections. It excludes microphone hardware, capture-device
queues, SwiftUI layout, and compositor paint. Apple Translation fallback and
automatic language routing are not measured by this runner. Their absence is a
coverage limitation, not a zero-latency result.

## Run a benchmark

Use an isolated checkout on Apple Silicon. Install the desired model packs
before collecting results. The runner never downloads a model or captures a
microphone. Keep other heavy work off the measurement machine.

Use the `bench` launcher for preparation, collection, reporting, and comparison.
It uses `uv` to pin Python 3.12.12 (Unicode 15.0.0) and the scoring requirements,
matching the retained foundation baseline. Mixing a system `python3` with a
different `uv` interpreter can produce incomparable runs even on the same Mac.

```sh
# Stamp the actual source identity, build optimized code, and record compiler,
# executable, dependency-lock and Metal-library hashes.
scripts/benchmarks/build.sh

# Synthetic regression probes: EN/JA, numbers, long utterances, noise, silence.
scripts/benchmarks/bench prepare_suite .build/benchmark-fixtures-v1

# Example: the production speech-to-translation path with pinned model roots.
scripts/benchmarks/bench run \
  .build/benchmark-fixtures-v1/manifest.jsonl .build/benchmark-run-v1 \
  --lanes speech translation pipeline \
  --whisper-model /path/to/whisper-large-v3-turbo-q4 \
  --phonon-model /path/to/mimi-phonon2 \
  --parakeet-model /path/to/ParakeetJapaneseEncoderQ4-v1

scripts/benchmarks/bench report .build/benchmark-run-v1
```

The model roots are explicit so the report can hash the exact artifacts. The
stable bundled translation pack is the default. Use `--translation-model` to
measure another pack. To use an already compiled matching MLX shader, set
`MIMI_MLX_METALLIB` when running `build.sh`; its SHA-256 is included in the build
receipt. Debug runs require `--allow-debug` and remain ineligible for performance
comparisons. Building updates the existing generated source-identity constant.

A fresh output directory is required. The protocol, selected cases, engine order,
inputs, every cold/warmup/warm attempt, stdout/stderr, and raw event traces are
retained. A receipt binds the completed files. Reporting refuses an
incomplete matrix or changed artifact. `metadata.json` records the run's actual
protocol overrides; `--case`, `--repetitions`, and `--warmups` are explicit,
recorded selections, not silent pruning.

For public human speech, materialize the repository's pinned FLEURS screens,
then import them without transcoding or changing their hashes:

```sh
uv run --no-project --python 3.12.12 --with 'datasets==4.0.0' \
  python scripts/speech/prepare_fleurs_en_screen_v1.py .build/fleurs-en
scripts/benchmarks/bench import_public_speech \
  .build/fleurs-en/manifest.jsonl .build/fleurs-en-common --language en-US
# Repeat with prepare_fleurs_ja_screen_v1.py and --language ja-JP.
```

The public FLEURS screen has already been used for development. It is not a new
holdout. Imported rows retain source references for ASR but have no translation
references; their translation quality stays unscored even when pipeline timing
is measured. Float WAVs are preserved as registered, rather than converted to
make a decoder convenient.

For translation quality, freeze the existing public 400-segment development
suite. Its long-document segments retain their parent document as the independent
unit. Existing exposure and review limitations remain attached to the rows.

```sh
scripts/benchmarks/bench prepare_suite .build/translation-development-v1 \
  --text-only \
  --public-translations Research/translation/benchmark/development-accuracy-v1.segments.jsonl
scripts/benchmarks/bench run \
  .build/translation-development-v1/manifest.jsonl .build/translation-run-v1 \
  --lanes translation
```

Do not use sealed promotion sources, references, or automated-claim material in
these development runs. Synthetic references are unreviewed engineering fixtures.
Neither a high overlap score nor these baselines authorizes model promotion.

## Timing contract

[Protocol v1](../configs/benchmarks/protocol-v1.json) fixes pacing, repetitions,
deadlines, randomization, quantiles, and uncertainty. Audio is delivered **at the
end of each 50 ms buffer**, against absolute monotonic deadlines. The runner
records schedule lateness rather than subtracting it from the result.

| Metric | Exact boundary |
|---|---|
| Speech preparation | Engine start requested → engine ready |
| Translation preparation | Startup prewarming requested → both directions ready, including model loading and preparation inference |
| Attempt-start latency | Fresh attempt origin → first text / first translation; includes preparation |
| First-text latency | Audio replay origin → first nonempty ASR publication; includes leading silence |
| Annotated onset latency | Human-annotated speech onset → first output; null without an annotation |
| Acoustic-activity latency | RMS-estimated onset → first text; a separate diagnostic, never substituted for speech onset |
| First-final latency | Replay origin → first application-finalized transcript; raw events distinguish engine finals from Apple's stop-time provisional promotion |
| ASR drain | Last audio buffer delivered → engine stop/flush returned |
| Translation queue | The specific correlated source revision arrived → its inference started |
| Translation service | Inference started → successful/failed result returned; includes actor scheduling, not just GPU kernels |
| Source-to-publication | Correlated source revision → translation assigned to the production worker's public model state |
| Translation tail | Last audio delivery → last finalized translation, bounded below by zero if completed earlier |
| Runtime-service RTF | Time awaiting ASR runtime calls, including stop, divided by audio duration; unavailable for Apple's closed implementation |
| Paced-wall RTF | Replay origin → speech stop, divided by audio duration; deliberately includes real-time waiting |
| Queue / drops | Exact engine queue counters at 16 kHz; unavailable Apple counters remain null |
| Memory | Process peak resident bytes; a cumulative high-water mark across repeats, not isolated per-attempt GPU memory |

Cold means a fresh subprocess, not flushed filesystem or GPU-driver caches.
Preparation is measured separately from publication latency. Setup is serialized
for attribution; attempt-start timing is harness startup, not desktop application
launch or the app's concurrent startup tasks. The same process and
model instances handle subsequent warmups and warm repeats. No per-token TTFT,
standardized Average Lagging, hardware capture latency, or paint latency is
claimed by a whole-sentence publication timestamp.

Tracing writes JSONL synchronously on the model's actor. Its small instrumentation
cost is part of these measurements. Compare runs under the same tracing protocol,
not with historical uninstrumented timings.

## Quality, failures, and statistics

- ASR uses NFKC, case folding, and punctuation/symbol/whitespace handling defined
  in `metrics.py`. English scores words; Japanese scores characters. Reports retain
  substitutions, deletions, insertions and reference-unit denominators. Corpus
  WER/CER is micro-aggregated; an empty output on a completed utterance scores all
  reference units as deletions. Empty-reference silence has no WER denominator.
- Translation uses pinned sacreBLEU 2.5.1, chrF2, character order 6, word order 0,
  beta 2. The metric signature is saved. This is lexical overlap, not semantic
  adequacy. Exceptions and timeouts are unscored and remain visible in coverage;
  completed empty translations are scored. Compare reference-text translation
  with pipeline translation to inspect propagation of ASR errors.
- Revision churn counts Unicode code points removed after the longest common prefix.
  Appends cost zero. Finalized utterance boundaries reset the comparison. Both the
  numerator and final-length-normalized rate are retained; an empty denominator
  yields null. This is a custom Mimi diagnostic.
- Success means the requested components returned complete nonempty output, or
  correctly remained empty for silence. It does not mean accurate translation.
  Native safety errors, empty output, hallucination, warnings, dropped audio,
  timeouts and unexecuted attempts remain separate observations.
- The external watchdog enforces an attempt deadline, kills the process group,
  retains the partial trace and marks later planned repeats `not_run`. Translation
  also has a separate bounded drain. There are no automatic retries. A later
  intentional rerun gets a new directory and never overwrites the failed run.
- Translation service-time quantiles use successful inference calls. Failed call
  durations remain in the raw request records beside failure counts.
- Observed latency always carries its observation and independent-unit counts.
  Deadline-capped attempt wall time reports the fixed outer cap for an observed
  failure; this is a conservative time-to-result diagnostic, not a measured
  successful latency.
- Quantiles use linear interpolation, type 7. Repeats and document segments remain
  clustered by `groupID`. Median intervals resample groups with their samples
  together. p90/p95 are suppressed below 20 independent groups and remain
  exploratory below 100. These intervals describe the registered fixture groups;
  they do not establish generalization to unseen speakers or meetings.

## Compare a future change

```sh
scripts/benchmarks/bench compare BASELINE CANDIDATE comparison.json --check
```

Comparison requires matching fixtures, protocol, evaluator, hardware, OS, build
configuration, power mode, Unicode normalization version, selected cases, and model artifacts. App source and
binary identities may differ: that is the change being evaluated. Use
`--allow-model-change` explicitly for a model comparison. Apple asset revisions
are not exposed, so their OS-managed identity remains a limitation.

The comparison command operates on warm attempts; cold preparation remains
available in the summaries and raw traces. The report pairs independent groups, retains all repeats, and gives both-success,
baseline-only, candidate-only and neither-success counts. Latency deltas use
paired group medians with group bootstrap intervals. Quality comparisons use the
same paired scored rows and report their coverage. A fast failure cannot improve
the successful paired latency result. `--check` fails for lost successful units,
increased audio loss or warnings, and lost queue telemetry. No arbitrary latency
or quality threshold is enabled before a defensible baseline and held-out quality
review exist. Treat ranking across many models or metrics as exploratory.

## Verification and sources

```sh
scripts/benchmarks/bench test_contract
.build/debug/Mimi --verify-benchmark-trace
```

These run in `scripts/test.sh`. CI also checks the retained baseline archive
with `scripts/verify-benchmark-evidence.py`, without replaying the models. They cover normalization and edit counts, timing
boundaries, source correlation, reused finals, failed inference, monotonic clocks,
late callbacks, partial UTF-8 traces, hard timeouts, missing observations, repeated
samples, changed cohorts, and paired reliability loss. The existing speech
qualification remains a separate source-bound release check.

The timing distinctions follow [Shangguan et al. (2021)](https://www.isca-archive.org/interspeech_2021/shangguan21_interspeech.pdf).
[SimulEval](https://aclanthology.org/2020.emnlp-demos.19/) defines simultaneous
translation metrics such as AL; this runner does not reuse that name for request
duration. [Xu et al. (2025)](https://aclanthology.org/2025.findings-naacl.393/)
explain why overlapping computation must not be added serially to consumed audio.
Translation scoring uses the [canonical sacreBLEU implementation](https://github.com/mjpost/sacrebleu).

## Retained initial profile

[Baseline profile v1](../configs/benchmarks/baseline-v1.json) declares a bounded
pilot covering every supported speech engine and all three lanes. It uses one
cold and one warm attempt, selected synthetic conditions, the first four
registered human-speech cases per language, and all 400 public translation
segments. This deliberately small speech sample cannot establish p95. The normal
protocol retains three warm repetitions; omit the pilot's selections to run the
full speech screens.

```sh
scripts/benchmarks/bench baseline .build/foundation-baseline \
  --synthetic .build/benchmark-fixtures-v1/manifest.jsonl \
  --human-en .build/fleurs-en-common/manifest.jsonl \
  --human-ja .build/fleurs-ja-common/manifest.jsonl \
  --translation .build/translation-development-v1/manifest.jsonl \
  --whisper-model /path/to/whisper-large-v3-turbo-q4 \
  --phonon-model /path/to/mimi-phonon2 \
  --parakeet-model /path/to/ParakeetJapaneseEncoderQ4-v1
```


The first retained collection is documented in
[the baseline overview](benchmarks/foundation-v1/README.md). Its portable archive
contains measurement rows, raw events, hashes, and receipts. Machine-specific
request files and console logs remain in the original local collection.
`scripts/package-benchmark-evidence.py` creates that portable export. Keep the
frozen audio and full raw collection outside `.build` if build caches will be
cleaned; the recorded run is preserved locally under
`Research/benchmarks/local/foundation-v1/`, which is gitignored.
