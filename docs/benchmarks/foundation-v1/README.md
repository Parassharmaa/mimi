# Initial benchmark baseline

Recorded October 10, 2026 with an optimized Mimi build on Mac15,6, 36 GiB RAM,
macOS 26.5.1. Low-power mode was off and the initial thermal state was nominal.
This measures production speech queues and translation-model publication, not
microphone capture or display paint.

The [registered profile](../../../configs/benchmarks/baseline-v1.json) uses one
fresh-process attempt and one warm attempt per case/engine/lane combination.
It is a bounded speech pilot plus the full public translation development suite.
Failures were retained; successful output is not a claim of semantic accuracy.

| Run | Planned attempts, cold + warm | Completed requested outputs | Error outcomes |
|---|---:|---:|---:|
| Synthetic clean/noisy/long/silent probes | 104 | 92 | 12 |
| Public English speech pilot | 24 | 20 | 4 |
| Public Japanese speech pilot | 24 | 24 | 0 |
| Public translation, 400 segments | 800 | 610 | 190 |

These counts include repeated and paired observations. They are not 952
independent examples. Translation has 200 document/sentence units, 100 per
direction; the speech pilot has four human utterances per language. Synthetic
noise variants share their original utterance's grouping.

## Warm reference-text translation

Each direction has 200 segment attempts from 100 independent document/sentence
units. Timing and chrF2 below are conditional on returned outputs. Safety
rejections remain in the completion denominator and in the raw traces.

| Direction | Completed / attempted segments | Units with latency / planned | p50 ms | p95 ms | chrF2 on scored outputs |
|---|---:|---:|---:|---:|---:|
| English → Japanese | 167 / 200 | 89 / 100 | 17.53 | 33.48 | 36.58 |
| Japanese → English | 138 / 200 | 75 / 100 | 20.77 | 38.70 | 53.99 |

The 95 rejected/failed warm requests comprise 86 protected-token rejections and
9 empty, untranslated, or malformed-result rejections. Cold attempts exhibit
the same aggregate counts. These failures must not disappear behind the fast
successful-request timings. Tail estimates remain exploratory because fewer
than 100 independent units produced timed outputs in either direction.

chrF2 is character overlap, not a human adequacy judgment. The public corpus has
already been exposed to development, and its review limitations remain intact.
This is not a promotion or general translation-quality claim.

## Warm public speech pilot

First-text time starts at clip replay, including leading silence. These four
utterances per language do not justify tail percentiles. WER/CER includes empty
completed ASR outputs as deletions. Phonon's latency uses only its two nonempty
outputs, while its error rate includes all four clips.

| Engine | Language | Nonempty finals / clips | First-text p50 ms | WER / CER % |
|---|---|---:|---:|---:|
| Apple Speech | English | 4 / 4 | 1,576.88 | 9.72 WER |
| Mimi Speech | English | 4 / 4 | 2,754.52 | 6.94 WER |
| Phonon | English | 2 / 4 | 1,998.82 | 68.06 WER |
| Apple Speech | Japanese | 4 / 4 | 3,025.22 | 5.84 CER |
| Mimi Speech | Japanese | 4 / 4 | 3,752.07 | 4.55 CER |
| Parakeet | Japanese | 4 / 4 | 2,376.58 | 5.84 CER |

Fresh-process speech preparation medians were 95–101 ms for Apple,
602–606 ms for Mimi Speech, 328 ms for Phonon, and 459 ms for Parakeet.
Both-direction translation prewarming took about 262–263 ms. These preparation
measurements are separate from the first-output times above. Filesystem and GPU
driver caches were not cleared.

## What to investigate next

- Reduce rejected or missing finalized translations while preserving number,
  placeholder, markup, and URL safety. The error trace identifies each source.
- Investigate Phonon's empty results on two registered English clips before
  treating its fast synthetic first output as a general advantage.
- Investigate nonempty Apple Speech output on the four-second Japanese-locale
  digital-silence probe. Both cold and warm runs recorded it.
- Check translation sensitivity to ASR formatting and segmentation. Some
  Japanese pipeline cases emitted an early partial translation but failed to
  publish a finalized translation.

These are baseline observations, not model fixes included in this change.

## Evidence and reproduction

- [Synthetic lane summaries](synthetic.md)
- [English speech summary](human-en.md)
- [Japanese speech summary](human-ja.md)
- [Public translation summary](public-translation.md)
- [Portable evidence archive](evidence.tar.gz) and [SHA-256](evidence.tar.gz.sha256)
- [Metric definitions and commands](../../BENCHMARKS.md)

The archive expands to `baseline/`. Each run contains `metadata.json`,
`attempts.jsonl`, `suite.jsonl`, native per-case `trace.jsonl`, process outcomes,
original and portable receipts, and summaries. Exports copy measurement values
byte-for-byte and omit machine-specific request files and console logs.

```sh
sha256sum -c evidence.tar.gz.sha256
# On macOS: shasum -a 256 -c evidence.tar.gz.sha256
tar -xzf evidence.tar.gz
../../../scripts/benchmarks/bench compare \
  baseline/public-translation baseline/public-translation self-check.json --check
```

Self-comparison passed with zero paired differences and no false reliability
regressions. The full raw collection and frozen audio are retained locally in
`Research/benchmarks/local/foundation-v1/`, outside Swift's build cache. Audio is
not embedded in the portable report archive. Public inputs can be materialized
from their pinned sources; retain the frozen synthetic WAVs for same-hash
comparisons, or establish a new baseline after resynthesis.
