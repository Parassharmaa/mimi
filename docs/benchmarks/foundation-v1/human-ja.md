# Mimi benchmark baseline

Observed 24/24 planned attempts, including cold runs and warmups.
Outcomes: `{"success": 24}`.

Warm latency uses observed events. Failed and missing results remain in the coverage denominator. A dash is unmeasured or insufficient independent units.

## Speech recognition

| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| apple | ja-JP | public-human-speech | 4 | 4/4 | 3025.22 | — | 5.84 | — |
| mimi-whisper | ja-JP | public-human-speech | 4 | 4/4 | 3752.07 | — | 4.55 | — |
| parakeet | ja-JP | public-human-speech | 4 | 4/4 | 2376.58 | — | 5.84 | — |

## Reference text to translation

| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |
|---|---|---|---:|---:|---:|---:|---:|---:|

## Speech to translation

| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |
|---|---|---|---:|---:|---:|---:|---:|---:|

## Interpretation

- Synthetic speech does not establish human-speech accuracy.
- chrF2 is lexical overlap; these references do not authorize semantic-quality claims or promotion.
- Cold means fresh process; filesystem and driver caches are uncontrolled.
- Latency ends at model publication and excludes microphone capture and compositor paint.
- Missing observations are null. Tails require at least 20 independent groups and remain exploratory below 100.
- Process peak RSS is cumulative within a process, not isolated peak memory for each warm attempt.

Detailed preparation, queue/service/publication latency, RTF, memory, churn, route counts, confidence intervals, failures, and denominators are in `summary.json`. Raw attempts are in `attempts.jsonl`.
