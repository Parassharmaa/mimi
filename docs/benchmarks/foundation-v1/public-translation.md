# Mimi benchmark baseline

Observed 800/800 planned attempts, including cold runs and warmups.
Outcomes: `{"error": 190, "success": 610}`.

Warm latency uses observed events. Failed and missing results remain in the coverage denominator. A dash is unmeasured or insufficient independent units.

## Speech recognition

| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |
|---|---|---|---:|---:|---:|---:|---:|---:|

## Reference text to translation

| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mimi-translation | en-US | public-development | 100 | 167/200 | 17.53 | 33.48 | — | 36.58 |
| mimi-translation | ja-JP | public-development | 100 | 138/200 | 20.77 | 38.70 | — | 53.99 |

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
