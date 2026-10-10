# Mimi benchmark baseline

Observed 104/104 planned attempts, including cold runs and warmups.
Outcomes: `{"error": 12, "success": 92}`.

Warm latency uses observed events. Failed and missing results remain in the coverage denominator. A dash is unmeasured or insufficient independent units.

## Speech recognition

| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| apple | en-US | clean | 1 | 1/1 | 1130.06 | — | 0.00 | — |
| apple | en-US | long | 1 | 1/1 | 1126.71 | — | 0.00 | — |
| apple | en-US | noise | 1 | 1/1 | 1135.23 | — | 0.00 | — |
| apple | en-US | silence | 1 | 1/1 | — | — | — | — |
| apple | ja-JP | clean | 1 | 1/1 | 1136.64 | — | 12.50 | — |
| apple | ja-JP | long | 1 | 1/1 | 1129.82 | — | 2.82 | — |
| apple | ja-JP | noise | 1 | 1/1 | 1135.96 | — | 12.50 | — |
| apple | ja-JP | silence | 1 | 0/1 | 4033.90 | — | — | — |
| mimi-whisper | en-US | clean | 1 | 1/1 | 2761.17 | — | 0.00 | — |
| mimi-whisper | en-US | long | 1 | 1/1 | 2762.62 | — | 3.33 | — |
| mimi-whisper | en-US | noise | 1 | 1/1 | 2755.27 | — | 0.00 | — |
| mimi-whisper | en-US | silence | 1 | 1/1 | — | — | — | — |
| mimi-whisper | ja-JP | clean | 1 | 1/1 | 3771.86 | — | 12.50 | — |
| mimi-whisper | ja-JP | long | 1 | 1/1 | 3768.14 | — | 2.82 | — |
| mimi-whisper | ja-JP | noise | 1 | 1/1 | 3778.45 | — | 25.00 | — |
| mimi-whisper | ja-JP | silence | 1 | 1/1 | — | — | — | — |
| parakeet | ja-JP | clean | 1 | 1/1 | 743.61 | — | 25.00 | — |
| parakeet | ja-JP | long | 1 | 1/1 | 742.46 | — | 9.86 | — |
| parakeet | ja-JP | noise | 1 | 1/1 | 756.25 | — | 12.50 | — |
| parakeet | ja-JP | silence | 1 | 1/1 | — | — | — | — |
| phonon | en-US | clean | 1 | 1/1 | 498.33 | — | 0.00 | — |
| phonon | en-US | long | 1 | 1/1 | 498.08 | — | 3.33 | — |
| phonon | en-US | noise | 1 | 1/1 | 911.77 | — | 0.00 | — |
| phonon | en-US | silence | 1 | 1/1 | — | — | — | — |

## Reference text to translation

| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| mimi-translation | en-US | clean | 1 | 1/1 | 11.71 | — | — | 34.01 |
| mimi-translation | en-US | long | 1 | 1/1 | 37.01 | — | — | 56.74 |
| mimi-translation | ja-JP | clean | 1 | 1/1 | 15.26 | — | — | 82.71 |
| mimi-translation | ja-JP | long | 1 | 0/1 | — | — | — | — |

## Speech to translation

| Engine | Language | Condition | Units | Success / planned | Primary p50 ms | p95 ms | WER / CER % | chrF2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| apple | en-US | clean | 1 | 1/1 | 1146.13 | — | 0.00 | 34.01 |
| apple | en-US | long | 1 | 1/1 | 1147.30 | — | 0.00 | 29.54 |
| apple | en-US | noise | 1 | 1/1 | 1131.02 | — | 0.00 | 34.01 |
| apple | en-US | silence | 1 | 1/1 | — | — | — | — |
| apple | ja-JP | clean | 1 | 1/1 | 1150.09 | — | 12.50 | 35.76 |
| apple | ja-JP | long | 1 | 1/1 | 1148.01 | — | 2.82 | 65.83 |
| apple | ja-JP | noise | 1 | 1/1 | 1153.62 | — | 12.50 | 35.76 |
| apple | ja-JP | silence | 1 | 0/1 | 4032.88 | — | — | — |
| mimi-whisper | en-US | clean | 1 | 1/1 | 2773.22 | — | 0.00 | 34.01 |
| mimi-whisper | en-US | long | 1 | 1/1 | 2761.75 | — | 3.33 | 41.66 |
| mimi-whisper | en-US | noise | 1 | 1/1 | 2777.61 | — | 0.00 | 34.01 |
| mimi-whisper | en-US | silence | 1 | 1/1 | — | — | — | — |
| mimi-whisper | ja-JP | clean | 1 | 0/1 | — | — | 12.50 | — |
| mimi-whisper | ja-JP | long | 1 | 1/1 | 3776.36 | — | 2.82 | 65.83 |
| mimi-whisper | ja-JP | noise | 1 | 1/1 | 3784.12 | — | 25.00 | 35.76 |
| mimi-whisper | ja-JP | silence | 1 | 1/1 | — | — | — | — |
| parakeet | ja-JP | clean | 1 | 0/1 | 755.42 | — | 25.00 | — |
| parakeet | ja-JP | long | 1 | 1/1 | 780.05 | — | 9.86 | 65.26 |
| parakeet | ja-JP | noise | 1 | 0/1 | 758.73 | — | 12.50 | — |
| parakeet | ja-JP | silence | 1 | 1/1 | — | — | — | — |
| phonon | en-US | clean | 1 | 1/1 | 516.18 | — | 0.00 | 34.01 |
| phonon | en-US | long | 1 | 1/1 | 508.57 | — | 3.33 | 41.66 |
| phonon | en-US | noise | 1 | 1/1 | 936.28 | — | 0.00 | 34.01 |
| phonon | en-US | silence | 1 | 1/1 | — | — | — | — |

## Interpretation

- Synthetic speech does not establish human-speech accuracy.
- chrF2 is lexical overlap; these references do not authorize semantic-quality claims or promotion.
- Cold means fresh process; filesystem and driver caches are uncontrolled.
- Latency ends at model publication and excludes microphone capture and compositor paint.
- Missing observations are null. Tails require at least 20 independent groups and remain exploratory below 100.
- Process peak RSS is cumulative within a process, not isolated peak memory for each warm attempt.

Detailed preparation, queue/service/publication latency, RTF, memory, churn, route counts, confidence intervals, failures, and denominators are in `summary.json`. Raw attempts are in `attempts.jsonl`.
