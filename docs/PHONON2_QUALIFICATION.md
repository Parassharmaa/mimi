# Phonon 2 PR qualification

The complete local gate passes for PR #15. CI must also pass before merging.
The Whisper finalization limit remains 1.1 seconds.

## Historical evidence manifests

The corrected product reports were captured with executable
`0a71d7179b4a993137b817cf9863db47689e3bbcfb075a64dff9ce51f9af9c52`.
Their suite hashes refer to the original long-form manifests, which must not
change when a new audio materialization is evaluated.

The Phonon integration restored the same pinned FLEURS source revision and
24 registered source positions using a newer ffmpeg toolchain. This changed
the WAV file hashes and therefore the manifest hashes. The historical reports
were unchanged, causing CI run 36755668441 to reject the replaced manifests.

The original manifests from commit `4630a84` are now preserved as
`manifest-product-corrected-v2.jsonl` and
`manifest-product-silence-1s-corrected-v2.jsonl` in each language's long-form
directory. The historical verifier reads these versioned manifests. Current
adaptive reports continue to use `manifest.jsonl` and
`manifest-silence-1s.jsonl`. No recorded report hashes or thresholds changed.

## Latency failure and resolution

Two identical English gapless replays measured 1.251469 and 1.251162 seconds
of post-audio finalization. Both exceed the registered limit. The final
endpoint near 248.3 seconds performs a complete-utterance Whisper decode;
the failure is not a deliberate wait in the stop method.

The audio runtime reconstructed unchanged vocabulary suppression masks for
every generated token. Dependency commit `159c4d3` prepares these masks once
per transcription chunk and preserves their original addition order. Exact
old/new logits and greedy tokens passed for float32 and float16, including
overlapping and invalid IDs. The existing dense and 4-bit tied-embedding
projection checks also passed in a standalone Swift harness against the
production MLX objects. Weights, inference parameters and segmentation did
not change.

The first dependency package test build ran out of disk space before reaching
the tests. Only the newly created incomplete build directory was removed.
The harness reused Mimi's compiled MLX objects; the actual unit-test sources
are committed in dependency PR #1.

Final implementation and embedded executable identity:
`c1ce2c74060b24a47fc76c12a109b2b1e18972d604ddd5da575b30d16c05ade1`.

| Registered control | Error rate | Post-audio finalization | Compute RTF |
|---|---:|---:|---:|
| English gapless, selected first run | 12.36% WER | 1.000356 s | paced input |
| English gapless, prescribed repeat | 12.36% WER | 0.995789 s | paced input |
| Japanese gapless | 6.03% CER | 0.972101 s | paced input |
| English paused | 5.90% WER | direct control | 0.360 |
| Japanese paused | 6.57% CER | direct control | 0.377 |

The first English run is selected, rather than the fastest repeat. Both
prescribed runs must pass. The repeat is retained as
`phonon-optimized-repeat-paced-en24-6.json`; the failed original runs are
retained as `phonon-integration-first-paced-attempt-en24-6.json` and
`phonon-integration-second-paced-attempt-en24-6.json`.

All four controls retain the exact prior hypotheses and finalized segments.
Paced runs have zero drops and backpressure events. Historical and current
source-bound evidence verifiers pass without relaxed thresholds.

The full `scripts/test.sh` suite passed after these replays: model selection,
speech exclusivity, benchmark tools, both evidence verifiers, translation
pack/runtime validation, self-tests, deterministic speech/session E2E,
light/dark UI fixtures, and the main-window lifecycle check.
