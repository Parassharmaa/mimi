# Phonon 2 PR qualification

PR #15 remains a draft until the complete local and CI gates pass. The
Whisper finalization limit remains 1.1 seconds.

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

## Current latency failure

Two identical English gapless replays measured 1.251469 and 1.251162 seconds
of post-audio finalization. Both exceed the registered limit. The final
endpoint near 248.3 seconds performs a complete-utterance Whisper decode;
the failure is not a deliberate wait in the stop method.

The audio runtime reconstructs unchanged vocabulary suppression masks for
every generated token. This is a candidate for reducing CPU preparation
overhead without changing logits, decoder tokens, or segmentation. Any speed
change must be followed by fresh reports with the updated source identity and
the same quality, queue-loss, and finalization gates.
