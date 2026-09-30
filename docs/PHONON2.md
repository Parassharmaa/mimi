# Phonon 2 in Mimi

Phonon 2 is an English-only option in the recording model picker and in
Settings > Voice Type > Model. Japanese remains available through Mimi Speech
Preview and Apple Speech. Phonon is not selected automatically and never
silently falls back to another model.

Mimi ships a 424,191,836-byte native MLX pack. It preserves all 264 five-value
encoder weight matrices exactly as two 2-bit planes with FP32 scale tables.
Other weights use BF16, matching the packed reference runtime. Python is only
needed to prepare the pack on a developer's machine, not to run Mimi.

The pinned source is [FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2/tree/1c388bcec35d19740bf36b0b675718223fa7904e).
Weights use CC BY 4.0. The app includes NVIDIA/Fermion attribution, upstream
licenses and a Mimi modification notice.

## Local verification

On Apple M3 Pro, the native debug build transcribed three natural earnings-call
excerpts of 32 to 34 seconds. Prepared whole-file processing took approximately
0.73 to 0.89 seconds, with terminology and output consistent with the packed
reference. These are small exploratory checks, not population accuracy claims.

The 32.08-second gaming excerpt was also replayed through Mimi's production
audio queue. First nonempty text appeared at 0.738 seconds, and finalization
finished 123 ms after input ended. The run had 68 partial/final updates and no
backpressure or audio-drop warnings. These timings exclude model preparation,
microphone capture and text-field insertion. Preparation took 0.836 seconds
with local model files and cached Metal kernels on that run.

Do not substitute Fermion's optimized Python or M5 throughput claims for these
native Swift measurements. The native port uses the packed encoder to retain
predictable memory. Further optimization can be evaluated separately.

For Japanese, a natural FLEURS test clip produced English-looking phonetic
fragments in the upstream Phonon runtime, not a usable Japanese transcript.
The model card and publisher documentation list English only. Mimi therefore
limits Phonon to English in the UI and rejects Japanese before capture.

The existing Whisper regression recordings were re-materialized from the same
pinned FLEURS revision and registered case positions using the current audio
toolchain. Artifact hashes are refreshed with the new replay reports; source
case selection, references and quality/latency thresholds are unchanged.

The refreshed Whisper controls retain 6.0% Japanese gapless CER, 12.4% English
gapless WER, 6.6% Japanese paused CER and 5.9% English paused WER, with no
audio drops. Japanese paced finalization passed at 1.068 s. English paced
finalization measured 1.251 s in both the first attempt and an identical retry,
above the existing 1.1 s gate. Both attempts are retained. The full regression
gate is therefore not green, and this integration remains a draft preview.

## Reproduction

`scripts/speech/fetch_phonon2_pack.sh` verifies the source archive and converts
it with pinned Fermion/MLX dependencies. `verify_phonon2_pack.py` verifies the
native pack inventory, checksums, conversion proof and size limit. Normal and
universal packaging both include the same model and notices.

The development CLI provides `--smoke-phonon2 <model-root> --audio <file>
--warm-runs 3`, and `--smoke-phonon2-live <model-root> --audio <file> --output
<report.json>`. The model-selection and speech-exclusivity checks cover English
routing, Japanese rejection without fallback, and startup ownership across
Apple's asynchronous asset check.

Benchmark inputs are public [Earnings-22](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard)
and [FLEURS](https://huggingface.co/datasets/google/fleurs). Recordings are not
included in the distributable app.
