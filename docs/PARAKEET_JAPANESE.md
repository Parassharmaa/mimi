# Fast Japanese transcription preview

Choose **Parakeet Japanese Preview (482 MB)** in Mimi's workspace model picker.
Download the optional model in Settings, then record Japanese normally. Its
replaceable drafts feed the live transcript and English translation immediately;
completed speech regions become immutable final segments. Apple Speech remains
the initial selection. Mimi Speech remains the higher accuracy Japanese option.
This preview applies to workspace transcription and captions; Voice Type keeps
its existing model choices.

The Japanese specialist is NVIDIA's Hybrid FastConformer TDT-CTC 0.6B, using a
custom MLX conversion. The encoder uses affine Q4/group64; prediction, embedding,
joint, and other residual tensors retain BF16 precision. The two verified files
total 481,857,364 bytes. This is 61.3% smaller than the pinned 1.25 GB BF16 input.
It is a Mimi derivative rather than an official NVIDIA quantized release.

Model inference runs on Apple Silicon with the existing native MLX Audio Swift
runtime. Native Silero VAD comes from the official Moonshine SDK, operating in
skip-transcription mode; Moonshine ASR weights are not needed. This is a bounded
live-window recognizer, like the English Phonon path, rather than an ASR encoder
with persistent streaming caches. Inference begins after 350ms of a detected
speech region and repeats every 500ms. Speech regions are limited to 15 seconds.
An eight-second input queue and a 40-second actor buffer bound retained PCM.
Completed native VAD source audio is released. No capture WAV is saved by this
live path, and no transcription API or Python process is used by the app.

The model pays its first-use Metal compilation during preparation with discarded
silent inference. This happens before capture starts. The first recording may
therefore show Preparing for longer, while subsequent recordings reuse the loaded
model. If a hypothesis becomes empty, its displayed draft is cleared; blank
native finals cannot leave an old provisional caption behind. Cancellation rejects
stale callbacks, and Stop drains the final input exactly once.

Assets download into a staging directory and pass pinned size/SHA-256 checks
before installation. Cancelling removes staging data. Removal unloads the model
and deletes only the installed optional pack. Installed models survive app
upgrades in Mimi's existing Application Support folder.

The registered 24-clip Japanese FLEURS screen verified all original audio hashes.
Normalized character error was 8.89% for the BF16 parent, 10.52% when every eligible
module was quantized, and 9.05% for the selected encoder-only Q4 variant. Keeping
prediction modules in BF16 recovered most of the accuracy at a 7.2 MB size cost.
The longer VAD look-behind experiment regressed to 18.79% and was rejected.

The selected variant's 331.56-second continuous replay measured 9.28% character
error, with no dropped audio. The prewarming investigation, repeated timing
results, memory measurements, source/executable hashes, and all failed or
superseded configurations are retained under `Research/speech/fast-japanese-v1`.
First nonempty text is a draft and can be wrong; it is not a measure of first
correct-word latency. Timings start at prerecorded audio delivery and exclude UI
paint. First speech-detection latency is reported separately from recording-start
latency. English Phonon uses a different English corpus and cannot provide a
paired Japanese accuracy comparison. Its blank trials remain failures in the
report rather than being counted as fast successful recognition.

These are selected local screens and continuous replays, not population accuracy
or a promotion gate for the default model. All weights, conversion modifications,
license terms and attribution are documented in
`App/Resources/SpeechLicenses/ParakeetJapanese`. Model weights are CC-BY-4.0;
MLX and the native VAD runtime retain their respective notices.
