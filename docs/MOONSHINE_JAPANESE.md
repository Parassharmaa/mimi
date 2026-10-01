# Moonshine Japanese streaming preview

Moonshine was evaluated as a research candidate, including a complete native
Swift integration prototype. The final workspace option uses Japanese Parakeet
instead, after its encoder-only Q4 variant improved the Japanese accuracy and
first useful-caption tradeoff. Moonshine remains accessible through developer
smoke commands for reproducible comparisons, but is not a workspace selection.

The model is Moonshine **Streaming Small Japanese**, 112.9 million parameters,
using the official native Swift/C++ runtime on CPU. This is the streaming model;
the publisher's older non-streaming Japanese Tiny/Base models have different
license terms. The streaming model and runtime are MIT. No Python, server, or
external transcription API is used by the application.

Eight optional model files total 121,803,780 bytes. Downloads are staged and
checked against pinned sizes and SHA-256 hashes before publication. Cancellation
removes the staging directory; removal unloads the native runtime before deleting
the installed pack. Assets use Mimi's existing Application Support location and
survive an application upgrade. No automatic model switch is performed.

Native inference lives on one actor. A bounded eight-second PCM queue connects
the existing audio capture routes to it. The first transcription pass follows
0.5 seconds of delivered audio; subsequent passes use a one-second cadence.
This schedules inference, not a promise that the model emits a useful word after
0.5 seconds. Empty hypotheses are possible and early text can be wrong.

The native VAD completes lines at speech boundaries or its 15-second maximum.
Mimi commits each completed line ID once and drains trailing input at Stop.
Cancelling a session invalidates its callbacks. The native `return_audio_data`
option is explicitly false: completed-line source audio is released rather than
retained and copied into every Swift transcript snapshot.

The preliminary 24-clip Japanese FLEURS screen used the registered seed-42
suite and verified every original audio hash. Final application accuracy was
**10.05% normalized character error**, with 24 nonempty finals. The standalone
C++ cadence screen measured 9.67%; the application uses 50ms capture buffers and
the Swift wrapper's forced final flush, so these are separate results. Older
screens on the same registered suite measured 6.57% for Mimi Speech Preview and
11.06% for Apple Speech; those are historical references, not freshly repeated
short-clip controls. Moonshine Tiny was rejected at about 15.2% CER despite its
lower compute cost.

The absolute preliminary first-text gate of one second was not met. The
standalone full paced screen measured 2.62 seconds median from recording start,
and 3.7ms median final flush. First nonempty text is not necessarily correct:
for example, one clip first produced `見て` and subsequently corrected it to
`インターネットで`. These results support an optional streaming preview, not
promotion to the default model or a claim of accurate instant recognition.

The final production timing repetitions and continuous-recording checks are
recorded in `Research/speech/fast-japanese-v1/runs`. The protocol identifies
first inference separately from two repeated recordings using an already loaded
model. English Phonon 2 uses a separate English corpus and is a speed reference;
it cannot provide a paired Japanese accuracy comparison. Whole-file Phonon RTF
also differs from its repeated live-window compute cost.

The CPU prepacking experiment is research-only. It preserved all 24 standalone
final transcripts and reduced unpaced mean compute RTF from 0.056 to 0.045. It
changes private native utility code and is not linked into Mimi. CoreML failed
on a zero-length decoder cache dimension and is not selected.

Pinned source, artifact, model, and dependency notices are recorded under
`App/Resources/SpeechLicenses/Moonshine` and in `MimiMoonshineModel.swift`.
Research scripts, raw failures, the interrupted audio-retaining run, and model
configuration are retained alongside the successful runs.
