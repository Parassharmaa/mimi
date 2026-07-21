# Teacher-student distillation plan

The current students are the two directional ElanMT Marian models exposed as
one Mimi translation engine. They are small enough to ship after 4-bit MLX
quantization, but the checked-in non-claimable canary still trails Apple by
about 6 chrF++ in both directions. Distillation targets that gap; it does not
change the promotion gate.

The July 2026 literature review refines this into Diverse Quality-aware
Regularized Distillation (DQRD-v1): uncertainty-plus-diversity source selection,
reviewed diverse teacher outputs, frozen-base KL/L2 preservation, a mixed-domain
curriculum, checkpoint averaging, and an optional human-preference DQO stage.
See `literature-review-2026-07.md`. Reasoning traces, vocabulary trimming, and a
single shared student are not part of the first run.

## Teacher and judge roles

1. A high-reasoning GPT-5.6 teacher proposes three translations for licensed,
   non-evaluation source text. It emits a compact `translation_brief` with
   register, terminology, protected tokens, and ambiguity flags.
2. The pipeline never requests or stores private chain-of-thought. Structured
   facts and error tags are auditable; free-form reasoning traces are not.
3. Deterministic checks reject changed numbers, dates, URLs, placeholders,
   copied text, script failures, pathological length, and near-matches to the
   protected benchmark.
4. A fast or instant-class model distinct from the teacher blindly scores
   adequacy, fluency, terminology, and critical errors. Its output only
   prioritizes review in the promotion lane. In the reviewer-free provisional
   SFT lane, two distinct judge models must each identify the same uniquely best
   threshold-passing candidate with no error tags, critical error, or protected-
   token failure.
5. For promotion-eligible data, two independent bilingual reviewers receive separately ordered, teacher-
   blinded packets. They must select the same candidate, or a distinct third
   reviewer must adjudicate the disagreement. Automated-consensus rows may enter
   only the explicitly enabled provisional SFT arm; they cannot enter DQO or
   support an Apple-beating or shipping claim.

Batch request builders use the Responses API with Structured Outputs and
`store: false`. They only create JSONL and never upload or submit it. The
separate `run_synthetic_batch.py` first validates the source-only request
contract offline. Its networked `submit` command requires the exact file
SHA-256 as an explicit confirmation, persists the uploaded file and batch IDs,
and checks for an existing matching batch before creation. Collection requires
a completed batch and proves an exact one-to-one `custom_id` match across the
success and error files. Reproducible commands pin the tested OpenAI Python SDK
2.46.0 and persist its actual version in batch state. The Batch API is suitable
for the eventual run because
requests are asynchronous and returned rows must be joined by `custom_id`, not
output order.

`prepare_bilingual_review_packets.py` omits teacher identity, candidate style,
teacher risk tags, judge identity, judge scores, and priority rank.
`prioritize_distillation_judgments.py` requires exact source/candidate coverage,
a judge model distinct from the teacher, and valid bounded scores; it emits
only a complete risk ordering. `approve_automated_consensus.py` consumes two
such independently produced files and admits only matching, uniquely best,
error-free selections under conservative score floors. It marks them
`two-judge-consensus-provisional` and `promotion_eligible: false`.
`approve_bilingual_selections.py` fails closed on missing
or duplicate decisions, reviewer reuse, candidate/source mismatches, or more
than one selected target. It records both source-level reviews and any third-
reviewer adjudication. This prevents the student from seeing conflicting
"gold" targets. The structured brief is a compact supervision/debug artifact,
not a hidden reasoning trace and not part of the target sequence.

The corrected provisional mining run uses the exact 4-bit student to score a
deterministic 1,500-row KFTT pool per direction, keeps 900 hard-but-aligned rows
with chrF++ at least 10, and adds 63 unambiguous project-owned Mimi UI pairs per
direction. Four one-to-many UI source mappings are rejected. A separate
hash-sampled set of 300 CC BY 4.0 English BTEC utterances supplies spoken/travel
EN→JA teacher sources; it has no licensed Japanese reference and is therefore
never treated as parallel gold.
The licensed reference and weak-student hypothesis are retained locally for
selection and review but are excluded from the GPT request. The merged batch has
2,226 requests. The deterministic split can tolerate about 42.3% uniformly
distributed review rejection while retaining the required 500 train and 50
validation targets in each direction. Because the final held-out suite is not
frozen, these files are an unsubmitted pipeline rehearsal and must be
regenerated before a real batch.

The earlier 600-row mined subset was also run as a no-GPT control using its existing
professional/project-owned references. With 1,800 KFTT replay rows per
direction, EN→JA development chrF++ rose 29.47→30.11 but its fused-kernel 4-bit
canary fell 29.33→27.78, so that checkpoint is rejected. JA→EN development rose
47.75→49.59 and its fused-kernel canary rose 55.92→56.52. This asymmetric result is
useful: high-quality replay alone can help, but it does not close the domain gap
or justify attributing future gains to the teacher without an explicit control.

A broader licensed-parallel control then mixed 2,000 capped NICT ALT news pairs,
all 63 unambiguous Mimi UI train pairs, and 4,126 KFTT replay rows per direction.
After 150 steps, EN→JA development chrF++ rose 28.575→29.117 while its fused
4-bit canary fell 29.33→27.59. JA→EN development rose 49.664→51.063 and its
fused canary reached 56.50 versus the 55.92 base. This second asymmetric result shows why the teacher batch is
mined from weak, domain-relevant sources and why each direction is selected
independently rather than assuming a larger parallel corpus will transfer.

A third no-GPT control used reciprocal conversational Tatoeba pairs instead of
ALT. `prepare_tatoeba_parallel.py` rejects one-to-many mappings and protected
matches, then requires moderate reference agreement from both pinned students.
Because ElanMT documents Tatoeba exposure, that score is only a training-noise
filter and never evaluation evidence. Training on 1,386 screened Tatoeba pairs,
63 Mimi UI pairs, and 2,898 KFTT replay rows per direction produced a fused
4-bit canary of 31.31 EN→JA and 55.92 JA→EN, versus 29.33/55.92 for the base.
The direction-selected preferred-v1 pair used this EN→JA child and the
hard-reference JA→EN child, reaching 31.31/56.52. Preferred-v2 later retained
this EN→JA child and replaced only JA→EN with the licensed-unified average. It
still does not close the Apple gap or replace the need for independently
reviewed teacher data.

A fourth no-GPT control applied the intended preservation objective to that
same EN→JA dataset: frozen-base KL on KFTT replay, L2-to-base, and a linearly
ramped domain weight. Full-precision development chrF++ improved 30.351→30.895
at step 150, and the checkpoint averager selected the best adjacent window at
steps 150/200/250 without violating the KFTT retention tolerance. After
4-bit/group-64 conversion, the single and averaged students both scored 30.81
on the non-claimable EN→JA canary and produced identical hypotheses. Both trail
the current 31.31 EN→JA child and are rejected. The result validates the
regularization/averaging machinery, but also confirms that selection must be
made after the exact shipping quantization and decoder.

## Student training

- Start from the pinned openly licensed ElanMT checkpoints, not random weights.
- Mix independently approved conversational/UI examples with the human KFTT
  corpus and Mimi's paired shipping UI copy so domain adaptation does not erase
  general translation ability.
- Train all Marian weights in float16 or float32, choose checkpoints on a
  separate reviewed development set, then quantize the selected checkpoint to
  4-bit MLX. Never train against the promotion suite.
- Use sequence-level distillation because the API teacher does not expose a
  reusable full-vocabulary probability distribution. Candidate diversity and
  human selection provide a cleaner signal than pretending text rationales are
  teacher logits.
- Oversample hard categories seen in current errors: macOS commands, retained
  English product terms, disfluencies, politeness, negation, ownership idioms,
  numbers, dates, and short live-caption fragments.
- Re-run Apple and the student from cold preparation plus three warm repetitions
  per case. Promotion still requires the paired-bootstrap quality gate, no
  critical meaning errors, acceptable latency/RSS, and a model pack no larger
  than 150 MiB.

The implemented training path is:

1. `build_distillation_dataset.py` creates per-direction train/dev JSONL from
   reviewed teacher output, a deterministic sample of high-quality KFTT replay,
   and capped samples from multiple prepared parallel corpora, including the
   source-file-grouped Mimi UI set and human-translated ALT. It rechecks the
   protected suite and records content hashes. Its reviewed-diversity mode
   admits one optional alternative only after matching independent approval;
   training samples one approved target per source per epoch while validation
   remains canonical.
2. `train_marian_distillation.py` updates every Marian parameter, evaluates
   greedy translation on the reviewed development set, and can add frozen-base
   KL from an explicitly declared preservation checkpoint on declared replay
   origins, summed L2-to-preservation-checkpoint, and a linear domain-loss
   curriculum with an unchanged replay floor. `smoke_train_marian_distillation.py`
   exercises a complete regularized MPS optimizer update against a declared
   pinned student.
3. `average_marian_checkpoints.py` deterministically selects the best three
   adjacent checkpoints by either aggregate chrF++ or an unweighted macro over
   predeclared development origins, rejects any window beyond the predeclared
   KFTT replay regression tolerance, and arithmetic-averages the full-precision
   tensors with a hash-bound manifest.
4. `prepare_elanmt_mlx.py` quantizes the averaged full-precision checkpoint;
   `package_elanmt_mlx.py` assembles both directions under one app interface.
   Every derived direction now carries the authenticated training-manifest and
   dataset-manifest digests, target source, effective data licenses, and the
   required KFTT/Tatoeba attribution records into the pair manifest. KFTT- or
   Tatoeba-derived candidates remain explicitly blocked from distribution
   until the share-alike and per-contributor attribution release work is done.

The licensed-unified no-GPT run exercised this exact path with capped ALT,
KFTT replay, screened Tatoeba, and Mimi UI. Both directions selected and
averaged steps 150/225/300. After exact 4-bit conversion, EN→JA regressed the
canary and was rejected. JA→EN preserved every canary output and improved the
400-case public stress direction over the previous child by +0.67 mean sentence
chrF++ (95% paired interval +0.08…+1.35). The preferred-v2 pair adopts only the
JA→EN child; this result validates the direction-selection mechanism but does
not satisfy the independent held-out Apple promotion gate.

The optional single-physical-model lane is separate and cannot silently replace
the specialists. `build_bidirectional_dataset.py` creates an exactly balanced
licensed train mixture while leaving validation unrepeated.
`train_bidirectional_marian.py` routes EN→JA and JA→EN rows to separate frozen
specialist teachers for token-level KL, trains one shared Marian student, and
selects on the unweighted macro-average directional chrF++. The student uses
explicit `<2ja>`/`<2en>` markers, which `prepare_elanmt_mlx.py` carries into the
pack manifest and the Python benchmark applies at inference. A one-update MPS
smoke passes, but the first 300-step pilots are quality failures; this lane is
compression research until both directions beat the specialist development
baselines after exact 4-bit conversion.

The trainer also has an exact shipping-quantizer ablation via
`--mlx-fake-quantization-bits 4 --mlx-fake-quantization-group-size 64`. It
continues from a declared `--initial-checkpoint`, trains through the pinned MLX
affine float16 quantize/dequantize path with a straight-through estimator, and
saves the validation-selected raw weights for authoritative conversion. Two
licensed-data EN→JA pilots slightly improved quantized development but regressed
the non-claimable canary to 30.81 and 30.88, so neither replaces the current
31.31 child. QAT should be reconsidered only after reviewed teacher targets add
new domain signal.

## Reviewer-free local-teacher control

The user-authorized training-only lane is now executed without a human-review
dependency. It is deliberately stricter than a single model-as-judge loop:

1. preferred-v2, CAT q4, and HPLT-v2 must be three distinct forward engines;
2. the same three model families independently backtranslate survivors;
3. deterministic script, protected-token, length, source-copy, and protected-
   suite gates must pass;
4. a pinned NLI model must accept the source/backtranslation semantics; and
5. a separately calibrated Qwen3-8B judge must return perfect adequacy and
   fluency, preserved meaning, no critical error, and no error tags.

The first EN→JA funnel retained 256 of 2,000 CC-BY-4.0 BTEC sources. All rows
are permanently `promotion_eligible: false`, never become DQO preferences, and
carry the original source license/provenance. The Qwen lane uses native batch
generation and a shared prompt-prefix K/V cache; it never requests or stores
reasoning traces.

Two regularized students improved full-precision and public-v2 development but
regressed the exact 4-bit canary. Linear parent/adapted checkpoint interpolation
then found a 15% adapted blend that preserves the canary and improves the
400-case EN→JA conversation slice by +0.17 mean sentence chrF++ (95%
+0.04…+0.35); overall public-v2 remains inconclusive. This blend is packaged as
developer-only preferred-v3. It is useful evidence for scaling high-quality
teacher targets in both directions, not product-promotion evidence.

The next matched experiment starts from 1,785 professional KFTT references
that are absent from the active base train sets and screened against both
protected suites. Qwen3-8B sees only source/direction and produces final
translations; pinned COMET-22, chrF++, baseline-delta, script, token, length,
copy, repetition, and contamination gates retain 27 EN→JA and 40 JA→EN rows.
Each teacher arm is matched by a control that uses the identical sources with
the professional reference target.

EN→JA Qwen/control both select step 75 at 30.908/30.872 development chrF++,
but their provenance-authenticated exact 4-bit packs produce identical 29.996
canary scores, down from preferred-v3's 31.305. The earlier unauthenticated
30.808 Qwen score is superseded by the hash-bound, exactly repeated rerun.
JA→EN Qwen and control both select step 0 because all trained checkpoints
regress aggregate validation. No arm reaches public-v2 or app integration.
The quantized Qwen arm is not stronger than the matched human control in this
tiny hard-row dose; neither supplies a promotion candidate. More accepted,
balanced domain coverage is required.

## Human-preference DQO, gated after SFT

The optional DQO arm is implemented but cannot run merely because preference
rows exist. `build_dqo_preferences.py` creates a pair only when the same
canonical candidate was selected by two independent bilingual reviewers. A
candidate is eligible as the loser only when neither reviewer selected it or
approved it as a valid diverse alternative. Adjudicated disagreements and
automated judge ranks never become preference pairs.

`evaluate_supervised_win.py` then requires a positive paired lower confidence
bound on reviewed development chrF++, a positive blind-human score bound, zero
candidate critical errors, at most 0.5 chrF++ general-retention regression, and
an integrity match from the full-precision checkpoint through the exact 4-bit
pair manifest. It produces `supervised-win-approved` only if every gate passes.

`train_marian_dqo.py` refuses to start without that report and its exact
checkpoint SHA-256. It uses the frozen winning SFT checkpoint as the reference
policy, length-normalized sequence log probabilities, the standard
chosen-versus-rejected log-ratio objective, and a small chosen-target SFT anchor.
No reasoning traces, automated preferences, held-out cases, or adjudicated
preferences are accepted. DQO remains an ablation: its output must repeat every
development, retention, quantization, parity, performance, and Apple-promotion
gate before it can replace the supervised candidate.

## License and provenance

Every seed carries its source license and provenance through the review queue.
Only distributable CC0, CC BY, CC BY-SA, or compatible project-owned source
text is eligible. Teacher model/revision, response ID, system fingerprint,
prompt hash, reviewer IDs, and final adjudication are retained. The resulting
model release must preserve ElanMT CC BY-SA 4.0 attribution and KFTT attribution
in Mimi's third-party notices.
