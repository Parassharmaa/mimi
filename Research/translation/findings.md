# Translation experiment findings

Last updated 2026-07-21. These results are research evidence, not a quality
claim. The checked-in 12-case canary has no independent bilingual adjudication
and every row is explicitly ineligible for promotion claims.

The 2026-07-21 literature refresh and next experiment are specified in
`strategy-lexically-constrained-distillation-2026-07-21.md`. The decision is to
use lexical constraints and accepted final translations for sequence
distillation, never teacher reasoning traces, and to reject or advance the
single bidirectional M2M-100 418M baseline on a small frozen gate before porting
it to MLX.

## Apple baseline on the development Mac

Hardware: Apple M3 Pro, 36 GiB RAM. Operating system: macOS 26.5.1. Apple
Translation used the high-fidelity strategy with installed English and Japanese
assets.

| Direction | chrF++ | p50 latency | p95 latency |
| --- | ---: | ---: | ---: |
| English→Japanese | 37.91 | 1.141 s | 1.416 s |
| Japanese→English | 61.79 | 1.149 s | 1.442 s |

The corrected warm-run Apple harness reported 104.7 MB peak process RSS and
0.311 s preparation. Quality is measured once per case; p50/p95 use three warm
repetitions so first-use asset/session setup does not distort the latency gate.

## MLX model experiments

| Candidate | Required model files | Peak RSS | EN→JA chrF++ | JA→EN chrF++ | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-0.6B 4-bit base | 351.4 MB | 686.7 MB | 13.89 | 40.39 | Fast but below Apple and above the size target. |
| Qwen3 1,000-step Tatoeba LoRA | about 363 MB | 685 MB | 20.92 | 48.10 | Best Qwen checkpoint, still below Apple and too large. |
| SmolLM2-135M local 4-bit base | 79.4 MB | 276.9 MB | 0.44 | 10.32 | Fits easily, but English-centric base cannot translate Japanese. |
| SmolLM2 4,000-step Tatoeba LoRA | 89.2 MB | 281.0 MB | 5.43 | 16.23 | Learned target scripts but not adequate meaning; rejected. |
| SmolLM2 2,000-step KFTT LoRA | 89.2 MB | 306.8 MB | 4.79 | 27.52 | Professional data helps JA→EN, but frozen English-centric embeddings cap EN→JA; stopped and rejected. |
| Old OPUS Marian pair, FP32 | 553.7 MB | 1,103.2 MB | 4.21 | 11.01 | Bible-domain checkpoints are unusable on live speech. |
| ElanMT-BT pair, upstream FP16 | 247.2 MB | 968.0 MB | 27.48 | 55.57 | First coherent compact baseline; above size target before quantization. |
| ElanMT-BT pair, fused MLX 4-bit pack | **73.4 MB** | **199.5 MB** | **29.33** | **55.92** | Exact deployed-kernel baseline; quality remains below Apple. |
| ElanMT hard-reference control, fused 4-bit | 73.4 MB mixed pack | 199.7 MB | 29.33 | 56.52 | EN→JA uses base; JA→EN is the best direction-specific control. |
| ElanMT licensed KFTT+ALT+UI control, fused 4-bit | 73.4 MB | 199.6 MB | 27.59 | 56.50 | ALT-assisted EN→JA regressed; JA→EN remains below the hard-reference control. |
| ElanMT screened conversational control, fused 4-bit | 73.4 MB | 199.7 MB | **31.31** | 55.92 | Best EN→JA direction; standalone JA→EN does not improve the fused base. |
| Direction-selected DQRD preferred-v1, fused 4-bit | **73.4 MB** | **199.7 MB** | **31.31** | **56.52** | Former development pair: conversational EN→JA plus hard-reference JA→EN behind one interface. |
| Preferred pair, 4-bit/group-32 | 81.0 MB | 207.5 MB | 25.63 | 51.26 | Rejected: smaller quantization groups regressed both directions. |
| Preferred pair, 4-bit/group-128 | 69.6 MB | 196.0 MB | 27.11 | 55.20 | Rejected: smaller pack, worse quality in both directions. |
| Preferred pair, 6-bit/group-64 | 103.6 MB | 230.7 MB | 27.95 | 55.57 | Rejected: higher precision regressed both directions. |
| Preferred pair, 8-bit/group-64 | 133.8 MB | 260.3 MB | 28.55 | 55.57 | Rejected: higher precision regressed both directions and approached the model cap. |
| Preferred 4-bit/group-64 pair, beam 2 | 73.4 MB | 200.2 MB | 30.40 | 54.16 | Rejected: slower and worse than greedy. |
| Preferred 4-bit/group-64 pair, beam 4 | 73.4 MB | 201.0 MB | 31.00 | 54.16 | Rejected: upstream-style beam search did not improve the canary. |
| Regularized conversational EN→JA + preferred JA→EN, best single checkpoint | 73.4 MB | 200.1 MB | 30.81 | 56.52 | Rejected: full-precision development improved, but the shipping-kernel canary regressed. |
| Regularized conversational EN→JA + preferred JA→EN, averaged checkpoints | 73.4 MB | 200.1 MB | 30.81 | 56.52 | Rejected: averaging steps 150/200/250 produced the same canary outputs as the single checkpoint after 4-bit quantization. |
| Regularized-parent exact-MLX QAT EN→JA + preferred JA→EN | 73.4 MB | 199.9 MB | 30.81 | 56.52 | Rejected: quantized development improved 30.616→30.651, but all canary translations matched the rejected regularized parent. |
| Shipping-best-parent exact-MLX QAT EN→JA + preferred JA→EN | 73.4 MB | 199.9 MB | 30.88 | 56.52 | Rejected: quantized development improved only 30.533→30.537 and the shipping canary regressed from 31.31. |
| Licensed-unified regularized averaged pair, fused 4-bit | 73.4 MB | 206.7 MB | 29.82 | 56.52 | Directional result: EN→JA rejected after canary regression; JA→EN preserved the canary and improved the 400-case stress slice. |
| Direction-selected DQRD preferred-v2, fused 4-bit | **73.4 MB** | **206.6 MB** | **31.31** | **56.52** | Former development pair: unchanged conversational EN→JA plus licensed-unified regularized averaged JA→EN. |
| HPLT v2 Transformer-base pair, FP16 | 448.9 MB | 1,136.1 MB | 36.55 | 55.91 | Rejected off the shelf: attractive EN→JA canary result collapses to 18.37/38.90 on the 400-case-per-direction stress evidence. |
| M2M100-418M multilingual model, FP16 | 1,941.9 MB | 4,463.3 MB | 24.10 | 47.65 | Rejected before stress: worse than the bilingual specialist in both directions and too slow; int4 size alone would not repair quality. |
| CAT-Translate-0.8B MLX 4-bit | 453.0 MB | 873.8 MB | 34.42 | 56.94 | One MIT-licensed bidirectional model; canary improves, but 800-case stress and COMET both lose to preferred-v2. |
| Hy-MT2-1.8B MLX sparse ternary | 464.2 MB | 810.3 MB | 35.62 | 60.70 | One Apache-2.0 bidirectional model; attractive canary is rejected by 22.64/44.09 on the matched 800-case stress set, opaque training-data rights, and a custom community runtime. |
| LMT-60-0.6B MLX 4-bit/group-64 | 346.9 MB | 745.6 MB | 31.38 | 54.15 | One Apache-2.0 Qwen3-based bidirectional model; rejected by 17.92/40.42 on the matched 800-case stress set, 141 critical-token failures, and incomplete training-data rights lineage. |
| CAT conversational/UI JA adapter, step 100 | 466.5 MB | 874.1 MB | 28.28 | 59.66 | Directional canary gain only: JA→EN conversation improves, overall stress stays 42.39 and EN→JA regresses; rejected. |
| Translate-15L T5-small, FP16 beam 4 | 244.6 MB | 956.1 MB | 0.00 | 4.11 | Apache-2.0 and fast only on short failed outputs; empty/repeated punctuation plus 2.60 s JA→EN p95 reject it before MLX porting. |
| Strict local-teacher EN→JA v2 + preferred-v2 JA→EN | 73.4 MB | 213.6 MB | 30.81 | 56.52 | Rejected as a full child: public-v2 EN→JA improves 30.13→30.30, but the canary regresses. |
| Direction-selected DQRD preferred-v3, fused 4-bit | **73.4 MB** | **212.7 MB** | **31.31** | **56.52** | Developer preferred: 15% local-teacher blend preserves the canary and improves the 400-case conversation slice; still not product quality. |

The SmolLM2 Tatoeba adapter was trained on variable-quality auxiliary data. Its
negative result motivated the KFTT-first run. It also showed that physically
using one generic decoder is not a useful goal if its tokenizer cannot model
Japanese well. The selected 73.4 MB pack instead contains two specialized tiny
students behind one bidirectional Mimi interface.

## Decoder acceleration result

Incremental greedy decoding now caches decoder self-attention and encoder
cross-attention K/V tensors in both Python MLX and Swift MLX. On the release
Swift app binary, 30 warm repetitions per canary row produced:

| Direction | Full-prefix p50 / p95 | KV-cache p50 / p95 | p95 speedup |
| --- | ---: | ---: | ---: |
| EN→JA | 50.9 / 54.1 ms | 29.4 / 32.4 ms | 1.67× |
| JA→EN | 49.0 / 61.9 ms | 27.7 / 34.2 ms | 1.81× |

Both paths match all 12 canary hypotheses. The 800-case public stress audit
shows 800/800 exact Python/Swift generated-token parity, and expanded v2 passes
2,400/2,400. Eight v1 and sixteen v2 rendered
JA→EN strings differ only in spaces before punctuation because Swift Tokenizers
and Transformers decode the same token sequence differently; the same rows
differ under full-prefix Swift, ruling out the cache. Actual Swift-string
scores are 33.2325/53.9897 chrF++ versus Python's 33.2325/53.9954. The model
pack remains exactly 73,403,714 bytes.

Preferred-v3 independently passes 2,400/2,400 exact generated-token parity from
its corrected minimal 73,402,252-byte pack. The packager now regenerates each
child manifest from the three physical payload files instead of retaining
source-conversion entries for files deliberately pruned from the pair.
The signed universal preferred-v3 candidate archive is 124,846,501 bytes at
SHA-256 `b7ab1de8b5af596a3d433559a625166b561acdb144f2c0ff1bceb0bc1674b598`;
the generated distribution report verifies all model and MLX 0.30.6 shader
bytes and passes the 150,000,000-byte download cap.

The remaining no-weight partial-caption idea does not pass the same identity
bar. Parallel teacher-forced verification of the previous target accepts only
1/240 drafts on coarse 50/75/100% source-growth traces. With finer 5% source
increments it also produces a false acceptance: the full-sequence verifier
keeps the prior 128-token EN→JA target while cached greedy decoding diverges at
token 7. This is a kernel/numerical-path mismatch even at zero bias. The
failure artifact SHA-256 is
`f054a4d05530394fdefed552bcea25daa771a2a1fbc1e850071d62779bfa423f`.
The branch is rejected without a Swift port; finalized-segment inference keeps
the existing exact K/V cache.

## Reviewer-free local teacher result

The local data funnel uses the MIT-licensed CAT q4 model only as the candidate
teacher. Preferred-v2 and CC-BY-4.0 HPLT-v2 are independent forward filters;
all three models independently backtranslate survivors. A pinned Apache-2.0
English NLI model then requires at least 0.9 entailment in both directions and
at most 0.1 contradiction. Finally, the Apache-2.0 Qwen3-8B 4-bit model judges
the exact source/candidate pair with a calibrated bilingual rubric and accepts
only adequacy=5, fluency=5, preserved meaning, no critical error, and no error
tags. No chain-of-thought is requested or retained.

The measured funnel is 2,000 BTEC sources → 542 surface-consensus rows → 283
roundtrip/NLI rows → 256 Qwen-approved targets. Qwen rejected 27 subtle errors
missed by the earlier filters, including possession/availability loss,
"checked luggage" rendered as inspected luggage, generic "check" for hotel
check-out, a computing sense of "platform," and minibar negation loss. Every
accepted row is marked `promotion_eligible: false`; its JSONL SHA-256 is
`4b4b19f706db904a0e1b46d0e1f5692eee39b75e544824cc51a758982a7edf86`.

Native MLX batch generation plus a shared 614-token system-prompt K/V prefix
preserved 16/16 judgments against the uncached path and cut the 16-row smoke
from about 40 seconds to about 13 seconds. The complete 283-row pass used batch
size 16, about 4.96 GB peak RSS, and a 4,623,784,971-byte research-only judge.
Neither the judge nor its cache enters Mimi's bundle.

The first regularized student selected step 75 at 30.847 full-precision
development chrF++ and reached 30.25 on public-v2 after 4-bit conversion, but
the canary regressed 31.31→30.81. A stronger 2→4 synthetic-loss curriculum with
doubled KL/L2 preservation selected step 100 at 30.853 and reached 30.30 on
public-v2, but repeated the canary regression. A seven-point interpolation
line search found one stable shipping point: 15% adapted / 85% parent. It
preserves all canary outputs and produces 30.17/55.95 public-v2 chrF++; its
EN→JA conversation delta over preferred-v2 is +0.17 mean sentence chrF++ (95%
+0.04…+0.35), while all-domain EN→JA is inconclusive (+0.07, -0.02…+0.15).
This is enough for a developer preferred-v3, not for product promotion.

## Current KFTT experiment

The pinned KFTT archive produced short, duplicate-controlled, protected-suite-
checked bidirectional rows. KFTT translations were produced and checked by
professional translators. The training split is never used as Mimi's promotion
suite, and the public KFTT test split is only an external smoke test because
pretrained-model contamination cannot be ruled out.

The first KFTT run confirmed that LoRA cannot repair SmolLM2's English-centric
tokenizer and frozen embeddings. The run was stopped after the saved 2,000-step
checkpoint once EN→JA remained repetitive and meaning-invalid.

The replacement students are the two 61M-parameter ElanMT-BT Marian models.
Their model cards document exclusively openly licensed training data, including
KFTT, Tatoeba, WikiMatrix, MDN, Wikimedia content translation, and a CC0
Wikidata parallel corpus. A direct MLX implementation exactly reproduced the
verified PyTorch output before quantization. Four-bit affine quantization of
linear layers and the shared embedding produces a minimal 73,403,427-byte pair.
Unlike a generic decoder, the Marian architecture preserves the dedicated pad
decoder-start embedding and generates coherent translations.

The signed universal Mimi app without translation weights is 79,870,674 bytes;
adding this pack gives 153,274,101 bytes (146.2 MiB) before `mlx.metallib` and
notices. Thus the requested sub-150-MB model target has ample room, while a
separate sub-150-MiB installed-app target is not yet claimed. The model-free
signed model-free ZIP is 18,638,719 bytes and passes its generated SHA-256
check (`7f2d34fe2c982e176e0d980321eb84291023756eb395aa36a0bc04b88c2de831`).
A temporary app clone with the exact pack under `Contents/Resources` was
re-signed and verified at 153,277,003 installed bytes; its ZIP was 83,429,490
bytes. This proves the weights are practical to distribute inside the app, but
that first measurement did not include `mlx.metallib` and does not promote or
bundle the model in Mimi's current release.

The missing-runtime check was later closed with the official prebuilt MLX
0.30.6 shader, exactly matching the pinned Swift runtime. The shader is
128,008,745 bytes. A temporary re-signed universal app containing both it and
the exact model pack occupied about 281.3 MB on disk and compressed to
124,734,275 bytes. The new structured release build then embedded the
direction-selected pair under `Contents/Resources/TranslationModels`, verified
every model/shader byte, passed universal `x86_64 arm64` and code-signature
checks, and produced a 124,820,906-byte ZIP at SHA-256
`6a167a14b16496e6a16ec64fcf7bf3722c716843d93c9535e46c62d73829e347`.
Thus the exact combined download clears the strict 150,000,000-byte cap, while
a sub-150-MB installed app is not claimed. This is development evidence, not a
promoted release.

The next quality experiment is sequence-level teacher-student distillation. A
high-reasoning GPT-5.6 teacher proposes multiple outputs plus structured facts,
not chain-of-thought. Deterministic filters are mandatory. The claim-eligible
lane uses independent bilingual review; the user-authorized training-only lane
requires the same unique error-free candidate from two distinct automated judge
models and permanently labels every selected row promotion-ineligible. See
`distillation.md`.

The exact 4-bit students have now mined a deterministic expanded provisional
seed set from licensed training data: 900 weak-but-aligned KFTT examples plus
63 unambiguous Mimi UI pairs in each direction. Selected KFTT chrF++ ranges from 10.02 to
35.41 (median 22.22) for EN→JA and 10.00 to 51.60 (median 35.51) for JA→EN.
Adding 300 hash-sampled CC BY 4.0 BTEC spoken/travel source utterances yields
2,226 source-only GPT-5.6 requests totaling 5,672,274 bytes. No request has been
uploaded or submitted. The sealed request contract validates offline at SHA-256
`7895c05a23ebd904bc2529e9c929d50d9679a9b6dbb107f9533194ff08dcd768`.
The pre-review split is 1,146/117 EN→JA
and 867/96 JA→EN, which permits about 42.3% uniformly distributed rejection
before missing the 500/50 train/dev minimum. The manifest deliberately marks
the set provisional, because mining and contamination screening must be
repeated after the final held-out suite is frozen.

The licensed-corpus pass also pinned NICT ALT (CC BY 4.0) and retained 16,011
short, unique human-translated pairs after conservative filtering. Dataset
assembly can now accept multiple parallel corpora but caps each at 2,000 train
and 200 validation rows per direction, so ALT news text cannot swamp reviewed
live-speech examples or Mimi UI copy. ParaNatCom is professionally translated
and CC BY 4.0, but remains outside the first run because its abstracts are not
sentence-aligned and its scientific register is a poor match. BSD remains
excluded because its noncommercial license cannot support shipping Mimi.

Before spending the teacher budget, both directions were trained once against
the licensed references in the earlier 600-row subset, mixed with 1,800
KFTT replay rows. Each direction had 2,465 post-deduplication train pairs and
414 validation pairs. EN→JA development chrF++ increased from 29.47 to 30.11 at
step 200, but the fused-kernel quantized canary fell from 29.33 to 27.78; it is rejected.
JA→EN development increased from 47.75 to 49.59 at step 150 and the fused
quantized canary increased from 55.92 to 56.52. The mixed base-EN→JA/tuned-JA→EN
pack is 73,403,570 bytes. It remains
development-only: the canary is non-claimable and the JA→EN score still trails
Apple's 61.79.

A second no-GPT control tested broader licensed parallel coverage rather than
only the mined hard subset. Each direction used 2,000 deterministically sampled
ALT train pairs, all 63 unambiguous Mimi UI train pairs, and 4,126 KFTT replay
rows; validation contained 200 ALT, 13 UI, and 921 KFTT rows. At step 150,
EN→JA development chrF++ rose 28.575→29.117 but the fused quantized canary fell
29.33→27.59. JA→EN development rose 49.664→51.063 and its fused canary reached
56.50 versus the 55.92 base. The exact two-direction pack is 73,403,570 bytes.
These results reinforce the need
for domain-relevant teacher examples and direction-specific checkpoint
selection; the canary is non-claimable and neither result changes promotion.

The next no-GPT control replaced ALT with conservatively screened conversational
Tatoeba. From a deterministic 3,900-pair scoring pool, the gate retained 1,537
reciprocal pairs after rejecting 1,484 ambiguous source mappings and 2,363 pairs
with insufficient agreement in at least one direction. ElanMT documents Tatoeba
in its pretraining, so model agreement is used only to remove likely noisy
training pairs and is explicitly not independent quality evidence. Each
direction then trained on 1,386 screened Tatoeba pairs, 63 Mimi UI pairs, and
2,898 KFTT replay rows. At step 150, EN→JA development chrF++ rose
30.351→30.579 and JA→EN rose 49.573→50.315. Shipping-kernel reruns scored
31.31/55.92 for the conversational pair, versus 29.33/55.92 for the fused base.
The independently selected hard-reference JA→EN direction scores 56.52, so the
new preferred logical pair combines conversational EN→JA with hard-reference
JA→EN and reaches 31.31/56.52. The canary remains non-claimable and the student
still trails Apple in both directions.

The same conversational EN→JA data was then used to exercise the full DQRD
regularization and checkpoint-averaging path without synthetic targets. Frozen-
base KL on KFTT replay, L2-to-base, and a 0.25→1.0 domain curriculum improved
the 1,085-case full-precision development score from 30.351 to 30.895 at step
150. The deterministic best adjacent window was steps 150/200/250. After exact
4-bit/group-64 conversion, however, both the best single checkpoint and the
three-checkpoint arithmetic average produced 30.81 EN→JA on the canary, below
the preferred 31.31; their six canary translations were identical. JA→EN stayed
56.52 because that direction was unchanged. Both 73,403,570-byte packs are
rejected, demonstrating again that full-precision development gains must be
confirmed after the actual shipping quantization and decoder.

An exact quantization-aware continuation was then implemented from the pinned
MLX 0.30.6 affine kernel rather than a generic INT4 approximation. The trainer
casts source weights to float16, reproduces MLX's group-64 signed scale and
zero-aligned edge adjustment, stores scale/bias at float16 precision, leaves
Linear biases floating point, excludes computed positional embeddings, and
uses a straight-through gradient. A value-level comparison against MLX,
tied-weight round trip, cache invalidation, and real MPS update/save/reload all
pass. Starting from the regularized parent, 100 QAT steps improved the
quantized 1,085-case development score 30.616→30.651 but produced the same six
EN→JA canary outputs and 30.81 score as its parent. Starting from the 31.31
shipping-best child at half the learning rate selected step 50, moving
quantized development only 30.533→30.537; its exact 73,403,599-byte pair scored
30.88/56.52. Both are rejected. The mechanism is retained for future reviewed
teacher targets, but more optimization on this licensed mixture is not
supported by the evidence.

The offline pipeline and one-update MPS training smoke now pass end to end,
including two-reviewer agreement, disagreement queuing, and independent third-
reviewer adjudication. The data builder requires one accepted target per source,
deterministically splits reviewed examples, samples KFTT replay, records
input/output hashes, and scans
again for protected-suite near matches. The trainer updates all Marian weights
and selects a checkpoint on reviewed-development chrF++. The Batch runner's
upload/create/status/collection lifecycle also passes against an offline SDK
fixture, including request-hash confirmation and exact collected-ID checks. No
real teacher batch has been submitted because this worktree has no API
credential. The optional human lane still requires its specified bilingual
reviews; the executed local lane instead uses strict multi-engine consensus and
Qwen judging, permanently marks accepted rows promotion-ineligible, and cannot
feed DQO.

The DQRD-v1 implementation now has executable controls rather than only a
literature plan. The legacy claim contract requires exactly 400 cases per
direction plus human attestations; reviewer-free development proceeds on the
larger public surface, but cannot mark it claim-eligible. Any replacement claim
suite still requires document-ID and text-level separation from every supplied
training file. The new hybrid source selector derives
sequence NLL and normalized mean-pooled encoder states from the exact 4-bit MLX
student, then samples across uncertainty thirds with greedy cosine k-center
coverage. A real KFTT/UI smoke passed for both directions against the preferred
73.4 MB pack. The Marian trainer's full MPS smoke also passed with frozen-base
KL, L2-to-base, a domain-loss curriculum, and an evaluated checkpoint artifact;
the standalone best-three-adjacent checkpoint averager passes its tensor/hash
contract. These are pipeline results only: the independent product-domain
held-out suite does not yet exist, so no quality or Apple-beating claim follows.

The claim suite was expanded from 400 to 800 distinct cases (400 per direction)
before authoring began. This doubles every domain stratum; the smallest
code-switching slice is now 20 rather than 10 distinct cases per direction.
The decision favors statistical power and slice visibility over annotation
cost: repeated ratings do not replace distinct translations as the effective
sample size, and classic paired-bootstrap work shows that even 300-sentence
samples can produce unstable significance conclusions. See
[Graham et al. (2020)](https://aclanthology.org/2020.emnlp-main.6/) and
[Koehn (2004)](https://aclanthology.org/W04-3250/).

The reviewed-diversity arm is also executable: an alternative target survives
only when both bilingual reviewers independently approve the same distinct
candidate, the dataset retains at most two audited variants on one source row,
the trainer samples one target deterministically per epoch, and development
scoring remains canonical. For learned evaluation, Mimi pins Apache-2.0
`Unbabel/wmt22-comet-da` at revision
`371e9839ca4e213dde891b066cf3080f75ec7e72` with
`unbabel-comet==2.2.7` and float32 mean-over-reference scoring. The legacy
promotion evaluator requires a positive paired COMET lower bound in both
directions in addition to chrF++ and blind human gates; XCOMET is excluded
because its checkpoint license is noncommercial. Reviewer-free development
reports the same automatic metrics but cannot silently bypass that gate.

The final planned training arm is now executable and fail-closed as well.
Conservative DQO preference construction uses only two-reviewer consensus and
never treats an approved diverse alternative, an adjudicated disagreement, or
an automated judge rank as a negative example. A separate supervised-win
evaluator binds reviewed development metrics, blind human scores, general
retention, zero critical errors, the full-precision checkpoint, and the exact
quantized pair manifest. The DQO trainer refuses to start unless that artifact
is approved and still matches the starting weights. The reviewer-free local
rows are explicitly barred from this path, so no DQO training has run and no
result is claimed.

The MLX benchmark runner now verifies every manifest-listed file before loading
the model and records the exact pair-manifest SHA-256 as `modelRevision`. A real
preferred-pack canary rerun produced the same non-claimable 31.31/56.52 chrF++
scores while binding its 73,403,570-byte pack to revision
`pair-manifest-sha256:48c2e256e309377f89a9ed8dd102a8d27d945e35c743798406d42dacfc7ddeb8`.
This closes an integrity field required by the promotion evaluator; it does not
change the failed quality gate.

The optional instant-model lane is now operational rather than advisory: its
Batch input has a separate sealed contract, its output must cover the exact
three candidates for every source, and a teacher-identical judge is rejected.
One judge can only order review work. The reviewer-free provisional SFT gate
requires two distinct judge models to choose the same uniquely best error-free
candidate; rows are permanently marked promotion-ineligible and remain barred
from DQO. In the claim-eligible lane, judge identity, scores, and rank are
removed from bilingual packets and approval still requires two matching human
selections or an independent third adjudicator.

The claim-ready benchmark is now operational rather than only declarative.
Separate tools create blinded reference-review packets, require a distinct
adjudicator, bind final references by hash, independently blind Apple and
candidate outputs for two bilingual scorers, and evaluate every promotion gate.
A quota-exact 800-case contract fixture passes when the candidate is strictly superior and
is rejected when one reviewer flags one candidate critical error. The evaluator
also refuses missing/misaligned cases, changed references, unmatched hardware or
OS, stale output assignments, insufficient warm runs, an oversized model,
excess memory, slow p95, or a failed non-Apple fail-closed artifact.

The repo also contains a clean project-owned domain source that does not need a
teacher: paired English/Japanese strings already shipping in Mimi's UI. The
extractor found 76 unambiguous pairs after language, duplicate, one-to-many, and
protected-suite checks, with 63 grouped into train and 13 into validation by
source file. Four conflicting source mappings are excluded. These
small, precise pairs supplement KFTT; they do not replace the larger reviewed
live-speech distillation set.

A historical 50-update EN→JA full-model ablation mixed the earlier 66-row UI
extraction with 198
deterministically sampled KFTT rows. Reviewed/public development chrF++ rose
from 28.38 to 28.87, but the separately screened 4-bit canary fell from 31.28
to 27.82. The checkpoint is rejected. This demonstrates why checkpoint
selection needs a domain-representative reviewed development set and why the UI
corpus should remain supporting replay rather than the main supervision signal.

## Licensed-unified regularized direction selection

The next no-GPT experiment used one reproducible licensed mixture in each
direction: 2,000 capped NICT ALT news pairs, 2,586 KFTT replay pairs, 1,386
reciprocal-agreement-filtered Tatoeba conversational pairs, and 63 Mimi UI
pairs. Validation contains 200/921/151/13 rows from the same four origins,
never repeats rows, and was independently screened against the protected
canary. Each training row retains source, license, provenance, origin, and
domain metadata.

The EN→JA train/validation JSONL hashes are
`0e0cbcb6d1fc4883ab3359763e324a91a0bccc7482818ec9c5fc091fc4e33ea3` /
`ab2b957f9b000f182bd8c459d34da34db02105e42e97e1efb6979097acdacc71`;
the JA→EN hashes are
`81615ff015931411b4563e1364e1f44ff9c5ae07de435eaf4e9c2bc942d7dafb` /
`00752124ad81c1f670362ec632420016e7e2aaf75251cd2ecc13b25d78b91096`.

Both specialists started from the direction-selected preferred-v1 child and
used that same checkpoint for frozen KL and L2 preservation. Non-KFTT rows had
a constant 2.0 loss weight, KFTT was the retention slice, and checkpoints were
saved every 75 updates through step 300. Selection maximized the unweighted
macro chrF++ across ALT, Tatoeba, and Mimi UI, subject to at most 0.5 chrF++
KFTT regression. The selector chose steps 150/225/300 in both directions and
arithmetic-averaged the full-precision tensors before exact MLX 4-bit/group-64
conversion. The EN→JA and JA→EN averaging manifests hash to
`626efc5e182aa8bb2c62e169f97854b21139b338e224a2b428c8f950223d8a6e`
and `bee6dd96814b80b471c88d452b62afba3f0e2baee81425a2833bc829c646e6fa`.

The full averaged pair is 73,403,858 bytes. EN→JA improves the 400-case public
stress direction from 33.24 to 33.45 corpus chrF++, but its paired sentence
delta over preferred-v1 is inconclusive (+0.37, 95% -0.11…+0.87) and its
non-claimable canary regresses 31.31→29.82, so that child is rejected. JA→EN
preserves every canary hypothesis and improves the public stress direction
53.57→53.95; its paired sentence delta is +0.67 (+0.08…+1.35). Preferred-v2
therefore keeps conversational-control EN→JA and changes only JA→EN to the
licensed-unified average. The resulting pair is 73,403,714 bytes, passes exact
12/12 Swift/Python parity, and is revision
`pair-manifest-sha256:6e5d8515b887944507ccb9c71634ae58f9471c257340da2776b25fb4f03f972c`.

This is a better development pair, not a promotion result. On the public
stress suite, preferred-v2 scores 33.24/53.95 and leads Apple overall by paired
mean sentence chrF++ +8.22/+5.54, but it still loses independently sourced ALT
news by -3.35/-4.87 with both 95% intervals below zero. The signed universal
archive embeds the exact pair plus MLX 0.30.6 shader, passes byte-for-byte
distribution verification, and is 124,847,075 bytes at SHA-256
`db4841203b312f5a6202d553fbdc9c00adbfae7541fdcfc52e4d7bc142f581df`.
The current app's Apple behavior remains temporarily unchanged and all fallback
checks pass; the intended final product path is validated local translation
with a non-Apple failure mode.

## Expanded-capacity baseline

The size envelope now prefers 150 MB but permits at most 500 MB when accuracy
requires it. HPLT v2 is the first larger dense control because both directional
checkpoints are CC BY 4.0, Marian-native Transformer-base models. EN→JA is pinned
at revision `0b07a399bf25965dc344fad25e7826c38bec53e6`; JA→EN is pinned at
`89c256961b845b265d2a1393883375eb47d79600`. Their native model SHA-256 values
are `940fcbc187435a6fe313e844df99255adcd86f9ca6d16697f27a2ab9fb901759`
and `ef6cc4648d585749ae5bbcc703f2962c5477251d42cb82368f503d17c5fd68e4`.
The new conversion staging script verifies the shared 64K SentencePiece order,
creates Transformers-compatible vocabulary/config files, and uses the upstream
Marian converter; the resulting FP16 pair is 448,893,412 bytes.

HPLT EN→JA initially looked strong on the six-case canary at 36.55 chrF++, but
the 400-case stress direction scored only 18.37: conversation 21.10, ALT news
27.40, and KFTT Wikipedia 13.41. HPLT JA→EN scored 55.91 on the canary and
38.90 on stress: 49.78/50.04/30.05 by the same domains. Both directions lose
Apple and preferred-v2 broadly. FP16 p95 reached 0.38/0.34 seconds with about
1.06 GB peak RSS. The unchanged HPLT pair is rejected before MLX porting; it may
remain a fine-tuning initialization or diverse teacher, but parameter count
alone did not improve translation.

The next dense control was the single MIT-licensed
`facebook/m2m100_418M` checkpoint pinned at
`55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`. It is structurally capable of
fitting the 500 MB ceiling only after aggressive 4-bit conversion: its 128,112
token embedding alone is about 262 MB at float16. The actual FP16 research run
is 1,941,931,012 bytes and reaches only 24.10/47.65 on the canary, with
0.49/0.39-second p95 latency and 4.46 GB peak RSS. Because it loses the 60.6M
parameter bilingual specialists before quantization, the unchanged M2M100 model
is rejected without spending time on an MLX port or public-stress run.

CyberAgent's newly released `CAT-Translate-0.8b` is a stronger architectural
match: one MIT-licensed bidirectional model trained specifically for Japanese
and English. The pinned community MLX q4 conversion at
`84cbdd97cf628fa98fcd5a757d2599ebee765cd7` is 453,006,430 bytes. It beats
preferred-v2's tiny canary in both directions (34.42/56.94), but the result does
not survive the 800-case stress suite: 24.90/42.47 chrF++, 0.369/0.442-second
p95, and about 874 MB peak RSS. Pinned COMET-22 confirms preferred-v2 is also
semantically stronger overall, scoring 0.8800/0.8449 versus CAT's
0.8618/0.8034. CAT only has a small COMET advantage on the news slices.

A contamination-screened 12,067-row licensed QLoRA control rejected three
near-overlapping training rows and evaluated steps 50/100/150/200; every
checkpoint regressed at least one canary direction. A second 2,898-row
Tatoeba/UI-only run used a 5× lower learning rate and twice the gradient
accumulation. Step 100 improves JA→EN canary 56.94→59.66 and the conversation
stress slice 57.63→61.63, but overall JA→EN stress is flat/slightly lower at
42.39, Wikipedia regresses, EN→JA falls to 28.28, and p95 grows. Its exact base
plus adapter footprint is 466,466,510 bytes. It is rejected for integration;
CAT remains a possible MIT-licensed diversity teacher.

## Encoder-heavy shallow-decoder speed ablation

The first literature-motivated depth experiment preserves all six parallel
encoder layers and keeps only decoder layers 0 and 5. MLX now infers variable
encoder/decoder depth from authenticated weights, and reproducible pruning
tools cover both quantized packs and full-precision recovery checkpoints.

Direct pruning reduces the two-direction 4-bit pack from 73,402,252 to
54,333,716 bytes. It destroys the converged representation, however: repetitive
outputs run to the 192-token bound, canary chrF++ is 0.1444/0.0750, and p50/p95
cached-decoding latency is 0.172/0.180 seconds EN→JA and 0.170/0.181 seconds
JA→EN under 30 matched warm runs. Fewer layers therefore do less work per token
but far more tokens end to end.

A second pilot applies 300 steps of supervised plus full-teacher logit
distillation to EN→JA. It authenticates 4,346 training and 1,085 validation rows,
uses every KFTT/Tatoeba/Mimi origin for `KL(full preferred || shallow student)`,
and never requests or stores reasoning traces. The known pathological step-0
full generation is skipped and declared; the only selection evaluation covers
all 1,085 validation rows with a 64-token cap. Development chrF++ recovers to
only 4.9813. Exact 4-bit cached canary is 1.9863 with
0.162/0.174-second p50/p95 under the same warm-30 protocol, and
the shallow-EN→JA plus intact-JA→EN pack is 63,869,086 bytes. It is rejected
before public-v2, COMET, Swift parity, or integration.

This does not refute a trained encoder-heavy architecture; it rejects naïve
post-hoc depth pruning and a short recovery schedule. The next valid arm must
train the shallow decoder from initialization, progressively drop teacher
layers, or distill intermediate representations in addition to output logits.

## Professional-reference local teacher ablation

The July teacher suite freezes 1,785 professional CC-BY-SA-3.0 KFTT pairs
(892 EN→JA, 893 JA→EN) after excluding active base-training sources, protected
canary/public-v2 rows, and near protected 5-gram matches. Its SHA-256 is
`b87f3ea53699a35ba816723bdfcb05e907ca31676e0d1576726251dbf1c2eca8`.
The exact preferred students provide a frozen baseline. Qwen3-8B receives only
source text and direction, never the reference or student hypothesis, and no
reasoning is requested or stored.

The full 1,785-row Qwen pass used batch 16 with a shared prompt-prefix cache,
took 1,551.7 seconds, and peaked at 4,975,116,288 bytes RSS. Pinned COMET-22,
chrF++, positive deltas over the student, token/script/length/copy/repetition
checks, and protected-suite screening retain 67 translations: 27 EN→JA and 40
JA→EN. The accepted JSONL SHA-256 is
`2f0724bffb2609c239e8df56940cedcfaaaa7bdaa68f1d13c4c8d21a2a347e5e`.

Matched teacher and professional-reference controls use the same source rows,
initial checkpoints, replay, optimization, and unchanged base validation.
EN→JA selects step 75 in both arms: Qwen reaches 30.9079 development chrF++,
the reference control 30.8720, and step 0 is 30.5951. After rebuilding the exact
4-bit/group-64 artifacts from corrected provenance manifests, the 73,410,348-
byte Qwen and 73,410,410-byte control pairs produce identical canary hypotheses
at 29.9961 EN→JA, below preferred-v3's 31.3055. Their authenticated delta is
-1.3573 mean sentence chrF++ with a 95% paired-bootstrap interval of
-3.1190…-0.0295. JA→EN selects step 0 in both arms; all steps 50/100/150
regress from 52.1805 aggregate development chrF++. The earlier unauthenticated
30.8081 Qwen result is superseded because it did not bind the compared reports,
suite content, or model revisions; the corrected Qwen run repeats exactly.
The rebuilt pair-manifest SHA-256 values are
`43595742e79da091426d6c4d0c11e137630b3b2a09c60f6dab0a9ce70d544ddd`
(Qwen) and
`1dd2c88c0c52211d0534824e92ca1c5c3eae52b45ae45d2aba34f85e06fef98e`
(control). Both embed the authenticated dataset-manifest digest, effective
licenses, Qwen teacher revision where applicable, KFTT's required notice, the
Tatoeba per-row attribution obligation, and an explicit distribution blocker.

All four arms are rejected. No public-v2, COMET candidate evaluation, Swift
parity run, app promotion, or fallback change is warranted. The experiment
shows that a strict local 8B teacher can improve full-precision development on
the tiny hard-source dose, but its quantized student is indistinguishable from
the matched human-target control on this canary and both regress. The next
teacher set needs more accepted examples and more
conversation/news/UI coverage, not reasoning traces or higher loss weight.

That expansion is now frozen before teacher filtering. The authenticated
licensed inventory retains 325,860/320,721 KFTT, 14,497/12,516 ALT, and
8,717/8,411 Tatoeba candidates for EN→JA/JA→EN after active-data,
prior-teacher, ambiguity, duplicate, and protected-suite exclusions. The exact
preferred-v3 pack scores 600 deterministic candidates in each of six
domain/direction cells; uncertainty-stratified encoder-cosine k-center selection
keeps 400 per cell. The resulting 2,400-row suite is evenly divided across
directions and conversation/news/Wikipedia, has SHA-256
`98da175c5a7d937afd280fec0db23757702c74dc8dd64f43e1eb3b2cd48d1198`,
and preserves 800 rows each of CC-BY-2.0-FR, CC-BY-4.0, and CC-BY-SA-3.0
provenance. A predeclared ten-row minimum in every domain/direction cell
prevents an aggregate teacher pass from hiding a collapsed slice.

The source-only Qwen3-8B generation completed all 2,400 rows in 1,598.2 seconds
at 4,973,641,728 bytes peak RSS. The strict reference/QE/structure filter found
281 potential improvements: 37/87 conversation, 49/92 news, and only 9/7
Wikipedia rows for EN→JA/JA→EN. Because both Wikipedia cells miss the
predeclared minimum of ten, the round is rejected as a unit and the filter does
not emit training data. The failure report SHA-256 is
`61a94ade4675cf9ebad7096dc570b3f57e83c8c9bf591ce48d7559adcc5d0e17`.

The predeclared retry path subsequently recovered the cell floor without
weakening a score threshold. It reused 1,600 unaffected rows exactly,
regenerated only the 800-row Wikipedia slice, and selected an alternative only
when it passed every frozen COMET-22, chrF++, baseline-delta, structure, and
language gate. A second literal EN→JA retry and one-case number/proper-noun
probes yielded no additional admissible candidates. The only validator repair
canonicalizes English month names with Japanese numeric months and excludes
that month numeral from the otherwise exact plain-number multiset; mismatched
months still fail. Policy v2 therefore admits the high-quality `August`↔`8月`
case without making numeric preservation permissive.

The final training-only set contains 290 rows: 96 EN→JA and 194 JA→EN, with
36/87 conversation, 50/94 news, and 10/13 Wikipedia examples. The authenticated
JSONL SHA-256 is
`f28b51052655d6fd4958fdaaadb561872de86cace2fcc03fa9893edff9f42382`.
Its EN→JA and JA→EN training manifests have SHA-256
`fc5a0afa7197806c0ed4fdb58cb39900c3c55d732489dc87d74bb62f81815717`
and
`9c3dbed41fd28301c7e06394210898f77bd0b35546627e0328ebdb43933e0e8d`.

The EN→JA student selects step 50 at 30.819 development chrF++, versus 30.595
at step 0. Exact 4-bit interpolation shows that only the 1% adapted / 99%
parent arm preserves all canary outputs; 2%, 5%, and 10% are rejected there.
A paired public-v2 rerun first exposed output drift in the untouched JA→EN
child because an ad-hoc environment had resolved MLX 0.32.0 instead of Mimi's
pinned 0.30.6 runtime. Reports now bind the benchmark and Marian-runtime source
hashes plus Python, MLX, Transformers, tokenizer, warm-up, cache, and position
contracts; same-engine mismatches fail before scoring. Under the authenticated
MLX 0.30.6 shipping contract, the 1% EN→JA arm changes 28/1,200 token sequences
and regresses by -0.07895 mean sentence chrF++ (95% paired interval
-0.14684…-0.02044). The untouched JA→EN child has 0/1,200 token changes. The
authenticated comparison SHA-256 is
`34b8f4b9eb9daddad4e266e18cd9f16845111f5d11ef62285c29ae45f128f409`.

JA→EN is clearer: step 0 remains best at 52.180 development chrF++, while
steps 50/100/150/200 reach 52.157/52.100/52.108/52.106. The selected output is
therefore the unchanged parent and needs no redundant conversion. Both
synthetic arms are rejected; no Swift integration, app default, or fallback is
changed.

The same frozen sources then support an all-human reference control. Adding
1,200 licensed balanced rows per direction to the existing replay mixture
raises EN→JA full-precision development from 30.5951 to 30.8713 at step 100;
JA→EN selects the exact step-0 parent because every trained checkpoint is
lower. The authenticated 4-bit pair is 73,411,676 bytes and has pair-manifest
SHA-256
`b3a5aff93abc1fb85dfb1b4f4ac31288b4275c1ee8590c139f410f426663150b`.
It improves public-v2 EN→JA by +0.3805 mean sentence chrF++ (95% paired
interval +0.1531…+0.6232), including +0.7106 on Wikipedia
(+0.2418…+1.2229), but regresses the six-case canary by -0.4765 and alters a
protected number/entity translation. A 5/10/15/25/50% parent-interpolation
sweep cannot isolate the broad gain: the first four blends repeat the same
-0.447 canary delta and 50% falls to -1.357. The entire family is rejected
without Swift integration or a default/fallback change.

Selective task-vector merging is also closed. Whole encoder, whole decoder,
tied embedding/output, and encoder-plus-decoder components independently
trigger a protected regression. Every one of the twelve single-layer merges
changes a canary output as well; decoder layer 5 is the least harmful at
-0.0295 mean sentence chrF++ but still degrades the macOS/UI case by -0.1771.
No component is eligible for public-v2 selection, and the regression canary is
not used to tune a per-weight mask.

## High-data shallow-student and sequence-KD result

The decoder-depth speed hypothesis now has a controlled high-data result. The
authenticated EN→JA builder selects 72,061 unique licensed sources and emits
165,050 repeated training rows: 50,000 KFTT, 56,000 ALT, 56,000 Tatoeba, and
3,050 project-owned UI rows. Validation is never repeated. The builder rejects
protected-suite and validation overlap before writing data and preserves row
license, provenance, attribution, and original identity. Its manifest SHA-256
is `864c66d6c873db96547847cc668bed1a51347e9ee56cf8f4357d762404aa5f07`.

The source-only sequence-distillation pass authenticates the full-precision
preferred-v3 teacher and the input dataset, deduplicates repeats before
generation, and never exposes human references or chain-of-thought. It accepts
72,050 unique teacher targets and rejects eleven immediate-EOS generations;
the repeated train split contains 165,039 rows. The output manifest SHA-256 is
`8d8f4ecc44f1e1009ff3a3c25914a0d917fba3f09ef0f4c6b0f255422ab8cfac`.

Matched 6-encoder/2-decoder training confirms that sequence targets help but
cannot compensate for the capacity cut. Human references reach 6.383, 9.974,
11.962, and 12.288 development chrF++ at steps 250/500/750/1,000. Sequence KD
reaches 6.754, 10.490, 12.401, and 12.823. This is consistently better at every
checkpoint and far better than the earlier 4.981 pilot, yet still unusable
against the intact parent's roughly 30.6 development score.

A 6-encoder/4-decoder arm is the first real latency/quality Pareto candidate.
It improves from 12.302 at initialization to 26.541/27.184/27.120/26.815 at
steps 250/500/750/1,000. Step 500 is best; averaging the best adjacent
500/750/1,000 checkpoints is reproducible at averaging-manifest SHA-256
`a2105a8f8c585c7ecb3c9fbd2a7b38494d4a224bdfd7bcedc48aea08938d92c3`.
After exact MLX 4-bit conversion, however, the single checkpoint scores only
27.057 EN→JA canary chrF++ versus 31.305 for preferred-v3. Its paired mean
sentence delta is -3.693 (-11.789…+2.554); the average is worse at 24.641.
The single checkpoint does reduce warm EN→JA p50/p95 from 29.30/31.86 ms to
22.64/25.86 ms and the mixed pair from 73,402,252 to 71,074,502 bytes, but peak
Python RSS increases from 205.8 MB to 229.9 MB. Quality rejects both before a
public-v2 run. Neither is quantized for JA→EN, packaged for Swift, or promoted.

This experiment also closes the proposed reasoning-trace route: canonical
final-sequence distillation already transfers the useful teacher signal without
reasoning, improves every matched shallow checkpoint, and remains auditable.
The remaining gap is architectural capacity and domain quality, not absent
private reasoning. The effective KFTT/Tatoeba training licenses also keep these
artifacts blocked pending share-alike and attribution review; the research
result is not a shipping-license decision.

Two source-capacity reallocations do not change the decision. Adding two
zero-residual/unit-layer-norm encoder blocks fails because Marian is post-norm:
the supposedly identity initialization scores 0.487, then only 4.592 and
13.708 at steps 250 and 500. The run is stopped at the predeclared comparison
point and never quantized. In contrast, widening every encoder FFN from 2,048
to 4,096 is exactly output-preserving at initialization: the additional `fc1`
features copy existing activations while their new `fc2` columns start at zero.
All six initial EN→JA canary outputs match the normal-width 6/4 model.

The wide arm reaches 26.567/27.135/27.208/27.245 development chrF++ at
250/500/750/1,000 steps. Its +0.061 best gain over normal width is too small
to survive exact MLX quantization: 4-bit canary chrF++ is 23.592, a significant
-7.751 mean sentence delta from preferred-v3 (-15.680…-0.898). Although the
78,177,088-byte mixed pair remains faster at 24.93/28.73 ms EN→JA p50/p95,
quality rejects it before public-v2. The source reallocation tools and MLX
loader now authenticate variable encoder FFN widths, but no Swift model or app
configuration consumes this rejected architecture.

## Live retranslation speed result

Self-speculative reuse of the previous caption translation was evaluated as a
zero-byte, zero-bias runtime alternative. Parallel teacher-forced verification
plus first-divergence cache continuation is not bit-stable with the incumbent
incremental MLX kernel: an unrestricted 128-token JA→EN partial draft produced
a different token at position 23. The guarded variant bypasses drafts over 64
tokens and falls back exactly when no prefix is accepted. It restores all
canary outputs, but early 25/50/75% source prefixes cause pathological
95–192-token generations on nearly every case, so useful draft acceptance is
zero. Total speed ratios are 0.992× EN→JA and 1.013× JA→EN. The rejection report
SHA-256 is
`b0f49fe43ec52e75e70f2e5d85260642722f4323f922cd81eb44eb3cbdb3101c`.
This is an EOS/prefix-training failure before it is a verifier optimization
problem. No Swift or app path changes.

## Release-clean four-engine audit

The first exact 148 MB routed pack was not actually release-clean. A recursive
hash-bound lineage audit traced its EN→JA generalist through a 15% checkpoint
interpolation to 256 local-teacher rows explicitly marked
`promotion_eligible: false` and `training_only: true`. The corrected release
contract records `blocked-promotion-ineligible-training-data`. On 1,400
public-v3 EN→JA cases, the interpolation's +0.062 mean sentence chrF++ advantage
over its human-only parent is inconclusive (-0.013…+0.146), so the parent is the
appropriate release candidate.

Re-fitting the formal router against that parent strengthens its grouped test
result from +0.398 to +0.432 mean sentence chrF++ (+0.153…+0.733), routing
160/386 cases and no conversation cases. The JA→EN model and legal router are
unchanged. The neural replacement
`elanmt-release-clean-human-routed-moe-v2-mlx-4bit-pack` is 148,075,038 bytes;
generated attributions and the release contract add 247,494 bytes, for
148,322,532 bytes combined. The trace authenticates 10 dataset files, five
training manifests, 264,300 dataset-row occurrences, and 9,305 unique Tatoeba
notices, with zero promotion-excluded weight-training rows. Four-engine peak RSS
is 401,031,168 bytes and router p95 is 0.101/0.052 ms EN→JA/JA→EN on the canary.

The portable router now has an independent native Swift implementation. Across
all 2,800 public-v3 cases it reproduces 2,800/2,800 Python route decisions with
a maximum absolute score delta of 5.11e-15. A temporary authenticated pair of
the two expert engines also passes 12/12 Swift/Python output-token parity under
cached decoding; the unchanged generalist pair retains its prior 12/12 result.
The developer-only Swift engine now accepts and hash-validates the MoE manifest,
loads the selected role, and passes cold generalist/expert smokes in both
directions without adding a user-facing switch.

On all 2,800 public-v3 cases, routing improves the human-only generalist by
+0.837 mean sentence chrF++ EN→JA (95% paired interval +0.613…+1.089) and
+1.295 JA→EN (+1.003…+1.605), reaching 29.18/53.44 corpus chrF++. The router
selects 588 EN→JA and 196 JA→EN experts before a strict URL/placeholder/markup/
digit-preservation guard sends 11 and 15 cases back to their generalists.
Conversation remains exactly unchanged. This is non-claimable public evidence.

The original router report counted 212 expert-selected cases where both expert
and generalist violate strict critical-token equality. That is not the total.
The initial complete audit found 520, but 22 were tokenizer artifacts caused by
sentence-final punctuation. The corrected audit finds 498 mismatches: 136
EN→JA expert, 101 EN→JA generalist, 61 JA→EN expert, and 200 JA→EN generalist-
derived outputs (including nine already tagged as critical-token fallback).
The Swift runtime now validates every neural path and has executable fail-closed
evidence for both directions, with only the proven single-percentage equivalence.
A broader number/negation/structure heuristic flags 707
routed outputs. These conservative flags mix real failures with valid number-
word, era-year, and formatting transformations, so they are not semantic error
rates and cannot weaken the zero-critical-error gate.

A reversible placeholder attempt does not solve the problem. Generic numeric
and URL labels cover 711/2,800 sources, but restoration fails in 328 generalist
and 329 expert outputs. Restored quality drops by -2.004 mean sentence chrF++
EN→JA and -3.067 JA→EN with wholly negative 95% intervals. The preprocessing
arm is rejected and is not present in the Swift runtime or release pack.

A deterministic exact-pack Apple diagnostic aligns the public-v2 Apple report
and public-v3 routed report by immutable source/reference content, leaving 647
post-hoc claim-ineligible cases per direction. chrF++ favors the local route by
+7.818 EN→JA (+6.050…+9.584) and +2.456 JA→EN (+0.954…+3.990), and local p95
is 56/65 ms versus 2.17/2.27 seconds. Pinned COMET-22 is nevertheless
inconclusive overall at -0.00265 (-0.00855…+0.00324) and -0.00371
(-0.01017…+0.00290). News is significantly worse in both directions, and
JA→EN conversation is also significantly worse. This is useful diagnostic
evidence and a clear rejection of an overall superiority claim.

Those smokes also prevented a false sense of completion. For the official legal
heading `（立入調査等）` / “(Site Inspection, etc.)”, the JA→EN generalist emits
“(Interesting research)” and the routed legal expert emits “(Interest Survey)”.
The implemented exact-memory ablation repairs it to the observed train-only
human medoid “(On-site Inspections)” without inspecting the test reference.

The memory builder NFKC-normalizes and collapses Unicode whitespace, requires
an exact source in at least two distinct laws, discards a document when that
document maps one normalized source to conflicting targets, selects only an
observed human target medoid, caps source/target length at 64/128 characters,
and requires exact critical-token preservation. The corrected v2 tokenizer
handles sentence-final integers without swallowing punctuation and emits 6,179
entries: 676 EN→JA and 5,503 JA→EN. The minified runtime is 615,743 bytes,
SHA-256 `d0cdc2416cf7d65a83b0914be96a94eb80812ef660edf3ad81672de18461633e`;
the complete deterministic audit is 635,338 bytes, SHA-256
`2f96c7542de1fc6dd09e341c8bdf5f647e7db622bb903bee9f72d75d0131229e`.

The threshold-independent validation slice contains every exact memory match in
the untouched law-grouped validation data: 9 EN→JA and 213 JA→EN. Against the
routed neural baseline, mean sentence chrF++ gains are +13.591 (95% interval
+0.573…+24.174) and +23.962 (+21.034…+26.957). Public-v3 is opened only after
that gate: it has one EN→JA and six JA→EN hits, all improved. Overall deltas are
+0.035 EN→JA (0…+0.105) and +0.167 JA→EN (+0.044…+0.319); legal deltas are
+0.246 and +1.171. These are retrieval results, explicitly separate from neural
generalization.

The v2 validation slice contains 9 EN→JA and 213 JA→EN matches. The native
implementation checks memory before routing, validates every entry
and manifest/provenance hash, and exactly matches Python on all 2,800 lookup
decisions and hypotheses. Cold neural role smokes still pass for the four
generalist/expert combinations, and a memory smoke selects “(On-site
Inspections)”. The resulting development pack is 148,691,509 bytes. Its release
evidence, including the full memory audit, is 885,471 bytes, for 149,576,980
combined—423,020 bytes below 150,000,000.

The four physical engines carry the same authenticated 2,400,891-byte
`tokenizer.json` (SHA-256
`84b75e5fb6540c393026cd01b212acdf8769f32df8d7adc88c8b748610401b7f`).
A fail-closed v2 repacker now verifies the complete source file table and all
four engine tokenizer records, refuses any mismatch or symlink, preserves
weights, routers, memory, configs, and lineage bytes, and stores that tokenizer
once at the pack root. The memory-v2 candidate falls from 148,691,509 to
141,488,564 bytes, saving 7,202,945 bytes. Regenerated hash-bound release
evidence is 885,505 bytes, so the complete 142,374,069-byte distribution set has
7,625,931 bytes of headroom below 150,000,000.

The fresh v2 residency artifact conservatively records 421,003,264 bytes peak
RSS while all four models are loaded; a same-runtime repeat observed
401,195,008 bytes. The larger value is retained rather than treating allocator
variation as a memory improvement. This size accounting covers the model pack
and required release sidecars, not a future signed app archive.

The developer-only Swift loader supports both layouts and authenticates the
shared payload before constructing the tokenizer. Cold v1/v2 smokes match
exactly for both generalists, both experts, and a translation-memory hit; the
four neural cases match all 37 decoder output token IDs. The parity report
SHA-256 is
`5eb077873d26c9e9339aaeafeb2366aea7ab0eebd5d5eacbc824afa830fa5090`.
This is a lossless packaging optimization, not new quality evidence, and it
does not change the normal app path or any promotion blocker.

The strict public-v3 final-output audit is now 498/2,800 failures rather than
520: the previous regex missed a sentence-final integer immediately followed
by punctuation. A broad typed-number policy could rescue 140 reference-
validated cases, but accepted 27 cases whose reference signature disagreed, so
it is rejected. A separately pre-gated single-explicit-digit-percentage rule
rescues one case with zero disagreements and passes adversarial Swift/Python
contracts; general word-number, kanji-number, ordinal, scale, and era
relaxations remain disabled.

The experimental app path now has executable non-Apple failure evidence. A
local load, integrity, token, or inference failure keeps the local lane selected,
preserves successful local translations, exposes a retryable error, and leaves
the source transcript visible. Floating live partials show source text instead
of invoking Apple while the local opt-in is active. The non-experimental app
default remains unchanged until every promotion gate passes.

This is still not promoted. The runtime memory is human and PDL-licensed, but
its source rows explicitly say `training_only: true` and
`promotion_eligible: false`. The hash-bound release contract therefore records
`blocked-training-only-runtime-memory-and-final-review`, and both pack and
memory retain `doesNotAuthorizeAppIntegration: true`. Final license/app review,
the independently sourced automated 400-distinct-case-per-direction suite, a
typed critical-error policy, and final license/app review remain required.

## Reviewer-free claim contract

The no-human-review authorization is now represented by a separate executable
contract rather than weakening the historical human benchmark. The new
`automated-claim-v1` manifest fixes exactly 400 newly sealed project-domain
cases per direction and retains the 120/80/60/60/60/20 domain allocation. It
forbids existing public benchmark material and paraphrases of every exposed
training, teacher, routing, exact-memory, development, and model-selection
asset.

Reference admission requires at least three final candidates per source, two
accepted references, and two independent judge model families distinct from
the generator and all training teachers. Every model revision, prompt, request,
response, source, reference, and evidence file is hash-bound. Neither generator
nor judges may retain reasoning traces. Both judges must award maximum scores,
preserve protected tokens, and report no error tags or critical errors for both
references. Exact/document and normalized character-5-gram checks are combined
with a separately pinned semantic-neighbor scan over a complete exposure
manifest.

Promotion then compares the candidate with the frozen best prior local model;
Apple is diagnostic-only. In each direction the candidate must reach at least
50 chrF++, 0.80 pinned COMET-22, and 8.5/10 mean blinded automated score, while
the lower 95% paired confidence bounds for chrF++, COMET, and automated
pairwise score must all be positive. A deterministic audit and either judge's
critical flag form a union veto, so one critical number, entity, negation,
omission, placeholder, URL, markup, or code-switching error rejects promotion.
The same evaluator requires exact Swift/MLX parity, a non-Apple local failure
path, 250 ms warm p95, current archive integrity, 768 MiB peak RSS, preferred
150 MB size, and the 500 MB hard ceiling.

Both validators have passing positive and negative contract tests. The source
side is also frozen: 800 unique deterministic project-owned scenarios, 400 per
direction with exact domain quotas, SHA-256
`f039ce456c55f051e8bbcc13ed9bc8270a722819308e008b39da7f30327ec16c`.
A release-lineage audit passed exact and normalized character-5-gram screening
at a 0.65 threshold across 1,358,264 exposed texts in 15 authenticated
training, validation, memory, canary, and public-development inputs. A stronger
schema-v2 freeze now covers 17 text-bearing assets and 14 evidence assets. It
counts 1,411,076 raw strings, records that the current model has no training
teacher inputs or outputs, and explicitly says `upstreamExactRowsComplete:
false`. The two opaque ElanMT bases are bounded by their exact May 2024 Hugging
Face revisions, both older than the private July 2026 sources; this is temporal
exclusion, not a fabricated exact-row scan.

The pinned Apache-2.0 multilingual MiniLM revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42` then exhaustively embedded all
599,317 normalized-unique controlled strings and all 800 frozen sources on MPS,
without a candidate prefilter. Zero cases exceeded the preregistered 0.82
threshold. Maximum similarity was 0.798585, mean was 0.607781, 21 cases were at
least 0.75, and inspection of the 30 closest pairs found related generic intents
rather than copied or paraphrased sentences. The report SHA-256 is
`ab0de643dd3555aab0b9abab79259bf40e23926120869a9a6d84961900c859f8`.

This still does not make the candidate claim-ready. The source rows have no
references and remain `claimEligible: false`; independent generator/judge
evidence remains pending. The final-output-only 800-request `gpt-5.6-sol`
generator batch is sealed at SHA-256
`d6f85b9af0f10767067c66d1332a6334233044ee28674694612ea2614db6822a`
with `store: false`, strict Structured Outputs, `reasoning.effort: none`, and a
1,024-token output allowance. Reference judges are pinned to the distinct
`gpt-4o-2024-08-06` and `gpt-4.1-2025-04-14` families. The exact v2 generator
file was submitted on 2026-07-21 after its SHA-256 was revalidated; responses
and batch state remain in the git-ignored private work area. The two judge
batches remain pending. The post-Batch lane is executable rather than prose: it rejects
partial coverage, unexpected model revisions, changed
source/request bindings, invalid Structured Outputs, visible or encrypted
reasoning material, and protected-token drift. Judge candidate order is
separately shuffled. Exactly two references are frozen only when both families
assign 4/4/4, preserve protected facts, emit no error tags, and set no critical
flag. If all three pass, the lowest normalized character-3-gram Jaccard pair is
selected deterministically. Exact numeric, opaque-ID, URL, placeholder, markup,
and code-switch checks complement hash-bound judge consensus for entity,
negation, and omission judgments. The end-to-end offline fixture passes and
proves that a visible reasoning summary or a single judge veto fails closed.
The checked-in public suites cannot be relabeled
because they have already influenced development or pretraining, use the wrong
domain mix, and generally provide only one reference.

The preceding high-reasoning submission is retained only as quarantined audit
evidence. All 800 transport requests returned without Batch-level errors, but
10 response bodies were `incomplete` because the 768-token allowance was
exhausted, and 300 contained one or more non-empty encrypted reasoning items. Its raw
response and content-free privacy-audit SHA-256 values are
`6aeb703450883f12ce7626a9deebce322a7f627b30d7dd1498e18ffe16c7dec3`
and
`b268d19185cae02fb2b1afbd24e6e484b27522b3ed2cf6078ec4aeb7aa681907`.
The collector rejected the first case and admitted zero translations. Official
GPT-5.6 [model guidance](https://developers.openai.com/api/docs/guides/latest-model#update-api-and-model-parameters)
says `store: false` returns encrypted reasoning items by default when reasoning
is used, so the replacement disables reasoning; no
reasoning payload is stripped, ignored, or silently accepted.

## Promotion status

A one-physical-model bidirectional ablation is now reproducible. The two
directional teachers have identical 32,001-token SentencePiece assets and
architecture, but their 50/50 parameter mean emits immediate EOS in all twelve
canary cases (0.0 chrF++ both directions), so direct weight averaging is
rejected. A balanced corpus builder combines 4,346 licensed rows per direction,
retains row-level license/provenance, repeats only the smaller training side,
never repeats validation, and independently re-screens all source and target
texts against the protected canary. The dual-teacher trainer sends each row to
the correct frozen specialist for token-level KL and selects on unweighted
macro-direction chrF++.

Starting from EN→JA, 300 unprefixed steps moved balanced development from
34.93/0.45 to 35.95/13.67. Its exact 4-bit/group-64 single pack is 39,138,120
bytes and scores 33.46/17.63 on the non-claimable canary, with about 52 ms warm
p50. Literature-aligned `<2ja>`/`<2en>` routing did not help at the same budget
(35.29/12.84 development). Starting instead from JA→EN preserved 50.03 JA→EN
but learned only 4.77 EN→JA at its best 200-step checkpoint. These pilots prove
the size and runtime shape but not sufficient quality; none replaces the
73,403,714-byte preferred-v2 directional pair.

The development evaluation surface is also expanded from twelve canary cases
to a deterministic 800-case public stress suite: 400 cases per direction from
200 KFTT, 100 ALT, and 100 Tatoeba human pairs. It is intentionally
`claimEligible: false`: it has one reference per case, does not match Mimi's
product-domain quotas, and KFTT/Tatoeba may overlap ElanMT pretraining. The
preferred-v2 pair scores 33.24 EN→JA and 53.95 JA→EN chrF++ over the 800 cases;
conversational slices are 39.77/68.09, news 31.30/53.64, and professional
Wikipedia 33.26/52.04. This is robustness evidence only and cannot replace the
sealed 400-per-direction independently authored promotion suite.

Apple high-fidelity Translation was then run on the identical 800 sources. To
avoid turning quality evaluation into 3,200 slow calls, the app benchmark now
accepts an explicit non-negative warm-run count while preserving three as the
default; this stress pass used zero repeats and the normal promotion workflow
still requires three. Apple scored 24.42/50.14 chrF++. Preferred-v2's
paired mean sentence-chrF++ deltas were +8.22 (95% bootstrap interval
+6.02…+10.56) EN→JA and +5.54 (+3.49…+7.70) JA→EN. This apparent overall win
does not generalize: on independently sourced ALT news the deltas reverse to
-3.35 (-5.53…-1.23) and -4.87 (-7.03…-2.82). KFTT Wikipedia contributes large
positive deltas, while Tatoeba JA→EN includes zero in its interval. The result
is strong evidence that public/pretraining overlap and domain mix can create a
misleading aggregate win; Apple remains the normal path until the private
product-domain suite passes both directions.

For higher-powered development checks, `public-stress-v2` expands to 2,400
cases, 1,200 per direction and 400 per direction/domain. Preferred-v3 scores
30.17/55.95 overall, with 41.40/67.76 conversation, 32.64/56.57 news, and
29.69/51.42 Wikipedia chrF++. Cached p95 is 50.9/59.2 ms and Python peak RSS is
212.7 MB. This v2 sample reduces slice variance and the accidental dominance of
KFTT in the v1 aggregate, but remains explicitly non-claimable and does not
replace the sealed product-domain suite.

Apple was then run on all 2,400 identical sources, scoring 26.45/54.29 corpus
chrF++ with 2.133/2.314-second p95. Preferred-v3 is roughly 41.9x/39.1x lower
at p95 and its paired sentence-chrF++ deltas are +7.59 (+6.22…+8.94) EN→JA
and +3.32 (+2.20…+4.47) JA→EN. That aggregate still masks a significant news
loss: -1.25 (-2.25…-0.24) and -2.84 (-3.83…-1.87).

The independently pinned COMET-22 comparison rejects any overall quality-win
interpretation. Relative to Apple, preferred-v3 is -0.00537
(-0.01064…-0.00016) EN→JA and +0.00007 (-0.00481…+0.00508) JA→EN. News loses
by -0.02902 (-0.03537…-0.02296) and -0.02078
(-0.02516…-0.01664); JA→EN conversation also loses. Wikipedia is the only
slice with positive intervals in both directions. The score reports bind the
same suite and exact pinned metric signature, and the comparison script
persists 10,000-sample paired-bootstrap intervals. Thus the 2,400-case run is
useful development evidence for speed and failure targeting, not promotion.

Not promoted. The preferred-v3 MLX candidate trails Apple on the non-claimable
canary and fails the broader public-v2 COMET/news gates; the 400-case-per-direction
product-domain held-out suite does not yet exist. Apple is only a diagnostic
baseline for the intended final system, but remains Mimi's temporary normal
path while no local candidate passes the absolute accuracy gate. The MLX code
requires an explicit environment gate,
local model pair, and matching `mlx.metallib`; any load or output-validation
failure clears candidate results and reruns the segments through Apple
Translation. Swift compilation passes, while live Swift inference remains
verified in both directions with a version-matched prebuilt shader: the first
EN→JA process completed in 3.0 seconds including cold shader/model setup, and a
fresh JA→EN process completed in 0.62 seconds with the shader cache warm. This
does not change the failed quality gate or enable the lane by default.

## Reverse-consistency expert reranker: rejected

A deterministic calibration/test ablation tested whether the packaged
generalists could cheaply judge the routed formal/legal experts. For each case
selected by the source router, both forward candidates were translated back by
the opposite-direction generalist. The expert survived only when typed
structure checks passed and its reverse chrF++ exceeded the generalist by the
calibrated direction-specific margin. This adds no model bytes but requires four
sequential translations on expert-eligible inputs.

The calibration set selected margins of 0.5 for EN→JA and 0.0 for JA→EN. On
the untouched 1,389-case public-stress-v3 test intersection, the resulting
selector regresses mean sentence chrF++ by -0.439 (-0.728…-0.198) EN→JA and
-0.725 (-1.060…-0.450) JA→EN against the current routed pack. The worst slice
is JA→EN legal at -5.425 (-7.624…-3.603). Modeled sequential p95 is 212.4 ms
EN→JA and 119.7 ms JA→EN, with JA→EN legal reaching 245.8 ms.

The quality loss is accompanied by worse structural safety: 241 exact
critical-token mismatches versus the routed baseline's 227, and 27 unsafe typed
acceptances versus 13. Peak RSS is intentionally unreported because no
end-to-end multi-model residency measurement was taken; a single constituent
report is not a valid selector RSS measurement. The ablation is therefore
closed, not ported to Swift, and cannot change Mimi's default.

## Relative self-likelihood expert reranker: rejected

The next bounded ablation scores the exact cached-greedy output under the model
that produced it. Mean chosen-token NLL includes EOS; all 1,516 regenerated
generalist/expert sequences match their saved candidate token IDs. A distinct
deterministic split calibrates only on source-router expert cases and chooses a
-0.15 expert-advantage margin in both directions.

On the untouched 690 EN→JA and 722 JA→EN cases, the selector regresses mean
sentence chrF++ by -0.491 (-0.763…-0.246) and -0.939
(-1.356…-0.575), respectively. The JA→EN legal slice loses 6.645 points
(-9.284…-4.370). Modeled p95 increases from 58.4 to 107.2 ms EN→JA and from
66.0 to 91.6 ms JA→EN because both forward candidates must run on the
expert-eligible population.

The selector also raises exact critical-token mismatches from 255 to 280 and
unsafe typed acceptances from 11 to 28. Relative self-likelihood is therefore
not calibrated across the independently fine-tuned generalist and expert, even
though architecture and tokenizer are shared. End-to-end RSS was not measured;
the selector report correctly leaves it `null`. This lane is closed and no
product code changes.

## Weighted source-router classifier: rejected

A canary-constrained, magnitude-weighted `LinearSVC` explores whether routing
should classify expert wins rather than regress their size. It preserves the
six-case EN→JA canary and improves over the generalist by +0.309 mean sentence
chrF++ on the grouped 386-case public test (+0.022…+0.611). Direct comparison
is essential: the already packaged ridge router gains more on the same cases.
The classifier is -0.123 behind it (-0.253…-0.008), after adding 43 routes and
removing 8. It is rejected before serialization or Swift work.

## SMaLL-100 single-model screen: rejected before MLX port

The 500 MB allowance makes one additional pretrained baseline worth testing.
`alirezamsh/small100` is an MIT-licensed, 330M-parameter, single-physical-model
translator covering Japanese and English. It distills M2M100 12B into a
12-layer encoder and shallow three-layer decoder, so it directly tests the
user-requested one-model and teacher/student direction. Revision
`8ab680e26a596d2e3d2d2d17ae0f68df1037328c` is pinned; the authenticated FP32
snapshot occupies 1,337,145,439 bytes. No 4-bit port is built unless quality
first justifies the engineering work.

It does not. Greedy decoding reaches 24.569/51.583 chrF++ EN→JA/JA→EN on the
non-claimable canary, with 0.258/0.192-second p95 and a conservative
3,495,542,784-byte process peak. Beam five changes quality to 24.260/53.332 but
raises p95 to 1.742/0.567 seconds. Both trail the compact Marian development
line by a wide margin. The greedy and beam score SHA-256 values are
`f3d8c4b3e4f7f01ee0ee3eaa1f538bf1d02d57c5ea15f314f4f235ea191e1e12`
and `8f8a715aa93ef4b93406fcaa73ee46b27704ecdf4a4305db3a8b6b06d1a9eaaf`.

A bounded bilingual adaptation control uses the already screened, exactly
balanced 8,692-row KFTT/Tatoeba/project-owned corpus and stores no reasoning
traces. MPS training requires eager attention because its scaled-dot-product
kernel does not support training dropout; non-reentrant checkpointing then
produces deterministic rank-16 q/v LoRA checkpoints with 1,179,648 trainable
parameters. At learning rate 1e-4, steps 50 and 100 reach 24.653/50.599 and
24.603/50.036. At 1e-3 the loss diverges and step 50 collapses to 1.500/5.052
with pathological 1.7-second generations. The three score SHA-256 values are
`e2f27f361428a193d367679d4042d4ed3f61237c58104198c45232a16a2e1ff2`,
`98eaa1b70dbd70fcffd1abd63810cf1373c5f0f265f796c1c684ec53ea066428`,
and `e6d915d7ebea2445e7957b04b0f671c0e4e1d7c24d66216b56a200d832ca8089`.
The model is rejected without public-stress, learned-metric, quantization, or
Swift work. The result supports shallow decoders as an architectural idea, not
this multilingual checkpoint as Mimi's student initialization.

## Frozen 800-source exact-pack audit and symmetric safety cascade

The preregistered automated claim surface now has a first candidate run without
pretending that source-only rows prove translation quality. The exact
141,488,564-byte shared-tokenizer pack was loaded from its authenticated
`75ae72c...` manifest and run on all 800 frozen project-owned sources, exactly
400 per direction, with three warm repetitions. The report preserves empty
references, sets every result `claimEligible: false`, and records
`claimBlocker: references-pending`.

The one-way shipping policy passes the latency limit but decisively fails the
runtime safety gate. Warm p95 is 112.6 ms EN→JA and 58.0 ms JA→EN; 414/800
outputs pass the exact critical-token and plausibility guards, while 386 fail
closed on critical-token mismatch. The observed process peak is 288,063,488
bytes, but the independently repeated four-engine worst case of 421,003,264
bytes remains the conservative residency claim. The report SHA-256 is
`58743cc357e966bb8c24ae8101d77d01e5deda2f1bfbe50ffa830232f98eecd7`.
The failures are not all semantic errors: localized dates and times often
change digit formatting. They nevertheless include visible meaning failures,
including dropped dates, repeated output, wrong product terminology, and lost
negation, so the guard cannot be weakened from this evidence.

A bounded algorithmic ablation tries the other already-bundled role whenever
the router-selected role fails the same guard. On public-stress-v3 it recovers
23 additional failures beyond the established expert-to-generalist fallback.
Mean sentence chrF++ changes by +0.019 EN→JA (-0.016…+0.064) and +0.057 JA→EN
(-0.024…+0.145), so neither direction has a positive lower bound. Modeled p95
remains below 85 ms on that public surface. The candidate and score SHA-256
values are `ebfed0d25687d69d2c5eaf171d6f8c7b09c9a732afa2b3c83edaf7c57dda1670`
and `1e7a4e1d93c41ac452ed0e94c03b3086a432d227db7c031da4f7210661425f19`.

The frozen product surface rejects the added runtime work. The second model is
attempted on 429/800 rows and changes 215 hypotheses, but it admits only five
additional structurally valid outputs. Four EN→JA alternatives visibly repair
a dropped date or repeated output; the only JA→EN rescue changes `13:05` to
`13:5`, illustrating that structural acceptance is not a quality judgment.
JA→EN p95 rises from 58.0 to 108.7 ms. The hash-bound comparison is
`e7610465462fa663c736e751803648d044bc16bea83daf51647f0100b4994d79`.
The symmetric cascade is therefore rejected without a Swift port. The evidence
redirects effort to stronger final-translation targets and preservation-aware
training rather than another inference-time selector.

## Preservation-aware curricula: EN public win, exact pack rejected

A licensed no-synthetic control turns already exposed human parallel rows into
text-identical preservation curricula; it adds labels and sampling policy, not
new source or target text. The EN→JA curriculum has 4,346 train and 1,085
validation rows. The selected step-150 checkpoint moves validation chrF++ from
30.579 to 30.782 and the critical-preservation slice from 27.694 to 29.070,
while replay is essentially flat at 31.099→31.090. Its training-manifest
SHA-256 is `8d94bfd0605cad9ef273362c57d5c645e3ecb6ae8f8dbf43a1d641b583f6675b`;
the exact 4-bit directory is 39,141,436 bytes.

That EN→JA checkpoint generalizes on the 1,400-case public-stress-v3 direction:
mean sentence chrF++ improves by +0.441 with a positive paired 95% interval
(+0.153…+0.738). Wikipedia improves +1.058 (+0.459…+1.729) and legal +0.634
(+0.065…+1.296); news is flat at -0.077 (-0.324…+0.142). Strict source/output
token mismatches fall only from 254 to 251 and the candidate still introduces
new individual failures, so the existing fail-closed guard remains mandatory.
The score SHA-256 is
`923a7e1270cdc341046218aea92c5436e733722156e8018c686a79efc90de7ec`.

JA→EN demonstrates why exact parent lineage matters. A first run from the old
conversational control is rejected at -1.468 public sentence chrF++
(-1.977…-0.998). The corrected run starts from the actual shipping averaged
checkpoint, authenticates both its `5b7894...` weight hash and `bee6dd...`
averaging manifest, and selects step 50. Validation changes 52.180→52.235 and
the critical slice 50.105→50.214. After exact 4-bit conversion, however,
public-stress changes by -0.109 (-0.300…+0.061), so it also fails the required
positive lower bound and is rejected. The training-manifest SHA-256 is
`d8699627da3c81044e6bf63cb27c5600ac3dcc6da82f7acaecc0acc8c303028e`;
the paired-score SHA-256 is
`a0325b0591e3046c83ee09ebdb2f93935436b65ac437b7c3be37ba19c194d6c2`.

The bounded packaging test therefore combines only the promising EN→JA child
with the incumbent JA→EN generalist and the unchanged experts, routers, and
translation memory. Shared-tokenizer size is 141,491,232 bytes, below the exact
150,000,000-byte preference; its manifest SHA-256 is
`8df1e11aab7996219877d292ea987facd738cba0dcaf5b8a49a2d26257647ebe`.
The exact 800-source app-shaped audit rejects it: fail-closed acceptance drops
from 414 to 395, with 405 critical-token failures. EN→JA/JA→EN p95 remains
real-time at 111.1/56.8 ms, but latency cannot rescue a safety regression.

A matched different-model comparison records 394 cases accepted by both packs,
20 accepted only by the incumbent, one accepted only by the candidate, and 385
failed by both. Retaining the old EN→JA generalist as a fifth fallback would
therefore raise the union to only 415/800—one case over the incumbent—for about
34.3 MB more weights and without reference-based quality evidence. That path is
also rejected. The exact comparison SHA-256 is
`501bd440bc4dca989948fa696f61b0a21669364369c9bfe5cf5b7028de91a7fa`.
No Swift port, release-contract replacement, or app-default change is made.

## July compact-model refresh: two more candidates closed

`LiquidAI/LFM2-350M-ENJP-MT` is technically credible: a single bidirectional
350M causal model, direct support in Mimi's pinned MLX Swift dependency, and a
published 381.6 MB MLX 8-bit bundle. It does not pass the license gate. LFM
Open License 1.0 excludes commercial use by a legal entity at or above USD 10M
annual revenue. Mimi therefore cannot treat it as unconditionally distributable
without a separate commercial agreement, and no local quality run is promoted.

`WhirlwindAI/Translate-15L` is Apache-2.0 and much smaller: the authenticated
60.5M-parameter FP32 snapshot is 244,616,918 bytes. Its repository metadata from
Transformers 5 requires a recorded `extra_special_tokens={}` compatibility
override under pinned Transformers 4.57.6; T5's equivalent `extra_ids` remain
unchanged. The model then fails both fair decoding controls on the 12-case
development canary. Greedy scores 0.00/1.37 chrF++ EN→JA/JA→EN. The model-card
beam-4 setting still scores 0.00/4.11, emits empty strings or long runs of
hyphens, and reaches 0.203/2.596-second p95. The beam report and score SHA-256
values are `871f50cd228d72e53dd77d11d2d7e942bf8d9b027ff5df1f91581bd35276ab42`
and `813801ccd9fd7f93e7db888e509621dc8085af5cc78c67c74d7503e76ca0d0db`.
It is rejected before MLX conversion, fine-tuning, or app work.

## CAT corpus audit and Hy-MT2 hard-ceiling control

CyberAgent's 7.11 GB `CAT-Translate-Dataset` is not admitted into Mimi's
training set. The public card identifies ODC-BY-1.0 plus Common Crawl terms,
corpus-level sources, and Apache-2.0 `gpt-oss` generators, but access requires
accepting a contact-sharing gate. All unauthenticated schema, preview, and
parquet endpoints return 401. More importantly, the public metadata does not
bind each row to its source URL/license, generator revision, and filtering
decision. The reusable result is the curriculum—not the rows: diversity-first
SFT, a smaller quality stage, MinHash/language/length filtering, and independent
format and length anti-hallucination objectives.

Tencent's `Hy-MT2-1.8B` is the strongest new hard-ceiling hypothesis. Its
Apache-2.0 Sherry checkpoint stores 1.8B parameters in a sparse-ternary grid.
The official 461,860,800-byte GGUF at revision
`9df5c824a00a744fb0512a29c640466f4d97dfb0` hashes to
`cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93`,
but fails to load in the referenced open llama.cpp STQ branch because it still
uses the older tensor layout/type ID.

The authenticated community MLX conversion at revision
`03d1df683157fde0a4ec80636e749867d0c13a5e` works through a custom Metal
kernel and totals 464,192,044 bytes. On the canary it appears excellent:
35.62/60.70 chrF++ EN→JA/JA→EN, about 810 MB peak process RSS, and warm p95
0.735/0.601 seconds. The required 800-case stress run reverses that conclusion.
Hy-MT2 scores only 22.64/44.09, versus preferred-v2's 33.23/54.00. Paired mean
sentence chrF++ deltas are -10.10 (-12.40…-7.91) and -11.85
(-13.94…-9.79). Stress p95 is 1.026/0.850 seconds, and 127/800 outputs fail the
exact critical-token audit.

The raw report, paired score, and structure-audit SHA-256 values are
`e3e7a8c63a70ba391a1c5d29b79ffe7682ca3a0bcd7b6c650bcc793caeff9ef5`,
`8ac289f115400194fec04643c623c9822446f4f72471d1631f07140dadcca332`,
and `712e70d3644ffbb74978dc8e74510b2c3fbad27127c075c1437c8b0b17653c6e`.
The model is rejected before COMET, Swift/Metal porting, fine-tuning, or app
integration. Independently, its report describes roughly one trillion training
tokens without enumerating source licenses, so Apache-2.0 weights would still
not satisfy Mimi's release-lineage gate even if quality had held.

## LMT-60-0.6B one-model control

NiuTrans's Apache-2.0 `LMT-60-0.6B` is a Qwen3-based 60-language causal
translator and therefore a cleaner one-model MLX/Swift hypothesis. The pinned
BF16 checkpoint at revision `dd189845cdc73346cef33c7a94f4b8bd8efdd4eb`
contains a 1,503,300,328-byte weight file with SHA-256
`c48c3b8d7b04d3c6e56452fa51ddefc03921043b0f70641cb80c5d3ac71b73e6`.
The exact local MLX 0.30.6 conversion uses affine 4-bit/group-64 weights. Its
authenticated snapshot is 346,929,488 bytes; `model.safetensors` is
335,450,548 bytes and hashes to
`8a2437b9f22eb7217be5f51bf5e10a7c36d5cc610f7f7f078740dd3e670af65f`.

The model-card prompt and real-time greedy decoding reach 31.38/54.15 canary
chrF++ with 0.297/0.301-second p95, but the matched 400-case-per-direction
stress run collapses to 17.92/40.42 versus preferred-v2's 33.23/54.00. Paired
mean sentence chrF++ deltas are -16.08 (-18.52…-13.78) EN→JA and -15.61
(-17.93…-13.37) JA→EN. Stress p95 is 0.491/0.397 seconds, peak process RSS is
745,619,456 bytes, and 141/800 outputs fail the exact critical-token audit.
Failures include mistranslated names, invented relations and locations, and
runaway repetition. The official card demonstrates beam 5; Mimi rejects this
candidate under the required real-time greedy control without claiming that
the result reproduces the paper's beam-decoded FLORES numbers.

The raw stress report, paired score, and structure-audit SHA-256 values are
`71b2a1f72f92290b93154824807784f02f301792125948d70cc9dbdc0fb895da`,
`58086acf192538f603fec561f6f5df1da13c43d3ab4bdb49262552f17fee2058`,
and `da0db56b6d1465886b922a200dd9f47fadc7f0ac979f91197f612ec298795861`.
No COMET run, Swift parity port, fine-tuning, or integration follows.

Release provenance independently fails. The paper describes 90B continued-
pretraining tokens from mixed monolingual and parallel sources, then roughly
567k curated SFT pairs. It names corpus families including CulturaX, MADLAD,
FineWeb2, OPUS, C4, OSCAR, WanJuanSiLu, and in-house data, plus billions of
synthetic parallel pairs, but does not bind every training row to a source
license or generator revision. The published `LMT-60-sft-data` repository also
has no license tag. Its exact 4,371,634-byte `en-ja.jsonl` at revision
`47914a5aac70e3e930aa8e7e8dae2969219319c3` hashes to
`5dd641719d2ec2f8727452e744a65226ac238e3384df2add79ff04de506c89fa`
and contains 13,169 pairs entirely from named public benchmark families: 997
FLORES-200, 1,997 NTREX-128, 2,584 IWSLT 2017, 594 IWSLT 2022, 3,991 WMT20
news, 1,018 WMT21 news, and 1,988 WMT22 news. The audit does not establish that
the exact evaluated splits were trained on, but it makes public-benchmark
interpretation contamination-sensitive and rules the shard out as independent
Mimi evidence. Apache-2.0 weights alone therefore cannot satisfy Mimi's
distributable-lineage gate.

## QuickMT architecture control

QuickMT supplies the strongest newly found small-Marian architecture signal,
but not a distributable Mimi candidate. The pinned CC-BY-4.0 releases use an
8-layer encoder/2-layer decoder with 1,024 hidden dimensions for EN→JA and a
12-layer encoder/2-layer decoder with 768 hidden dimensions for JA→EN. Their
official CTranslate2 int8 exports are 401,699,775 and 407,101,843 bytes;
together with tokenizers/configuration the local pair is 813,661,710 bytes.

The exact model-card tokenization and CPU int8 beam-1 canary reaches 26.54
chrF++ EN→JA and 57.98 JA→EN, with 0.076/0.077-second warm p95 and
1,776,779,264-byte peak process RSS. Three of twelve rows fail the exact
critical-token audit. EN→JA is already well below Mimi's compact incumbent, so
the technical weights are rejected before broader stress, custom MLX/Swift
porting, fine-tuning, or integration. The raw report, score, and structural
audit SHA-256 values are
`a584845607c3b73258640cfa33b2416fafaaeecfc29b0db6d16a5aa8f79c9d51`,
`6cf3a589c2231924b833348696c5dd5bb377d8e0c5508dd0fd1acc4c56e0ef38`,
and `aa701e0a680038e539919b32a951f6cfec0ee283f11f11455248a0d107124519`.

The training card independently blocks reuse. Its 63,285,158-row mixture has
no license declaration and combines benchmark train/dev/test families,
restricted or unclear corpora, and large MADLAD/NewsCrawl backtranslations
without row-level rights or generator identities. QuickMT therefore remains an
architecture blueprint—train the shallow decoder from initialization—not a
teacher, training dataset, or shipping model.

## Progressive five-decoder recovery closes post-hoc depth pruning

A narrower follow-up retained decoder layers 0, 1, 2, 3, and 5 from the exact
six-layer EN→JA incumbent, then ran the licensed 123,050-row recovery curriculum
with frozen-parent KL. This removes only one decoder layer, so it is the least
aggressive post-hoc latency test after the rejected four- and two-layer arms.

The untouched pruned initializer scores 23.592 chrF++ on the exact 1,285-case
licensed validation set. Recovery reaches 26.766 at step 125 and 27.066 at step
250, still 3.543 points below the intact parent's 30.609 matched baseline. The
predeclared stop therefore fired before quantization or canary work. The final
training manifest and selected weight SHA-256 values are
`e4ddf9e5feb38a17e5e889de91226ae2ee2fea232f18182246f50cffc463d34f`
and `10f7038f2f530d8ad05cfe9c3ffa35ae4ba5796b80e0fd151cdefdde3fdf3836`.
Together with the earlier 6→4 and 6→2 failures, this closes decoder-layer
deletion for the converged ElanMT family. Any future shallow decoder must be
initialized and distilled as shallow from the beginning.

## Token-local negative-space adaptation: bounded but not a winner

The NSL-MT literature prompted a reviewer-free training control that adds no
inference parameters. Authenticated licensed references remain the only
positive targets. A deterministic builder selected 8,000 unique positives per
direction and created 31,621 EN→JA / 31,590 JA→EN rejected strings spanning
number, unit, URL, placeholder, negation, omission, and duplication errors.
The negative strings are used only as negative evidence; there are no free-form
synthetic translations, human reviews, or reasoning traces. The train hashes
are `b5775c05fd3c303e80c1fee8f573d0d85b117c3f80c634778ffe47d499c45b44`
and `455646da68d6a2745ae26a16415c8c883e2d6ff30165de659edbe15c70973d24`.

Rather than reproduce the paper's unbounded whole-sequence `log P(v|x)` term,
Mimi applies severity-weighted `-log(1-p(v_t))` only at the first divergent bad
token under the correct target prefix, alongside ordinary positive
cross-entropy. The paper-range alpha 0.3 EN→JA arm barely moves bad-token
probability. Alpha 3.0 is the stronger bounded calibration: EN→JA licensed-dev
chrF++ changes 30.604→30.652 while mean bad-token probability changes
0.023303→0.022463; JA→EN changes 52.180→52.211 and
0.006860→0.006721.

The shipping-shaped q4 gate rejects promotion. EN→JA falls to 29.996 canary
chrF++ and retains the critical number failure; the report/score/audit hashes
are `73ffba8c2af8f39612c90f27c31757e2a1de7025c2528d1ff222549b8dbc057a`,
`56aadab1a7ddc6a4b06780b80eb6d1cc04adf36be2f2204b3b0716d5bd30b02e`,
and `283c109fb451425a71c444e7d8aac8dc1375bfd0904edad5204ddc1d4b208937`.
JA→EN produces exactly the incumbent's 6/6 hypotheses and token sequences at
56.520 chrF++; its corresponding hashes are
`e1449bc1dc86101f3d1678d0cca3c5708acec39307a328243f5d10c381dc8e8b`,
`ae0a2e02ca1047c83a5f2eec4d86d27a0d0771a09dad5a1703acfc04548e2efc`,
and `567bd48b619eee581ac87af21dd3541acbbe5873f63d71c104c0a8d8a87cdb17`.
The arm stops before stress, learned metrics, Swift parity, or integration.

## Source-only critical taxonomy and typed n-best rescue: diagnostic, rejected

The exact 800-source pack audit's 386 strict critical-token failures are now
classified without references or a quality claim. A hardened atomic parser
finds 290 exact Gregorian-date/24-hour-time surface normalizations: 39 EN→JA
and 251 JA→EN. Another 33 are broader word-number or percent normalizations
covered only by already rejected policy families. The remaining 63 show
concrete unsafe structure: 25 introductions/duplications, 19 drops, 9 mixed
substitutions, 8 date value/drop/multiplicity changes, and 2 invented-time
changes. Admitting the 290 would raise structural acceptance to 704/800, but
one such output still turns “keep Hana in the loop” into a confinement meaning;
structural equivalence cannot authorize semantic quality.

The parser is deliberately narrow: NFKC; at most one valid Gregorian date and
one 24-hour time; exact value, order, multiplicity, residual digits, percent,
URL, placeholder, printf, and markup preservation; and fail-closed rejection
of AM/PM, time zones, ranges, eras, abbreviations, and ambiguity. Public-v3
does not validate it for promotion. Only 12/25 candidate cases match the
narrow reference signature; 13 references use additional Japanese-era forms.
The existing percentage arm has only 1/1 public evidence and is now labeled
insufficient rather than statistically passed.

A literature-derived failure-only beam-4 n-best control runs on 43 eligible
strict failures. It finds nine typed candidates but needs 0.461-second EN→JA
and 0.339-second JA→EN triggered p95. Several preserve the date while altering
negation or conditional meaning. The report SHA-256 is
`80623a88a6100bc8c7172d58620be23fecc44733c13c1786c6455c1f36ad0be6`.
The arm is rejected before Swift work, hard constrained decoding, integration,
or any default change. The reusable output is the failure taxonomy and
adversarial validator; promotion waits for the sealed 400+400 references and
independent automated judges.

`Yokii2/quickmt-ja-en-v2` is also excluded. Its roughly 211.3 MB JA→EN
CTranslate2 engine fits the relaxed size ceiling, but its unreproducible 31M-row
Patchouli subset derives Japanese text from unknown-license CC-100 and English
targets from an unspecified “Mistral Small 4” teacher. The architecture is not
a current Mimi MLX/Swift drop-in, and the repository's CC-BY-4.0 label does not
establish distributable training lineage.

## Target-vocabulary projection shortlist: exact but rejected at canary

The final no-credential decoding-speed control leaves the incumbent network and
greedy decisions unchanged, but projects each decoder state onto a smaller
target-language vocabulary. The static shortlist is derived only from the
authenticated shared tokenizer—no corpus row, held-out source, reference, or
benchmark output is used. EN→JA keeps Japanese-script, common numeric/symbol,
SentencePiece-space, and short Latin tokens; JA→EN keeps Unicode-Latin,
numeric/symbol, SentencePiece-space, and short Latin tokens. Each sentence adds
its source token IDs and all leading-`▁` surface equivalents. The authenticated
artifact SHA-256 is
`ff2452673637c601f7de182b9a3ab0a0aea927eb348042bf10c014e03934c790`.

The exact q4 projection-subset unit test matches full-vocabulary logits, and the
12-case runtime canary preserves every output token ID, hypothesis, route,
guard decision, and failure result. Median candidate vocabulary sizes are
20,613/32,001 EN→JA and 15,196/32,001 JA→EN. That mathematical parity does not
produce a material wall-clock win on this Apple-Silicon runtime:

| Direction | Metric | Full projection | Shortlist | Candidate delta |
|---|---:|---:|---:|---:|
| EN→JA | warm p50 | 0.032508 s | 0.032566 s | +0.18% |
| EN→JA | warm p95 | 0.034904 s | 0.034607 s | -0.85% |
| JA→EN | warm p50 | 0.030300 s | 0.030159 s | -0.47% |
| JA→EN | warm p95 | 0.038647 s | 0.038228 s | -1.08% |

Peak RSS increases from 285,802,496 to 309,755,904 bytes (+23,953,408;
+8.38%) because the process retains static subset projections. Preparation time
increases from 0.0683 to 0.1557 seconds (+127.89%). The predeclared gate requires
at least 5% improvement at both p50 and p95 in both directions, exact output
parity, and no peak-RSS increase. The comparison report therefore stops the arm
at canary; its SHA-256 is
`c13035d0ed4a90673aa9b559e498f3414d13a61c93f994e1e258f8d31cf06134`.
There is no 800-source run, Swift port, app integration, or default change.

## MLX 0.31.2 runtime upgrade: token drift and latency regression, rejected

MLX 0.31.2 adds a small-M split-K quantized matrix-multiplication path, making a
runtime-only upgrade the lowest-risk follow-up to the failed output shortlist.
Mimi reran the exact 12-case, 30-warm-repeat canary between fresh MLX 0.30.6
controls. Model files, tokenizer, Python runtime, Transformers, Tokenizers,
generation policy, and benchmark suite were unchanged.

The upgrade is not numerically exact. Two EN→JA cases change output token IDs
and hypotheses: `canary-en-003` changes the date/platform connective, while
`canary-en-004` adds the missing space-key instruction and changes the sentence
structure. The latter may read better, but a canary cannot establish quality and
runtime upgrades must preserve tokens before a sealed evaluation permits drift.

Against the immediately following 0.30.6 control, 0.31.2 regresses EN→JA warm
p50/p95 by 4.30%/24.60% and JA→EN by 5.92%/7.11%. Peak RSS rises 2,965,504 bytes
(1.04%). The candidate report SHA-256 is
`148fa4925b39f02a22ccfcd7112b4fad4b0579559249fd53beb753c289eeab5e`;
the rejected comparison SHA-256 is
`94a0b03c8f305caa275cf6ce141679331d4e17dc0ed5877fe7515fe1c7682395`.
Stop at canary. Mimi remains pinned to MLX 0.30.6 for this model/runtime.

## Packed q4 attention projections: exact and leaner, but slower

An opt-in MLX path concatenates the quantized output rows, scales, affine
biases, and linear biases for encoder/decoder self-QKV and decoder cross-KV.
It never dequantizes or changes row order. The q4 projection contract is
bit-exact, and the isolated alternating-order benchmark clears its continuation
gate: self-QKV improves 11.15% at M=1 and 11.59% at the median of M=1/8/16/32/64.
The microbenchmark SHA-256 is
`11dadcf86c4b882e5ec8ed8339fd1e045aff4c3d141eed03963b0b1d169f6f76`.

End-to-end behavior is exact across all 12 canary outputs, tokens, routes, and
guards, but latency regresses. EN→JA p50/p95 increase 1.37%/6.86%; JA→EN
increases 1.13%/0.74%. Packing does reduce peak RSS by 23,101,440 bytes (-8.04%)
and preparation time by 17.53%, because the concatenated arrays replace rather
than duplicate the individual projections. That is useful implementation
evidence, but it does not meet the real-time goal's ≥5% p50/p95 speed gate.
The canary comparison SHA-256 is
`6488e6fa1b45bd5049fd1601b6d3c1600ac36a7d8cdc522f72219e542ac384ac`.
Stop before 800-source, Swift, packaging, or default work.

## Stable-block MLX compilation: exact, below the isolated floor

Before compiling cache-length-dependent decoder steps, Mimi benchmarks only two
fixed `[1,1,512]` decoder subgraphs: residual-add plus LayerNorm, and the q4 FFN
plus residual-add plus LayerNorm. Both compiled outputs are bit-exact. Across
seven alternating-order blocks, median improvement is only 0.19% for the
residual block and 4.77% for the FFN block, below the predeclared 10% isolated
continuation threshold. The report SHA-256 is
`81f0c0d57a70b20b26030e551339cde28574d268b796ce6daa94389da4f35d5f`.
Stop before whole-layer/whole-decoder compilation, shape-cache experiments,
custom Metal fusion, full canary, or Swift work.

## Same-depth SSRU decoder: insufficient Apple-Silicon speed headroom

The remaining trained-runtime hypothesis replaces each decoder self-attention
sub-layer with the Simple Recurrent Unit recurrence reported by Kim et al.:
`f_t = sigmoid(W_t x_t + b_f)`,
`c_t = f_t c_{t-1} + (1-f_t) W x_t`, and `o_t = ReLU(c_t)`. Before creating a
new student, Mimi measures an authenticated q4 compute proxy that retains all
six decoder layers, cross-attention, FFNs, residuals, and layer norms. It packs
the two SSRU projections into one call and alternates measurement order against
the incumbent cached Transformer layer at source/prefix length 16.

Across seven 1,000-iteration blocks on Apple Silicon, the median Transformer
layer time is 0.00050599 seconds and the SSRU proxy is 0.00047211 seconds: only
a 5.2267% layer-level improvement. That misses the predeclared 10% continuation
floor before accounting for unchanged encoder, cross-attention, FFN, output
projection, tokenization, and synchronization costs. The proxy evaluates speed
only—not SSRU translation quality—and therefore cannot authorize a model.

The arm stops before student initialization, sequence distillation, q4 quality
evaluation, Swift implementation, or app integration. Its report SHA-256 is
`ce7f1d9257696150ad4706d4ab6d4ce5385ebcf62859ae4044cbc70675aa93f7`.
The broader 12-encoder/1-decoder design remains literature-supported but is a
different pretraining architecture, not a justified mutation of the current
small incumbent. Mimi should spend the next teacher budget on the sealed
400+400 product-domain references and quality-preserving distillation rather
than train this same-depth SSRU arm.

## Release-contract completeness repair: more lineage, still blocked

The release builder no longer hardcodes `provenanceComplete=true`. It accepts
the two selected expert checkpoints explicitly, authenticates their weight and
training-manifest hashes, and recursively follows direct training, interpolation,
and checkpoint-averaging ancestry. The audit-v2 contract now binds all four
selected engines, eight training manifests, two transformation manifests,
twelve dataset files, and 9,305 Tatoeba attribution records.

The stronger trace exposes 256 provisional local-teacher consensus ancestor
rows marked training-only and promotion-ineligible. It also confirms that all
four engine manifests lack a hash-bound full-precision-to-MLX conversion record.
Consequently, the regenerated contract reports `provenanceComplete=false`,
`modelPromotionEligible=false`, and an explicit blocker union. Its SHA-256 is
`ec7769e1b45252566f8515c8bfd94095f9ee3b5c3946d165c010c33f6a4aa2c9`.

The development-only stager now preserves rather than overwrites incomplete
provenance. The v3 staged manifest SHA-256 is
`4add7e677378965d7dc5e4398b5bb01c1acbbc2d7125485c97acf243f9b77601`;
normal staging and archive verification reject it, while the explicit local
development path remains labeled `blocked-development-only`. This repair does
not make the model distributable and does not change Mimi's default.

## Conversion provenance rebuilt with byte-identical weights

Each selected full-precision checkpoint was re-converted under MLX 0.30.6 after
the converter gained a hash-bound transformation record. All four q4
`model.safetensors` SHA-256 values exactly match the incumbent, and all twelve
non-manifest files in the final shared-tokenizer pack are byte-identical. The
new manifests bind source and output weights, 4-bit/group-64/float16 settings,
the converter SHA-256, and Python/MLX/tokenizer runtime versions.

The provenance-complete v3 pack is 141,492,266 bytes, only 3,702 bytes larger
than v2 metadata-wise, with manifest SHA-256
`deda4fe0d6c9ca3fd069ca99f7c45a42b5bab1fcfda3ff861cb3a0bdee40c2ee`.
Its audit-v3 contract computes `provenanceComplete=true` and hashes to
`8c4baec93d53f499914201f3a42179eb7a6071490e86e5cda2fd952946db4d45`.
The v4 development-stage manifest hashes to
`e78ff01d2a85a2c76e41580afd9ff7b27ad7584d1aff765afe54381676d359a5`.

The rebuilt pack also preserves all 12 canary hypotheses, output token IDs,
routes, guard decisions, and failure results after excluding timing fields. Its
three-warm-run canary report SHA-256 is
`abf72b7608e0bd06b8bfdc29fa1a7efa39e881764f0ac600fd03d6fbb8aebfd5`.

This closes transformation provenance only. The pack still contains 256
promotion-ineligible ancestor rows and the promotion-ineligible exact memory;
the 400+400 reference run, license compatibility, portable inventory, and app
distribution reviews also remain open. It is not integrated or promoted.

## Human-only lineage cleanup: quality-neutral, one safety regression

The selected EN→JA formal expert inherited 256 provisional local-teacher rows
only through its initializer. A matched 1,000-step retrain starts and preserves
against the human-only conversational checkpoint instead, while keeping the
same 123,050 licensed human/project-owned training rows, validation set, seed,
optimizer, KL/L2 preservation terms, and evaluation cadence. It uses no
synthetic targets or reasoning traces. Step 750 is selected at 31.047909
reviewed-dev chrF++; step 1,000 reaches 31.044249. The old expert's best is
31.059595 at step 1,000, while its matched step-750 score is 31.045468. The
clean checkpoint is therefore effectively tied, not an accuracy improvement.
Its full-precision weight SHA-256 is
`eb9aa8db0e99d371036b0c55635cfdb3a5ee4d5715e32ab58bebe6173aa17ee5`.

The q4 routed pack replaces only that expert, removes the blocked exact memory,
and retains the two generalists, JA→EN legal expert, and source routers. It is
140,875,806 bytes with manifest SHA-256
`8fd2dd3ecf39ab86ff535e8a2f77576390898fb92a7600a923efd90dbed8704e`,
9,124,194 bytes below the preferred ceiling. Peak RSS on the 2,800-case run is
306,741,248 bytes; warm p95 is 75.13 ms EN→JA and 68.25 ms JA→EN.

Against an otherwise identical memory-free pack containing the old formal
expert, mean sentence chrF++ changes +0.0083 EN→JA (95% paired bootstrap
-0.0270…+0.0490) and exactly 0.0 JA→EN. EN→JA legal improves +0.0831
(+0.0101…+0.1790), but one long legal source loses an appended-table number;
both the clean expert and generalist fallback fail the strict token guard.
Runtime acceptance consequently drops from 2,296/2,800 to 2,295/2,800, with
493 critical-token mismatches and 12 implausible outputs. The paired score
artifact SHA-256 is
`7d34e98b8595522051f89fbeb2b6c88221e2de3699d863fe563cd4222fa4f47b`;
the runtime comparison hashes to
`12f29802a2bb3a172a8b9f89ff5882455a69dd38066f6429a35492047c5a0245`.

Against the memory-bearing development incumbent, the memory-free pack is
-0.0267 EN→JA (inconclusive) and -0.1673 JA→EN (-0.3185…-0.0441), with the
JA→EN legal slice at -1.1713. Seven exact-memory hits explain this different
comparison; three of them account for the incumbent's extra runtime
acceptances. The memory remains policy-blocked, so this does not justify
putting it into an app. It does show that deleting a useful lookup layer has a
measurable cost that must be recovered with distributable neural training or a
separately authorized memory.

The release-contract builder now also authenticates adjacent dataset policy
manifests. Clean row flags cannot override a dataset manifest that omits or
denies `promotion_eligible`; missing policy coverage also blocks promotion.
The v4 contract has complete model/conversion provenance and zero excluded
rows, but four dataset manifests still do not authorize promotion. Its SHA-256
is `b56eb15418d8661626fdfc428611783671eabf2c24ba88456e2d23165fa0af0b`.

Decision: retain v4 as the best lineage-clean research pack, but do not stage,
integrate, or change Mimi's default. It fails the zero-critical-regression gate,
the dataset-policy and license reviews remain open, and the sealed 400+400
automated reference evaluation is still pending.

## Law-group validation rejects every clean checkpoint

A new validation-only safety suite samples 400 complete human-translated legal
units from 42 Japanese Law Translation validation law groups and emits both
directions, for 800 cases total. The law groups are disjoint from the
authenticated JLT training and public-test groups. Selection uses no model
output. It reserves 160 critical-structure, 80 negation, 80 enumeration, 40
long-form, and 40 general source units, rejects exact and near overlap with the
protected suites, and is explicitly ineligible for claims, promotion, or app
integration. The suite SHA-256 is
`352b04c12a17480ffd3e41ea89afef6caf00f0b0aae640050398898a3e81bc91`;
its manifest hashes to
`aa8d813cc59a6cceac9db97564f49f957434cc40f612fdc572a5215a7c79f46a`.

Before looking at the remaining clean checkpoints, Mimi registered steps
250/500/750/1,000 and required all of: paired legal sentence-chrF++ 95% lower
bound at least -0.1 against the old neural expert, no increase from 208 exact
critical-token mismatches, no increase from 101 negation mismatches, and
general-development chrF++ at least 30.9096. The baseline report was then
refreshed solely to add exact declared-model hashes; all 400 hypotheses and
structure counts remained identical. No selection rule changed. The final
contract SHA-256 is
`6324b5bd6772b33e81b24553c40f2eae80c589412f1ebc0ccf474bfd131f9f00`.

All four clean checkpoints fail:

- Step 250: legal corpus chrF++ 16.6840; paired mean -1.8704 with 95% CI
  -2.5860…-1.1896; 174 critical and 102 negation mismatches; general retention
  also fails.
- Step 500: 17.5746; mean -0.4365, CI -0.8106…-0.0543; 193 critical and 103
  negation mismatches.
- Step 750: 17.9295; mean -0.2866, CI -0.5624…-0.0206; 204 critical and 102
  negation mismatches.
- Step 1,000: 18.1166; mean -0.0685, CI -0.2542…+0.1110; 208 critical and 101
  negation mismatches. It passes structure and general-development retention,
  but not the preregistered legal non-inferiority bound.

The baseline legal corpus score is 18.3041. The selector therefore emits
`clean-checkpoint-family-rejected`, with no selected step and no promotion or
integration authorization. Its SHA-256 is
`aeea1867d0371499643db7e5b7ee7a0a65e6b55131dd533e6bda2ff5142b84af`.
The monotonic recovery makes a smaller learning-rate continuation from the
clean step-1,000 checkpoint a justified next experiment, but this validation
suite is now development evidence; it cannot replace the still-sealed 400+400
product-domain evaluation.

## Clean continuation wins legal quality, but safety and confirmation reject it

A preregistered 1,000-step continuation starts from and preserves against the
clean step-1,000 checkpoint, uses a new deterministic shuffle, halves the
learning rate to 1e-6, and retains the same licensed human/project-owned
123,050-row dataset and KL/L2 preservation terms. Its contract binds the exact
initial weights, training manifest, dataset, hyperparameters, selector, MLX
benchmark, runtime, and structure-audit implementations and hashes to
`048192f788d66e9a3c379a375e14d22597779d0507e83e057d0b5ba904eee380`.

All continuation checkpoints retain the preregistered general-development
floor. More importantly, every one significantly beats the old neural expert
on the 400-case EN→JA validation law set. Step 250 reaches 19.1576 legal corpus
chrF++ and +0.9222 mean sentence chrF++ (95% +0.3981…+1.4900). Steps
500/750/1,000 reach +1.0269/+1.5160/+1.4443 with positive lower bounds.
Nevertheless, every checkpoint is rejected: step 250 changes negation
mismatches from 101 to 104, while later steps also exceed the 208 exact
critical-token baseline. The locked selector emits no selected checkpoint; its
SHA-256 is
`e1622b64d2fbf9d6e131d66e9f1dc195ac29e38554bc7e8b3bc0f60c2fcbf96a`.

A validation-developed, reference-free wrapper then compares the step-250
output with the already-clean step-750 expert and chooses the lower frozen
`(critical+negation, critical, negation)` mismatch tuple, keeping the primary
on ties. On validation it falls back 18/400 times, retains a significant
+0.5396 quality gain (+0.0570…+1.0582), and improves exact critical/negation
counts to 194/99. This is post-hoc development evidence, not a claim.

Before testing that rule, Mimi freezes another 400 paired human legal units
from the 61 disjoint JLT test-law groups, producing 400 cases per direction.
It rejects 3,145 exact or near overlaps with JLT train/validation, canary,
public-v3, automated-claim sources, and the first legal validation suite. The
test suite and manifest SHA-256 values are
`ea27ac27bb23e99dd3d4fe29b70bab7ebb660fcac9309b3d06f08e9124ca91ca`
and `4015fa87366eeb953cdcf666d3931fb99a556ef6c249a355f65905abb3b83676`.

The independent EN→JA result is promising but inconclusive. Corpus chrF++
improves 19.2917→19.9002; exact critical mismatches fall 180→161 and negation
mismatches 76→74. Only 22/400 cases invoke the alternate; warm p95 is 62.57 ms;
the conservative five-model size before tokenizer deduplication is 180,019,246
bytes. Paired mean sentence chrF++ is +0.5227, but its 95% interval is
-0.0816…+1.0782, missing the preregistered strictly positive lower bound. The
evaluator therefore emits `structure-fallback-rejected`. The contract, result,
and independent structure-audit SHA-256 values are
`af0878aab7baaab22b94b28be64b477ca00be9acccd8735326567974ee227974`,
`a713673b3d1ae9454bf4a4b7b22d87df9871342d0380820a7ec05654d103b3f3`,
and `7bac8dbaa5ad7bf2528e6384a67a1a80abce30d572581d17ee6d5debab202b27`.
No model, router, pack, Swift runtime, or app default changes.

A final bounded calibration arm starts from the quality-winning clean step-250
checkpoint and applies the already-authenticated token-local negative-space
dataset: 31,621 deterministic rejected pairs, including 2,114 negation
reversals and 6,093 number substitutions. Rejected strings are used only as
negative evidence; every positive target remains a licensed human/project-owned
reference. Alpha is fixed at 3.0 for 125 steps before training. The contract
SHA-256 is
`3bedd13711a4bc10732f3ff187e2f019871a3e3e16c976fd01b5da86a4998f80`.

The arm stops before q4 conversion or legal evaluation. Mean rejected-token
probability improves from 0.0252594 to 0.0242013, but general-development
chrF++ falls from 30.94675 to 30.86728, below the preregistered 30.90 floor.
The rejected full-precision weight and training-manifest SHA-256 values are
`684b66658f2cb443983f08361dfde3dba5fe9758fb318e5497acda2503bf6f93`
and `abfadcdecb81687d9e8dd3390760e863da36eb0e1efde257bfe3e73a63bc72ef`.
No downstream benchmark or runtime work is authorized.

## V4 distributability review remains fail-closed

A primary-source compliance-engineering review now binds the lineage-clean
140,875,806-byte pack and its existing release contract. ElanMT's adapted
weights remain proposed under CC BY-SA 4.0; KFTT retains its CC BY-SA 3.0 and
NICT notice; Tatoeba retains 9,305 per-row contributor/license records; ALT is
explicitly limited to the CC BY parallel sentence text rather than the
separately CC BY-NC-SA annotated treebanks; and Japanese Law Translation keeps
PDL 1.0 source, editing, unofficial-text, and no-warranty notices. The review
does not claim that CC BY-SA makes a GPL-style source-code offer mandatory, and
does not decide the jurisdiction-specific question of whether training text
makes the resulting weights adapted material.

The channel decision is deliberately split. A direct signed/notarized macOS
release is conditionally feasible after separate model-license scoping,
complete offline notices, use of the completed repository-relative portable
inventory, and every quality/policy gate. Mac App Store distribution remains blocked pending
qualified review or written permission because the Standard EULA and any
effective technological measure may conflict with ShareAlike's
no-additional-restrictions condition. The review and machine-readable matrix
hash to
`0c5f390d55e935b266f7110ac2fd2927d74c2d93966db00cad34c1ce126e8ca5`
and
`6b838a6479b4d9022124f39d258fe0203d17fc825bc617aece9ffbe06451af25`.
Distribution, app integration, and a default change remain unauthorized.

The portable-inventory blocker is now closed mechanically. A cloned pack
rewrites only the three manifests that contained local worktree paths; all
weights, tokenizers, routers, and tokenizer configurations retain their exact
hashes. The portable pack is 140,875,791 bytes with manifest SHA-256
`71f330302559a0948c1b35f6def2d6107b4b57728cf53e1e99e0279141d84e79`.
Its repository-relative contract and complete file inventory hash to
`3d8ccfdfb95d2365de21a0912bd2c370c8ebf6061c903d758798720a8dc0a8f6`
and
`fb62b633e56c5c2fc5aa983e71f8475db6da2588c99479fae6031dcd5b8f01e7`.
The real staging path authenticates that inventory and emits only
`blocked-development-only`; it does not weaken any remaining blocker.
The exact source/portable MLX replay matches 12/12 routes, hypotheses, generated
token IDs, safety/fallback decisions, and failure reasons. Its comparison
SHA-256 is
`885502fa4d31b595ccc76b74993ff24ad3e4d40a88d6f5965c6172c4544d1b58`.
Mimi's native Swift loader validates the portable pack, and cold smokes pass
for both directions' generalist and expert roles. The original two canary
critical-token failures are reproduced exactly, so no quality gate changes.

The public-license-file blocker is now closed mechanically in a successor
audit, not by editing the immutable portable-v2 record. The
`elanmt-release-clean-human-only-routed-moe-v4-portable-licensed-audit-v3`
directory contains five official Creative Commons legal codes, the official
Japanese PDL 1.0 PDF, its explicitly non-controlling English reference, and
the Japanese Law Translation site terms. The eight documents total 413,215
bytes, and the complete license directory including its manifest is 418,162
bytes. The pinned source-lock, release contract, portable inventory, and
license-bundle manifest hash to
`103ec604642ba75407c8236df3ba59a35f55dce3d554222d52b902557f7b0615`,
`1da6991a7f6640b749301b04d73115c57637d1cbd4319bd275f093f46ab348f1`,
`55d33deb8f6302152c934b601e3e55c7243cddc592bbd7c4a52f8b39befa9d04`,
and
`54c9334674e45dd3ba3ea55845fc0b1801a23dbe6b62a19b77eeb22eb4728bac`.

The source freezer fetches only HTTPS, requires exact pinned bytes and media
types, rejects changed sources before creating output, and records source URL,
resolved URL, language, purpose, and canonical/reference status. The Japanese
Law Translation HTML contains a randomized CakePHP anti-CSRF token; exactly
that single non-substantive value is replaced with a documented placeholder
before its normalized hash is checked. Repeated fetches reproduce the same
normalized snapshot. The real staging path independently verifies the nested
license manifest plus every portable-inventory entry. Its staged manifest
hashes to
`dd78ed1affaffcb91841767de551c2c23b156ef1292b62590f700e83bf3e734b`
and still says `blocked-development-only`, `doesNotAuthorizeDistribution=true`,
`doesNotAuthorizeAppIntegration=true`, and `modelPromotionEligible=false`.
Only `public-license-text-bundle-pending` was removed; policy, sealed quality,
license compatibility, and app-distribution review remain blockers.

## M2M-100 418M fails the frozen feasibility gate

The single-model bidirectional alternative was tested exactly once on the
source-frozen 40+40 diagnostic screen at Hugging Face revision
`55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636`, greedy decoding, FP16 MPS, and one
warm run. The 1,941,931,012-byte local snapshot peaked at 3,774,595,072 bytes
resident. Its authenticated p95 was 0.7441 seconds EN→JA and 1.0490 seconds
JA→EN, versus 0.0638 and 0.0806 seconds for the 140,875,791-byte Marian pack.

Quality fails decisively. M2M-100 scores 18.6602 chrF++ EN→JA and 40.8343
JA→EN, regressions of 9.8307 and 11.1008 points from Marian. Seven of eight
domain slices regress by more than the preregistered 1.0-point limit; only
JA→EN human-translated news improves (+2.2911). Critical-token failures are
7/40 EN→JA and 9/40 JA→EN, compared with Marian's 7 and 8 strict mismatches,
so neither direction satisfies the required strict improvement.

The candidate report and score SHA-256 values are
`4d98c098383573b932a0dfbbe86b8b8597962c76dae92f21d1f9f085a0095e38`
and
`1315bd9279297435034d0ffcea0d1b8ae7904b8571e0d79ed69d937a456a8b85`.
M2M-100 is rejected before quantization or MLX porting. This saves that effort
for constrained Marian distillation and does not change Mimi's runtime or
default.
