# Literature review: the next Mimi translation strategy

Last updated 2026-07-21. This review asks a narrow question: what is the most
credible way to improve Mimi's 73.4 MB English↔Japanese student beyond the
current 31.31/56.52 non-claimable canary result without weakening licensing,
contamination controls, MLX deployment, or the Apple fallback?

The product constraint was subsequently broadened: 150 MB remains preferred,
but the model bundle may reach 500 MB if that buys real accuracy while retaining
real-time local inference. Apple is now a comparison baseline only; exceeding
Apple is insufficient without strong absolute quality and critical-error gates,
and the intended final failure path must not depend on Apple Translation.

## Executive decision

The next experiment should be **diverse, quality-aware, regularized
sequence-level distillation**, run separately for each direction.

The important change is not a larger raw corpus or private teacher reasoning.
It is a better combination of:

1. uncertainty-plus-diversity source selection;
2. several high-quality teacher translations rather than only the teacher mode;
3. bilingual selection of a canonical target and, in a controlled ablation, one
   genuinely distinct valid alternative;
4. mixed-domain curriculum training with an explicit frozen-base distillation
   loss to prevent catastrophic forgetting;
5. checkpoint averaging; and
6. an optional small preference-optimization stage using reviewer choices.

This keeps the deployed Marian architecture and 4-bit pack unchanged. It also
explains Mimi's measurements better than “add more good data”: capped ALT
improved its own development mix but severely regressed EN→JA on the canary,
whereas screened conversational data retained EN→JA and improved JA→EN. The
failure is domain adaptation and forgetting, not simply insufficient corpus
size.

## What the literature says

### 1. Sequence-level distillation remains the right basic mechanism

Sequence-level knowledge distillation trains a small autoregressive student on
complete teacher translations rather than requiring access to every teacher
token probability. That fits an API teacher and a Marian student with different
architectures and tokenizers. It is still the foundation of recent compact NMT
systems.

The newer result is that always taking the teacher's single most likely output
throws away useful information. Galiano-Jiménez et al. find that sampled or
n-best translations can have slightly lower individual teacher quality yet
produce better students through greater lexical and structural diversity. Their
ablations attribute the gain to genuinely different translations, not merely
duplicating examples. See [Beyond the Mode: Sequence-Level Distillation of
Multilingual Translation Models for Low-Resource Language
Pairs](https://aclanthology.org/2025.findings-naacl.372/).

**Mimi implication:** retain multiple teacher candidates through review. The
first training control should still use one canonical target per source. A
pre-registered second arm may retain one additional reviewer-approved,
meaning-preserving alternative and sample one target per source per epoch. Do
not train on unreviewed diversity and do not put rationales in the target.

### 2. Select informative and diverse sources, not merely the hardest rows

Pure uncertainty mining tends to return repetitive failures and noise. Pure
diversity mining tends to return broad but easy examples. Hybrid Uncertainty
and Diversity Sampling stratifies by model uncertainty, clusters embeddings,
and selects diverse items within uncertainty strata; it outperformed either
criterion alone in multi-domain NMT adaptation. See [To Label or Not to Label:
Hybrid Active Learning for Neural Machine
Translation](https://aclanthology.org/2025.coling-main.206/).

Curriculum work reaches a compatible conclusion: data should gradually move
toward relevant, clean domains rather than using one static mixture. See
[Learning a Multi-Domain Curriculum for Neural Machine
Translation](https://aclanthology.org/2020.acl-main.689/) and [Dynamic Data
Selection and Weighting for Iterative
Back-Translation](https://aclanthology.org/2020.emnlp-main.475/).

**Mimi implication:** mine each direction independently. Use the current
student's token entropy or sequence margin for uncertainty and its mean-pooled
encoder state for diversity, avoiding a new embedding-model dependency. Apply
the benchmark's domain quotas before selection. Sample across uncertainty
strata rather than choosing the lowest chrF or highest entropy tail.

### 3. Mimi's ALT regression is consistent with catastrophic forgetting

Saunders and DeNeefe show that what an adapted NMT model forgets is connected to
the target-vocabulary coverage of the adaptation data. This is directly
relevant to Mimi: news-heavy ALT changed the target distribution and improved
the ALT/KFTT development mixture while hurting conversational EN→JA. See
[Domain adapted machine translation: What does catastrophic forgetting forget
and why?](https://aclanthology.org/2024.emnlp-main.704/).

Earlier NMT work demonstrates two practical controls. L2 regularization toward
the pretrained parameters reduces overfitting, while an auxiliary cross-entropy
or KL term keeps the adapted model's output distribution close to the frozen
general model. See [Regularization techniques for fine-tuning in neural machine
translation](https://aclanthology.org/D17-1156/) and [Regularized Training
Objective for Continued Training for Domain Adaptation in Neural Machine
Translation](https://aclanthology.org/W18-2705/).

**Mimi implication:** replay alone is insufficient. During domain training,
keep a frozen copy of the pinned ElanMT student and add a preservation loss on
licensed replay examples:

```text
L = L_selected_target
  + alpha * KL(base_token_distribution || adapted_token_distribution)
  + beta  * ||adapted_parameters - base_parameters||²
```

The base and student share the exact architecture and tokenizer, so this is
straightforward token-level distillation even though GPT logits are
unavailable. The frozen base is a preservation teacher; GPT supplies the new
sequence-level knowledge.

### 4. Encoder alignment is plausible, but only as a secondary ablation

Encoder-aware sequence-level KD adds a cosine alignment loss between teacher
and student encoder representations and reports better low-resource and
out-of-domain generalization than vanilla sequence KD. See [Encoder-Aware
Sequence-Level Knowledge Distillation for Low-Resource Neural Machine
Translation](https://aclanthology.org/2025.loresmt-1.15/).

An API LLM does not expose compatible encoder states, so the paper cannot be
copied literally. Mimi can instead align the adapted encoder with the frozen
base encoder on replay rows. This may preserve general bilingual structure, but
it may also resist useful domain adaptation. It belongs after the KL/L2
baseline, not in the first experiment.

### 5. Preference optimization is a credible second stage

Direct Quality Optimization adapts DPO to NMT using a translation quality
estimator as a preference proxy. The WMT 2025 paper reports improvements across
BLEU, COMET, CometKiwi, BLEURT, and human MQM evaluation, including transfer to
languages not present in preference training. See [Cross-lingual
Human-Preference Alignment for Neural Machine Translation with Direct Quality
Optimization](https://aclanthology.org/2025.wmt-1.2/).

**Mimi implication:** the bilingual review workflow already creates a stronger
signal than an automatic proxy: selected versus rejected teacher candidates.
If regularized supervised distillation first improves the reviewed development
set, run a short DQO/DPO-style stage using the frozen supervised checkpoint as
the reference policy. Never manufacture preference pairs from tiny score
differences, and never let the instant judge alone define them.

### 6. Candidate quality estimators are useful triage, not ground truth

QE-fusion improves candidate pools by combining spans selected with a quality
estimator, and xCOMET provides sentence scores plus error-span detection for
critical omissions and hallucinations. See [Don't Rank, Combine! Combining
Machine Translation Hypotheses Using Quality
Estimation](https://aclanthology.org/2024.acl-long.653/) and [xCOMET:
Transparent Machine Translation Evaluation through Fine-grained Error
Detection](https://aclanthology.org/2024.tacl-1.54/).

Learned metrics also have domain, version, precision, and reporting pitfalls;
scores are not automatically comparable across setups. See [Pitfalls and
Outlooks in Using COMET](https://aclanthology.org/2024.wmt-1.121/).

**Mimi implication:** a pinned xCOMET/MetricX-style evaluator may order review
and flag spans, but it cannot by itself approve data or promote a model. In the
reviewer-free training lane, acceptance must instead require multiple distinct
translation engines, roundtrip semantic checks, and a separately calibrated
bilingual judge; every such row remains promotion-ineligible. Store every model
revision, software version, precision, and metric signature. Human comparison
remains optional stronger evidence rather than a prerequisite for experiments.

### 7. Checkpoint averaging is low-risk and nearly free

Checkpoint averaging is widely effective in NMT and adds no deployed parameters
or latency. Simple averaging of several converged checkpoints generally captures
most of the benefit; elaborate weighting offers little extra in a flat region.
See [Revisiting Checkpoint Averaging for Neural Machine
Translation](https://aclanthology.org/2022.findings-aacl.18/).

**Mimi implication:** save checkpoints on the reviewed development schedule and
average the best three adjacent checkpoints selected without looking at the
canary or held-out suite. Quantize only the averaged full-precision model.

### 7a. Exact 4-bit-aware continuation is valid, but not a substitute for domain data

Four-bit NMT work shows that retraining through quantization error can preserve
quality while keeping biases in floating point. More recent parameter-efficient
QAT work likewise trains through fake-quantized weights with a straight-through
estimator. See [Compressing Neural Machine Translation Models with 4-bit
Precision](https://aclanthology.org/2020.ngt-1.4/), [L4Q: Parameter Efficient
Quantization-Aware Fine-Tuning on Large Language
Models](https://aclanthology.org/2025.acl-long.99/), and [PE-QAT: Parameter-
Efficient Quantization-Aware Training for Large Language
Models](https://aclanthology.org/2026.acl-srw.63/).

**Mimi implication:** a QAT ablation must reproduce the pinned MLX affine
quantizer exactly—including float16 source weights, groupwise signed scales,
zero-aligned edge adjustment, float16 stored scale/bias, and floating-point
Linear biases—rather than use a generic symmetric INT4 approximation. Select
on licensed development data, then run the authoritative MLX converter. The two
implemented EN→JA pilots slightly improved their quantized development scores,
but their shipping-kernel canaries reached only 30.81 and 30.88 versus the
current 31.31. QAT is therefore retained as a tested tool for future reviewed
teacher data, not as a reason to continue tuning the current licensed mixture.

### 8. Cyclical distillation is interesting but too risky for the first run

CycleDistill reports large first-iteration gains when bootstrapping low-resource
MT from monolingual text with LLM translations. See [CycleDistill: Bootstrapping
Machine Translation using LLMs with Cyclical
Distillation](https://aclanthology.org/2025.wat-1.7/). Mimi is not truly
low-resource, however, and repeated self-training can amplify teacher or student
errors. One reviewed teacher iteration is justified; automatic cycles are not.

Likewise, Self-Evolution KD reports gains by adapting the mixture of teacher
token distributions and one-hot ground truth according to token difficulty, but
it requires teacher softmax distributions that the API teacher does not expose.
See [Self-Evolution Knowledge Distillation for LLM-based Machine
Translation](https://aclanthology.org/2025.coling-main.686/). It is not a fit for
the current GPT API lane.

### 9. A single physical bidirectional model is a later compression experiment

Bidirectional Training improves bilingual alignment without adding parameters
and complements data distillation. See [Improving Neural Machine Translation by
Bidirectional Training](https://aclanthology.org/2021.emnlp-main.263/).

Niu et al. train one standard model for both directions by swapping every
parallel pair, balancing the two directions, sharing the vocabulary and tied
embeddings, and prepending a target-language marker such as `<2en>`. Their
single-model result reduces deployment cost, but the paper also reports a small
high-resource quality regression and notes that unbalanced language pairs may
require oversampling. See [Bi-Directional Neural Machine Translation with
Synthetic Parallel Data](https://aclanthology.org/W18-2710/).

Multi-teacher distillation is the natural extension: individual bilingual
models teach one shared multilingual student. Sun et al. demonstrate one shared
encoder/decoder plus multilingual distillation, while Saleh et al. show that
negative transfer remains a real risk and motivate selective teacher routing.
See [Knowledge Distillation for Multilingual Unsupervised Neural Machine
Translation](https://aclanthology.org/2020.acl-main.324/) and [Multilingual
Neural Machine Translation: Can Linguistic Hierarchies
Help?](https://aclanthology.org/2021.findings-emnlp.114/).

Mimi now implements that ablation with deterministically balanced licensed
batches, a direction-specific frozen teacher for token-level KL, macro-average
directional development selection, and optional `<2ja>`/`<2en>` source markers.
A 50/50 parameter mean collapses to immediate EOS, proving the independently
trained specialists are not neuron-aligned. The 300-step EN-initialized student
reaches 35.95/13.67 development chrF++; adding markers reaches 35.29/12.84. The
reciprocal marked JA-initialized student preserves 50.03 JA→EN but reaches only
4.77 EN→JA. One 4-bit physical model is 39.14 MB, but none of these pilots is a
quality candidate. The 73.4 MB specialist pair remains preferred while longer
shared-student work is treated as compression research, not promotion evidence.

An 800-case public stress comparison reinforces the evaluation literature's
warning about test composition. Preferred-v2 beats Apple in aggregate on
KFTT/ALT/Tatoeba by +8.22/+5.54 mean sentence chrF++, yet loses the separately
sourced ALT news slice by -3.35/-4.87 with paired 95% intervals below zero.
KFTT and Tatoeba are documented ElanMT training/pretraining sources, so the
aggregate is not independent evidence. Mimi must select on a separate reviewed
development set and make its only promotion claim on the sealed product-domain
suite; a large public sample is not automatically an unbiased sample.

The licensed-unified regularization ablation supports direction-wise selection
rather than all-or-nothing corpus adoption. A frozen-preferred KL/L2 objective,
macro-domain selection, KFTT retention gate, and three-checkpoint average
produced a JA→EN child that preserves the canary and improves the 400-case public
stress direction over preferred-v1 by +0.67 mean sentence chrF++ (95%
+0.08…+1.35). The identically trained EN→JA child has an inconclusive +0.37
stress delta (-0.11…+0.87) and regresses the canary. Preferred-v2 therefore
keeps the former EN→JA child and adopts only the new JA→EN child. This is exactly
the kind of asymmetric negative transfer hierarchical or directional selection
is designed to contain; it is still not held-out promotion evidence.

### 10. Vocabulary trimming is not worth the risk

Mimi is already far below the model-size target with a 32K vocabulary. A 2024
negative-results study found BPE trimming inconsistent and capable of heavy
quality degradation. See [An Analysis of BPE Vocabulary Trimming in Neural
Machine Translation](https://aclanthology.org/2024.insights-1.7/).

**Mimi implication:** preserve the verified tokenizer and spend the size budget
on quality. Do not retrain or trim the vocabulary in the next round.

### 11. The 500 MB ceiling changes the architecture search, not the evidence gate

The first larger dense baseline is HPLT v2. HPLT publishes directional
Transformer-base Marian checkpoints trained on its parallel corpus under CC BY
4.0: [EN→JA](https://huggingface.co/HPLT/translate-en-ja-v2.0-hplt) and
[JA→EN](https://huggingface.co/HPLT/translate-ja-en-v2.0-hplt). Each native
checkpoint is 307,935,566 bytes and converts to about 219.5 MB float16 before
quantization. The architecture remains MLX-portable: six 512-wide layers, eight
heads, a 2,048-wide FFN, and a 64K joint SentencePiece vocabulary.

The direct evidence rejects HPLT as an off-the-shelf replacement. EN→JA reaches
36.55 on the tiny canary but only 18.37 on the 400-case public stress direction;
JA→EN reaches 55.91 and 38.90. The large canary reversal demonstrates why model
selection cannot use a few attractive examples. HPLT may remain a fine-tuning
initialization or teacher-diversity source, but quantizing the unchanged pair is
not worth shipping work.

The next dense baseline was the MIT-licensed 418M-parameter
[M2M100](https://huggingface.co/facebook/m2m100_418M). A 4-bit port could fit
inside 500 MB, but the FP16 canary scores only 24.10/47.65 with roughly
0.49/0.39-second p95 latency, so it is rejected before porting. mT5-small is
Apache 2.0 but spends a
large fraction of its roughly 300M parameters on a 250,112-token multilingual
vocabulary, making it a less attractive bilingual shipping candidate unless
fine-tuning evidence says otherwise.

The missed single-model compression control is
[SMaLL-100](https://aclanthology.org/2022.emnlp-main.571/), a distilled
M2M100-compatible model with a 12-layer encoder, three-layer decoder, and about
330M parameters. Its paper reports 3.6× lower size and 4.3× faster inference
than M2M100 1.2B, and explicitly motivates shallow decoding plus balanced
distillation for resource-constrained translation. The pinned
[MIT checkpoint](https://huggingface.co/alirezamsh/small100/tree/8ab680e26a596d2e3d2d2d17ae0f68df1037328c)
supports Japanese and English in one physical model. That makes it a better
architectural hypothesis than unchanged M2M100, but evidence still controls
the decision.

The authenticated 1,337,145,439-byte FP32 snapshot reaches only 24.57/51.58
chrF++ on Mimi's non-claimable EN→JA/JA→EN canary under greedy decoding, with
0.258/0.192-second p95 and a conservative 3.50 GB process peak. Paper-default
beam five lowers EN→JA to 24.26, raises JA→EN only to 53.33, and increases p95
to 1.742/0.567 seconds. A provenance-bound balanced rank-16 LoRA pilot trains
1,179,648 parameters over 8,692 licensed rows. Its step-50/100 greedy results
are 24.65/50.60 and 24.60/50.04; a 10× learning-rate control collapses to
1.50/5.05. The unchanged and lightly adapted model are rejected before an MLX
port. Shallow-decoder distillation remains supported as a strategy, but the
multilingual checkpoint's Japanese quality is too far behind the existing
bilingual students.

A more recent and much more relevant baseline is CyberAgent's June 2026
[CAT-Translate-0.8B](https://huggingface.co/cyberagent/CAT-Translate-0.8b), an
MIT-licensed single causal model specialized for both Japanese↔English
directions. Its training recipe—synthetic parallel data, diversity-first then
quality-first SFT, followed by multi-objective GRPO with semantic, lexical,
format, and length rewards—is strong independent support for Mimi's staged
distillation plan. The published
[MLX 4-bit conversion](https://huggingface.co/hotchpotch/CAT-Translate-0.8b-mlx-q4)
is pinned at `84cbdd97cf628fa98fcd5a757d2599ebee765cd7` and occupies 453,006,430
bytes, so it is the first credible one-model baseline inside the hard 500 MB
model ceiling.

Direct measurement nevertheless rejects it as Mimi's runtime. CAT scores
34.42/56.94 on the tiny canary but only 24.90/42.47 chrF++ over the 800 public
stress cases, versus preferred-v2's 33.23/54.00. COMET-22 also favors
preferred-v2 overall (0.8800/0.8449 versus 0.8618/0.8034), so the gap is not
merely CAT producing less reference-literal paraphrases. CAT is 4–8× slower,
with 0.369/0.442-second public-stress p95 and roughly 874 MB peak RSS.

Two local QLoRA controls reinforce directional selection. A 12,067-row
licensed multi-domain adapter regressed every 50/100/150/200-step canary. A
lower-rate conversational/UI adapter improves JA→EN canary 56.94→59.66 at step
100 and the conversational stress slice 57.63→61.63, but its all-domain JA→EN
stress score is essentially flat at 42.39 and remains far below preferred-v2's
54.00; EN→JA regresses sharply. Base plus one adapter is 466,466,510 bytes.
This branch is rejected for runtime integration but CAT remains useful as a
distributable diversity teacher and as evidence for quality-first staged SFT.

The newly published CAT training corpus does not pass Mimi's release-lineage
gate. Its card identifies ODC-BY-1.0 plus Common Crawl terms, several source
corpora, and Apache-2.0 `gpt-oss-20b`/`gpt-oss-120b` generators, but access is
gated and all unauthenticated schema/preview/parquet endpoints return 401. The
public metadata is corpus-level: it does not bind every row to a source URL,
source license, generator revision, and filtering decision. Mimi therefore
adopts CAT's two-stage recipe without importing its 7.11 GB release. Diversity
first, quality second, MinHash, language/length filtering, and independent
format/length anti-hallucination objectives are reusable; unverifiable rows are
not.

Two July refresh candidates do not alter that conclusion. Liquid AI's
[LFM2-350M-ENJP-MT](https://huggingface.co/LiquidAI/LFM2-350M-ENJP-MT) is a
single bidirectional 350M causal model with a published 381.6 MB MLX 8-bit
pack and direct LFM2 support in Mimi's pinned MLX Swift dependency. Its LFM
Open License 1.0, however, excludes commercial use by legal entities at or
above USD 10M annual revenue. It therefore fails Mimi's unconditional
distributability gate absent a separate agreement and is not advanced as a
shipping candidate.

[Translate-15L](https://huggingface.co/WhirlwindAI/Translate-15L) is the
opposite trade-off: Apache-2.0, one bidirectional T5-small model, 60.5M
parameters, and a 244,616,918-byte authenticated FP32 snapshot. Its card does
not publish a legible Japanese score and warns that lower-resource languages
remain limited. Exact local screening confirms that limitation. Greedy decoding
scores 0.00/1.37 canary chrF++ EN→JA/JA→EN. The card's beam-4 setting still
scores only 0.00/4.11, emits empty strings or long repeated hyphens, and reaches
2.60-second JA→EN p95. Both reports bind revision
`ce860c33668440b031e30f50cc31377c6b6fac59`; the beam score SHA-256 is
`813801ccd9fd7f93e7db888e509621dc8085af5cc78c67c74d7503e76ca0d0db`.
No MLX conversion or fine-tuning follows.

Tencent's May 2026
[Hy-MT2-1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B) adds a more serious
hard-ceiling control. It is one Apache-2.0 bidirectional model trained for
real-world translation. AngelSlim's Sherry quantization constrains most linear
weights to a sparse ternary grid, placing the official file at 461,860,800
bytes. This is exactly the sort of accuracy-first exception the 500 MB ceiling
was intended to test: it spends much more than the 150 MB preference while
remaining a single on-device model.

The runtime evidence needs two qualifications. First, the official GGUF at
revision `9df5c824a00a744fb0512a29c640466f4d97dfb0` hashes to
`cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93`
but fails to load in the referenced open llama.cpp STQ branch because the file
still uses an earlier tensor layout/type ID. That exact pairing cannot ship.
Second, the community Apache-2.0
[MLX conversion](https://huggingface.co/kuotient/Hy-MT2-1.8B-1.25Bit-MLX)
at revision `03d1df683157fde0a4ec80636e749867d0c13a5e` carries a custom Python/Metal
kernel, not an already supported MLX Swift model. Its authenticated files total
464,192,044 bytes.

The MLX research control is nevertheless functional on this Apple Silicon
machine. Greedy canary chrF++ is 35.62 EN→JA and 60.70 JA→EN, above the current
31.31/56.52 generalist pair, with about 805 MB process RSS. Warm p95 is
0.735/0.601 seconds. The matched 800-case stress run rejects that canary signal:
Hy-MT2 reaches only 22.64/44.09 versus preferred-v2's 33.23/54.00, with paired
mean sentence chrF++ deltas -10.10 (-12.40…-7.91) and -11.85
(-13.94…-9.79). Stress p95 is 1.026/0.850 seconds and 127/800 outputs fail the
exact critical-token audit. No learned-metric run, Swift/Metal port,
fine-tuning, or app integration follows.

Even a quality win would remain legally conditional. The paper describes about
one trillion multilingual monolingual/parallel mid-training tokens and broad
family-specific business, domain, and instruction data, but does not name the
source corpora or their licenses. Apache-2.0 permission on the released weights
does not by itself meet Mimi's stricter requirement to prove the commercial
rights of every training source. Hy-MT2 can test the technical ceiling now; it
cannot become a shipping model or release teacher without a publisher-supplied
data-rights inventory.

NiuTrans's [LMT-60-0.6B](https://huggingface.co/NiuTrans/LMT-60-0.6B) is a
useful counterexample to the idea that a conventional one-model architecture is
automatically safer. It is an Apache-2.0 Qwen3-based 60-language translator,
so MLX conversion is routine and the same architecture has a plausible MLX
Swift path. The paper reports strong Japanese FLORES results, and the model
card's prescribed prompt produces a nearly competitive local canary after an
exact affine q4/group-64 conversion: 31.38/54.15 chrF++ at 346.9 MB.

The matched 800-case test rejects it. Real-time greedy decoding scores only
17.92/40.42 against preferred-v2's 33.23/54.00; paired mean sentence chrF++
deltas are -16.08 (-18.52…-13.78) and -15.61 (-17.93…-13.37). It also has
141 exact critical-token failures, visible entity hallucination and repetition,
0.491/0.397-second p95, and 745.6 MB peak RSS. The public recipe demonstrates
beam 5, so this is specifically a failure under Mimi's real-time decoding
contract, not a claimed reproduction of the paper's FLORES setting. The scale
of the loss makes a slower beam sweep a poor next investment.

The provenance result is independently negative. The
[LMT paper](https://arxiv.org/abs/2511.07003) describes 90B mixed monolingual
and bilingual tokens, names broad corpus families, and adds billions of
synthetic pairs, but does not enumerate per-row rights and generator versions.
The released [SFT dataset](https://huggingface.co/datasets/NiuTrans/LMT-60-sft-data)
has no license tag. Its pinned EN–JA shard has 13,169 rows, all labeled as
FLORES-200, NTREX-128, IWSLT 2017/2022, or WMT news 2020/2021/2022. This does
not prove that an exact reported test split was present, but it turns public
benchmark scores into contamination-sensitive diagnostics and makes the shard
unsuitable as independent Mimi training or promotion evidence. Apache-2.0
checkpoint terms cannot fill either evidence gap. For Mimi, LMT remains a
rejected technical control rather than a release model or teacher.

### 11a. Reviewer-free distillation can generate training signal, but needs layered independence

Recent compact-NMT work reinforces two points. [Mind the Gap: Diverse NMT
Models for Resource-Constrained
Environments](https://aclanthology.org/2025.nodalida-1.21/) reports that
sequence-level distillation can yield much faster students, but that teacher
quality/capacity and target script materially affect success. [A Self-
Distillation Recipe for Neural Machine
Translation](https://aclanthology.org/2025.findings-acl.261/) adds warm-up and
gradient adaptation specifically to prevent a teacher from pushing the student
against the translation objective. These results support strict teacher
selection plus frozen-parent preservation, not indiscriminate synthetic scale.

Mimi's executed control agrees. A three-forward/three-reverse-engine funnel,
strict English mutual entailment, and calibrated Qwen3-8B bilingual judge kept
256 of 2,000 BTEC sources. The first regularized student improved public-v2
slightly but regressed the quantized canary; doubling teacher dose improved the
public result further but repeated the regression. A 15% adapted / 85% parent
weight blend preserves every canary output and improves the 400-case EN→JA
conversation slice by +0.17 mean sentence chrF++ (+0.04…+0.35), although its
all-domain change is inconclusive. This is preferred-v3 development evidence,
not a quality claim. Reasoning traces were neither requested nor stored.

The next teacher round should therefore scale accepted high-quality targets in
both directions, not merely raise their loss weight. The now-direction-aware
pipeline can use an MIT-licensed multilingual MiniLM NLI checkpoint for
Japanese roundtrips and the calibrated Qwen judge for either direction. GPT-5.6
remains the preferred candidate teacher once a credentialed, hash-confirmed
batch can be submitted; CAT remains the local diversity teacher.

### 11b. Optimize repeated work before adding a speculative model

The live Marian runtime already removed the dominant autoregressive redundancy
with exact decoder self-attention and encoder cross-attention K/V caching. The
offline Qwen judge exposed the same principle at a different level: native
batch generation plus a shared 614-token prompt-prefix cache cut a 16-row smoke
from roughly 40 to 13 seconds with exact judgment parity.

For live partial captions, [Self-Speculative Biased Decoding for Faster Re-
Translation](https://arxiv.org/abs/2509.21740) is a better fit than a second
draft model: reuse the previous translation as a draft when the source grows,
verify it in one pass, and resume only at divergence. It needs no auxiliary
weights. Mimi should test this only when the local model receives growing
speech partials; finalized segments already complete around 28–34 ms in Swift.

That test is now complete for Mimi's bidirectional Marian kernel, and the
no-bias form is rejected. Coarse 50/75/100% partial growth accepted only 1/240
previous outputs. Five-percent source increments then found a false acceptance:
parallel teacher-forced verification retained a 128-token draft even though
authoritative cached greedy generation diverged at token 7. Serial cached-greedy
verification would restore exactness but repeat the autoregressive work it was
meant to remove. No Swift implementation or product switch follows. The lesson
is architecture-specific: exact decoder K/V caching remains useful, but a
single parallel verification pass is not an identity proof for this quantized
encoder-decoder path.

If a larger causal translator later becomes the quality winner, speculative
decoding becomes relevant again. [Speculative Decoding Across
Languages](https://arxiv.org/abs/2605.30580) warns that small draft models have
weak non-English acceptance and finds that task-specific distillation can help,
while very cheap n-gram drafts may still win on end-to-end speed. The decision
must therefore use measured accepted tokens per second and total latency, not
draft-model perplexity alone.

### 11c. The next architecture should be encoder-heavy and direction-aware

The speed literature does not support making a sub-500 MB sparse MoE the next
default. Sparse routing reduces active arithmetic, not the expert weights that
must remain in memory. A June 2026 consumer/edge study found inference cost on
bandwidth-bound devices followed total rather than active parameters; its MoE
control was slower than a comparable dense model on Apple M2 Pro and RTX 4070
Ti despite lower nominal active FLOPs. This result is narrow, but it matches the
constraint Mimi actually has: model bytes and unified-memory traffic. See
[Does Mixture-of-Experts Actually Help Inference on Consumer and Edge
Hardware?](https://arxiv.org/abs/2606.21428). If specialization is useful, the
more relevant design is a dense shared backbone with tiny routed adapters;
[Mixture-of-Adapters for NMT](https://aclanthology.org/2024.findings-naacl.154/)
was explicitly designed to avoid ordinary MoE's parameter inefficiency and
training instability.

Encoder-decoder asymmetry is a more credible latency lever. Only decoder layers
sit on the sequential token path, so capacity moved into a parallel source
encoder is cheaper at inference. A 2025 NMT study that paired an LLM encoder
with a conventional translation decoder reports quality matching or exceeding
its tested baselines with 2.4–6.5× inference speedups and 75% less decoder KV
cache: [Beyond Decoder-only](https://aclanthology.org/2025.findings-acl.490/).
Those headline numbers do not transfer automatically to Mimi, but they justify
the next controlled architecture sweep: shared vocabulary/embeddings and a
deeper shared encoder, followed by either one shallow direction-token decoder
or two one-to-two-layer direction-specific decoders inside one package.

Direction-specific capacity is important even behind one API. Recent
multilingual post-training analysis finds asymmetric conflict and synergy by
translation direction and improves adaptation with direction-aware training
and group-wise merging: [Asymmetric Conflict and Synergy in Post-training for
Multilingual MT](https://aclanthology.org/2025.findings-acl.944/). Mimi's own
results agree: the best EN→JA and JA→EN checkpoints repeatedly come from
different data and regularization choices. "One model" should therefore mean a
single shared package and inference interface, not a requirement to share every
decoder parameter.

Fully non-autoregressive translation remains a high-risk accuracy experiment.
A broad 2024 evaluation finds modern NAT systems still below autoregressive
translation under stronger automatic and human evaluation and less robust out
of distribution: [What Have We Achieved on Non-autoregressive
Translation?](https://aclanthology.org/2024.findings-acl.452/). Multi-token
prediction is also not a drop-in optimization for a tiny student: smaller
models struggle unless trained with a forward curriculum, although that
curriculum can retain self-speculative benefits. See [Pre-Training Curriculum
for Multi-Token Prediction](https://aclanthology.org/2025.acl-long.1243/).

The revised experiment order is therefore:

1. finish strict final-translation sequence distillation and the matched human-
   reference control on the current fast specialists;
2. sweep shared-encoder/shallow-decoder students in roughly 100–250 MB int4,
   comparing a shared decoder against two tiny direction-specific decoders;
3. quantize selectively—keeping embeddings, output projection, or sensitive
   layers at higher precision only when the held-out gain justifies their bytes;
4. prototype previous-output reuse for growing live-caption partials;
5. test n-gram speculative decoding and multi-token heads only after profiling
   proves sequential decoder launches remain the bottleneck;
6. keep NAT and routed adapters as research arms, not the first shipping path.

Reasoning traces remain excluded. The teacher emits only final translations;
the student needs the target sequence distribution, not a second explanatory
output mode that it would never use in Mimi.

The first controlled 6-encoder/2-decoder implementation is now complete and
clarifies what the cited work does *not* imply. Post-hoc retention of decoder
layers 0 and 5 creates a 54.33 MB two-way int4 pack but collapses canary chrF++
to 0.144/0.075; repetition to the 192-token cap makes matched cached-decoding
p50 roughly 0.17 seconds rather than improving on the incumbent.
A 300-step EN→JA recovery run combines licensed references with token-level KL
from the intact preferred model and reaches only 4.981 validation chrF++ and
1.986 exact-int4 canary chrF++. The architecture is rejected in this form. The
next sweep must train a shallow decoder from the beginning, use progressive
layer dropping, or add intermediate-state distillation; direct surgery on a
converged decoder is not an efficiency algorithm.

The higher-data follow-up resolves part of that uncertainty. Canonical
source-only sequence distillation improves a 6-encoder/2-decoder student over a
matched human-reference arm at every 250-step evaluation and finishes at
12.823 versus 12.288 development chrF++. This supports final-sequence KD, not
reasoning traces. A 6-encoder/4-decoder student reaches 27.184 and cuts exact
quantized EN→JA warm p50/p95 by about 23%/19%, but loses 3.69 mean sentence
chrF++ on the canary. Checkpoint averaging loses more. Decoder depth therefore
is a genuine speed lever, but four layers remain below Mimi's protected quality
floor even with 72,050 teacher-generated unique sequences.

The classic deep-encoder/shallow-decoder result is real but does not transfer
by naive post-hoc layer insertion here. [Kasai et al.](https://arxiv.org/abs/2006.10369)
show that deep encoders, shallow autoregressive decoders, careful speed
measurement, and sequence KD can rival non-autoregressive systems. Mimi's
post-norm Marian cannot append a truly identity Transformer block using only
zero residual weights and unit layer norms; the 8-encoder/4-decoder initializer
collapses before slowly recovering. An exactly preserving alternative widens
encoder FFNs behind zero output columns. It produces only +0.061 development
chrF++ and then collapses to 23.592 on the exact 4-bit canary. The lesson is
that encoder-heavy allocation should be trained as an architecture from a
strong pretraining stage, not retrofitted into a converged six-layer teacher.

For sparse capacity, Mimi should prefer sentence/task routing before custom
token-level MoE kernels. [Beyond Distillation: Task-level Mixture-of-Experts for
Efficient Inference](https://aclanthology.org/2021.findings-emnlp.304/) reports
that task routing can expose deployable sub-networks with better throughput than
token routing, while [StableMoE](https://aclanthology.org/2022.acl-long.489/)
shows that unstable routing harms sample efficiency and advocates a learned,
then frozen lightweight router. That maps cleanly to Mimi: calibrate a router on
held-out correctness, freeze it, and select one directional/domain expert per
segment. It keeps total stored weights auditable and only activates one expert.

An even lower-risk version is a confidence-gated cascade. The 73.4 MB student
handles easy segments; low sequence confidence, repetition, abnormal length,
script failure, or entity/number mismatch routes to the larger model. Decoder
KV caching is implemented before judging larger-model latency because the
current MLX prototype recomputes the full target prefix every token. The small
student becomes a speculative draft only if measured acceptance reduces
end-to-end latency without changing output quality.

That prerequisite is now measured rather than hypothetical. Release Swift/MLX
incremental decoding gives 1.67× EN→JA and 1.81× JA→EN p95 speedups on 180 warm
canary translations per direction, with 2,400/2,400 generated-token parity against
the Python MLX implementation and no extra model bytes. This makes KV caching
the default inside the developer-gated runtime. Speculative decoding is not the
next default: this 61M-parameter specialist already completes typical segments
in roughly 28–34 ms, so a second draft model and verification pass must beat
that measured floor before earning bundle or implementation complexity.

The next no-retraining cache experiment has a negative result. Replacing
per-token self-K/V concatenation with capacity-rounded block growth preserved
every canary token for block sizes 16, 64, and 192, but made warm p50/p95 about
3–11% slower against stable baseline-before/baseline-after runs. At these short
outputs, MLX functional slice updates cost more than the avoided copies. Mimi
should not port this storage strategy to Swift. A precomputed sinusoidal
position table and a pure compiled decode step remain cleaner experiments
because they remove repeated graph construction without changing cache
storage.

The exact position-table ablation is also complete. It preserves all canary
tokens. The metadata-authenticated rerun improves EN→JA by only 1–2%; JA→EN
p95 improves by under 1.3% while p50 sits between its two surrounding
baselines. That effect is too small relative to run drift to justify a Swift
port. Compilation remains the only untested micro-kernel idea; it should
proceed only as a pure fixed-step graph with the same token-parity gate.

That compile probe is now closed without adding a product path. MLX 0.30.6
preserves exact results when the decoder step is compiled for one fixed cache
shape, but shapeless compilation fails shape inference in the quantized
`AddMM` primitive. The current cache length grows every target token, so the
fixed-shape function would recompile at every step. A viable compiled decoder
therefore requires a masked fixed-capacity cache first; that is a new design,
not a safe flag on the existing path, and must overcome the negative block-
cache result.

There is a distinct app-level win that model microbenchmarks miss: retain both
direction specialists after their first validated load. The developer-gated
engine previously held only one runtime, so a language switch could discard
and reload the other specialist. It now uses a configuration-scoped two-slot
cache and clears both entries when the model directory changes. This is an
output-identical switch-latency optimization; packaged alternating-direction
latency and peak RSS remain the acceptance measurements.

The most workload-specific recent proposal is
[Self-Speculative Biased Decoding for Faster Live Translation](https://arxiv.org/abs/2509.21740).
It reuses the previous translation as a free draft for a growing source,
verifies draft positions in parallel, and reports up to 1.7× retranslation
speedup. Mimi cannot use the paper's positive draft bias because it intentionally
changes the verifier distribution; the only admissible first arm is bias zero.
Conventional seq2seq speculative decoding is also credible—
[Xia et al.](https://aclanthology.org/2023.findings-emnlp.257/) report roughly
5× on larger translation/summarization systems—but it adds a trained drafter
and promises comparable rather than identical quality. The broader draft-model
study by [Yan et al.](https://aclanthology.org/2025.naacl-long.328/) finds that
draft latency, not language-model quality alone, dominates realized speedup.
That makes a second 30 MB Marian draft unattractive beside Mimi's existing
roughly 30 ms specialist unless acceptance is exceptionally high.

The previous-output experiment is now closed as a negative result on Mimi's
current model. First-divergence cache continuation exposes numerical drift
between parallel causal prefill and the authoritative incremental MLX kernel;
one unrestricted canary partial changes token 23. A guarded 64-token/zero-prefix
fallback restores exact output parity, but early partial sources trigger
95–192-token repetitive targets and accept no useful draft prefix. The measured
0.992×/1.013× total ratios are noise. The next prerequisite is prefix-aware
EOS/length training plus real 180 ms ASR traces, not a Swift verifier port.

Runtime vocabulary shrinking is another published NMT-specific lever:
[Shi and Knight](https://aclanthology.org/P17-2091/) report 2× decoding speed
without BLEU loss using alignment-derived target vocabularies. It remains a
poor immediate Mimi fit. The shared 32,001-token English/Japanese vocabulary,
code-switching, copied product names, numbers, and unseen entities make a
static shortlist difficult to certify, while exact fallback requires computing
the omitted logits and erases the gain. Keep it behind a 100% token-parity gate
rather than treating the paper's corpus BLEU result as a universal guarantee.

## Quality-first update: filtered ensemble sequence distillation

The strongest recent evidence still favors final-sequence distillation, not
reasoning-trace imitation. [Enis and Hopkins (2024)](https://arxiv.org/abs/2404.13813)
show that strong LLM translations can be compressed into a conventional NMT
student, while also finding contamination in a common public benchmark. That
combination matters for Mimi: a stronger teacher is useful only when teacher
sources and evaluation remain rigorously separated. A 2025 confidence-filtered
NMT study likewise finds that low-confidence teacher outputs can be removed
without losing translation quality, reducing both noise and training cost
([Sanchez-Cartagena et al.](https://doi.org/10.3390/app15148091)).

WMT25 systems reinforce the candidate-selection pattern. English→Japanese
submissions generated multiple translations and selected with document-level
reranking, MBR, COMET, or LaBSE; the Shy-Hunyuan system used multiple expert
teachers before supervised distillation
([WMT25 findings](https://aclanthology.org/2025.wmt-1.22/)). These systems are
far larger than Mimi's budget and their learned selectors cannot be copied
blindly, but the transferable lesson is to spend teacher compute on target
diversity and selection rather than expose private rationales to the student.

The next Mimi arm therefore uses a release-clean human-parallel backbone, caps
every domain, and admits synthetic targets only when they beat or complement a
licensed training reference under independent structure, semantic, and
confidence checks. The compact student receives only the selected translation
sequence. Frozen-preferred KL and L2 regularization limit catastrophic
forgetting, and the exact 4-bit runtime remains unchanged. Reasoning may be used
internally by a teacher service when supported, but neither hidden reasoning nor
free-form explanations are requested, stored, filtered, or trained on.

### Executed update: route narrow experts instead of globally replacing the student

The release-clean corpus expansion confirms that domain adaptation is much
easier than global replacement. A 123,050-row full-depth EN→JA continuation
improves the 1,400-case public-v3 direction by +0.639 mean sentence chrF++ (95%
paired bootstrap +0.257…+1.026), driven by legal and Wikipedia, but loses on
conversation and the protected post-quantization canary. Treating that same
checkpoint as a formal-language expert changes the decision: a frozen
source-only character n-gram ridge router routes 153/386 document/law-grouped
held-out cases, routes no conversation cases, and gains +0.398 mean sentence
chrF++ with a positive lower confidence bound (+0.130…+0.689).

This supports a small asymmetric mixture of experts before a new large
bidirectional architecture. Keep the proven 73.4 MB two-direction pair as the
generalist, add only direction/domain experts that earn their bytes, and ship a
portable deterministic router whose inputs are source text alone. One EN→JA
expert keeps the research bundle near 111 MB in the current duplicated-tokenizer
layout, or roughly 106 MB with one shared tokenizer; four 4-bit direction/expert
weight files plus shared tokenizer data are expected near 142 MB, still under
the preferred 150 MB budget. Those are packaging estimates until an exact
shared-tokenizer Swift bundle is built and measured. The router result is also
development evidence, not permission to integrate: the independently sourced claim suite,
Apple comparison, critical-error suite, exact Swift parity, latency, RSS, and
bundle-integrity gates still apply.

The symmetric follow-up succeeded as a more sharply bounded specialist. A
JA→EN legal continuation improves the 200-case public legal slice by +9.867
mean sentence chrF++ (95% paired bootstrap +8.128…+11.718). Its source-only
legal router then routes 35/359 document/law-grouped test cases—34 legal and one
Wikipedia, with none from conversation or news—and gains +0.918 overall with a
positive interval (+0.427…+1.569); it routes no product-canary case.

The first four-engine packaging result exposed a broader lesson for compact
distillation: quality-only lineage selection is insufficient. The preferred-v3
EN→JA interpolation included 256 automated-consensus rows explicitly marked
promotion-ineligible. Its +0.062 mean sentence chrF++ advantage over the
human-only parent on 1,400 public-v3 cases is inconclusive (-0.013…+0.146), so
the safer response is to discard the interpolation, not relax the data gate.
Re-fitting the expert-delta router against the human-only parent improves the
held-out development result to +0.432 (+0.153…+0.733), routing 160/386 cases
and none from conversation.

Packaging the two authenticated human-only generalists, both human-data
experts, and both portable routers now produces a 148,075,038-byte model pack.
The 247,494-byte attribution/provenance payload brings the combined size to
148,322,532 bytes without tokenizer deduplication. Its hash-bound trace covers
10 dataset files, five training manifests, 264,300 dataset-row occurrences,
and 9,305 unique Tatoeba attribution records, finding zero promotion-excluded
rows. Loading all four engines peaks at 401,031,168 bytes RSS, while router p95
is 0.101/0.052 ms on the canary. A direct Swift port reproduces all 2,800
public-v3 Python routes with maximum score delta 5.11e-15, and the two exact
expert engines pass 12/12 Swift/Python output-token parity under cached
decoding. The developer-gated Swift loader also validates the exact MoE pack
and passes cold selection/decoding smokes for all four engine roles. This leaves
the claim, critical-error, final license, and app-distribution gates unchanged.
In fact the smoke exposes a useful counterexample: both JA→EN engines
mistranslate the short legal heading `（立入調査等）`.

The compact terminology-memory ablation is now complete. It admits only short
exact normalized sources observed in at least two distinct law documents,
drops conflicting within-document observations, selects an observed human
target medoid, and requires exact critical-token equality. Its corrected 6,179
entries add 615,743 runtime bytes. On every untouched validation match it improves the
routed model by +13.591 mean sentence chrF++ EN→JA (9 cases) and +23.962 JA→EN
(213 cases), both with positive paired lower bounds. Public-v3 contains only
seven hits, all improved. This supports a narrow systems conclusion: exact
retrieval is useful for repeated, licensed formulas, but its sparse scores must
remain separate from neural generalization and cannot justify an overall claim.

The full source-routed public-v3 replay supports the narrow-expert strategy but
not promotion. It improves mean sentence chrF++ over the human-only generalists
by +0.837 EN→JA (95% paired interval +0.613…+1.089) and +1.295 JA→EN
(+1.003…+1.605), while leaving conversation exactly unchanged. A strict
critical-token guard falls back from 26 expert outputs. The earlier 212 count is
only the expert-routed subset where both engines fail. Auditing all selected
outputs with the corrected tokenizer finds 498 mismatches: 197 expert-selected
and 301 generalist-derived paths. The native runtime now fails closed for all
strict failures except the separately proven single-percentage equivalence and
passes failure smokes in both
directions. This is conservative structural screening rather than a semantic-
error estimate, and it reinforces the need for a pre-registered typed policy
and frozen critical-error suite.

The combined four-model, router, exact-memory, attribution, and audit payload is
149,576,980 bytes, still below the preferred 150,000,000-byte boundary. Native
lookup parity is exact on 2,800/2,800 public decisions and hypotheses. However,
the memory source rows are explicitly training-only and promotion-ineligible;
the release contract preserves that block. The experiment therefore validates
the technical shape without authorizing app integration.

Generic placeholder shielding is not the answer for a tokenizer that was not
trained on those labels. Across 711 sources containing numeric or URL
replacements, `[NUMn]`/`[URLn]` restoration fails in 328 generalist outputs and
329 expert outputs; restored quality drops by -2.004/-3.067 mean sentence
chrF++ EN→JA/JA→EN with wholly negative confidence intervals. The appropriate
literature-aligned next route is typed constrained decoding or training-time
placeholder exposure, evaluated behind exact output/quality gates—not ad-hoc
preprocessing in the shipping runtime.

The exact-pack Apple diagnostic sharpens the evaluation lesson. A deterministic
content intersection supplies 647 post-hoc claim-ineligible cases per direction.
chrF++ favors the routed pack, and its p95 is roughly 39×/35× lower, but pinned
COMET-22 is inconclusive overall (-0.00265/-0.00371). Both news directions and
JA→EN conversation significantly favor Apple, while Wikipedia favors the local
model. Compact-MT selection therefore needs domain-balanced learned-metric and
critical-error gates; aggregate lexical overlap plus latency is insufficient.

Two runtime ablations narrow the implementation strategy. Full-sentence
shallow-model draft verification is counterproductive for this small Marian
pair: acceptance is low and serial draft cost makes inference about four times
slower. Increasing nominal weight precision is not a quality fix either; the
current 4-bit/group-64 conversion beats tested 4-bit groups 32/128 and 6-/8-bit
variants on the canary. Continue to use exact decoder K/V caching, and profile
token-level speculation only if a future larger winner makes draft overhead a
small fraction of verified decoding.

## Evaluation-strategy update

Recent evaluation work reinforces three decisions in the new contract.
[Zeng et al. (2024)](https://aclanthology.org/2024.findings-acl.710/) identify
both public-test leakage and limited reference diversity as causes of misleading
MT evaluation, and show that carefully selected multiple references improve
alignment with human judgments. That supports two accepted references per case,
but not unfiltered self-generated references: Mimi requires model-family-
independent acceptance, immutable hashes, and an exposure scan before freezing.

[Dabre et al. (2025)](https://aclanthology.org/2025.mtsummit-1.29/) report that
byte-level n-gram variants can equal or improve segment-level correlation over
chrF/chrF++ for Japanese and other non-Latin targets. bytF++ is therefore worth
adding as a diagnostic in a future metrics pass, while the preregistered gate
keeps chrF++ and pinned COMET-22 so thresholds do not move after results are
visible.

Recent public sets remain useful external audits, not the product claim core.
[WMT24](https://www2.statmt.org/wmt24/translation-task.html) supplies a recent
English→Japanese general-domain test, and
[WMT21](https://www.statmt.org/wmt21/translation-task.html) supplies both
directions, but their source rights and public exposure make them inappropriate
for a distributable, contamination-resistant private core. FLORES+ is symmetric
and CC-BY-SA-4.0 through the
[Open Language Data Initiative](https://github.com/openlanguagedata/flores),
yet its public visibility and English-origin translationese make it a control,
not sole evidence. CoVoST 2 offers open conversational/speech-adjacent
English↔Japanese translations in the
[official corpus repository](https://github.com/facebookresearch/covost), but
its age and likely pretraining exposure likewise limit it to robustness checks.

The core decision is consequently new sealed project-owned/CC0 product-domain
sources. Their reference generator must not be GPT-5.6 if GPT-5.6 outputs train
the student; generator, reference judges, and pairwise engine judges are pinned
and separated by role/model family. Only final translations and structured
scores/tags are retained—never reasoning traces.

## Recommended experiment: DQRD-v1

The working name is **Diverse Quality-aware Regularized Distillation v1**.

### Stage A: freeze evaluation before generation

1. Finish the expanded 800-case held-out suite with two independently accepted
   references per case and document-level splits. This is 400 distinct cases
   per direction. Under the reviewer-free contract, an isolated generator may
   propose final references, but it must differ from every training teacher and
   two separately pinned judge families must accept both references with no
   error or critical tags. Validate provenance, request/response hashes,
   complete exposure coverage, semantic contamination, schema, and critical
   structures before any candidate output exists.
   It doubles the initial design so the 5% code-switching stratum has 20 cases
   per direction and reduces the risk that a small apparent Apple delta is a
   low-power artifact. Count distinct translations, not repeated ratings, as
   the effective sample size; see
   [Graham et al. (2020)](https://aclanthology.org/2020.emnlp-main.6/) and the
   paired-bootstrap analysis in
   [Koehn (2004)](https://aclanthology.org/W04-3250/).
2. Freeze a separate reviewed domain-development set. The held-out suite is
   opened only for the final candidate; the 12-case canary remains plumbing and
   gross-regression evidence only.
3. Re-run exact and 5-gram contamination screening against every training and
   teacher-source file.

### Stage B: select teacher sources

1. The initial 2,226-source KFTT rehearsal is complete; use its negative matched
   control result to require a larger domain-balanced dose rather than more
   weight on the same narrow source distribution.
2. The next expansion is now frozen at 2,400 reference-backed train rows:
   400 KFTT, 400 ALT, and 400 Tatoeba cases per direction, selected only after
   active-data, prior-teacher, ambiguity, duplicate, and protected-suite
   screening. Keep BTEC and project-owned source-only additions in a separate
   lane because they cannot support the same hidden-reference comparison.
3. Compute student uncertainty, encoder embeddings, domain, length, named
   entities, numbers, placeholders, and code-switching flags.
4. Select within the benchmark-domain quotas using uncertainty strata plus
   embedding diversity. Cap near-duplicate clusters and cap every origin.

The executed balanced round validates the cell-level stop rule. Qwen3-8B
produced 281 individually qualifying translations, but only 9/7 in the two
Wikipedia cells; consequently no synthetic training set was emitted. The
matched human-reference arm improves the 2,400-row public development suite but
changes a protected number/entity canary output, and 5–50% checkpoint blending
does not recover it. This is evidence for stronger candidate diversity and
explicit entity preservation during training—not for weakening admission or
regression floors.

The bounded retry result sharpens that conclusion. Alternative Qwen candidates
plus a semantically correct `August`↔`8月` protected-token canonicalization
raise the strict set to 290 and clear all six ten-row cell floors without
changing any metric threshold. Yet the matched students still do not advance:
JA→EN selects the frozen parent, while the only canary-preserving EN→JA blend
regresses public-v2 by -0.079 sentence chrF++ under the pinned shipping runtime
(95% paired interval -0.147…-0.020). Better filtered targets alone are therefore
insufficient at this dose. The next quality arm
should increase target diversity and domain coverage or move capacity into an
encoder-heavy student trained from initialization; it should not add reasoning
traces, increase synthetic loss weight, or tune further masks against the
canary.

### Stage C: generate and review diverse targets

1. GPT produces several independent candidate translations plus the existing
   structured translation brief. It never returns or stores chain-of-thought.
2. Deterministic gates check language, numbers, dates, units, URLs, placeholders,
   named entities, script, length, copied source, and protected-suite similarity.
3. A distinct pinned QE/judge lane adds review priority and error-span hints.
4. Two independently pinned bilingual judge models see only source, candidate,
   and compact criteria. Admit a target only on exact consensus plus every
   deterministic gate; reject all disagreements. Store scores/tags, never
   reasoning traces.
5. In a pre-registered diversity arm only, the same independent-consensus rule
   may admit one additional meaning-equivalent but lexically distinct target.
   Training samples one admitted target per source per epoch; validation always
   uses the licensed canonical reference.

### Stage D: regularized per-direction training

Run a small, pre-declared matrix rather than an open-ended sweep:

| Arm | Teacher targets | Preservation | Purpose |
| --- | --- | --- | --- |
| A | canonical only | replay only | Current sequence-KD baseline |
| B | canonical only | replay + frozen-base KL + L2-to-base | Primary recommendation |
| C | canonical + one approved alternative | same as B | Test diversity finding |
| D | best of B/C | short reviewed-preference DQO stage | Optional human-evidence lane only |

Use the same initialization, update budget, batch size, quantization, and source
mixture in every arm. Begin with mostly professional/general replay and gradually
increase reviewed domain examples, while retaining a fixed replay floor. Tune
EN→JA and JA→EN independently because every Mimi ablation so far is asymmetric.

Save frequent full-precision checkpoints. Select using a composite frozen-dev
criterion that rewards domain chrF++/learned-metric quality but rejects general
KFTT retention loss or any new critical error. Average the best three adjacent
checkpoints, then quantize to 4-bit MLX.

### Stage E: decision rules

Before opening the held-out suite, require in both directions:

- a positive change on the reviewed domain-development set;
- no more than 0.5 chrF++ regression on the professional general-development
  set;
- no new critical meaning, number, entity, negation, or omission error;
- successful exact-pack Swift/MLX inference; and
- the existing size, latency, RSS, integrity, and non-Apple failure-path gates.

Run exactly one selected candidate on the claim suite. Report chrF++, sacreBLEU,
a pinned learned metric with signature, domain slices, paired bootstrap
confidence intervals, plus the deterministic adversarial critical-error suite.
Optional blind-human comparison strengthens the evidence but is not required.
The automated product contract is now preregistered: it compares against the
frozen best prior local model, requires positive lower confidence bounds in both
directions, strong absolute quality floors, two blind automated judge families,
and zero critical errors. Apple remains a separately reported diagnostic rather
than an authorization dependency.

## Priority ranking

| Strategy | Expected value | Risk/cost | Decision |
| --- | --- | --- | --- |
| Frozen-base KL/L2 regularization + mixed curriculum | High | Low; training-only extra model | **Do next** |
| Uncertainty + diversity source selection | High | Moderate implementation | **Do next** |
| Strict automated teacher consensus + calibrated judge | Medium-high | High rejection rate; provisional only | **Scale both directions** |
| GPT-5.6 diverse final translations | High if filtering holds | API credential/cost | **Next teacher source** |
| Checkpoint averaging | Medium | Very low | **Do next** |
| Human-preference DQO | Medium-high | Requires human evidence | **Optional; not a blocker** |
| Encoder alignment to frozen base | Medium | May over-constrain adaptation | **Secondary ablation** |
| One shared bidirectional student | Size win, uncertain quality | Architecture/training risk | **Later** |
| Cyclical self-training | Uncertain for EN↔JA | Confirmation-bias risk | **Not first run** |
| Teacher reasoning traces | No demonstrated MT target value | Noise/privacy/audit cost | **Do not use** |
| Vocabulary trimming | Little size need | Documented quality risk | **Do not use** |

## Bottom line

The literature does not support training a tiny translator on hidden reasoning
traces. It supports transferring diverse final translations, selecting the
right sources, and preventing a small adapted model from forgetting its strong
general prior. Mimi's first reviewer-free control validates that direction but
also shows that 256 accepted EN→JA rows are too little for a robust quality
promotion: they produce a narrow conversation gain only after parent blending.
The next quality gain should come from more strictly filtered targets in both
directions and a broader independent development surface, not from shrinking
the tokenizer or forcing one physical model too early. Runtime work should
retain exact decoder K/V caching. Further decoding acceleration belongs behind
an exact-output parity gate; previous-output parallel verification is closed
for the current Marian kernel, and an extra bundled draft network is not
justified at the present 51/59 ms p95.

## July 2026 refresh: distillation and PEFT boundaries

[Evolving Knowledge Distillation](https://arxiv.org/abs/2605.09924) adds a
useful ordering result: a fixed small NMT student first learns from a
capacity-near junior teacher, then from progressively stronger teachers. In
the paper's homogeneous Transformer experiments, the staged student reaches
34.24 BLEU against a 34.32 senior teacher, whereas direct senior-to-student
distillation reaches 31.09. The released implementation is MIT-licensed, but
the evidence is on German/English and English/Czech fairseq systems and uses
same-family token-distribution KD; it is not direct proof for Marian plus a
closed causal/API teacher.

Mimi can still adopt the safe part of the strategy. The current full-precision
Marian teacher supplies a first, architecture-matched sequence/KL stage; only
after that stage preserves the frozen development surface should independently
accepted stronger final translations enter a second stage. CAT, Hy-MT2, and
LMT-60 cannot serve as the stronger stage because each loses the matched
multi-domain stress control. GPT-5.6 remains the prospective senior teacher
only after the sealed final-answer/no-trace lane and two-family filtering are
available. Compare staged junior→senior training against a direct senior-only
control with the same accepted rows and update budget. This is progressive
curriculum distillation, not training on reasoning traces.

[Song et al. (2026)](https://aclanthology.org/2026.loresmt-1.1/) report that
distillation from strong teachers can let 2B–3B decoder-only small language
models match much larger systems in a 200-language low-resource study, with
full-parameter fine-tuning outperforming LoRA and decoder-only teachers
outperforming encoder–decoder teachers. The transferable lesson for Mimi is to
keep testing strong final-translation teachers and full fine-tuning. The paper
does not justify deploying a causal 2B–3B model here: even ideal 4-bit weights
alone are roughly 1.0–1.5 GB, above the hard 500 MB bundle cap, and Japanese is
not the low-resource setting studied.

[Stackhouse and Debenedetto (2026)](https://aclanthology.org/2026.americasnlp-6.4/)
find OFT closest to full fine-tuning on NLLB-200-distilled-600M across 13
indigenous-to-Spanish pairs, while LoRA provides a strong parameter-efficiency
tradeoff. OFT is a useful training-only ablation if full fine-tuning the current
61M Marian student proves unstable. It does not make NLLB a shipping baseline:
the [official NLLB-200 distilled model card](https://huggingface.co/facebook/nllb-200-distilled-600M)
labels the weights CC-BY-NC-4.0 and explicitly says the research model is not
released for production deployment. Those restrictions are incompatible with
Mimi's distributable macOS bundle.

The refreshed priority is therefore unchanged at deployment time: distill
diverse final translations into the distributable compact Marian architecture,
compare full fine-tuning with a preservation-aware parameter-efficient control,
and quantize the winner. Teacher reasoning traces remain excluded; only final
candidate translations and independently auditable acceptance metadata enter
the pipeline.

## Executed algorithm check: reverse translation is not a safe router judge

The reverse-consistency intuition was tested directly rather than assumed. A
calibrated selector used opposite-direction generalist round trips to veto
formal/legal expert choices. On an untouched 1,389-case public test, both
paired confidence intervals are below zero: -0.439 mean sentence chrF++ EN→JA
(-0.728…-0.198) and -0.725 JA→EN (-1.060…-0.450). It also increases exact
critical-token mismatches from 227 to 241 and unsafe typed acceptances from 13
to 27. The result rejects reverse chrF++ as a routing signal for the current
pack and favors training-time preservation or source-side routing over serial
self-consistency decoding.

## Executed algorithm check: model confidence is not cross-expert quality

The exact packaged engines also test normalized generation confidence. The
generalist's and expert's mean chosen-token NLL values are internally valid and
reproduce every saved greedy token sequence, but their relative scale is not a
quality estimator across independently fine-tuned checkpoints. A calibrated
-0.15 margin regresses untouched-test mean sentence chrF++ by -0.491 EN→JA and
-0.939 JA→EN, with wholly negative paired intervals. JA→EN legal loses 6.645
points; exact critical-token mismatches rise from 255 to 280.

This closes another inference-time selector family for the current pack.
Output-side self-consistency and cross-model confidence both add serial work and
worsen safety. The next quality effort should alter the student during training
with stronger final-translation targets and preservation losses, or improve the
source-only router with features available before decoding. It should not add a
second candidate decoder to the real-time path.

The corresponding source-side classification control does not replace the
incumbent either. Weighting a linear expert-win classifier by observed delta
magnitude looks positive against the generalist, but a direct paired test is
-0.123 sentence chrF++ behind the current ridge router
(-0.253…-0.008). This reinforces the evaluation rule: every new router must
beat the deployed router, not merely beat the unrouted base. With inference-time
and source-only alternatives exhausted, new quality now depends on better
licensed/consensus teacher targets and training-time preservation.

## Executed algorithm check: symmetric safety fallback is too sparse

One remaining no-weight idea was to make the existing critical-token fallback
symmetric. The app-shaped policy already tries the generalist when a routed
expert fails. The ablation also tries the expert after a router-selected
generalist failure, accepting the alternate only when it passes the same exact
token policy. This is safer than reranking by confidence because the alternate
cannot turn a structurally valid first output into a failure.

Public-stress-v3 provides weakly positive but inconclusive evidence: the extra
direction recovers 23 cases and changes mean sentence chrF++ by +0.019 EN→JA
(-0.016…+0.064) and +0.057 JA→EN (-0.024…+0.145). The sealed 800-source
product audit is more decisive for runtime value. It tries a second model 429
times, changes 215 hypotheses, and recovers only five strict failures. JA→EN
p95 nearly doubles from 58.0 to 108.7 ms; the one JA→EN structural rescue is
merely `13:05` becoming `13:5`. Without references, that cannot count as a
quality win.

This closes symmetric fallback for the current four-engine pack. Conditional
second decoding is still real-time, but its sparse product-domain benefit does
not justify runtime complexity, extra energy, or a Swift parity surface. The
stronger lesson is that hundreds of safety failures originate in the student
outputs themselves. More teacher-quality training data, explicit
number/entity/negation preservation, and a new untouched final suite are higher
value than trying another bundled role after generation.

## Executed algorithm check: preservation needs product-domain support

The preservation-aware full-fine-tuning result refines the literature-derived
strategy. Reweighting licensed examples that already preserve numbers,
placeholders, and bilingual negation can improve a compact Marian student
without reasoning traces or synthetic text. EN→JA gains +0.441 mean sentence
chrF++ on 1,400 public cases with a positive paired interval, while the exact
4-bit architecture and latency remain unchanged. This supports transferring
final translations and structural behavior into the existing student rather
than deploying a much larger decoder-only teacher.

It does not establish a shipping win. On the new 400-source product-domain
EN→JA surface, substituting the adapted generalist inside the exact routed pack
reduces fail-closed acceptance by 19 cases. The public reference corpora reward
Wikipedia and legal phrasing; the sealed sources emphasize meetings,
conversation, UI, and omission-sensitive structures. This is a concrete
domain-transfer boundary, not evidence for weakening the guard.

The JA→EN control adds a second lesson: adapt the actual current checkpoint,
not an older base with the same architecture. Correcting that lineage changes
a decisive -1.468 public regression into an inconclusive -0.109, but still does
not produce the required positive lower bound. Preservation fine-tuning remains
the right family of experiment; its next inputs must be independently filtered
product-domain final translations, with enough replay to protect the current
general prior. Hidden reasoning traces remain unnecessary and excluded.

## July follow-up: QuickMT and negative-space learning

[QuickMT EN→JA](https://huggingface.co/quickmt/quickmt-en-ja) and
[JA→EN](https://huggingface.co/quickmt/quickmt-ja-en) provide direct modern
evidence for encoder-heavy, shallow autoregressive translation: 8e/2d at hidden
size 1,024 and 12e/2d at hidden size 768. Mimi's exact CTranslate2 canary does
not validate the released pair as a replacement—26.54/57.98 chrF++, an
813.7 MB two-model footprint, and 1.78 GB process RSS—but it strengthens the
architecture hypothesis. The failed 6→2, 6→4, and now 6→5 post-hoc ElanMT arms
clarify the implementation rule: decoder depth must be shallow during
pretraining/distillation; deleting layers from a converged six-layer decoder is
not the algorithm described by the literature. QuickMT's 63.3M-row training
mixture has no usable row-level license/contamination inventory, so neither its
data nor outputs are admitted as Mimi training evidence.

[NSL-MT](https://aclanthology.org/2026.findings-acl.465/) contributes a second,
orthogonal idea: generate three to five linguistically invalid outputs per
correct pair and combine ordinary positive cross-entropy with
severity-weighted negative evidence. The paper uses `alpha=0.7` (with 0.3–0.9
sensitivity) and reports gains across low-resource African-language settings.
That is promising but not direct evidence for a strong high-resource
English↔Japanese Marian student. Its whole-sequence `log P(v|x)` term is also
unbounded below and can be satisfied by suppressing an arbitrary token in the
bad sentence.

Mimi therefore implements a safer, explicitly non-paper-faithful control. It
keeps authenticated licensed references as the only positive targets and
creates deterministic number, unit, URL, placeholder, negation, omission, and
duplication corruptions used only as negative evidence. The added bounded term
is `-log(1-p(v_t))` at the first divergent token under the correct target
prefix, severity weighted. This changes no inference architecture, bundle size,
or latency. It uses no free-form synthetic translation, human reviewer, or
reasoning trace. Promotion still requires the ordinary untouched canary/stress
and sealed 400+400 gates; success on the generated negatives cannot validate
translation quality by itself.

The executed alpha sweep resolves the immediate hypothesis. At alpha 0.3 the
EN→JA bad-token probability is effectively unchanged. Alpha 3.0 lowers it by
about 3.6% relative while preserving a tiny positive full-precision validation
delta, but exact q4 canary chrF++ falls to 29.996. JA→EN lowers bad-token
probability by about 2.0% and adds 0.031 full-precision validation chrF++, yet
its q4 canary tokens are exactly identical to the incumbent. The localized
objective is stable and auditable, but neither direction passes the first
shipping-quality gate. Do not increase alpha blindly: the useful next
negative-space experiment would mine errors from independently judged
product-domain final translations and pair them with stronger positive targets,
not generate increasingly severe generic corruptions.

## Typed temporal validation and failure-triggered n-best decoding

Numerical adequacy needs a separate gate from aggregate translation metrics.
[Wang et al. (2021)](https://aclanthology.org/2021.findings-acl.415/) show that
small numerical errors can survive normal BLEU evaluation and propose exact
format fuzzing plus CI-style input/output checks. Earlier typed transduction
work by [Tu, Zhou, and Zong (2012)](https://aclanthology.org/2012.iwslt-papers.9/)
supports converting number/date/time expressions through a language-neutral
representation. That evidence justifies a narrow validator; it does not make a
regex-only result a correct translation.

Lexically constrained decoding is a possible repair mechanism. Dynamic beam
allocation ([Post and Vilar, 2018](https://aclanthology.org/N18-1119/)), FSA
multi-stack decoding ([Hasler et al., 2018](https://aclanthology.org/N18-2081/)),
and vectorized DBA ([Hu et al., 2019](https://aclanthology.org/N19-1090/)) can
force phrases or alternatives, but published results also show latency and
placement/duplication risks. [Mitra](https://aclanthology.org/2024.eamt-1.12/)
strengthens disjunctive surface constraints but is much slower than
unconstrained decoding. Training-time source factors
([Dinu et al., 2019](https://aclanthology.org/P19-1294/)) remain the more
promising eventual path when clean target exposure is available.

Mimi's no-credential control therefore tries beam-4 n-best filtering before
hard forcing. A fail-closed temporal parser accepts at most one valid Gregorian
date and one unambiguous 24-hour time, preserves order/multiplicity and every
residual digit/protected token, and rejects AM/PM, time zones, ranges, eras, and
unsupported forms. On the frozen 800-source surface, 290/386 strict failures
are candidate temporal surface normalizations, 33 match separately rejected
word-number/percent normalizations, and 63 contain concrete drops,
substitutions, duplication, or inventions. This is diagnostic, not evidence
that 290 translations are correct.

Public-reference evidence is insufficient: the temporal arm finds 25 strict
v1 candidates, only 12 of which the deliberately narrower reference parser can
validate; 13 references add Japanese-era notation and remain fail-closed. The
old percentage arm has only one validated case, so its status is now
`insufficient-evidence`, not a promotion pass. The minimum future gate is zero
unsafe accepts over at least 300 held-out cases per direction plus per-format,
adversarial, and Swift/Python parity evidence.

Failure-triggered beam-4 n-best adds only nine source-only typed candidates.
Triggered p95 is 0.461 seconds EN→JA and 0.339 seconds JA→EN, and typed date
preservation still permits visible semantic regressions such as losing the
meaning of “unless.” The runtime strategy is rejected. Hard constrained beam
search would add more latency without solving semantic placement, so it is not
implemented. The current strict runtime and app default remain unchanged.

## Exact output-projection shortlisting on MLX

Vocabulary shortlisting attacks the decoder's final matrix multiplication
without changing the encoder, decoder layers, quantized weights, tokenizer, or
greedy policy. It is attractive here because the shared Marian vocabulary has
32,001 entries while the target scripts are largely separable. The safe form is
not a benchmark-mined top-token list: Mimi builds static candidates from the
authenticated tokenizer's Unicode surfaces and adds source-local surface
variants. A prepared static projection avoids re-slicing the large tied output
matrix on every sentence, while a small dynamic extension covers input-local
terms.

The implementation is exact for any output whose winning tokens remain in the
shortlist, and the q4 subset-logit tests plus the 12-case canary establish exact
token/hypothesis/routing parity for this control. The shortlist reduces the
median projected vocabulary to 20,613 EN→JA and 15,196 JA→EN, but measured warm
p50/p95 changes range from a 0.18% regression to a 1.08% improvement—well below
the 5% continuation floor. Retained subset projections increase peak RSS by
23.95 MB (+8.38%) and more than double preparation time. At this model scale,
MLX decoder-layer work and dispatch overhead dominate enough that reducing the
final projection is not a useful end-to-end optimization. The arm stops before
the full source audit or Swift work.

This result closes tokenizer-script output shortlisting for the current
141,488,564-byte routed pack. Revisit only with a fused/gathered projection
kernel that does not retain duplicate materialized weights, or with a different
architecture whose output projection is proven to dominate Instruments traces.

## Quantized-kernel upgrade and next exact runtime sequence

[MLX 0.31.2](https://github.com/ml-explore/mlx/releases/tag/v0.31.2) includes a
new small-M split-K quantized matrix-multiplication path; the underlying
[MLX change](https://github.com/ml-explore/mlx/pull/3120) reports meaningful
microbenchmark gains for several small matrix shapes. That is not evidence of a
Marian batch-one translation gain: autoregressive token decoding may select a
different vector path, and Mimi's encoder/decoder shapes and dispatch mix matter.

The exact paired canary rejects the upgrade. Relative to a following 0.30.6
control, MLX 0.31.2 regresses p50/p95 in both directions and changes two EN→JA
token sequences. The observed drift also demonstrates why runtime-library
upgrades belong inside the same output-parity and quality gates as model changes.

The next exact, no-retraining runtime experiment is offline QKV packing: replace
the three self-attention Q/K/V quantized projections with one concatenated
projection and split views, and likewise pack encoder-attention K/V. It must
replace rather than duplicate weights. Require bit/tolerance-bounded projection
tests, exact decoded-token parity, an isolated projection microbenchmark gain of
at least 10%, then at least 5% end-to-end p50 and p95 improvement in each
direction with no RSS or bundle regression. Only after that should Mimi try
stable-subgraph `mx.compile`, residual-add/LayerNorm fusion, or the higher-risk
same-depth SSRU decoder distillation described by
[Kim et al.](https://arxiv.org/abs/2010.02416).

The QKV/KV experiment is now resolved. Packed q4 projections are bit-exact and
clear the isolated ≥10% self-QKV gate, but the full batch-one translator is
1–7% slower at direction-separated p50/p95. The likely dispatch win is too small
relative to attention, FFN, cache, tokenization, and synchronization work; the
larger packed kernels may also have less favorable scheduling. A useful side
effect is an 8.04% peak-RSS reduction after replacing the individual projection
objects. Keep the implementation research-only; do not present memory savings
as a latency win.

The next exact lane is `mx.compile` on stable, fixed-shape subgraphs, measured
with recompilation counters and first-use latency. Cache-length-dependent token
steps must remain outside shapeless compilation until shape safety is proven.
If compile does not clear an isolated ≥10% block gate, stop before custom Metal
residual-add/LayerNorm work.

That stop gate now fires. `mx.compile` preserves exact values, but fixed-shape
residual+LayerNorm improves only 0.19% and the full FFN residual block 4.77% at
the median of alternating timing blocks. This is not enough headroom to justify
dynamic cache-shape compilation, graph-cache memory, first-use stalls, or a
custom Metal maintenance surface. The remaining plausible speed research is a
trained architectural change—same-depth SSRU first—rather than more Python
graph rearrangement around this Marian implementation.

That same-depth SSRU screen is now complete and negative. The paper's stronger
architecture combines SSRU with removing decoder FFNs and reallocating depth to
a much deeper encoder; it does not establish that swapping only self-attention
inside Mimi's existing 6e/6d post-norm checkpoint will yield an end-to-end win.
Mimi therefore preregistered a cheaper Apple-Silicon proxy before training: six
decoder layers and their cross-attention/FFNs remain, while an exact-shape q4
SSRU recurrence replaces only self-attention compute. Across alternating timing
blocks it improves the measured decoder layer by 5.2267%, below the 10% floor.
Because every other translation component remains unchanged, expected full
sentence gain is smaller still.

Do not train or port that same-depth arm. A purpose-pretrained 12e/1d SSRU model
is not disproved, but existing Mimi evidence makes it a high-cost later project:
one shared Marian pilot fits in 39.1 MB but loses substantial direction quality,
6e/2d and 6e/4d students fail the protected quality floor, and naive deep-encoder
retrofits collapse or lose after q4. The defensible next use of teacher compute
is completing the independently filtered 400+400 product-domain reference set,
then distilling into the already fast specialists. Reconsider 12e/1d only with
a distributable pretrained initialization or a training plan large enough to
learn the architecture from its beginning—not by grafting it onto this model.
