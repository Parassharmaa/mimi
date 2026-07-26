# V21 tagged critical-value translation strategy

Date: 2026-07-26

Status: preregistered research strategy. It does not authorize model training,
protected evaluation, app replacement, release, or public upload.

## Decision

V21 should train the Marian specialists to copy source-bound value identities
through atomic tags. It should not add another full fallback, reranker, or
unconstrained beam search.

V19's exact Swift screen failed closed on 71 of 400 segments. Sixty-eight
failures contain numbers, two contain numbers plus opaque identifiers, and one
contains the unsupported Japanese decimal form `三・〇`. V20 increased the loss
of every token on a critical row by 2 to 3 times, but it did not directly
supervise value tokens. Its focus validation regressed in both directions.

The next model should learn where each protected value belongs while a
deterministic sidecar preserves what the value is.

```text
source
  -> conservative typed-value tagger
  -> tagged source plus immutable value sidecar
  -> V21 Marian specialist with cached greedy decoding
  -> exact tag identity and multiplicity check
  -> deterministic target-language value restoration
  -> existing repetition, plausibility, and critical-token guards
  -> return, local fallback, or fail closed
```

## Why this direction

The evidence rules out several nearby approaches:

| Approach | Result |
|---|---|
| V20 row-weighted maximum likelihood | No focus gain in either direction |
| Beam-4 recovery | Only 9 of 43 candidates recovered; triggered p95 0.34 to 0.46 seconds |
| Existing critical specialists | Failed independent transfer and safety gates |
| Round-trip and self-likelihood reranking | Regressed quality or safety |
| FP16 HPLT fallback | Recovered 14 of 71 and pushed the pair above 500 MB |

This is consistent with three primary findings:

- [Dinu et al. 2019](https://aclanthology.org/P19-1294/) show that source-side
  terminology factors can improve constraint adherence without requiring a
  general beam-search constraint at every step.
- [Post and Vilar 2018](https://aclanthology.org/N18-1119/) make dynamic
  constrained decoding practical, but it remains a beam-search mechanism with
  runtime cost.
- [Wang et al. 2021](https://aclanthology.org/2021.findings-acl.415/) show that
  numerical translation needs dedicated adversarial evaluation beyond BLEU.

V21 therefore starts with tag-aware greedy decoding. A narrow constrained
decoder is only a fallback experiment if trained greedy decoding nearly passes.

## Representation

Reserve 32 atomic tokenizer items, `<v00>` through `<v31>`. The current
400-segment suite has at most 13 typed values in one segment, so 32 leaves
headroom.

Example:

```text
Source: The scene had four helicopters and 140 firefighters.
Tagged: The scene had <v00> helicopters and <v01> firefighters.
Model:  現場には<v00>機のヘリコプターと<v01>人の消防士がいた。
Final:  現場には四機のヘリコプターと140人の消防士がいた。
```

Digit-bearing values, identifiers, URLs, and placeholders are restored exactly.
Japanese kanji numbers and English number words are restored using a
target-language lexical form only when the parser proves a one-to-one mapping.
Ambiguous parsing, missing or duplicated tags, unsupported ranges, and unsafe
lexicalization fail closed. Missing tags are never inserted silently after
generation.

## Data

Use only the existing authenticated human-reference V20 corpus:

| Direction | Critical focus rows | Surface transformations |
|---|---:|---:|
| EN→JA | 18,174 | 7,724 |
| JA→EN | 18,122 | 7,777 |

Keep the existing exact and near-duplicate screen against every registered
evaluation suite. Admit only rows with an unambiguous one-to-one canonical
value alignment for the first cell. No synthetic translation, teacher output,
or reasoning trace is needed.

## Objective

Replace V20's row-wide weighting with token-specific supervision:

```text
L = normal cross entropy
  + 8 * correct-tag cross entropy
  + extra-tag unlikelihood
  + 0.5 * frozen-parent KL on plain replay
  + 1e-5 * parent L2
```

Use balanced tagged-focus and plain-replay batches. The unlikelihood term
penalizes a tag at ordinary target positions and penalizes the wrong tag at an
expected tag position.

## Smallest first cell

Run JA→EN first because it accounts for 42 of V19's 71 failures.

1. Start from the full-precision V19 legal JA→EN parent.
2. Select 4,096 unambiguous tagged focus rows and an equal plain replay sample.
3. Train 100 updates with checkpoints at steps 50 and 100.
4. Give only the 32 new embedding rows a high learning rate.
5. Keep the existing network on a low-rate, parent-regularized continuation.
6. Select on the frozen 153-case typed validation slice and the full
   1,285-case validation set.

Convert to q4/group-64 only if all validation gates pass:

| Gate | Requirement |
|---|---:|
| Exact tag identity and multiplicity | at least 98% |
| Typed-validation strict failures | at least 50% lower |
| Full validation chrF++ | no worse than parent by 0.10 |
| Worst domain chrF++ | no worse than parent by 0.50 |
| New repetition, empty, negation, or protected-token failures | zero |

The exact JA→EN Swift continuation gate is 42 failures to 21 or fewer, with
zero new failures among the 158 previously accepted segments. Only then repeat
the experiment for EN→JA.

## Optional constrained decoding

If the only residual failure is a missing tag, test a failure-triggered finite
state or dynamic beam decoder over the atomic tag IDs:

- mask tags already used;
- prohibit end-of-sequence until every required tag appears;
- allocate beam states by a tag-coverage bit mask;
- preserve the existing semantic, plausibility, repetition, and critical-token
  guards after restoration.

This decoder is secondary. Prior untrained beam-4 recovery was too slow and
weak to justify making beam search the default.

## Size and runtime

The Swift Marian runtime currently hardcodes a 32,001-token vocabulary. A
passing V21 requires a manifest-driven vocabulary size and tag-aware
restoration, but no new neural architecture.

Thirty-two tied 512-dimensional q4 embedding rows add about 9 KB per
specialist, plus tokenizer metadata. Replacing the two V19 experts keeps the
four-engine research pack near 146.8 MB, within the 150 MB preferred ceiling.

Final promotion still requires:

- zero failed-closed segments on the exact 400-segment Swift screen;
- zero new failures among every previously accepted segment;
- adversarial number, date, time, unit, identifier, and legal-article tests;
- chrF++, BLEU, COMET-22, and independent semantic-judge non-inferiority;
- long-document and Apple comparisons on identical held-out items;
- isolated p95 at most 175 ms with no more than 10% latency or RSS regression;
- model bytes below 150 MB preferred and 500 MB hard;
- completed attribution and share-alike review.

V21 does not weaken Mimi's guards and does not use Apple Translation as a
runtime fallback.
