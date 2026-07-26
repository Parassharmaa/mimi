# V20 typed-numeric preservation strategy

Date: 2026-07-26

Status: **bounded directional continuation authorized; app integration and
publication remain unauthorized**

## Why this arm

V19's stored outputs passed chrF++, BLEU, and COMET, but Mimi's exact Swift
critical-token guard rejected 71/400 segments after both compact local parents
were tried. Several failures were genuine severe meaning changes, such as
`300 miles (480 km)` becoming `3 km`, Article 74 becoming 71, and list item
226 becoming 222. Broadly relaxing the guard is therefore unsafe.

The independent HPLT fallback recovered only 14/71 strict-surface failures,
left 57 unresolved, admitted at least one lexical-number omission, and pushed
the pair above 500 MB. V18's one-model distillation path missed the JA→EN
quality floor by more than 30 chrF++ points. V20 instead changes the
directional experts themselves while retaining the V19 architecture and strict
guard.

## Data

V20 reuses the existing 123,050-row-per-direction licensed human corpus. It
does not generate or rewrite translations. The builder:

1. authenticates parent train and validation hashes;
2. permits only the declared distributable license set;
3. requires provenance and attribution for non-project-owned rows;
4. excludes Unicode-NFKC exact matches against nine registered evaluation
   suites, including segmented source/reference strings;
5. requires train/validation normalized pair separation; and
6. labels rows whose non-empty bilingual typed signature is preserved by the
   human target.

After screening, EN→JA has 123,011 train rows, including 18,174 focus rows and
7,724 bilingual surface transformations. JA→EN has 123,012 train rows,
including 18,122 focus rows and 7,777 transformations. The screen removes
39/38 train rows and no validation rows. All text remains identical to the
authenticated parent; there are zero synthetic rows and no reasoning traces.

## Frozen experiment

Each V19 full-precision expert receives one 100-update continuation at
`5e-7`, effective batch 16. Focus token loss ramps from 2× to 3× while the
other 104k rows remain preservation replay with frozen-parent KL 0.5 and
L2-to-parent `1e-5`. Only steps 50 and 100 are eligible.

A checkpoint must improve its typed-numeric validation slice by at least 0.1
chrF++ while keeping full validation within 0.1 and every domain within 0.5 of
the exact step-0 parent. Failure retires that direction before q4.

Passing full-precision checkpoints still receive no app authority. They must
survive exact q4/group-64 conversion, four public regression suites, the full
400-segment Swift cascade screen, COMET and semantic non-inferiority, isolated
latency/RSS, bundle size, and license review. The strict runtime guard remains
unchanged. Mimi's current bundled resources remain byte-pinned unless the
exact Swift screen reaches zero failed-closed segments without a new safety or
quality regression.

## Result

Both directions are rejected before q4 conversion.

| Direction | Step | Full validation Δ chrF++ | Typed-number Δ chrF++ | Decision |
|---|---:|---:|---:|---|
| EN→JA | 50 | +0.0258 | −0.0102 | reject |
| EN→JA | 100 | +0.0054 | −0.0102 | reject |
| JA→EN | 50 | −0.0328 | −0.1253 | reject |
| JA→EN | 100 | −0.0548 | −0.1798 | reject |

No checkpoint reaches the frozen `+0.10` focus-slice threshold. EN→JA changes
too little and JA→EN moves in the wrong direction. Continuing the same
weighted-MLE objective is not justified. No q4 conversion, public/protected
suite evaluation, app change, bundle replacement, or upload is authorized.

The machine-readable result is
`typed-numeric-preservation-v20-result-2026-07-26.json`. The next credible arm
must supervise critical-value retention directly—through a token-level
auxiliary objective or source-conditioned constrained decoding—rather than
merely increasing the weight of otherwise ordinary sequence likelihood.
