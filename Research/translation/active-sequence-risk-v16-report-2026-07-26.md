# V16 active sequence-risk result

Date: 2026-07-26

Status: **rejected before semantic judging**

## Execution

The single preregistered arm ran exactly 50 updates from the distributable safe
parent under contract SHA-256
`4a3c8ee0fa08a97bf9707501cb6c1b4d36d11b70bec58dfe2c16a64b720ff6c9`.
Only the registered step-25 and step-50 checkpoints were written.

Checkpoint identities:

| Step | Model SHA-256 | Training-manifest SHA-256 |
| ---: | --- | --- |
| 25 | `23c2e17600f5a32e062462beae4c3fa9ee320beee5dde1899d70d02b0e2404e2` | `ce7f87762d120c30656ffd53785228e092773f50c44a30a54f948dd84e2576db` |
| 50 | `48268212f292f91f09e224f742f4d684296549e9855f81d2f32cf52119525763` | `64553f46244e0732a1742f37ccc4f5421a40545c042f351c01c5e54a98ba7606` |

PCGrad detected an omission/repetition conflict in 46 of 50 updates. Mean
pre-projection cosine was `-0.0308` with range `-0.0784` to `+0.0188`; mean
post-rule cosine was `+0.0331`, and every post-rule value was positive. The
implementation therefore performed the intended conflict correction.

## Authoritative held-out result

The tracked result
`active-sequence-risk-v16-presemantic-result-2026-07-26.json` has SHA-256
`fd986b7099b72affd7a9dc3d51c1413911b087c0d7a21a044c95f24f74f646c1`.
Neither checkpoint clears all frozen pre-semantic gates.

### Translation quality

| Suite delta from safe parent | Step 25 | Step 50 | Required |
| --- | ---: | ---: | ---: |
| Fresh V16 corpus chrF++ | +0.064 | +0.085 | at least +0.200 |
| Fresh V16 mean sentence chrF++ | +0.034 | +0.049 | at least +0.200 |
| Fresh V16 long chrF++ | +0.189 | +0.200 | at least +0.150 |
| Fresh V16 worst stratum | -0.767 | -0.767 | at least -0.500 |
| V12 mean sentence chrF++ | -0.014 | -0.007 | at least +0.150 |
| V14 mean sentence chrF++ | +0.062 | +0.110 | at least +0.150 |
| V15 mean sentence chrF++ | +0.037 | +0.078 | at least +0.150 |
| V15 omission-risk chrF++ | -0.032 | +0.086 | at least 0.000 |

The fresh worst-stratum result is the seven-case terminology slice. Step 50
does improve critical (`+0.223`) and long (`+0.200`) translation, but its
fresh paired mean interval remains compatible with no gain (`-0.012` to
`+0.113`). V12 aggregate quality is slightly below the safe parent, including
`-0.201` on its omission-risk slice at step 50.

No checkpoint creates a new repetition or generation-limit failure on any of
the four suites. New exact/typed detector disagreements do exist, but they are
not sent to semantic judges because the non-semantic gates already fail.

### Active sequence risks

| Delta from safe parent | Step 25 | Step 50 | Required |
| --- | ---: | ---: | ---: |
| All-pair preference accuracy | +0.0044 | +0.0059 | at least +0.0250 |
| Omission mean margin | +0.0007 | +0.0010 | at least +0.0200 |
| Repetition mean margin | +0.0049 | +0.0067 | at least +0.0200 |
| Active fraction | -0.0059 | -0.0103 | at most -0.0200 |

The optimization rule is functioning, but the bounded arm is too weak to
produce the registered behavioral change. This is not grounds to increase the
learning rate, loss weights, or update count after seeing the result. Such a
change would be a new experiment requiring a new evidence review and contract.

## Decision

V16 stops at the pre-semantic boundary. Sonnet 5 and Opus 5 judging, q4
conversion, COMET, protected evaluation, runtime measurement, bundle
replacement, app changes, release, and public upload were not run. The
existing bundled translator and application routing remain unchanged.
