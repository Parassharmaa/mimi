# V13 dual-Claude safety-taxonomy calibration

Date: 2026-07-26

Status: **calibration complete; V12 rejection unchanged**

## Outcome

V13 independently assessed the 27 V12 cases that introduced at least one
registered deterministic failure. Every case contained four anonymous
candidates: the licensed reference, safe parent, V12 step 50, and V12 step
100. Exact `claude-sonnet-5` and `claude-opus-5` judged the same blinded
payloads. Each of seven shards per model contains canonical-model usage
evidence; fallback models were forbidden. Candidate origin was hidden and no
reasoning trace was requested or stored.

The main calibration result is that the deterministic negation detector was
wrong about the *kind* of error in every newly flagged case. Neither judge
applied a polarity tag to any of the five registered negation cases at either
checkpoint. Four of those five cases still contain other critical errors,
including wrong legal citations, amounts, and ministry names. The result
therefore improves the taxonomy without making unsafe translations pass.

| Registered V12 event | Step 50 supported by either / both judges | Step 100 supported by either / both judges |
| --- | ---: | ---: |
| exact critical structure | 8 / 3 of 13 | 7 / 3 of 15 |
| typed critical structure | 6 / 5 of 7 | 7 / 5 of 8 |
| negation / polarity | **0 / 0 of 5** | **0 / 0 of 5** |
| repetition / generation | **1 / 1 of 1** | **1 / 1 of 1** |

The loop is unambiguous: both judges tag the repeating `Article (25)` output
as repetition/nontermination. The strongest deterministic signal is typed
structure, where 13 of 15 checkpoint-events receive related semantic support
from at least one judge. Exact surface mismatch is much noisier because it
mixes real changes with equivalent `(ii)`/`(2)`-style representations.

## Judge agreement and limits

Across all 108 candidate assessments, the judges agree on the critical-error
boolean for 93 candidates (86.1%) and on the representation-only boolean for
86 (79.6%). A fail-closed view produces these counts:

| Anonymous candidate role | Candidates | Critical by either judge | Clean by both |
| --- | ---: | ---: | ---: |
| licensed reference | 27 | 2 | 25 |
| safe parent | 27 | 20 | 5 |
| V12 step 50 | 27 | 19 | 6 |
| V12 step 100 | 27 | 20 | 6 |

These rows are **not** a comparative translation-quality benchmark. The sample
was deliberately selected from known V12 detector disagreements, so it is
heavily enriched for hard and broken translations. It cannot estimate
whole-suite win rate or replace chrF++, BLEU, COMET, latency, memory, and
bundle-size evaluation.

The licensed reference itself is judged critical twice. Both judges agree that
one reference renders `法人` as “legal specialist” and omits a following-item
clause; only Sonnet marks another water-buffalo terminology issue critical.
This is useful evidence that licensed references and LLM judges are fallible.
Future gates must retain independent references, deterministic structure
checks, dual-judge disagreement handling, and a protected held-out suite.

## Evaluator change for the next experiment

V13 does not alter V12's frozen decision. It calibrates the future evaluator:

- tokenize legal `No.` as the abbreviation for “number,” never as a polarity
  cue;
- accept mathematically equivalent upper-bound language such as “not more
  than” when the boundary and amount are preserved;
- distinguish equivalent numbering surfaces from changed article, item, or
  paragraph identity;
- keep repetition/nontermination as a deterministic hard failure;
- send newly introduced exact, typed, polarity, and omission disagreements
  through the same dual-model semantic audit, failing closed when the judges
  disagree;
- report detector category and semantic error category separately.

The next training arm should use model-generated rollout prefixes and explicit
EOS/repetition recovery. It should also create legal-number and citation
counterfactuals conditioned on the model's own bad states. V12's 97.7%
teacher-forced chosen-token preference shows that another gold-prefix-only
ranking run is unlikely to fix exposure-driven loops.

## Reproducible evidence

- Frozen contract:
  `canonical-safety-taxonomy-v13-contract-2026-07-26.json`
  (`993ee9d767f8f6b6f48a3f86197f10f81a6d6d08061b672f306de512ddaea2e1`)
- Collected result:
  `canonical-safety-taxonomy-v13-result-2026-07-26.json`
  (`d9b92810760ed5f0973a00927e0680b2604feca59b8e9c0ed44c03cc17fb05c5`)
- Exact judged sources per model: 27
- Exact verified shards per model: 7
- Sonnet run cost recorded by the CLI: USD 0.6638272
- Opus run cost recorded by the CLI: USD 0.616579

The work-directory request, response, shard, and mapping files remain local
research evidence and are intentionally ignored. The tracked result binds
their hashes and exact-model manifests. No q4 conversion, protected
evaluation, COMET run, training, bundle replacement, app change, release, or
public upload is authorized.
