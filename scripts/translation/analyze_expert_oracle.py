#!/usr/bin/env python3
"""Measure a reference-aware expert oracle as a non-deployable upper bound."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"translation report has no results: {path}")
    keyed = {str(row["caseID"]): row for row in rows}
    if len(keyed) != len(rows):
        raise SystemExit(f"translation report has duplicate case IDs: {path}")
    return keyed


def group_id(identifier: str) -> str:
    if identifier.startswith("development-accuracy-v1:document:"):
        return identifier.rsplit(":segment-", 1)[0]
    return identifier


def sentence_chrf(hypothesis: str, references: list[str]) -> float:
    return sacrebleu.sentence_chrf(
        hypothesis,
        references,
        word_order=2,
        eps_smoothing=True,
    ).score


def parse_expert(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expert must use LABEL=REPORT.json")
    label, raw_path = value.split("=", 1)
    if not label or label == "baseline" or not raw_path:
        raise argparse.ArgumentTypeError(
            "expert label must be non-empty and cannot be 'baseline'"
        )
    return label, Path(raw_path)


def bootstrap_grouped_mean(
    rows: list[dict],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[group_id(row["caseID"])].append(row["oracleDelta"])
    group_values = list(grouped.values())
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        sampled = [
            value
            for _ in group_values
            for value in generator.choice(group_values)
        ]
        means.append(sum(sampled) / len(sampled))
    means.sort()
    return [
        means[int((len(means) - 1) * 0.025)],
        means[int((len(means) - 1) * 0.975)],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), required=True)
    parser.add_argument(
        "--expert",
        action="append",
        type=parse_expert,
        required=True,
        metavar="LABEL=REPORT.json",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--sticky-documents",
        action="store_true",
        help="select one expert for every segment in a development document",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise SystemExit("bootstrap sample count must be positive")
    labels = [label for label, _ in args.expert]
    if len(set(labels)) != len(labels):
        raise SystemExit("expert labels must be unique")

    direction = DIRECTIONS[args.direction]
    baseline = load_report(args.baseline_report)
    experts = {label: load_report(path) for label, path in args.expert}
    rows: list[dict] = []
    selection_counts: Counter[str] = Counter()
    domain_deltas: dict[str, list[float]] = defaultdict(list)
    for identifier, baseline_row in baseline.items():
        if (
            baseline_row.get("sourceLanguage"),
            baseline_row.get("targetLanguage"),
        ) != direction:
            continue
        references = baseline_row.get("references")
        if not isinstance(references, list) or not references:
            raise SystemExit(f"baseline row has no references: {identifier}")
        options = [
            {
                "label": "baseline",
                "hypothesis": baseline_row["hypothesis"],
                "score": sentence_chrf(baseline_row["hypothesis"], references),
            }
        ]
        for label, expert_rows in experts.items():
            expert_row = expert_rows.get(identifier)
            if expert_row is None:
                raise SystemExit(f"expert {label} is missing {identifier}")
            for key in ("source", "references", "sourceLanguage", "targetLanguage"):
                if expert_row.get(key) != baseline_row.get(key):
                    raise SystemExit(
                        f"expert {label} differs from baseline for {identifier}: {key}"
                    )
            options.append(
                {
                    "label": label,
                    "hypothesis": expert_row["hypothesis"],
                    "score": sentence_chrf(expert_row["hypothesis"], references),
                }
            )
        rows.append(
            {
                "caseID": identifier,
                "domain": str(baseline_row["domain"]),
                "options": options,
                "references": references,
            }
        )
    if not rows:
        raise SystemExit("baseline report contains no rows for the requested direction")

    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[group_id(row["caseID"])].append(row)
    for group in grouped_rows.values():
        if args.sticky_documents:
            labels = [option["label"] for option in group[0]["options"]]
            group_scores = {
                label: sum(
                    next(
                        option["score"]
                        for option in row["options"]
                        if option["label"] == label
                    )
                    for row in group
                )
                / len(group)
                for label in labels
            }
            selected_label = max(
                labels,
                key=lambda label: (
                    group_scores[label],
                    label == "baseline",
                    label,
                ),
            )
        for row in group:
            selected = (
                next(
                    option
                    for option in row["options"]
                    if option["label"] == selected_label
                )
                if args.sticky_documents
                else max(
                    row["options"],
                    key=lambda option: (
                        option["score"],
                        option["label"] == "baseline",
                        option["label"],
                    ),
                )
            )
            baseline = row["options"][0]
            delta = selected["score"] - baseline["score"]
            row.update(
                {
                    "baselineSentenceChrFPlusPlus": baseline["score"],
                    "oracleSentenceChrFPlusPlus": selected["score"],
                    "oracleDelta": delta,
                    "selectedExpert": selected["label"],
                    "baselineHypothesis": baseline["hypothesis"],
                    "oracleHypothesis": selected["hypothesis"],
                }
            )
            selection_counts[selected["label"]] += 1
            domain_deltas[row["domain"]].append(delta)

    baseline_hypotheses = [row["baselineHypothesis"] for row in rows]
    oracle_hypotheses = [row["oracleHypothesis"] for row in rows]
    references = [[row["references"][0] for row in rows]]
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "reference-aware expert complementarity upper bound",
        "promotionEligible": False,
        "deployableRouter": False,
        "direction": args.direction,
        "inputs": {
            "baseline": {
                "path": str(args.baseline_report.resolve()),
                "sha256": sha256(args.baseline_report),
            },
            "experts": {
                label: {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
                for label, path in args.expert
            },
        },
        "contract": {
            "selectionSignal": (
                "maximum mean segment chrF++ per document against held-out references"
                if args.sticky_documents
                else "maximum sentence chrF++ against held-out reference"
            ),
            "documentRoutingSticky": args.sticky_documents,
            "documentBootstrapGrouping": (
                "all segments sharing a development-accuracy-v1 document ID"
            ),
            "bootstrapSamples": args.bootstrap_samples,
            "seed": args.seed,
            "warning": (
                "This oracle reads references and is only an upper bound. "
                "It cannot be used for inference or promotion."
            ),
        },
        "summary": {
            "cases": len(rows),
            "selectionCounts": dict(sorted(selection_counts.items())),
            "baselineCorpusChrFPlusPlus": sacrebleu.corpus_chrf(
                baseline_hypotheses,
                references,
                word_order=2,
            ).score,
            "oracleCorpusChrFPlusPlus": sacrebleu.corpus_chrf(
                oracle_hypotheses,
                references,
                word_order=2,
            ).score,
            "meanSentenceChrFPlusPlusDelta": (
                sum(row["oracleDelta"] for row in rows) / len(rows)
            ),
            "groupedBootstrap95": bootstrap_grouped_mean(
                rows,
                samples=args.bootstrap_samples,
                seed=args.seed,
            ),
            "domains": {
                domain: {
                    "cases": len(deltas),
                    "meanSentenceChrFPlusPlusDelta": sum(deltas) / len(deltas),
                }
                for domain, deltas in sorted(domain_deltas.items())
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
