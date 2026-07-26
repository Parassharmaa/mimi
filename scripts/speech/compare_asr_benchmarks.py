#!/usr/bin/env python3
"""Compare two Mimi ASR benchmark reports on the exact same suite."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path


def normalized_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        if unicodedata.category(character)[0] in {"L", "N"}
        else " "
        for character in normalized
    )


def is_ascii_alphanumeric(character: str) -> bool:
    return character.isascii() and character.isalnum()


def contains_term_surface(hypothesis: str, candidate: str) -> bool:
    normalized_hypothesis = normalized_match_text(hypothesis)
    candidate_characters = [
        character
        for character in normalized_match_text(candidate)
        if unicodedata.category(character)[0] in {"L", "N"}
    ]
    if not candidate_characters:
        return False

    pattern = r"\s*".join(re.escape(character) for character in candidate_characters)
    if is_ascii_alphanumeric(candidate_characters[0]):
        pattern = rf"(?<![0-9a-z]){pattern}"
    if is_ascii_alphanumeric(candidate_characters[-1]):
        pattern = rf"{pattern}(?![0-9a-z])"
    return re.search(pattern, normalized_hypothesis) is not None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def protected_term_counts(
    suite_row: dict,
    hypothesis: str,
    *,
    include_aliases: bool,
) -> tuple[int, int]:
    kept = 0
    terms = suite_row.get("protectedTerms", [])
    aliases = suite_row.get("protectedTermAliases", {})
    for term in terms:
        surfaces = [term]
        if include_aliases:
            surfaces.extend(aliases.get(term, []))
        kept += any(contains_term_surface(hypothesis, surface) for surface in surfaces)
    return kept, len(terms)


def report_error_rate(report: dict) -> float:
    return report.get(
        "corpusErrorRate",
        report.get(
            "corpusCharacterErrorRate",
            report.get("corpusWordErrorRate"),
        ),
    )


def row_edits(row: dict) -> int:
    return row.get(
        "editDistance",
        row.get("characterEdits", row.get("wordEdits")),
    )


def row_reference_units(row: dict) -> int:
    return row.get(
        "referenceUnits",
        row.get("referenceCharacters", row.get("referenceWords")),
    )


def row_error_rate(row: dict) -> float:
    return row.get(
        "errorRate",
        row.get("characterErrorRate", row.get("wordErrorRate")),
    )


def bootstrap_pairwise(
    case_ids: list[str],
    suite: dict[str, dict],
    results: list[dict[str, dict]],
    *,
    iterations: int = 10_000,
    seed: int = 20_260_726,
) -> dict:
    randomizer = random.Random(seed)
    cer_differences: list[float] = []
    term_differences: list[float] = []
    case_values: dict[str, dict] = {}
    for case_id in case_ids:
        term_counts = [
            protected_term_counts(
                suite[case_id],
                result[case_id]["hypothesis"],
                include_aliases=True,
            )
            for result in results
        ]
        case_values[case_id] = {
            "edits": [row_edits(result[case_id]) for result in results],
            "referenceUnits": row_reference_units(results[0][case_id]),
            "protectedKept": [count[0] for count in term_counts],
            "protectedTotal": term_counts[0][1],
        }

    for _ in range(iterations):
        sample = randomizer.choices(case_ids, k=len(case_ids))
        reference_units = sum(
            case_values[case_id]["referenceUnits"] for case_id in sample
        )
        left_edits = sum(case_values[case_id]["edits"][0] for case_id in sample)
        right_edits = sum(case_values[case_id]["edits"][1] for case_id in sample)
        cer_differences.append((right_edits - left_edits) / max(1, reference_units))
        protected_total = sum(
            case_values[case_id]["protectedTotal"] for case_id in sample
        )
        left_kept = sum(case_values[case_id]["protectedKept"][0] for case_id in sample)
        right_kept = sum(case_values[case_id]["protectedKept"][1] for case_id in sample)
        term_differences.append((right_kept - left_kept) / max(1, protected_total))

    return {
        "iterations": iterations,
        "seed": seed,
        "errorRateDifferenceRightMinusLeft95CI": [
            percentile(cer_differences, 0.025),
            percentile(cer_differences, 0.975),
        ],
        "aliasAwareProtectedTermRecallDifferenceRightMinusLeft95CI": [
            percentile(term_differences, 0.025),
            percentile(term_differences, 0.975),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    suite = {
        row["caseID"]: row
        for row in (
            json.loads(line)
            for line in args.suite.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    reports = [
        json.loads(args.left.read_text(encoding="utf-8")),
        json.loads(args.right.read_text(encoding="utf-8")),
    ]
    metrics = [report.get("metric", "cer") for report in reports]
    if metrics[0] != metrics[1]:
        raise SystemExit("benchmark reports use different error metrics")
    results = [{row["caseID"]: row for row in report["results"]} for report in reports]
    expected_ids = set(suite)
    if any(set(result) != expected_ids for result in results):
        raise SystemExit("benchmark reports do not cover the exact suite")

    engines: list[dict] = []
    for report, result in zip(reports, results, strict=True):
        protected_exact_total = 0
        protected_exact_kept = 0
        protected_alias_kept = 0
        for case_id, suite_row in suite.items():
            exact_kept, term_total = protected_term_counts(
                suite_row,
                result[case_id]["hypothesis"],
                include_aliases=False,
            )
            alias_kept, _ = protected_term_counts(
                suite_row,
                result[case_id]["hypothesis"],
                include_aliases=True,
            )
            protected_exact_total += term_total
            protected_exact_kept += exact_kept
            protected_alias_kept += alias_kept
        engines.append(
            {
                "engine": report["engine"],
                "metric": report.get("metric", "cer"),
                "corpusErrorRate": report_error_rate(report),
                "meanRealTimeFactor": report["meanRealTimeFactor"],
                "protectedTermsTotal": protected_exact_total,
                "exactProtectedTermsKept": protected_exact_kept,
                "exactProtectedTermRecall": protected_exact_kept
                / max(1, protected_exact_total),
                "aliasAwareProtectedTermsKept": protected_alias_kept,
                "aliasAwareProtectedTermRecall": protected_alias_kept
                / max(1, protected_exact_total),
            }
        )

    left_wins = 0
    right_wins = 0
    ties = 0
    for case_id in sorted(expected_ids):
        left_error = row_error_rate(results[0][case_id])
        right_error = row_error_rate(results[1][case_id])
        if left_error < right_error:
            left_wins += 1
        elif right_error < left_error:
            right_wins += 1
        else:
            ties += 1

    comparison = {
        "format": "mimi-asr-comparison-v2",
        "metric": metrics[0],
        "caseCount": len(suite),
        "engines": engines,
        "pairwise": {
            "leftEngine": reports[0]["engine"],
            "leftWins": left_wins,
            "rightEngine": reports[1]["engine"],
            "rightWins": right_wins,
            "ties": ties,
            "corpusErrorRateDifferenceRightMinusLeft": engines[1]["corpusErrorRate"]
            - engines[0]["corpusErrorRate"],
            "aliasAwareProtectedTermRecallDifferenceRightMinusLeft": engines[1][
                "aliasAwareProtectedTermRecall"
            ]
            - engines[0]["aliasAwareProtectedTermRecall"],
            "pairedClipBootstrap": bootstrap_pairwise(
                sorted(expected_ids),
                suite,
                results,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
