#!/usr/bin/env python3
"""Evaluate a canary-constrained weighted source-only expert classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from evaluate_expert_router import (
    align,
    bootstrap_interval,
    feature_text,
    load_suite,
    report_rows,
    routed_summary,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_interval(values: np.ndarray, samples: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    return [float(value) for value in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("expert_report", type=Path)
    parser.add_argument("canary", type=Path)
    parser.add_argument("canary_baseline_report", type=Path)
    parser.add_argument("canary_expert_report", type=Path)
    parser.add_argument("current_router_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=("en-ja", "ja-en"), required=True)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.c <= 0 or args.bootstrap_samples < 1:
        raise SystemExit("C and bootstrap sample count must be positive")

    language_direction = {
        "en-ja": ("en-US", "ja-JP"),
        "ja-en": ("ja-JP", "en-US"),
    }[args.direction]
    rows = align(
        load_suite(args.suite, language_direction),
        report_rows(args.baseline_report),
        report_rows(args.expert_report),
        args.direction,
    )
    splits = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "tune", "test")
    }
    if any(len(values) < 50 for values in splits.values()):
        raise SystemExit("router split is unexpectedly small")
    canary_rows = align(
        load_suite(args.canary, language_direction),
        report_rows(args.canary_baseline_report),
        report_rows(args.canary_expert_report),
        f"{args.direction}-canary",
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=16_384,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(
        [feature_text(row["source"]) for row in splits["train"]]
    )
    labels = np.array([int(row["expertDelta"] > 0) for row in splits["train"]])
    if len(set(labels.tolist())) != 2:
        raise SystemExit("expert-delta classifier requires both target classes")
    absolute_deltas = np.array(
        [abs(float(row["expertDelta"])) for row in splits["train"]]
    )
    sample_weights = np.sqrt(absolute_deltas + 0.1)
    sample_weights /= sample_weights.mean()
    classifier = LinearSVC(C=args.c, max_iter=100_000, random_state=0)
    classifier.fit(train_matrix, labels, sample_weight=sample_weights)

    def predict(values: list[dict]) -> np.ndarray:
        return classifier.decision_function(
            vectorizer.transform([feature_text(row["source"]) for row in values])
        )

    tune_predictions = predict(splits["tune"])
    canary_predictions = predict(canary_rows)
    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(tune_predictions, np.linspace(0.0, 1.0, 101)),
                [float(tune_predictions.max()) + 1e-6],
            ]
        )
    )
    best: tuple[float, float, int, float] | None = None
    for minimum_source_characters in (0, 40, 60, 80, 100, 120, 160, 200):
        tune_lengths = np.array(
            [len(row["source"]) >= minimum_source_characters for row in splits["tune"]]
        )
        canary_lengths = np.array(
            [len(row["source"]) >= minimum_source_characters for row in canary_rows]
        )
        for threshold in thresholds:
            canary_route = (canary_predictions >= threshold) & canary_lengths
            canary_gain = float(
                np.where(
                    canary_route,
                    [row["expertDelta"] for row in canary_rows],
                    0.0,
                ).mean()
            )
            if canary_gain < 0:
                continue
            tune_route = (tune_predictions >= threshold) & tune_lengths
            tune_gain = float(
                np.where(
                    tune_route,
                    [row["expertDelta"] for row in splits["tune"]],
                    0.0,
                ).mean()
            )
            candidate = (
                tune_gain,
                -float(tune_route.sum()),
                minimum_source_characters,
                float(threshold),
            )
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise SystemExit("no canary-preserving classifier threshold exists")
    _, _, minimum_source_characters, threshold = best

    test_predictions = predict(splits["test"])
    test_summary = routed_summary(
        splits["test"], test_predictions, threshold, minimum_source_characters
    )
    test_summary["pairedBootstrap95"] = bootstrap_interval(
        splits["test"],
        test_predictions,
        threshold,
        minimum_source_characters,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    test_summary["domains"] = {}
    for domain in sorted({row["domain"] for row in splits["test"]}):
        indices = [
            index for index, row in enumerate(splits["test"]) if row["domain"] == domain
        ]
        test_summary["domains"][domain] = routed_summary(
            [splits["test"][index] for index in indices],
            test_predictions[indices],
            threshold,
            minimum_source_characters,
        )
    tune_summary = routed_summary(
        splits["tune"], tune_predictions, threshold, minimum_source_characters
    )
    canary_summary = routed_summary(
        canary_rows, canary_predictions, threshold, minimum_source_characters
    )

    current = json.loads(args.current_router_report.read_text(encoding="utf-8"))
    current_test = current.get("test", {})
    current_ids = set(current_test.get("routedIDs", []))
    if int(current_test.get("cases", -1)) != len(splits["test"]):
        raise SystemExit("current router report uses a different held-out population")
    new_route = (test_predictions >= threshold) & np.array(
        [len(row["source"]) >= minimum_source_characters for row in splits["test"]]
    )
    versus_current = np.array(
        [
            row["expertDelta"]
            * (int(selected) - int(row["id"] in current_ids))
            for row, selected in zip(splits["test"], new_route, strict=True)
        ]
    )
    current_comparison = {
        "meanSentenceChrFPlusPlusDelta": float(versus_current.mean()),
        "pairedBootstrap95": paired_interval(
            versus_current, args.bootstrap_samples, args.seed
        ),
        "addedRoutes": int(
            sum(
                selected and row["id"] not in current_ids
                for row, selected in zip(splits["test"], new_route, strict=True)
            )
        ),
        "removedRoutes": int(
            sum(
                not selected and row["id"] in current_ids
                for row, selected in zip(splits["test"], new_route, strict=True)
            )
        ),
    }

    coefficient_hash = hashlib.sha256(
        classifier.coef_.astype(np.float64).tobytes()
        + classifier.intercept_.astype(np.float64).tobytes()
    ).hexdigest()
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "development-only canary-constrained weighted source router ablation",
        "claimEligible": False,
        "direction": args.direction,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in (
                ("suite", args.suite),
                ("baselineReport", args.baseline_report),
                ("expertReport", args.expert_report),
                ("canary", args.canary),
                ("canaryBaselineReport", args.canary_baseline_report),
                ("canaryExpertReport", args.canary_expert_report),
                ("currentRouterReport", args.current_router_report),
            )
        },
        "splitCounts": {name: len(values) for name, values in splits.items()},
        "classifier": {
            "family": "LinearSVC",
            "C": args.c,
            "randomState": 0,
            "target": "expert sentence chrF++ delta greater than zero",
            "sampleWeight": "sqrt(abs(expert delta) + 0.1), normalized to mean 1",
            "features": "source-only TF-IDF character 2-5 grams plus source-length bin",
            "vocabularySize": len(vectorizer.vocabulary_),
            "coefficientSHA256": coefficient_hash,
            "sklearnVersion": sklearn.__version__,
        },
        "routing": {
            "minimumSourceCharacters": minimum_source_characters,
            "scoreThreshold": threshold,
            "selectionRule": "maximize tune mean gain, then fewer routes, subject to non-negative canary gain",
        },
        "tuning": tune_summary,
        "canary": canary_summary,
        "test": test_summary,
        "versusCurrentRouter": current_comparison,
        "decision": {
            "passesCurrentRouterGate": current_comparison["pairedBootstrap95"][0] > 0,
            "doesNotAuthorizeAppIntegration": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tuning": tune_summary,
                "canary": canary_summary,
                "test": test_summary,
                "versusCurrentRouter": current_comparison,
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
