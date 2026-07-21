#!/usr/bin/env python3
"""Evaluate a source-only formal-text gate for a Marian expert checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sacrebleu
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from source_expert_router import SourceExpertRouter


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def load_suite(path: Path, direction: tuple[str, str]) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        row
        for row in rows
        if (row.get("sourceLanguage"), row.get("targetLanguage")) == direction
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise SystemExit("directional suite must be non-empty with unique IDs")
    return rows


def report_rows(path: Path) -> dict[str, dict]:
    payload = load_json(path)
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"translation report has no results: {path}")
    return {str(row["caseID"]): row for row in rows}


def group_id(identifier: str) -> str:
    if ":jlt:" in identifier:
        return identifier.split(":tu-", 1)[0]
    if ":alt:SNT." in identifier:
        prefix, remainder = identifier.split(":alt:SNT.", 1)
        return f"{prefix}:alt:{remainder.split('.', 1)[0]}"
    return identifier


def split_name(identifier: str, direction: str) -> str:
    digest = hashlib.sha256(
        f"mimi-expert-router-v1:{direction}:{group_id(identifier)}".encode()
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 50:
        return "train"
    if bucket < 75:
        return "tune"
    return "test"


def feature_text(source: str) -> str:
    length_bin = min(len(source) // 20, 20)
    return f"{source}\n__MIMI_LENGTH_BIN_{length_bin}__"


def sentence_chrf(hypothesis: str, references: list[str]) -> float:
    return sacrebleu.sentence_chrf(
        hypothesis,
        references,
        word_order=2,
        eps_smoothing=True,
    ).score


def align(
    suite: list[dict],
    baseline: dict[str, dict],
    expert: dict[str, dict],
    direction_name: str,
) -> list[dict]:
    aligned: list[dict] = []
    for row in suite:
        identifier = str(row["id"])
        if identifier not in baseline or identifier not in expert:
            raise SystemExit(f"reports are missing aligned case: {identifier}")
        base = baseline[identifier]
        candidate = expert[identifier]
        for report_row in (base, candidate):
            if (
                report_row.get("source") != row.get("source")
                or report_row.get("references") != row.get("references")
            ):
                raise SystemExit(f"report content differs for {identifier}")
        base_score = sentence_chrf(base["hypothesis"], row["references"])
        expert_score = sentence_chrf(candidate["hypothesis"], row["references"])
        aligned.append(
            {
                "id": identifier,
                "domain": row["domain"],
                "source": row["source"],
                "references": row["references"],
                "baselineHypothesis": base["hypothesis"],
                "expertHypothesis": candidate["hypothesis"],
                "baselineSentenceChrFPlusPlus": base_score,
                "expertSentenceChrFPlusPlus": expert_score,
                "expertDelta": expert_score - base_score,
                "split": split_name(identifier, direction_name),
            }
        )
    return aligned


def routed_summary(
    rows: list[dict],
    predictions: np.ndarray,
    threshold: float,
    minimum_source_characters: int,
) -> dict:
    route = (predictions >= threshold) & np.array(
        [len(row["source"]) >= minimum_source_characters for row in rows]
    )
    deltas = np.array([row["expertDelta"] for row in rows])
    routed_deltas = np.where(route, deltas, 0.0)
    hypotheses = [
        row["expertHypothesis"] if selected else row["baselineHypothesis"]
        for row, selected in zip(rows, route, strict=True)
    ]
    references = [[row["references"][0] for row in rows]]
    baseline_hypotheses = [row["baselineHypothesis"] for row in rows]
    return {
        "cases": len(rows),
        "routedCases": int(route.sum()),
        "routeRate": float(route.mean()),
        "meanSentenceChrFPlusPlusDelta": float(routed_deltas.mean()),
        "positiveRoutedCases": int(np.sum(route & (deltas > 0))),
        "negativeRoutedCases": int(np.sum(route & (deltas < 0))),
        "equalRoutedCases": int(np.sum(route & (deltas == 0))),
        "baselineCorpusChrFPlusPlus": sacrebleu.corpus_chrf(
            baseline_hypotheses,
            references,
            word_order=2,
        ).score,
        "routedCorpusChrFPlusPlus": sacrebleu.corpus_chrf(
            hypotheses,
            references,
            word_order=2,
        ).score,
        "routedIDs": [row["id"] for row, selected in zip(rows, route, strict=True) if selected],
    }


def bootstrap_interval(
    rows: list[dict],
    predictions: np.ndarray,
    threshold: float,
    minimum_source_characters: int,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    route = (predictions >= threshold) & np.array(
        [len(row["source"]) >= minimum_source_characters for row in rows]
    )
    deltas = np.where(route, [row["expertDelta"] for row in rows], 0.0)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(rows), size=(samples, len(rows)))
    means = deltas[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("expert_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--direction", choices=("en-ja", "ja-en"), required=True)
    parser.add_argument(
        "--training-target",
        choices=("expert-delta", "legal-domain"),
        default="expert-delta",
    )
    parser.add_argument("--minimum-domain-precision", type=float, default=0.0)
    parser.add_argument("--canary", type=Path)
    parser.add_argument("--canary-baseline-report", type=Path)
    parser.add_argument("--canary-expert-report", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    canary_values = (
        args.canary,
        args.canary_baseline_report,
        args.canary_expert_report,
    )
    if any(canary_values) and not all(canary_values):
        raise SystemExit("canary suite and both canary reports must be supplied together")
    if args.bootstrap_samples < 1:
        raise SystemExit("bootstrap sample count must be positive")
    if not 0.0 <= args.minimum_domain_precision <= 1.0:
        raise SystemExit("minimum domain precision must be between zero and one")
    if args.minimum_domain_precision and args.training_target != "legal-domain":
        raise SystemExit("minimum domain precision requires legal-domain training")

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
    if any(len(split) < 50 for split in splits.values()):
        raise SystemExit("router split is unexpectedly small")

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
    train_targets = np.array(
        [
            (
                row["expertDelta"]
                if args.training_target == "expert-delta"
                else float(row["domain"] == "ministry-published-legal")
            )
            for row in splits["train"]
        ]
    )
    tune_matrix = vectorizer.transform(
        [feature_text(row["source"]) for row in splits["tune"]]
    )
    canary_rows = None
    canary_matrix = None
    if all(canary_values):
        canary_rows = align(
            load_suite(args.canary, language_direction),
            report_rows(args.canary_baseline_report),
            report_rows(args.canary_expert_report),
            f"{args.direction}-canary",
        )
        canary_matrix = vectorizer.transform(
            [feature_text(row["source"]) for row in canary_rows]
        )

    best: tuple[float, float, float, float, float, Ridge, np.ndarray] | None = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        regressor = Ridge(alpha=alpha, solver="lsqr")
        regressor.fit(train_matrix, train_targets)
        predictions = regressor.predict(tune_matrix)
        canary_predictions_for_model = (
            regressor.predict(canary_matrix) if canary_matrix is not None else None
        )
        thresholds = np.unique(
            np.concatenate(
                [
                    np.quantile(predictions, np.linspace(0.0, 1.0, 101)),
                    np.array(
                        [
                            0.0,
                            float(predictions.max())
                            + max(1e-6, abs(float(predictions.max())) * 1e-6),
                        ]
                    ),
                ]
            )
        )
        for minimum_source_characters in (0, 40, 60, 80, 100, 120, 160, 200):
            tune_lengths = np.array(
                [
                    len(row["source"]) >= minimum_source_characters
                    for row in splits["tune"]
                ]
            )
            for threshold in thresholds:
                if canary_predictions_for_model is not None:
                    canary_route = (
                        canary_predictions_for_model >= threshold
                    ) & np.array(
                        [
                            len(row["source"]) >= minimum_source_characters
                            for row in canary_rows
                        ]
                    )
                    canary_gain = float(
                        np.where(
                            canary_route,
                            [row["expertDelta"] for row in canary_rows],
                            0.0,
                        ).mean()
                    )
                    if canary_gain < 0:
                        continue
                routed = (predictions >= threshold) & tune_lengths
                if args.minimum_domain_precision and bool(routed.any()):
                    legal_precision = float(
                        np.mean(
                            [
                                row["domain"] == "ministry-published-legal"
                                for row, selected in zip(
                                    splits["tune"], routed, strict=True
                                )
                                if selected
                            ]
                        )
                    )
                    if legal_precision < args.minimum_domain_precision:
                        continue
                gain = float(
                    np.where(
                        routed,
                        [row["expertDelta"] for row in splits["tune"]],
                        0.0,
                    ).mean()
                )
                candidate = (
                    gain,
                    -float(routed.sum()),
                    float(minimum_source_characters),
                    float(threshold),
                    -alpha,
                    regressor,
                    predictions,
                )
                if best is None or candidate[:5] > best[:5]:
                    best = candidate
    assert best is not None
    (
        tune_gain,
        _,
        minimum_source_characters,
        threshold,
        negative_alpha,
        regressor,
        tune_predictions,
    ) = best
    minimum_source_characters = int(minimum_source_characters)
    alpha = -negative_alpha
    test_matrix = vectorizer.transform(
        [feature_text(row["source"]) for row in splits["test"]]
    )
    test_predictions = regressor.predict(test_matrix)
    tune_summary = routed_summary(
        splits["tune"],
        tune_predictions,
        threshold,
        minimum_source_characters,
    )
    test_summary = routed_summary(
        splits["test"],
        test_predictions,
        threshold,
        minimum_source_characters,
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
            index
            for index, row in enumerate(splits["test"])
            if row["domain"] == domain
        ]
        domain_rows = [splits["test"][index] for index in indices]
        domain_predictions = test_predictions[indices]
        test_summary["domains"][domain] = routed_summary(
            domain_rows,
            domain_predictions,
            threshold,
            minimum_source_characters,
        )

    canary_summary = None
    if canary_rows is not None:
        canary_predictions = regressor.predict(
            vectorizer.transform([feature_text(row["source"]) for row in canary_rows])
        )
        canary_summary = routed_summary(
            canary_rows,
            canary_predictions,
            threshold,
            minimum_source_characters,
        )

    router_model = {
        "schemaVersion": 1,
        "format": "mimi-source-expert-router-v1",
        "direction": args.direction,
        "trainingTarget": args.training_target,
        "minimumDomainPrecision": args.minimum_domain_precision,
        "features": {
            "analyzer": "unicode-codepoint-character",
            "ngramRange": [2, 5],
            "lowercase": True,
            "minimumDocumentFrequency": 2,
            "sublinearTermFrequency": True,
            "inverseDocumentFrequency": "smooth-idf",
            "normalization": "l2",
            "sourceLengthBin": "append newline then __MIMI_LENGTH_BIN_{min(chars//20,20)}__",
        },
        "vocabulary": {
            ngram: int(index) for ngram, index in vectorizer.vocabulary_.items()
        },
        "inverseDocumentFrequency": [float(value) for value in vectorizer.idf_],
        "ridge": {
            "alpha": alpha,
            "coefficients": [float(value) for value in regressor.coef_],
            "intercept": float(regressor.intercept_),
        },
        "routing": {
            "minimumSourceCharacters": minimum_source_characters,
            "scoreThreshold": threshold,
        },
    }
    portable_router = SourceExpertRouter(router_model)
    portable_score_deltas = []
    for validation_rows, sklearn_predictions in (
        (splits["tune"], tune_predictions),
        (splits["test"], test_predictions),
    ):
        portable_predictions = np.array(
            [portable_router.score(row["source"]) for row in validation_rows]
        )
        portable_score_deltas.extend(
            np.abs(portable_predictions - sklearn_predictions).tolist()
        )
        sklearn_routes = (sklearn_predictions >= threshold) & np.array(
            [
                len(row["source"]) >= minimum_source_characters
                for row in validation_rows
            ]
        )
        portable_routes = np.array(
            [portable_router.routes_to_expert(row["source"]) for row in validation_rows]
        )
        if not np.array_equal(portable_routes, sklearn_routes):
            raise SystemExit("portable router changes a tuned or held-out route")
    if canary_rows is not None:
        portable_canary_predictions = np.array(
            [portable_router.score(row["source"]) for row in canary_rows]
        )
        portable_score_deltas.extend(
            np.abs(portable_canary_predictions - canary_predictions).tolist()
        )
        portable_canary_routes = np.array(
            [portable_router.routes_to_expert(row["source"]) for row in canary_rows]
        )
        sklearn_canary_routes = (
            canary_predictions >= threshold
        ) & np.array(
            [len(row["source"]) >= minimum_source_characters for row in canary_rows]
        )
        if not np.array_equal(portable_canary_routes, sklearn_canary_routes):
            raise SystemExit("portable router changes a canary route")
    maximum_portable_score_delta = max(portable_score_deltas, default=0.0)
    if maximum_portable_score_delta > 1e-5:
        raise SystemExit(
            "portable router score mismatch: "
            f"maximum absolute delta {maximum_portable_score_delta}"
        )
    model_bytes = (
        json.dumps(
            router_model,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()
    if args.model_output is not None:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        args.model_output.write_bytes(model_bytes)
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "development-only source-only routed Marian expert ablation",
        "promotionEligible": False,
        "direction": args.direction,
        "inputs": {
            "suite": {"path": str(args.suite.resolve()), "sha256": sha256(args.suite)},
            "baselineReport": {
                "path": str(args.baseline_report.resolve()),
                "sha256": sha256(args.baseline_report),
            },
            "expertReport": {
                "path": str(args.expert_report.resolve()),
                "sha256": sha256(args.expert_report),
            },
        },
        "splitContract": {
            "grouped": ["JLT law", "ALT document"],
            "buckets": {"train": "0-49", "tune": "50-74", "test": "75-99"},
            "counts": {name: len(split) for name, split in splits.items()},
            "domains": {
                name: dict(Counter(row["domain"] for row in split))
                for name, split in splits.items()
            },
        },
        "router": {
            "features": "source-only TF-IDF character 2-5 grams plus source-length bin",
            "trainingTarget": args.training_target,
            "minimumDomainPrecision": args.minimum_domain_precision,
            "maximumFeatures": 16_384,
            "ridgeAlpha": alpha,
            "threshold": threshold,
            "minimumSourceCharacters": minimum_source_characters,
            "portableModel": {
                "path": str(args.model_output.resolve()) if args.model_output else None,
                "bytes": len(model_bytes),
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
            },
            "sklearnVersion": sklearn.__version__,
            "privateReasoningTracesUsed": False,
            "portableRuntimeValidation": {
                "exactRouteParity": True,
                "maximumAbsoluteScoreDelta": maximum_portable_score_delta,
                "tolerance": 1e-5,
            },
        },
        "tuning": {"selectedMeanGain": tune_gain, **tune_summary},
        "tuningConstraint": "non-negative mean sentence chrF++ delta on product canary",
        "test": test_summary,
        "canary": canary_summary,
        "decision": {
            "passesHeldoutRouterAblation": (
                test_summary["pairedBootstrap95"][0] > 0
                and (
                    canary_summary is None
                    or canary_summary["meanSentenceChrFPlusPlusDelta"] >= 0
                )
            ),
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
            {"tuning": tune_summary, "test": test_summary, "canary": canary_summary},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
