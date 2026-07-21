#!/usr/bin/env python3
"""Offline contract test for held-out review, contamination, and promotion gates."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "Research/translation/benchmark/manifest.json"


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def run(*arguments: str, stdout=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=True,
        stdout=stdout,
        text=True,
    )


def score(engine: str) -> dict:
    return {
        "adequacy": 4 if engine == "candidate" else 1,
        "fluency": 4 if engine == "candidate" else 1,
        "terminology": 2 if engine == "candidate" else 0,
        "criticalError": False,
        "errorTags": [],
        "notes": "offline contract fixture",
    }


def main() -> None:
    weights = {
        "meeting-and-live-speech": 120,
        "everyday-conversation": 80,
        "macos-and-technical-ui": 60,
        "numbers-dates-and-entities": 60,
        "politeness-ambiguity-and-omission": 60,
        "code-switching": 20,
    }
    draft: list[dict] = []
    for source_language, target_language, prefix in (
        ("en-US", "ja-JP", "en"),
        ("ja-JP", "en-US", "ja"),
    ):
        index = 0
        for domain, count in weights.items():
            for _ in range(count):
                index += 1
                source = (
                    f"Original English benchmark sentence number {index} for Mimi."
                    if prefix == "en"
                    else f"Mimi用の日本語ベンチマーク原文、第{index}番です。"
                )
                references = (
                    [f"Mimi用の翻訳文、第{index}番です。", f"これはMimiの訳文{index}です。"]
                    if prefix == "en"
                    else [f"Mimi benchmark translation number {index}.", f"This is Mimi translation {index}."]
                )
                draft.append(
                    {
                        "id": f"heldout-{prefix}-{index:03d}",
                        "documentID": f"commissioned-{prefix}-{index:03d}",
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language,
                        "domain": domain,
                        "source": source,
                        "references": references,
                        "sourceAuthorID": f"source-author-{prefix}",
                        "referenceAuthorIDs": [
                            f"reference-author-a-{prefix}",
                            f"reference-author-b-{prefix}",
                        ],
                        "split": "heldout-draft",
                        "license": "project-owned",
                        "provenance": "offline validator fixture",
                        "reviewStatus": "bootstrap-unreviewed",
                        "claimEligible": False,
                        "sourceGeneratedByAI": False,
                        "referenceGeneratedByAI": False,
                    }
                )

    with tempfile.TemporaryDirectory(prefix="mimi-benchmark-contract-") as temporary:
        work = Path(temporary)
        authoring_template = work / "authoring-template.jsonl"
        run(
            "scripts/translation/prepare_benchmark_authoring_template.py",
            MANIFEST,
            str(authoring_template),
            stdout=subprocess.DEVNULL,
        )
        template_rows = [
            json.loads(line) for line in authoring_template.read_text().splitlines()
        ]
        assert len(template_rows) == 800
        assert sum(row["sourceLanguage"] == "en-US" for row in template_rows) == 400
        assert all(row["claimEligible"] is False and not row["source"] for row in template_rows)
        draft_path = work / "draft.jsonl"
        reference_review = work / "reference-review"
        adjudication = work / "adjudication"
        suite, review_records, rejected = (
            work / "suite.jsonl",
            work / "review-records.jsonl",
            work / "rejected.jsonl",
        )
        validation, training = work / "validation.json", work / "train.jsonl"
        write_jsonl(draft_path, draft)

        run(
            "scripts/translation/prepare_benchmark_reference_review.py",
            str(draft_path), str(reference_review),
            "--reviewer", "reviewer-a", "--reviewer", "reviewer-b",
            stdout=subprocess.DEVNULL,
        )
        for reviewer in ("reviewer-a", "reviewer-b"):
            path = reference_review / f"{reviewer}.responses.jsonl"
            responses = [json.loads(line) for line in path.read_text().splitlines()]
            for response in responses:
                response["decision"] = "approve"
                response["attestations"] = {
                    "human": True,
                    "bilingualQualified": True,
                    "independent": True,
                    "noAIAssistance": True,
                }
            write_jsonl(path, responses)
        invalid_review_path = work / "invalid-review-a.jsonl"
        invalid_reviews = [
            json.loads(line)
            for line in (reference_review / "reviewer-a.responses.jsonl").read_text().splitlines()
        ]
        invalid_reviews[0]["attestations"]["noAIAssistance"] = False
        write_jsonl(invalid_review_path, invalid_reviews)
        invalid_adjudication = subprocess.run(
            [
                sys.executable,
                "scripts/translation/prepare_benchmark_adjudication.py",
                str(draft_path),
                str(invalid_review_path),
                str(reference_review / "reviewer-b.responses.jsonl"),
                str(work / "invalid-adjudication"),
                "--adjudicator",
                "reviewer-c",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert invalid_adjudication.returncode != 0
        assert "required human attestations" in invalid_adjudication.stderr
        run(
            "scripts/translation/prepare_benchmark_adjudication.py",
            str(draft_path),
            str(reference_review / "reviewer-a.responses.jsonl"),
            str(reference_review / "reviewer-b.responses.jsonl"),
            str(adjudication), "--adjudicator", "reviewer-c",
            stdout=subprocess.DEVNULL,
        )
        adjudication_path = adjudication / "reviewer-c.responses.jsonl"
        adjudications = [json.loads(line) for line in adjudication_path.read_text().splitlines()]
        for response in adjudications:
            response["decision"] = "approve"
            response["attestations"] = {
                "human": True,
                "bilingualQualified": True,
                "independent": True,
                "noAIAssistance": True,
            }
        write_jsonl(adjudication_path, adjudications)
        run(
            "scripts/translation/finalize_benchmark_suite.py",
            str(draft_path),
            str(reference_review / "reviewer-a.responses.jsonl"),
            str(reference_review / "reviewer-b.responses.jsonl"),
            str(adjudication_path), str(suite), str(review_records), str(rejected),
            stdout=subprocess.DEVNULL,
        )
        assert not rejected.read_text(encoding="utf-8")
        final_suite = [json.loads(line) for line in suite.read_text().splitlines()]
        assert len(final_suite) == 800
        assert all(row["claimEligible"] and row["reviewStatus"] == "adjudicated" for row in final_suite)

        write_jsonl(training, [{"source": "A completely unrelated training sentence."}])
        validation_command = [
            "scripts/translation/validate_benchmark_suite.py",
            str(suite), MANIFEST, str(review_records),
            "--training-jsonl", str(training), "--output", str(validation),
        ]
        run(*validation_command, stdout=subprocess.DEVNULL)
        truncated_suite = work / "truncated-suite.jsonl"
        truncated_reviews = work / "truncated-review-records.jsonl"
        omitted_id = final_suite[0]["id"]
        write_jsonl(truncated_suite, [row for row in final_suite if row["id"] != omitted_id])
        review_rows = [json.loads(line) for line in review_records.read_text().splitlines()]
        write_jsonl(
            truncated_reviews,
            [row for row in review_rows if row["caseID"] != omitted_id],
        )
        truncated = subprocess.run(
            [
                sys.executable,
                "scripts/translation/validate_benchmark_suite.py",
                str(truncated_suite),
                MANIFEST,
                str(truncated_reviews),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert truncated.returncode != 0
        assert "need exactly 400" in truncated.stderr
        write_jsonl(training, [{"source": final_suite[0]["source"]}])
        contaminated = subprocess.run(
            [sys.executable, *validation_command], cwd=ROOT, capture_output=True, text=True
        )
        assert contaminated.returncode != 0
        assert "exact-match contamination" in contaminated.stderr
        write_jsonl(training, [{
            "source": "A completely unrelated sentence from a protected document.",
            "documentID": final_suite[0]["documentID"],
        }])
        document_contaminated = subprocess.run(
            [sys.executable, *validation_command], cwd=ROOT, capture_output=True, text=True
        )
        assert document_contaminated.returncode != 0
        assert "document-level contamination" in document_contaminated.stderr
        write_jsonl(training, [{"source": "A completely unrelated training sentence."}])

        candidate_results, apple_results = [], []
        for row in final_suite:
            common = {
                "caseID": row["id"],
                "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"],
                "domain": row["domain"],
                "source": row["source"],
                "references": row["references"],
                "latencySeconds": 0.12,
                "warmLatencySeconds": [0.08, 0.09, 0.10],
                "claimEligible": True,
            }
            candidate_results.append({**common, "hypothesis": row["references"][0]})
            apple_results.append({
                **common,
                "hypothesis": row["source"],
                "latencySeconds": 1.2,
                "warmLatencySeconds": [0.9, 1.0, 1.1],
            })
        base_report = {
            "schemaVersion": 1,
            "createdAt": "2026-07-17T00:00:00Z",
            "operatingSystem": "macOS fixture",
            "hardware": "Apple fixture",
            "preparationSeconds": 0.1,
        }
        model_bundle = work / "model-bundle"
        (model_bundle / "en-ja").mkdir(parents=True)
        (model_bundle / "ja-en").mkdir()
        (model_bundle / "en-ja" / "model.safetensors").write_bytes(b"en-ja fixture")
        (model_bundle / "ja-en" / "model.safetensors").write_bytes(b"ja-en fixture")
        root_manifest = model_bundle / "manifest.json"
        root_manifest.write_text(json.dumps({
            "format": "mimi-mlx-marian-pair-v1",
            "interface": "bidirectional-en-ja",
            "files": {},
        }), encoding="utf-8")
        model_revision = f"pair-manifest-sha256:{hashlib.sha256(root_manifest.read_bytes()).hexdigest()}"
        model_bytes = sum(
            path.stat().st_size for path in model_bundle.rglob("*") if path.is_file()
        )
        candidate_report = work / "candidate.json"
        apple_report = work / "apple.json"
        candidate_report.write_text(json.dumps({
            **base_report,
            "engine": "mlx-fixture",
            "modelRevision": model_revision,
            "peakResidentBytes": 177_000_000,
            "modelBytes": model_bytes,
            "results": candidate_results,
        }), encoding="utf-8")
        apple_report.write_text(json.dumps({
            **base_report,
            "engine": "apple-translation-high-fidelity",
            "modelRevision": None,
            "peakResidentBytes": 105_000_000,
            "modelBytes": None,
            "results": apple_results,
        }), encoding="utf-8")
        metric_configuration = json.loads((ROOT / MANIFEST).read_text())["measurement"][
            "learnedMetric"
        ]
        metric_signature_value = {
            "metric": metric_configuration["name"],
            "modelRepository": metric_configuration["modelRepository"],
            "modelRevision": metric_configuration["modelRevision"],
            "modelLicense": metric_configuration["modelLicense"],
            "package": metric_configuration["package"],
            "packageVersion": metric_configuration["packageVersion"],
            "setuptoolsVersion": metric_configuration["setuptoolsVersion"],
            "precision": metric_configuration["precision"],
            "multipleReferenceAggregation": metric_configuration[
                "multipleReferenceAggregation"
            ],
        }
        metric_signature = hashlib.sha256(
            json.dumps(
                metric_signature_value,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        def learned_report(engine: str, engine_path: Path, score_value: float) -> dict:
            return {
                "schemaVersion": 1,
                **metric_signature_value,
                "signatureSHA256": metric_signature,
                "engine": engine,
                "suiteSHA256": hashlib.sha256(suite.read_bytes()).hexdigest(),
                "engineReportSHA256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
                "results": [
                    {
                        "caseID": row["id"],
                        "score": score_value,
                        "referenceScores": [score_value for _ in row["references"]],
                    }
                    for row in final_suite
                ],
            }

        candidate_learned = work / "candidate-comet.json"
        apple_learned = work / "apple-comet.json"
        candidate_learned.write_text(
            json.dumps(learned_report("mlx-fixture", candidate_report, 0.9)),
            encoding="utf-8",
        )
        apple_learned.write_text(
            json.dumps(
                learned_report("apple-translation-high-fidelity", apple_report, 0.1)
            ),
            encoding="utf-8",
        )

        comparison = work / "comparison"
        run(
            "scripts/translation/prepare_engine_comparison_packets.py",
            str(candidate_report), str(apple_report), str(comparison),
            "--reviewer", "human-a", "--reviewer", "human-b",
            stdout=subprocess.DEVNULL,
        )
        assignments = {
            (row["reviewerID"], row["caseID"]): row
            for row in [
                json.loads(line)
                for line in (comparison / "sealed-assignments.jsonl").read_text().splitlines()
            ]
        }
        for reviewer in ("human-a", "human-b"):
            path = comparison / f"{reviewer}.responses.jsonl"
            responses = [json.loads(line) for line in path.read_text().splitlines()]
            for response in responses:
                assignment = assignments[(reviewer, response["caseID"])]
                response["outputA"] = score(assignment["outputAEngine"])
                response["outputB"] = score(assignment["outputBEngine"])
            write_jsonl(path, responses)

        fallback = work / "fallback.json"
        fallback.write_text(json.dumps({
            "schemaVersion": 1,
            "status": "passed",
            "appleDefaultWhenExperimentalDisabled": True,
            "candidateFailureDoesNotUseApple": True,
            "candidateFailurePreservesLocalResults": True,
            "candidateFailureShowsRetryableError": True,
            "applePartialsWhenExperimentalDisabled": True,
            "experimentalPartialsDoNotUseApple": True,
            "invalidModelPackRejected": True,
        }), encoding="utf-8")
        parity = work / "parity.json"
        parity.write_text(json.dumps({
            "schemaVersion": 1,
            "status": "passed",
            "engine": "swift-mlx-marian-exact-output-parity",
            "modelRevision": model_revision,
            "suiteSHA256": hashlib.sha256(suite.read_bytes()).hexdigest(),
            "pythonReportSHA256": hashlib.sha256(candidate_report.read_bytes()).hexdigest(),
            "pairManifestSHA256": model_revision.removeprefix("pair-manifest-sha256:"),
            "cases": len(final_suite),
            "exactMatches": len(final_suite),
            "results": [
                {
                    "caseID": row["id"],
                    "direction": (
                        "en-ja" if row["sourceLanguage"] == "en-US" else "ja-en"
                    ),
                    "pythonHypothesis": row["references"][0],
                    "swiftHypothesis": row["references"][0],
                    "exactMatch": True,
                }
                for row in final_suite
            ],
        }), encoding="utf-8")
        metallib = work / "mlx.metallib"
        metallib.write_bytes(b"version-pinned mlx metallib fixture")
        archive = work / "Mimi-macOS.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr("Mimi.app/Contents/MacOS/Mimi", b"fixture executable")
            zipped.writestr("Mimi.app/Contents/Info.plist", b"fixture plist")
            zipped.write(metallib, "Mimi.app/Contents/MacOS/mlx.metallib")
            for path in sorted(model_bundle.rglob("*")):
                if path.is_file():
                    zipped.write(
                        path,
                        "Mimi.app/Contents/Resources/TranslationModels/"
                        + path.relative_to(model_bundle).as_posix(),
                    )
        distribution = work / "distribution.json"
        maximum_archive = json.loads((ROOT / MANIFEST).read_text())[
            "promotionGate"
        ]["maximumDistributionArchiveBytes"]
        run(
            "scripts/translation/verify_translation_distribution.py",
            str(archive), str(model_bundle), str(metallib), str(distribution),
            "--maximum-archive-bytes", str(maximum_archive),
            stdout=subprocess.DEVNULL,
        )
        promotion = work / "promotion.json"
        promotion_command = [
            "scripts/translation/evaluate_translation_promotion.py",
            str(suite), MANIFEST, str(validation), str(candidate_report), str(apple_report),
            str(candidate_learned), str(apple_learned),
            str(comparison / "sealed-assignments.jsonl"),
            str(comparison / "human-a.responses.jsonl"),
            str(comparison / "human-b.responses.jsonl"),
            str(fallback), str(parity), str(distribution), str(promotion),
        ]
        run(*promotion_command, stdout=subprocess.DEVNULL)
        assert json.loads(promotion.read_text())["promote"] is True

        first = final_suite[0]["id"]
        review_path = comparison / "human-a.responses.jsonl"
        responses = [json.loads(line) for line in review_path.read_text().splitlines()]
        response = next(value for value in responses if value["caseID"] == first)
        assignment = assignments[("human-a", first)]
        candidate_label = "outputA" if assignment["outputAEngine"] == "candidate" else "outputB"
        response[candidate_label]["criticalError"] = True
        write_jsonl(review_path, responses)
        rejected_promotion = subprocess.run(
            [sys.executable, *promotion_command], cwd=ROOT, capture_output=True, text=True
        )
        assert rejected_promotion.returncode == 2
        assert json.loads(promotion.read_text())["promote"] is False

    print("Mimi held-out benchmark and promotion contract smoke passed.")


if __name__ == "__main__":
    main()
