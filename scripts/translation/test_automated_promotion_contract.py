#!/usr/bin/env python3
"""End-to-end contract test for reviewer-free local-model promotion."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/translation/evaluate_automated_translation_promotion.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def fixture(work: Path) -> dict[str, Path]:
    suite_rows = []
    for direction, prefix in (("en-US>ja-JP", "en"), ("ja-JP>en-US", "ja")):
        for index in (1, 2):
            en_ja = prefix == "en"
            suite_rows.append(
                {
                    "id": f"{prefix}-{index}",
                    "sourceLanguage": "en-US" if en_ja else "ja-JP",
                    "targetLanguage": "ja-JP" if en_ja else "en-US",
                    "domain": "conversation",
                    "source": f"Source sentence {index}." if en_ja else f"原文{index}です。",
                    "references": [f"正確な翻訳{index}です。"] if en_ja else [f"Accurate translation {index}."],
                    "claimEligible": True,
                }
            )
    suite = work / "suite.jsonl"
    write_jsonl(suite, suite_rows)
    learned = {
        "name": "COMET-22",
        "modelRepository": "Unbabel/wmt22-comet-da",
        "modelRevision": "371e9839ca4e213dde891b066cf3080f75ec7e72",
        "modelLicense": "Apache-2.0",
        "package": "unbabel-comet",
        "packageVersion": "2.2.7",
        "setuptoolsVersion": "80.9.0",
        "precision": "float32",
        "multipleReferenceAggregation": "mean",
    }
    manifest = work / "manifest.json"
    write_json(
        manifest,
        {
            "schemaVersion": 1,
            "suiteID": "automated-promotion-fixture",
            "randomSeed": 7,
            "minimumCasesPerDirection": 2,
            "directions": ["en-US>ja-JP", "ja-JP>en-US"],
            "domains": {"conversation": 1.0},
            "measurement": {"warmRuns": 3, "learnedMetric": learned},
            "promotionGate": {
                "pairedBootstrapSamples": 200,
                "confidenceLevel": 0.95,
                "minimumChrFPlusPlus": {"en-US>ja-JP": 50.0, "ja-JP>en-US": 50.0},
                "minimumLearnedMetric": {"en-US>ja-JP": 0.8, "ja-JP>en-US": 0.8},
                "minimumChrFDeltaLowerBound": 0.0,
                "minimumLearnedMetricDeltaLowerBound": 0.0,
                "minimumAutomatedPairwiseDeltaLowerBound": 0.0,
                "minimumAutomatedCandidateMeanScore": 8.5,
                "maximumCriticalMeaningErrors": 0,
                "maximumWarmP95LatencySeconds": 0.25,
                "maximumPeakResidentBytes": 805306368,
                "preferredModelBytes": 150000000,
                "maximumModelBytes": 500000000,
                "maximumDistributionArchiveBytes": 500000000,
            },
        },
    )
    validation = work / "validation.json"
    write_json(
        validation,
        {
            "status": "claim-ready-automated-suite-validated",
            "suiteID": "automated-promotion-fixture",
            "suite": {"sha256": sha256(suite)},
            "manifest": {"sha256": sha256(manifest)},
        },
    )

    def engine_report(path: Path, engine: str, revision: str, candidate: bool) -> None:
        values = []
        for row in suite_rows:
            hypothesis = row["references"][0] if candidate else "wrong output"
            values.append(
                {
                    "caseID": row["id"],
                    "sourceLanguage": row["sourceLanguage"],
                    "targetLanguage": row["targetLanguage"],
                    "domain": row["domain"],
                    "source": row["source"],
                    "references": row["references"],
                    "claimEligible": True,
                    "hypothesis": hypothesis,
                    "latencySeconds": 0.05,
                    "warmLatencySeconds": [0.04, 0.05, 0.06],
                }
            )
        write_json(
            path,
            {
                "schemaVersion": 1,
                "engine": engine,
                "modelRevision": revision,
                "hardware": "fixture-mac",
                "operatingSystem": "fixture-macos",
                "modelBytes": 149000000 if candidate else 140000000,
                "peakResidentBytes": 300000000,
                "trainingTeacherModels": ["training-teacher"] if candidate else [],
                "results": values,
            },
        )

    candidate = work / "candidate.json"
    baseline = work / "baseline.json"
    engine_report(candidate, "candidate-local", "candidate-revision", True)
    engine_report(baseline, "prior-local", "baseline-revision", False)

    learned_report_fields = {"metric": learned["name"], **{
        key: value for key, value in learned.items() if key != "name"
    }}
    expected_signature = hashlib.sha256(
        json.dumps(learned_report_fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def learned_report(path: Path, engine: str, engine_path: Path, score: float) -> None:
        write_json(
            path,
            {
                "schemaVersion": 1,
                "engine": engine,
                **learned_report_fields,
                "signatureSHA256": expected_signature,
                "engineReportSHA256": sha256(engine_path),
                "suiteSHA256": sha256(suite),
                "results": [{"caseID": row["id"], "score": score} for row in suite_rows],
            },
        )

    candidate_learned = work / "candidate-learned.json"
    baseline_learned = work / "baseline-learned.json"
    learned_report(candidate_learned, "candidate-local", candidate, 0.9)
    learned_report(baseline_learned, "prior-local", baseline, 0.5)

    candidate_report = json.loads(candidate.read_text(encoding="utf-8"))
    baseline_report = json.loads(baseline.read_text(encoding="utf-8"))

    def judge(path: Path, model: str, family: str) -> None:
        write_json(
            path,
            {
                "schemaVersion": 1,
                "purpose": "blinded-automated-engine-comparison",
                "suiteSHA256": sha256(suite),
                "candidateReportSHA256": sha256(candidate),
                "baselineReportSHA256": sha256(baseline),
                "judgeModel": model,
                "judgeModelFamily": family,
                "judgeRevision": "revision-1",
                "promptSHA256": "a" * 64,
                "reasoningTracesStored": False,
                "results": [
                    {
                        "caseID": row["id"],
                        "blinded": True,
                        "candidateHypothesisSHA256": text_hash(
                            candidate_report["results"][index]["hypothesis"]
                        ),
                        "baselineHypothesisSHA256": text_hash(
                            baseline_report["results"][index]["hypothesis"]
                        ),
                        "requestSHA256": "b" * 64,
                        "responseSHA256": "c" * 64,
                        "candidate": {
                            "adequacy": 4,
                            "fluency": 4,
                            "terminology": 2,
                            "criticalError": False,
                            "errorTags": [],
                        },
                        "baseline": {
                            "adequacy": 1,
                            "fluency": 2,
                            "terminology": 1,
                            "criticalError": False,
                            "errorTags": [],
                        },
                    }
                    for index, row in enumerate(suite_rows)
                ],
            },
        )

    judge_a = work / "judge-a.json"
    judge_b = work / "judge-b.json"
    judge(judge_a, "pairwise-a", "family-a")
    judge(judge_b, "pairwise-b", "family-b")
    critical = work / "critical.json"
    write_json(
        critical,
        {
            "schemaVersion": 1,
            "status": "passed",
            "suiteSHA256": sha256(suite),
            "candidateReportSHA256": sha256(candidate),
            "results": [
                {
                    "caseID": row["id"],
                    "hypothesisSHA256": text_hash(candidate_report["results"][index]["hypothesis"]),
                    "criticalError": False,
                    "errorTags": [],
                }
                for index, row in enumerate(suite_rows)
            ],
        },
    )
    failure = work / "failure.json"
    write_json(
        failure,
        {
            "status": "passed",
            "appleDefaultWhenExperimentalDisabled": True,
            "candidateFailureDoesNotUseApple": True,
            "candidateFailurePreservesLocalResults": True,
            "candidateFailureShowsRetryableError": True,
            "applePartialsWhenExperimentalDisabled": True,
            "experimentalPartialsDoNotUseApple": True,
        },
    )
    parity = work / "parity.json"
    write_json(
        parity,
        {
            "schemaVersion": 1,
            "status": "passed",
            "engine": "swift-mlx-marian-exact-output-parity",
            "modelRevision": "candidate-revision",
            "suiteSHA256": sha256(suite),
            "pythonReportSHA256": sha256(candidate),
            "cases": len(suite_rows),
            "exactMatches": len(suite_rows),
            "results": [{"caseID": row["id"], "exactMatch": True} for row in suite_rows],
        },
    )
    archive = work / "Mimi-fixture.zip"
    archive.write_bytes(b"fixture archive")
    distribution = work / "distribution.json"
    write_json(
        distribution,
        {
            "schemaVersion": 1,
            "status": "passed",
            "modelRevision": "candidate-revision",
            "archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": sha256(archive),
            },
            "modelBundle": {"bytes": 149000000},
        },
    )
    return {
        "suite": suite,
        "manifest": manifest,
        "validation": validation,
        "candidate": candidate,
        "baseline": baseline,
        "candidate_learned": candidate_learned,
        "baseline_learned": baseline_learned,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "critical": critical,
        "failure": failure,
        "parity": parity,
        "distribution": distribution,
        "output": work / "promotion.json",
    }


def run(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(EVALUATOR),
            str(paths["suite"]),
            str(paths["manifest"]),
            str(paths["validation"]),
            str(paths["candidate"]),
            str(paths["baseline"]),
            str(paths["candidate_learned"]),
            str(paths["baseline_learned"]),
            str(paths["judge_a"]),
            str(paths["judge_b"]),
            str(paths["critical"]),
            str(paths["failure"]),
            str(paths["parity"]),
            str(paths["distribution"]),
            str(paths["output"]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-automated-promotion-") as temporary:
        paths = fixture(Path(temporary))
        result = run(paths)
        assert result.returncode == 0, result.stderr
        output = json.loads(paths["output"].read_text(encoding="utf-8"))
        assert output["promote"] is True
        assert output["appleComparisonRole"] == "diagnostic-only-not-a-promotion-input"
        assert output["preferredSizeTargetMet"] is True

        critical = json.loads(paths["critical"].read_text(encoding="utf-8"))
        original_critical = copy.deepcopy(critical)
        critical["status"] = "failed"
        critical["results"][0]["criticalError"] = True
        critical["results"][0]["errorTags"] = ["number-change"]
        write_json(paths["critical"], critical)
        result = run(paths)
        assert result.returncode == 2, result.stderr
        output = json.loads(paths["output"].read_text(encoding="utf-8"))
        assert output["promote"] is False
        assert output["directions"]["en-US>ja-JP"]["candidateCriticalCaseIDs"] == ["en-1"]
        write_json(paths["critical"], original_critical)

        judge_b = json.loads(paths["judge_b"].read_text(encoding="utf-8"))
        original_judge_b = copy.deepcopy(judge_b)
        judge_b["judgeModel"] = "pairwise-a"
        write_json(paths["judge_b"], judge_b)
        result = run(paths)
        assert result.returncode != 0 and "distinct models" in result.stderr
        write_json(paths["judge_b"], original_judge_b)

        for judge_path in (paths["judge_a"], paths["judge_b"]):
            judge = json.loads(judge_path.read_text(encoding="utf-8"))
            for row in judge["results"]:
                row["baseline"] = copy.deepcopy(row["candidate"])
            write_json(judge_path, judge)
        result = run(paths)
        assert result.returncode == 2, result.stderr
        output = json.loads(paths["output"].read_text(encoding="utf-8"))
        gates = {
            gate["name"]: gate
            for gate in output["directions"]["en-US>ja-JP"]["gates"]
        }
        assert gates["automated-pairwise-paired-bootstrap-lower"]["passed"] is False

    print("Mimi automated local-model promotion contract passed.")


if __name__ == "__main__":
    main()
