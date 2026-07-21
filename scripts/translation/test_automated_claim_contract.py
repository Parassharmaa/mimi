#!/usr/bin/env python3
"""Contract tests for the separate reviewer-free benchmark validator."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "scripts/translation/validate_automated_benchmark_suite.py"
FINALIZER = ROOT / "scripts/translation/assemble_automated_claim_reference_suite.py"
HASH = "a" * 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


DOMAIN_COUNTS = {
    "meeting-and-live-speech": 120,
    "everyday-conversation": 80,
    "macos-and-technical-ui": 60,
    "numbers-dates-and-entities": 60,
    "politeness-ambiguity-and-omission": 60,
    "code-switching": 20,
}


def case(identifier: str, direction: str, domain: str, index: int) -> dict:
    en_ja = direction == "en-US>ja-JP"
    source = f"Fresh meeting sentence {index}." if en_ja else f"新規会議文{index}です。"
    references = (
        [f"新しい会議文{index}です。", f"会議用の新規文{index}です。"]
        if en_ja
        else [f"This is fresh meeting sentence {index}.", f"A new meeting sentence {index}."]
    )
    return {
        "id": identifier,
        "documentID": f"document-{identifier}",
        "sourceLanguage": "en-US" if en_ja else "ja-JP",
        "targetLanguage": "ja-JP" if en_ja else "en-US",
        "domain": domain,
        "source": source,
        "references": references,
        "acceptedReferenceCandidateIDs": [f"{identifier}:candidate-1", f"{identifier}:candidate-2"],
        "split": "heldout-automated",
        "reviewStatus": "automated-two-judge-consensus-v1",
        "claimEligible": True,
        "sourceGeneratedByAI": False,
        "referenceGeneratedByAI": True,
        "publicBenchmarkOrigin": False,
        "paraphraseOfExistingMaterial": False,
        "sourceCreatedAt": "2026-07-20",
        "license": "Project-owned",
        "provenance": "sealed contract fixture",
    }


def build_fixture(work: Path) -> dict[str, Path]:
    expected_final_rows: list[dict] = []
    for direction, prefix in (("en-US>ja-JP", "en"), ("ja-JP>en-US", "ja")):
        direction_index = 0
        for domain, count in DOMAIN_COUNTS.items():
            for _ in range(count):
                direction_index += 1
                expected_final_rows.append(
                    case(
                        f"{prefix}-{direction_index:03d}",
                        direction,
                        domain,
                        direction_index,
                    )
                )
    source_rows = [
        {
            **row,
            "acceptedReferenceCandidateIDs": [],
            "claimEligible": False,
            "referenceGeneratedByAI": None,
            "references": [],
            "reviewStatus": "references-pending",
            "split": "heldout-automated-source-draft",
        }
        for row in expected_final_rows
    ]
    sources = work / "sources.jsonl"
    write_jsonl(sources, source_rows)
    frozen_sources_sha256 = sha256(sources)
    manifest = work / "manifest.json"
    write_json(
        manifest,
        {
            "schemaVersion": 1,
            "suiteID": "fixture",
            "frozenSources": {
                "path": str(sources),
                "sha256": frozen_sources_sha256,
                "cases": len(source_rows),
                "claimEligible": False,
            },
            "exactCasesPerDirection": 400,
            "directions": ["en-US>ja-JP", "ja-JP>en-US"],
            "domains": {
                "meeting-and-live-speech": 0.30,
                "everyday-conversation": 0.20,
                "macos-and-technical-ui": 0.15,
                "numbers-dates-and-entities": 0.15,
                "politeness-ambiguity-and-omission": 0.15,
                "code-switching": 0.05,
            },
            "sourcePolicy": {
                "minimumCreationDate": "2026-07-20",
                "allowedLicenses": ["Project-owned"],
            },
            "referencePolicy": {
                "mode": "automated-two-judge-consensus-v1",
                "minimumGeneratedCandidatesPerCase": 3,
                "exactAcceptedReferencesPerCase": 2,
                "minimumAdequacy": 4,
                "maximumScore": 4,
                "minimumFluency": 4,
                "minimumTerminology": 4,
                "minimumIndependentBilingualJudges": 2,
                "requiresPinnedGeneratorRevision": True,
                "requiresPinnedJudgeRevisions": True,
                "requiresPromptHashes": True,
                "requiresRequestAndResponseHashes": True,
                "requiresNoReasoningTraceRetention": True,
                "requiresStoreFalse": True,
                "requiresExactCoverage": True,
                "criticalErrorIfAnyJudgeFlags": True,
            },
            "contaminationPolicy": {
                "requiredExposureScopes": [
                    "training",
                    "development",
                    "teacher-input",
                    "teacher-output",
                    "router",
                    "model-selection",
                    "exact-memory",
                ],
                "characterNgramSize": 5,
                "maximumTrainHeldoutJaccard": 0.65,
                "forbidTrainingDocumentIDOverlap": True,
                "maximumSemanticSimilarity": 0.82,
            },
        },
    )
    generator = work / "generator.json"
    generator_results = []
    for row in expected_final_rows:
        candidates = [
            {
                "candidateID": identifier,
                "text": text,
                "sha256": text_hash(text),
            }
            for identifier, text in [
                (row["acceptedReferenceCandidateIDs"][0], row["references"][0]),
                (row["acceptedReferenceCandidateIDs"][1], row["references"][1]),
                (f"{row['id']}:candidate-3", f"unused alternative {row['id']}"),
            ]
        ]
        generator_results.append(
            {
                "caseID": row["id"],
                "sourceSHA256": text_hash(row["source"]),
                "requestSHA256": HASH,
                "responseSHA256": "b" * 64,
                "candidates": candidates,
            }
        )
    write_json(
        generator,
        {
            "schemaVersion": 1,
            "purpose": "benchmark-reference-generation",
            "generatorModel": "reference-generator",
            "generatorModelFamily": "family-generator",
            "generatorRevision": "revision-1",
            "promptSHA256": "c" * 64,
            "reasoningTracesStored": False,
            "store": False,
            "sourceSuiteSHA256": frozen_sources_sha256,
            "requestFileSHA256": "1" * 64,
            "rawBatchOutputSHA256": "2" * 64,
            "results": generator_results,
        },
    )

    def judge(path: Path, role: str, model: str, family: str) -> None:
        results = []
        for row, generated in zip(expected_final_rows, generator_results, strict=True):
            assessments = [
                {
                    "candidateID": row["acceptedReferenceCandidateIDs"][index],
                    "referenceSHA256": text_hash(reference),
                    "adequacy": 4,
                    "fluency": 4,
                    "terminology": 4,
                    "criticalError": False,
                    "protectedTokensPreserved": True,
                    "errorTags": [],
                    "acceptAsReference": True,
                }
                for index, reference in enumerate(row["references"])
            ]
            unused = generated["candidates"][2]
            assessments.append(
                {
                    "candidateID": unused["candidateID"],
                    "referenceSHA256": unused["sha256"],
                    "adequacy": 3,
                    "fluency": 4,
                    "terminology": 4,
                    "criticalError": False,
                    "protectedTokensPreserved": True,
                    "errorTags": ["addition"],
                    "acceptAsReference": False,
                }
            )
            results.append(
                {
                    "caseID": row["id"],
                    "sourceSHA256": text_hash(row["source"]),
                    "requestSHA256": "d" * 64,
                    "responseSHA256": "e" * 64,
                    "assessments": assessments,
                }
            )
        write_json(
            path,
            {
                "schemaVersion": 1,
                "purpose": "benchmark-reference-review",
                "judgeRole": role,
                "judgeModel": model,
                "judgeModelFamily": family,
                "judgeRevision": "revision-1",
                "promptSHA256": "f" * 64,
                "reasoningTracesStored": False,
                "store": False,
                "sourceSuiteSHA256": frozen_sources_sha256,
                "generatorReportSHA256": sha256(generator),
                "requestFileSHA256": "3" * 64,
                "rawBatchOutputSHA256": "4" * 64,
                "results": results,
            },
        )

    judge_a = work / "judge-a.json"
    judge_b = work / "judge-b.json"
    judge(judge_a, "reference-judge-a", "judge-a", "family-a")
    judge(judge_b, "reference-judge-b", "judge-b", "family-b")
    suite = work / "suite.jsonl"
    decision = work / "decision.json"
    finalized = subprocess.run(
        [
            "python3",
            str(FINALIZER),
            str(sources),
            str(generator),
            str(judge_a),
            str(judge_b),
            str(suite),
            str(decision),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert finalized.returncode == 0, finalized.stderr
    suite_rows = [json.loads(line) for line in suite.read_text(encoding="utf-8").splitlines()]
    assert len(suite_rows) == 800
    assert all(row["claimEligible"] is True for row in suite_rows)
    structural = work / "structural.json"
    write_json(
        structural,
        {
            "schemaVersion": 1,
            "status": "passed",
            "suiteSHA256": sha256(suite),
            "judgeReportASHA256": sha256(judge_a),
            "judgeReportBSHA256": sha256(judge_b),
            "results": [
                {
                    "caseID": row["id"],
                    "sourceSHA256": text_hash(row["source"]),
                    "referenceSHA256s": [text_hash(value) for value in row["references"]],
                    "criticalError": False,
                    "errorTags": [],
                    "checks": {
                        name: True
                        for name in (
                            "numbers",
                            "entities",
                            "negation",
                            "placeholders",
                            "urls",
                            "markup",
                            "codeSwitching",
                            "omission",
                        )
                    },
                }
                for row in suite_rows
            ],
        },
    )
    extraction = work / "exposure.jsonl"
    write_jsonl(extraction, [{"source": "Unrelated legacy material about weather."}])
    upstream_revision = "1" * 40
    release = work / "release-contract.json"
    write_json(
        release,
        {
            "schemaVersion": 1,
            "upstreamModels": {
                f"fixture/base@{upstream_revision}": {
                    "repository": "fixture/base",
                    "revision": upstream_revision,
                    "license": "CC-BY-SA-4.0",
                }
            },
        },
    )
    metadata = {
        "createdAt": "2024-05-20T01:51:18.000Z",
        "id": "fixture/base",
        "lastModified": "2024-05-20T01:53:38.000Z",
        "sha": upstream_revision,
    }
    exposure = work / "exposure-manifest.json"
    write_json(
        exposure,
        {
            "schemaVersion": 2,
            "coverageBasis": "exact-project-controlled-plus-upstream-revision-temporal-exclusion",
            "projectControlledExposureComplete": True,
            "upstreamExactRowsComplete": False,
            "frozenSourcesSHA256": frozen_sources_sha256,
            "trainingTeacherModelsComplete": True,
            "trainingTeacherModels": ["training-teacher"],
            "releaseContract": {"path": str(release), "sha256": sha256(release)},
            "upstreamRevisionAttestations": [
                {
                    "repository": "fixture/base",
                    "revision": upstream_revision,
                    "license": "CC-BY-SA-4.0",
                    "revisionAPIURL": (
                        "https://huggingface.co/api/models/fixture/base/revision/"
                        f"{upstream_revision}"
                    ),
                    "modelCardURL": (
                        "https://huggingface.co/fixture/base/blob/"
                        f"{upstream_revision}/README.md"
                    ),
                    "revisionMetadata": metadata,
                    "revisionMetadataSHA256": canonical_hash(metadata),
                }
            ],
            "evidenceAssetCount": 1,
            "evidenceAssets": [
                {
                    "path": str(release),
                    "sha256": sha256(release),
                    "purpose": "fixture release lineage",
                }
            ],
            "assetCount": 1,
            "assets": [
                {
                    "path": str(extraction),
                    "sha256": sha256(extraction),
                    "projectControlled": True,
                    "scopes": [
                        "training",
                        "development",
                        "router",
                        "model-selection",
                        "exact-memory",
                    ],
                    "textExtractionJSONL": str(extraction),
                    "textExtractionSHA256": sha256(extraction),
                }
            ],
            "zeroTextScopeAttestations": [
                {
                    "scope": scope,
                    "reason": "fixture release uses no training teacher text",
                    "evidenceAssetSHA256s": [sha256(release)],
                }
                for scope in ("teacher-input", "teacher-output")
            ],
        },
    )
    semantic = work / "semantic.json"
    write_json(
        semantic,
        {
            "schemaVersion": 1,
            "status": "passed",
            "suiteSHA256": sha256(suite),
            "exposureManifestSHA256": sha256(exposure),
            "threshold": 0.82,
            "embedderModel": "fixture-embedder",
            "embedderRevision": "revision-1",
            "exhaustiveUniqueExposureTextScan": True,
            "candidatePrefilterUsed": False,
            "sourceAndReferencesScanned": True,
            "queryTextCount": len(suite_rows) * 3,
            "comparisonLanguagePolicy": "same-declared-language-via-deterministic-script-bucket",
            "results": [
                {"caseID": row["id"], "maximumSimilarity": 0.1}
                for row in suite_rows
            ],
        },
    )
    return {
        "sources": sources,
        "suite": suite,
        "manifest": manifest,
        "generator": generator,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "structural": structural,
        "exposure": exposure,
        "extraction": extraction,
        "release": release,
        "semantic": semantic,
        "output": work / "validation.json",
    }


def run(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python3",
            str(VALIDATOR),
            str(paths["suite"]),
            str(paths["manifest"]),
            str(paths["generator"]),
            str(paths["judge_a"]),
            str(paths["judge_b"]),
            str(paths["structural"]),
            str(paths["exposure"]),
            str(paths["semantic"]),
            "--output",
            str(paths["output"]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-automated-claim-") as temporary:
        work = Path(temporary)
        paths = build_fixture(work)
        result = run(paths)
        assert result.returncode == 0, result.stderr
        validation = json.loads(paths["output"].read_text(encoding="utf-8"))
        assert validation["status"] == "claim-ready-automated-suite-validated"
        assert validation["directions"]["en-US>ja-JP"]["cases"] == 400
        assert validation["directions"]["ja-JP>en-US"]["cases"] == 400

        original_sources = paths["sources"].read_text(encoding="utf-8")
        original_manifest_text = paths["manifest"].read_text(encoding="utf-8")
        paths["sources"].write_text(
            "\n".join(original_sources.splitlines()[:-1]) + "\n", encoding="utf-8"
        )
        shortened_manifest = json.loads(original_manifest_text)
        shortened_manifest["frozenSources"].update(
            {
                "sha256": sha256(paths["sources"]),
                "cases": 799,
            }
        )
        write_json(paths["manifest"], shortened_manifest)
        rejected_finalization = subprocess.run(
            [
                "python3",
                str(FINALIZER),
                str(paths["sources"]),
                str(paths["generator"]),
                str(paths["judge_a"]),
                str(paths["judge_b"]),
                str(work / "short-final-suite.jsonl"),
                str(work / "short-final-decision.json"),
                "--manifest",
                str(paths["manifest"]),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected_finalization.returncode != 0
        assert "need exactly 400" in rejected_finalization.stderr
        paths["sources"].write_text(original_sources, encoding="utf-8")
        paths["manifest"].write_text(original_manifest_text, encoding="utf-8")

        original_suite = paths["suite"].read_text(encoding="utf-8")
        paths["suite"].write_text("\n".join(original_suite.splitlines()[:-1]) + "\n", encoding="utf-8")
        result = run(paths)
        assert result.returncode != 0 and "need exactly 400" in result.stderr
        paths["suite"].write_text(original_suite, encoding="utf-8")

        judge_b = json.loads(paths["judge_b"].read_text(encoding="utf-8"))
        original_judge_b = copy.deepcopy(judge_b)
        judge_b["judgeModel"] = "judge-a"
        write_json(paths["judge_b"], judge_b)
        result = run(paths)
        assert result.returncode != 0 and "distinct model families" in result.stderr
        write_json(paths["judge_b"], original_judge_b)

        judge_b = copy.deepcopy(original_judge_b)
        judge_b["judgeRevision"] = ""
        write_json(paths["judge_b"], judge_b)
        result = run(paths)
        assert result.returncode != 0 and "unpinned" in result.stderr
        write_json(paths["judge_b"], original_judge_b)

        judge_b = copy.deepcopy(original_judge_b)
        judge_b["promptSHA256"] = "not-a-hash"
        write_json(paths["judge_b"], judge_b)
        result = run(paths)
        assert result.returncode != 0 and "prompt" in result.stderr
        write_json(paths["judge_b"], original_judge_b)

        judge_b = copy.deepcopy(original_judge_b)
        judge_b["results"][0]["responseSHA256"] = "not-a-hash"
        write_json(paths["judge_b"], judge_b)
        result = run(paths)
        assert result.returncode != 0 and "response" in result.stderr
        write_json(paths["judge_b"], original_judge_b)

        judge_b = copy.deepcopy(original_judge_b)
        judge_b["results"].pop()
        write_json(paths["judge_b"], judge_b)
        result = run(paths)
        assert result.returncode != 0 and "exact frozen suite" in result.stderr
        write_json(paths["judge_b"], original_judge_b)

        judge_b = copy.deepcopy(original_judge_b)
        judge_b["results"][0]["assessments"][0]["acceptAsReference"] = False
        write_json(paths["judge_b"], judge_b)
        result = run(paths)
        assert result.returncode != 0 and "found a reference error" in result.stderr
        write_json(paths["judge_b"], original_judge_b)

        generator = json.loads(paths["generator"].read_text(encoding="utf-8"))
        original_generator = copy.deepcopy(generator)
        generator["results"][0]["candidates"][0]["sha256"] = "0" * 64
        write_json(paths["generator"], generator)
        result = run(paths)
        assert result.returncode != 0 and "candidate hash mismatch" in result.stderr
        write_json(paths["generator"], original_generator)

        generator = copy.deepcopy(original_generator)
        generator["reasoningTracesStored"] = True
        write_json(paths["generator"], generator)
        result = run(paths)
        assert result.returncode != 0 and "reasoning traces" in result.stderr
        write_json(paths["generator"], original_generator)

        exposure = json.loads(paths["exposure"].read_text(encoding="utf-8"))
        original_exposure = copy.deepcopy(exposure)
        exposure["upstreamExactRowsComplete"] = True
        write_json(paths["exposure"], exposure)
        result = run(paths)
        assert result.returncode != 0 and "opaque upstream rows" in result.stderr
        write_json(paths["exposure"], original_exposure)

        exposure = copy.deepcopy(original_exposure)
        exposure["upstreamRevisionAttestations"][0]["revisionMetadata"]["createdAt"] = (
            "2026-07-20T00:00:00Z"
        )
        exposure["upstreamRevisionAttestations"][0]["revisionMetadataSHA256"] = canonical_hash(
            exposure["upstreamRevisionAttestations"][0]["revisionMetadata"]
        )
        write_json(paths["exposure"], exposure)
        result = run(paths)
        assert result.returncode != 0 and "not temporally excluded" in result.stderr
        write_json(paths["exposure"], original_exposure)

        judge_b = json.loads(paths["judge_b"].read_text(encoding="utf-8"))
        original_judge_b = copy.deepcopy(judge_b)
        judge_b["results"][0]["assessments"][0]["criticalError"] = True
        write_json(paths["judge_b"], judge_b)
        result = run(paths)
        assert result.returncode != 0 and "found a reference error" in result.stderr
        write_json(paths["judge_b"], original_judge_b)

        structural = json.loads(paths["structural"].read_text(encoding="utf-8"))
        original_structural = copy.deepcopy(structural)
        structural["results"][0]["checks"]["numbers"] = False
        write_json(paths["structural"], structural)
        result = run(paths)
        assert result.returncode != 0 and "structural check" in result.stderr
        write_json(paths["structural"], original_structural)

        structural = copy.deepcopy(original_structural)
        structural["status"] = "failed"
        structural["results"][0]["criticalError"] = True
        structural["results"][0]["errorTags"] = ["numbers"]
        write_json(paths["structural"], structural)
        result = run(paths)
        assert result.returncode != 0 and "structural report" in result.stderr
        write_json(paths["structural"], original_structural)

        write_jsonl(paths["extraction"], [{"source": "Fresh meeting sentence 1."}])
        exposure = json.loads(paths["exposure"].read_text(encoding="utf-8"))
        exposure["assets"][0]["sha256"] = sha256(paths["extraction"])
        exposure["assets"][0]["textExtractionSHA256"] = sha256(paths["extraction"])
        write_json(paths["exposure"], exposure)
        semantic = json.loads(paths["semantic"].read_text(encoding="utf-8"))
        semantic["exposureManifestSHA256"] = sha256(paths["exposure"])
        write_json(paths["semantic"], semantic)
        result = run(paths)
        assert result.returncode != 0 and "exact-match contamination" in result.stderr

    print("Mimi reviewer-free automated claim contract passed.")


if __name__ == "__main__":
    main()
