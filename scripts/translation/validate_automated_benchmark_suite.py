#!/usr/bin/env python3
"""Fail-closed validation for Mimi's reviewer-free claim benchmark.

This is intentionally separate from validate_benchmark_suite.py. The legacy
validator remains the authority for the human-authored, human-adjudicated lane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from validate_benchmark_suite import apportioned, normalized, scan_training


REQUIRED_STRUCTURAL_CHECKS = (
    "numbers",
    "entities",
    "negation",
    "placeholders",
    "urls",
    "markup",
    "codeSwitching",
    "omission",
)


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    output = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(value, dict) for value in output):
        raise SystemExit(f"expected JSON objects in: {path}")
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_hash(value: object, label: str) -> str:
    candidate = str(value or "").strip().lower()
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        raise SystemExit(f"invalid SHA-256 for {label}")
    return candidate


def index_results(report: dict, label: str, case_ids: set[str]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for row in report.get("results", []):
        case_id = str(row.get("caseID", "")).strip()
        if not case_id or case_id in output or case_id not in case_ids:
            raise SystemExit(f"{label} has missing, duplicate, or unknown case: {case_id}")
        output[case_id] = row
    if set(output) != case_ids:
        raise SystemExit(f"{label} does not cover the exact frozen suite")
    return output


def resolve(base: Path, value: object, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise SystemExit(f"missing path for {label}")
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def parse_utc_timestamp(value: object, label: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise SystemExit(f"missing timestamp for {label}")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise SystemExit(f"invalid timestamp for {label}") from error
    if parsed.tzinfo is None:
        raise SystemExit(f"timestamp must include a timezone for {label}")
    return parsed


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_exposure_manifest(
    path: Path, manifest: dict
) -> tuple[list[Path], set[str], dict]:
    exposure = load(path)
    if exposure.get("schemaVersion") != 2:
        raise SystemExit("exposure manifest must use truthful schema v2")
    if (
        exposure.get("projectControlledExposureComplete") is not True
        or exposure.get("upstreamExactRowsComplete") is not False
        or exposure.get("coverageBasis")
        != "exact-project-controlled-plus-upstream-revision-temporal-exclusion"
    ):
        raise SystemExit(
            "exposure manifest must separate complete project-controlled exposure "
            "from opaque upstream rows"
        )
    frozen = manifest.get("frozenSources", {})
    if exposure.get("frozenSourcesSHA256") != require_hash(
        frozen.get("sha256"), "frozen source suite"
    ):
        raise SystemExit("exposure manifest is not bound to the frozen source suite")
    if exposure.get("trainingTeacherModelsComplete") is not True:
        raise SystemExit("exposure manifest must attest complete training-teacher coverage")
    teacher_models = {
        str(value).strip() for value in exposure.get("trainingTeacherModels", []) if str(value).strip()
    }
    required_scopes = set(manifest["contaminationPolicy"]["requiredExposureScopes"])
    observed_scopes: set[str] = set()
    extraction_paths: list[Path] = []
    seen_extractions: set[str] = set()
    seen_assets: set[str] = set()
    base = path.parent

    release_record = exposure.get("releaseContract")
    if not isinstance(release_record, dict):
        raise SystemExit("exposure manifest lacks its release contract")
    release_path = resolve(base, release_record.get("path"), "release contract")
    if not release_path.is_file() or sha256(release_path) != require_hash(
        release_record.get("sha256"), "release contract"
    ):
        raise SystemExit("missing or changed exposure release contract")
    release = load(release_path)
    upstream = release.get("upstreamModels")
    if not isinstance(upstream, dict) or not upstream:
        raise SystemExit("release contract has no pinned upstream models")
    attestations = exposure.get("upstreamRevisionAttestations")
    if not isinstance(attestations, list) or len(attestations) != len(upstream):
        raise SystemExit("upstream revision attestations do not cover the exact release lineage")
    source_cutoff = date.fromisoformat(manifest["sourcePolicy"]["minimumCreationDate"])
    attested_keys: set[str] = set()
    for index, attestation in enumerate(attestations):
        if not isinstance(attestation, dict):
            raise SystemExit(f"invalid upstream revision attestation: {index}")
        repository = str(attestation.get("repository", "")).strip()
        revision = str(attestation.get("revision", "")).strip()
        key = f"{repository}@{revision}"
        if key in attested_keys or key not in upstream:
            raise SystemExit(f"unknown or duplicate upstream revision attestation: {key}")
        attested_keys.add(key)
        record = upstream[key]
        if (
            record.get("repository") != repository
            or record.get("revision") != revision
            or record.get("license") != attestation.get("license")
        ):
            raise SystemExit(f"upstream revision attestation mismatch: {key}")
        metadata = attestation.get("revisionMetadata")
        if not isinstance(metadata, dict) or metadata != {
            "createdAt": metadata.get("createdAt"),
            "id": repository,
            "lastModified": metadata.get("lastModified"),
            "sha": revision,
        }:
            raise SystemExit(f"invalid pinned revision metadata: {key}")
        if canonical_sha256(metadata) != require_hash(
            attestation.get("revisionMetadataSHA256"), f"revision metadata {key}"
        ):
            raise SystemExit(f"revision metadata hash mismatch: {key}")
        if parse_utc_timestamp(metadata.get("createdAt"), key).date() >= source_cutoff:
            raise SystemExit(f"upstream revision is not temporally excluded: {key}")
        parse_utc_timestamp(metadata.get("lastModified"), key)
        api_url = str(attestation.get("revisionAPIURL", ""))
        card_url = str(attestation.get("modelCardURL", ""))
        if revision not in api_url or revision not in card_url or repository not in card_url:
            raise SystemExit(f"upstream revision evidence is not pinned: {key}")
    if attested_keys != set(upstream):
        raise SystemExit("upstream revision attestation set mismatch")

    evidence_assets = exposure.get("evidenceAssets")
    if not isinstance(evidence_assets, list) or not evidence_assets:
        raise SystemExit("exposure manifest has no evidence assets")
    evidence_hashes: set[str] = set()
    seen_evidence_paths: set[str] = set()
    for index, evidence in enumerate(evidence_assets):
        if not isinstance(evidence, dict):
            raise SystemExit(f"invalid exposure evidence asset: {index}")
        evidence_path = resolve(base, evidence.get("path"), f"evidence asset {index}")
        evidence_key = str(evidence_path)
        evidence_hash = require_hash(evidence.get("sha256"), f"evidence asset {index}")
        if evidence_key in seen_evidence_paths:
            raise SystemExit(f"duplicate exposure evidence asset: {evidence_path}")
        if not evidence_path.is_file() or sha256(evidence_path) != evidence_hash:
            raise SystemExit(f"missing or changed exposure evidence asset: {evidence_path}")
        seen_evidence_paths.add(evidence_key)
        evidence_hashes.add(evidence_hash)
    if exposure.get("evidenceAssetCount") != len(evidence_assets):
        raise SystemExit("exposure manifest evidenceAssetCount does not match evidenceAssets")

    assets = exposure.get("assets")
    if not isinstance(assets, list) or not assets:
        raise SystemExit("exposure manifest has no assets")
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise SystemExit(f"invalid exposure asset: {index}")
        if asset.get("projectControlled") is not True:
            raise SystemExit(f"exposure asset is not project-controlled: {index}")
        asset_path = resolve(base, asset.get("path"), f"exposure asset {index}")
        key = str(asset_path)
        if key in seen_assets:
            raise SystemExit(f"duplicate exposure asset: {asset_path}")
        seen_assets.add(key)
        if not asset_path.is_file() or sha256(asset_path) != require_hash(
            asset.get("sha256"), f"exposure asset {index}"
        ):
            raise SystemExit(f"missing or changed exposure asset: {asset_path}")
        scopes = {str(value).strip() for value in asset.get("scopes", []) if str(value).strip()}
        if not scopes or not scopes <= required_scopes:
            raise SystemExit(f"invalid exposure scopes for: {asset_path}")
        observed_scopes.update(scopes)
        extraction = resolve(
            base,
            asset.get("textExtractionJSONL"),
            f"text extraction for exposure asset {index}",
        )
        if not extraction.is_file() or sha256(extraction) != require_hash(
            asset.get("textExtractionSHA256"), f"text extraction {index}"
        ):
            raise SystemExit(f"missing or changed text extraction: {extraction}")
        if not rows(extraction):
            raise SystemExit(f"empty exposure text extraction: {extraction}")
        extraction_key = str(extraction)
        if extraction_key not in seen_extractions:
            extraction_paths.append(extraction)
            seen_extractions.add(extraction_key)

    zero_text = exposure.get("zeroTextScopeAttestations", [])
    if not isinstance(zero_text, list):
        raise SystemExit("invalid zero-text scope attestations")
    zero_scopes: set[str] = set()
    for index, attestation in enumerate(zero_text):
        if not isinstance(attestation, dict):
            raise SystemExit(f"invalid zero-text scope attestation: {index}")
        scope = str(attestation.get("scope", "")).strip()
        reason = str(attestation.get("reason", "")).strip()
        supporting = {
            require_hash(value, f"zero-text attestation evidence {index}")
            for value in attestation.get("evidenceAssetSHA256s", [])
        }
        if (
            not scope
            or scope in zero_scopes
            or scope not in required_scopes
            or not reason
            or not supporting
            or not supporting <= evidence_hashes
        ):
            raise SystemExit(f"invalid zero-text scope attestation: {scope}")
        zero_scopes.add(scope)
    observed_scopes.update(zero_scopes)
    if observed_scopes != required_scopes:
        raise SystemExit(
            "exposure manifest scope mismatch; "
            f"missing={sorted(required_scopes - observed_scopes)}"
        )
    if exposure.get("assetCount") != len(assets):
        raise SystemExit("exposure manifest assetCount does not match assets")
    return extraction_paths, teacher_models, exposure


def validate_generator(
    report: dict,
    manifest: dict,
    suite: dict[str, dict],
    training_teachers: set[str],
) -> tuple[str, str, dict[str, dict]]:
    if report.get("schemaVersion") != 1 or report.get("purpose") != "benchmark-reference-generation":
        raise SystemExit("invalid reference-generator report")
    model = str(report.get("generatorModel", "")).strip()
    family = str(report.get("generatorModelFamily", "")).strip()
    revision = str(report.get("generatorRevision", "")).strip()
    if not model or not family or not revision or model in training_teachers:
        raise SystemExit("reference generator is unpinned or overlaps a training teacher")
    require_hash(report.get("promptSHA256"), "reference generator prompt")
    require_hash(report.get("requestFileSHA256"), "reference generator request file")
    require_hash(report.get("rawBatchOutputSHA256"), "reference generator raw response file")
    if report.get("sourceSuiteSHA256") != require_hash(
        manifest.get("frozenSources", {}).get("sha256"), "frozen source suite"
    ):
        raise SystemExit("reference generator is not bound to the frozen source suite")
    if report.get("reasoningTracesStored") is not False or report.get("store") is not False:
        raise SystemExit("reference generator must retain no reasoning traces and use store=false")
    indexed = index_results(report, "reference-generator report", set(suite))
    minimum_candidates = int(manifest["referencePolicy"]["minimumGeneratedCandidatesPerCase"])
    for case_id, evidence in indexed.items():
        source = str(suite[case_id]["source"])
        if evidence.get("sourceSHA256") != text_sha256(source):
            raise SystemExit(f"generator source hash mismatch: {case_id}")
        require_hash(evidence.get("requestSHA256"), f"generator request {case_id}")
        require_hash(evidence.get("responseSHA256"), f"generator response {case_id}")
        candidates = evidence.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < minimum_candidates:
            raise SystemExit(f"insufficient generated reference candidates: {case_id}")
        candidate_ids: set[str] = set()
        candidate_text: dict[str, str] = {}
        for candidate in candidates:
            identifier = str(candidate.get("candidateID", "")).strip()
            text = str(candidate.get("text", "")).strip()
            if not identifier or identifier in candidate_ids or not text:
                raise SystemExit(f"invalid generated candidate: {case_id}")
            if candidate.get("sha256") != text_sha256(text):
                raise SystemExit(f"generated candidate hash mismatch: {case_id}/{identifier}")
            candidate_ids.add(identifier)
            candidate_text[identifier] = text
        accepted_ids = [str(value) for value in suite[case_id].get("acceptedReferenceCandidateIDs", [])]
        references = [str(value) for value in suite[case_id]["references"]]
        if len(accepted_ids) != len(references) or any(
            candidate_text.get(identifier) != reference
            for identifier, reference in zip(accepted_ids, references)
        ):
            raise SystemExit(f"suite references are not bound to generator candidates: {case_id}")
    return model, family, indexed


def validate_judge(
    report: dict,
    label: str,
    expected_role: str,
    manifest: dict,
    suite: dict[str, dict],
    forbidden_models: set[str],
    generator_report_sha256: str,
) -> tuple[str, str]:
    if report.get("schemaVersion") != 1 or report.get("purpose") != "benchmark-reference-review":
        raise SystemExit(f"invalid {label} report")
    model = str(report.get("judgeModel", "")).strip()
    family = str(report.get("judgeModelFamily", "")).strip()
    revision = str(report.get("judgeRevision", "")).strip()
    if not model or not family or not revision or model in forbidden_models:
        raise SystemExit(f"{label} is unpinned or not independent")
    require_hash(report.get("promptSHA256"), f"{label} prompt")
    require_hash(report.get("requestFileSHA256"), f"{label} request file")
    require_hash(report.get("rawBatchOutputSHA256"), f"{label} raw response file")
    if (
        report.get("judgeRole") != expected_role
        or report.get("reasoningTracesStored") is not False
        or report.get("store") is not False
        or report.get("sourceSuiteSHA256")
        != require_hash(manifest.get("frozenSources", {}).get("sha256"), "frozen source suite")
        or report.get("generatorReportSHA256") != generator_report_sha256
    ):
        raise SystemExit(f"{label} retained data or is not bound to the frozen evidence")
    indexed = index_results(report, label, set(suite))
    minimum = int(manifest["referencePolicy"]["minimumAdequacy"])
    maximum = int(manifest["referencePolicy"]["maximumScore"])
    for case_id, result in indexed.items():
        if result.get("sourceSHA256") != text_sha256(str(suite[case_id]["source"])):
            raise SystemExit(f"{label} source hash mismatch: {case_id}")
        require_hash(result.get("requestSHA256"), f"{label} request {case_id}")
        require_hash(result.get("responseSHA256"), f"{label} response {case_id}")
        assessments = result.get("assessments")
        references = [str(value) for value in suite[case_id]["references"]]
        if not isinstance(assessments, list) or len(assessments) < len(references):
            raise SystemExit(f"{label} assessment coverage mismatch: {case_id}")
        by_hash = {str(value.get("referenceSHA256", "")): value for value in assessments}
        by_candidate = {str(value.get("candidateID", "")): value for value in assessments}
        expected_hashes = {text_sha256(value) for value in references}
        expected_candidates = {
            identifier: text_sha256(reference)
            for identifier, reference in zip(
                suite[case_id].get("acceptedReferenceCandidateIDs", []), references
            )
        }
        if (
            len(by_hash) != len(assessments)
            or len(by_candidate) != len(assessments)
            or not expected_hashes <= set(by_hash)
            or any(
                identifier not in by_candidate
                or by_candidate[identifier].get("referenceSHA256") != reference_hash
                for identifier, reference_hash in expected_candidates.items()
            )
        ):
            raise SystemExit(f"{label} reference hashes mismatch: {case_id}")
        for assessment in (by_hash[value] for value in expected_hashes):
            scores = [assessment.get(field) for field in ("adequacy", "fluency", "terminology")]
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
                for value in scores
            ):
                raise SystemExit(f"{label} rejected or underscored a reference: {case_id}")
            if (
                assessment.get("criticalError") is not False
                or assessment.get("protectedTokensPreserved") is not True
                or assessment.get("errorTags") != []
                or assessment.get("acceptAsReference") is not True
            ):
                raise SystemExit(f"{label} found a reference error: {case_id}")
    return model, family


def validate_structural_report(
    path: Path,
    suite_path: Path,
    suite: dict[str, dict],
    judge_report_a: Path,
    judge_report_b: Path,
) -> None:
    report = load(path)
    if (
        report.get("schemaVersion") != 1
        or report.get("status") != "passed"
        or report.get("suiteSHA256") != sha256(suite_path)
        or report.get("judgeReportASHA256") != sha256(judge_report_a)
        or report.get("judgeReportBSHA256") != sha256(judge_report_b)
    ):
        raise SystemExit("invalid deterministic structural report")
    indexed = index_results(report, "deterministic structural report", set(suite))
    for case_id, result in indexed.items():
        references = [str(value) for value in suite[case_id]["references"]]
        if (
            result.get("sourceSHA256") != text_sha256(str(suite[case_id]["source"]))
            or result.get("referenceSHA256s") != [text_sha256(value) for value in references]
            or result.get("criticalError") is not False
            or result.get("errorTags") != []
        ):
            raise SystemExit(f"deterministic structural failure: {case_id}")
        checks = result.get("checks", {})
        if any(checks.get(name) is not True for name in REQUIRED_STRUCTURAL_CHECKS):
            raise SystemExit(f"missing deterministic structural check: {case_id}")


def validate_semantic_report(
    path: Path,
    suite_path: Path,
    exposure_path: Path,
    manifest: dict,
    suite: dict[str, dict],
) -> None:
    report = load(path)
    threshold = float(manifest["contaminationPolicy"]["maximumSemanticSimilarity"])
    expected_query_texts = sum(1 + len(row.get("references", [])) for row in suite.values())
    if (
        report.get("schemaVersion") != 1
        or report.get("status") != "passed"
        or report.get("suiteSHA256") != sha256(suite_path)
        or report.get("exposureManifestSHA256") != sha256(exposure_path)
        or float(report.get("threshold", math.nan)) != threshold
        or not str(report.get("embedderModel", "")).strip()
        or not str(report.get("embedderRevision", "")).strip()
        or report.get("exhaustiveUniqueExposureTextScan") is not True
        or report.get("candidatePrefilterUsed") is not False
        or report.get("sourceAndReferencesScanned") is not True
        or report.get("queryTextCount") != expected_query_texts
        or report.get("comparisonLanguagePolicy")
        != "same-declared-language-via-deterministic-script-bucket"
    ):
        raise SystemExit("invalid semantic contamination report")
    indexed = index_results(report, "semantic contamination report", set(suite))
    for case_id, result in indexed.items():
        score = result.get("maximumSimilarity")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) > threshold
        ):
            raise SystemExit(f"semantic contamination threshold exceeded: {case_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("generator_report", type=Path)
    parser.add_argument("judge_report_a", type=Path)
    parser.add_argument("judge_report_b", type=Path)
    parser.add_argument("structural_report", type=Path)
    parser.add_argument("exposure_manifest", type=Path)
    parser.add_argument("semantic_contamination_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = load(args.manifest)
    suite_rows = rows(args.suite)
    if manifest.get("schemaVersion") != 1:
        raise SystemExit("unsupported automated benchmark manifest")
    suite = {str(row.get("id", "")): row for row in suite_rows}
    if not suite or len(suite) != len(suite_rows) or "" in suite:
        raise SystemExit("suite IDs must be non-empty and unique")

    extraction_paths, training_teachers, _ = validate_exposure_manifest(
        args.exposure_manifest, manifest
    )
    expected_directions = set(manifest["directions"])
    allowed_domains = set(manifest["domains"])
    exact = int(manifest["exactCasesPerDirection"])
    accepted_reference_count = int(manifest["referencePolicy"]["exactAcceptedReferencesPerCase"])
    minimum_date = date.fromisoformat(manifest["sourcePolicy"]["minimumCreationDate"])
    allowed_licenses = set(manifest["sourcePolicy"]["allowedLicenses"])
    by_direction: dict[str, list[dict]] = defaultdict(list)
    seen_sources: dict[str, str] = {}
    document_ids: set[str] = set()
    heldout_text: list[tuple[str, str]] = []
    for case_id, row in suite.items():
        direction = f"{row.get('sourceLanguage')}>{row.get('targetLanguage')}"
        if direction not in expected_directions or row.get("domain") not in allowed_domains:
            raise SystemExit(f"unsupported direction or domain: {case_id}")
        if (
            row.get("split") != "heldout-automated"
            or row.get("reviewStatus") != manifest["referencePolicy"]["mode"]
            or row.get("claimEligible") is not True
            or row.get("sourceGeneratedByAI") is not False
            or row.get("referenceGeneratedByAI") is not True
            or row.get("publicBenchmarkOrigin") is not False
            or row.get("paraphraseOfExistingMaterial") is not False
        ):
            raise SystemExit(f"case does not satisfy automated heldout declarations: {case_id}")
        if row.get("license") not in allowed_licenses or not str(row.get("provenance", "")).strip():
            raise SystemExit(f"case has an unsupported license or no provenance: {case_id}")
        created = date.fromisoformat(str(row.get("sourceCreatedAt", "")))
        if created < minimum_date:
            raise SystemExit(f"case predates the preregistered source cutoff: {case_id}")
        document_id = str(row.get("documentID", "")).strip()
        source = str(row.get("source", "")).strip()
        references = [str(value).strip() for value in row.get("references", [])]
        if (
            not document_id
            or not source
            or len(references) != accepted_reference_count
            or not all(references)
            or len({normalized(value) for value in references}) != len(references)
        ):
            raise SystemExit(f"case has invalid document/source/references: {case_id}")
        normalized_source = normalized(source)
        if normalized_source in seen_sources:
            raise SystemExit(f"duplicate source: {seen_sources[normalized_source]} and {case_id}")
        seen_sources[normalized_source] = case_id
        document_ids.add(document_id)
        heldout_text.extend([(case_id, source), *((case_id, value) for value in references)])
        by_direction[direction].append(row)

    if set(by_direction) != expected_directions:
        raise SystemExit("suite does not contain every required direction")
    directions: dict[str, dict] = {}
    for direction, values in by_direction.items():
        if len(values) != exact:
            raise SystemExit(f"{direction} has {len(values)} cases; need exactly {exact}")
        expected_domains = apportioned(exact, manifest["domains"])
        actual_domains = Counter(str(value["domain"]) for value in values)
        if dict(actual_domains) != expected_domains:
            raise SystemExit(
                f"{direction} domain quota mismatch; actual={dict(actual_domains)} "
                f"expected={expected_domains}"
            )
        directions[direction] = {
            "cases": len(values),
            "domains": dict(sorted(actual_domains.items())),
        }

    generator_report = load(args.generator_report)
    generator_model, generator_family, _ = validate_generator(
        generator_report, manifest, suite, training_teachers
    )
    judge_a = load(args.judge_report_a)
    judge_b = load(args.judge_report_b)
    model_a, family_a = validate_judge(
        judge_a,
        "reference judge A",
        "reference-judge-a",
        manifest,
        suite,
        training_teachers | {generator_model},
        sha256(args.generator_report),
    )
    model_b, family_b = validate_judge(
        judge_b,
        "reference judge B",
        "reference-judge-b",
        manifest,
        suite,
        training_teachers | {generator_model},
        sha256(args.generator_report),
    )
    if model_a == model_b or family_a == family_b or generator_family in {family_a, family_b}:
        raise SystemExit("reference judges and generator must use distinct model families")

    validate_structural_report(
        args.structural_report,
        args.suite,
        suite,
        args.judge_report_a,
        args.judge_report_b,
    )
    validate_semantic_report(
        args.semantic_contamination_report,
        args.suite,
        args.exposure_manifest,
        manifest,
        suite,
    )
    scanned = scan_training(
        extraction_paths,
        heldout_text,
        document_ids,
        int(manifest["contaminationPolicy"]["characterNgramSize"]),
        float(manifest["contaminationPolicy"]["maximumTrainHeldoutJaccard"]),
        bool(manifest["contaminationPolicy"]["forbidTrainingDocumentIDOverlap"]),
    )

    output = {
        "schemaVersion": 1,
        "status": "claim-ready-automated-suite-validated",
        "suiteID": manifest["suiteID"],
        "suite": {"path": str(args.suite), "sha256": sha256(args.suite)},
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        "referenceEvidence": {
            "generatorReportSHA256": sha256(args.generator_report),
            "judgeReportASHA256": sha256(args.judge_report_a),
            "judgeReportBSHA256": sha256(args.judge_report_b),
            "structuralReportSHA256": sha256(args.structural_report),
        },
        "contaminationEvidence": {
            "exposureManifestSHA256": sha256(args.exposure_manifest),
            "semanticReportSHA256": sha256(args.semantic_contamination_report),
            "exposureAssets": len(extraction_paths),
            "exposureTextsScanned": scanned,
        },
        "directions": directions,
        "generatorModel": generator_model,
        "judgeModels": sorted([model_a, model_b]),
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
