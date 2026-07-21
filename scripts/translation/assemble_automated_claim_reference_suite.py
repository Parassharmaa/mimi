#!/usr/bin/env python3
"""Admit exactly two references per case after unanimous blinded consensus."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from collect_automated_claim_reference_candidates import index, rows, sha256, text_sha256


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "Research/translation/benchmark/automated-claim-v1.manifest.json"
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def normalized(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def character_ngrams(value: str, size: int = 3) -> set[str]:
    value = normalized(value)
    if not value:
        return set()
    return {value[index : index + size] for index in range(max(1, len(value) - size + 1))}


def jaccard(left: str, right: str) -> float:
    left_ngrams = character_ngrams(left)
    right_ngrams = character_ngrams(right)
    union = left_ngrams | right_ngrams
    return 1.0 if not union else len(left_ngrams & right_ngrams) / len(union)


def require_sha256(value: object, label: str) -> str:
    candidate = str(value or "").strip().lower()
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        raise SystemExit(f"invalid SHA-256 for {label}")
    return candidate


def apportioned(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {domain: total * weight for domain, weight in weights.items()}
    counts = {domain: math.floor(value) for domain, value in raw.items()}
    order = sorted(weights, key=lambda domain: (-(raw[domain] - counts[domain]), domain))
    for domain in order[: total - sum(counts.values())]:
        counts[domain] += 1
    return counts


def validate_frozen_sources(manifest: dict, source_path: Path, source_rows: list[dict]) -> None:
    if manifest.get("schemaVersion") != 1:
        raise SystemExit("unsupported automated benchmark manifest")
    policy = manifest.get("referencePolicy", {})
    required_policy = {
        "mode": "automated-two-judge-consensus-v1",
        "minimumGeneratedCandidatesPerCase": 3,
        "exactAcceptedReferencesPerCase": 2,
        "minimumIndependentBilingualJudges": 2,
        "minimumAdequacy": 4,
        "minimumFluency": 4,
        "minimumTerminology": 4,
        "maximumScore": 4,
        "requiresPinnedGeneratorRevision": True,
        "requiresPinnedJudgeRevisions": True,
        "requiresPromptHashes": True,
        "requiresRequestAndResponseHashes": True,
        "requiresNoReasoningTraceRetention": True,
        "requiresStoreFalse": True,
        "requiresExactCoverage": True,
        "criticalErrorIfAnyJudgeFlags": True,
    }
    if any(policy.get(name) != value for name, value in required_policy.items()):
        raise SystemExit("automated reference policy is missing or weakened")
    frozen = manifest.get("frozenSources", {})
    if (
        frozen.get("sha256") != sha256(source_path)
        or frozen.get("cases") != len(source_rows)
        or frozen.get("claimEligible") is not False
    ):
        raise SystemExit("manifest is not bound to the exact frozen source suite")

    exact = int(manifest["exactCasesPerDirection"])
    expected_directions = set(manifest["directions"])
    domain_weights = manifest["domains"]
    allowed_domains = set(domain_weights)
    by_direction: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        case_id = str(row.get("id", "")).strip()
        direction = f"{row.get('sourceLanguage')}>{row.get('targetLanguage')}"
        if direction not in expected_directions or row.get("domain") not in allowed_domains:
            raise SystemExit(f"unsupported frozen source direction or domain: {case_id}")
        if (
            row.get("split") != "heldout-automated-source-draft"
            or row.get("reviewStatus") != "references-pending"
            or row.get("claimEligible") is not False
            or row.get("sourceGeneratedByAI") is not False
            or row.get("referenceGeneratedByAI") is not None
            or row.get("references") != []
            or row.get("acceptedReferenceCandidateIDs") != []
            or row.get("publicBenchmarkOrigin") is not False
            or row.get("paraphraseOfExistingMaterial") is not False
        ):
            raise SystemExit(f"source is not a frozen claim-ineligible draft: {case_id}")
        by_direction[direction].append(row)
    if set(by_direction) != expected_directions:
        raise SystemExit("frozen sources do not contain every required direction")
    expected_domains = apportioned(exact, domain_weights)
    for direction, values in by_direction.items():
        if len(values) != exact:
            raise SystemExit(f"{direction} has {len(values)} frozen sources; need exactly {exact}")
        actual_domains = Counter(str(value["domain"]) for value in values)
        if dict(actual_domains) != expected_domains:
            raise SystemExit(
                f"{direction} source domain quota mismatch; actual={dict(actual_domains)} "
                f"expected={expected_domains}"
            )


def report_results(report: dict, label: str, expected: set[str]) -> dict[str, dict]:
    results = report.get("results")
    if not isinstance(results, list):
        raise SystemExit(f"{label} has no result list")
    indexed = index(results, "caseID", label)
    if set(indexed) != expected:
        raise SystemExit(f"{label} does not have exact source-suite coverage")
    return indexed


def accepted_assessments(result: dict, label: str) -> dict[str, dict]:
    assessments = result.get("assessments")
    if not isinstance(assessments, list):
        raise SystemExit(f"{label} has no assessments")
    indexed = index(assessments, "candidateID", label)
    for candidate_id, assessment in indexed.items():
        scores = [assessment.get(field) for field in ("adequacy", "fluency", "terminology")]
        consistent = (
            scores == [4, 4, 4]
            and assessment.get("criticalError") is False
            and assessment.get("protectedTokensPreserved") is True
            and assessment.get("errorTags") == []
        )
        if assessment.get("acceptAsReference") is not consistent:
            raise SystemExit(f"{label} has contradictory acceptance: {candidate_id}")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("generator_report", type=Path)
    parser.add_argument("judge_report_a", type=Path)
    parser.add_argument("judge_report_b", type=Path)
    parser.add_argument("output_suite", type=Path)
    parser.add_argument("decision_report", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    for output in (args.output_suite, args.decision_report):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    source_rows = rows(args.sources)
    sources = index(source_rows, "id", "source suite")
    manifest = load(args.manifest)
    validate_frozen_sources(manifest, args.sources, source_rows)
    expected = set(sources)
    generator = load(args.generator_report)
    judge_a = load(args.judge_report_a)
    judge_b = load(args.judge_report_b)
    generated = report_results(generator, "generator report", expected)
    judged_a = report_results(judge_a, "judge A report", expected)
    judged_b = report_results(judge_b, "judge B report", expected)

    if (
        generator.get("schemaVersion") != 1
        or generator.get("purpose") != "benchmark-reference-generation"
        or generator.get("sourceSuiteSHA256") != sha256(args.sources)
        or generator.get("reasoningTracesStored") is not False
        or generator.get("store") is not False
        or not str(generator.get("generatorModel", "")).strip()
        or not str(generator.get("generatorModelFamily", "")).strip()
        or not str(generator.get("generatorRevision", "")).strip()
    ):
        raise SystemExit("invalid or unbound reference-generator report")
    require_sha256(generator.get("promptSHA256"), "reference generator prompt")
    require_sha256(generator.get("requestFileSHA256"), "reference generator request file")
    require_sha256(generator.get("rawBatchOutputSHA256"), "reference generator raw response file")
    generator_hash = sha256(args.generator_report)
    generator_model = str(generator.get("generatorModel", ""))
    generator_family = str(generator.get("generatorModelFamily", ""))
    judge_models: set[str] = set()
    judge_families: set[str] = set()
    judge_roles: set[str] = set()
    for label, expected_role, report in (
        ("judge A", "reference-judge-a", judge_a),
        ("judge B", "reference-judge-b", judge_b),
    ):
        if (
            report.get("schemaVersion") != 1
            or report.get("purpose") != "benchmark-reference-review"
            or report.get("sourceSuiteSHA256") != sha256(args.sources)
            or report.get("generatorReportSHA256") != generator_hash
            or report.get("reasoningTracesStored") is not False
            or report.get("store") is not False
            or not str(report.get("judgeRevision", "")).strip()
        ):
            raise SystemExit(f"invalid or unbound {label} report")
        require_sha256(report.get("promptSHA256"), f"{label} prompt")
        require_sha256(report.get("requestFileSHA256"), f"{label} request file")
        require_sha256(report.get("rawBatchOutputSHA256"), f"{label} raw response file")
        model = str(report.get("judgeModel", ""))
        family = str(report.get("judgeModelFamily", ""))
        if not model or not family or model == generator_model or family == generator_family:
            raise SystemExit(f"{label} is not independent from the generator")
        judge_models.add(model)
        judge_families.add(family)
        judge_roles.add(str(report.get("judgeRole", "")))
        if report.get("judgeRole") != expected_role:
            raise SystemExit(f"{label} has the wrong role attestation")
    if (
        len(judge_models) != 2
        or len(judge_families) != 2
        or judge_roles != {"reference-judge-a", "reference-judge-b"}
    ):
        raise SystemExit("reference judges must use two distinct model families")

    admitted_rows: list[dict] = []
    decisions: list[dict] = []
    failed_cases: list[str] = []
    for case_id, source in sources.items():
        generated_result = generated[case_id]
        candidates = generated_result.get("candidates")
        if (
            generated_result.get("sourceSHA256") != text_sha256(str(source["source"]))
            or not isinstance(candidates, list)
            or len(candidates) != 3
        ):
            raise SystemExit(f"invalid generator result: {case_id}")
        require_sha256(generated_result.get("requestSHA256"), f"generator request {case_id}")
        require_sha256(generated_result.get("responseSHA256"), f"generator response {case_id}")
        candidate_by_id = index(candidates, "candidateID", f"generator candidates for {case_id}")
        a = accepted_assessments(judged_a[case_id], f"judge A/{case_id}")
        b = accepted_assessments(judged_b[case_id], f"judge B/{case_id}")
        for label, result in (("judge A", judged_a[case_id]), ("judge B", judged_b[case_id])):
            if result.get("sourceSHA256") != text_sha256(str(source["source"])):
                raise SystemExit(f"{label} source hash mismatch: {case_id}")
            require_sha256(result.get("requestSHA256"), f"{label} request {case_id}")
            require_sha256(result.get("responseSHA256"), f"{label} response {case_id}")
        if set(a) != set(candidate_by_id) or set(b) != set(candidate_by_id):
            raise SystemExit(f"judge candidate coverage mismatch: {case_id}")
        consensus: list[str] = []
        for candidate_id, candidate in candidate_by_id.items():
            text = str(candidate.get("text", "")).strip()
            text_hash = text_sha256(text)
            if not text or candidate.get("sha256") != text_hash:
                raise SystemExit(f"invalid generator candidate: {case_id}/{candidate_id}")
            for label, assessment in (("judge A", a[candidate_id]), ("judge B", b[candidate_id])):
                if assessment.get("referenceSHA256") != text_hash:
                    raise SystemExit(f"{label} candidate hash mismatch: {case_id}/{candidate_id}")
            if a[candidate_id]["acceptAsReference"] and b[candidate_id]["acceptAsReference"]:
                consensus.append(candidate_id)

        selected: list[str] = []
        pair_similarity: float | None = None
        if len(consensus) >= 2:
            ranked_pairs = sorted(
                combinations(consensus, 2),
                key=lambda pair: (
                    jaccard(str(candidate_by_id[pair[0]]["text"]), str(candidate_by_id[pair[1]]["text"])),
                    pair,
                ),
            )
            chosen = set(ranked_pairs[0])
            selected = [candidate_id for candidate_id in candidate_by_id if candidate_id in chosen]
            pair_similarity = jaccard(
                str(candidate_by_id[selected[0]]["text"]),
                str(candidate_by_id[selected[1]]["text"]),
            )
            row = copy.deepcopy(source)
            row.update(
                {
                    "acceptedReferenceCandidateIDs": selected,
                    "claimEligible": True,
                    "referenceGeneratedByAI": True,
                    "references": [str(candidate_by_id[value]["text"]) for value in selected],
                    "reviewStatus": "automated-two-judge-consensus-v1",
                    "split": "heldout-automated",
                }
            )
            admitted_rows.append(row)
        else:
            failed_cases.append(case_id)
        decisions.append(
            {
                "caseID": case_id,
                "consensusAcceptedCandidateIDs": consensus,
                "selectedReferenceCandidateIDs": selected,
                "selectedPairCharacter3GramJaccard": pair_similarity,
                "status": "admitted" if selected else "insufficient-unanimous-candidates",
            }
        )

    suite_payload = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in admitted_rows
    )
    decision = {
        "schemaVersion": 1,
        "status": "passed" if not failed_cases else "failed",
        "policy": "two-blinded-distinct-family-judges-unanimous-perfect-consensus-v1",
        "selection": "lowest normalized character-3-gram Jaccard; candidate-ID tie break",
        "sourceSuiteSHA256": sha256(args.sources),
        "manifestSHA256": sha256(args.manifest),
        "generatorReportSHA256": generator_hash,
        "judgeReportASHA256": sha256(args.judge_report_a),
        "judgeReportBSHA256": sha256(args.judge_report_b),
        "generatorModel": generator_model,
        "generatorRevision": generator["generatorRevision"],
        "judgeModels": sorted(judge_models),
        "judgeRevisions": sorted(
            [str(judge_a["judgeRevision"]), str(judge_b["judgeRevision"])]
        ),
        "cases": len(source_rows),
        "admittedCases": len(admitted_rows),
        "outputSuiteSHA256": (
            hashlib.sha256(suite_payload.encode("utf-8")).hexdigest() if not failed_cases else None
        ),
        "failedCases": failed_cases,
        "results": decisions,
    }
    args.decision_report.parent.mkdir(parents=True, exist_ok=True)
    args.decision_report.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failed_cases:
        raise SystemExit(
            f"{len(failed_cases)} cases have fewer than two unanimous perfect candidates; "
            "no suite written"
        )
    args.output_suite.parent.mkdir(parents=True, exist_ok=True)
    args.output_suite.write_text(suite_payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "cases": len(admitted_rows),
                "references": len(admitted_rows) * 2,
                "suite": str(args.output_suite),
                "suiteSHA256": sha256(args.output_suite),
                "decisionReport": str(args.decision_report),
                "decisionReportSHA256": sha256(args.decision_report),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
