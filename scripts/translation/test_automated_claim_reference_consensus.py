#!/usr/bin/env python3
"""End-to-end offline contract test for automated reference consensus."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COLLECT_CANDIDATES = ROOT / "scripts/translation/collect_automated_claim_reference_candidates.py"
PREPARE_JUDGES = ROOT / "scripts/translation/prepare_automated_claim_reference_judge_batch.py"
COLLECT_JUDGES = ROOT / "scripts/translation/collect_automated_claim_reference_judgments.py"
ASSEMBLE = ROOT / "scripts/translation/assemble_automated_claim_reference_suite.py"
AUDIT = ROOT / "scripts/translation/audit_automated_claim_reference_structures.py"
RUNNER = ROOT / "scripts/translation/run_synthetic_batch.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def run(*arguments: str, success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["python3", *arguments], cwd=ROOT, text=True, capture_output=True, check=False
    )
    if success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0, result.stdout
    return result


def response(case_id: str, model: str, payload: dict, number: int) -> dict:
    return {
        "custom_id": case_id,
        "response": {
            "status_code": 200,
            "body": {
                "id": f"resp-{model}-{number}",
                "status": "completed",
                "model": model,
                "system_fingerprint": "fixture-fingerprint",
                "output_text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        },
        "error": None,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-reference-consensus-") as temporary:
        work = Path(temporary)
        sources = work / "sources.jsonl"
        source_rows = [
            {
                "id": "en-ja-1",
                "documentID": "document-en-ja-1",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "macos-and-technical-ui",
                "source": "Open case M-7 at 5 in {room}.",
                "references": [],
                "acceptedReferenceCandidateIDs": [],
                "claimEligible": False,
                "referenceGeneratedByAI": None,
                "reviewStatus": "references-pending",
                "split": "heldout-automated-source-draft",
                "sourceGeneratedByAI": False,
                "publicBenchmarkOrigin": False,
                "paraphraseOfExistingMaterial": False,
                "sourceCreatedAt": "2026-07-20",
                "license": "Project-owned",
                "provenance": "fixture",
            },
            {
                "id": "ja-en-1",
                "documentID": "document-ja-en-1",
                "sourceLanguage": "ja-JP",
                "targetLanguage": "en-US",
                "domain": "macos-and-technical-ui",
                "source": "案件R-8を6時に{room}で開いてください。",
                "references": [],
                "acceptedReferenceCandidateIDs": [],
                "claimEligible": False,
                "referenceGeneratedByAI": None,
                "reviewStatus": "references-pending",
                "split": "heldout-automated-source-draft",
                "sourceGeneratedByAI": False,
                "publicBenchmarkOrigin": False,
                "paraphraseOfExistingMaterial": False,
                "sourceCreatedAt": "2026-07-20",
                "license": "Project-owned",
                "provenance": "fixture",
            },
        ]
        write_jsonl(sources, source_rows)
        manifest = work / "manifest.json"
        write_json(
            manifest,
            {
                "schemaVersion": 1,
                "suiteID": "automated-reference-consensus-fixture",
                "frozenSources": {
                    "path": str(sources),
                    "sha256": sha256(sources),
                    "cases": len(source_rows),
                    "claimEligible": False,
                },
                "exactCasesPerDirection": 1,
                "directions": ["en-US>ja-JP", "ja-JP>en-US"],
                "domains": {"macos-and-technical-ui": 1.0},
                "referencePolicy": {
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
                },
            },
        )
        plan = work / "plan.json"
        plan_value = {
            "schemaVersion": 1,
            "suiteSHA256": sha256(sources),
            "generator": {
                "role": "final-translation-candidate-generator",
                "model": "gpt-5.6-sol",
                "family": "gpt-5.6",
                "revision": "gpt-5.6-sol",
                "reasoningEffort": "none",
                "reasoningSummaryRequested": False,
                "minimumCandidatesPerCase": 3,
                "store": False,
            },
            "judges": [
                {
                    "role": "reference-judge-a",
                    "model": "gpt-4o-2024-08-06",
                    "family": "gpt-4o",
                    "revision": "gpt-4o-2024-08-06",
                    "reasoningEffort": None,
                    "reasoningSummaryRequested": False,
                    "store": False,
                },
                {
                    "role": "reference-judge-b",
                    "model": "gpt-4.1-2025-04-14",
                    "family": "gpt-4.1",
                    "revision": "gpt-4.1-2025-04-14",
                    "reasoningEffort": None,
                    "reasoningSummaryRequested": False,
                    "store": False,
                },
            ],
        }
        write_json(plan, plan_value)
        generator_prompt = "Return three final translations only. Do not return reasoning."
        prompt_hash = hashlib.sha256(generator_prompt.encode()).hexdigest()
        generator_requests = work / "generator-requests.jsonl"
        request_rows = []
        for source in source_rows:
            request_rows.append(
                {
                    "custom_id": source["id"],
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": "gpt-5.6-sol",
                        "store": False,
                        "reasoning": {"effort": "none"},
                        "input": [
                            {"role": "developer", "content": generator_prompt},
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "source_id": source["id"],
                                        "source_language": source["sourceLanguage"],
                                        "target_language": source["targetLanguage"],
                                        "domain": source["domain"],
                                        "source": source["source"],
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            },
                        ],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "fixture",
                                "strict": True,
                                "schema": {"type": "object"},
                            }
                        },
                        "max_output_tokens": 768,
                        "metadata": {
                            "pipeline": "mimi-benchmark-reference-generator-v1",
                            "prompt_sha256": prompt_hash,
                        },
                    },
                }
            )
        write_jsonl(generator_requests, request_rows)
        run(str(RUNNER), "validate", str(generator_requests))

        translations = {
            "en-ja-1": [
                "5時に{room}で案件M-7を開いてください。",
                "{room}で案件M-7を5時に開いて。",
                "5時になったら{room}でM-7の案件を開きます。",
            ],
            "ja-en-1": [
                "Open case R-8 at 6 in {room}.",
                "At 6, please open case R-8 in {room}.",
                "Please open R-8, the case, in {room} at 6.",
            ],
        }
        generator_output = work / "generator-output.jsonl"
        generator_responses = [
            response(
                source["id"],
                "gpt-5.6-sol",
                {"source_id": source["id"], "translations": translations[source["id"]]},
                number,
            )
            for number, source in enumerate(source_rows, start=1)
        ]
        write_jsonl(generator_output, generator_responses)
        generator_report = work / "generator-report.json"
        candidate_queue = work / "candidate-queue.jsonl"
        run(
            str(COLLECT_CANDIDATES),
            str(sources),
            str(generator_requests),
            str(plan),
            str(generator_output),
            str(generator_report),
            str(candidate_queue),
        )

        changed_token_output = work / "generator-changed-token-output.jsonl"
        changed_token_responses = copy.deepcopy(generator_responses)
        changed_payload = json.loads(
            changed_token_responses[0]["response"]["body"]["output_text"]
        )
        changed_payload["translations"][0] = changed_payload["translations"][0].replace(
            "{room}", "{other}"
        )
        changed_token_responses[0]["response"]["body"]["output_text"] = json.dumps(
            changed_payload, ensure_ascii=False, sort_keys=True
        )
        write_jsonl(changed_token_output, changed_token_responses)
        rejected = run(
            str(COLLECT_CANDIDATES),
            str(sources),
            str(generator_requests),
            str(plan),
            str(changed_token_output),
            str(work / "changed-token-generator.json"),
            str(work / "changed-token-queue.jsonl"),
            success=False,
        )
        assert "changed a URL, placeholder, or markup token" in rejected.stderr

        trace_output = work / "generator-trace-output.jsonl"
        traced = copy.deepcopy(generator_responses)
        traced[0]["response"]["body"]["output"] = [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "trace"}]}
        ]
        write_jsonl(trace_output, traced)
        rejected = run(
            str(COLLECT_CANDIDATES),
            str(sources),
            str(generator_requests),
            str(plan),
            str(trace_output),
            str(work / "rejected-generator.json"),
            str(work / "rejected-queue.jsonl"),
            success=False,
        )
        assert "reasoning trace" in rejected.stderr

        judge_prompt = work / "judge-prompt.txt"
        judge_prompt.write_text(
            "Blindly score three translations. Return structured scores only; no reasoning.",
            encoding="utf-8",
        )
        judge_reports: list[Path] = []
        for role, model in (
            ("reference-judge-a", "gpt-4o-2024-08-06"),
            ("reference-judge-b", "gpt-4.1-2025-04-14"),
        ):
            judge_requests = work / f"{role}-requests.jsonl"
            run(
                str(PREPARE_JUDGES),
                str(sources),
                str(generator_report),
                str(plan),
                str(judge_prompt),
                role,
                str(judge_requests),
            )
            run(str(RUNNER), "validate", str(judge_requests))
            requests = [json.loads(line) for line in judge_requests.read_text().splitlines()]
            judge_output = work / f"{role}-output.jsonl"
            responses = []
            for number, request in enumerate(requests, start=1):
                request_input = json.loads(request["body"]["input"][1]["content"])
                assessments = []
                for candidate in request_input["candidates"]:
                    accepted = not candidate["candidate_id"].endswith("candidate-3")
                    assessments.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "adequacy": 4 if accepted else 3,
                            "fluency": 4,
                            "terminology": 4,
                            "protected_tokens_preserved": True,
                            "critical_error": False,
                            "error_tags": [] if accepted else ["addition"],
                            "accept_as_reference": accepted,
                        }
                    )
                responses.append(
                    response(
                        request["custom_id"],
                        model,
                        {"source_id": request["custom_id"], "assessments": assessments},
                        number,
                    )
                )
            write_jsonl(judge_output, responses)
            judge_report = work / f"{role}-report.json"
            run(
                str(COLLECT_JUDGES),
                str(sources),
                str(generator_report),
                str(judge_requests),
                str(plan),
                role,
                str(judge_output),
                str(judge_report),
            )
            judge_reports.append(judge_report)

        suite = work / "suite.jsonl"
        decisions = work / "decisions.json"
        run(
            str(ASSEMBLE),
            str(sources),
            str(generator_report),
            str(judge_reports[0]),
            str(judge_reports[1]),
            str(suite),
            str(decisions),
            "--manifest",
            str(manifest),
        )
        frozen = [json.loads(line) for line in suite.read_text().splitlines()]
        assert len(frozen) == 2
        assert all(len(value["references"]) == 2 and value["claimEligible"] for value in frozen)
        audit = work / "audit.json"
        run(str(AUDIT), str(suite), str(judge_reports[0]), str(judge_reports[1]), str(audit))
        assert json.loads(audit.read_text())["status"] == "passed"

        changed_number_suite = work / "changed-number-suite.jsonl"
        changed_number_rows = copy.deepcopy(frozen)
        changed_number_rows[0]["references"][0] = changed_number_rows[0]["references"][0].replace(
            "5時", "9時"
        )
        write_jsonl(changed_number_suite, changed_number_rows)
        changed_number_judges: list[Path] = []
        changed_candidate_id = changed_number_rows[0]["acceptedReferenceCandidateIDs"][0]
        changed_reference_hash = hashlib.sha256(
            changed_number_rows[0]["references"][0].encode("utf-8")
        ).hexdigest()
        for index_value, judge_report in enumerate(judge_reports):
            changed_judge = json.loads(judge_report.read_text(encoding="utf-8"))
            result = next(
                value for value in changed_judge["results"] if value["caseID"] == "en-ja-1"
            )
            assessment = next(
                value
                for value in result["assessments"]
                if value["candidateID"] == changed_candidate_id
            )
            assessment["referenceSHA256"] = changed_reference_hash
            path = work / f"changed-number-judge-{index_value}.json"
            write_json(path, changed_judge)
            changed_number_judges.append(path)
        changed_number_audit = work / "changed-number-audit.json"
        rejected = run(
            str(AUDIT),
            str(changed_number_suite),
            str(changed_number_judges[0]),
            str(changed_number_judges[1]),
            str(changed_number_audit),
            success=False,
        )
        assert "deterministic reference audit failed" in rejected.stderr
        changed_number_report = json.loads(changed_number_audit.read_text(encoding="utf-8"))
        changed_result = next(
            value for value in changed_number_report["results"] if value["caseID"] == "en-ja-1"
        )
        assert changed_result["checks"]["numbers"] is False
        assert changed_result["criticalError"] is True

        bad_judge = work / "bad-judge.json"
        bad_value = json.loads(judge_reports[1].read_text())
        assessment = bad_value["results"][0]["assessments"][0]
        assessment.update(
            {
                "adequacy": 3,
                "criticalError": True,
                "errorTags": ["omission"],
                "acceptAsReference": False,
            }
        )
        write_json(bad_judge, bad_value)
        rejected_suite = work / "rejected-suite.jsonl"
        rejection_report = work / "rejection-decisions.json"
        rejected = run(
            str(ASSEMBLE),
            str(sources),
            str(generator_report),
            str(judge_reports[0]),
            str(bad_judge),
            str(rejected_suite),
            str(rejection_report),
            "--manifest",
            str(manifest),
            success=False,
        )
        assert "fewer than two" in rejected.stderr
        assert not rejected_suite.exists()
        assert json.loads(rejection_report.read_text())["status"] == "failed"

    print("Mimi automated reference consensus contracts passed.")


if __name__ == "__main__":
    main()
