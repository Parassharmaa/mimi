#!/usr/bin/env python3
"""Convert a complete GPT reference Batch response into hash-bound candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from typed_critical_token_policy import mask_protected, normalize
from validate_benchmark_suite import normalized


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def rows(path: Path) -> list[dict]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values or not all(isinstance(value, dict) for value in values):
        raise SystemExit(f"expected non-empty JSONL objects: {path}")
    return values


def index(values: list[dict], key: str, label: str) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for value in values:
        identifier = str(value.get(key, "")).strip()
        if not identifier or identifier in output:
            raise SystemExit(f"{label} has an empty or duplicate ID: {identifier}")
        output[identifier] = value
    return output


def visible_reasoning_trace(value: object) -> bool:
    if isinstance(value, dict):
        if value.get("type") == "reasoning" and value.get("summary"):
            return True
        for key, child in value.items():
            if key in {"summary_text", "encrypted_content"} and child:
                return True
            if visible_reasoning_trace(child):
                return True
    elif isinstance(value, list):
        return any(visible_reasoning_trace(child) for child in value)
    return False


def output_text(body: dict) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    found: list[str] = []
    for item in body.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    found.append(text)
    if len(found) != 1:
        raise ValueError("response does not contain exactly one final output text")
    return found[0]


def japanese_text(value: str) -> bool:
    return any(
        0x3040 <= ord(character) <= 0x30FF
        or 0x3400 <= ord(character) <= 0x4DBF
        or 0x4E00 <= ord(character) <= 0x9FFF
        for character in unicodedata.normalize("NFKC", value)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("model_plan", type=Path)
    parser.add_argument("batch_output", type=Path)
    parser.add_argument("generator_report", type=Path)
    parser.add_argument("candidate_queue", type=Path)
    args = parser.parse_args()
    for output in (args.generator_report, args.candidate_queue):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")

    source_rows = rows(args.sources)
    request_rows = rows(args.requests)
    response_rows = rows(args.batch_output)
    sources = index(source_rows, "id", "source suite")
    requests = index(request_rows, "custom_id", "request file")
    responses = index(response_rows, "custom_id", "batch output")
    if set(sources) != set(requests) or set(sources) != set(responses):
        raise SystemExit("sources, requests, and Batch responses do not have exact ID coverage")

    plan = json.loads(args.model_plan.read_text(encoding="utf-8"))
    generator = plan.get("generator", {})
    if (
        plan.get("suiteSHA256") != sha256(args.sources)
        or generator.get("store") is not False
        or generator.get("reasoningSummaryRequested") is not False
        or generator.get("minimumCandidatesPerCase") != 3
    ):
        raise SystemExit("model plan is not bound to the frozen source suite")
    prompt_hashes = {
        str(request.get("body", {}).get("metadata", {}).get("prompt_sha256", ""))
        for request in request_rows
    }
    if len(prompt_hashes) != 1 or any(len(value) != 64 for value in prompt_hashes):
        raise SystemExit("request file does not contain one pinned prompt hash")
    prompt_hash = next(iter(prompt_hashes))

    report_results: list[dict] = []
    queue_rows: list[dict] = []
    response_ids: set[str] = set()
    system_fingerprints: set[str] = set()
    for case_id in sources:
        source = sources[case_id]
        request = requests[case_id]
        body_request = request.get("body")
        try:
            request_input = json.loads(body_request["input"][1]["content"])
            developer_prompt = body_request["input"][0]["content"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid generator request: {case_id}") from error
        expected_input = {
            "source_id": case_id,
            "source_language": source["sourceLanguage"],
            "target_language": source["targetLanguage"],
            "domain": source["domain"],
            "source": source["source"],
        }
        if (
            request.get("method") != "POST"
            or request.get("url") != "/v1/responses"
            or request_input != expected_input
            or body_request.get("model") != generator.get("model")
            or body_request.get("store") is not False
            or body_request.get("reasoning") != {"effort": generator.get("reasoningEffort")}
            or body_request.get("metadata", {}).get("pipeline")
            != "mimi-benchmark-reference-generator-v1"
            or hashlib.sha256(str(developer_prompt).encode("utf-8")).hexdigest() != prompt_hash
        ):
            raise SystemExit(f"generator request is not bound to the frozen source: {case_id}")
        batch_row = responses[case_id]
        if batch_row.get("error") not in (None, {}):
            raise SystemExit(f"Batch response has an error: {case_id}")
        response = batch_row.get("response")
        if not isinstance(response, dict) or response.get("status_code") != 200:
            raise SystemExit(f"Batch response is not HTTP 200: {case_id}")
        body = response.get("body")
        if not isinstance(body, dict) or body.get("status") not in (None, "completed"):
            raise SystemExit(f"incomplete Responses API body: {case_id}")
        if visible_reasoning_trace(body):
            raise SystemExit(f"visible or encrypted reasoning trace found: {case_id}")
        response_id = str(body.get("id", "")).strip()
        if not response_id or response_id in response_ids:
            raise SystemExit(f"empty or duplicate response ID: {case_id}")
        response_ids.add(response_id)
        if body.get("model") != generator.get("model"):
            raise SystemExit(f"generator model revision mismatch: {case_id}")
        fingerprint = str(body.get("system_fingerprint") or "").strip()
        if fingerprint:
            system_fingerprints.add(fingerprint)
        try:
            payload = json.loads(output_text(body))
        except (ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid Structured Output: {case_id}: {error}") from error
        if not isinstance(payload, dict) or set(payload) != {"source_id", "translations"}:
            raise SystemExit(f"unexpected reference candidate schema: {case_id}")
        translations = payload.get("translations")
        if payload.get("source_id") != case_id or not isinstance(translations, list) or len(translations) != 3:
            raise SystemExit(f"invalid candidate coverage: {case_id}")
        translations = [str(value).strip() for value in translations]
        if (
            any(not value for value in translations)
            or len({normalized(value) for value in translations}) != 3
        ):
            raise SystemExit(f"empty or duplicate reference candidates: {case_id}")
        target_language = str(source["targetLanguage"])
        for translation in translations:
            if target_language == "ja-JP" and not japanese_text(translation):
                raise SystemExit(f"candidate does not contain Japanese output: {case_id}")
            if target_language == "en-US" and not any(character.isascii() and character.isalpha() for character in translation):
                raise SystemExit(f"candidate does not contain English output: {case_id}")
            _, source_protected = mask_protected(normalize(str(source["source"])))
            _, output_protected = mask_protected(normalize(translation))
            if source_protected != output_protected:
                raise SystemExit(f"candidate changed a URL, placeholder, or markup token: {case_id}")
        candidates = [
            {
                "candidateID": f"{case_id}:candidate-{index_value}",
                "text": translation,
                "sha256": text_sha256(translation),
            }
            for index_value, translation in enumerate(translations, start=1)
        ]
        report_results.append(
            {
                "caseID": case_id,
                "sourceSHA256": text_sha256(str(source["source"])),
                "requestSHA256": canonical_sha256(request),
                "responseSHA256": canonical_sha256(batch_row),
                "responseID": response_id,
                "candidates": candidates,
            }
        )
        queue_rows.append(
            {
                "caseID": case_id,
                "sourceLanguage": source["sourceLanguage"],
                "targetLanguage": source["targetLanguage"],
                "domain": source["domain"],
                "source": source["source"],
                "sourceSHA256": text_sha256(str(source["source"])),
                "candidates": candidates,
                "generatorModelRedactedForJudges": True,
            }
        )

    report = {
        "schemaVersion": 1,
        "purpose": "benchmark-reference-generation",
        "generatorModel": generator["model"],
        "generatorModelFamily": generator["family"],
        "generatorRevision": generator["revision"],
        "promptSHA256": prompt_hash,
        "reasoningTracesStored": False,
        "store": False,
        "sourceSuiteSHA256": sha256(args.sources),
        "requestFileSHA256": sha256(args.requests),
        "rawBatchOutputSHA256": sha256(args.batch_output),
        "systemFingerprints": sorted(system_fingerprints),
        "results": report_results,
    }
    args.generator_report.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_queue.parent.mkdir(parents=True, exist_ok=True)
    args.generator_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.candidate_queue.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in queue_rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(report_results),
                "candidates": len(queue_rows) * 3,
                "generatorReport": str(args.generator_report),
                "generatorReportSHA256": sha256(args.generator_report),
                "candidateQueue": str(args.candidate_queue),
                "candidateQueueSHA256": sha256(args.candidate_queue),
                "reasoningTracesStored": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
