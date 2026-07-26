#!/usr/bin/env python3
"""Run Mimi's blinded consensus judge requests through authenticated Claude CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_synthetic_batch import request_contract

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
JUDGE_PIPELINE = "mimi-translation-judge-v1"
ALLOWED_AUXILIARY_MODELS = {"claude-haiku-4-5"}
WRAPPER = """Act only as Mimi's blinded English-Japanese translation judge.
Do not use tools, inspect files, or explain your work.
Assess every source independently and return only the supplied structured output.
Keep source_id and candidate_id values exactly unchanged.
Do not infer candidate origin and never reveal or simulate chain-of-thought.
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing file: {path}") from error


def exclusive_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(value)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing file: {path}") from error


def read_requests(path: Path) -> tuple[list[dict], dict]:
    contract = request_contract(path)
    parsed = []
    schema_hashes: dict[str, dict] = {}
    developer_prompts: set[str] = set()
    for request in rows(path):
        custom_id = str(request["custom_id"])
        body = request["body"]
        if body.get("metadata", {}).get("pipeline") != JUDGE_PIPELINE:
            raise SystemExit(f"{custom_id}: expected the Mimi judge pipeline")
        source = json.loads(body["input"][1]["content"])
        if set(source) != {
            "source_id",
            "source_language",
            "target_language",
            "domain",
            "source",
            "candidates",
        }:
            raise SystemExit(f"{custom_id}: invalid blinded judge input fields")
        if source["source_id"] != custom_id:
            raise SystemExit(f"{custom_id}: source_id mismatch")
        candidates = source["candidates"]
        if (
            not isinstance(candidates, list)
            or len(candidates) not in {2, 3, 4}
            or len({candidate["candidate_id"] for candidate in candidates})
            != len(candidates)
            or any(set(candidate) != {"candidate_id", "translation"} for candidate in candidates)
        ):
            raise SystemExit(f"{custom_id}: invalid blinded candidates")
        serialized = json.dumps(source, ensure_ascii=False)
        if any(
            forbidden in serialized
            for forbidden in (
                "candidate_origin",
                "licensed-reference",
                "reference_provenance",
                "teacher_model",
            )
        ):
            raise SystemExit(f"{custom_id}: candidate-origin leakage")
        schema = body["text"]["format"]["schema"]
        schema_hash = sha256_bytes(canonical_bytes(schema))
        schema_hashes[schema_hash] = schema
        developer_prompt = str(body["input"][0]["content"])
        developer_prompts.add(developer_prompt)
        parsed.append(
            {
                "custom_id": custom_id,
                "source": source,
                "characters": len(serialized),
            }
        )
    if len(schema_hashes) != 1 or len(developer_prompts) != 1:
        raise SystemExit("judge requests must share one prompt and one schema")
    return parsed, {
        **contract,
        "schema": next(iter(schema_hashes.values())),
        "schema_hash": next(iter(schema_hashes)),
        "developer_prompt": next(iter(developer_prompts)),
    }


def build_shards(
    request_rows: list[dict],
    maximum_items: int,
    maximum_characters: int,
) -> list[dict]:
    if maximum_items < 1 or maximum_characters < 1:
        raise SystemExit("shard limits must be positive")
    groups: list[list[dict]] = []
    current: list[dict] = []
    characters = 0
    for row in request_rows:
        if current and (
            len(current) >= maximum_items
            or characters + row["characters"] > maximum_characters
        ):
            groups.append(current)
            current = []
            characters = 0
        current.append(row)
        characters += row["characters"]
    if current:
        groups.append(current)
    return [
        {
            "index": index,
            "custom_ids": [row["custom_id"] for row in group],
            "items": len(group),
            "characters": sum(row["characters"] for row in group),
            "source_payload_sha256": sha256_bytes(
                canonical_bytes([row["source"] for row in group])
            ),
        }
        for index, group in enumerate(groups)
    ]


def expected_manifest(
    requests: Path,
    run_directory: Path,
    maximum_items: int,
    maximum_characters: int,
) -> tuple[dict, list[dict], dict]:
    request_rows, contract = read_requests(requests)
    shards = build_shards(request_rows, maximum_items, maximum_characters)
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "transport": "claude-cli-claude-ai-login",
            "request_path": str(requests.resolve()),
            "request_sha256": contract["request_sha256"],
            "request_count": contract["request_count"],
            "judge_model": contract["model"],
            "prompt_sha256": contract["prompt_sha256"],
            "structured_output_schema_sha256": contract["schema_hash"],
            "maximum_items": maximum_items,
            "maximum_characters": maximum_characters,
            "shard_count": len(shards),
            "run_directory": str(run_directory.resolve()),
            "candidate_origin_exposed": False,
            "reasoning_trace_stored": False,
            "shards": shards,
        },
        request_rows,
        contract,
    )


def load_or_create_manifest(
    requests: Path,
    run_directory: Path,
    maximum_items: int,
    maximum_characters: int,
) -> tuple[dict, list[dict], dict]:
    expected, request_rows, contract = expected_manifest(
        requests,
        run_directory,
        maximum_items,
        maximum_characters,
    )
    manifest_path = run_directory / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise SystemExit("Claude judge manifest differs from current requests/config")
    else:
        exclusive_json(manifest_path, expected)
    return expected, request_rows, contract


def batch_schema(item_schema: dict, count: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": item_schema,
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def validate_payload(payload: Any, source: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) != {"source_id", "assessments"}:
        raise ValueError("invalid Claude judge result fields")
    if payload["source_id"] != source["source_id"]:
        raise ValueError("Claude judge source_id mismatch")
    assessments = payload["assessments"]
    expected_ids = {
        candidate["candidate_id"] for candidate in source["candidates"]
    }
    if not isinstance(assessments, list) or len(assessments) != len(expected_ids):
        raise ValueError("Claude judge assessment count mismatch")
    found_ids = {
        assessment.get("candidate_id")
        for assessment in assessments
        if isinstance(assessment, dict)
    }
    if found_ids != expected_ids:
        raise ValueError(
            "Claude judge candidate coverage mismatch: "
            f"expected={sorted(expected_ids)} found={sorted(str(value) for value in found_ids)}"
        )
    return payload


def verified_primary_model_usage(envelope: dict, requested_model: str) -> dict:
    """Prove that Claude CLI actually used the exact requested judge model.

    Claude CLI may make a small auxiliary Haiku request for its own bookkeeping.
    That request is not a translation judgment. Any other model identity is
    rejected so an overload fallback cannot silently enter the evidence.
    """
    model_usage = envelope.get("modelUsage")
    if not isinstance(model_usage, dict) or not model_usage:
        raise ValueError("Claude result has no modelUsage evidence")
    matching: list[tuple[str, dict]] = []
    unexpected: list[str] = []
    auxiliary: list[str] = []
    for usage_key, raw in model_usage.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Claude modelUsage entry is malformed: {usage_key}")
        canonical_model = str(raw.get("canonicalModel", "")).strip()
        output_tokens = raw.get("outputTokens")
        if (
            not canonical_model
            or isinstance(output_tokens, bool)
            or not isinstance(output_tokens, int)
            or output_tokens < 0
        ):
            raise ValueError(f"Claude modelUsage identity is incomplete: {usage_key}")
        if usage_key == requested_model and canonical_model == requested_model:
            matching.append((usage_key, raw))
        elif canonical_model in ALLOWED_AUXILIARY_MODELS:
            auxiliary.append(canonical_model)
        elif output_tokens > 0:
            unexpected.append(f"{usage_key}->{canonical_model}")
    if len(matching) != 1:
        raise ValueError(
            f"requested Claude model was not proven exactly once: {requested_model}"
        )
    if unexpected:
        raise ValueError(
            "unexpected non-auxiliary Claude model usage: " + ", ".join(unexpected)
        )
    usage_key, usage = matching[0]
    if int(usage["outputTokens"]) <= 0:
        raise ValueError(f"requested Claude model produced no output: {requested_model}")
    return {
        "usage_key": usage_key,
        "canonical_model": str(usage["canonicalModel"]),
        "provider": usage.get("provider"),
        "output_tokens": int(usage["outputTokens"]),
        "auxiliary_models": sorted(set(auxiliary)),
    }


def shard_paths(run_directory: Path, index: int) -> tuple[Path, Path]:
    stem = f"{index:05d}"
    return (
        run_directory / "shards" / f"{stem}.results.jsonl",
        run_directory / "shards" / f"{stem}.metadata.json",
    )


def validate_shard(path: Path, expected_ids: list[str]) -> list[dict]:
    values = rows(path)
    if [str(row.get("custom_id")) for row in values] != expected_ids:
        raise ValueError("Claude shard IDs or order do not match")
    return values


def run_shard(
    manifest: dict,
    request_rows: list[dict],
    contract: dict,
    run_directory: Path,
    index: int,
    executable: str,
) -> dict:
    shard = manifest["shards"][index]
    expected_ids = [str(value) for value in shard["custom_ids"]]
    result_path, metadata_path = shard_paths(run_directory, index)
    if result_path.exists() or metadata_path.exists():
        if not result_path.is_file() or not metadata_path.is_file():
            raise SystemExit(f"Claude shard {index} is incomplete")
        validate_shard(result_path, expected_ids)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("result_sha256") != sha256_path(result_path):
            raise SystemExit(f"Claude shard {index} hash mismatch")
        if (
            metadata.get("verified_primary_model", {}).get("canonical_model")
            != manifest["judge_model"]
        ):
            raise SystemExit(f"Claude shard {index} model identity is unverified")
        return {"shard": index, "status": "already-complete"}

    by_id = {row["custom_id"]: row for row in request_rows}
    source_rows = [by_id[custom_id]["source"] for custom_id in expected_ids]
    if sha256_bytes(canonical_bytes(source_rows)) != shard["source_payload_sha256"]:
        raise SystemExit(f"Claude shard {index} source hash mismatch")
    prompt = (
        WRAPPER
        + "\n"
        + contract["developer_prompt"].strip()
        + "\n\nBlinded judge inputs:\n"
        + json.dumps(source_rows, ensure_ascii=False, separators=(",", ":"))
    )
    schema = batch_schema(contract["schema"], len(source_rows))
    command = [
        executable,
        "-p",
        "--safe-mode",
        "--no-session-persistence",
        "--tools",
        "",
        "--model",
        str(manifest["judge_model"]),
        "--effort",
        "low",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        prompt,
    ]
    started = utc_now()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"Claude judge shard {index} failed: {(completed.stderr or completed.stdout)[-2000:]}"
        )
    try:
        envelope = json.loads(completed.stdout)
        verified_model = verified_primary_model_usage(
            envelope, str(manifest["judge_model"])
        )
        structured = envelope["structured_output"]
        values = structured["results"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Claude judge shard {index} returned invalid JSON: {error}") from error
    if not isinstance(values, list) or len(values) != len(source_rows):
        raise SystemExit(f"Claude judge shard {index} result count mismatch")
    by_source: dict[str, dict] = {}
    try:
        for value in values:
            source_id = str(value.get("source_id", ""))
            if source_id in by_source or source_id not in expected_ids:
                raise ValueError("duplicate or unknown Claude source_id")
            source = by_id[source_id]["source"]
            by_source[source_id] = validate_payload(value, source)
    except (AttributeError, ValueError) as error:
        raise SystemExit(f"Claude judge shard {index}: {error}") from error
    if set(by_source) != set(expected_ids):
        raise SystemExit(f"Claude judge shard {index} is missing results")

    collected = []
    for position, source_id in enumerate(expected_ids):
        payload = by_source[source_id]
        collected.append(
            {
                "custom_id": source_id,
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": (
                            f"claude-cli-{index:05d}-{position:02d}-"
                            f"{sha256_bytes(canonical_bytes(payload))[:16]}"
                        ),
                        "model": manifest["judge_model"],
                        "system_fingerprint": None,
                        "output_text": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                },
            }
        )
    result_bytes = b"".join(canonical_bytes(row) + b"\n" for row in collected)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "transport": "claude-cli-claude-ai-login",
        "shard": index,
        "custom_ids": expected_ids,
        "request_sha256": manifest["request_sha256"],
        "source_payload_sha256": shard["source_payload_sha256"],
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "result_sha256": sha256_bytes(result_bytes),
        "result_count": len(collected),
        "judge_model": manifest["judge_model"],
        "verified_primary_model": verified_model,
        "reasoning_trace_stored": False,
        "candidate_origin_exposed": False,
        "duration_api_ms": envelope.get("duration_api_ms"),
        "duration_ms": envelope.get("duration_ms"),
        "total_cost_usd": envelope.get("total_cost_usd"),
        "usage": envelope.get("usage"),
        "model_usage": envelope.get("modelUsage"),
        "started_at": started,
        "finished_at": utc_now(),
    }
    exclusive_bytes(result_path, result_bytes)
    exclusive_json(metadata_path, metadata)
    return {
        "shard": index,
        "status": "completed",
        "results": len(collected),
        "cost_usd": envelope.get("total_cost_usd"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", type=Path)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-items", type=int, default=8)
    parser.add_argument("--maximum-characters", type=int, default=16_000)
    parser.add_argument("--claude-executable", default="claude")
    parser.add_argument("--maximum-shards", type=int)
    args = parser.parse_args()

    manifest, request_rows, contract = load_or_create_manifest(
        args.requests,
        args.run_directory,
        args.maximum_items,
        args.maximum_characters,
    )
    executable = shutil.which(args.claude_executable) or args.claude_executable
    completed_now = 0
    for index in range(manifest["shard_count"]):
        result = run_shard(
            manifest,
            request_rows,
            contract,
            args.run_directory,
            index,
            executable,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if result["status"] == "completed":
            completed_now += 1
            if (
                args.maximum_shards is not None
                and completed_now >= args.maximum_shards
            ):
                break

    result_rows: dict[str, dict] = {}
    missing = []
    for shard in manifest["shards"]:
        result_path, _ = shard_paths(args.run_directory, int(shard["index"]))
        if not result_path.is_file():
            missing.append(int(shard["index"]))
            continue
        for row in validate_shard(
            result_path,
            [str(value) for value in shard["custom_ids"]],
        ):
            result_rows[str(row["custom_id"])] = row
    if missing:
        print(
            json.dumps(
                {
                    "status": "partial",
                    "complete_shards": manifest["shard_count"] - len(missing),
                    "remaining_shards": len(missing),
                },
                sort_keys=True,
            )
        )
        return
    if args.output.exists():
        existing = rows(args.output)
        if [row["custom_id"] for row in existing] != contract["custom_ids"]:
            raise SystemExit("existing Claude output does not match request IDs")
        print(json.dumps({"status": "already-collected", "output": str(args.output)}))
        return
    output_bytes = b"".join(
        canonical_bytes(result_rows[custom_id]) + b"\n"
        for custom_id in contract["custom_ids"]
    )
    exclusive_bytes(args.output, output_bytes)
    total_cost = 0.0
    for shard in manifest["shards"]:
        _, metadata_path = shard_paths(args.run_directory, int(shard["index"]))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        total_cost += float(metadata.get("total_cost_usd") or 0.0)
    summary = {
        "status": "collected",
        "sources": len(result_rows),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_bytes(output_bytes),
        "judge_model": manifest["judge_model"],
        "total_cost_usd": total_cost,
        "candidate_origin_exposed": False,
        "reasoning_trace_stored": False,
    }
    exclusive_json(
        args.output.with_suffix(args.output.suffix + ".manifest.json"),
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
