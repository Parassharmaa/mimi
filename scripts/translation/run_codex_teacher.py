#!/usr/bin/env python3
"""Run Mimi's source-only translation teacher through authenticated Codex CLI.

This is a resumable alternative transport for the already sealed Responses API
request corpus. It deliberately keeps the original request file authoritative,
never passes licensed references or student hypotheses to the teacher, and
collects Batch-compatible JSONL for the existing independent filtering pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_synthetic_batch import request_contract


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_VERSION = 1
SOURCE_KEYS = {
    "source_id",
    "source_language",
    "target_language",
    "domain",
    "source",
}
STYLES = {
    "natural-spoken",
    "concise-caption",
    "meaning-conservative",
}
RISK_TAGS = {
    "ambiguity",
    "register",
    "terminology",
    "omission",
    "addition",
    "protected-token",
}
REGISTER_VALUES = {"casual", "neutral", "polite", "technical"}
TOKEN_USAGE_RE = re.compile(r"tokens used\s*[\r\n]+([\d,]+)", re.IGNORECASE)
CODEX_WRAPPER = """You are acting only as Mimi's translation-data teacher.
Do not use tools, inspect files, or explain your work.
Translate every source object independently into its requested target language.
Keep every source_id exactly unchanged and emit every input exactly once.
Return only JSON matching the supplied output schema.
Never reveal, request, reconstruct, or simulate private chain-of-thought.
The inputs below are source-only training candidates with no human references or student outputs.
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json_object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error


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


def read_request_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = request_contract(path)
    rows: list[dict[str, Any]] = []
    schemas: dict[str, dict[str, Any]] = {}
    developer_prompts: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        request = json_object(json.loads(line), f"request line {line_number}")
        custom_id = str(request["custom_id"])
        body = json_object(request["body"], f"request body {custom_id}")
        messages = body["input"]
        developer_prompt = str(messages[0]["content"])
        developer_prompts.add(developer_prompt)
        try:
            source = json_object(
                json.loads(messages[1]["content"]),
                f"source payload {custom_id}",
            )
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            raise SystemExit(f"invalid source payload for {custom_id}: {error}") from error
        if set(source) != SOURCE_KEYS:
            raise SystemExit(
                f"{custom_id}: Codex teacher accepts only the sealed source-only fields"
            )
        if source["source_id"] != custom_id:
            raise SystemExit(f"{custom_id}: source_id does not match custom_id")
        if any(
            forbidden in source
            for forbidden in (
                "reference",
                "reference_translation",
                "student_hypothesis",
                "student_output",
            )
        ):
            raise SystemExit(f"{custom_id}: forbidden teacher leakage field")
        text_format = json_object(body["text"]["format"], f"schema {custom_id}")
        schema = json_object(text_format["schema"], f"schema body {custom_id}")
        schema_hash = sha256_bytes(canonical_bytes(schema))
        schemas[schema_hash] = schema
        rows.append(
            {
                "custom_id": custom_id,
                "source": source,
                "source_characters": len(str(source["source"])),
                "developer_prompt": developer_prompt,
                "model": str(body["model"]),
                "schema_hash": schema_hash,
            }
        )
    if len(schemas) != 1:
        raise SystemExit("request corpus must use exactly one Structured Outputs schema")
    if len(developer_prompts) != 1:
        raise SystemExit("request corpus must use exactly one teacher prompt")
    if len(rows) != contract["request_count"]:
        raise SystemExit("parsed request count does not match sealed request contract")
    return rows, {
        **contract,
        "schema": next(iter(schemas.values())),
        "schema_hash": next(iter(schemas)),
        "developer_prompt": next(iter(developer_prompts)),
    }


def build_shards(
    rows: list[dict[str, Any]],
    maximum_items: int,
    maximum_source_characters: int,
) -> list[dict[str, Any]]:
    if maximum_items < 1:
        raise SystemExit("--maximum-items must be positive")
    if maximum_source_characters < 1:
        raise SystemExit("--maximum-source-characters must be positive")
    shards: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_characters = 0
    for row in rows:
        characters = int(row["source_characters"])
        would_overflow = current and (
            len(current) >= maximum_items
            or current_characters + characters > maximum_source_characters
        )
        if would_overflow:
            shards.append(current)
            current = []
            current_characters = 0
        current.append(row)
        current_characters += characters
    if current:
        shards.append(current)

    result: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        sources = [row["source"] for row in shard]
        result.append(
            {
                "index": index,
                "item_count": len(shard),
                "source_characters": sum(
                    int(row["source_characters"]) for row in shard
                ),
                "custom_ids": [str(row["custom_id"]) for row in shard],
                "source_payload_sha256": sha256_bytes(canonical_bytes(sources)),
            }
        )
    return result


def expected_manifest(
    request_path: Path,
    run_directory: Path,
    maximum_items: int,
    maximum_source_characters: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rows, contract = read_request_rows(request_path)
    shards = build_shards(rows, maximum_items, maximum_source_characters)
    manifest = {
        "schema_version": PIPELINE_VERSION,
        "transport": "codex-cli-chatgpt-login",
        "request_path": str(request_path.resolve()),
        "request_sha256": contract["request_sha256"],
        "request_count": contract["request_count"],
        "request_model": contract["model"],
        "teacher_model": contract["model"],
        "reasoning_effort": "none",
        "source_only": True,
        "reasoning_traces_stored": False,
        "developer_prompt_sha256": contract["prompt_sha256"],
        "structured_output_schema_sha256": contract["schema_hash"],
        "maximum_items": maximum_items,
        "maximum_source_characters": maximum_source_characters,
        "shard_count": len(shards),
        "run_directory": str(run_directory.resolve()),
        "shards": shards,
    }
    return manifest, rows, contract


def prepare_command(arguments: argparse.Namespace) -> None:
    manifest, _, _ = expected_manifest(
        arguments.requests,
        arguments.run_directory,
        arguments.maximum_items,
        arguments.maximum_source_characters,
    )
    manifest_path = arguments.run_directory / "manifest.json"
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing != manifest:
            raise SystemExit(
                "existing Codex teacher manifest differs; use a new run directory"
            )
        print(
            json.dumps(
                {
                    "status": "already-prepared",
                    "request_count": manifest["request_count"],
                    "shard_count": manifest["shard_count"],
                    "manifest": str(manifest_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    exclusive_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "prepared",
                "request_count": manifest["request_count"],
                "shard_count": manifest["shard_count"],
                "maximum_items": manifest["maximum_items"],
                "maximum_source_characters": manifest[
                    "maximum_source_characters"
                ],
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_bound_run(
    request_path: Path,
    run_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = run_directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing run manifest: {manifest_path}; run prepare first")
    manifest = load_json(manifest_path)
    expected, rows, contract = expected_manifest(
        request_path,
        run_directory,
        int(manifest["maximum_items"]),
        int(manifest["maximum_source_characters"]),
    )
    if manifest != expected:
        raise SystemExit("run manifest is not bound to the current sealed request corpus")
    return manifest, rows, contract


def batch_schema(item_schema: dict[str, Any], item_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "minItems": item_count,
                "maxItems": item_count,
                "items": item_schema,
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def shard_prompt(
    developer_prompt: str,
    sources: list[dict[str, Any]],
) -> str:
    return (
        CODEX_WRAPPER
        + "\n"
        + developer_prompt.strip()
        + "\n\nSource-only inputs:\n"
        + json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )


def validate_teacher_payload(
    value: Any,
    expected_custom_id: str,
) -> dict[str, Any]:
    payload = json_object(value, f"teacher result {expected_custom_id}")
    if set(payload) == {"source_id", "canonical_translation", "risk_tags"}:
        if payload["source_id"] != expected_custom_id:
            raise ValueError(f"{expected_custom_id}: source_id mismatch")
        if (
            not isinstance(payload["canonical_translation"], str)
            or not payload["canonical_translation"].strip()
        ):
            raise ValueError(
                f"{expected_custom_id}: empty canonical teacher translation"
            )
        risk_tags = payload["risk_tags"]
        if (
            not isinstance(risk_tags, list)
            or any(
                not isinstance(tag, str) or tag not in RISK_TAGS
                for tag in risk_tags
            )
            or len(risk_tags) != len(set(risk_tags))
        ):
            raise ValueError(f"{expected_custom_id}: invalid canonical risk tags")
        return payload
    if set(payload) != {"source_id", "translation_brief", "candidates"}:
        raise ValueError(f"{expected_custom_id}: unexpected teacher result fields")
    if payload["source_id"] != expected_custom_id:
        raise ValueError(f"{expected_custom_id}: source_id mismatch")
    brief = json_object(
        payload["translation_brief"],
        f"translation brief {expected_custom_id}",
    )
    if set(brief) != {"register", "terms", "preserve", "ambiguities"}:
        raise ValueError(f"{expected_custom_id}: invalid translation_brief fields")
    if brief["register"] not in REGISTER_VALUES:
        raise ValueError(f"{expected_custom_id}: invalid register")
    if not isinstance(brief["terms"], list):
        raise ValueError(f"{expected_custom_id}: terms must be a list")
    for term in brief["terms"]:
        term_object = json_object(term, f"term {expected_custom_id}")
        if set(term_object) != {"source", "target"} or not all(
            isinstance(term_object[key], str) for key in ("source", "target")
        ):
            raise ValueError(f"{expected_custom_id}: invalid term")
    if not all(
        isinstance(brief[key], list)
        and all(isinstance(item, str) for item in brief[key])
        for key in ("preserve", "ambiguities")
    ):
        raise ValueError(f"{expected_custom_id}: invalid preserve or ambiguities")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError(f"{expected_custom_id}: exactly three candidates required")
    styles: set[str] = set()
    for candidate_value in candidates:
        candidate = json_object(
            candidate_value,
            f"candidate {expected_custom_id}",
        )
        if set(candidate) != {"translation", "style", "risk_tags"}:
            raise ValueError(f"{expected_custom_id}: invalid candidate fields")
        if not isinstance(candidate["translation"], str) or not candidate[
            "translation"
        ].strip():
            raise ValueError(f"{expected_custom_id}: empty candidate translation")
        style = candidate["style"]
        if style not in STYLES:
            raise ValueError(f"{expected_custom_id}: invalid candidate style")
        styles.add(style)
        risk_tags = candidate["risk_tags"]
        if (
            not isinstance(risk_tags, list)
            or any(tag not in RISK_TAGS for tag in risk_tags)
            or len(set(risk_tags)) != len(risk_tags)
        ):
            raise ValueError(f"{expected_custom_id}: invalid risk tags")
    if styles != STYLES:
        raise ValueError(f"{expected_custom_id}: candidate styles are incomplete")
    return payload


def result_paths(run_directory: Path, index: int) -> tuple[Path, Path]:
    stem = f"{index:05d}"
    shard_directory = run_directory / "shards"
    return (
        shard_directory / f"{stem}.results.jsonl",
        shard_directory / f"{stem}.metadata.json",
    )


def validate_completed_shard(
    result_path: Path,
    metadata_path: Path,
    expected_ids: list[str],
) -> list[dict[str, Any]]:
    if not result_path.is_file() or not metadata_path.is_file():
        raise ValueError("result and metadata must both exist")
    metadata = load_json(metadata_path)
    if metadata.get("result_sha256") != sha256_path(result_path):
        raise ValueError(f"{result_path}: result hash does not match metadata")
    rows = [
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    found_ids = [str(row.get("custom_id")) for row in rows]
    if found_ids != expected_ids:
        raise ValueError(f"{result_path}: result IDs or order do not match shard")
    for row, expected_id in zip(rows, expected_ids):
        body = json_object(
            row.get("response", {}).get("body"),
            f"collected response {expected_id}",
        )
        validate_teacher_payload(json.loads(body["output_text"]), expected_id)
    return rows


def codex_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_one_shard(
    request_path: Path,
    run_directory: Path,
    index: int,
    codex_executable: str,
) -> dict[str, Any]:
    manifest, rows, contract = load_bound_run(request_path, run_directory)
    if index < 0 or index >= manifest["shard_count"]:
        raise SystemExit(
            f"shard index {index} is outside 0..{manifest['shard_count'] - 1}"
        )
    shard = manifest["shards"][index]
    expected_ids = [str(value) for value in shard["custom_ids"]]
    result_path, metadata_path = result_paths(run_directory, index)
    if result_path.exists() or metadata_path.exists():
        try:
            completed_rows = validate_completed_shard(
                result_path,
                metadata_path,
                expected_ids,
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"incomplete or invalid existing shard {index}: {error}"
            ) from error
        return {
            "status": "already-complete",
            "shard": index,
            "results": len(completed_rows),
            "result": str(result_path),
        }

    by_id = {str(row["custom_id"]): row for row in rows}
    source_rows = [by_id[custom_id]["source"] for custom_id in expected_ids]
    if sha256_bytes(canonical_bytes(source_rows)) != shard["source_payload_sha256"]:
        raise SystemExit(f"shard {index}: source payload hash mismatch")
    prompt = shard_prompt(contract["developer_prompt"], source_rows)
    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
    executable = shutil.which(codex_executable) or codex_executable
    version = codex_version(executable)
    started_at = utc_now()
    with tempfile.TemporaryDirectory(prefix=f"mimi-codex-teacher-{index:05d}-") as work:
        temporary = Path(work)
        schema_path = temporary / "schema.json"
        output_path = temporary / "output.json"
        schema_path.write_text(
            json.dumps(
                batch_schema(contract["schema"], len(expected_ids)),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        command = [
            executable,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-m",
            str(manifest["teacher_model"]),
            "-c",
            'model_reasoning_effort="none"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]
        process = subprocess.run(
            command,
            cwd=ROOT,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        transcript = process.stdout or ""
        if process.returncode != 0 or not output_path.is_file():
            failure_path = (
                run_directory
                / "failures"
                / f"{index:05d}-{sha256_bytes(transcript.encode())[:12]}.json"
            )
            exclusive_json(
                failure_path,
                {
                    "schema_version": PIPELINE_VERSION,
                    "shard": index,
                    "returncode": process.returncode,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "codex_version": version,
                    "prompt_sha256": prompt_sha256,
                    "transcript_tail": transcript[-8000:],
                },
            )
            raise SystemExit(
                f"Codex teacher shard {index} failed; diagnostic: {failure_path}"
            )
        try:
            output = json_object(
                json.loads(output_path.read_text(encoding="utf-8")),
                f"Codex output shard {index}",
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise SystemExit(f"Codex shard {index} returned invalid JSON: {error}") from error

    values = output.get("results")
    if not isinstance(values, list) or len(values) != len(expected_ids):
        raise SystemExit(
            f"Codex shard {index}: expected {len(expected_ids)} results"
        )
    by_output_id: dict[str, dict[str, Any]] = {}
    try:
        for value in values:
            candidate_id = str(json_object(value, "teacher result").get("source_id"))
            if candidate_id in by_output_id:
                raise ValueError(f"duplicate source_id: {candidate_id}")
            by_output_id[candidate_id] = validate_teacher_payload(value, candidate_id)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Codex shard {index}: {error}") from error
    if set(by_output_id) != set(expected_ids):
        unknown = set(by_output_id) - set(expected_ids)
        missing = set(expected_ids) - set(by_output_id)
        raise SystemExit(
            f"Codex shard {index}: {len(unknown)} unknown and {len(missing)} missing IDs"
        )

    collected_rows: list[dict[str, Any]] = []
    for position, custom_id in enumerate(expected_ids):
        payload = by_output_id[custom_id]
        response_id = (
            f"codex-cli-{index:05d}-{position:03d}-"
            f"{sha256_bytes(canonical_bytes(payload))[:16]}"
        )
        collected_rows.append(
            {
                "custom_id": custom_id,
                "response": {
                    "body": {
                        "id": response_id,
                        "model": f"{manifest['teacher_model']}-via-codex-cli",
                        "system_fingerprint": None,
                        "output_text": json.dumps(
                            payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                },
                "mimi_teacher_transport": {
                    "transport": "codex-cli-chatgpt-login",
                    "codex_version": version,
                    "reasoning_effort": "none",
                    "reasoning_trace_stored": False,
                    "prompt_sha256": prompt_sha256,
                },
            }
        )
    result_bytes = b"".join(
        canonical_bytes(row) + b"\n" for row in collected_rows
    )
    token_match = TOKEN_USAGE_RE.search(transcript)
    token_usage = int(token_match.group(1).replace(",", "")) if token_match else None
    metadata = {
        "schema_version": PIPELINE_VERSION,
        "transport": "codex-cli-chatgpt-login",
        "shard": index,
        "request_sha256": manifest["request_sha256"],
        "source_payload_sha256": shard["source_payload_sha256"],
        "prompt_sha256": prompt_sha256,
        "result_sha256": sha256_bytes(result_bytes),
        "result_count": len(collected_rows),
        "custom_ids": expected_ids,
        "teacher_model": manifest["teacher_model"],
        "reasoning_effort": "none",
        "source_only": True,
        "reasoning_trace_stored": False,
        "codex_version": version,
        "reported_tokens_used": token_usage,
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    exclusive_bytes(result_path, result_bytes)
    exclusive_json(metadata_path, metadata)
    return {
        "status": "completed",
        "shard": index,
        "results": len(collected_rows),
        "reported_tokens_used": token_usage,
        "result": str(result_path),
        "metadata": str(metadata_path),
    }


def run_command(arguments: argparse.Namespace) -> None:
    manifest, _, _ = load_bound_run(arguments.requests, arguments.run_directory)
    completed = 0
    for index in range(arguments.start_shard, manifest["shard_count"]):
        if arguments.maximum_shards is not None and completed >= arguments.maximum_shards:
            break
        result = run_one_shard(
            arguments.requests,
            arguments.run_directory,
            index,
            arguments.codex_executable,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        if result["status"] == "completed":
            completed += 1


def status_command(arguments: argparse.Namespace) -> None:
    manifest, _, _ = load_bound_run(arguments.requests, arguments.run_directory)
    complete: list[int] = []
    invalid: dict[int, str] = {}
    tokens = 0
    for shard in manifest["shards"]:
        index = int(shard["index"])
        result_path, metadata_path = result_paths(arguments.run_directory, index)
        if not result_path.exists() and not metadata_path.exists():
            continue
        try:
            validate_completed_shard(
                result_path,
                metadata_path,
                [str(value) for value in shard["custom_ids"]],
            )
            complete.append(index)
            metadata = load_json(metadata_path)
            tokens += int(metadata.get("reported_tokens_used") or 0)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            invalid[index] = str(error)
    print(
        json.dumps(
            {
                "request_count": manifest["request_count"],
                "shard_count": manifest["shard_count"],
                "complete_shards": len(complete),
                "remaining_shards": manifest["shard_count"] - len(complete),
                "completed_indices": complete,
                "invalid_shards": invalid,
                "reported_tokens_used": tokens,
            },
            indent=2,
            sort_keys=True,
        )
    )


def collect_command(arguments: argparse.Namespace) -> None:
    manifest, _, contract = load_bound_run(
        arguments.requests,
        arguments.run_directory,
    )
    collected_by_id: dict[str, dict[str, Any]] = {}
    shard_hashes: list[str] = []
    for shard in manifest["shards"]:
        index = int(shard["index"])
        result_path, metadata_path = result_paths(arguments.run_directory, index)
        try:
            shard_rows = validate_completed_shard(
                result_path,
                metadata_path,
                [str(value) for value in shard["custom_ids"]],
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"cannot collect incomplete shard {index}: {error}") from error
        shard_hashes.append(sha256_path(result_path))
        for row in shard_rows:
            custom_id = str(row["custom_id"])
            if custom_id in collected_by_id:
                raise SystemExit(f"duplicate collected custom_id: {custom_id}")
            collected_by_id[custom_id] = row
    expected_ids = [str(value) for value in contract["custom_ids"]]
    if set(collected_by_id) != set(expected_ids):
        raise SystemExit("collected Codex results do not match the sealed request IDs")
    output = b"".join(
        canonical_bytes(collected_by_id[custom_id]) + b"\n"
        for custom_id in expected_ids
    )
    exclusive_bytes(arguments.output, output)
    summary = {
        "schema_version": PIPELINE_VERSION,
        "transport": "codex-cli-chatgpt-login",
        "request_sha256": manifest["request_sha256"],
        "request_count": len(expected_ids),
        "output": str(arguments.output.resolve()),
        "output_sha256": sha256_bytes(output),
        "shard_result_sha256": shard_hashes,
        "source_only": True,
        "reasoning_trace_stored": False,
        "collected_at": utc_now(),
    }
    exclusive_json(arguments.output.with_suffix(arguments.output.suffix + ".manifest.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="freeze deterministic source-only Codex shards",
    )
    prepare.add_argument("requests", type=Path)
    prepare.add_argument("run_directory", type=Path)
    prepare.add_argument("--maximum-items", type=int, default=32)
    prepare.add_argument("--maximum-source-characters", type=int, default=8_000)
    prepare.set_defaults(handler=prepare_command)

    run = commands.add_parser(
        "run",
        help="run missing shards sequentially through cached ChatGPT login",
    )
    run.add_argument("requests", type=Path)
    run.add_argument("run_directory", type=Path)
    run.add_argument("--start-shard", type=int, default=0)
    run.add_argument("--maximum-shards", type=int)
    run.add_argument("--codex-executable", default="codex")
    run.set_defaults(handler=run_command)

    status = commands.add_parser("status", help="validate resumable shard state")
    status.add_argument("requests", type=Path)
    status.add_argument("run_directory", type=Path)
    status.set_defaults(handler=status_command)

    collect = commands.add_parser(
        "collect",
        help="collect all shards into Batch-compatible JSONL",
    )
    collect.add_argument("requests", type=Path)
    collect.add_argument("run_directory", type=Path)
    collect.add_argument("output", type=Path)
    collect.set_defaults(handler=collect_command)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
