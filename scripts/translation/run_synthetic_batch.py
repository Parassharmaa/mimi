#!/usr/bin/env python3
"""Validate, submit, inspect, and collect Mimi's OpenAI Batch API run.

Network operations are deliberately separate from request construction. Submission
requires the operator to paste the exact SHA-256 printed by ``validate``.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_ENDPOINT = "/v1/responses"
MAX_BATCH_BYTES = 200_000_000
STATE_SCHEMA_VERSION = 1
EXPECTED_BODY_KEYS = {
    "model",
    "store",
    "reasoning",
    "input",
    "text",
    "max_output_tokens",
    "metadata",
}
TEACHER_PIPELINE = "mimi-translation-v1"
JUDGE_PIPELINE = "mimi-translation-judge-v1"
REFERENCE_GENERATOR_PIPELINE = "mimi-benchmark-reference-generator-v1"
REFERENCE_JUDGE_PIPELINE = "mimi-benchmark-reference-judge-v1"
EXPECTED_TEACHER_SOURCE_KEYS = {
    "source_id",
    "source_language",
    "target_language",
    "domain",
    "source",
}
EXPECTED_JUDGE_SOURCE_KEYS = EXPECTED_TEACHER_SOURCE_KEYS | {"candidates"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def request_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"request file does not exist: {path}")
    raw = path.read_bytes()
    if not raw:
        raise SystemExit("request file is empty")
    if len(raw) > MAX_BATCH_BYTES:
        raise SystemExit(
            f"request file is {len(raw)} bytes; the Batch API limit is {MAX_BATCH_BYTES}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"request file is not UTF-8: {error}") from error

    custom_ids: list[str] = []
    models: set[str] = set()
    prompt_hashes: set[str] = set()
    pipelines: set[str] = set()
    reasoning_efforts: set[str] = set()
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            request = json_object(json.loads(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise SystemExit(f"invalid request JSON at line {line_number}: {error}") from error
        if set(request) != {"custom_id", "method", "url", "body"}:
            raise SystemExit(f"line {line_number}: request keys do not match the sealed contract")
        custom_id = str(request["custom_id"])
        if not custom_id or custom_id in seen:
            raise SystemExit(f"line {line_number}: empty or duplicate custom_id: {custom_id!r}")
        seen.add(custom_id)
        custom_ids.append(custom_id)
        if request["method"] != "POST" or request["url"] != API_ENDPOINT:
            raise SystemExit(f"line {line_number}: only POST {API_ENDPOINT} is permitted")

        body = json_object(request["body"])
        body_keys = set(body)
        if frozenset(body_keys) not in {
            frozenset(EXPECTED_BODY_KEYS),
            frozenset(EXPECTED_BODY_KEYS - {"reasoning"}),
        }:
            raise SystemExit(f"line {line_number}: response body keys do not match the sealed contract")
        if body.get("store") is not False:
            raise SystemExit(f"line {line_number}: store must be false")
        reasoning_value = body.get("reasoning")
        if reasoning_value is None:
            effort = "not-supported"
        else:
            reasoning = json_object(reasoning_value)
            effort = reasoning.get("effort")
            if not isinstance(effort, str):
                raise SystemExit(f"line {line_number}: reasoning effort must be a string")
        reasoning_efforts.add(effort)
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise SystemExit(f"line {line_number}: model must be a non-empty string")
        models.add(model)

        messages = body.get("input")
        if not isinstance(messages, list) or len(messages) != 2:
            raise SystemExit(f"line {line_number}: input must contain developer and user messages")
        developer_message = json_object(messages[0])
        user_message = json_object(messages[1])
        if set(developer_message) != {"role", "content"} or developer_message["role"] != "developer":
            raise SystemExit(f"line {line_number}: first message must be the developer prompt")
        if set(user_message) != {"role", "content"} or user_message["role"] != "user":
            raise SystemExit(f"line {line_number}: second message must be the source-only user input")
        developer_prompt = developer_message.get("content")
        if not isinstance(developer_prompt, str) or not developer_prompt:
            raise SystemExit(f"line {line_number}: developer prompt must be non-empty text")
        metadata = json_object(body.get("metadata"))
        pipeline = metadata.get("pipeline")
        if pipeline not in {
            TEACHER_PIPELINE,
            JUDGE_PIPELINE,
            REFERENCE_GENERATOR_PIPELINE,
            REFERENCE_JUDGE_PIPELINE,
        }:
            raise SystemExit(f"line {line_number}: unknown Mimi Batch pipeline: {pipeline!r}")
        pipelines.add(pipeline)
        if pipeline in {TEACHER_PIPELINE, REFERENCE_GENERATOR_PIPELINE} and effort != "none":
            raise SystemExit(
                f"line {line_number}: teacher reasoning effort must be none so the "
                "final-output-only contract cannot retain encrypted reasoning items"
            )
        if pipeline == JUDGE_PIPELINE and effort not in {"none", "minimal", "low"}:
            raise SystemExit(f"line {line_number}: judge reasoning effort must be none, minimal, or low")
        if pipeline == REFERENCE_JUDGE_PIPELINE and effort not in {"none", "not-supported"}:
            raise SystemExit(
                f"line {line_number}: reference judge reasoning effort must be none or omitted"
            )

        try:
            source = json_object(json.loads(user_message["content"]))
        except (TypeError, json.JSONDecodeError, ValueError) as error:
            raise SystemExit(f"line {line_number}: user content is not a JSON object: {error}") from error
        expected_source_keys = (
            EXPECTED_TEACHER_SOURCE_KEYS
            if pipeline in {TEACHER_PIPELINE, REFERENCE_GENERATOR_PIPELINE}
            else EXPECTED_JUDGE_SOURCE_KEYS
        )
        if set(source) != expected_source_keys:
            if pipeline in {TEACHER_PIPELINE, REFERENCE_GENERATOR_PIPELINE}:
                raise SystemExit(
                    f"line {line_number}: teacher input must contain source-only fields; "
                    "references and student hypotheses are forbidden"
                )
            raise SystemExit(f"line {line_number}: judge input keys do not match the blinded contract")
        if source["source_id"] != custom_id:
            raise SystemExit(f"line {line_number}: source_id does not match custom_id")
        if {source["source_language"], source["target_language"]} != {"en-US", "ja-JP"}:
            raise SystemExit(f"line {line_number}: unsupported language direction")
        if not isinstance(source["source"], str) or not source["source"].strip():
            raise SystemExit(f"line {line_number}: source must be non-empty text")
        if pipeline in {JUDGE_PIPELINE, REFERENCE_JUDGE_PIPELINE}:
            candidates = source.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 3:
                raise SystemExit(f"line {line_number}: judge input must contain three candidates")
            candidate_ids: set[str] = set()
            for candidate in candidates:
                candidate = json_object(candidate)
                if set(candidate) != {"candidate_id", "translation"}:
                    raise SystemExit(f"line {line_number}: judge candidate keys are invalid")
                candidate_id = candidate.get("candidate_id")
                translation = candidate.get("translation")
                if (
                    not isinstance(candidate_id, str)
                    or not candidate_id
                    or candidate_id in candidate_ids
                    or not isinstance(translation, str)
                    or not translation.strip()
                ):
                    raise SystemExit(f"line {line_number}: judge candidates must be unique, non-empty text")
                candidate_ids.add(candidate_id)

        declared_prompt_hash = metadata.get("prompt_sha256")
        actual_prompt_hash = sha256_bytes(developer_prompt.encode("utf-8"))
        if declared_prompt_hash != actual_prompt_hash:
            raise SystemExit(f"line {line_number}: prompt_sha256 does not match the developer prompt")
        prompt_hashes.add(actual_prompt_hash)

        text_config = json_object(body.get("text"))
        output_format = json_object(text_config.get("format"))
        if output_format.get("type") != "json_schema" or output_format.get("strict") is not True:
            raise SystemExit(f"line {line_number}: strict Structured Outputs are required")

    if not custom_ids:
        raise SystemExit("request file contains no requests")
    if (
        len(models) != 1
        or len(prompt_hashes) != 1
        or len(pipelines) != 1
        or len(reasoning_efforts) != 1
    ):
        raise SystemExit("all requests must use one model, pipeline, reasoning effort, and prompt")
    return {
        "request_path": str(path.resolve()),
        "request_sha256": sha256_bytes(raw),
        "request_bytes": len(raw),
        "request_count": len(custom_ids),
        "custom_ids": custom_ids,
        "model": next(iter(models)),
        "prompt_sha256": next(iter(prompt_hashes)),
        "pipeline": next(iter(pipelines)),
        "reasoning_effort": next(iter(reasoning_efforts)),
    }


def public_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "custom_ids"}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"batch state does not exist: {path}")
    try:
        state = json_object(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"invalid batch state: {error}") from error
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise SystemExit("unsupported batch state schema")
    return state


def api_client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for this network operation")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit(
            "the OpenAI Python SDK is required; run this command with `uv run --with openai`"
        ) from error
    return OpenAI()


def sdk_version() -> str | None:
    try:
        import openai
    except ImportError:
        return None
    return getattr(openai, "__version__", None)


def model_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise SystemExit(f"unexpected OpenAI SDK object: {type(value).__name__}")


def bind_state_to_contract(state: dict[str, Any], contract: dict[str, Any]) -> None:
    for key in (
        "request_sha256",
        "request_bytes",
        "request_count",
        "model",
        "prompt_sha256",
        "pipeline",
        "reasoning_effort",
    ):
        if state.get(key) != contract[key]:
            raise SystemExit(f"batch state {key} does not match the current request file")


def validate_command(arguments: argparse.Namespace) -> None:
    print(json.dumps(public_contract(request_contract(arguments.requests)), indent=2, sort_keys=True))


def find_existing_batch(client: Any, state: dict[str, Any]) -> Any | None:
    matches = []
    for batch in client.batches.list(limit=100):
        metadata = getattr(batch, "metadata", None) or {}
        if (
            getattr(batch, "input_file_id", None) == state["input_file_id"]
            and metadata.get("request_sha256") == state["request_sha256"]
        ):
            matches.append(batch)
    if len(matches) > 1:
        raise SystemExit("multiple matching batches exist; inspect them before continuing")
    return matches[0] if matches else None


def submit_command(arguments: argparse.Namespace) -> None:
    contract = request_contract(arguments.requests)
    supplied = arguments.confirm_input_sha256.lower()
    if not hmac.compare_digest(supplied, contract["request_sha256"]):
        raise SystemExit(
            "submission confirmation does not match the request SHA-256; run `validate` and paste it"
        )
    client = api_client()
    openai_sdk_version = sdk_version()

    if arguments.state.exists():
        state = load_state(arguments.state)
        bind_state_to_contract(state, contract)
        if state.get("batch_id"):
            raise SystemExit(f"request file is already bound to batch {state['batch_id']}; not resubmitting")
    else:
        with arguments.requests.open("rb") as request_file:
            uploaded = client.files.create(file=request_file, purpose="batch")
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            **public_contract(contract),
            "phase": "uploaded",
            "input_file_id": uploaded.id,
            "uploaded_file": model_dict(uploaded),
            "uploaded_at": utc_now(),
            "openai_sdk_version": openai_sdk_version,
        }
        save_state(arguments.state, state)

    existing = find_existing_batch(client, state)
    if existing is None:
        state["phase"] = "creating-batch"
        state["batch_creation_started_at"] = utc_now()
        save_state(arguments.state, state)
        try:
            existing = client.batches.create(
                input_file_id=state["input_file_id"],
                endpoint=API_ENDPOINT,
                completion_window="24h",
                metadata={
                    "pipeline": contract["pipeline"],
                    "request_sha256": state["request_sha256"],
                },
            )
        except Exception:
            state["phase"] = "batch-creation-uncertain"
            state["batch_creation_failed_at"] = utc_now()
            save_state(arguments.state, state)
            raise

    batch = model_dict(existing)
    state.update({
        "phase": "submitted",
        "batch_id": batch["id"],
        "batch": batch,
        "last_checked_at": utc_now(),
    })
    save_state(arguments.state, state)
    print(json.dumps({
        "batch_id": state["batch_id"],
        "input_file_id": state["input_file_id"],
        "request_sha256": state["request_sha256"],
        "status": batch["status"],
        "state": str(arguments.state),
    }, indent=2, sort_keys=True))


def refresh_batch(client: Any, state_path: Path, state: dict[str, Any]) -> dict[str, Any]:
    batch_id = state.get("batch_id")
    if not batch_id:
        raise SystemExit("batch state has no batch_id; submission did not complete")
    batch = model_dict(client.batches.retrieve(batch_id))
    state["batch"] = batch
    state["phase"] = batch["status"]
    state["last_checked_at"] = utc_now()
    save_state(state_path, state)
    return batch


def status_command(arguments: argparse.Namespace) -> None:
    state = load_state(arguments.state)
    batch = refresh_batch(api_client(), arguments.state, state)
    print(json.dumps({
        "batch_id": batch["id"],
        "status": batch["status"],
        "request_counts": batch.get("request_counts"),
        "output_file_id": batch.get("output_file_id"),
        "error_file_id": batch.get("error_file_id"),
        "state": str(arguments.state),
    }, indent=2, sort_keys=True))


def file_content(client: Any, file_id: str) -> bytes:
    response = client.files.content(file_id)
    if hasattr(response, "read"):
        content = response.read()
    else:
        content = getattr(response, "content", None)
        if content is None:
            content = getattr(response, "text", None)
    if isinstance(content, str):
        return content.encode("utf-8")
    if not isinstance(content, bytes):
        raise SystemExit(f"unexpected content response for {file_id}")
    return content


def result_ids(raw: bytes, label: str) -> set[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SystemExit(f"{label} is not UTF-8: {error}") from error
    found: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json_object(json.loads(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise SystemExit(f"invalid {label} JSON at line {line_number}: {error}") from error
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id or custom_id in found:
            raise SystemExit(f"{label} has an empty or duplicate custom_id at line {line_number}")
        found.add(custom_id)
    return found


def exclusive_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite collected file: {path}") from error


def collect_command(arguments: argparse.Namespace) -> None:
    state = load_state(arguments.state)
    request_path = Path(state["request_path"])
    contract = request_contract(request_path)
    bind_state_to_contract(state, contract)
    client = api_client()
    batch = refresh_batch(client, arguments.state, state)
    if batch["status"] != "completed":
        raise SystemExit(f"batch status is {batch['status']!r}; collection requires 'completed'")

    output_file_id = batch.get("output_file_id")
    error_file_id = batch.get("error_file_id")
    output = file_content(client, output_file_id) if output_file_id else b""
    errors = file_content(client, error_file_id) if error_file_id else b""
    output_ids = result_ids(output, "batch output")
    error_ids = result_ids(errors, "batch error output")
    if output_ids & error_ids:
        raise SystemExit("the success and error files contain overlapping custom_ids")
    expected_ids = set(contract["custom_ids"])
    unknown = (output_ids | error_ids) - expected_ids
    missing = expected_ids - (output_ids | error_ids)
    if unknown or missing:
        raise SystemExit(
            f"collected IDs do not match requests: {len(unknown)} unknown, {len(missing)} missing"
        )
    counts = batch.get("request_counts") or {}
    if counts.get("total") not in (None, len(expected_ids)):
        raise SystemExit("Batch API request_counts.total does not match the input file")
    if counts.get("completed") not in (None, len(output_ids)):
        raise SystemExit("Batch API request_counts.completed does not match the output file")
    if counts.get("failed") not in (None, len(error_ids)):
        raise SystemExit("Batch API request_counts.failed does not match the error file")

    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite collected file: {arguments.output}")
    if error_file_id and arguments.error_output.exists():
        raise SystemExit(f"refusing to overwrite collected file: {arguments.error_output}")
    exclusive_write(arguments.output, output)
    if error_file_id:
        exclusive_write(arguments.error_output, errors)

    state["phase"] = "collected"
    state["collection"] = {
        "collected_at": utc_now(),
        "output_path": str(arguments.output.resolve()),
        "output_sha256": sha256_bytes(output),
        "output_count": len(output_ids),
        "error_path": str(arguments.error_output.resolve()) if error_file_id else None,
        "error_sha256": sha256_bytes(errors) if error_file_id else None,
        "error_count": len(error_ids),
        "complete_without_request_errors": not error_ids,
    }
    save_state(arguments.state, state)
    print(json.dumps(state["collection"], indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a request JSONL without network access")
    validate.add_argument("requests", type=Path)
    validate.set_defaults(handler=validate_command)

    submit = commands.add_parser("submit", help="upload and submit a hash-confirmed batch")
    submit.add_argument("requests", type=Path)
    submit.add_argument("state", type=Path)
    submit.add_argument("--confirm-input-sha256", required=True)
    submit.set_defaults(handler=submit_command)

    status = commands.add_parser("status", help="retrieve and save the current batch status")
    status.add_argument("state", type=Path)
    status.set_defaults(handler=status_command)

    collect = commands.add_parser("collect", help="download and verify a completed batch")
    collect.add_argument("state", type=Path)
    collect.add_argument("output", type=Path)
    collect.add_argument("--error-output", type=Path)
    collect.set_defaults(handler=collect_command)
    return root


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "collect" and arguments.error_output is None:
        arguments.error_output = arguments.output.with_name(f"{arguments.output.stem}.errors.jsonl")
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
