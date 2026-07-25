#!/usr/bin/env python3
"""Run a pinned local MLX model as a blinded multi-candidate MT judge."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
JUDGE_PIPELINE = "mimi-translation-judge-v1"
LANGUAGE_NAMES = {"en-US": "English", "ja-JP": "Japanese"}
ALLOWED_ERROR_TAGS = {
    "meaning-reversal",
    "negation",
    "number-or-date",
    "named-entity",
    "omission",
    "addition",
    "register",
    "terminology",
    "disfluency",
}
ERROR_TAG_ALIASES = {
    # Qwen occasionally emits this narrower description even after schema
    # repair. Repetition is a fluency defect, so retain the negative judgment
    # under the existing closed-vocabulary tag instead of dropping or
    # weakening it.
    "repetition": "disfluency",
}
SYSTEM_PROMPT = """You are a strict professional bilingual English-Japanese translation judge.
Compare every anonymous candidate directly against the same source and against each other.
Silently establish the complete source meaning first. Score adequacy, fluency, and terminology
as integers from 0 to 4. Use adequacy 4 only when every meaning is preserved with no meaningful
addition or omission. Treat changed negation, meaning reversal, wrong numbers or dates, and wrong
named entities as critical. protected_tokens_preserved must be false when exact URLs, placeholders,
markup, or required critical surfaces are corrupted.
Set protected_tokens_preserved to true when the source has no such protected token.

Use the score range discriminatively: when one candidate is materially better, its total score
must be higher. Do not reward verbosity or invent a difference between genuinely equivalent
translations. An exact tie is an abstention and is preferable to an arbitrary winner.
Never infer candidate origin. Do not reveal reasoning or repair candidates.
Return one compact JSON object only. It must have exactly two keys: source_id and
assessments. Copy the actual source_id and every actual candidate_id from the input;
do not emit placeholder text. Include exactly one assessment for every candidate.
Each assessment must be a JSON array of exactly seven values in this order:
candidate_id string, adequacy integer, fluency integer, terminology integer,
protected_tokens_preserved boolean, critical_error boolean, error_tags array.
Do not use objects for individual assessments. error_tags may only contain:
meaning-reversal, negation, number-or-date, named-entity, omission, addition,
register, terminology, disfluency. Never add explanations or extra values."""


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def validate_source(source: Any, custom_id: str) -> dict:
    required = {
        "source_id",
        "source_language",
        "target_language",
        "domain",
        "source",
        "candidates",
    }
    if not isinstance(source, dict) or set(source) != required:
        raise ValueError(f"invalid blinded source fields: {custom_id}")
    if source["source_id"] != custom_id:
        raise ValueError(f"source ID mismatch: {custom_id}")
    if (
        source["source_language"] not in LANGUAGE_NAMES
        or source["target_language"] not in LANGUAGE_NAMES
        or source["source_language"] == source["target_language"]
    ):
        raise ValueError(f"unsupported language direction: {custom_id}")
    candidates = source["candidates"]
    if not isinstance(candidates, list) or len(candidates) not in {3, 4}:
        raise ValueError(f"invalid candidate count: {custom_id}")
    candidate_ids: list[str] = []
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or set(candidate) != {"candidate_id", "translation"}
            or not isinstance(candidate["candidate_id"], str)
            or not candidate["candidate_id"]
            or not isinstance(candidate["translation"], str)
            or not candidate["translation"].strip()
        ):
            raise ValueError(f"invalid blinded candidate: {custom_id}")
        candidate_ids.append(candidate["candidate_id"])
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"duplicate candidate ID: {custom_id}")
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
        raise ValueError(f"candidate-origin leakage: {custom_id}")
    return source


def parse_payload(
    text: str,
    source: dict,
    *,
    allow_source_id_correction: bool = False,
) -> dict:
    text = re.sub(r"(?s)^.*?</think>\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match is None:
        raise ValueError("judge returned no JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ValueError(f"judge returned invalid JSON: {error}") from error
    if not isinstance(payload, dict) or set(payload) != {"source_id", "assessments"}:
        raise ValueError("judge returned unexpected result fields")
    source_id_corrected = payload["source_id"] != source["source_id"]
    if source_id_corrected and not allow_source_id_correction:
        raise ValueError("judge returned a different source ID")
    assessments = payload["assessments"]
    expected_ids = {
        candidate["candidate_id"] for candidate in source["candidates"]
    }
    if not isinstance(assessments, list) or len(assessments) != len(expected_ids):
        raise ValueError("judge returned an invalid assessment count")
    normalized: list[dict] = []
    normalized_error_tag_count = 0
    found_ids: set[str] = set()
    for assessment_values in assessments:
        if not isinstance(assessment_values, list) or len(assessment_values) != 7:
            raise ValueError("judge returned an invalid assessment tuple")
        assessment = dict(zip(
            (
                "candidate_id",
                "adequacy",
                "fluency",
                "terminology",
                "protected_tokens_preserved",
                "critical_error",
                "error_tags",
            ),
            assessment_values,
            strict=True,
        ))
        candidate_id = assessment["candidate_id"]
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in expected_ids
            or candidate_id in found_ids
        ):
            raise ValueError("judge returned an unknown or duplicate candidate ID")
        found_ids.add(candidate_id)
        for score_name in ("adequacy", "fluency", "terminology"):
            score = assessment[score_name]
            if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
                raise ValueError(f"judge returned an invalid {score_name} score")
        for boolean_name in ("protected_tokens_preserved", "critical_error"):
            if not isinstance(assessment[boolean_name], bool):
                raise ValueError(f"judge returned an invalid {boolean_name} value")
        error_tags = assessment["error_tags"]
        if isinstance(error_tags, list) and all(
            isinstance(tag, str) for tag in error_tags
        ):
            mapped_tags = [ERROR_TAG_ALIASES.get(tag, tag) for tag in error_tags]
            normalized_error_tag_count += sum(
                original != mapped
                for original, mapped in zip(error_tags, mapped_tags, strict=True)
            )
            assessment["error_tags"] = list(dict.fromkeys(mapped_tags))
            error_tags = assessment["error_tags"]
        if (
            not isinstance(error_tags, list)
            or any(
                not isinstance(tag, str) or tag not in ALLOWED_ERROR_TAGS
                for tag in error_tags
            )
            or len(error_tags) != len(set(error_tags))
        ):
            raise ValueError("judge returned invalid error tags")
        normalized.append(assessment)
    if found_ids != expected_ids:
        raise ValueError("judge did not cover the exact candidate set")
    normalized.sort(key=lambda value: value["candidate_id"])
    result = {"source_id": source["source_id"], "assessments": normalized}
    if source_id_corrected:
        result["_mimi_source_id_corrected"] = True
    if normalized_error_tag_count:
        result["_mimi_normalized_error_tags"] = normalized_error_tag_count
    return result


def read_requests(path: Path) -> list[dict]:
    parsed: list[dict] = []
    seen: set[str] = set()
    for request in rows(path):
        custom_id = str(request.get("custom_id", ""))
        if not custom_id or custom_id in seen:
            raise ValueError("request IDs must be non-empty and unique")
        seen.add(custom_id)
        body = request.get("body", {})
        if body.get("metadata", {}).get("pipeline") != JUDGE_PIPELINE:
            raise ValueError(f"unexpected judge pipeline: {custom_id}")
        inputs = body.get("input")
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(f"invalid request input: {custom_id}")
        source = validate_source(json.loads(inputs[1]["content"]), custom_id)
        parsed.append({"custom_id": custom_id, "source": source})
    if not parsed:
        raise ValueError("request file is empty")
    return parsed


def prompt_for(
    tokenizer: Any,
    source: dict,
    correction_attempt: int | None = None,
    correction_error: str | None = None,
) -> str:
    required_ids = [
        candidate["candidate_id"] for candidate in source["candidates"]
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    **source,
                    "source_language": LANGUAGE_NAMES[source["source_language"]],
                    "target_language": LANGUAGE_NAMES[source["target_language"]],
                    "output_contract": {
                        "required_source_id": source["source_id"],
                        "required_candidate_ids": required_ids,
                        "correction_attempt": correction_attempt,
                        "prior_schema_error": correction_error,
                        "instruction": (
                            "Return fresh valid JSON only. Use exactly the required "
                            "fields and enumerated tags."
                            if correction_attempt is not None
                            else "Return valid JSON only."
                        ),
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    template_args = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **template_args
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **template_args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-license", default="Apache-2.0")
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path("Research/translation/models/hf-cache"),
    )
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--maximum-retries", type=int, default=2)
    parser.add_argument("--no-shared-prefix-cache", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for output in (args.output, args.report):
        if output.exists() and output.stat().st_size:
            raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    if args.batch_size < 1 or args.max_tokens < 1 or args.maximum_retries < 0:
        raise SystemExit(
            "batch-size and max-tokens must be positive; maximum-retries cannot be negative"
        )
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive")

    try:
        requests = read_requests(args.requests)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid local judge request: {error}") from error
    if args.limit is not None:
        requests = requests[: args.limit]

    # Keep heavyweight MLX dependencies out of import-time validation tests.
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import batch_generate, load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=args.hf_home,
        )
    )
    model, tokenizer = load(str(snapshot))
    mx.eval(model.parameters())
    prompts = [prompt_for(tokenizer, row["source"]) for row in requests]
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    shared_prefix_cache = None
    shared_prefix_tokens = 0
    if not args.no_shared_prefix_cache:
        shared_prefix_tokens = min(map(len, encoded_prompts))
        for index in range(shared_prefix_tokens):
            token = encoded_prompts[0][index]
            if any(prompt[index] != token for prompt in encoded_prompts[1:]):
                shared_prefix_tokens = index
                break
        if shared_prefix_tokens:
            shared_prefix_cache = make_prompt_cache(model)
            model(
                mx.array(
                    [encoded_prompts[0][:shared_prefix_tokens]], dtype=mx.int32
                ),
                cache=shared_prefix_cache,
            )
            mx.eval([cache.state for cache in shared_prefix_cache])

    sampler = make_sampler(temp=0.0)
    judge_model = f"{args.model}@{args.revision}:multi-candidate-v1"
    fingerprint = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    results: list[dict] = []
    corrected_responses = 0
    corrected_source_ids = 0
    normalized_error_tags = 0
    for start in range(0, len(requests), args.batch_size):
        batch_rows = requests[start : start + args.batch_size]
        batch_prompts = [
            prompt[shared_prefix_tokens:]
            for prompt in encoded_prompts[start : start + args.batch_size]
        ]
        prompt_caches = (
            [copy.deepcopy(shared_prefix_cache) for _ in batch_prompts]
            if shared_prefix_cache is not None
            else None
        )
        response = batch_generate(
            model,
            tokenizer,
            batch_prompts,
            prompt_caches=prompt_caches,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        accepted_payloads: dict[int, tuple[dict, str, int]] = {}
        pending: dict[int, tuple[str, str]] = {}
        for index, (row, raw) in enumerate(
            zip(batch_rows, response.texts, strict=True)
        ):
            try:
                payload = parse_payload(
                    raw,
                    row["source"],
                    allow_source_id_correction=True,
                )
            except ValueError as error:
                pending[index] = (str(error), raw)
            else:
                accepted_payloads[index] = (payload, raw, 0)

        for attempt in range(1, args.maximum_retries + 1):
            if not pending:
                break
            retry_indices = sorted(pending)
            retry_prompts = [
                tokenizer.encode(
                    prompt_for(
                        tokenizer,
                        batch_rows[index]["source"],
                        correction_attempt=attempt,
                        correction_error=pending[index][0],
                    )
                )
                for index in retry_indices
            ]
            retry_response = batch_generate(
                model,
                tokenizer,
                retry_prompts,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            next_pending: dict[int, tuple[str, str]] = {}
            for index, raw in zip(retry_indices, retry_response.texts, strict=True):
                try:
                    payload = parse_payload(
                        raw,
                        batch_rows[index]["source"],
                        allow_source_id_correction=True,
                    )
                except ValueError as error:
                    next_pending[index] = (str(error), raw)
                else:
                    accepted_payloads[index] = (payload, raw, attempt)
                    corrected_responses += 1
            pending = next_pending

        if pending:
            index = sorted(pending)[0]
            error, raw = pending[index]
            row = batch_rows[index]
            raise SystemExit(
                f"invalid local judge result after {args.maximum_retries} retries "
                f"for {row['custom_id']}: {error}; response={raw[:2000]!r}"
            )

        for index, row in enumerate(batch_rows):
            payload, raw, retry_count = accepted_payloads[index]
            source_id_corrected = bool(
                payload.pop("_mimi_source_id_corrected", False)
            )
            normalized_error_tags += int(
                payload.pop("_mimi_normalized_error_tags", 0)
            )
            corrected_source_ids += int(source_id_corrected)
            response_id = "qwen-consensus-" + hashlib.sha256(
                (row["custom_id"] + raw).encode()
            ).hexdigest()[:24]
            results.append(
                {
                    "custom_id": row["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": response_id,
                            "model": judge_model,
                            "system_fingerprint": fingerprint,
                            "retry_count": retry_count,
                            "source_id_corrected": source_id_corrected,
                            "output_text": json.dumps(
                                payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    },
                }
            )
        print(f"judged {len(results)}/{len(requests)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in results
        ),
        encoding="utf-8",
    )
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "blinded local multi-candidate MT consensus judge",
        "claimEligible": False,
        "candidateOriginExposed": False,
        "reasoningTraceStored": False,
        "judgeModel": args.model,
        "judgeRevision": args.revision,
        "judgeIdentity": judge_model,
        "judgeLicense": args.model_license,
        "modelBytes": directory_bytes(snapshot),
        "batchSize": args.batch_size,
        "maximumOutputTokens": args.max_tokens,
        "maximumRetries": args.maximum_retries,
        "correctedResponses": corrected_responses,
        "correctedSourceIds": corrected_source_ids,
        "normalizedErrorTags": normalized_error_tags,
        "sharedPrefixCacheTokens": shared_prefix_tokens,
        "systemPromptSHA256": fingerprint,
        "requestInput": {
            "path": str(args.requests.resolve()),
            "sha256": sha256(args.requests),
            "limitedRows": args.limit,
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "rows": len(results),
        },
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(results),
                "output": str(args.output),
                "output_sha256": sha256(args.output),
                "report": str(args.report),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
