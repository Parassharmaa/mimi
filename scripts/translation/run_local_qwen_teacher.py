#!/usr/bin/env python3
"""Generate reference-hidden EN<->JA targets with a pinned local Qwen teacher."""

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
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
LANGUAGE_NAMES = {"en-US": "English", "ja-JP": "Japanese"}
SYSTEM_PROMPT = """You are a professional English-Japanese translation engine.
Translate the supplied source faithfully into the requested target language.
Preserve every fact, entity, number, date, placeholder, URL, negation, modality,
speaker role, and level of formality. Produce natural target-language wording.
Return only the final translation. Do not explain, analyze, or reveal reasoning."""
LITERAL_SYSTEM_PROMPT = SYSTEM_PROMPT + """
Prefer an explicit, close translation over paraphrase. Preserve source clause
order when natural, and do not compress, generalize, or omit modifiers."""
NUMBER_PRESERVING_SYSTEM_PROMPT = LITERAL_SYSTEM_PROMPT + """
Never introduce an Arabic numeral that is not written as an Arabic numeral in
the source. Render spelled-out numbers, dates, and month names with target-
language words or kanji rather than converting them to digits."""
PROPER_NOUN_SYSTEM_PROMPT = LITERAL_SYSTEM_PROMPT + """
Treat romanized Japanese personal and place names as proper nouns. Use their
established Japanese kanji only when known; otherwise transliterate them. Never
replace a proper name with a similar-looking common noun or unrelated name."""
PROMPT_PROFILES = {
    "baseline": SYSTEM_PROMPT,
    "literal": LITERAL_SYSTEM_PROMPT,
    "number-preserving": NUMBER_PRESERVING_SYSTEM_PROMPT,
    "proper-noun": PROPER_NOUN_SYSTEM_PROMPT,
}


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing suite: {path}")
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


def clean_output(text: str) -> str:
    text = re.sub(r"(?s)^.*?</think>\s*", "", text).strip()
    text = re.sub(
        r"(?i)^(?:final\s+)?(?:translation|english|japanese)\s*:\s*",
        "",
        text,
    ).strip()
    pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("「", "」"))
    for opening, closing in pairs:
        if len(text) >= 2 and text.startswith(opening) and text.endswith(closing):
            text = text[len(opening):-len(closing)].strip()
            break
    return text


def prompt_for(
    tokenizer,
    source: str,
    source_language: str,
    target_language: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    try:
        source_name = LANGUAGE_NAMES[source_language]
        target_name = LANGUAGE_NAMES[target_language]
    except KeyError as error:
        raise SystemExit(f"unsupported language: {error.args[0]}") from error
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Source language: {source_name}\n"
                f"Target language: {target_name}\n"
                f"Source text:\n{source}"
            ),
        },
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def reusable_results(
    report_path: Path,
    suite: list[dict],
    *,
    suite_path: Path,
    model: str,
    revision: str,
    model_license: str,
) -> tuple[dict, dict[str, dict]]:
    if not report_path.is_file():
        raise SystemExit(f"missing reusable teacher report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("claimEligible") is not False
        or report.get("referenceExposedToTeacher") is not False
        or report.get("studentHypothesisExposedToTeacher") is not False
        or report.get("reasoningTraceRequestedOrStored") is not False
        or report.get("modelRepository") != model
        or report.get("modelRevision") != revision
        or report.get("modelLicense") != model_license
        or report.get("suite", {}).get("sha256") != sha256(suite_path)
    ):
        raise SystemExit("reusable report violates the pinned hidden-reference contract")
    indexed: dict[str, dict] = {}
    for result in report.get("results", []):
        identifier = str(result.get("caseID", ""))
        if not identifier or identifier in indexed:
            raise SystemExit("reusable report has missing or duplicate case IDs")
        indexed[identifier] = result
    if set(indexed) != {str(row["id"]) for row in suite}:
        raise SystemExit("reusable report does not cover the exact full suite")
    for row in suite:
        result = indexed[str(row["id"])]
        for field in ("sourceLanguage", "targetLanguage", "domain", "source", "references"):
            if result.get(field) != row.get(field):
                raise SystemExit(f"reusable report disagrees with suite {field}: {row['id']}")
    return report, indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-license", default="Apache-2.0")
    parser.add_argument("--hf-home", type=Path, default=Path("Research/translation/models/hf-cache"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-shared-prefix-cache", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse-report", type=Path)
    parser.add_argument("--retry-domain", action="append", default=[])
    parser.add_argument("--retry-direction", choices=("en-ja", "ja-en"), action="append", default=[])
    parser.add_argument("--retry-case-id", action="append", default=[])
    parser.add_argument("--prompt-profile", choices=tuple(PROMPT_PROFILES), default="baseline")
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive")
    if args.batch_size < 1 or args.max_tokens < 1:
        raise SystemExit("batch-size and max-tokens must be positive")
    if not 0.0 <= args.temperature <= 2.0:
        raise SystemExit("temperature must be between zero and two")
    if not 0.0 <= args.top_p <= 1.0:
        raise SystemExit("top-p must be between zero and one")
    if args.seed < 0:
        raise SystemExit("seed cannot be negative")
    if args.retry_domain and args.reuse_report is None:
        raise SystemExit("retry-domain requires reuse-report")
    if args.retry_direction and args.reuse_report is None:
        raise SystemExit("retry-direction requires reuse-report")
    if args.retry_case_id and args.reuse_report is None:
        raise SystemExit("retry-case-id requires reuse-report")
    if args.reuse_report is not None and not args.retry_domain:
        raise SystemExit("reuse-report requires at least one retry-domain")
    if args.reuse_report is not None and args.limit is not None:
        raise SystemExit("limit cannot be combined with reuse-report")

    suite = rows(args.suite)
    identifiers = [str(row.get("id", "")) for row in suite]
    if not suite or any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise SystemExit("suite is empty or has missing/duplicate IDs")
    for row in suite:
        if row.get("claimEligible") is not False or row.get("referenceExposedToTeacher") is not False:
            raise SystemExit(f"suite lacks reference-hidden training-only marking: {row.get('id')}")
        if not str(row.get("source", "")).strip():
            raise SystemExit(f"suite has an empty source: {row.get('id')}")
    reusable_report = None
    reused_results: dict[str, dict] = {}
    if args.reuse_report is not None:
        reusable_report, reused_results = reusable_results(
            args.reuse_report,
            suite,
            suite_path=args.suite,
            model=args.model,
            revision=args.revision,
            model_license=args.model_license,
        )
        unknown_domains = set(args.retry_domain) - {str(row.get("domain")) for row in suite}
        if unknown_domains:
            raise SystemExit(f"retry-domain not present in suite: {sorted(unknown_domains)}")
        retry_directions = {
            {"en-ja": ("en-US", "ja-JP"), "ja-en": ("ja-JP", "en-US")}[direction]
            for direction in args.retry_direction
        }
        unknown_case_ids = set(args.retry_case_id) - set(identifiers)
        if unknown_case_ids:
            raise SystemExit(f"retry-case-id not present in suite: {sorted(unknown_case_ids)}")
        generation_suite = [
            row
            for row in suite
            if str(row.get("domain")) in set(args.retry_domain)
            and (
                not retry_directions
                or (row.get("sourceLanguage"), row.get("targetLanguage"))
                in retry_directions
            )
            and (
                not args.retry_case_id
                or str(row.get("id")) in set(args.retry_case_id)
            )
        ]
    else:
        generation_suite = suite[:args.limit] if args.limit is not None else suite

    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    from huggingface_hub import snapshot_download
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    preparation_started = time.perf_counter()
    local_model = Path(args.model)
    if local_model.is_dir():
        snapshot = local_model.resolve()
        engine_revision = "local"
    else:
        snapshot = Path(snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            cache_dir=args.hf_home,
        ))
        engine_revision = args.revision
    model, tokenizer = load(str(snapshot))
    mx.eval(model.parameters())
    preparation_seconds = time.perf_counter() - preparation_started
    mx.random.seed(args.seed)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

    # This is the central non-leakage boundary: prompt_for receives only source
    # and language IDs, never the suite row, student hypothesis, or references.
    prompts = [
        prompt_for(
            tokenizer,
            str(row["source"]),
            str(row["sourceLanguage"]),
            str(row["targetLanguage"]),
            system_prompt=PROMPT_PROFILES[args.prompt_profile],
        )
        for row in generation_suite
    ]
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    shared_prefix_cache = None
    shared_prefix_tokens = 0
    if not args.no_shared_prefix_cache and len(encoded_prompts) > 1:
        shared_prefix_tokens = min(map(len, encoded_prompts))
        for index in range(shared_prefix_tokens):
            token = encoded_prompts[0][index]
            if any(prompt[index] != token for prompt in encoded_prompts[1:]):
                shared_prefix_tokens = index
                break
        if shared_prefix_tokens:
            shared_prefix_cache = make_prompt_cache(model)
            model(
                mx.array([encoded_prompts[0][:shared_prefix_tokens]], dtype=mx.int32),
                cache=shared_prefix_cache,
            )
            mx.eval([cache.state for cache in shared_prefix_cache])

    generated: list[tuple[str, float]] = []
    generation_started = time.perf_counter()
    for start in range(0, len(generation_suite), args.batch_size):
        batch_prompts = [
            prompt[shared_prefix_tokens:]
            for prompt in encoded_prompts[start:start + args.batch_size]
        ]
        if any(not prompt for prompt in batch_prompts):
            raise SystemExit("shared prefix consumed a complete source prompt")
        prompt_caches = (
            [copy.deepcopy(shared_prefix_cache) for _ in batch_prompts]
            if shared_prefix_cache is not None else None
        )
        started = time.perf_counter()
        response = batch_generate(
            model,
            tokenizer,
            batch_prompts,
            prompt_caches=prompt_caches,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        per_case = elapsed / len(batch_prompts)
        generated.extend((clean_output(text), per_case) for text in response.texts)
        print(f"translated {len(generated)}/{len(generation_suite)}", flush=True)
    generation_seconds = time.perf_counter() - generation_started

    generated_results: dict[str, dict] = {}
    for row, (hypothesis, latency) in zip(generation_suite, generated, strict=True):
        if not hypothesis:
            raise SystemExit(f"teacher produced an empty translation: {row['id']}")
        generated_results[str(row["id"])] = {
            "caseID": row["id"],
            "sourceLanguage": row["sourceLanguage"],
            "targetLanguage": row["targetLanguage"],
            "domain": row["domain"],
            "source": row["source"],
            "references": row["references"],
            "claimEligible": False,
            "hypothesis": hypothesis,
            "latencySeconds": latency,
            "warmLatencySeconds": [],
        }
    if reusable_report is not None:
        combined_results = {**reused_results, **generated_results}
        results = [combined_results[str(row["id"])] for row in suite]
    else:
        results = [generated_results[str(row["id"])] for row in generation_suite]
    retry_suffix = ":targeted-retry" if reusable_report is not None else ""
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": (
            f"mlx-lm:{args.model}@{engine_revision[:12]}:reference-hidden-translation"
            f"{retry_suffix}"
        ),
        "purpose": "local sequence teacher for training-only EN-JA distillation",
        "claimEligible": False,
        "referenceExposedToTeacher": False,
        "studentHypothesisExposedToTeacher": False,
        "reasoningTraceRequestedOrStored": False,
        "modelRepository": args.model,
        "modelRevision": engine_revision,
        "modelLicense": args.model_license,
        "modelBytes": directory_bytes(snapshot),
        "batchSize": args.batch_size,
        "sharedPrefixCacheEnabled": not args.no_shared_prefix_cache,
        "sharedPrefixCacheTokens": shared_prefix_tokens,
        "systemPromptProfile": args.prompt_profile,
        "systemPromptSHA256": hashlib.sha256(
            PROMPT_PROFILES[args.prompt_profile].encode()
        ).hexdigest(),
        "sampling": {
            "temperature": args.temperature,
            "topP": args.top_p,
            "seed": args.seed,
        },
        "suite": {
            "path": str(args.suite.resolve()),
            "sha256": sha256(args.suite),
            "limitedRows": args.limit,
        },
        "targetedRetry": (
            {
                "parentReport": {
                    "path": str(args.reuse_report.resolve()),
                    "sha256": sha256(args.reuse_report),
                },
                "domains": sorted(set(args.retry_domain)),
                "directions": sorted(set(args.retry_direction)),
                "caseIDs": sorted(set(args.retry_case_id)),
                "regeneratedRows": len(generation_suite),
                "reusedRows": len(suite) - len(generation_suite),
            }
            if args.reuse_report is not None else None
        ),
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "generationSeconds": generation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(results)} reference-hidden teacher translations to {args.output}")


if __name__ == "__main__":
    main()
