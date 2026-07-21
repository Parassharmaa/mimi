#!/usr/bin/env python3
"""Run a pinned, permissively licensed local LLM as a provisional MT judge."""

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

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import batch_generate, load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"
DEFAULT_REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
ERROR_TAGS = {
    "omission", "addition", "mistranslation", "negation", "entity", "number",
    "tense-aspect", "pronoun-role", "register", "unnatural", "source-copy", "other",
}
LANGUAGE_NAMES = {"en-US": "English", "ja-JP": "Japanese"}
SYSTEM_PROMPT = """You are a strict professional bilingual English-Japanese translation reviewer.
Silently form your own best translation and align every source meaning before judging.
Judge whether the candidate is publication-ready for the exact source in the requested direction.
Reject any lost or added meaning, changed subject/object or pronoun role, wrong negation,
tense/aspect, entity, number, idiom, phrasal verb, speech act, register, or unnatural language.
Surface word overlap is not evidence of correctness. Do not repair the candidate and do not
reveal reasoning. Return one compact JSON object only with exactly these fields:
{"adequacy":1,"fluency":1,"meaning_preserved":false,"critical_error":false,
"error_tags":["mistranslation"],"verdict":"reject"}
Scores are integers 1-5. Use verdict=accept only when adequacy=5, fluency=5,
meaning_preserved=true, critical_error=false, and error_tags is empty.

Calibration examples:
English: Where is the station? Japanese: 駅はどこですか？
{"adequacy":5,"fluency":5,"meaning_preserved":true,"critical_error":false,"error_tags":[],"verdict":"accept"}
English: I want that. Japanese: それが欲しい。
{"adequacy":5,"fluency":5,"meaning_preserved":true,"critical_error":false,"error_tags":[],"verdict":"accept"}
English: Do you have chocolate made in Holland? Japanese: オランダでチョコレートは作られていますか？
{"adequacy":2,"fluency":5,"meaning_preserved":false,"critical_error":false,"error_tags":["omission","mistranslation"],"verdict":"reject"}
English: I had checked two suitcases. Japanese: 私は2つのスーツケースをチェックしました。
{"adequacy":2,"fluency":4,"meaning_preserved":false,"critical_error":false,"error_tags":["mistranslation"],"verdict":"reject"}
English: I'm checking out tomorrow morning. Japanese: 明日朝、チェックします。
{"adequacy":2,"fluency":4,"meaning_preserved":false,"critical_error":false,"error_tags":["omission","mistranslation"],"verdict":"reject"}
English: I'll introduce you to my family doctor. Japanese: 私の家族医師を紹介します。
{"adequacy":3,"fluency":3,"meaning_preserved":false,"critical_error":false,"error_tags":["omission","unnatural"],"verdict":"reject"}
English: At the turn of the century. Japanese: 100年周期で。
{"adequacy":1,"fluency":3,"meaning_preserved":false,"critical_error":true,"error_tags":["mistranslation"],"verdict":"reject"}
Japanese: 明日の朝、チェックアウトします。 English: I'll check it tomorrow morning.
{"adequacy":2,"fluency":5,"meaning_preserved":false,"critical_error":false,"error_tags":["mistranslation"],"verdict":"reject"}
Japanese: 駅はどこですか？ English: Where is the station?
{"adequacy":5,"fluency":5,"meaning_preserved":true,"critical_error":false,"error_tags":[],"verdict":"accept"}"""


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


def parse_verdict(text: str, identifier: str) -> dict:
    text = re.sub(r"(?s)^.*?</think>\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match is None:
        raise SystemExit(f"judge returned no JSON object: {identifier}: {text[:160]}")
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise SystemExit(f"judge returned invalid JSON: {identifier}: {error}") from error
    required = {
        "adequacy", "fluency", "meaning_preserved", "critical_error", "error_tags", "verdict"
    }
    if set(verdict) != required:
        raise SystemExit(f"judge returned an unexpected schema: {identifier}")
    if verdict["adequacy"] not in range(1, 6) or verdict["fluency"] not in range(1, 6):
        raise SystemExit(f"judge returned an invalid score: {identifier}")
    if not isinstance(verdict["meaning_preserved"], bool) or not isinstance(
        verdict["critical_error"], bool
    ):
        raise SystemExit(f"judge returned an invalid boolean: {identifier}")
    if not isinstance(verdict["error_tags"], list) or any(
        not isinstance(tag, str) or not tag.strip()
        for tag in verdict["error_tags"]
    ):
        raise SystemExit(f"judge returned invalid error tags: {identifier}")
    verdict["error_tags"] = list(dict.fromkeys(
        tag if tag in ERROR_TAGS else "other"
        for tag in verdict["error_tags"]
    ))
    if verdict["verdict"] not in {"accept", "reject"}:
        raise SystemExit(f"judge returned an invalid verdict: {identifier}")
    strict_accept = (
        verdict["adequacy"] == 5
        and verdict["fluency"] == 5
        and verdict["meaning_preserved"] is True
        and verdict["critical_error"] is False
        and verdict["error_tags"] == []
    )
    if (verdict["verdict"] == "accept") != strict_accept:
        raise SystemExit(f"judge verdict contradicts its scores: {identifier}")
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-license", default="Apache-2.0")
    parser.add_argument("--hf-home", type=Path, default=Path("Research/translation/models/hf-cache"))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-shared-prefix-cache", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive")
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")

    candidates = rows(args.candidates)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    identifiers = [str(row.get("id", "")) for row in candidates]
    if not candidates or any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise SystemExit("candidate input is empty or has missing/duplicate IDs")
    if any(row.get("promotion_eligible") is not False for row in candidates):
        raise SystemExit("local judge only accepts explicitly promotion-ineligible candidates")

    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    snapshot = Path(snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.hf_home,
    ))
    model, tokenizer = load(str(snapshot))
    mx.eval(model.parameters())
    sampler = make_sampler(temp=0.0)
    prompts: list[str] = []
    for row in candidates:
        source_language = str(row.get("source_language", ""))
        target_language = str(row.get("target_language", ""))
        if source_language not in LANGUAGE_NAMES or target_language not in LANGUAGE_NAMES:
            raise SystemExit(f"unsupported candidate language pair: {row.get('id')}")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "source_language": LANGUAGE_NAMES[source_language],
                        "target_language": LANGUAGE_NAMES[target_language],
                        "source": row["source"],
                        "candidate": row["target"],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        template_args = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_args
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, **template_args)
        prompts.append(prompt)

    results = []
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
                mx.array([encoded_prompts[0][:shared_prefix_tokens]], dtype=mx.int32),
                cache=shared_prefix_cache,
            )
            mx.eval([cache.state for cache in shared_prefix_cache])

    for start in range(0, len(candidates), args.batch_size):
        batch_rows = candidates[start:start + args.batch_size]
        batch_prompts = [
            prompt[shared_prefix_tokens:]
            for prompt in encoded_prompts[start:start + args.batch_size]
        ]
        if any(not prompt for prompt in batch_prompts):
            raise SystemExit("shared prompt prefix consumed a complete candidate prompt")
        prompt_caches = (
            [copy.deepcopy(shared_prefix_cache) for _ in batch_prompts]
            if shared_prefix_cache is not None else None
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
        for row, raw in zip(batch_rows, response.texts, strict=True):
            results.append({
                "candidateID": row["id"],
                "sourceID": row.get("source_id"),
                "source": row["source"],
                "candidate": row["target"],
                "judgment": parse_verdict(raw, str(row["id"])),
            })
        print(f"judged {len(results)}/{len(candidates)}", flush=True)

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "strict local bilingual judge for provisional synthetic training data",
        "claimEligible": False,
        "judgeModel": args.model,
        "judgeRevision": args.revision,
        "judgeLicense": args.model_license,
        "modelBytes": directory_bytes(snapshot),
        "batchSize": args.batch_size,
        "sharedPrefixCacheTokens": shared_prefix_tokens,
        "systemPromptSHA256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "candidateInput": {
            "path": str(args.candidates.resolve()),
            "sha256": sha256(args.candidates),
            "limitedRows": args.limit,
        },
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    accepted = sum(row["judgment"]["verdict"] == "accept" for row in results)
    print(f"wrote {len(results)} judgments ({accepted} strict accepts) to {args.output}")


if __name__ == "__main__":
    main()
