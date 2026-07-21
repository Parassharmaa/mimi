#!/usr/bin/env python3
"""Benchmark exact full-sentence draft verification for MLX Marian."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from benchmark_marian_partial_retranslation import (
    DIRECTIONS,
    generate_with_parallel_draft_verification,
    load_direction_model,
    load_suite,
    percentile,
    select_rows,
    sha256,
    token_sha256,
)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def timed_teacher(model, source_tokens: list[int], maximum_tokens: int):
    started = time.perf_counter()
    output = model.generate_cached(source_tokens, maximum_tokens)
    mx.synchronize()
    return output, time.perf_counter() - started


def timed_draft_verified(
    teacher,
    draft,
    source_tokens: list[int],
    maximum_tokens: int,
):
    started = time.perf_counter()
    draft_output = draft.generate_cached(source_tokens, maximum_tokens)
    output, diagnostics = generate_with_parallel_draft_verification(
        teacher,
        source_tokens,
        draft_output,
        maximum_tokens,
    )
    mx.synchronize()
    diagnostics["generatedDraftTokens"] = len(draft_output)
    return output, diagnostics, time.perf_counter() - started


def summarize(cases: list[dict]) -> dict:
    baseline = [case["teacherLatencySeconds"] for case in cases]
    candidate = [case["draftVerifiedLatencySeconds"] for case in cases]
    attempted = [case["verification"] for case in cases if case["verification"]["attempted"]]
    draft_tokens = sum(item["draftTokenCount"] for item in attempted)
    accepted_tokens = sum(item["acceptedDraftTokens"] for item in attempted)
    return {
        "cases": len(cases),
        "exactOutputParity": all(case["exactOutputParity"] for case in cases),
        "attemptedDrafts": len(attempted),
        "bypassedDrafts": len(cases) - len(attempted),
        "draftTokens": draft_tokens,
        "acceptedDraftTokens": accepted_tokens,
        "draftTokenAcceptanceRate": accepted_tokens / draft_tokens if draft_tokens else 0.0,
        "fullDraftAndEOSAcceptanceRate": (
            sum(item["fullDraftAndEOSAccepted"] for item in attempted) / len(attempted)
            if attempted
            else 0.0
        ),
        "teacherTotalSeconds": sum(baseline),
        "draftVerifiedTotalSeconds": sum(candidate),
        "totalSpeedup": sum(baseline) / sum(candidate),
        "teacherP50Seconds": statistics.median(baseline),
        "teacherP95Seconds": percentile(baseline, 0.95),
        "draftVerifiedP50Seconds": statistics.median(candidate),
        "draftVerifiedP95Seconds": percentile(candidate, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja-teacher", type=Path, required=True)
    parser.add_argument("--ja-en-teacher", type=Path, required=True)
    parser.add_argument("--en-ja-draft", type=Path, required=True)
    parser.add_argument("--ja-en-draft", type=Path, required=True)
    parser.add_argument("--cases-per-domain", type=int, default=10)
    parser.add_argument("--maximum-tokens", type=int, default=128)
    parser.add_argument("--warm-runs", type=int, default=2)
    args = parser.parse_args()
    if args.cases_per_domain < 1 or args.maximum_tokens < 1 or args.warm_runs < 0:
        raise SystemExit("case/token counts must be positive and warm runs non-negative")

    selected = select_rows(load_suite(args.suite), args.cases_per_domain)
    teacher_paths = {
        ("en-US", "ja-JP"): args.en_ja_teacher,
        ("ja-JP", "en-US"): args.ja_en_teacher,
    }
    draft_paths = {
        ("en-US", "ja-JP"): args.en_ja_draft,
        ("ja-JP", "en-US"): args.ja_en_draft,
    }
    cases: list[dict] = []
    model_records: dict[str, dict] = {}

    for direction, rows in selected.items():
        teacher, teacher_tokenizer, teacher_record = load_direction_model(
            teacher_paths[direction]
        )
        draft, draft_tokenizer, draft_record = load_direction_model(draft_paths[direction])
        teacher_tokenizer_sha = sha256(teacher_paths[direction] / "tokenizer.json")
        draft_tokenizer_sha = sha256(draft_paths[direction] / "tokenizer.json")
        if teacher_tokenizer_sha != draft_tokenizer_sha:
            raise SystemExit(f"teacher and draft tokenizers differ for {direction}")
        label = DIRECTIONS[direction]
        model_records[label] = {
            "teacher": teacher_record,
            "draft": draft_record,
            "sharedTokenizerSHA256": teacher_tokenizer_sha,
        }

        warm_tokens = teacher_tokenizer.encode(str(rows[0]["source"]))
        for _ in range(args.warm_runs):
            timed_teacher(teacher, warm_tokens, args.maximum_tokens)
            timed_draft_verified(teacher, draft, warm_tokens, args.maximum_tokens)

        for index, row in enumerate(rows):
            source_tokens = teacher_tokenizer.encode(str(row["source"]))
            if source_tokens != draft_tokenizer.encode(str(row["source"])):
                raise SystemExit(f"teacher and draft tokenization differs for {row['id']}")
            if index % 2 == 0:
                baseline, baseline_latency = timed_teacher(
                    teacher, source_tokens, args.maximum_tokens
                )
                candidate, verification, candidate_latency = timed_draft_verified(
                    teacher, draft, source_tokens, args.maximum_tokens
                )
            else:
                candidate, verification, candidate_latency = timed_draft_verified(
                    teacher, draft, source_tokens, args.maximum_tokens
                )
                baseline, baseline_latency = timed_teacher(
                    teacher, source_tokens, args.maximum_tokens
                )
            exact = baseline == candidate
            if not exact:
                raise SystemExit(f"model-draft verification changed output for {row['id']}")
            cases.append(
                {
                    "id": row["id"],
                    "direction": f"{direction[0]}>{direction[1]}",
                    "domain": row["domain"],
                    "sourceTokenCount": len(source_tokens),
                    "outputTokenCount": len(baseline),
                    "outputTokenSHA256": token_sha256(baseline),
                    "teacherLatencySeconds": baseline_latency,
                    "draftVerifiedLatencySeconds": candidate_latency,
                    "exactOutputParity": exact,
                    "verification": verification,
                }
            )

        del teacher, teacher_tokenizer, draft, draft_tokenizer
        gc.collect()
        mx.clear_cache()

    directions = {
        f"{direction[0]}>{direction[1]}": summarize(
            [
                case
                for case in cases
                if case["direction"] == f"{direction[0]}>{direction[1]}"
            ]
        )
        for direction in DIRECTIONS
    }
    unique_paths = set(teacher_paths.values()) | set(draft_paths.values())
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "development-only exact model-draft acceleration ablation",
        "promotionEligible": False,
        "algorithm": {
            "name": "shallow-model-full-draft-parallel-teacher-verification",
            "outputContract": "exact cached-greedy teacher token parity",
            "maximumTokens": args.maximum_tokens,
            "warmRuns": args.warm_runs,
        },
        "suite": {"path": str(args.suite.resolve()), "sha256": sha256(args.suite)},
        "models": model_records,
        "runtime": {
            "mlx": getattr(mx, "__version__", "unknown"),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "combinedUndeduplicatedModelBytes": sum(
                directory_bytes(path) for path in unique_paths
            ),
        },
        "directions": directions,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(directions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
