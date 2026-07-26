#!/usr/bin/env python3
"""Mine free-running v12-step-50 prefixes for the v14 repair arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from audit_translation_structures import critical_tokens, tokens
from evaluate_canonical_sequence_v10_internal import generation_rows
from train_marian_distillation import checkpoint_identity, sha256
from transformers import MarianTokenizer
from typed_critical_token_policy import typed_preserves

EXPERIMENT = "canonical-rollout-repair-v14-ja-en"
SOURCE_EXPERIMENT = "canonical-safety-repair-v12-ja-en"
REPOSITORY = "Mitsua/elan-mt-bt-ja-en"
REVISION = "539f80eb05306e27a166b45e4264c7fa2eb4de97"
DATASET_MANIFEST_SHA256 = (
    "1cd2e3629513f4662c6c9ffd6854d463bd638f08c8001bdb73027db0dc03d245"
)
STEP50_MODEL_SHA256 = "cf67fb44a4e9a0991c95b5e87578a4427610cc8e4110fbd3bb03909356600f2b"
STEP50_MANIFEST_SHA256 = (
    "68de542ff5476800aab88b075cac05b5e3dd3da21330d15d1d7910a587a9c533"
)
HARD_LIMIT = 2_048


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def stable_rank(seed: int, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{seed}\0{row.get('id', '')}\0"
            f"{row.get('source', '')}\0{row.get('target', '')}"
        ).encode()
    ).hexdigest()


def first_repeat_trigger(
    token_ids: list[int],
) -> dict[str, int] | None:
    """Return the earliest third contiguous phrase repetition."""
    matches = []
    maximum_width = min(16, len(token_ids) // 3)
    for width in range(3, maximum_width + 1):
        for start in range(len(token_ids) - width * 3 + 1):
            phrase = token_ids[start : start + width]
            if (
                token_ids[start + width : start + width * 2] == phrase
                and token_ids[start + width * 2 : start + width * 3] == phrase
            ):
                matches.append(
                    {
                        "phrase_start": start,
                        "phrase_width": width,
                        "trigger_index": start + width * 2,
                    }
                )
    if not matches:
        return None
    return min(
        matches,
        key=lambda item: (
            item["trigger_index"],
            item["phrase_width"],
            item["phrase_start"],
        ),
    )


def failure_flags(
    row: dict[str, Any],
    generated: dict[str, Any],
) -> list[str]:
    source = str(row["source"])
    hypothesis = str(generated["hypothesis"])
    flags = []
    if critical_tokens(source) != critical_tokens(hypothesis):
        flags.append("exact")
    if not typed_preserves(
        source,
        hypothesis,
        "ja-JP",
        "en-US",
    ):
        flags.append("typed")
    if tokens(source)["negative"] != tokens(hypothesis)["negative"]:
        flags.append("negation-detector")
    if (
        generated["reached_generation_limit"]
        or generated["repeated_token_loop"]
        or not hypothesis.strip()
    ):
        flags.append("generation")
    return flags


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default="mps",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.batch_size < 1:
        raise SystemExit("batch size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    root = Path(__file__).resolve().parents[2]
    dataset_manifest_path = args.dataset_directory / "manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    train_path = args.dataset_directory / "train.jsonl"
    if (
        sha256(dataset_manifest_path) != DATASET_MANIFEST_SHA256
        or dataset_manifest.get("experiment") != EXPERIMENT
        or dataset_manifest.get("status") != "frozen-ready-for-rollout-mining"
        or dataset_manifest.get("outputs", {}).get("train", {}).get("sha256")
        != sha256(train_path)
        or dataset_manifest.get("counts", {}).get("train") != 7_104
    ):
        raise SystemExit("v14 dataset identity differs")

    model_path = args.checkpoint / "model.safetensors"
    training_manifest_path = args.checkpoint / "mimi_training_manifest.json"
    training_manifest = load_json(training_manifest_path)
    if (
        sha256(model_path) != STEP50_MODEL_SHA256
        or sha256(training_manifest_path) != STEP50_MANIFEST_SHA256
        or checkpoint_identity(args.checkpoint) != ("ja-en", REPOSITORY, REVISION)
        or training_manifest.get("experiment") != SOURCE_EXPERIMENT
        or training_manifest.get("checkpoint_step") != 50
        or training_manifest.get("promotion_eligible") is not False
    ):
        raise SystemExit("v12 step-50 rollout checkpoint differs")

    rows = load_jsonl(train_path)
    if len(rows) != 7_104:
        raise SystemExit("v14 train row count differs")
    tokenizer = MarianTokenizer.from_pretrained(args.checkpoint)
    generated = generation_rows(
        args.checkpoint,
        tokenizer,
        rows,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        maximum_source_tokens=args.max_source_tokens,
        maximum_target_tokens=args.max_target_tokens,
    )
    generated_by_id = {str(value["id"]): value for value in generated}
    if set(generated_by_id) != {str(row["id"]) for row in rows}:
        raise SystemExit("rollout generation does not cover v14 train")

    rollout_rows = []
    recovery_rows = []
    hard_candidates = []
    flag_counts: Counter[str] = Counter()
    for row in rows:
        value = generated_by_id[str(row["id"])]
        token_ids = [int(token) for token in value["output_token_ids"]]
        flags = failure_flags(row, value)
        for flag in flags:
            flag_counts[flag] += 1
        repeat = first_repeat_trigger(token_ids)
        recovery = None
        if repeat is not None:
            trigger = int(repeat["trigger_index"])
            recovery = {
                **repeat,
                "rejected_token_id": token_ids[trigger],
                "recovery_token_id": int(tokenizer.eos_token_id),
                "recovery_reason": "third-contiguous-phrase-repetition",
            }
        elif value["reached_generation_limit"] and token_ids:
            trigger = len(token_ids) - 1
            recovery = {
                "phrase_start": None,
                "phrase_width": None,
                "trigger_index": trigger,
                "rejected_token_id": token_ids[trigger],
                "recovery_token_id": int(tokenizer.eos_token_id),
                "recovery_reason": "generation-limit",
            }
        rollout = {
            "id": str(row["id"]),
            "source_sha256": hashlib.sha256(str(row["source"]).encode()).hexdigest(),
            "reference_sha256": hashlib.sha256(str(row["target"]).encode()).hexdigest(),
            "hypothesis": str(value["hypothesis"]),
            "output_token_ids": token_ids,
            "terminated": bool(value["terminated"]),
            "reached_generation_limit": bool(value["reached_generation_limit"]),
            "repeated_token_loop": bool(value["repeated_token_loop"]),
            "failure_flags": flags,
            "recovery": recovery,
        }
        rollout_rows.append(rollout)
        if recovery is not None:
            recovery_rows.append(
                {
                    "id": str(row["id"]),
                    "source": str(row["source"]),
                    "target": str(row["target"]),
                    "source_language": "ja-JP",
                    "target_language": "en-US",
                    "origin": str(row["origin"]),
                    "source_license": str(row["source_license"]),
                    "source_provenance": str(row["source_provenance"]),
                    "v14_stratum": str(row["v14_stratum"]),
                    "rollout_token_ids": token_ids,
                    "failure_flags": flags,
                    **recovery,
                    "rollout_is_positive_target": False,
                }
            )
        if flags:
            hard_candidates.append(
                {
                    **row,
                    "rollout_hypothesis": str(value["hypothesis"]),
                    "rollout_token_ids": token_ids,
                    "rollout_failure_flags": flags,
                    "rollout_is_positive_target": False,
                }
            )

    hard_rows = sorted(
        hard_candidates,
        key=lambda row: (
            0 if "generation" in row["rollout_failure_flags"] else 1,
            stable_rank(args.seed, row),
        ),
    )[:HARD_LIMIT]
    if not hard_rows:
        raise SystemExit("v14 rollout mining found no hard rows")
    if not recovery_rows:
        raise SystemExit(
            "v14 rollout mining found no real EOS/repetition recovery rows"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "rollouts": args.output / "rollouts.jsonl",
        "hard": args.output / "hard.jsonl",
        "recovery": args.output / "recovery.jsonl",
    }
    for key, values in (
        ("rollouts", rollout_rows),
        ("hard", hard_rows),
        ("recovery", recovery_rows),
    ):
        write_jsonl(output_paths[key], values)

    manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "rollout-mining-complete",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "direction": "ja-en",
        "operation": ("greedy free-running rollout mining from rejected v12 step 50"),
        "checkpoint": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "license": "CC-BY-SA-4.0",
            "model": record(model_path, root),
            "training_manifest": record(
                training_manifest_path,
                root,
            ),
            "promotion_eligible": False,
        },
        "dataset": {
            "manifest": record(dataset_manifest_path, root),
            "train": record(train_path, root),
        },
        "generation": {
            "device": args.device,
            "batch_size": args.batch_size,
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "do_sample": False,
            "num_beams": 1,
        },
        "selection": {
            "seed": args.seed,
            "hard_limit": HARD_LIMIT,
            "hard_policy": (
                "all generation failures first, then deterministic "
                "structure disagreements by stable SHA-256 rank"
            ),
            "recovery_policy": (
                "prefer EOS before the third contiguous 3-16-token phrase "
                "occurrence; otherwise prefer EOS at the generation limit"
            ),
        },
        "counts": {
            "rollouts": len(rollout_rows),
            "hard": len(hard_rows),
            "recovery": len(recovery_rows),
            "failure_flags": dict(sorted(flag_counts.items())),
            "hard_strata": dict(
                sorted(Counter(str(row["v14_stratum"]) for row in hard_rows).items())
            ),
            "recovery_reasons": dict(
                sorted(
                    Counter(
                        str(row["recovery_reason"]) for row in recovery_rows
                    ).items()
                )
            ),
        },
        "outputs": {
            key: {
                **record(path, root),
                "rows": len(values),
            }
            for key, path, values in (
                ("rollouts", output_paths["rollouts"], rollout_rows),
                ("hard", output_paths["hard"], hard_rows),
                ("recovery", output_paths["recovery"], recovery_rows),
            )
        },
        "rollout_strings_are_positive_targets": False,
        "recovery_target_is_only_eos": True,
        "free_form_synthetic_translations_used_as_targets": False,
        "private_reasoning_traces_used": False,
        "human_reviewer_required": False,
        "promotion_eligible": False,
        "training_authorized": False,
        "protected_evaluation_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": display_path(args.output, root),
                "manifest_sha256": sha256(manifest_path),
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
