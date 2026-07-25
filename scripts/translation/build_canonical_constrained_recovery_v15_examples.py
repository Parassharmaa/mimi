#!/usr/bin/env python3
"""Build licensed-continuation recovery and omission contrasts for v15."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from filter_training_dataset_against_protected import normalized, sha256
from transformers import MarianTokenizer

EXPERIMENT = "canonical-constrained-recovery-v15-ja-en"
DATASET_MANIFEST_SHA256 = (
    "9b412e0a7d49234ab374f4e47fc71e0f70e9cb432af6f792d26e7ce56910c523"
)
V14_ROLLOUT_MANIFEST_SHA256 = (
    "f93ecd7d724e37f468321cca8fbf3e9ac472ee290bb2f979daa380cb5dddd4e4"
)
INITIAL_MODEL_SHA256 = (
    "cf67fb44a4e9a0991c95b5e87578a4427610cc8e4110fbd3bb03909356600f2b"
)
RECOVERY_LIMIT = 2_048
ALIGNED_RECOVERY_LIMIT = 1_024
OMISSION_LIMIT = 2_048


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not values:
        raise SystemExit(f"expected non-empty JSONL: {path}")
    return values


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


def stable_rank(seed: int, role: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{seed}\0{role}\0{row.get('id', '')}\0"
            f"{row.get('source', '')}\0{row.get('target', '')}"
        ).encode()
    ).hexdigest()


def target_token_ids(
    tokenizer: MarianTokenizer,
    text: str,
    *,
    maximum_tokens: int,
) -> list[int]:
    encoded = tokenizer(
        text_target=text,
        truncation=True,
        max_length=maximum_tokens,
    )
    return [int(value) for value in encoded["input_ids"]]


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def aligned_recovery(
    row: dict[str, Any],
    rollout: dict[str, Any],
    target_ids: list[int],
    *,
    special_ids: set[int],
    maximum_tokens: int,
) -> dict[str, Any] | None:
    rollout_ids = [int(value) for value in rollout["rollout_token_ids"]]
    matcher = difflib.SequenceMatcher(
        None,
        target_ids,
        rollout_ids,
        autojunk=False,
    )
    for (
        tag,
        target_start,
        target_end,
        rollout_start,
        rollout_end,
    ) in matcher.get_opcodes():
        if tag == "equal":
            continue
        if (
            target_start < 2
            or target_start >= len(target_ids)
            or rollout_start >= len(rollout_ids)
            or tag == "delete"
        ):
            continue
        correct = int(target_ids[target_start])
        rejected = int(rollout_ids[rollout_start])
        generated_fragment = rollout_ids[
            rollout_start : min(rollout_end, rollout_start + 3)
        ]
        perturbed = [*target_ids[:target_start], *generated_fragment]
        if (
            correct == rejected
            or correct in special_ids
            or rejected in special_ids
            or len(perturbed) >= maximum_tokens
        ):
            continue
        return {
            "id": f"v15-recovery:aligned:{row['id']}",
            "source": row["source"],
            "target": row["target"],
            "source_language": row["source_language"],
            "target_language": row["target_language"],
            "source_license": row["source_license"],
            "source_provenance": row["source_provenance"],
            "origin": row["origin"],
            "v15_stratum": row["v15_stratum"],
            "contrast_role": "aligned-free-running-first-error",
            "clean_prefix_token_ids": target_ids[:target_start],
            "perturbed_prefix_token_ids": perturbed,
            "correct_token_id": correct,
            "rejected_token_id": rejected,
            "positive_target_source": "licensed-human-reference",
            "generated_strings_are_positive_targets": False,
            "rollout_hypothesis_sha256": hashlib.sha256(
                str(rollout["rollout_hypothesis"]).encode()
            ).hexdigest(),
        }
    return None


def repeated_prefix_recovery(
    row: dict[str, Any],
    target_ids: list[int],
    *,
    seed: int,
    special_ids: set[int],
    maximum_tokens: int,
) -> dict[str, Any] | None:
    usable_end = len(target_ids) - 1
    if usable_end < 10:
        return None
    digest = hashlib.sha256(f"{seed}\0repeat\0{row['id']}".encode()).digest()
    width = 3 + digest[0] % 6
    if usable_end <= width + 2:
        return None
    span = usable_end - width - 1
    end = width + 1 + int.from_bytes(digest[1:5], "big") % span
    phrase = target_ids[end - width : end]
    clean = target_ids[:end]
    perturbed = [*clean, *phrase]
    correct = int(target_ids[end])
    rejected = int(phrase[0])
    if (
        correct == rejected
        or correct in special_ids
        or rejected in special_ids
        or len(perturbed) >= maximum_tokens
    ):
        return None
    return {
        "id": f"v15-recovery:reference-repeat:{row['id']}",
        "source": row["source"],
        "target": row["target"],
        "source_language": row["source_language"],
        "target_language": row["target_language"],
        "source_license": row["source_license"],
        "source_provenance": row["source_provenance"],
        "origin": row["origin"],
        "v15_stratum": row["v15_stratum"],
        "contrast_role": "licensed-reference-third-repeat-counterfactual",
        "clean_prefix_token_ids": clean,
        "perturbed_prefix_token_ids": perturbed,
        "correct_token_id": correct,
        "rejected_token_id": rejected,
        "repeat_phrase_width": width,
        "positive_target_source": "licensed-human-reference-continuation",
        "generated_strings_are_positive_targets": False,
    }


def omission_contrast(
    row: dict[str, Any],
    target_ids: list[int],
    *,
    seed: int,
    special_ids: set[int],
) -> dict[str, Any] | None:
    usable_end = len(target_ids) - 1
    if usable_end < 8:
        return None
    digest = hashlib.sha256(f"{seed}\0omission\0{row['id']}".encode()).digest()
    width = 2 + digest[0] % 5
    if usable_end <= width + 2:
        return None
    start = 2 + int.from_bytes(digest[1:5], "big") % (usable_end - width - 1)
    after = start + width
    correct = int(target_ids[start])
    rejected = int(target_ids[after])
    if correct == rejected or correct in special_ids or rejected in special_ids:
        return None
    return {
        "id": f"v15-omission:{row['id']}",
        "source": row["source"],
        "target": row["target"],
        "source_language": row["source_language"],
        "target_language": row["target_language"],
        "source_license": row["source_license"],
        "source_provenance": row["source_provenance"],
        "origin": row["origin"],
        "v15_stratum": row["v15_stratum"],
        "contrast_role": "licensed-reference-token-span-deletion",
        "clean_prefix_token_ids": target_ids[:start],
        "correct_token_id": correct,
        "rejected_token_id": rejected,
        "omitted_token_ids": target_ids[start:after],
        "omitted_token_count": width,
        "positive_target_source": "licensed-human-reference",
        "generated_strings_are_positive_targets": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("v14_rollout_directory", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--maximum-target-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    root = Path(__file__).resolve().parents[2]
    dataset_manifest_path = args.dataset_directory / "manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    train_path = args.dataset_directory / "train.jsonl"
    if (
        sha256(dataset_manifest_path) != DATASET_MANIFEST_SHA256
        or dataset_manifest.get("experiment") != EXPERIMENT
        or dataset_manifest.get("status")
        != "frozen-ready-for-contrastive-example-building"
        or dataset_manifest.get("outputs", {}).get("train", {}).get("sha256")
        != sha256(train_path)
        or dataset_manifest.get("counts", {}).get("train") != 7_104
        or dataset_manifest.get("distribution_provenance", {}).get(
            "all_positive_targets_are_licensed_human_references"
        )
        is not True
    ):
        raise SystemExit("v15 dataset identity or safety state differs")

    rollout_manifest_path = args.v14_rollout_directory / "manifest.json"
    rollout_manifest = load_json(rollout_manifest_path)
    hard_path = args.v14_rollout_directory / "hard.jsonl"
    if (
        sha256(rollout_manifest_path) != V14_ROLLOUT_MANIFEST_SHA256
        or rollout_manifest.get("experiment") != "canonical-rollout-repair-v14-ja-en"
        or rollout_manifest.get("outputs", {}).get("hard", {}).get("sha256")
        != sha256(hard_path)
        or rollout_manifest.get("rollout_strings_are_positive_targets") is not False
    ):
        raise SystemExit("v14 rollout evidence differs")

    checkpoint_model_path = args.checkpoint / "model.safetensors"
    if sha256(checkpoint_model_path) != INITIAL_MODEL_SHA256:
        raise SystemExit("v15 tokenizer checkpoint differs")
    tokenizer = MarianTokenizer.from_pretrained(args.checkpoint)
    special_ids = {
        int(value) for value in tokenizer.all_special_ids if value is not None
    }

    train_rows = load_jsonl(train_path)
    hard_by_source = {
        normalized(str(row["source"])): row for row in load_jsonl(hard_path)
    }
    tokenized = {
        str(row["id"]): target_token_ids(
            tokenizer,
            str(row["target"]),
            maximum_tokens=args.maximum_target_tokens,
        )
        for row in train_rows
    }

    aligned_candidates = []
    for row in train_rows:
        rollout = hard_by_source.get(normalized(str(row["source"])))
        if rollout is None:
            continue
        value = aligned_recovery(
            row,
            rollout,
            tokenized[str(row["id"])],
            special_ids=special_ids,
            maximum_tokens=args.maximum_target_tokens,
        )
        if value is not None:
            aligned_candidates.append(value)
    aligned_candidates.sort(
        key=lambda row: (stable_rank(args.seed, "aligned", row), str(row["id"]))
    )
    aligned_rows = aligned_candidates[:ALIGNED_RECOVERY_LIMIT]
    used_sources = {normalized(str(row["source"])) for row in aligned_rows}

    repeated_candidates = []
    for row in train_rows:
        if normalized(str(row["source"])) in used_sources:
            continue
        value = repeated_prefix_recovery(
            row,
            tokenized[str(row["id"])],
            seed=args.seed,
            special_ids=special_ids,
            maximum_tokens=args.maximum_target_tokens,
        )
        if value is not None:
            repeated_candidates.append(value)
    repeated_candidates.sort(
        key=lambda row: (
            stable_rank(args.seed, "reference-repeat", row),
            str(row["id"]),
        )
    )
    recovery_rows = [
        *aligned_rows,
        *repeated_candidates[: RECOVERY_LIMIT - len(aligned_rows)],
    ]
    if len(recovery_rows) != RECOVERY_LIMIT:
        raise SystemExit(f"insufficient v15 recovery contrasts: {len(recovery_rows)}")

    omission_candidates = []
    for row in train_rows:
        value = omission_contrast(
            row,
            tokenized[str(row["id"])],
            seed=args.seed,
            special_ids=special_ids,
        )
        if value is not None:
            omission_candidates.append(value)
    omission_candidates.sort(
        key=lambda row: (
            0 if str(row["v15_stratum"]).endswith(("omission-risk", "long")) else 1,
            stable_rank(args.seed, "omission", row),
            str(row["id"]),
        )
    )
    omission_rows = omission_candidates[:OMISSION_LIMIT]
    if len(omission_rows) != OMISSION_LIMIT:
        raise SystemExit(f"insufficient v15 omission contrasts: {len(omission_rows)}")

    args.output.mkdir(parents=True, exist_ok=True)
    recovery_path = args.output / "recovery.jsonl"
    omission_path = args.output / "omission.jsonl"
    write_jsonl(recovery_path, recovery_rows)
    write_jsonl(omission_path, omission_rows)

    output_manifest = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "status": "contrastive-examples-frozen",
        "direction": "ja-en",
        "purpose": (
            "licensed-continuation clean/recovery ordering and "
            "licensed-reference omission contrast"
        ),
        "promotion_eligible": False,
        "training_only": True,
        "private_reasoning_traces_used": False,
        "free_form_synthetic_translations_used_as_targets": False,
        "generated_strings_are_positive_targets": False,
        "positive_target_source": "licensed human references only",
        "selection": {
            "seed": args.seed,
            "recovery_limit": RECOVERY_LIMIT,
            "aligned_recovery_limit": ALIGNED_RECOVERY_LIMIT,
            "omission_limit": OMISSION_LIMIT,
            "aligned_policy": (
                "first token-level non-delete difference between a free-running "
                "v12-step-50 rollout and its licensed reference"
            ),
            "repetition_policy": (
                "append one extra copy of a deterministic licensed-reference "
                "phrase, then prefer the true licensed continuation over the "
                "token that would begin a third contiguous copy"
            ),
            "omission_policy": (
                "prefer the first token of a deterministic licensed-reference "
                "span over the token after that deleted span"
            ),
        },
        "counts": {
            "recovery": len(recovery_rows),
            "recovery_roles": dict(
                sorted(
                    Counter(str(row["contrast_role"]) for row in recovery_rows).items()
                )
            ),
            "omission": len(omission_rows),
            "omission_strata": dict(
                sorted(
                    Counter(str(row["v15_stratum"]) for row in omission_rows).items()
                )
            ),
        },
        "dataset": {
            "manifest": record(dataset_manifest_path, root),
            "train": record(train_path, root),
        },
        "v14_rollout_evidence": {
            "manifest": record(rollout_manifest_path, root),
            "hard": record(hard_path, root),
            "rollout_strings_are_positive_targets": False,
        },
        "tokenizer_checkpoint": {
            "path": display_path(args.checkpoint, root),
            "model": record(checkpoint_model_path, root),
        },
        "outputs": {
            "recovery": {
                **record(recovery_path, root),
                "rows": len(recovery_rows),
            },
            "omission": {
                **record(omission_path, root),
                "rows": len(omission_rows),
            },
        },
        "distribution_provenance": {
            "all_positive_targets_are_licensed_human_references": True,
            "all_rows_have_source_license": all(
                bool(row.get("source_license"))
                for row in [*recovery_rows, *omission_rows]
            ),
            "all_rows_have_source_provenance": all(
                bool(row.get("source_provenance"))
                for row in [*recovery_rows, *omission_rows]
            ),
            "generated_strings_are_context_or_negative_evidence_only": True,
            "private_reasoning_traces_used": False,
        },
    }
    manifest_output_path = args.output / "manifest.json"
    manifest_output_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": sha256(manifest_output_path),
                "counts": output_manifest["counts"],
                "status": output_manifest["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
