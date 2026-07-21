#!/usr/bin/env python3
"""Generate source-only Marian sequence-distillation targets for a licensed dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DIRECTIONS = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing JSONL input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_authenticated_dataset(
    directory: Path,
    direction: str,
) -> tuple[list[dict], list[dict], dict, dict]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("input dataset has no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("direction") != direction:
        raise SystemExit("input dataset direction differs")
    expected_languages = DIRECTIONS[direction]
    output_rows: dict[str, list[dict]] = {}
    records: dict[str, dict] = {}
    for split in ("train", "valid"):
        path = directory / f"{split}.jsonl"
        expected_hash = manifest.get("outputs", {}).get(split, {}).get("sha256")
        if expected_hash != sha256(path):
            raise SystemExit(f"input manifest does not authenticate {split}.jsonl")
        rows = load_jsonl(path)
        identifiers = [str(row.get("id", "")) for row in rows]
        if (
            not rows
            or not all(identifiers)
            or len(identifiers) != len(set(identifiers))
            or any(
                (row.get("source_language"), row.get("target_language"))
                != expected_languages
                for row in rows
            )
        ):
            raise SystemExit(f"input {split} has invalid IDs or direction")
        output_rows[split] = rows
        records[split] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}
    return output_rows["train"], output_rows["valid"], manifest, {
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        **records,
    }


def unique_training_sources(rows: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for row in rows:
        original_id = str(row.get("original_id") or row["id"])
        source = str(row.get("source", ""))
        if not source.strip():
            raise SystemExit(f"training row has empty source: {row['id']}")
        previous = by_id.get(original_id)
        if previous is not None and previous["source"] != source:
            raise SystemExit(f"repeated training ID has inconsistent source: {original_id}")
        by_id[original_id] = {"id": original_id, "source": source}
    return [by_id[identifier] for identifier in sorted(by_id)]


def materialize_dataset(
    train_rows: list[dict],
    valid_rows: list[dict],
    targets: dict[str, str],
    teacher_revision: str,
    rejected: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    rejected = rejected or set()
    output: list[dict] = []
    for row in train_rows:
        original_id = str(row.get("original_id") or row["id"])
        if original_id in rejected:
            continue
        target = targets.get(original_id, "").strip()
        if not target:
            raise SystemExit(f"missing teacher target: {original_id}")
        output.append(
            {
                **row,
                "target": target,
                "reference_target_sha256": text_sha256(str(row["target"])),
                "target_source": "marian-source-only-sequence-distillation",
                "teacher_model_revision": teacher_revision,
            }
        )
    return output, valid_rows


def load_partial_targets(
    path: Path,
    sources: dict[str, str],
    teacher_weights_sha256: str,
) -> tuple[dict[str, str], set[str]]:
    if not path.is_file():
        return {}, set()
    targets: dict[str, str] = {}
    rejected: set[str] = set()
    for row in load_jsonl(path):
        identifier = str(row.get("id", ""))
        if identifier not in sources or identifier in targets or identifier in rejected:
            raise SystemExit("partial teacher targets contain unknown or duplicate IDs")
        if row.get("source_sha256") != text_sha256(sources[identifier]):
            raise SystemExit(f"partial target source identity differs: {identifier}")
        if row.get("teacher_weights_sha256") != teacher_weights_sha256:
            raise SystemExit("partial targets use a different teacher checkpoint")
        rejection = str(row.get("rejected", "")).strip()
        target = str(row.get("target", "")).strip()
        if rejection:
            if rejection != "empty-target" or target:
                raise SystemExit(f"partial target has invalid rejection: {identifier}")
            rejected.add(identifier)
        elif target:
            targets[identifier] = target
        else:
            raise SystemExit(f"partial target is empty without rejection: {identifier}")
    return targets, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset", type=Path)
    parser.add_argument("teacher_checkpoint", type=Path)
    parser.add_argument("identity_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.max_source_tokens, args.max_target_tokens) < 1:
        raise SystemExit("batch and token limits must be positive")
    if args.output_directory.exists() and any(args.output_directory.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    args.output_directory.mkdir(parents=True, exist_ok=True)

    train_rows, valid_rows, input_manifest, input_record = load_authenticated_dataset(
        args.input_dataset, args.direction
    )
    sources = unique_training_sources(train_rows)
    source_by_id = {row["id"]: row["source"] for row in sources}
    weights_path = args.teacher_checkpoint / "model.safetensors"
    if not weights_path.is_file():
        raise SystemExit("teacher checkpoint has no model.safetensors")
    identity = json.loads(args.identity_manifest.read_text(encoding="utf-8"))
    teacher_weights_sha = sha256(weights_path)
    if identity.get("source_weights_sha256") != teacher_weights_sha:
        raise SystemExit("identity manifest does not authenticate teacher weights")
    repository = str(identity.get("source_repository", ""))
    revision = str(identity.get("source_revision", ""))
    if not repository or not revision:
        raise SystemExit("identity manifest lacks teacher repository/revision")
    teacher_revision = f"{repository}@{revision}"

    partial_path = args.output_directory / "teacher-targets.partial.jsonl"
    final_targets_path = args.output_directory / "teacher-targets.jsonl"
    if final_targets_path.exists():
        raise SystemExit("output already contains completed teacher targets")
    targets, rejected = load_partial_targets(
        partial_path, source_by_id, teacher_weights_sha
    )
    missing = [
        row for row in sources if row["id"] not in targets and row["id"] not in rejected
    ]

    if missing:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        tokenizer = MarianTokenizer.from_pretrained(args.teacher_checkpoint)
        model = MarianMTModel.from_pretrained(args.teacher_checkpoint)
        if args.device == "mps":
            if not torch.backends.mps.is_available():
                raise SystemExit("MPS is not available")
            model = model.to(device="mps", dtype=torch.float16)
        else:
            model = model.to(args.device)
        model.eval()
        with partial_path.open("a", encoding="utf-8") as partial:
            for start in range(0, len(missing), args.batch_size):
                batch = missing[start : start + args.batch_size]
                encoded = tokenizer(
                    [row["source"] for row in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_source_tokens,
                )
                encoded = {key: value.to(args.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        max_length=args.max_target_tokens,
                        num_beams=1,
                        do_sample=False,
                    )
                translations = tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                for row, translation in zip(batch, translations, strict=True):
                    target = translation.strip()
                    record = {
                        "id": row["id"],
                        "source_sha256": text_sha256(row["source"]),
                        "teacher_weights_sha256": teacher_weights_sha,
                    }
                    if target:
                        record["target"] = target
                        targets[row["id"]] = target
                    else:
                        record["rejected"] = "empty-target"
                        rejected.add(row["id"])
                    partial.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                partial.flush()
                completed = min(start + len(batch), len(missing))
                if completed % (args.batch_size * 25) == 0 or completed == len(missing):
                    print(
                        json.dumps(
                            {
                                "generated_this_run": completed,
                                "remaining": len(missing) - completed,
                            }
                        ),
                        flush=True,
                    )
        del model
    if set(targets) | rejected != set(source_by_id) or set(targets) & rejected:
        raise SystemExit("teacher target generation is incomplete")

    partial_path.replace(final_targets_path)
    distilled_train, distilled_valid = materialize_dataset(
        train_rows, valid_rows, targets, teacher_revision, rejected
    )
    train_path = args.output_directory / "train.jsonl"
    valid_path = args.output_directory / "valid.jsonl"
    write_jsonl(train_path, distilled_train)
    write_jsonl(valid_path, distilled_valid)
    manifest = {
        "schema_version": 1,
        "experiment": "source-only canonical sequence distillation for shallow Marian",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "direction": args.direction,
        "target_source": "marian-source-only-sequence-distillation",
        "promotion_eligible": False,
        "private_reasoning_traces_used": False,
        "references_exposed_to_teacher": False,
        "input_dataset": input_record,
        "input_dataset_manifest_sha256": sha256(args.input_dataset / "manifest.json"),
        "input_effective_licenses": input_manifest.get("effective_licenses"),
        "teacher": {
            "checkpoint": str(args.teacher_checkpoint),
            "weights_sha256": teacher_weights_sha,
            "identity_manifest": {
                "path": str(args.identity_manifest),
                "sha256": sha256(args.identity_manifest),
            },
            "repository": repository,
            "revision": revision,
            "license": "CC-BY-SA-4.0",
        },
        "generation": {
            "algorithm": "greedy-batched-full-precision-source-only",
            "batch_size": args.batch_size,
            "maximum_source_tokens": args.max_source_tokens,
            "maximum_target_tokens": args.max_target_tokens,
            "unique_sources": len(sources),
            "device": args.device,
            "script_sha256": sha256(Path(__file__).resolve()),
            "python_version": platform.python_version(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "sentencepiece", "sacremoses")
            },
        },
        "counts": {"train": len(distilled_train), "valid": len(distilled_valid)},
        "rejected": {
            "empty_teacher_targets_unique_sources": len(rejected),
            "empty_teacher_target_training_rows": len(train_rows) - len(distilled_train),
        },
        "effective_licenses": {
            "train": dict(
                sorted(Counter(row["source_license"] for row in distilled_train).items())
            ),
            "valid": dict(
                sorted(Counter(row["source_license"] for row in distilled_valid).items())
            ),
        },
        "outputs": {
            "teacher_targets": {
                "path": str(final_targets_path),
                "sha256": sha256(final_targets_path),
            },
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
    }
    (args.output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
