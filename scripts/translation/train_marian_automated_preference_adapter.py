#!/usr/bin/env python3
"""Train a mergeable Marian adapter from automated teacher-over-current pairs.

This is an experimental, preregistered preference objective.  It is separate
from Mimi's human-only DQO path and cannot authorize promotion by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup

from build_canonical_pairwise_preference_dataset import (
    ORIGIN,
    PURPOSE,
    REVIEW_STATUS,
)
from marian_low_rank_adapter import (
    adapter_modules,
    apply_low_rank_adapters,
    capture_trainable_state,
    load_trainable_state,
    merge_low_rank_adapters,
    save_adapter,
)
from train_marian_dqo import (
    PreferenceCollator,
    PreferenceRows,
    dqo_loss,
    evaluate,
    hardware_name,
    pair_logps,
)


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"preference split is empty or malformed: {path}")
    return rows


def validate_rows(rows: list[dict], direction: str, split: str) -> None:
    expected = LANGUAGES[direction]
    identifiers: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        identifier = str(row.get("id", "")).strip()
        source_id = str(row.get("source_id", "")).strip()
        if (
            not identifier
            or identifier in identifiers
            or not source_id
            or source_id in sources
        ):
            raise SystemExit(f"{split} contains an empty or duplicate ID/source")
        identifiers.add(identifier)
        sources.add(source_id)
        if (row.get("source_language"), row.get("target_language")) != expected:
            raise SystemExit(f"{split} contains the wrong language direction: {identifier}")
        if (
            row.get("origin") != ORIGIN
            or row.get("review_status") != REVIEW_STATUS
            or row.get("promotion_eligible") is not True
            or row.get("private_reasoning_traces_used") is not False
        ):
            raise SystemExit(f"{split} row lacks preference evidence: {identifier}")
        judges = {
            str(model).strip()
            for model in row.get("judge_model_ids", [])
            if str(model).strip()
        }
        evidence = row.get("judge_evidence")
        if (
            len(judges) != 2
            or not isinstance(evidence, list)
            or len(evidence) != 2
            or {item.get("judge_model") for item in evidence} != judges
            or any(item.get("pareto_preferred") is not True for item in evidence)
        ):
            raise SystemExit(f"{split} row lacks unanimous independent judges: {identifier}")
        texts = [
            str(row.get(field, "")).strip()
            for field in ("source", "chosen", "rejected")
        ]
        if not all(texts) or texts[1] == texts[2]:
            raise SystemExit(f"{split} contains empty or identical preferences: {identifier}")
        if not row.get("source_license") or not row.get("source_provenance"):
            raise SystemExit(f"{split} lacks license/provenance: {identifier}")


def load_preferences(
    directory: Path,
    direction: str,
) -> tuple[list[dict], list[dict], dict]:
    manifest_path = directory / "manifest.json"
    train_path = directory / "train.jsonl"
    valid_path = directory / "valid.jsonl"
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != "canonical teacher-over-current preference v4"
        or manifest.get("purpose") != PURPOSE
        or manifest.get("direction") != direction
        or manifest.get("promotion_eligible") is not True
        or manifest.get("claim_eligible") is not False
        or manifest.get("private_reasoning_traces_used") is not False
    ):
        raise SystemExit("preference dataset manifest has an invalid experiment contract")
    review = manifest.get("review_policy")
    if (
        not isinstance(review, dict)
        or review.get("human_reviewer_required") is not False
        or review.get("independent_judge_models") != 2
        or review.get("teacher_not_a_judge") is not True
        or review.get("absolute_teacher_quality_required") is not True
        or review.get("unanimous_pareto_preference_over_current_required") is not True
        or review.get("strict_total_improvement_per_judge") is not True
    ):
        raise SystemExit("preference dataset has a weakened automated review policy")
    for split, path in (("train", train_path), ("valid", valid_path)):
        record = manifest.get("outputs", {}).get(split, {})
        if (
            record.get("sha256") != sha256(path)
            or not isinstance(record.get("rows"), int)
        ):
            raise SystemExit(f"preference dataset does not authenticate {split}")
    effective = manifest.get("effective_licenses")
    if not isinstance(effective, dict) or any(
        not isinstance(effective.get(split), dict) or not effective[split]
        for split in ("train", "valid")
    ):
        raise SystemExit("preference dataset lacks effective licenses")
    train = load_jsonl(train_path)
    valid = load_jsonl(valid_path)
    validate_rows(train, direction, "train")
    validate_rows(valid, direction, "valid")
    if manifest["outputs"]["train"]["rows"] != len(train) or manifest["outputs"][
        "valid"
    ]["rows"] != len(valid):
        raise SystemExit("preference row counts differ from manifest")
    if {row["source_id"] for row in train} & {
        row["source_id"] for row in valid
    }:
        raise SystemExit("preference sources leak across train and validation")
    for split, rows in (("train", train), ("valid", valid)):
        actual = Counter(str(row["source_license"]) for row in rows)
        if dict(sorted(actual.items())) != effective[split]:
            raise SystemExit(f"effective {split} licenses differ from rows")
    return train, valid, manifest


def validate_contract(
    path: Path,
    *,
    dataset_directory: Path,
    parent: Path,
    direction: str,
) -> dict:
    contract = load_json(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("experiment")
        != "canonical-pairwise-preference-adapter-v4-ja-en"
        or contract.get("status") != "preregistered-ready-for-training"
        or contract.get("direction") != direction
        or contract.get("promotion_authorized") is not False
        or contract.get("app_change_authorized") is not False
        or contract.get("public_upload_authorized") is not False
        or contract.get("private_reasoning_traces_used") is not False
    ):
        raise SystemExit("v4 experiment contract is invalid or already authorizes promotion")
    if contract.get("dataset", {}).get("directory") != str(dataset_directory):
        raise SystemExit("v4 contract binds a different dataset directory")
    for name in ("manifest.json", "train.jsonl", "valid.jsonl"):
        expected = contract.get("dataset", {}).get("files", {}).get(name, {}).get(
            "sha256"
        )
        if expected != sha256(dataset_directory / name):
            raise SystemExit(f"v4 contract dataset hash differs: {name}")
    if contract.get("parent", {}).get("path") != str(parent):
        raise SystemExit("v4 contract binds a different parent checkpoint")
    if contract.get("parent", {}).get("model_sha256") != sha256(
        parent / "model.safetensors"
    ):
        raise SystemExit("v4 contract parent model hash differs")
    implementation = contract.get("implementation", {}).get("trainer", {})
    this_path = Path(__file__).resolve()
    if implementation.get("sha256") != sha256(this_path):
        raise SystemExit("v4 contract trainer implementation hash differs")
    return contract


def save_merged_candidate(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    output: Path,
    manifest: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "mimi_automated_preference_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("experiment_contract", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--preset",
        choices=("consultation-v1", "all-projections"),
        default="consultation-v1",
    )
    parser.add_argument("--encoder-top-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--evaluation-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--chosen-sft-weight", type=float, default=0.02)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--checkpoint-directory", type=Path)
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if args.checkpoint_directory is not None:
        if args.checkpoint_directory.exists() and any(args.checkpoint_directory.iterdir()):
            raise SystemExit(
                f"refusing to overwrite non-empty checkpoints: {args.checkpoint_directory}"
            )
        args.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if not (args.parent_checkpoint / "model.safetensors").is_file():
        raise SystemExit("parent checkpoint is incomplete")
    if min(
        args.rank,
        args.batch_size,
        args.gradient_accumulation,
        args.max_steps,
        args.evaluation_steps,
    ) < 1:
        raise SystemExit("rank, batch, accumulation, steps, and evaluation must be positive")
    if (
        args.alpha <= 0
        or not 0 <= args.dropout < 1
        or args.learning_rate <= 0
        or args.beta <= 0
        or args.chosen_sft_weight < 0
    ):
        raise SystemExit("invalid adapter or optimization hyperparameters")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    contract = validate_contract(
        args.experiment_contract,
        dataset_directory=args.preference_directory,
        parent=args.parent_checkpoint,
        direction=args.direction,
    )
    registered = contract.get("training", {})
    actual_hyperparameters = {
        "seed": args.seed,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "preset": args.preset,
        "encoder_top_layers": args.encoder_top_layers,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "evaluation_steps": args.evaluation_steps,
        "beta": args.beta,
        "chosen_sft_weight": args.chosen_sft_weight,
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
    }
    if registered != actual_hyperparameters:
        raise SystemExit("training arguments differ from the preregistered v4 recipe")

    train_rows, valid_rows, dataset_manifest = load_preferences(
        args.preference_directory, args.direction
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = MarianTokenizer.from_pretrained(args.parent_checkpoint)
    model = MarianMTModel.from_pretrained(args.parent_checkpoint).to(device)
    reference = MarianMTModel.from_pretrained(args.parent_checkpoint).to(device)
    reference.eval()
    reference.requires_grad_(False)
    selected_modules = apply_low_rank_adapters(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        preset=args.preset,
        encoder_layers=model.config.encoder_layers,
        encoder_top_layers=args.encoder_top_layers,
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    collator = PreferenceCollator(
        tokenizer, args.max_source_tokens, args.max_target_tokens
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PreferenceRows(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        PreferenceRows(valid_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, args.max_steps),
        num_training_steps=args.max_steps,
    )
    optimizer.zero_grad(set_to_none=True)

    base_metrics = evaluate(
        model, reference, valid_loader, device, args.beta
    )
    history = [{"step": 0, **base_metrics}]
    best = history[0]
    best_state = capture_trainable_state(model)
    checkpoints: list[dict] = []
    update_step = 0
    micro_step = 0
    epoch = 0
    model.train()
    while update_step < args.max_steps:
        for pair in train_loader:
            chosen = {
                key: value.to(device) for key, value in pair["chosen"].items()
            }
            rejected = {
                key: value.to(device) for key, value in pair["rejected"].items()
            }
            policy_chosen, policy_rejected = pair_logps(model, chosen, rejected)
            with torch.inference_mode():
                reference_chosen, reference_rejected = pair_logps(
                    reference, chosen, rejected
                )
            preference_loss, relative_margin = dqo_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                args.beta,
            )
            chosen_sft_loss = -policy_chosen.mean()
            combined = preference_loss + args.chosen_sft_weight * chosen_sft_loss
            (combined / args.gradient_accumulation).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            if (
                update_step % args.evaluation_steps == 0
                or update_step == args.max_steps
            ):
                metrics = evaluate(
                    model, reference, valid_loader, device, args.beta
                )
                record = {
                    "step": update_step,
                    **metrics,
                    "last_train_objective": {
                        "preference_loss": float(preference_loss.detach().cpu()),
                        "chosen_sft_loss": float(chosen_sft_loss.detach().cpu()),
                        "relative_margin": float(
                            relative_margin.mean().detach().cpu()
                        ),
                    },
                }
                history.append(record)
                if args.checkpoint_directory is not None:
                    adapter_path = (
                        args.checkpoint_directory
                        / f"step-{update_step:07d}.safetensors"
                    )
                    save_adapter(model, adapter_path)
                    checkpoints.append(
                        {
                            "step": update_step,
                            "path": str(adapter_path),
                            "sha256": sha256(adapter_path),
                            "bytes": adapter_path.stat().st_size,
                            "metrics": record,
                        }
                    )
                if (
                    metrics["relative_pair_accuracy"],
                    metrics["relative_margin"],
                    -metrics["loss"],
                ) > (
                    best["relative_pair_accuracy"],
                    best["relative_margin"],
                    -best["loss"],
                ):
                    best = record
                    best_state = capture_trainable_state(model)
                print(
                    json.dumps(
                        {"current": record, "best": best}, ensure_ascii=False
                    )
                )
            if update_step >= args.max_steps:
                break
        epoch += 1

    load_trainable_state(model, best_state)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    adapter_path = args.output_directory / "adapters.safetensors"
    save_adapter(model, adapter_path)
    merged_modules = merge_low_rank_adapters(model)
    if merged_modules != selected_modules or adapter_modules(model):
        raise RuntimeError("adapter merge did not preserve the selected module set")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "operation": "automated-consensus teacher-over-current preference adapter",
        "status": "trained-not-promotion-evaluated",
        "direction": args.direction,
        "student_repository": contract["parent"]["repository"],
        "student_revision": contract["parent"]["revision"],
        "license": "CC-BY-SA-4.0",
        "private_reasoning_traces_used": False,
        "human_reviewers_used": False,
        "promotion_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
        "parent": contract["parent"],
        "experiment_contract": {
            "path": str(args.experiment_contract),
            "sha256": sha256(args.experiment_contract),
        },
        "preferences": {
            "directory": str(args.preference_directory),
            "manifest_sha256": sha256(args.preference_directory / "manifest.json"),
            "train_sha256": sha256(args.preference_directory / "train.jsonl"),
            "valid_sha256": sha256(args.preference_directory / "valid.jsonl"),
            "train_pairs": len(train_rows),
            "valid_pairs": len(valid_rows),
            "target_source": dataset_manifest["target_source"],
            "effective_licenses": dataset_manifest["effective_licenses"],
            "review_policy": dataset_manifest["review_policy"],
        },
        "adapter": {
            "format": "mimi-marian-low-rank-v1",
            "rank": args.rank,
            "alpha": args.alpha,
            "scale": args.alpha / args.rank,
            "dropout": args.dropout,
            "preset": args.preset,
            "encoder_top_layers": args.encoder_top_layers,
            "selected_modules": selected_modules,
            "trainable_parameters": trainable_count,
            "path": str(adapter_path),
            "sha256": sha256(adapter_path),
            "bytes": adapter_path.stat().st_size,
            "merged_for_deployment": True,
        },
        "objective": {
            "loss": (
                "-log sigmoid(beta * ((logp_policy_chosen-logp_policy_rejected) - "
                "(logp_parent_chosen-logp_parent_rejected))) + "
                "chosen_sft_weight * chosen_nll"
            ),
            "sequence_log_probability": "mean target-token log probability",
            "reference_policy": "frozen exact current JA-to-EN parent",
            "preference_source": REVIEW_STATUS,
            "selection": (
                "highest validation relative-pair accuracy, then relative margin, "
                "then lower loss"
            ),
        },
        "hyperparameters": actual_hyperparameters,
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "best": best,
        "history": history,
        "checkpoints": checkpoints,
    }
    save_merged_candidate(
        model, tokenizer, args.output_directory, manifest
    )
    print(
        json.dumps(
            {
                "output": str(args.output_directory),
                "best": best,
                "adapter_bytes": adapter_path.stat().st_size,
                "trainable_parameters": trainable_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
