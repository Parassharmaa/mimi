#!/usr/bin/env python3
"""Train the preregistered Sonnet-5/Opus-5 JA-to-EN preference arm."""

from __future__ import annotations

import argparse
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

from train_marian_automated_preference_adapter import (
    load_json,
    load_jsonl,
    sha256,
    validate_rows,
)
from train_marian_automated_preference_full import capture_state, save_candidate
from train_marian_distillation import l2_to_frozen_base
from train_marian_dqo import (
    PreferenceCollator,
    PreferenceRows,
    dqo_loss,
    evaluate,
    hardware_name,
    pair_logps,
)


EXPERIMENT = "canonical-pairwise-preference-claude5-v7-ja-en"
DATASET_EXPERIMENT = "canonical-pairwise-v7-ja-en-claude5"
REQUIRED_JUDGES = {"claude-sonnet-5", "claude-opus-5"}


def load_claude5_preferences(
    directory: Path,
) -> tuple[list[dict], list[dict], dict]:
    manifest_path = directory / "manifest.json"
    train_path = directory / "train.jsonl"
    valid_path = directory / "valid.jsonl"
    manifest = load_json(manifest_path)
    review = manifest.get("review_policy")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != DATASET_EXPERIMENT
        or manifest.get("purpose")
        != "preregistered automated-consensus teacher-over-current preference experiment"
        or manifest.get("direction") != "ja-en"
        or set(manifest.get("required_judge_model_ids", [])) != REQUIRED_JUDGES
        or manifest.get("promotion_eligible") is not True
        or manifest.get("claim_eligible") is not False
        or manifest.get("private_reasoning_traces_used") is not False
        or not isinstance(review, dict)
        or review.get("human_reviewer_required") is not False
        or review.get("independent_judge_models") != 2
        or review.get("teacher_not_a_judge") is not True
        or review.get("absolute_teacher_quality_required") is not True
        or review.get("unanimous_pareto_preference_over_current_required") is not True
        or review.get("strict_total_improvement_per_judge") is not True
    ):
        raise SystemExit("Claude 5 preference dataset contract is invalid")
    for split, path in (("train", train_path), ("valid", valid_path)):
        record = manifest.get("outputs", {}).get(split, {})
        if record.get("sha256") != sha256(path) or not isinstance(
            record.get("rows"), int
        ):
            raise SystemExit(f"Claude 5 preference dataset does not authenticate {split}")
    effective = manifest.get("effective_licenses")
    if not isinstance(effective, dict) or any(
        not isinstance(effective.get(split), dict) or not effective[split]
        for split in ("train", "valid")
    ):
        raise SystemExit("Claude 5 preference dataset lacks effective licenses")
    train = load_jsonl(train_path)
    valid = load_jsonl(valid_path)
    validate_rows(train, "ja-en", "train")
    validate_rows(valid, "ja-en", "valid")
    if (
        manifest["outputs"]["train"]["rows"] != len(train)
        or manifest["outputs"]["valid"]["rows"] != len(valid)
        or {row["source_id"] for row in train}
        & {row["source_id"] for row in valid}
    ):
        raise SystemExit("Claude 5 preference counts or split isolation differ")
    for split, split_rows in (("train", train), ("valid", valid)):
        actual_licenses = dict(
            sorted(Counter(str(row["source_license"]) for row in split_rows).items())
        )
        if actual_licenses != effective[split]:
            raise SystemExit(f"Claude 5 effective {split} licenses differ")
        if any(
            set(row.get("judge_model_ids", [])) != REQUIRED_JUDGES
            for row in split_rows
        ):
            raise SystemExit(f"Claude 5 {split} row has an unexpected judge")
    return train, valid, manifest


def validate_contract(
    path: Path,
    *,
    dataset_directory: Path,
    parent: Path,
) -> dict:
    contract = load_json(path)
    if (
        contract.get("schema_version") != 1
        or contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "preregistered-ready-for-training"
        or contract.get("direction") != "ja-en"
        or contract.get("promotion_authorized") is not False
        or contract.get("app_change_authorized") is not False
        or contract.get("public_upload_authorized") is not False
        or contract.get("private_reasoning_traces_used") is not False
    ):
        raise SystemExit("Claude 5 preference contract is invalid")
    if contract.get("dataset", {}).get("directory") != str(dataset_directory):
        raise SystemExit("Claude 5 contract binds a different preference dataset")
    for name in ("manifest.json", "train.jsonl", "valid.jsonl"):
        if contract.get("dataset", {}).get("files", {}).get(name, {}).get(
            "sha256"
        ) != sha256(dataset_directory / name):
            raise SystemExit(f"Claude 5 preference dataset hash differs: {name}")
    if (
        contract.get("parent", {}).get("path") != str(parent)
        or contract.get("parent", {}).get("model_sha256")
        != sha256(parent / "model.safetensors")
        or contract.get("implementation", {}).get("trainer", {}).get("sha256")
        != sha256(Path(__file__).resolve())
    ):
        raise SystemExit("Claude 5 contract parent or trainer binding differs")
    required = set(
        contract.get("dataset", {}).get("required_judge_model_ids", [])
    )
    if required != {"claude-sonnet-5", "claude-opus-5"}:
        raise SystemExit("Claude 5 preference contract has the wrong judges")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("experiment_contract", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--evaluation-steps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--chosen-sft-weight", type=float, default=0.02)
    parser.add_argument("--l2-to-parent-weight", type=float, default=0.10)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if not (args.parent_checkpoint / "model.safetensors").is_file():
        raise SystemExit("parent checkpoint is incomplete")
    if min(
        args.batch_size,
        args.gradient_accumulation,
        args.max_steps,
        args.evaluation_steps,
    ) < 1:
        raise SystemExit("batch, accumulation, steps, and evaluation must be positive")
    if (
        args.learning_rate <= 0
        or args.beta <= 0
        or args.chosen_sft_weight < 0
        or args.l2_to_parent_weight < 0
    ):
        raise SystemExit("invalid optimization hyperparameters")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    contract = validate_contract(
        args.experiment_contract,
        dataset_directory=args.preference_directory,
        parent=args.parent_checkpoint,
    )
    actual_hyperparameters = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "evaluation_steps": args.evaluation_steps,
        "beta": args.beta,
        "chosen_sft_weight": args.chosen_sft_weight,
        "l2_to_parent_weight": args.l2_to_parent_weight,
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
    }
    if contract.get("training") != actual_hyperparameters:
        raise SystemExit("arguments differ from the preregistered Claude 5 recipe")

    train_rows, valid_rows, dataset_manifest = load_claude5_preferences(
        args.preference_directory
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
    base_parameters = {
        name: parameter.detach() for name, parameter in reference.named_parameters()
    }
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
        model.parameters(),
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

    base_metrics = evaluate(model, reference, valid_loader, device, args.beta)
    history = [{"step": 0, **base_metrics}]
    best = history[0]
    best_state = capture_state(model)
    update_step = 0
    micro_step = 0
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
            l2_loss = l2_to_frozen_base(model, base_parameters)
            combined = (
                preference_loss
                + args.chosen_sft_weight * chosen_sft_loss
                + args.l2_to_parent_weight * l2_loss
            )
            (combined / args.gradient_accumulation).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                        "l2_to_parent": float(l2_loss.detach().cpu()),
                        "relative_margin": float(
                            relative_margin.mean().detach().cpu()
                        ),
                    },
                }
                history.append(record)
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
                    best_state = capture_state(model)
                print(
                    json.dumps(
                        {"current": record, "best": best}, ensure_ascii=False
                    ),
                    flush=True,
                )
            if update_step >= args.max_steps:
                break

    model.load_state_dict(best_state, strict=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "operation": "Claude-5-consensus full-parameter preference update",
        "status": "trained-not-promotion-evaluated",
        "direction": "ja-en",
        "student_repository": contract["parent"]["repository"],
        "student_revision": contract["parent"]["revision"],
        "license": contract["parent"]["license"],
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
            "required_judge_model_ids": dataset_manifest[
                "required_judge_model_ids"
            ],
            "effective_licenses": dataset_manifest["effective_licenses"],
        },
        "objective": {
            "loss": (
                "DPO-style parent-relative sequence preference plus chosen SFT "
                "and L2-to-parent regularization"
            ),
            "sequence_log_probability": "mean target-token log probability",
            "reference_policy": "frozen exact current JA-to-EN parent",
            "selection": (
                "highest validation relative-pair accuracy, then relative margin, "
                "then lower loss"
            ),
        },
        "hyperparameters": actual_hyperparameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "best": best,
        "history": history,
    }
    save_candidate(model, tokenizer, args.output_directory, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output_directory),
                "best": best,
                "model_sha256": sha256(
                    args.output_directory / "model.safetensors"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
