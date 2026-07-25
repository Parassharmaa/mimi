#!/usr/bin/env python3
"""Train the preregistered V16 full-sequence PCGrad arm."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from diagnose_active_sequence_risk_v16 import (
    display_path,
    load_json,
    load_jsonl,
    sequence_scores,
    token_batch,
)
from torch.utils.data import DataLoader, Dataset
from train_marian_distillation import (
    Collator,
    TranslationRows,
    checkpoint_identity,
    checkpoint_lineage_manifests,
    evaluate,
    frozen_base_kl,
    hardware_name,
    l2_to_frozen_base,
    load_rows,
    move,
    sha256,
    synchronize,
)
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    get_linear_schedule_with_warmup,
)

EXPERIMENT = "active-sequence-risk-v16-ja-en"


class RiskRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def list_collator(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def next_batch(iterator: Any, loader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def trainable_parameters(model: MarianMTModel) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def gradient_values(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [None if value is None else value.detach() for value in values]


def gradient_dot(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> torch.Tensor:
    total = None
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            continue
        value = (left_value.float() * right_value.float()).sum()
        total = value if total is None else total + value
    if total is None:
        raise RuntimeError("gradient pair has no shared parameters")
    return total


def gradient_norm(values: list[torch.Tensor | None]) -> float:
    squared = gradient_dot(values, values)
    return math.sqrt(float(squared.detach().cpu()))


def symmetric_pcgrad(
    omission: list[torch.Tensor | None],
    repetition: list[torch.Tensor | None],
    *,
    epsilon: float,
) -> tuple[
    list[torch.Tensor | None],
    list[torch.Tensor | None],
    dict[str, float | bool],
]:
    dot = gradient_dot(omission, repetition)
    omission_squared = gradient_dot(omission, omission)
    repetition_squared = gradient_dot(repetition, repetition)
    denominator = torch.sqrt(omission_squared * repetition_squared).clamp_min(epsilon)
    cosine = dot / denominator
    conflict = bool((dot < 0).detach().cpu())
    if not conflict:
        return (
            omission,
            repetition,
            {
                "conflict": False,
                "cosine_before": float(cosine.detach().cpu()),
                "cosine_after": float(cosine.detach().cpu()),
            },
        )
    omission_scale = dot / repetition_squared.clamp_min(epsilon)
    repetition_scale = dot / omission_squared.clamp_min(epsilon)
    projected_omission = []
    projected_repetition = []
    for omission_value, repetition_value in zip(omission, repetition):
        if omission_value is None or repetition_value is None:
            projected_omission.append(omission_value)
            projected_repetition.append(repetition_value)
            continue
        projected_omission.append(
            omission_value - omission_scale.to(omission_value.dtype) * repetition_value
        )
        projected_repetition.append(
            repetition_value
            - repetition_scale.to(repetition_value.dtype) * omission_value
        )
    after = gradient_dot(projected_omission, projected_repetition)
    after_denominator = torch.sqrt(
        gradient_dot(projected_omission, projected_omission)
        * gradient_dot(projected_repetition, projected_repetition)
    ).clamp_min(epsilon)
    return (
        projected_omission,
        projected_repetition,
        {
            "conflict": True,
            "cosine_before": float(cosine.detach().cpu()),
            "cosine_after": float((after / after_denominator).detach().cpu()),
        },
    )


def assign_combined_gradients(
    parameters: list[torch.nn.Parameter],
    mle: list[torch.Tensor | None],
    omission: list[torch.Tensor | None],
    repetition: list[torch.Tensor | None],
    *,
    omission_weight: float,
    repetition_weight: float,
) -> None:
    for parameter, mle_value, omission_value, repetition_value in zip(
        parameters,
        mle,
        omission,
        repetition,
    ):
        combined = None
        for value, weight in (
            (mle_value, 1.0),
            (omission_value, omission_weight),
            (repetition_value, repetition_weight),
        ):
            if value is None:
                continue
            weighted = value * weight
            combined = weighted if combined is None else combined + weighted
        parameter.grad = None if combined is None else combined.to(parameter.dtype)


def sequence_ranking_loss(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    rows: list[dict[str, Any]],
    *,
    target_margin: float,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    sources = [str(row["source"]) for row in rows]
    chosen_batch = token_batch(
        tokenizer,
        sources,
        [str(row["chosen"]) for row in rows],
        maximum_source_tokens=maximum_source_tokens,
        maximum_target_tokens=maximum_target_tokens,
        device=device,
    )
    rejected_batch = token_batch(
        tokenizer,
        sources,
        [str(row["rejected"]) for row in rows],
        maximum_source_tokens=maximum_source_tokens,
        maximum_target_tokens=maximum_target_tokens,
        device=device,
    )
    chosen_scores, _ = sequence_scores(model, chosen_batch)
    rejected_scores, _ = sequence_scores(model, rejected_batch)
    margins = chosen_scores - rejected_scores
    losses = F.relu(target_margin - margins)
    return losses.mean(), {
        "loss": float(losses.mean().detach().cpu()),
        "mean_margin": float(margins.mean().detach().cpu()),
        "preference_accuracy": float(margins.gt(0).float().mean().detach().cpu()),
        "active_fraction": float(losses.gt(0).float().mean().detach().cpu()),
    }


@torch.inference_mode()
def evaluate_active_risks(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    target_margin: float,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    margins_by_role: dict[str, list[float]] = {"omission": [], "repetition": []}
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        sources = [str(row["source"]) for row in batch_rows]
        chosen = token_batch(
            tokenizer,
            sources,
            [str(row["chosen"]) for row in batch_rows],
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
            device=device,
        )
        rejected = token_batch(
            tokenizer,
            sources,
            [str(row["rejected"]) for row in batch_rows],
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
            device=device,
        )
        chosen_scores, _ = sequence_scores(model, chosen)
        rejected_scores, _ = sequence_scores(model, rejected)
        for row, value in zip(batch_rows, (chosen_scores - rejected_scores).cpu()):
            margins_by_role[str(row["risk_role"])].append(float(value))
    synchronize(device)
    model.train()

    def summarize(values: list[float]) -> dict[str, float | int]:
        return {
            "cases": len(values),
            "mean_margin": sum(values) / len(values),
            "preference_accuracy": sum(value > 0 for value in values) / len(values),
            "active_fraction": sum(value < target_margin for value in values)
            / len(values),
        }

    by_role = {role: summarize(values) for role, values in margins_by_role.items()}
    all_values = [*margins_by_role["omission"], *margins_by_role["repetition"]]
    return {
        "all": summarize(all_values),
        "by_role": by_role,
    }


def validate_contract(contract_path: Path, *, root: Path) -> dict[str, Any]:
    contract = load_json(contract_path)
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "preregistered-ready-for-one-arm-training"
        or contract.get("direction") != "ja-en"
        or contract.get("training", {}).get("gradient_rule")
        != "unprojected-mle-plus-symmetric-pcgrad-between-omission-and-repetition"
        or contract.get("training", {}).get("one_arm_only") is not True
        or contract.get("training", {}).get(
            "post_result_hyperparameter_changes_forbidden"
        )
        is not True
        or contract.get("protected_evaluation_authorized") is not False
        or contract.get("app_change_authorized") is not False
    ):
        raise SystemExit("V16 contract safety state differs")
    for section, field in (
        ("dataset", "manifest"),
        ("dataset", "train"),
        ("dataset", "fresh_valid"),
        ("dataset", "active"),
        ("dataset", "v12_regression"),
        ("dataset", "v14_regression"),
        ("dataset", "v15_regression"),
        ("initial_checkpoint", "model"),
        ("initial_checkpoint", "training_manifest"),
        ("preservation_checkpoint", "model"),
        ("preservation_checkpoint", "training_manifest"),
    ):
        item = contract[section][field]
        if sha256(root / item["path"]) != item["sha256"]:
            raise SystemExit(f"V16 bound input differs: {section}.{field}")
    return contract


def save_candidate(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    output: Path,
    manifest: dict[str, Any],
) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty checkpoint: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "mimi_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("checkpoint_directory", type=Path)
    parser.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default="mps",
    )
    args = parser.parse_args()
    if args.checkpoint_directory.exists() and any(args.checkpoint_directory.iterdir()):
        raise SystemExit(
            "refusing to overwrite non-empty checkpoint directory: "
            f"{args.checkpoint_directory}"
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    root = Path(__file__).resolve().parents[2]
    contract = validate_contract(args.contract, root=root)
    training = contract["training"]
    if training["checkpoint_steps"] != [
        training["evaluation_steps"],
        training["max_steps"],
    ]:
        raise SystemExit("V16 checkpoint schedule differs")

    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    train_rows = load_rows(root / contract["dataset"]["train"]["path"], "ja-en")
    fresh_rows = load_rows(root / contract["dataset"]["fresh_valid"]["path"], "ja-en")
    active_rows = load_jsonl(root / contract["dataset"]["active"]["path"])
    omission_rows = [row for row in active_rows if row["risk_role"] == "omission"]
    repetition_rows = [row for row in active_rows if row["risk_role"] == "repetition"]
    if (
        len(train_rows) != contract["dataset"]["counts"]["train"]
        or len(fresh_rows) != contract["dataset"]["counts"]["fresh_valid"]
        or len(active_rows) != contract["dataset"]["counts"]["active"]
        or len(omission_rows) != contract["dataset"]["counts"]["active_omission"]
        or len(repetition_rows) != contract["dataset"]["counts"]["active_repetition"]
    ):
        raise SystemExit("V16 bound row counts differ")

    initial_checkpoint = root / contract["initial_checkpoint"]["path"]
    preservation_checkpoint = root / contract["preservation_checkpoint"]["path"]
    expected_identity = (
        "ja-en",
        contract["initial_checkpoint"]["repository"],
        contract["initial_checkpoint"]["revision"],
    )
    for label, checkpoint in (
        ("initial", initial_checkpoint),
        ("preservation", preservation_checkpoint),
    ):
        if checkpoint_identity(checkpoint) != expected_identity:
            raise SystemExit(f"V16 {label} checkpoint identity differs")

    tokenizer = MarianTokenizer.from_pretrained(initial_checkpoint)
    model = MarianMTModel.from_pretrained(initial_checkpoint).to(device)
    preservation_model = MarianMTModel.from_pretrained(preservation_checkpoint).to(
        device
    )
    preservation_model.eval()
    preservation_model.requires_grad_(False)
    base_parameters = {
        name: parameter.detach()
        for name, parameter in preservation_model.named_parameters()
    }
    parameters = trainable_parameters(model)

    collator = Collator(
        tokenizer,
        training["max_source_tokens"],
        training["max_target_tokens"],
        set(),
    )
    train_loader = DataLoader(
        TranslationRows(train_rows),
        batch_size=training["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collator,
    )
    fresh_loader = DataLoader(
        TranslationRows(fresh_rows),
        batch_size=training["evaluation_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )
    omission_loader = DataLoader(
        RiskRows(omission_rows),
        batch_size=training["risk_batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        collate_fn=list_collator,
    )
    repetition_loader = DataLoader(
        RiskRows(repetition_rows),
        batch_size=training["risk_batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 2),
        collate_fn=list_collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training["warmup_steps"],
        num_training_steps=training["max_steps"],
    )
    train_iterator = iter(train_loader)
    omission_iterator = iter(omission_loader)
    repetition_iterator = iter(repetition_loader)

    initial_translation = evaluate(
        model,
        tokenizer,
        fresh_loader,
        fresh_rows,
        device,
        training["max_target_tokens"],
    )
    initial_risks = evaluate_active_risks(
        model,
        tokenizer,
        active_rows,
        batch_size=training["evaluation_batch_size"],
        target_margin=training["sequence_target_margin"],
        maximum_source_tokens=training["max_source_tokens"],
        maximum_target_tokens=training["max_target_tokens"],
        device=device,
    )
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "fresh_translation": initial_translation,
            "active_risks": initial_risks,
        }
    ]
    pcgrad_history: list[dict[str, Any]] = []
    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "operation": (
            "licensed-reference MLE plus full-sequence omission/repetition "
            "ranking with symmetric PCGrad between safety gradients"
        ),
        "paper_faithful_reproduction": False,
        "direction": "ja-en",
        "student_repository": expected_identity[1],
        "student_revision": expected_identity[2],
        "license": "CC-BY-SA-4.0",
        "contract": {
            "path": display_path(args.contract, root),
            "sha256": sha256(args.contract),
        },
        "initial_checkpoint": {
            "path": display_path(initial_checkpoint, root),
            "model_sha256": sha256(initial_checkpoint / "model.safetensors"),
            "training_manifest_sha256": sha256(
                initial_checkpoint / "mimi_training_manifest.json"
            ),
            "lineage_manifests": checkpoint_lineage_manifests(initial_checkpoint),
        },
        "preservation_checkpoint": {
            "path": display_path(preservation_checkpoint, root),
            "model_sha256": sha256(preservation_checkpoint / "model.safetensors"),
            "training_manifest_sha256": sha256(
                preservation_checkpoint / "mimi_training_manifest.json"
            ),
            "lineage_manifests": checkpoint_lineage_manifests(preservation_checkpoint),
        },
        "dataset_manifest": contract["dataset"]["manifest"],
        "dataset_provenance": {
            "direction": "ja-en",
            "experiment": EXPERIMENT,
            "positive_target_source": "authenticated licensed-human references only",
            "effective_licenses": contract["dataset"]["effective_licenses"],
            "promotion_eligible": False,
            "generated_strings_are_positive_targets": False,
            "private_reasoning_traces_used": False,
        },
        "dataset": {
            "train_sha256": contract["dataset"]["train"]["sha256"],
            "fresh_valid_sha256": contract["dataset"]["fresh_valid"]["sha256"],
            "active_sha256": contract["dataset"]["active"]["sha256"],
            "train_rows": len(train_rows),
            "fresh_valid_rows": len(fresh_rows),
            "active_rows": len(active_rows),
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "hyperparameters": training,
        "objective": {
            "mle": "token cross-entropy on licensed-human references",
            "omission": (
                "length-normalized complete licensed reference over a "
                "deterministic full-sequence deletion negative"
            ),
            "repetition": (
                "length-normalized complete licensed reference over a "
                "deterministic full-sequence repetition negative"
            ),
            "gradient_rule": training["gradient_rule"],
            "retention": "safe-parent token KL plus parameter L2",
        },
        "capacity_change": False,
        "moe_added": False,
        "scheduled_sampling_used": False,
        "private_reasoning_traces_used": False,
        "promotion_eligible": False,
        "app_change_authorized": False,
        "bundle_replacement_authorized": False,
    }

    for step in range(1, training["max_steps"] + 1):
        raw_mle, train_iterator = next_batch(train_iterator, train_loader)
        omission_batch, omission_iterator = next_batch(
            omission_iterator,
            omission_loader,
        )
        repetition_batch, repetition_iterator = next_batch(
            repetition_iterator,
            repetition_loader,
        )
        mle_batch = move(raw_mle, device)
        mle_batch.pop("preservation_mask")
        student = model(**mle_batch)
        with torch.inference_mode():
            parent = preservation_model(**mle_batch)
        preservation_mask = torch.ones(
            mle_batch["labels"].shape[0],
            dtype=torch.bool,
            device=device,
        )
        kl = frozen_base_kl(
            student.logits,
            parent.logits,
            mle_batch["labels"],
            preservation_mask,
        )
        l2 = l2_to_frozen_base(model, base_parameters)
        mle_loss = (
            student.loss
            + training["frozen_parent_kl_weight"] * kl
            + training["l2_to_parent_weight"] * l2
        )
        mle_gradients = gradient_values(mle_loss, parameters)

        omission_loss, omission_metrics = sequence_ranking_loss(
            model,
            tokenizer,
            omission_batch,
            target_margin=training["sequence_target_margin"],
            maximum_source_tokens=training["max_source_tokens"],
            maximum_target_tokens=training["max_target_tokens"],
            device=device,
        )
        omission_gradients = gradient_values(omission_loss, parameters)
        repetition_loss, repetition_metrics = sequence_ranking_loss(
            model,
            tokenizer,
            repetition_batch,
            target_margin=training["sequence_target_margin"],
            maximum_source_tokens=training["max_source_tokens"],
            maximum_target_tokens=training["max_target_tokens"],
            device=device,
        )
        repetition_gradients = gradient_values(repetition_loss, parameters)
        projected_omission, projected_repetition, projection = symmetric_pcgrad(
            omission_gradients,
            repetition_gradients,
            epsilon=training["pcgrad_epsilon"],
        )
        optimizer.zero_grad(set_to_none=True)
        assign_combined_gradients(
            parameters,
            mle_gradients,
            projected_omission,
            projected_repetition,
            omission_weight=training["omission_weight"],
            repetition_weight=training["repetition_weight"],
        )
        combined_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            training["maximum_gradient_norm"],
        )
        optimizer.step()
        scheduler.step()
        synchronize(device)
        pcgrad_history.append(
            {
                "step": step,
                "mle_loss": float(mle_loss.detach().cpu()),
                "mle_gradient_norm": gradient_norm(mle_gradients),
                "omission": omission_metrics,
                "omission_gradient_norm": gradient_norm(omission_gradients),
                "repetition": repetition_metrics,
                "repetition_gradient_norm": gradient_norm(repetition_gradients),
                "projection": projection,
                "combined_gradient_norm_before_clip": float(
                    combined_norm.detach().cpu()
                ),
            }
        )

        if step in training["checkpoint_steps"]:
            translation = evaluate(
                model,
                tokenizer,
                fresh_loader,
                fresh_rows,
                device,
                training["max_target_tokens"],
            )
            risks = evaluate_active_risks(
                model,
                tokenizer,
                active_rows,
                batch_size=training["evaluation_batch_size"],
                target_margin=training["sequence_target_margin"],
                maximum_source_tokens=training["max_source_tokens"],
                maximum_target_tokens=training["max_target_tokens"],
                device=device,
            )
            history.append(
                {
                    "step": step,
                    "fresh_translation": translation,
                    "active_risks": risks,
                }
            )
            checkpoint = args.checkpoint_directory / f"step-{step:07d}"
            manifest = {
                **common,
                "checkpoint_step": step,
                "training_history": history,
                "pcgrad_history": pcgrad_history,
            }
            save_candidate(model, tokenizer, checkpoint, manifest)
            print(
                json.dumps(
                    {
                        "checkpoint": display_path(checkpoint, root),
                        "step": step,
                        "fresh_translation": translation,
                        "active_risks": risks,
                        "projection": projection,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
