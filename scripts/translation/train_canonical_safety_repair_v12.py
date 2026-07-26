#!/usr/bin/env python3
"""Run the preregistered v12 constraint-aware JA-to-EN repair arm."""

from __future__ import annotations

import argparse
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
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
    sha256,
    synchronize,
)
from train_marian_negative_space import (
    NegativeRows,
    NegativeSpaceCollator,
    divergence_metrics,
    evaluate_negatives,
    load_negative_space,
    move,
    split_metadata,
    token_local_unlikelihood,
)
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    get_linear_schedule_with_warmup,
)

EXPERIMENT = "canonical-safety-repair-v12-ja-en"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def ranking_margin_loss(
    logits: torch.Tensor,
    positions: torch.Tensor,
    chosen_token_ids: torch.Tensor,
    rejected_token_ids: torch.Tensor,
    severity: torch.Tensor,
    *,
    target_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    margins, preferred = divergence_metrics(
        logits,
        positions,
        chosen_token_ids,
        rejected_token_ids,
    )
    losses = F.relu(target_margin - margins)
    weighted = (losses * severity).sum() / severity.sum().clamp_min(1e-6)
    return weighted, preferred


def validate_contract(
    contract_path: Path,
    *,
    root: Path,
) -> dict[str, Any]:
    contract = load_json(contract_path)
    if (
        contract.get("experiment") != EXPERIMENT
        or contract.get("status") != "preregistered-ready-for-one-arm-training"
        or contract.get("direction") != "ja-en"
        or contract.get("training", {}).get("one_arm_only") is not True
        or contract.get("training", {}).get(
            "post_result_hyperparameter_changes_forbidden"
        )
        is not True
        or contract.get("protected_evaluation_authorized") is not False
        or contract.get("app_change_authorized") is not False
    ):
        raise SystemExit("v12 contract safety state differs")
    for section, field in (
        ("dataset", "manifest"),
        ("dataset", "train"),
        ("dataset", "valid"),
        ("negative_dataset", "manifest"),
        ("negative_dataset", "train"),
        ("negative_dataset", "valid"),
        ("initial_checkpoint", "model"),
        ("initial_checkpoint", "training_manifest"),
        ("preservation_checkpoint", "model"),
        ("preservation_checkpoint", "training_manifest"),
    ):
        item = contract[section][field]
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"v12 bound input differs: {section}.{field}")
    return contract


def checkpoint_manifest(
    *,
    common: dict[str, Any],
    history: list[dict[str, Any]],
    step: int,
    checkpoint_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        **common,
        "history": history,
        "checkpoint_step": step,
        "checkpoint_metrics": checkpoint_metrics,
        "promotion_eligible": False,
        "quantization_authorized": False,
        "protected_evaluation_authorized": False,
        "app_change_authorized": False,
        "public_upload_authorized": False,
    }


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
            f"refusing to overwrite non-empty output: {args.checkpoint_directory}"
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    root = Path(__file__).resolve().parents[2]
    contract = validate_contract(args.contract, root=root)
    training = contract["training"]
    expected_checkpoints = training["checkpoint_steps"]
    if expected_checkpoints != [
        training["evaluation_steps"],
        training["max_steps"],
    ]:
        raise SystemExit("v12 checkpoint schedule differs")

    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    negative_directory = root / contract["negative_dataset"]["directory"]
    (
        train_rows,
        negative_valid_rows,
        negative_manifest,
        parent_directory,
    ) = load_negative_space(negative_directory, "ja-en")
    expected_parent = root / contract["dataset"]["directory"]
    if parent_directory.resolve() != expected_parent.resolve():
        raise SystemExit("v12 negative parent directory differs")
    parent_valid_rows = load_rows(parent_directory / "valid.jsonl", "ja-en")
    if len(parent_valid_rows) != contract["dataset"]["counts"]["valid"]:
        raise SystemExit("v12 positive validation row count differs")

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
            raise SystemExit(f"v12 {label} checkpoint identity differs")

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
    if training["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    negative_collator = NegativeSpaceCollator(
        tokenizer,
        training["max_source_tokens"],
        training["max_target_tokens"],
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        NegativeRows(train_rows),
        batch_size=training["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=negative_collator,
    )
    negative_valid_loader = DataLoader(
        NegativeRows(negative_valid_rows),
        batch_size=training["evaluation_batch_size"],
        shuffle=False,
        collate_fn=negative_collator,
    )
    positive_collator = Collator(
        tokenizer,
        training["max_source_tokens"],
        training["max_target_tokens"],
        set(),
    )
    positive_valid_loader = DataLoader(
        TranslationRows(parent_valid_rows),
        batch_size=training["evaluation_batch_size"],
        shuffle=False,
        collate_fn=positive_collator,
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
    optimizer.zero_grad(set_to_none=True)

    initial_translation = evaluate(
        model,
        tokenizer,
        positive_valid_loader,
        parent_valid_rows,
        device,
        training["max_target_tokens"],
    )
    initial_negatives = evaluate_negatives(
        model,
        negative_valid_loader,
        device,
    )
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "translation": initial_translation,
            "negative_space": initial_negatives,
        }
    ]
    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "operation": (
            "constraint-aware token-local safety repair with licensed-human "
            "references, ranking, unlikelihood, and frozen-parent retention"
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
        "dataset_manifest": {
            "path": contract["dataset"]["manifest"]["path"],
            "sha256": contract["dataset"]["manifest"]["sha256"],
            "schema_version": 1,
            "direction": "ja-en",
            "experiment": EXPERIMENT,
            "target_source": "licensed human references only",
            "effective_licenses": contract["dataset"]["effective_licenses"],
            "promotion_eligible": False,
            "authenticated_outputs": ["train", "valid"],
            "outputs_authenticated": True,
            "negative_dataset_manifest": {
                "path": contract["negative_dataset"]["manifest"]["path"],
                "sha256": contract["negative_dataset"]["manifest"]["sha256"],
            },
        },
        "dataset": {
            "directory": contract["dataset"]["directory"],
            "train_sha256": contract["dataset"]["train"]["sha256"],
            "valid_sha256": contract["dataset"]["valid"]["sha256"],
            "train_rows": contract["dataset"]["counts"]["train"],
            "valid_rows": contract["dataset"]["counts"]["valid"],
        },
        "negative_dataset": {
            "directory": contract["negative_dataset"]["directory"],
            "manifest_sha256": contract["negative_dataset"]["manifest"]["sha256"],
            "train_sha256": negative_manifest["outputs"]["train"]["sha256"],
            "valid_sha256": negative_manifest["outputs"]["valid"]["sha256"],
            "train_pairs": len(train_rows),
            "valid_pairs": len(negative_valid_rows),
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "hyperparameters": training,
        "objective": {
            "positive": (
                "ordinary token cross-entropy on authenticated licensed-human "
                "references"
            ),
            "negative": (
                "severity-weighted token-local unlikelihood on the first "
                "divergent rejected token"
            ),
            "ranking": (
                "severity-weighted hinge ranking of the chosen token above the "
                "rejected token at the same first divergence"
            ),
            "retention": (
                "teacher-forced token KL and parameter L2 to the frozen safe parent"
            ),
            "negative_strings_are_positive_targets": False,
            "free_form_synthetic_translations_used": False,
            "private_reasoning_traces_used": False,
            "human_review_required": False,
        },
    }

    update_step = 0
    micro_step = 0
    model.train()
    while update_step < training["max_steps"]:
        for batch in train_loader:
            batch = move(batch, device)
            model_inputs, metadata = split_metadata(batch)
            outputs = model(**model_inputs)
            positive_loss = outputs.loss
            negative_loss, rejected_probability = token_local_unlikelihood(
                outputs.logits,
                metadata["negative_positions"],
                metadata["rejected_token_ids"],
                metadata["severity"],
            )
            ranking_loss, preferred = ranking_margin_loss(
                outputs.logits,
                metadata["negative_positions"],
                metadata["chosen_token_ids"],
                metadata["rejected_token_ids"],
                metadata["severity"],
                target_margin=training["ranking_target_margin"],
            )
            with torch.no_grad():
                parent_logits = preservation_model(**model_inputs).logits
            preservation_mask = torch.ones(
                model_inputs["labels"].shape[0],
                dtype=torch.bool,
                device=device,
            )
            kl_loss = frozen_base_kl(
                outputs.logits,
                parent_logits,
                model_inputs["labels"],
                preservation_mask,
            )
            l2_loss = l2_to_frozen_base(model, base_parameters)
            combined = (
                positive_loss
                + training["negative_weight"] * negative_loss
                + training["ranking_weight"] * ranking_loss
                + training["frozen_parent_kl_weight"] * kl_loss
                + training["l2_to_parent_weight"] * l2_loss
            )
            (combined / training["gradient_accumulation"]).backward()
            micro_step += 1
            if micro_step % training["gradient_accumulation"]:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1

            if (
                update_step % training["evaluation_steps"] == 0
                or update_step == training["max_steps"]
            ):
                translation = evaluate(
                    model,
                    tokenizer,
                    positive_valid_loader,
                    parent_valid_rows,
                    device,
                    training["max_target_tokens"],
                )
                negatives = evaluate_negatives(
                    model,
                    negative_valid_loader,
                    device,
                )
                checkpoint_metrics = {
                    "translation": translation,
                    "negative_space": negatives,
                    "last_train_objective": {
                        "positive_cross_entropy": float(positive_loss.detach().cpu()),
                        "token_local_unlikelihood": float(negative_loss.detach().cpu()),
                        "ranking_margin_loss": float(ranking_loss.detach().cpu()),
                        "chosen_token_preference_accuracy": float(
                            preferred.float().mean().detach().cpu()
                        ),
                        "mean_rejected_token_probability": float(
                            rejected_probability.mean().detach().cpu()
                        ),
                        "frozen_parent_kl": float(kl_loss.detach().cpu()),
                        "l2_to_parent": float(l2_loss.detach().cpu()),
                        "combined": float(combined.detach().cpu()),
                    },
                }
                history.append(
                    {
                        "step": update_step,
                        **checkpoint_metrics,
                    }
                )
                output = args.checkpoint_directory / f"step-{update_step:07d}"
                save_candidate(
                    model,
                    tokenizer,
                    output,
                    checkpoint_manifest(
                        common=common,
                        history=history,
                        step=update_step,
                        checkpoint_metrics=checkpoint_metrics,
                    ),
                )
                print(
                    json.dumps(
                        history[-1],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if update_step >= training["max_steps"]:
                break

    synchronize(device)


if __name__ == "__main__":
    main()
