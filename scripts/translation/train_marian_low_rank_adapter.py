#!/usr/bin/env python3
"""Train mergeable low-rank adapters on one Marian translation direction."""

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
import torch.nn.functional as F
import transformers
from torch import nn
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup

from marian_low_rank_adapter import (
    LowRankLinear,
    adapter_modules,
    apply_low_rank_adapters,
    capture_trainable_state,
    load_trainable_state,
    merge_low_rank_adapters,
    save_adapter,
    unfreeze_layer_norm_scales,
)
from train_marian_distillation import (
    Collator,
    TranslationRows,
    checkpoint_identity,
    checkpoint_lineage_manifests,
    curriculum_domain_weight,
    evaluate,
    frozen_base_kl,
    hardware_name,
    load_rows,
    move,
    save_candidate,
)
from training_manifest_provenance import (
    authenticate_dataset_manifest,
    derive_target_provenance,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_smoothed_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    preservation_mask: torch.Tensor,
    focus_weight: float,
    label_smoothing: float,
) -> torch.Tensor:
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(labels.shape)
    token_mask = labels.ne(-100)
    row_weights = torch.where(
        preservation_mask,
        torch.ones_like(preservation_mask, dtype=token_losses.dtype),
        torch.full_like(preservation_mask, focus_weight, dtype=token_losses.dtype),
    )
    weights = token_mask.to(token_losses.dtype) * row_weights[:, None]
    return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)


def module_by_name(model: nn.Module, name: str) -> nn.Module:
    module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def initialize_from_checkpoint(
    model: MarianMTModel,
    checkpoint: Path,
) -> dict[str, float]:
    specialist = MarianMTModel.from_pretrained(checkpoint)
    captures: dict[str, float] = {}
    for name, adapter in adapter_modules(model).items():
        source = module_by_name(specialist, name)
        if not isinstance(source, nn.Linear):
            raise SystemExit(f"SVD source is not Linear: {name}")
        delta = (
            source.weight.detach().float().cpu()
            - adapter.base.weight.detach().float().cpu()
        )
        captures[name] = adapter.initialize_from_delta(delta)
    del specialist
    return captures


def validate_checkpoint_identity(
    checkpoint: Path,
    *,
    direction: str,
    repository: str,
    revision: str,
    label: str,
) -> None:
    if not (checkpoint / "model.safetensors").is_file():
        raise SystemExit(f"{label} checkpoint is incomplete: {checkpoint}")
    actual = checkpoint_identity(checkpoint)
    expected = (direction, repository, revision)
    if actual is not None and actual != expected:
        raise SystemExit(
            f"{label} checkpoint identity differs: expected {expected}, found {actual}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=("en-ja", "ja-en"), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--preset",
        choices=("consultation-v1", "all-projections"),
        default="consultation-v1",
    )
    parser.add_argument("--encoder-top-layers", type=int, default=3)
    parser.add_argument("--train-layer-norm-scales", action="store_true")
    parser.add_argument("--svd-initialization-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--evaluation-steps", type=int, default=50)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--frozen-base-kl-weight", type=float, default=0.0)
    parser.add_argument("--focus-loss-weight-start", type=float, default=1.0)
    parser.add_argument("--focus-loss-weight-end", type=float, default=2.0)
    parser.add_argument("--curriculum-ramp-steps", type=int)
    parser.add_argument(
        "--focus-origin",
        action="append",
        default=[],
        help="Origins receiving the focus loss weight; defaults to the balanced hard-reference origin.",
    )
    parser.add_argument(
        "--training-description",
        default=(
            "direct low-rank adaptation on licensed human parallel references; "
            "no synthetic reasoning traces"
        ),
    )
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
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
        or not 0 <= args.label_smoothing < 1
        or min(
            args.frozen_base_kl_weight,
            args.focus_loss_weight_start,
            args.focus_loss_weight_end,
        ) < 0
    ):
        raise SystemExit("invalid adapter, smoothing, or loss-weight configuration")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    validate_checkpoint_identity(
        args.initial_checkpoint,
        direction=args.direction,
        repository=args.repository,
        revision=args.revision,
        label="initial",
    )
    if args.svd_initialization_checkpoint is not None:
        validate_checkpoint_identity(
            args.svd_initialization_checkpoint,
            direction=args.direction,
            repository=args.repository,
            revision=args.revision,
            label="SVD initialization",
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_path = args.dataset_directory / "train.jsonl"
    valid_path = args.dataset_directory / "valid.jsonl"
    train_rows = load_rows(train_path, args.direction)
    valid_rows = load_rows(valid_path, args.direction)
    dataset_manifest, dataset_manifest_metadata = authenticate_dataset_manifest(
        args.dataset_directory,
        direction=args.direction,
        train_path=train_path,
        valid_path=valid_path,
    )
    target_provenance = derive_target_provenance(
        dataset_manifest,
        train_rows,
        fallback_training_description=args.training_description,
    )

    tokenizer = MarianTokenizer.from_pretrained(args.initial_checkpoint)
    model = MarianMTModel.from_pretrained(args.initial_checkpoint).to(device)
    selected_modules = apply_low_rank_adapters(
        model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        preset=args.preset,
        encoder_layers=model.config.encoder_layers,
        encoder_top_layers=args.encoder_top_layers,
    )
    layer_norm_scales = (
        unfreeze_layer_norm_scales(model) if args.train_layer_norm_scales else []
    )
    svd_captured_energy: dict[str, float] = {}
    if args.svd_initialization_checkpoint is not None:
        svd_captured_energy = initialize_from_checkpoint(
            model, args.svd_initialization_checkpoint
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    base_model = None
    if args.frozen_base_kl_weight > 0:
        base_model = MarianMTModel.from_pretrained(args.initial_checkpoint).to(device)
        base_model.eval()
        base_model.requires_grad_(False)

    focus_origins = set(args.focus_origin or ["human-balanced-hard-reference"])
    observed_origins = {str(row.get("origin", "unknown")) for row in train_rows}
    missing_focus = focus_origins - observed_origins
    if missing_focus:
        raise SystemExit(f"focus origins are absent from training data: {sorted(missing_focus)}")
    preservation_origins = observed_origins - focus_origins
    collator = Collator(
        tokenizer,
        args.max_source_tokens,
        args.max_target_tokens,
        preservation_origins,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = TranslationRows(
        train_rows, seed=args.seed, sample_target_variants=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        TranslationRows(valid_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
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

    history: list[dict] = []
    initial_metrics = evaluate(
        model, tokenizer, valid_loader, valid_rows, device, args.max_target_tokens
    )
    history.append({"step": 0, **initial_metrics})
    best = {"step": 0, **initial_metrics}
    best_state = capture_trainable_state(model)

    update_step = 0
    micro_step = 0
    epoch = 0
    model.train()
    while update_step < args.max_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            batch = move(batch, device)
            preservation_mask = batch.pop("preservation_mask")
            labels = batch["labels"]
            outputs = model(**batch)
            focus_weight = curriculum_domain_weight(
                update_step,
                args.curriculum_ramp_steps or args.max_steps,
                args.focus_loss_weight_start,
                args.focus_loss_weight_end,
            )
            sequence_loss = weighted_smoothed_cross_entropy(
                outputs.logits,
                labels,
                preservation_mask,
                focus_weight,
                args.label_smoothing,
            )
            kl_loss = outputs.logits.new_zeros((), dtype=torch.float32)
            if (
                args.frozen_base_kl_weight > 0
                and bool(preservation_mask.any())
            ):
                assert base_model is not None
                with torch.inference_mode():
                    base_logits = base_model(**batch).logits
                kl_loss = frozen_base_kl(
                    outputs.logits,
                    base_logits,
                    labels,
                    preservation_mask,
                )
            combined_loss = (
                sequence_loss + args.frozen_base_kl_weight * kl_loss
            )
            (combined_loss / args.gradient_accumulation).backward()
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
                    model,
                    tokenizer,
                    valid_loader,
                    valid_rows,
                    device,
                    args.max_target_tokens,
                )
                record = {
                    "step": update_step,
                    **metrics,
                    "training_objective": {
                        "sequence_loss": float(sequence_loss.detach().cpu()),
                        "frozen_base_kl": float(kl_loss.detach().cpu()),
                        "focus_loss_weight": focus_weight,
                    },
                }
                history.append(record)
                if (metrics["chrf_pp"], -metrics["loss"]) > (
                    best["chrf_pp"],
                    -best["loss"],
                ):
                    best = record
                    best_state = capture_trainable_state(model)
                print(json.dumps({"current": record, "best": best}, ensure_ascii=False))
            if update_step >= args.max_steps:
                break
        epoch += 1

    load_trainable_state(model, best_state)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    adapter_path = args.output_directory / "adapters.safetensors"
    save_adapter(model, adapter_path)
    adapter_config = {
        "format": "mimi-marian-low-rank-v1",
        "rank": args.rank,
        "alpha": args.alpha,
        "scale": args.alpha / args.rank,
        "dropout": args.dropout,
        "preset": args.preset,
        "encoder_top_layers": args.encoder_top_layers,
        "selected_modules": selected_modules,
        "train_layer_norm_scales": args.train_layer_norm_scales,
        "layer_norm_scales": layer_norm_scales,
        "merged_for_deployment": True,
    }
    adapter_config_path = args.output_directory / "adapter_config.json"
    adapter_config_path.write_text(
        json.dumps(adapter_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    merged_modules = merge_low_rank_adapters(model)
    if merged_modules != selected_modules or adapter_modules(model):
        raise RuntimeError("adapter merge did not preserve the selected module set")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "direction": args.direction,
        "student_repository": args.repository,
        "student_revision": args.revision,
        "license": "CC-BY-SA-4.0",
        "approach": "direct mergeable low-rank Marian adaptation",
        "private_reasoning_traces_used": False,
        "training_description": target_provenance["training_description"],
        "initial_checkpoint": {
            "path": str(args.initial_checkpoint.resolve()),
            "model_sha256": sha256(args.initial_checkpoint / "model.safetensors"),
            "lineage_manifests": checkpoint_lineage_manifests(
                args.initial_checkpoint
            ),
        },
        "svd_initialization_checkpoint": (
            {
                "path": str(args.svd_initialization_checkpoint.resolve()),
                "model_sha256": sha256(
                    args.svd_initialization_checkpoint / "model.safetensors"
                ),
                "captured_delta_energy_by_module": svd_captured_energy,
            }
            if args.svd_initialization_checkpoint is not None
            else None
        ),
        "dataset": {
            "train_path": str(train_path.resolve()),
            "train_sha256": sha256(train_path),
            "train_rows": len(train_rows),
            "valid_path": str(valid_path.resolve()),
            "valid_sha256": sha256(valid_path),
            "valid_rows": len(valid_rows),
            "origins": dict(
                sorted(Counter(str(row.get("origin", "unknown")) for row in train_rows).items())
            ),
        },
        "dataset_manifest": dataset_manifest_metadata,
        "adapter": {
            **adapter_config,
            "adapter_path": str(adapter_path.resolve()),
            "adapter_sha256": sha256(adapter_path),
            "adapter_bytes": adapter_path.stat().st_size,
            "config_path": str(adapter_config_path.resolve()),
            "config_sha256": sha256(adapter_config_path),
            "trainable_parameters": trainable_count,
            "total_parameters_including_wrappers": total_count,
            "trainable_fraction": trainable_count / total_count,
        },
        "objective": {
            "sequence_target": target_provenance["sequence_target"],
            "label_smoothing": args.label_smoothing,
            "focus_origins": sorted(focus_origins),
            "preservation_origins": sorted(preservation_origins),
            "frozen_base_kl_weight": args.frozen_base_kl_weight,
            "selection": "highest validation chrF++; lower loss tie-break",
        },
        "hyperparameters": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "evaluation_steps": args.evaluation_steps,
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "gradient_checkpointing": args.gradient_checkpointing,
            "focus_loss_weight_start": args.focus_loss_weight_start,
            "focus_loss_weight_end": args.focus_loss_weight_end,
            "curriculum_ramp_steps": args.curriculum_ramp_steps or args.max_steps,
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "best": best,
        "history": history,
    }
    save_candidate(model, tokenizer, args.output_directory, manifest)
    print(json.dumps(
        {
            "output": str(args.output_directory),
            "best": best,
            "adapter_bytes": adapter_path.stat().st_size,
            "trainable_parameters": trainable_count,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
