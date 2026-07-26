#!/usr/bin/env python3
"""Run the preregistered v14 rollout-conditioned JA-to-EN repair arm."""

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

EXPERIMENT = "canonical-rollout-repair-v14-ja-en"


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


class RecoveryRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class RecoveryCollator:
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        *,
        decoder_start_token_id: int,
        max_source_tokens: int,
        max_target_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens

    def __call__(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        source = self.tokenizer(
            [str(row["source"]) for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_source_tokens,
            return_tensors="pt",
        )
        prefixes = []
        positions = []
        for row in rows:
            tokens = [int(value) for value in row["rollout_token_ids"]]
            trigger = int(row["trigger_index"])
            if not 0 < trigger < len(tokens):
                raise ValueError(f"invalid rollout recovery trigger: {row['id']}")
            prefix = tokens[:trigger][: self.max_target_tokens - 1]
            if not prefix:
                raise ValueError(f"empty rollout recovery prefix: {row['id']}")
            decoder = [self.decoder_start_token_id, *prefix]
            prefixes.append(decoder)
            positions.append(len(decoder) - 1)
        width = max(len(prefix) for prefix in prefixes)
        decoder_input_ids = torch.full(
            (len(rows), width),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
        )
        decoder_attention_mask = torch.zeros(
            (len(rows), width),
            dtype=torch.long,
        )
        for index, prefix in enumerate(prefixes):
            decoder_input_ids[index, : len(prefix)] = torch.tensor(
                prefix,
                dtype=torch.long,
            )
            decoder_attention_mask[index, : len(prefix)] = 1
        return {
            **source,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "recovery_positions": torch.tensor(
                positions,
                dtype=torch.long,
            ),
            "recovery_token_ids": torch.tensor(
                [int(row["recovery_token_id"]) for row in rows],
                dtype=torch.long,
            ),
            "rejected_token_ids": torch.tensor(
                [int(row["rejected_token_id"]) for row in rows],
                dtype=torch.long,
            ),
        }


def recovery_logits(
    model: MarianMTModel,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch["decoder_input_ids"],
        decoder_attention_mask=batch["decoder_attention_mask"],
    )
    positions = batch["recovery_positions"]
    return outputs.logits[
        torch.arange(
            outputs.logits.shape[0],
            device=outputs.logits.device,
        ),
        positions,
    ]


def recovery_losses(
    logits: torch.Tensor,
    recovery_token_ids: torch.Tensor,
    rejected_token_ids: torch.Tensor,
    *,
    target_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits.float(), dim=-1)
    rejected_probability = probabilities.gather(
        1,
        rejected_token_ids[:, None],
    ).squeeze(1)
    unlikelihood = -torch.log1p(-rejected_probability.clamp(max=1 - 1e-6)).mean()
    recovery_scores = logits.gather(
        1,
        recovery_token_ids[:, None],
    ).squeeze(1)
    rejected_scores = logits.gather(
        1,
        rejected_token_ids[:, None],
    ).squeeze(1)
    margins = recovery_scores - rejected_scores
    ranking = F.relu(target_margin - margins).mean()
    preferred = recovery_scores > rejected_scores
    return (
        unlikelihood,
        ranking,
        rejected_probability,
        preferred,
    )


def scheduled_decoder_inputs(
    model: MarianMTModel,
    labels: torch.Tensor,
    teacher_forced_logits: torch.Tensor,
    *,
    probability: float,
    generator: torch.Generator,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
    predictions = teacher_forced_logits.detach().argmax(dim=-1)
    eligible = labels[:, :-1].ne(-100) & predictions[:, :-1].ne(pad_token_id)
    draws = torch.rand(
        eligible.shape,
        generator=generator,
        device="cpu",
    ).to(eligible.device)
    replacements = eligible & draws.lt(probability)
    decoder_input_ids[:, 1:] = torch.where(
        replacements,
        predictions[:, :-1],
        decoder_input_ids[:, 1:],
    )
    return decoder_input_ids, replacements


@torch.inference_mode()
def evaluate_recovery(
    model: MarianMTModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    probabilities = []
    preferences = []
    margins = []
    for raw_batch in loader:
        batch = move(raw_batch, device)
        logits = recovery_logits(model, batch)
        values = F.softmax(logits.float(), dim=-1)
        rejected = values.gather(
            1,
            batch["rejected_token_ids"][:, None],
        ).squeeze(1)
        recovery_scores = logits.gather(
            1,
            batch["recovery_token_ids"][:, None],
        ).squeeze(1)
        rejected_scores = logits.gather(
            1,
            batch["rejected_token_ids"][:, None],
        ).squeeze(1)
        probabilities.extend(float(value) for value in rejected.detach().cpu())
        preferences.extend(
            bool(value) for value in (recovery_scores > rejected_scores).detach().cpu()
        )
        margins.extend(
            float(value) for value in (recovery_scores - rejected_scores).detach().cpu()
        )
    synchronize(device)
    model.train()
    return {
        "cases": len(probabilities),
        "mean_rejected_token_probability": sum(probabilities) / len(probabilities),
        "recovery_preference_accuracy": sum(preferences) / len(preferences),
        "mean_recovery_margin": sum(margins) / len(margins),
    }


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
        raise SystemExit("v14 contract safety state differs")
    for section, field in (
        ("dataset", "manifest"),
        ("dataset", "hard_train"),
        ("dataset", "valid"),
        ("rollout_dataset", "manifest"),
        ("rollout_dataset", "hard"),
        ("rollout_dataset", "recovery"),
        ("initial_checkpoint", "model"),
        ("initial_checkpoint", "training_manifest"),
        ("preservation_checkpoint", "model"),
        ("preservation_checkpoint", "training_manifest"),
    ):
        item = contract[section][field]
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"v14 bound input differs: {section}.{field}")
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
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
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
    expected_checkpoints = training["checkpoint_steps"]
    if expected_checkpoints != [
        training["evaluation_steps"],
        training["max_steps"],
    ]:
        raise SystemExit("v14 checkpoint schedule differs")

    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    scheduled_generator = torch.Generator().manual_seed(seed + 1)
    device = torch.device(args.device)

    hard_path = root / contract["rollout_dataset"]["hard"]["path"]
    recovery_path = root / contract["rollout_dataset"]["recovery"]["path"]
    valid_path = root / contract["dataset"]["valid"]["path"]
    hard_rows = load_rows(hard_path, "ja-en")
    recovery_rows = load_jsonl(recovery_path)
    valid_rows = load_rows(valid_path, "ja-en")
    if (
        len(hard_rows) != contract["rollout_dataset"]["counts"]["hard"]
        or len(recovery_rows) != contract["rollout_dataset"]["counts"]["recovery"]
        or len(valid_rows) != contract["dataset"]["counts"]["valid"]
    ):
        raise SystemExit("v14 bound row counts differ")

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
            raise SystemExit(f"v14 {label} checkpoint identity differs")

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

    collator = Collator(
        tokenizer,
        training["max_source_tokens"],
        training["max_target_tokens"],
        set(),
    )
    train_loader = DataLoader(
        TranslationRows(hard_rows),
        batch_size=training["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        TranslationRows(valid_rows),
        batch_size=training["evaluation_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )
    recovery_collator = RecoveryCollator(
        tokenizer,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        max_source_tokens=training["max_source_tokens"],
        max_target_tokens=training["max_target_tokens"],
    )
    recovery_loader = DataLoader(
        RecoveryRows(recovery_rows),
        batch_size=training["recovery_batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 2),
        collate_fn=recovery_collator,
    )
    recovery_evaluation_loader = DataLoader(
        RecoveryRows(recovery_rows),
        batch_size=training["evaluation_batch_size"],
        shuffle=False,
        collate_fn=recovery_collator,
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
        valid_loader,
        valid_rows,
        device,
        training["max_target_tokens"],
    )
    initial_recovery = evaluate_recovery(
        model,
        recovery_evaluation_loader,
        device,
    )
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "translation": initial_translation,
            "recovery": initial_recovery,
        }
    ]
    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "operation": (
            "parallel scheduled sampling on free-running hard rows plus "
            "explicit EOS recovery at mined repetition prefixes"
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
        "rollout_dataset_manifest": contract["rollout_dataset"]["manifest"],
        "dataset_provenance": {
            "direction": "ja-en",
            "experiment": EXPERIMENT,
            "target_source": ("authenticated licensed-human references only"),
            "effective_licenses": contract["dataset"]["effective_licenses"],
            "promotion_eligible": False,
            "rollout_strings_are_positive_targets": False,
            "recovery_target_is_only_eos": True,
        },
        "dataset": {
            "directory": contract["dataset"]["directory"],
            "hard_train_sha256": contract["dataset"]["hard_train"]["sha256"],
            "valid_sha256": contract["dataset"]["valid"]["sha256"],
            "hard_train_rows": len(hard_rows),
            "valid_rows": len(valid_rows),
            "recovery_rows": len(recovery_rows),
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "hyperparameters": training,
        "objective": {
            "positive": (
                "ordinary token cross-entropy on licensed-human "
                "references from free-running hard rows"
            ),
            "scheduled_sampling": (
                "cross-entropy under one-step model-predicted decoder "
                "tokens sampled from the teacher-forced pass"
            ),
            "recovery": (
                "EOS-over-repeated-token ranking and token-local "
                "unlikelihood under actual free-running bad prefixes"
            ),
            "retention": (
                "teacher-forced token KL and parameter L2 to the frozen safe parent"
            ),
            "rollout_strings_are_positive_targets": False,
            "free_form_synthetic_translations_used_as_targets": False,
            "private_reasoning_traces_used": False,
            "human_review_required": False,
        },
    }

    update_step = 0
    micro_step = 0
    recovery_iterator = iter(recovery_loader)
    model.train()
    while update_step < training["max_steps"]:
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            batch.pop("preservation_mask")
            teacher_forced = model(**batch)
            scheduled_inputs, replacements = scheduled_decoder_inputs(
                model,
                batch["labels"],
                teacher_forced.logits,
                probability=training["scheduled_sampling_probability"],
                generator=scheduled_generator,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            scheduled = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=scheduled_inputs,
                labels=batch["labels"],
            )
            try:
                raw_recovery = next(recovery_iterator)
            except StopIteration:
                recovery_iterator = iter(recovery_loader)
                raw_recovery = next(recovery_iterator)
            recovery_batch = move(raw_recovery, device)
            logits = recovery_logits(model, recovery_batch)
            (
                recovery_unlikelihood,
                recovery_ranking,
                rejected_probability,
                recovery_preferred,
            ) = recovery_losses(
                logits,
                recovery_batch["recovery_token_ids"],
                recovery_batch["rejected_token_ids"],
                target_margin=training["recovery_ranking_target_margin"],
            )
            with torch.no_grad():
                parent_logits = preservation_model(**batch).logits
            preservation_mask = torch.ones(
                batch["labels"].shape[0],
                dtype=torch.bool,
                device=device,
            )
            kl_loss = frozen_base_kl(
                teacher_forced.logits,
                parent_logits,
                batch["labels"],
                preservation_mask,
            )
            l2_loss = l2_to_frozen_base(model, base_parameters)
            combined = (
                teacher_forced.loss
                + training["scheduled_sampling_weight"] * scheduled.loss
                + training["recovery_unlikelihood_weight"] * recovery_unlikelihood
                + training["recovery_ranking_weight"] * recovery_ranking
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
                    valid_loader,
                    valid_rows,
                    device,
                    training["max_target_tokens"],
                )
                recovery = evaluate_recovery(
                    model,
                    recovery_evaluation_loader,
                    device,
                )
                checkpoint_metrics = {
                    "translation": translation,
                    "recovery": recovery,
                    "last_train_objective": {
                        "positive_cross_entropy": float(
                            teacher_forced.loss.detach().cpu()
                        ),
                        "scheduled_sampling_cross_entropy": float(
                            scheduled.loss.detach().cpu()
                        ),
                        "scheduled_replacement_fraction": float(
                            replacements.float().mean().detach().cpu()
                        ),
                        "recovery_unlikelihood": float(
                            recovery_unlikelihood.detach().cpu()
                        ),
                        "recovery_ranking": float(recovery_ranking.detach().cpu()),
                        "recovery_preference_accuracy": float(
                            recovery_preferred.float().mean().detach().cpu()
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
