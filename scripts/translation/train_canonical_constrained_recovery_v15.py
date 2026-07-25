#!/usr/bin/env python3
"""Train the preregistered v15 constrained-recovery Marian arm."""

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

EXPERIMENT = "canonical-constrained-recovery-v15-ja-en"


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


class ContrastRows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def pack_prefixes(
    prefixes: list[list[int]],
    *,
    decoder_start_token_id: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = [[decoder_start_token_id, *prefix] for prefix in prefixes]
    if any(len(value) < 2 for value in values):
        raise ValueError("contrastive prefix must contain at least one target token")
    width = max(len(value) for value in values)
    ids = torch.full(
        (len(values), width),
        pad_token_id,
        dtype=torch.long,
    )
    mask = torch.zeros((len(values), width), dtype=torch.long)
    positions = []
    for index, value in enumerate(values):
        ids[index, : len(value)] = torch.tensor(value, dtype=torch.long)
        mask[index, : len(value)] = 1
        positions.append(len(value) - 1)
    return ids, mask, torch.tensor(positions, dtype=torch.long)


class PrefixContrastCollator:
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        *,
        decoder_start_token_id: int,
        max_source_tokens: int,
        max_target_tokens: int,
        includes_perturbed_prefix: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens
        self.includes_perturbed_prefix = includes_perturbed_prefix

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        source = self.tokenizer(
            [str(row["source"]) for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_source_tokens,
            return_tensors="pt",
        )
        clean_prefixes = [
            [int(value) for value in row["clean_prefix_token_ids"]][
                : self.max_target_tokens - 1
            ]
            for row in rows
        ]
        clean_ids, clean_mask, clean_positions = pack_prefixes(
            clean_prefixes,
            decoder_start_token_id=self.decoder_start_token_id,
            pad_token_id=int(self.tokenizer.pad_token_id),
        )
        batch = {
            **source,
            "clean_decoder_input_ids": clean_ids,
            "clean_decoder_attention_mask": clean_mask,
            "clean_positions": clean_positions,
            "correct_token_ids": torch.tensor(
                [int(row["correct_token_id"]) for row in rows],
                dtype=torch.long,
            ),
            "rejected_token_ids": torch.tensor(
                [int(row["rejected_token_id"]) for row in rows],
                dtype=torch.long,
            ),
        }
        if self.includes_perturbed_prefix:
            perturbed_prefixes = [
                [int(value) for value in row["perturbed_prefix_token_ids"]][
                    : self.max_target_tokens - 1
                ]
                for row in rows
            ]
            perturbed_ids, perturbed_mask, perturbed_positions = pack_prefixes(
                perturbed_prefixes,
                decoder_start_token_id=self.decoder_start_token_id,
                pad_token_id=int(self.tokenizer.pad_token_id),
            )
            batch.update(
                {
                    "perturbed_decoder_input_ids": perturbed_ids,
                    "perturbed_decoder_attention_mask": perturbed_mask,
                    "perturbed_positions": perturbed_positions,
                }
            )
        return batch


def prefix_logits(
    model: MarianMTModel,
    batch: dict[str, torch.Tensor],
    *,
    prefix: str,
) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        decoder_input_ids=batch[f"{prefix}_decoder_input_ids"],
        decoder_attention_mask=batch[f"{prefix}_decoder_attention_mask"],
    )
    positions = batch[f"{prefix}_positions"]
    return outputs.logits[
        torch.arange(outputs.logits.shape[0], device=outputs.logits.device),
        positions,
    ]


def constrained_recovery_losses(
    clean_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    correct_token_ids: torch.Tensor,
    rejected_token_ids: torch.Tensor,
    *,
    recovery_margin: float,
    clean_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_log_probabilities = F.log_softmax(clean_logits.float(), dim=-1)
    perturbed_log_probabilities = F.log_softmax(
        perturbed_logits.float(),
        dim=-1,
    )
    clean_correct = clean_log_probabilities.gather(
        1,
        correct_token_ids[:, None],
    ).squeeze(1)
    perturbed_correct = perturbed_log_probabilities.gather(
        1,
        correct_token_ids[:, None],
    ).squeeze(1)
    perturbed_rejected = perturbed_log_probabilities.gather(
        1,
        rejected_token_ids[:, None],
    ).squeeze(1)
    recovery_margins = perturbed_correct - perturbed_rejected
    clean_margins = clean_correct - perturbed_correct
    recovery_loss = F.relu(recovery_margin - recovery_margins).mean()
    constrained_loss = F.relu(clean_margin - clean_margins).mean()
    return (
        recovery_loss,
        constrained_loss,
        recovery_margins,
        clean_margins,
        perturbed_rejected.exp(),
    )


def omission_losses(
    clean_logits: torch.Tensor,
    correct_token_ids: torch.Tensor,
    rejected_token_ids: torch.Tensor,
    *,
    target_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probabilities = F.log_softmax(clean_logits.float(), dim=-1)
    correct = log_probabilities.gather(
        1,
        correct_token_ids[:, None],
    ).squeeze(1)
    rejected = log_probabilities.gather(
        1,
        rejected_token_ids[:, None],
    ).squeeze(1)
    margins = correct - rejected
    return F.relu(target_margin - margins).mean(), margins


@torch.inference_mode()
def evaluate_contrasts(
    model: MarianMTModel,
    recovery_loader: DataLoader,
    omission_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    recovery_margins = []
    clean_margins = []
    rejected_probabilities = []
    for raw_batch in recovery_loader:
        batch = move(raw_batch, device)
        _, _, recovery, clean, rejected = constrained_recovery_losses(
            prefix_logits(model, batch, prefix="clean"),
            prefix_logits(model, batch, prefix="perturbed"),
            batch["correct_token_ids"],
            batch["rejected_token_ids"],
            recovery_margin=0.0,
            clean_margin=0.0,
        )
        recovery_margins.extend(float(value) for value in recovery.detach().cpu())
        clean_margins.extend(float(value) for value in clean.detach().cpu())
        rejected_probabilities.extend(float(value) for value in rejected.detach().cpu())
    omission_margins = []
    for raw_batch in omission_loader:
        batch = move(raw_batch, device)
        _, margins = omission_losses(
            prefix_logits(model, batch, prefix="clean"),
            batch["correct_token_ids"],
            batch["rejected_token_ids"],
            target_margin=0.0,
        )
        omission_margins.extend(float(value) for value in margins.detach().cpu())
    synchronize(device)
    model.train()
    return {
        "recovery_cases": len(recovery_margins),
        "mean_recovery_margin": sum(recovery_margins) / len(recovery_margins),
        "recovery_preference_accuracy": sum(value > 0 for value in recovery_margins)
        / len(recovery_margins),
        "mean_clean_over_recovery_margin": sum(clean_margins) / len(clean_margins),
        "clean_over_recovery_accuracy": sum(value > 0 for value in clean_margins)
        / len(clean_margins),
        "mean_rejected_token_probability": sum(rejected_probabilities)
        / len(rejected_probabilities),
        "omission_cases": len(omission_margins),
        "mean_omission_margin": sum(omission_margins) / len(omission_margins),
        "omission_preference_accuracy": sum(value > 0 for value in omission_margins)
        / len(omission_margins),
    }


def validate_contract(contract_path: Path, *, root: Path) -> dict[str, Any]:
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
        raise SystemExit("v15 contract safety state differs")
    for section, field in (
        ("dataset", "manifest"),
        ("dataset", "train"),
        ("dataset", "fresh_valid"),
        ("dataset", "v12_regression"),
        ("dataset", "v14_regression"),
        ("contrastive_examples", "manifest"),
        ("contrastive_examples", "recovery"),
        ("contrastive_examples", "omission"),
        ("initial_checkpoint", "model"),
        ("initial_checkpoint", "training_manifest"),
        ("preservation_checkpoint", "model"),
        ("preservation_checkpoint", "training_manifest"),
    ):
        item = contract[section][field]
        path = root / item["path"]
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"v15 bound input differs: {section}.{field}")
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


def next_batch(
    iterator: Any,
    loader: DataLoader,
) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


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
        raise SystemExit("v15 checkpoint schedule differs")

    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    train_rows = load_rows(root / contract["dataset"]["train"]["path"], "ja-en")
    fresh_valid_rows = load_rows(
        root / contract["dataset"]["fresh_valid"]["path"],
        "ja-en",
    )
    recovery_rows = load_jsonl(
        root / contract["contrastive_examples"]["recovery"]["path"]
    )
    omission_rows = load_jsonl(
        root / contract["contrastive_examples"]["omission"]["path"]
    )
    if (
        len(train_rows) != contract["dataset"]["counts"]["train"]
        or len(fresh_valid_rows) != contract["dataset"]["counts"]["fresh_valid"]
        or len(recovery_rows) != contract["contrastive_examples"]["counts"]["recovery"]
        or len(omission_rows) != contract["contrastive_examples"]["counts"]["omission"]
    ):
        raise SystemExit("v15 bound row counts differ")

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
            raise SystemExit(f"v15 {label} checkpoint identity differs")

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
    fresh_valid_loader = DataLoader(
        TranslationRows(fresh_valid_rows),
        batch_size=training["evaluation_batch_size"],
        shuffle=False,
        collate_fn=collator,
    )
    recovery_collator = PrefixContrastCollator(
        tokenizer,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        max_source_tokens=training["max_source_tokens"],
        max_target_tokens=training["max_target_tokens"],
        includes_perturbed_prefix=True,
    )
    omission_collator = PrefixContrastCollator(
        tokenizer,
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        max_source_tokens=training["max_source_tokens"],
        max_target_tokens=training["max_target_tokens"],
        includes_perturbed_prefix=False,
    )
    recovery_loader = DataLoader(
        ContrastRows(recovery_rows),
        batch_size=training["contrast_batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
        collate_fn=recovery_collator,
    )
    omission_loader = DataLoader(
        ContrastRows(omission_rows),
        batch_size=training["contrast_batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 2),
        collate_fn=omission_collator,
    )
    recovery_evaluation_loader = DataLoader(
        ContrastRows(recovery_rows),
        batch_size=training["evaluation_batch_size"],
        shuffle=False,
        collate_fn=recovery_collator,
    )
    omission_evaluation_loader = DataLoader(
        ContrastRows(omission_rows),
        batch_size=training["evaluation_batch_size"],
        shuffle=False,
        collate_fn=omission_collator,
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
        fresh_valid_loader,
        fresh_valid_rows,
        device,
        training["max_target_tokens"],
    )
    initial_contrasts = evaluate_contrasts(
        model,
        recovery_evaluation_loader,
        omission_evaluation_loader,
        device,
    )
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "fresh_translation": initial_translation,
            "contrasts": initial_contrasts,
        }
    ]
    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "operation": (
            "licensed-reference MLE plus clean/recovery ordering and "
            "token-span omission contrast"
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
        "contrastive_examples_manifest": contract["contrastive_examples"]["manifest"],
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
            "train_rows": len(train_rows),
            "fresh_valid_rows": len(fresh_valid_rows),
            "recovery_rows": len(recovery_rows),
            "omission_rows": len(omission_rows),
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "hyperparameters": training,
        "objective": {
            "positive": "token cross-entropy on licensed-human references",
            "recovery": (
                "licensed continuation over generated or repeated continuation "
                "under a model-like perturbed prefix"
            ),
            "constrained_recovery": (
                "licensed continuation under its clean prefix over the same "
                "continuation under a perturbed prefix"
            ),
            "omission": (
                "first licensed span token over the token after the "
                "deterministically deleted span"
            ),
            "retention": "teacher-forced KL and parameter L2 to the safe parent",
            "generated_strings_are_positive_targets": False,
            "free_form_synthetic_translations_used_as_targets": False,
            "private_reasoning_traces_used": False,
            "human_review_required": False,
        },
    }

    update_step = 0
    micro_step = 0
    recovery_iterator = iter(recovery_loader)
    omission_iterator = iter(omission_loader)
    model.train()
    while update_step < training["max_steps"]:
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            batch.pop("preservation_mask")
            teacher_forced = model(**batch)

            raw_recovery, recovery_iterator = next_batch(
                recovery_iterator,
                recovery_loader,
            )
            recovery_batch = move(raw_recovery, device)
            (
                recovery_loss,
                constrained_loss,
                recovery_margins,
                clean_margins,
                rejected_probabilities,
            ) = constrained_recovery_losses(
                prefix_logits(model, recovery_batch, prefix="clean"),
                prefix_logits(model, recovery_batch, prefix="perturbed"),
                recovery_batch["correct_token_ids"],
                recovery_batch["rejected_token_ids"],
                recovery_margin=training["recovery_target_margin"],
                clean_margin=training["clean_over_recovery_target_margin"],
            )

            raw_omission, omission_iterator = next_batch(
                omission_iterator,
                omission_loader,
            )
            omission_batch = move(raw_omission, device)
            omission_loss, omission_margins = omission_losses(
                prefix_logits(model, omission_batch, prefix="clean"),
                omission_batch["correct_token_ids"],
                omission_batch["rejected_token_ids"],
                target_margin=training["omission_target_margin"],
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
                + training["recovery_weight"] * recovery_loss
                + training["constrained_recovery_weight"] * constrained_loss
                + training["omission_weight"] * omission_loss
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
                    fresh_valid_loader,
                    fresh_valid_rows,
                    device,
                    training["max_target_tokens"],
                )
                contrasts = evaluate_contrasts(
                    model,
                    recovery_evaluation_loader,
                    omission_evaluation_loader,
                    device,
                )
                checkpoint_metrics = {
                    "fresh_translation": translation,
                    "contrasts": contrasts,
                    "last_train_objective": {
                        "positive_cross_entropy": float(
                            teacher_forced.loss.detach().cpu()
                        ),
                        "recovery_ranking": float(recovery_loss.detach().cpu()),
                        "constrained_recovery_ranking": float(
                            constrained_loss.detach().cpu()
                        ),
                        "omission_ranking": float(omission_loss.detach().cpu()),
                        "recovery_preference_accuracy": float(
                            recovery_margins.gt(0).float().mean().detach().cpu()
                        ),
                        "clean_over_recovery_accuracy": float(
                            clean_margins.gt(0).float().mean().detach().cpu()
                        ),
                        "omission_preference_accuracy": float(
                            omission_margins.gt(0).float().mean().detach().cpu()
                        ),
                        "mean_rejected_token_probability": float(
                            rejected_probabilities.mean().detach().cpu()
                        ),
                        "frozen_parent_kl": float(kl_loss.detach().cpu()),
                        "l2_to_parent": float(l2_loss.detach().cpu()),
                        "combined": float(combined.detach().cpu()),
                    },
                }
                history.append({"step": update_step, **checkpoint_metrics})
                manifest = {
                    **common,
                    "history": history,
                    "checkpoint_step": update_step,
                    "checkpoint_metrics": checkpoint_metrics,
                    "promotion_eligible": False,
                    "quantization_authorized": False,
                    "protected_evaluation_authorized": False,
                    "app_change_authorized": False,
                    "public_upload_authorized": False,
                }
                save_candidate(
                    model,
                    tokenizer,
                    args.checkpoint_directory / f"step-{update_step:07d}",
                    manifest,
                )
                print(
                    json.dumps(history[-1], ensure_ascii=False, sort_keys=True),
                    flush=True,
                )
            if update_step >= training["max_steps"]:
                break
    synchronize(device)


if __name__ == "__main__":
    main()
