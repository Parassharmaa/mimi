#!/usr/bin/env python3
"""Token-local negative-space adaptation for one authenticated Marian checkpoint.

This is an NSL-MT-inspired safety arm, not a paper-faithful reproduction.  It
retains ordinary cross-entropy on the licensed correct reference and adds
severity-weighted unlikelihood only at the first token where a deterministic
corruption diverges.  Localizing the penalty prevents the unbounded shortcut of
making an arbitrary token in the entire rejected sequence improbable.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup

from train_marian_distillation import (
    Collator,
    TranslationRows,
    checkpoint_identity,
    checkpoint_lineage_manifests,
    evaluate,
    hardware_name,
    load_rows,
    sha256,
    synchronize,
)


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_negative_space(
    directory: Path, direction: str
) -> tuple[list[dict], list[dict], dict, Path]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("negative-space dataset lacks manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment")
        != "deterministic token-local negative-space Marian adaptation"
        or manifest.get("direction") != direction
        or manifest.get("promotion_eligible") is not False
        or manifest.get("private_reasoning_traces_used") is not False
        or manifest.get("free_form_synthetic_translations_used") is not False
        or manifest.get("human_review_required") is not False
    ):
        raise SystemExit("negative-space manifest safety contract differs")
    train_path, valid_path = directory / "train.jsonl", directory / "valid.jsonl"
    for name, path in (("train", train_path), ("valid", valid_path)):
        if manifest.get("outputs", {}).get(name, {}).get("sha256") != sha256(path):
            raise SystemExit(f"negative-space manifest does not authenticate {name}")

    expected = LANGUAGES[direction]

    def validate(rows: list[dict], split: str) -> list[dict]:
        identifiers: set[str] = set()
        for row in rows:
            identifier = str(row.get("id", ""))
            if not identifier or identifier in identifiers:
                raise SystemExit(f"{split} negative-space IDs are empty or duplicated")
            identifiers.add(identifier)
            if (row.get("source_language"), row.get("target_language")) != expected:
                raise SystemExit(f"{split} negative-space direction differs")
            if (
                not all(str(row.get(field, "")).strip() for field in (
                    "source", "chosen", "rejected", "violation_type", "source_license"
                ))
                or row["chosen"] == row["rejected"]
                or row.get("negative_generation")
                != "deterministic-target-corruption-used-only-as-negative-evidence"
                or not 0 < float(row.get("severity", 0)) <= 1
            ):
                raise SystemExit(f"{split} negative-space row is unsafe: {identifier}")
        if not rows:
            raise SystemExit(f"{split} negative-space rows are empty")
        return rows

    train_rows = validate(load_jsonl(train_path), "train")
    valid_rows = validate(load_jsonl(valid_path), "validation")
    if {row["parent_id"] for row in train_rows} & {row["parent_id"] for row in valid_rows}:
        raise SystemExit("negative-space parent IDs leak across train and validation")

    parent = manifest.get("parent", {})
    parent_directory = Path(str(parent.get("directory", "")))
    parent_manifest_path = parent_directory / "manifest.json"
    if not parent_manifest_path.is_file() or parent.get("manifest_sha256") != sha256(parent_manifest_path):
        raise SystemExit("negative-space parent manifest is missing or differs")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "valid"):
        parent_path = parent_directory / f"{split}.jsonl"
        expected_hash = parent.get(f"{split}_sha256")
        if expected_hash != sha256(parent_path) or expected_hash != parent_manifest.get("outputs", {}).get(split, {}).get("sha256"):
            raise SystemExit(f"negative-space parent {split} binding differs")
    return train_rows, valid_rows, manifest, parent_directory


class NegativeRows(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def truncate_labels(labels: torch.Tensor, pad_token_id: int, eos_token_id: int, maximum: int) -> torch.Tensor:
    if labels.shape[1] > maximum:
        labels = labels[:, :maximum].clone()
        labels[:, -1] = eos_token_id
    return labels


def first_divergence(chosen: list[int], rejected: list[int]) -> tuple[int, int, int]:
    for index, (chosen_token, rejected_token) in enumerate(zip(chosen, rejected)):
        if chosen_token != rejected_token:
            return index, chosen_token, rejected_token
    if len(chosen) != len(rejected):
        raise ValueError("target sequences differ only beyond their encoded EOS token")
    raise ValueError("chosen and rejected target token sequences are identical")


class NegativeSpaceCollator:
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        max_source_tokens: int,
        max_target_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        chosen = self.tokenizer(
            [row["source"] for row in rows],
            text_target=[row["chosen"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_source_tokens,
            return_tensors="pt",
        )
        rejected = self.tokenizer(
            text_target=[row["rejected"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_target_tokens,
            return_tensors="pt",
        )["input_ids"]
        labels = truncate_labels(
            chosen["labels"],
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.max_target_tokens,
        )
        rejected = truncate_labels(
            rejected,
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.max_target_tokens,
        )
        positions: list[int] = []
        chosen_tokens: list[int] = []
        rejected_tokens: list[int] = []
        for index, row in enumerate(rows):
            chosen_ids = labels[index][labels[index] != self.tokenizer.pad_token_id].tolist()
            rejected_ids = rejected[index][rejected[index] != self.tokenizer.pad_token_id].tolist()
            try:
                position, chosen_token, rejected_token = first_divergence(chosen_ids, rejected_ids)
            except ValueError as error:
                raise RuntimeError(f"negative pair has no usable divergence: {row['id']}: {error}") from error
            positions.append(position)
            chosen_tokens.append(chosen_token)
            rejected_tokens.append(rejected_token)
        labels[labels == self.tokenizer.pad_token_id] = -100
        chosen["labels"] = labels
        chosen["negative_positions"] = torch.tensor(positions, dtype=torch.long)
        chosen["chosen_token_ids"] = torch.tensor(chosen_tokens, dtype=torch.long)
        chosen["rejected_token_ids"] = torch.tensor(rejected_tokens, dtype=torch.long)
        chosen["severity"] = torch.tensor([float(row["severity"]) for row in rows], dtype=torch.float32)
        return chosen


def token_local_unlikelihood(
    logits: torch.Tensor,
    positions: torch.Tensor,
    rejected_token_ids: torch.Tensor,
    severity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(logits.shape[0], device=logits.device)
    probabilities = F.softmax(logits.float()[rows, positions], dim=-1)
    rejected_probabilities = probabilities.gather(1, rejected_token_ids[:, None]).squeeze(1)
    losses = -torch.log1p(-rejected_probabilities.clamp(max=1.0 - 1e-6))
    weighted = (losses * severity).sum() / severity.sum().clamp_min(1e-6)
    return weighted, rejected_probabilities


def divergence_metrics(
    logits: torch.Tensor,
    positions: torch.Tensor,
    chosen_token_ids: torch.Tensor,
    rejected_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(logits.shape[0], device=logits.device)
    selected = logits.float()[rows, positions]
    chosen = selected.gather(1, chosen_token_ids[:, None]).squeeze(1)
    rejected = selected.gather(1, rejected_token_ids[:, None]).squeeze(1)
    return chosen - rejected, chosen.gt(rejected)


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def split_metadata(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    metadata = {
        key: batch.pop(key)
        for key in ("negative_positions", "chosen_token_ids", "rejected_token_ids", "severity")
    }
    return batch, metadata


@torch.inference_mode()
def evaluate_negatives(
    model: MarianMTModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    losses: list[float] = []
    probabilities: list[float] = []
    margins: list[float] = []
    preferences: list[bool] = []
    for batch in loader:
        batch = move(batch, device)
        model_inputs, metadata = split_metadata(batch)
        logits = model(**model_inputs).logits
        loss, rejected_probability = token_local_unlikelihood(
            logits,
            metadata["negative_positions"],
            metadata["rejected_token_ids"],
            metadata["severity"],
        )
        margin, preferred = divergence_metrics(
            logits,
            metadata["negative_positions"],
            metadata["chosen_token_ids"],
            metadata["rejected_token_ids"],
        )
        losses.extend([float(loss)] * len(margin))
        probabilities.extend(rejected_probability.cpu().tolist())
        margins.extend(margin.cpu().tolist())
        preferences.extend(preferred.cpu().tolist())
    synchronize(device)
    model.train()
    return {
        "pairs": len(margins),
        "token_local_unlikelihood": sum(losses) / len(losses),
        "mean_rejected_token_probability": sum(probabilities) / len(probabilities),
        "mean_chosen_minus_rejected_logit": sum(margins) / len(margins),
        "chosen_token_preference_accuracy": sum(preferences) / len(preferences),
    }


def save_candidate(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    output: Path,
    manifest: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # Keep the specialized filename for easy experiment discovery and the
    # canonical filename so generic checkpoint identity/conversion tooling
    # carries the exact dataset lineage into any later q4 control.
    for name in (
        "mimi_negative_space_training_manifest.json",
        "mimi_training_manifest.json",
    ):
        (output / name).write_text(payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("negative_directory", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=161803)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=125)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--evaluation-steps", type=int, default=125)
    parser.add_argument("--negative-weight", type=float, default=0.3)
    parser.add_argument("--max-source-tokens", type=int, default=128)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if not (args.initial_checkpoint / "model.safetensors").is_file():
        raise SystemExit("initial checkpoint lacks model.safetensors")
    if min(args.batch_size, args.gradient_accumulation, args.max_steps, args.evaluation_steps) < 1:
        raise SystemExit("batch, accumulation, steps, and evaluation interval must be positive")
    if args.learning_rate <= 0 or args.negative_weight <= 0:
        raise SystemExit("learning-rate and negative-weight must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    expected_identity = (args.direction, args.repository, args.revision)
    actual_identity = checkpoint_identity(args.initial_checkpoint)
    if actual_identity != expected_identity:
        raise SystemExit(f"initial checkpoint identity differs: expected {expected_identity}, found {actual_identity}")

    train_rows, negative_valid_rows, negative_manifest, parent_directory = load_negative_space(
        args.negative_directory, args.direction
    )
    parent_manifest_path = parent_directory / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_valid_rows = load_rows(parent_directory / "valid.jsonl", args.direction)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = MarianTokenizer.from_pretrained(args.initial_checkpoint)
    model = MarianMTModel.from_pretrained(args.initial_checkpoint).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    negative_collator = NegativeSpaceCollator(
        tokenizer, args.max_source_tokens, args.max_target_tokens
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        NegativeRows(train_rows), batch_size=args.batch_size, shuffle=True,
        generator=generator, collate_fn=negative_collator,
    )
    negative_valid_loader = DataLoader(
        NegativeRows(negative_valid_rows), batch_size=args.batch_size, shuffle=False,
        collate_fn=negative_collator,
    )
    positive_collator = Collator(
        tokenizer, args.max_source_tokens, args.max_target_tokens, set()
    )
    positive_valid_loader = DataLoader(
        TranslationRows(parent_valid_rows), batch_size=args.batch_size, shuffle=False,
        collate_fn=positive_collator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, args.max_steps),
        num_training_steps=args.max_steps,
    )
    optimizer.zero_grad(set_to_none=True)

    initial_translation = evaluate(
        model, tokenizer, positive_valid_loader, parent_valid_rows, device, args.max_target_tokens
    )
    initial_negatives = evaluate_negatives(model, negative_valid_loader, device)
    history = [{"step": 0, "translation": initial_translation, "negative_space": initial_negatives}]
    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "operation": "NSL-MT-inspired token-local negative-space adaptation",
        "paper_faithful_reproduction": False,
        "direction": args.direction,
        "student_repository": args.repository,
        "student_revision": args.revision,
        "license": "CC-BY-SA-4.0",
        "initial_checkpoint": {
            "path": str(args.initial_checkpoint),
            "model_sha256": sha256(args.initial_checkpoint / "model.safetensors"),
            "lineage_manifests": checkpoint_lineage_manifests(args.initial_checkpoint),
        },
        "dataset": {
            "directory": str(args.negative_directory),
            "manifest_sha256": sha256(args.negative_directory / "manifest.json"),
            "train_sha256": negative_manifest["outputs"]["train"]["sha256"],
            "valid_sha256": negative_manifest["outputs"]["valid"]["sha256"],
            "train_pairs": len(train_rows),
            "valid_pairs": len(negative_valid_rows),
            "parent_directory": str(parent_directory),
            "parent_valid_sha256": negative_manifest["parent"]["valid_sha256"],
            "parent_valid_rows": len(parent_valid_rows),
        },
        "dataset_manifest": {
            "path": str(parent_manifest_path.resolve()),
            "sha256": sha256(parent_manifest_path),
            "schema_version": parent_manifest.get("schema_version"),
            "direction": parent_manifest.get("direction"),
            "experiment": negative_manifest.get("experiment"),
            "target_source": "licensed-human-reference-with-deterministic-negative-only-corruptions",
            "effective_licenses": parent_manifest.get("effective_licenses"),
            "promotion_eligible": False,
            "authenticated_outputs": ["train", "valid"],
            "outputs_authenticated": True,
            "negative_dataset_manifest": {
                "path": str((args.negative_directory / "manifest.json").resolve()),
                "sha256": sha256(args.negative_directory / "manifest.json"),
            },
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "hyperparameters": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "evaluation_steps": args.evaluation_steps,
            "negative_weight": args.negative_weight,
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "objective": {
            "positive": "ordinary token cross-entropy on authenticated licensed human/project-owned references",
            "negative": "severity-weighted -log(1-p(rejected_token)) at first divergent target token under the correct prefix",
            "motivation": "localized bounded unlikelihood avoids the whole-sequence probability shortcut in the reviewed paper objective",
            "negative_strings_are_positive_targets": False,
            "free_form_synthetic_translations_used": False,
            "private_reasoning_traces_used": False,
            "human_review_required": False,
        },
    }

    update_step = 0
    micro_step = 0
    epoch = 0
    model.train()
    while update_step < args.max_steps:
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
            combined = positive_loss + args.negative_weight * negative_loss
            (combined / args.gradient_accumulation).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            if update_step % args.evaluation_steps == 0 or update_step == args.max_steps:
                translation = evaluate(
                    model, tokenizer, positive_valid_loader, parent_valid_rows, device, args.max_target_tokens
                )
                negatives = evaluate_negatives(model, negative_valid_loader, device)
                record = {
                    "step": update_step,
                    "translation": translation,
                    "negative_space": negatives,
                    "last_train_objective": {
                        "positive_cross_entropy": float(positive_loss.detach().cpu()),
                        "token_local_unlikelihood": float(negative_loss.detach().cpu()),
                        "mean_rejected_token_probability": float(rejected_probability.mean().detach().cpu()),
                        "combined": float(combined.detach().cpu()),
                    },
                }
                history.append(record)
                print(json.dumps(record, ensure_ascii=False))
            if update_step >= args.max_steps:
                break
        epoch += 1

    final = history[-1]
    manifest = {
        **common,
        "history": history,
        "final": final,
        "general_validation_delta_chrf_pp": (
            final["translation"]["chrf_pp"] - initial_translation["chrf_pp"]
        ),
        "negative_validation_delta_rejected_probability": (
            final["negative_space"]["mean_rejected_token_probability"]
            - initial_negatives["mean_rejected_token_probability"]
        ),
        "promotion_eligible": False,
    }
    save_candidate(model, tokenizer, args.output_directory, manifest)


if __name__ == "__main__":
    main()
