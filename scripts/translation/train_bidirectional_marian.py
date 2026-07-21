#!/usr/bin/env python3
"""Distill two directional Marian teachers into one shared bidirectional student."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sacrebleu
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup


DIRECTIONS = {"en-ja": 0, "ja-en": 1}
SOURCE_PREFIXES = {"en-ja": "<2ja> ", "ja-en": "<2en> "}
TOKENIZER_ASSETS = ("source.spm", "target.spm", "vocab.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def load_rows(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"dataset is empty: {path}")
    identifiers: set[str] = set()
    for row in rows:
        if row.get("direction") not in DIRECTIONS:
            raise SystemExit(f"row lacks a valid direction: {row.get('id')}")
        identifier = str(row.get("id", ""))
        if not identifier or identifier in identifiers:
            raise SystemExit(f"missing or duplicate row ID: {identifier}")
        identifiers.add(identifier)
    return rows


def validate_model_compatibility(paths: list[Path]) -> None:
    for path in paths:
        if not (path / "model.safetensors").is_file():
            raise SystemExit(f"model checkpoint is incomplete: {path}")
    reference_config = json.loads((paths[0] / "config.json").read_text(encoding="utf-8"))
    reference_config.pop("_name_or_path", None)
    for path in paths[1:]:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        config.pop("_name_or_path", None)
        if config != reference_config:
            raise SystemExit(f"model architecture differs: {path}")
    for name in TOKENIZER_ASSETS:
        digests = {sha256(path / name) for path in paths}
        if len(digests) != 1:
            raise SystemExit(f"model tokenizer asset differs: {name}")


class Rows(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class Collator:
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
        batch = self.tokenizer(
            [SOURCE_PREFIXES[row["direction"]] + row["source"] for row in rows],
            text_target=[row["target"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_source_tokens,
            return_tensors="pt",
        )
        labels = batch["labels"]
        if labels.shape[1] > self.max_target_tokens:
            labels = labels[:, : self.max_target_tokens]
            labels[:, -1] = self.tokenizer.eos_token_id
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        teacher_inputs = self.tokenizer(
            [row["source"] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_source_tokens,
            return_tensors="pt",
        )
        batch["teacher_input_ids"] = teacher_inputs["input_ids"]
        batch["teacher_attention_mask"] = teacher_inputs["attention_mask"]
        batch["direction_ids"] = torch.tensor(
            [DIRECTIONS[row["direction"]] for row in rows], dtype=torch.long
        )
        return batch


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def teacher_student_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_mask = labels.ne(-100)
    divergences = F.kl_div(
        F.log_softmax(student_logits.float() / temperature, dim=-1),
        F.softmax(teacher_logits.float() / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * temperature**2
    return (divergences * token_mask).sum(), token_mask.sum()


def directional_teacher_kl(
    student_logits: torch.Tensor,
    teacher_batch: dict[str, torch.Tensor],
    direction_ids: torch.Tensor,
    teachers: dict[int, MarianMTModel],
    temperature: float,
) -> torch.Tensor:
    total = student_logits.new_zeros((), dtype=torch.float32)
    tokens = student_logits.new_zeros((), dtype=torch.long)
    for direction_id, teacher in teachers.items():
        indices = torch.nonzero(direction_ids == direction_id, as_tuple=False).flatten()
        if not len(indices):
            continue
        subset = {
            key: value.index_select(0, indices)
            for key, value in teacher_batch.items()
        }
        with torch.inference_mode():
            teacher_logits = teacher(**subset).logits
        subtotal, count = teacher_student_kl(
            student_logits.index_select(0, indices),
            teacher_logits,
            subset["labels"],
            temperature,
        )
        total = total + subtotal
        tokens = tokens + count
    return total / tokens.clamp_min(1)


def validation_subset(rows: list[dict], limit_per_direction: int, seed: int) -> list[dict]:
    if limit_per_direction <= 0:
        return rows
    output: list[dict] = []
    for direction_index, direction in enumerate(DIRECTIONS):
        current = [row for row in rows if row["direction"] == direction]
        random.Random(seed + direction_index).shuffle(current)
        output.extend(current[:limit_per_direction])
    return sorted(output, key=lambda row: (row["direction"], row["id"]))


@torch.inference_mode()
def evaluate(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    loader: DataLoader,
    rows: list[dict],
    device: torch.device,
    max_new_tokens: int,
) -> dict:
    model.eval()
    hypotheses: list[str] = []
    losses: list[tuple[float, int]] = []
    for batch in loader:
        batch = move(batch, device)
        batch.pop("direction_ids")
        batch.pop("teacher_input_ids")
        batch.pop("teacher_attention_mask")
        outputs = model(**batch)
        batch_size = int(batch["input_ids"].shape[0])
        losses.append((float(outputs.loss), batch_size))
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
        hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    by_direction: dict[str, dict] = {}
    for direction in DIRECTIONS:
        indices = [index for index, row in enumerate(rows) if row["direction"] == direction]
        by_direction[direction] = {
            "cases": len(indices),
            "chrf_pp": sacrebleu.corpus_chrf(
                [hypotheses[index] for index in indices],
                [[rows[index]["target"] for index in indices]],
                word_order=2,
            ).score,
        }
    macro = sum(value["chrf_pp"] for value in by_direction.values()) / len(by_direction)
    model.train()
    return {
        "loss": sum(value * count for value, count in losses) / sum(
            count for _, count in losses
        ),
        "macro_direction_chrf_pp": macro,
        "directions": by_direction,
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
    (output / "mimi_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("en_ja_teacher", type=Path)
    parser.add_argument("ja_en_teacher", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--evaluation-steps", type=int, default=100)
    parser.add_argument("--validation-limit-per-direction", type=int, default=128)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--teacher-kl-weight", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-float16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if min(
        args.batch_size,
        args.gradient_accumulation,
        args.max_steps,
        args.evaluation_steps,
        args.teacher_temperature,
    ) <= 0:
        raise SystemExit("batch, steps, intervals, and temperature must be positive")
    if args.teacher_kl_weight < 0 or args.validation_limit_per_direction < 0:
        raise SystemExit("KL weight and validation limit must be non-negative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_paths = [args.en_ja_teacher, args.ja_en_teacher, args.initial_checkpoint]
    validate_model_compatibility(model_paths)
    tokenizer = MarianTokenizer.from_pretrained(args.en_ja_teacher)

    train_path = args.dataset_directory / "train.jsonl"
    valid_path = args.dataset_directory / "valid.jsonl"
    dataset_manifest_path = args.dataset_directory / "manifest.json"
    train_rows = load_rows(train_path)
    all_valid_rows = load_rows(valid_path)
    valid_rows = validation_subset(
        all_valid_rows, args.validation_limit_per_direction, args.seed
    )
    if any(
        sum(row["direction"] == direction for row in train_rows)
        != sum(row["direction"] == next(iter(DIRECTIONS)) for row in train_rows)
        for direction in DIRECTIONS
    ):
        raise SystemExit("training mixture is not direction-balanced")

    student = MarianMTModel.from_pretrained(args.initial_checkpoint).to(device)
    teachers: dict[int, MarianMTModel] = {}
    for direction, path in (
        ("en-ja", args.en_ja_teacher),
        ("ja-en", args.ja_en_teacher),
    ):
        teacher = MarianMTModel.from_pretrained(path)
        if args.teacher_float16:
            teacher = teacher.to(dtype=torch.float16)
        teacher = teacher.to(device)
        teacher.eval()
        teacher.requires_grad_(False)
        teachers[DIRECTIONS[direction]] = teacher
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        student.config.use_cache = False

    collator = Collator(tokenizer, args.max_source_tokens, args.max_target_tokens)
    train_loader = DataLoader(
        Rows(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        Rows(valid_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, args.max_steps),
        num_training_steps=args.max_steps,
    )
    optimizer.zero_grad(set_to_none=True)

    common_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "direction": "bidirectional",
        "license": "CC-BY-SA-4.0",
        "training_description": (
            "balanced licensed targets plus direction-specific token-level KL from two "
            "frozen specialist teachers and explicit target-language source markers; "
            "no chain-of-thought"
        ),
        "source_prefixes": SOURCE_PREFIXES,
        "teachers": {
            "en-ja": {
                "path": str(args.en_ja_teacher),
                "model_sha256": sha256(args.en_ja_teacher / "model.safetensors"),
            },
            "ja-en": {
                "path": str(args.ja_en_teacher),
                "model_sha256": sha256(args.ja_en_teacher / "model.safetensors"),
            },
        },
        "initial_checkpoint": {
            "path": str(args.initial_checkpoint),
            "model_sha256": sha256(args.initial_checkpoint / "model.safetensors"),
        },
        "dataset": {
            "manifest_path": str(dataset_manifest_path),
            "manifest_sha256": sha256(dataset_manifest_path),
            "train_path": str(train_path),
            "train_sha256": sha256(train_path),
            "train_rows": len(train_rows),
            "valid_path": str(valid_path),
            "valid_sha256": sha256(valid_path),
            "valid_rows": len(all_valid_rows),
            "selection_valid_rows": len(valid_rows),
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
            "validation_limit_per_direction": args.validation_limit_per_direction,
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "teacher_kl_weight": args.teacher_kl_weight,
            "teacher_temperature": args.teacher_temperature,
            "teacher_float16": args.teacher_float16,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "selection": (
            "maximum unweighted macro-average direction chrF++ on deterministic licensed "
            "development subsets; tie-break minimum aggregate development loss"
        ),
        "private_chain_of_thought_stored": False,
    }

    base_metrics = evaluate(
        student,
        tokenizer,
        valid_loader,
        valid_rows,
        device,
        args.max_target_tokens,
    )
    history = [{"step": 0, **base_metrics}]
    best = history[0]
    save_candidate(
        student,
        tokenizer,
        args.output_directory,
        {**common_manifest, "best": best, "history": history},
    )

    update_step = 0
    micro_step = 0
    epoch = 0
    student.train()
    while update_step < args.max_steps:
        for batch in train_loader:
            batch = move(batch, device)
            direction_ids = batch.pop("direction_ids")
            teacher_batch = {
                "input_ids": batch.pop("teacher_input_ids"),
                "attention_mask": batch.pop("teacher_attention_mask"),
                "labels": batch["labels"],
            }
            outputs = student(**batch)
            ce_loss = outputs.loss
            kl_loss = directional_teacher_kl(
                outputs.logits,
                teacher_batch,
                direction_ids,
                teachers,
                args.teacher_temperature,
            )
            combined = ce_loss + args.teacher_kl_weight * kl_loss
            (combined / args.gradient_accumulation).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            if update_step % args.evaluation_steps == 0 or update_step == args.max_steps:
                metrics = evaluate(
                    student,
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
                        "cross_entropy": float(ce_loss.detach().cpu()),
                        "teacher_kl": float(kl_loss.detach().cpu()),
                    },
                }
                history.append(record)
                if (
                    metrics["macro_direction_chrf_pp"],
                    -metrics["loss"],
                ) > (best["macro_direction_chrf_pp"], -best["loss"]):
                    best = record
                    save_candidate(
                        student,
                        tokenizer,
                        args.output_directory,
                        {**common_manifest, "best": best, "history": history},
                    )
                print(json.dumps({"current": record, "best": best}, ensure_ascii=False))
            if update_step >= args.max_steps:
                break
        epoch += 1

    final_manifest = {**common_manifest, "best": best, "history": history}
    manifest_path = args.output_directory / "mimi_training_manifest.json"
    manifest_path.write_text(
        json.dumps(final_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.output_directory), "best": best},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
