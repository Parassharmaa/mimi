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
VALIDATION_CACHE_POLICY = {
    "loss_forward": False,
    "greedy_generation": True,
}
STUDENT_TEACHER_COMPATIBILITY_KEYS = (
    "activation_function",
    "d_model",
    "decoder_attention_heads",
    "decoder_layers",
    "encoder_attention_heads",
    "encoder_layers",
    "max_position_embeddings",
    "model_type",
    "normalize_before",
    "normalize_embedding",
    "scale_embedding",
    "share_encoder_decoder_embeddings",
    "static_position_embeddings",
    "vocab_size",
)


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


def load_model_configuration(path: Path) -> dict:
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


def validate_model_compatibility(
    en_ja_teacher: Path,
    ja_en_teacher: Path,
    student: Path,
) -> dict:
    paths = [en_ja_teacher, ja_en_teacher, student]
    for path in paths:
        if not (path / "model.safetensors").is_file():
            raise SystemExit(f"model checkpoint is incomplete: {path}")

    teacher_configurations = [
        load_model_configuration(en_ja_teacher),
        load_model_configuration(ja_en_teacher),
    ]
    normalized_teachers = []
    for configuration in teacher_configurations:
        normalized = dict(configuration)
        normalized.pop("_name_or_path", None)
        normalized_teachers.append(normalized)
    if normalized_teachers[0] != normalized_teachers[1]:
        raise SystemExit("directional teacher architectures differ")

    teacher_configuration = teacher_configurations[0]
    student_configuration = load_model_configuration(student)
    mismatches = {
        key: {
            "teacher": teacher_configuration.get(key),
            "student": student_configuration.get(key),
        }
        for key in STUDENT_TEACHER_COMPATIBILITY_KEYS
        if teacher_configuration.get(key) != student_configuration.get(key)
    }
    if mismatches:
        raise SystemExit(
            "student/teacher architecture is incompatible: "
            + json.dumps(mismatches, sort_keys=True)
        )
    dimensions = {}
    for key in ("encoder_ffn_dim", "decoder_ffn_dim"):
        teacher_dimension = int(teacher_configuration.get(key, -1))
        student_dimension = int(student_configuration.get(key, -1))
        if teacher_dimension <= 0 or student_dimension < teacher_dimension:
            raise SystemExit(
                f"student {key} must be at least the teacher width "
                f"{teacher_dimension}: {student_dimension}"
            )
        dimensions[key] = {
            "teacher": teacher_dimension,
            "student": student_dimension,
        }
    for name in TOKENIZER_ASSETS:
        digests = {sha256(path / name) for path in paths}
        if len(digests) != 1:
            raise SystemExit(f"model tokenizer asset differs: {name}")
    return {
        "shared_architecture": {
            key: teacher_configuration.get(key)
            for key in STUDENT_TEACHER_COMPATIBILITY_KEYS
        },
        "ffn_dimensions": dimensions,
        "tokenizer_assets": {
            name: sha256(en_ja_teacher / name) for name in TOKENIZER_ASSETS
        },
    }


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


def masked_encoder_mse(
    student_states: torch.Tensor,
    teacher_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if student_states.shape != teacher_states.shape:
        raise ValueError(
            f"student/teacher encoder states differ: "
            f"{tuple(student_states.shape)} != {tuple(teacher_states.shape)}"
        )
    token_mask = attention_mask.to(dtype=student_states.dtype).unsqueeze(-1)
    squared = (student_states.float() - teacher_states.float()).square()
    return (squared * token_mask).sum(), token_mask.sum() * student_states.shape[-1]


def directional_teacher_losses(
    student_logits: torch.Tensor,
    teacher_batch: dict[str, torch.Tensor],
    direction_ids: torch.Tensor,
    teachers: dict[int, MarianMTModel],
    temperature: float,
    student_encoder_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kl_total = student_logits.new_zeros((), dtype=torch.float32)
    kl_tokens = student_logits.new_zeros((), dtype=torch.long)
    encoder_total = student_logits.new_zeros((), dtype=torch.float32)
    encoder_values = student_logits.new_zeros((), dtype=torch.long)
    for direction_id, teacher in teachers.items():
        indices = torch.nonzero(direction_ids == direction_id, as_tuple=False).flatten()
        if not len(indices):
            continue
        subset = {
            key: value.index_select(0, indices)
            for key, value in teacher_batch.items()
        }
        with torch.inference_mode():
            teacher_outputs = teacher(**subset, return_dict=True)
        subtotal, count = teacher_student_kl(
            student_logits.index_select(0, indices),
            teacher_outputs.logits,
            subset["labels"],
            temperature,
        )
        kl_total = kl_total + subtotal
        kl_tokens = kl_tokens + count
        if student_encoder_states is not None:
            encoder_subtotal, value_count = masked_encoder_mse(
                student_encoder_states.index_select(0, indices),
                teacher_outputs.encoder_last_hidden_state,
                subset["attention_mask"],
            )
            encoder_total = encoder_total + encoder_subtotal
            encoder_values = encoder_values + value_count
    return (
        kl_total / kl_tokens.clamp_min(1),
        encoder_total / encoder_values.clamp_min(1),
    )


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
        outputs = model(
            **batch,
            use_cache=VALIDATION_CACHE_POLICY["loss_forward"],
        )
        batch_size = int(batch["input_ids"].shape[0])
        losses.append((float(outputs.loss), batch_size))
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=VALIDATION_CACHE_POLICY["greedy_generation"],
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
    parser.add_argument("--encoder-alignment-weight", type=float, default=0.0)
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
    if (
        args.teacher_kl_weight < 0
        or args.encoder_alignment_weight < 0
        or args.validation_limit_per_direction < 0
    ):
        raise SystemExit(
            "KL, encoder-alignment weight, and validation limit must be non-negative"
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    compatibility = validate_model_compatibility(
        args.en_ja_teacher,
        args.ja_en_teacher,
        args.initial_checkpoint,
    )
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
            "encoder_alignment_weight": args.encoder_alignment_weight,
            "teacher_float16": args.teacher_float16,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "selection": (
            "maximum unweighted macro-average direction chrF++ on deterministic licensed "
            "development subsets; tie-break minimum aggregate development loss"
        ),
        "student_teacher_compatibility": compatibility,
        "encoder_alignment": (
            "projection-free token-state MSE on a separate unprefixed student encoder "
            "pass with the exact teacher input IDs and attention mask"
            if args.encoder_alignment_weight
            else "disabled"
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
            student_encoder_states = None
            if args.encoder_alignment_weight:
                student_encoder_states = student.model.encoder(
                    input_ids=teacher_batch["input_ids"],
                    attention_mask=teacher_batch["attention_mask"],
                    return_dict=True,
                ).last_hidden_state
            kl_loss, encoder_alignment_loss = directional_teacher_losses(
                outputs.logits,
                teacher_batch,
                direction_ids,
                teachers,
                args.teacher_temperature,
                student_encoder_states,
            )
            combined = (
                ce_loss
                + args.teacher_kl_weight * kl_loss
                + args.encoder_alignment_weight * encoder_alignment_loss
            )
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
                        "encoder_alignment": float(
                            encoder_alignment_loss.detach().cpu()
                        ),
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
