#!/usr/bin/env python3
"""Run a bounded, provenance-bound bilingual LoRA pilot on SMaLL-100."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import M2M100ForConditionalGeneration, get_linear_schedule_with_warmup

from run_small100_benchmark import (
    DEFAULT_REPOSITORY,
    DEFAULT_REVISION,
    MODEL_FILES,
    load_tokenizer,
    sha256,
)


DIRECTIONS = {"en-ja": "ja", "ja-en": "en"}


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    required = {
        "id",
        "direction",
        "source",
        "target",
        "source_license",
        "source_provenance",
    }
    if not rows:
        raise SystemExit("SMaLL-100 training data is empty")
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise SystemExit("SMaLL-100 training row lacks provenance or parallel text")
        if row["direction"] not in DIRECTIONS:
            raise SystemExit(f"unsupported training direction: {row['direction']}")
        if not row["source"].strip() or not row["target"].strip():
            raise SystemExit(f"empty training text: {row['id']}")
        if row["id"] in identifiers and "balance_repeat_index" not in row:
            raise SystemExit(f"undeclared duplicate training ID: {row['id']}")
        identifiers.add(row["id"])
    return rows


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def checkpoint_manifest(
    checkpoint: Path,
    *,
    args: argparse.Namespace,
    snapshot: Path,
    train_path: Path,
    dataset_manifest_path: Path,
    rows: list[dict],
    step: int,
    trainable_parameters: int,
    total_parameters: int,
    last_loss: float,
) -> None:
    adapter = checkpoint / "adapter_model.safetensors"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "operation": "bounded-balanced-en-ja-small100-lora-pilot",
        "promotion_eligible": False,
        "does_not_authorize_app_integration": True,
        "base_model": {
            "repository": args.repository,
            "revision": args.revision,
            "model_sha256": sha256(snapshot / "model.safetensors"),
            "license": "MIT",
        },
        "dataset": {
            "manifest_path": str(dataset_manifest_path),
            "manifest_sha256": sha256(dataset_manifest_path),
            "train_path": str(train_path),
            "train_sha256": sha256(train_path),
            "rows": len(rows),
            "directions": dict(Counter(row["direction"] for row in rows)),
            "licenses": dict(Counter(row["source_license"] for row in rows)),
            "all_rows_retain_provenance": True,
        },
        "training": {
            "step": step,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": ["q_proj", "v_proj"],
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "seed": args.seed,
            "last_loss": last_loss,
        },
        "parameters": {
            "trainable": trainable_parameters,
            "total": total_parameters,
        },
        "adapter": {
            "bytes": adapter.stat().st_size,
            "sha256": sha256(adapter),
        },
        "private_chain_of_thought_stored": False,
    }
    (checkpoint / "mimi_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--hf-home", type=Path, default=Path("Research/translation/models/hf-cache")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    for label in (
        "max_steps",
        "save_every",
        "report_every",
        "gradient_accumulation_steps",
        "rank",
        "alpha",
        "max_source_tokens",
        "max_target_tokens",
    ):
        if getattr(args, label) < 1:
            raise SystemExit(f"{label.replace('_', '-')} must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")

    train_path = args.dataset / "train.jsonl"
    dataset_manifest_path = args.dataset / "manifest.json"
    rows = load_rows(train_path)
    if not dataset_manifest_path.is_file():
        raise SystemExit("dataset manifest is missing")
    counts = Counter(row["direction"] for row in rows)
    if counts["en-ja"] != counts["ja-en"]:
        raise SystemExit(f"training directions are not balanced: {dict(counts)}")

    local_model = Path(args.repository)
    snapshot = (
        local_model.resolve()
        if local_model.is_dir()
        else Path(
            snapshot_download(
                repo_id=args.repository,
                revision=args.revision,
                cache_dir=args.hf_home,
                allow_patterns=MODEL_FILES,
            )
        )
    )
    tokenizer = load_tokenizer(snapshot)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    base = M2M100ForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=dtype,
        use_safetensors=True,
        attn_implementation="eager",
    )
    configuration = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, configuration).to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "rows": len(rows),
                "trainableParameters": trainable_parameters,
                "totalParameters": total_parameters,
            }
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, args.max_steps),
        num_training_steps=args.max_steps,
    )
    optimizer.zero_grad(set_to_none=True)
    randomizer = random.Random(args.seed)
    order = list(range(len(rows)))
    randomizer.shuffle(order)
    cursor = 0
    args.output.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []

    for step in range(1, args.max_steps + 1):
        for _ in range(args.gradient_accumulation_steps):
            if cursor == len(order):
                randomizer.shuffle(order)
                cursor = 0
            row = rows[order[cursor]]
            cursor += 1
            tokenizer.tgt_lang = DIRECTIONS[row["direction"]]
            batch = tokenizer(
                row["source"],
                text_target=row["target"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_source_tokens,
            )
            if batch["labels"].shape[-1] > args.max_target_tokens:
                batch["labels"] = batch["labels"][:, : args.max_target_tokens]
            batch = batch.to(device)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite SMaLL-100 LoRA loss at step {step}")
            losses.append(float(loss.detach().cpu()))
            (loss / args.gradient_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % args.report_every == 0 or step == 1:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": sum(losses[-args.gradient_accumulation_steps :])
                        / args.gradient_accumulation_steps,
                        "learningRate": scheduler.get_last_lr()[0],
                    }
                ),
                flush=True,
            )
        if step % args.save_every == 0 or step == args.max_steps:
            checkpoint = args.output / f"checkpoint-{step:06d}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            checkpoint_manifest(
                checkpoint,
                args=args,
                snapshot=snapshot,
                train_path=train_path,
                dataset_manifest_path=dataset_manifest_path,
                rows=rows,
                step=step,
                trainable_parameters=trainable_parameters,
                total_parameters=total_parameters,
                last_loss=losses[-1],
            )
            print(
                json.dumps(
                    {"checkpoint": str(checkpoint), "bytes": directory_bytes(checkpoint)}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
