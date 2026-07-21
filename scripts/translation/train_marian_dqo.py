#!/usr/bin/env python3
"""Post-SFT Marian Direct Quality Optimization on human preference pairs.

This command is fail-closed: it refuses to start unless a hash-bound
`supervised-win-approved` report proves that the exact starting checkpoint won
on reviewed development data, retained general translation quality, and had no
new critical errors.
"""

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
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup


LANGUAGES = {
    "en-ja": ("en-US", "ja-JP"),
    "ja-en": ("ja-JP", "en-US"),
}
REQUIRED_WIN_GATES = {
    "reviewed-development-chrf-win",
    "blind-human-development-win",
    "no-new-critical-errors",
    "general-retention",
    "exact-bundle-checkpoint-binding",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_supervised_win(path: Path, checkpoint: Path, direction: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schemaVersion") != 1
        or report.get("status") != "supervised-win-approved"
        or report.get("approved") is not True
        or report.get("direction") != direction
    ):
        raise SystemExit("DQO requires an approved supervised-win report for this direction")
    gates = report.get("gates")
    if not isinstance(gates, list):
        raise SystemExit("supervised-win report has no gates")
    gate_map = {str(value.get("name")): value for value in gates if isinstance(value, dict)}
    if not REQUIRED_WIN_GATES <= set(gate_map) or any(
        gate_map[name].get("passed") is not True for name in REQUIRED_WIN_GATES
    ):
        raise SystemExit("supervised-win report does not pass every required DQO prerequisite")
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file():
        raise SystemExit(f"supervised checkpoint lacks model.safetensors: {checkpoint}")
    checkpoint_record = report.get("supervisedCheckpoint", {})
    if checkpoint_record.get("modelSHA256") != sha256(model_path):
        raise SystemExit("supervised-win report is not bound to the starting checkpoint")
    return report


def load_preferences(directory: Path, direction: str) -> tuple[list[dict], list[dict], dict]:
    train_path, valid_path = directory / "train.jsonl", directory / "valid.jsonl"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("preference dataset lacks manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("purpose") != "post-supervised-win human-preference DQO only"
        or manifest.get("direction") != direction
    ):
        raise SystemExit("preference manifest has invalid purpose or direction")
    if (
        manifest.get("train", {}).get("sha256") != sha256(train_path)
        or manifest.get("valid", {}).get("sha256") != sha256(valid_path)
    ):
        raise SystemExit("preference data hashes disagree with manifest")
    expected = LANGUAGES[direction]

    def validate(rows: list[dict], name: str) -> list[dict]:
        identifiers: set[str] = set()
        for row in rows:
            identifier = str(row.get("id", "")).strip()
            if not identifier or identifier in identifiers:
                raise SystemExit(f"{name} preferences contain empty or duplicate ID: {identifier}")
            identifiers.add(identifier)
            if (row.get("source_language"), row.get("target_language")) != expected:
                raise SystemExit(f"{name} preferences contain the wrong direction: {identifier}")
            if (
                row.get("origin") != "two-reviewer-human-preference"
                or row.get("review_status") != "two-reviewer-selected-over-unapproved-candidate"
                or len(set(row.get("reviewer_ids", []))) != 2
            ):
                raise SystemExit(f"{name} preference lacks two-reviewer evidence: {identifier}")
            texts = [str(row.get(field, "")).strip() for field in ("source", "chosen", "rejected")]
            if not all(texts) or texts[1] == texts[2]:
                raise SystemExit(f"{name} preference has empty or identical choices: {identifier}")
        if not rows:
            raise SystemExit(f"{name} preferences are empty")
        return rows

    train = validate(load_jsonl(train_path), "train")
    valid = validate(load_jsonl(valid_path), "validation")
    if {row["source_id"] for row in train} & {row["source_id"] for row in valid}:
        raise SystemExit("preference sources leak across train and validation")
    return train, valid, manifest


class PreferenceRows(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class PreferenceCollator:
    def __init__(self, tokenizer: MarianTokenizer, max_source_tokens: int, max_target_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens

    def tokenize(self, rows: list[dict], target_field: str) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            [row["source"] for row in rows],
            text_target=[row[target_field] for row in rows],
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
        return batch

    def __call__(self, rows: list[dict]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "chosen": self.tokenize(rows, "chosen"),
            "rejected": self.tokenize(rows, "rejected"),
        }


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sequence_log_probabilities(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    # Length normalization avoids teaching the preference objective to prefer
    # shorter captions merely because summed log probabilities are less negative.
    return (token_logps * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)


def dqo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    relative_margin = (policy_chosen - policy_rejected) - (
        reference_chosen - reference_rejected
    )
    return -F.logsigmoid(beta * relative_margin).mean(), relative_margin


def pair_logps(
    model: MarianMTModel,
    chosen: dict[str, torch.Tensor],
    rejected: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen_output = model(**chosen)
    rejected_output = model(**rejected)
    return (
        sequence_log_probabilities(chosen_output.logits, chosen["labels"]),
        sequence_log_probabilities(rejected_output.logits, rejected["labels"]),
    )


@torch.inference_mode()
def evaluate(
    model: MarianMTModel,
    reference: MarianMTModel,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict:
    model.eval()
    losses: list[float] = []
    margins: list[float] = []
    policy_preferences: list[float] = []
    for pair in loader:
        chosen, rejected = move(pair["chosen"], device), move(pair["rejected"], device)
        policy_chosen, policy_rejected = pair_logps(model, chosen, rejected)
        reference_chosen, reference_rejected = pair_logps(reference, chosen, rejected)
        loss, relative = dqo_loss(
            policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
        )
        losses.extend([float(loss)] * len(relative))
        margins.extend(relative.cpu().tolist())
        policy_preferences.extend((policy_chosen - policy_rejected).cpu().tolist())
    model.train()
    return {
        "loss": sum(losses) / len(losses),
        "relative_margin": sum(margins) / len(margins),
        "relative_pair_accuracy": sum(value > 0 for value in margins) / len(margins),
        "policy_pair_accuracy": sum(value > 0 for value in policy_preferences) / len(policy_preferences),
        "pairs": len(margins),
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
    (output / "mimi_dqo_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preference_directory", type=Path)
    parser.add_argument("supervised_checkpoint", type=Path)
    parser.add_argument("supervised_win_report", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=tuple(LANGUAGES), required=True)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--evaluation-steps", type=int, default=25)
    parser.add_argument("--beta", type=float, default=0.10)
    parser.add_argument("--chosen-sft-weight", type=float, default=0.10)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-directory", type=Path)
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if args.checkpoint_directory is not None:
        if args.checkpoint_directory.exists() and any(args.checkpoint_directory.iterdir()):
            raise SystemExit(f"refusing to overwrite non-empty checkpoints: {args.checkpoint_directory}")
        args.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if min(args.batch_size, args.gradient_accumulation, args.max_steps, args.evaluation_steps) < 1:
        raise SystemExit("batch, accumulation, steps, and evaluation interval must be positive")
    if args.learning_rate <= 0 or args.beta <= 0 or args.chosen_sft_weight < 0:
        raise SystemExit("learning-rate/beta must be positive and chosen-sft-weight non-negative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    win = validate_supervised_win(
        args.supervised_win_report, args.supervised_checkpoint, args.direction
    )
    train_rows, valid_rows, preference_manifest = load_preferences(
        args.preference_directory, args.direction
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = MarianTokenizer.from_pretrained(args.supervised_checkpoint)
    model = MarianMTModel.from_pretrained(args.supervised_checkpoint).to(device)
    reference = MarianMTModel.from_pretrained(args.supervised_checkpoint).to(device)
    reference.eval()
    reference.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    collator = PreferenceCollator(tokenizer, args.max_source_tokens, args.max_target_tokens)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PreferenceRows(train_rows), batch_size=args.batch_size, shuffle=True,
        generator=generator, collate_fn=collator,
    )
    valid_loader = DataLoader(
        PreferenceRows(valid_rows), batch_size=args.batch_size, shuffle=False,
        collate_fn=collator,
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

    base_metrics = evaluate(model, reference, valid_loader, device, args.beta)
    history = [{"step": 0, **base_metrics}]
    best = history[0]
    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "operation": "post-SFT human-preference direct-quality-optimization",
        "direction": args.direction,
        "license": "CC-BY-SA-4.0",
        "starting_checkpoint": {
            "path": str(args.supervised_checkpoint),
            "model_sha256": sha256(args.supervised_checkpoint / "model.safetensors"),
        },
        "supervised_win_report": {
            "path": str(args.supervised_win_report),
            "sha256": sha256(args.supervised_win_report),
            "status": win["status"],
            "candidate_model_revision": win["candidateModelRevision"],
        },
        "preferences": {
            "directory": str(args.preference_directory),
            "manifest_sha256": sha256(args.preference_directory / "manifest.json"),
            "train_sha256": preference_manifest["train"]["sha256"],
            "valid_sha256": preference_manifest["valid"]["sha256"],
            "train_pairs": len(train_rows),
            "valid_pairs": len(valid_rows),
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
            "beta": args.beta,
            "chosen_sft_weight": args.chosen_sft_weight,
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "objective": {
            "loss": "-log sigmoid(beta * ((logp_policy_chosen-logp_policy_rejected) - (logp_frozen_sft_chosen-logp_frozen_sft_rejected))) + chosen_sft_weight * chosen_nll",
            "sequence_log_probability": "mean token log probability to avoid length preference",
            "reference_policy": "frozen exact supervised-win checkpoint",
            "preference_source": "two-reviewer consensus only; no adjudicated or automated preferences",
            "private_chain_of_thought": False,
        },
    }
    save_candidate(model, tokenizer, args.output_directory, {**common, "best": best, "history": history})

    update_step = 0
    micro_step = 0
    epoch = 0
    checkpoints: list[dict] = []
    model.train()
    while update_step < args.max_steps:
        for pair in train_loader:
            chosen, rejected = move(pair["chosen"], device), move(pair["rejected"], device)
            policy_chosen, policy_rejected = pair_logps(model, chosen, rejected)
            with torch.inference_mode():
                reference_chosen, reference_rejected = pair_logps(reference, chosen, rejected)
            preference_loss, relative_margin = dqo_loss(
                policy_chosen, policy_rejected, reference_chosen, reference_rejected, args.beta
            )
            chosen_sft_loss = -policy_chosen.mean()
            combined = preference_loss + args.chosen_sft_weight * chosen_sft_loss
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
                metrics = evaluate(model, reference, valid_loader, device, args.beta)
                record = {
                    "step": update_step,
                    **metrics,
                    "last_train_objective": {
                        "preference_loss": float(preference_loss.detach().cpu()),
                        "chosen_sft_loss": float(chosen_sft_loss.detach().cpu()),
                        "relative_margin": float(relative_margin.mean().detach().cpu()),
                    },
                }
                history.append(record)
                if args.checkpoint_directory is not None:
                    checkpoint = args.checkpoint_directory / f"step-{update_step:07d}"
                    checkpoints.append({"step": update_step, "path": str(checkpoint), "metrics": record})
                    save_candidate(
                        model, tokenizer, checkpoint,
                        {**common, "checkpoint_step": update_step, "checkpoint_metrics": record, "history": history},
                    )
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
                    save_candidate(model, tokenizer, args.output_directory, {**common, "best": best, "history": history})
                print(json.dumps({"current": record, "best": best}, ensure_ascii=False))
            if update_step >= args.max_steps:
                break
        epoch += 1

    (args.output_directory / "mimi_dqo_training_manifest.json").write_text(
        json.dumps(
            {**common, "best": best, "history": history, "checkpoints": checkpoints, "epochs": epoch},
            ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output_directory), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
