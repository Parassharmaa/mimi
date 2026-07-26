#!/usr/bin/env python3
"""Distill two directional Marian teachers into one shared bidirectional student."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
ROOT = Path(__file__).resolve().parents[2]
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


def resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "sentencepiece": importlib.metadata.version("sentencepiece"),
        "sacremoses": importlib.metadata.version("sacremoses"),
        "sacrebleu": sacrebleu.__version__,
    }


def same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def validate_launch_contract(
    contract_path: Path,
    args: argparse.Namespace,
) -> tuple[dict, str]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("experiment") != "shared-bidirectional-v18-wide-dense-phase1"
        or contract.get("status") != "phase1-capacity-control-authorized"
        or contract.get("training_authorized") is not True
    ):
        raise SystemExit("V18 launch contract does not authorize this experiment")
    for field in (
        "app_change_authorized",
        "promotion_authorized",
        "public_upload_authorized",
        "private_reasoning_traces_used",
    ):
        if contract.get(field) is not False:
            raise SystemExit(f"V18 launch contract must keep {field}=false")

    trainer_record = contract.get("tools", {}).get("trainer", {})
    current_trainer = Path(__file__).resolve()
    if (
        not same_path(
            resolved(Path(str(trainer_record.get("path", "")))),
            current_trainer,
        )
        or trainer_record.get("sha256") != sha256(current_trainer)
    ):
        raise SystemExit("V18 contract does not authenticate this trainer")

    dataset = contract["dataset"]
    dataset_manifest = args.dataset_directory / "manifest.json"
    if (
        not same_path(
            resolved(Path(dataset["manifest"]["materialized_path"])),
            dataset_manifest,
        )
        or sha256(dataset_manifest) != dataset["manifest"]["sha256"]
        or sha256(args.dataset_directory / "train.jsonl") != dataset["train_sha256"]
        or sha256(args.dataset_directory / "valid.jsonl") != dataset["valid_sha256"]
    ):
        raise SystemExit("V18 launch dataset differs from the contract")

    selection = contract["phase1_training"]["selection_artifact"]
    if args.selection_file is None:
        raise SystemExit("--selection-file is required by the V18 contract")
    if (
        not same_path(resolved(Path(selection["path"])), args.selection_file)
        or sha256(args.selection_file) != selection["sha256"]
    ):
        raise SystemExit("V18 selection artifact differs from the contract")

    model_records = (
        (
            args.initial_checkpoint,
            contract["initialization"]["expected_widened_weights_sha256"],
            "initial checkpoint",
        ),
        (
            args.en_ja_teacher,
            contract["teachers"]["en-ja"]["weights_sha256"],
            "EN-JA teacher",
        ),
        (
            args.ja_en_teacher,
            contract["teachers"]["ja-en"]["weights_sha256"],
            "JA-EN teacher",
        ),
    )
    for path, expected, label in model_records:
        if sha256(path / "model.safetensors") != expected:
            raise SystemExit(f"V18 {label} weights differ from the contract")

    phase = contract["phase1_training"]
    expected_steps = list(
        range(args.evaluation_steps, args.max_steps + 1, args.evaluation_steps)
    )
    if expected_steps != phase["checkpoint_steps"]:
        raise SystemExit("V18 scheduled checkpoint steps differ from the contract")
    parameters = phase["hyperparameters"]
    actual_parameters = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "drop_last": args.drop_last,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "evaluation_steps": args.evaluation_steps,
        "selection_validation_per_direction": args.validation_limit_per_direction,
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
        "teacher_kl_weight": args.teacher_kl_weight,
        "teacher_temperature": args.teacher_temperature,
        "encoder_alignment_weight": args.encoder_alignment_weight,
        "teacher_float16": args.teacher_float16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    differences = {
        key: {"contract": parameters.get(key), "launch": value}
        for key, value in actual_parameters.items()
        if parameters.get(key) != value
    }
    if differences:
        raise SystemExit(
            "V18 launch hyperparameters differ from the contract: "
            + json.dumps(differences, sort_keys=True)
        )
    if package_versions() != contract["runtime"]["packages"]:
        raise SystemExit(
            "V18 runtime packages differ from the contract: "
            + json.dumps(package_versions(), sort_keys=True)
        )
    return contract, sha256(contract_path)


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


def validate_frozen_selection(rows: list[dict], cases_per_direction: int) -> None:
    if cases_per_direction <= 0:
        raise SystemExit("a frozen selection requires a positive direction count")
    for direction in DIRECTIONS:
        count = sum(row["direction"] == direction for row in rows)
        if count != cases_per_direction:
            raise SystemExit(
                f"frozen selection has {count} {direction} rows, "
                f"expected {cases_per_direction}"
            )
    if any(not str(row.get("selection_stratum", "")).strip() for row in rows):
        raise SystemExit("frozen selection row lacks a selection stratum")


@torch.inference_mode()
def evaluate(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    loader: DataLoader,
    rows: list[dict],
    device: torch.device,
    max_new_tokens: int,
    *,
    loss_use_cache: bool = VALIDATION_CACHE_POLICY["loss_forward"],
    generation_use_cache: bool = VALIDATION_CACHE_POLICY["greedy_generation"],
    return_generated_token_ids: bool = False,
) -> dict:
    model.eval()
    hypotheses: list[str] = []
    generated_token_ids: list[list[int]] = []
    losses: list[tuple[float, int]] = []
    for batch in loader:
        batch = move(batch, device)
        batch.pop("direction_ids")
        batch.pop("teacher_input_ids")
        batch.pop("teacher_attention_mask")
        outputs = model(
            **batch,
            use_cache=loss_use_cache,
        )
        batch_size = int(batch["input_ids"].shape[0])
        losses.append((float(outputs.loss), batch_size))
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=generation_use_cache,
        )
        if return_generated_token_ids:
            generated_token_ids.extend(generated.detach().cpu().tolist())
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
    result = {
        "loss": sum(value * count for value, count in losses) / sum(
            count for _, count in losses
        ),
        "macro_direction_chrf_pp": macro,
        "directions": by_direction,
    }
    if return_generated_token_ids:
        result["generated_token_ids"] = generated_token_ids
    return result


def write_summary_manifest(output: Path, manifest: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / f".mimi_training_manifest.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "mimi_training_manifest.json")


def recursive_file_records(directory: Path) -> dict[str, dict[str, int | str]]:
    return {
        str(path.relative_to(directory)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "mimi_checkpoint_manifest.json"
    }


def save_immutable_checkpoint(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    output: Path,
    *,
    step: int,
    metrics: dict,
    objective: dict,
    contract_sha256: str,
) -> tuple[Path, dict]:
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    destination = checkpoints / f"step-{step:07d}"
    temporary = checkpoints / f".step-{step:07d}.{os.getpid()}.tmp"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite immutable checkpoint: {destination}")
    if temporary.exists():
        raise SystemExit(f"stale temporary checkpoint requires inspection: {temporary}")
    temporary.mkdir()
    inference_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        model.save_pretrained(temporary, safe_serialization=True)
    finally:
        model.config.use_cache = inference_cache
    tokenizer.save_pretrained(temporary)
    checkpoint_manifest = {
        "schema_version": 1,
        "experiment": "shared-bidirectional-v18-wide-dense-phase1",
        "step": step,
        "contract_sha256": contract_sha256,
        "metrics": metrics,
        "training_objective": objective,
        "immutable_scheduled_checkpoint": True,
        "inference_config_use_cache": True,
        "files": recursive_file_records(temporary),
    }
    (temporary / "mimi_checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination, checkpoint_manifest


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_rolling_resume_state(
    output: Path,
    *,
    checkpoint: Path,
    checkpoint_manifest: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    train_generator: torch.Generator,
    epoch_start_generator_state: torch.Tensor,
    post_iterator_generator_state: torch.Tensor,
    progress: dict,
    history: list[dict],
    best: dict,
    contract_sha256: str,
) -> dict:
    resume = output / "resume"
    resume.mkdir(parents=True, exist_ok=True)
    step = int(progress["update_step"])
    if (
        int(checkpoint_manifest.get("step", -1)) != step
        or checkpoint_manifest.get("contract_sha256") != contract_sha256
    ):
        raise SystemExit("resume state disagrees with its scheduled checkpoint")
    temporary_state = resume / f".step-{step:07d}.{os.getpid()}.pt.tmp"
    final_state = resume / f"step-{step:07d}.pt"
    if final_state.exists() or temporary_state.exists():
        raise SystemExit(f"refusing to overwrite resume state for step {step}")
    state = {
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": sha256(
            checkpoint / "mimi_checkpoint_manifest.json"
        ),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "rng": cpu_tree(rng_state()),
        "train_generator_state": train_generator.get_state(),
        "epoch_start_generator_state": epoch_start_generator_state,
        "post_iterator_generator_state": post_iterator_generator_state,
        "progress": progress,
        "history": history,
        "best": best,
    }
    torch.save(state, temporary_state)
    os.replace(temporary_state, final_state)
    state_manifest = {
        "schema_version": 1,
        "step": step,
        "contract_sha256": contract_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": state["checkpoint_manifest_sha256"],
        "state": {
            "path": str(final_state),
            "bytes": final_state.stat().st_size,
            "sha256": sha256(final_state),
        },
        "rolling_policy": (
            "only the newest full optimizer/RNG resume state is retained; every "
            "scheduled model/tokenizer checkpoint remains immutable"
        ),
    }
    temporary_manifest = resume / f".latest.{os.getpid()}.json.tmp"
    temporary_manifest.write_text(
        json.dumps(state_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, resume / "latest.json")
    for previous in resume.glob("step-*.pt"):
        if previous != final_state:
            previous.unlink()
    return state_manifest


def authenticate_checkpoint(checkpoint: Path, contract_sha256: str) -> dict:
    manifest_path = checkpoint / "mimi_checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"checkpoint manifest is missing: {checkpoint}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract_sha256") != contract_sha256
        or manifest.get("immutable_scheduled_checkpoint") is not True
    ):
        raise SystemExit("resume checkpoint is not authenticated by this contract")
    for name, record in manifest.get("files", {}).items():
        path = checkpoint / name
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256(path) != record["sha256"]
        ):
            raise SystemExit(f"resume checkpoint file differs: {path}")
    return manifest


def mean_objectives(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        raise ValueError("cannot aggregate an empty objective list")
    return {
        key: sum(item[key] for item in items) / len(items)
        for key in items[0]
    }


def load_authenticated_resume_state(
    output: Path,
    checkpoint: Path,
    contract_sha256: str,
) -> tuple[dict, dict]:
    checkpoint_manifest = authenticate_checkpoint(checkpoint, contract_sha256)
    latest_path = output / "resume" / "latest.json"
    if not latest_path.is_file():
        raise SystemExit("authenticated rolling resume manifest is missing")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    state_record = latest.get("state", {})
    state_path = Path(str(state_record.get("path", "")))
    if (
        latest.get("contract_sha256") != contract_sha256
        or not same_path(Path(str(latest.get("checkpoint", ""))), checkpoint)
        or latest.get("checkpoint_manifest_sha256")
        != sha256(checkpoint / "mimi_checkpoint_manifest.json")
        or not state_path.is_file()
        or state_path.stat().st_size != int(state_record.get("bytes", -1))
        or sha256(state_path) != state_record.get("sha256")
    ):
        raise SystemExit("rolling resume state is not authenticated")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if (
        state.get("contract_sha256") != contract_sha256
        or not same_path(Path(str(state.get("checkpoint", ""))), checkpoint)
        or int(state.get("progress", {}).get("update_step", -1))
        != int(checkpoint_manifest["step"])
    ):
        raise SystemExit("rolling resume payload disagrees with its checkpoint")
    return state, latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("en_ja_teacher", type=Path)
    parser.add_argument("ja_en_teacher", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--validate-only", action="store_true")
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
    parser.add_argument(
        "--teacher-float16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop an incomplete training microbatch so every update has exact size",
    )
    args = parser.parse_args()

    if args.resume_checkpoint is None:
        if args.output_directory.exists() and any(args.output_directory.iterdir()):
            raise SystemExit(
                f"refusing to overwrite non-empty output: {args.output_directory}"
            )
    elif not args.output_directory.is_dir():
        raise SystemExit("resume requires an existing output directory")
    if args.contract is None:
        raise SystemExit("--contract is required for a full V18 training run")
    if args.drop_last is not True:
        raise SystemExit("V18 requires --drop-last for exact-size accumulated updates")
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
    launch_contract, contract_sha256 = validate_launch_contract(args.contract, args)
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
    if args.selection_file is not None:
        valid_rows = load_rows(args.selection_file)
        validate_frozen_selection(valid_rows, args.validation_limit_per_direction)
    else:
        valid_rows = validation_subset(
            all_valid_rows, args.validation_limit_per_direction, args.seed
        )
    if any(
        sum(row["direction"] == direction for row in train_rows)
        != sum(row["direction"] == next(iter(DIRECTIONS)) for row in train_rows)
        for direction in DIRECTIONS
    ):
        raise SystemExit("training mixture is not direction-balanced")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "V18 launch validated",
                    "contract_sha256": contract_sha256,
                    "train_rows": len(train_rows),
                    "selection_rows": len(valid_rows),
                    "exact_examples_per_update": (
                        args.batch_size * args.gradient_accumulation
                    ),
                },
                sort_keys=True,
            )
        )
        return

    model_source = args.resume_checkpoint or args.initial_checkpoint
    if args.resume_checkpoint is not None:
        authenticate_checkpoint(args.resume_checkpoint, contract_sha256)
    student = MarianMTModel.from_pretrained(model_source).to(device)
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
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Rows(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        drop_last=args.drop_last,
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
    if len(train_loader) % args.gradient_accumulation:
        raise SystemExit(
            "drop-last training loader does not divide into exact accumulated updates"
        )

    resume_state = None
    resume_latest = None
    if args.resume_checkpoint is not None:
        resume_state, resume_latest = load_authenticated_resume_state(
            args.output_directory,
            args.resume_checkpoint,
            contract_sha256,
        )
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])

    existing_summary = {}
    existing_summary_path = args.output_directory / "mimi_training_manifest.json"
    if resume_state is not None and existing_summary_path.is_file():
        existing_summary = json.loads(
            existing_summary_path.read_text(encoding="utf-8")
        )
    common_manifest = {
        "schema_version": 1,
        "created_at": existing_summary.get("created_at")
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
            "selection_path": str(args.selection_file),
            "selection_sha256": sha256(args.selection_file),
        },
        "contract": {
            "path": str(args.contract),
            "sha256": contract_sha256,
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "runtime": package_versions(),
        "device": args.device,
        "hyperparameters": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "drop_last": args.drop_last,
            "dropped_rows_per_epoch": len(train_rows)
            - len(train_loader) * args.batch_size,
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
            "maximum unweighted macro-average direction chrF++ on the exact frozen "
            "direction/domain/length-stratified selection artifact; tie-break minimum "
            "aggregate development loss"
        ),
        "checkpoint_policy": {
            "scheduled_steps": launch_contract["phase1_training"]["checkpoint_steps"],
            "immutable_model_and_tokenizer_at_every_scheduled_step": True,
            "best_is_manifest_pointer_only": True,
            "rolling_authenticated_full_resume_state": True,
            "rolling_resume_states_retained": 1,
        },
        "student_teacher_compatibility": compatibility,
        "encoder_alignment": (
            "projection-free token-state MSE on a separate unprefixed student encoder "
            "pass with the exact teacher input IDs and attention mask"
            if args.encoder_alignment_weight
            else "disabled"
        ),
        "private_chain_of_thought_stored": False,
    }

    if resume_state is None:
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
        best_artifact = {
            "kind": "authenticated-initial-checkpoint",
            "path": str(args.initial_checkpoint),
            "weights_sha256": sha256(
                args.initial_checkpoint / "model.safetensors"
            ),
        }
        checkpoints: list[dict] = []
        resume_manifest = None
        update_step = 0
        micro_step = 0
        epoch = 0
        start_batch_index = 0
        write_summary_manifest(
            args.output_directory,
            {
                **common_manifest,
                "best": best,
                "best_artifact": best_artifact,
                "history": history,
                "checkpoints": checkpoints,
                "latest_resume_state": resume_manifest,
                "completed": False,
            },
        )
        resume_epoch_start_state = None
        resume_post_iterator_state = None
    else:
        best_state = resume_state["best"]
        best = best_state["metrics"]
        best_artifact = best_state["artifact"]
        checkpoints = best_state["checkpoints"]
        history = resume_state["history"]
        progress = resume_state["progress"]
        update_step = int(progress["update_step"])
        micro_step = int(progress["micro_step"])
        epoch = int(progress["epoch"])
        start_batch_index = int(progress["next_batch_index"])
        resume_manifest = resume_latest
        resume_epoch_start_state = resume_state["epoch_start_generator_state"]
        resume_post_iterator_state = resume_state["post_iterator_generator_state"]
        restore_rng_state(resume_state["rng"])
        if micro_step % args.gradient_accumulation:
            raise SystemExit("resume checkpoint is not on an optimizer boundary")

    interval_objectives: list[dict[str, float]] = []
    micro_objectives: list[torch.Tensor] = []
    objective_names = (
        "cross_entropy",
        "teacher_kl",
        "encoder_alignment",
        "combined",
    )
    student.train()
    while update_step < args.max_steps:
        resuming_current_epoch = resume_epoch_start_state is not None
        if resuming_current_epoch:
            train_generator.set_state(resume_epoch_start_state)
        epoch_start_generator_state = train_generator.get_state().clone()
        train_iterator = iter(train_loader)
        post_iterator_generator_state = train_generator.get_state().clone()
        if (
            resuming_current_epoch
            and resume_post_iterator_state is not None
            and not torch.equal(
                post_iterator_generator_state, resume_post_iterator_state
            )
        ):
            raise SystemExit("resume DataLoader permutation state differs")
        for batch_index, batch in enumerate(train_iterator):
            if batch_index < start_batch_index:
                continue
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
            micro_objectives.append(
                torch.stack(
                    (
                        ce_loss.detach().float(),
                        kl_loss.detach().float(),
                        encoder_alignment_loss.detach().float(),
                        combined.detach().float(),
                    )
                )
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
            if len(micro_objectives) != args.gradient_accumulation:
                raise RuntimeError(
                    "optimizer update did not aggregate the frozen microbatch count"
                )
            objective_values = (
                torch.stack(micro_objectives).mean(dim=0).cpu().tolist()
            )
            update_objective = {
                name: float(value)
                for name, value in zip(objective_names, objective_values)
            }
            interval_objectives.append(update_objective)
            micro_objectives.clear()
            if update_step % args.evaluation_steps == 0 or update_step == args.max_steps:
                metrics = evaluate(
                    student,
                    tokenizer,
                    valid_loader,
                    valid_rows,
                    device,
                    args.max_target_tokens,
                )
                training_objective = {
                    "last_update_mean_over_microbatches": update_objective,
                    "interval_mean_over_updates": mean_objectives(
                        interval_objectives
                    ),
                    "updates_in_interval": len(interval_objectives),
                    "microbatches_per_update": args.gradient_accumulation,
                    "examples_per_update": (
                        args.batch_size * args.gradient_accumulation
                    ),
                }
                record = {
                    "step": update_step,
                    **metrics,
                    "training_objective": training_objective,
                }
                history.append(record)
                checkpoint, checkpoint_manifest = save_immutable_checkpoint(
                    student,
                    tokenizer,
                    args.output_directory,
                    step=update_step,
                    metrics=metrics,
                    objective=training_objective,
                    contract_sha256=contract_sha256,
                )
                checkpoint_record = {
                    "step": update_step,
                    "path": str(checkpoint),
                    "manifest_sha256": sha256(
                        checkpoint / "mimi_checkpoint_manifest.json"
                    ),
                    "metrics": metrics,
                }
                checkpoints.append(checkpoint_record)
                if (
                    metrics["macro_direction_chrf_pp"],
                    -metrics["loss"],
                ) > (best["macro_direction_chrf_pp"], -best["loss"]):
                    best = record
                    best_artifact = {
                        "kind": "immutable-scheduled-checkpoint",
                        **checkpoint_record,
                    }
                progress = {
                    "update_step": update_step,
                    "micro_step": micro_step,
                    "epoch": epoch,
                    "next_batch_index": batch_index + 1,
                }
                resume_manifest = save_rolling_resume_state(
                    args.output_directory,
                    checkpoint=checkpoint,
                    checkpoint_manifest=checkpoint_manifest,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_generator=train_generator,
                    epoch_start_generator_state=epoch_start_generator_state,
                    post_iterator_generator_state=post_iterator_generator_state,
                    progress=progress,
                    history=history,
                    best={
                        "metrics": best,
                        "artifact": best_artifact,
                        "checkpoints": checkpoints,
                    },
                    contract_sha256=contract_sha256,
                )
                write_summary_manifest(
                    args.output_directory,
                    {
                        **common_manifest,
                        "best": best,
                        "best_artifact": best_artifact,
                        "history": history,
                        "checkpoints": checkpoints,
                        "latest_resume_state": resume_manifest,
                        "completed": update_step == args.max_steps,
                    },
                )
                print(
                    json.dumps(
                        {
                            "current": record,
                            "best": best,
                            "checkpoint": checkpoint_record,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                interval_objectives.clear()
            if update_step >= args.max_steps:
                break
        epoch += 1
        start_batch_index = 0
        resume_epoch_start_state = None
        resume_post_iterator_state = None

    print(
        json.dumps(
            {
                "output": str(args.output_directory),
                "best": best,
                "best_artifact": best_artifact,
                "checkpoints": checkpoints,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
