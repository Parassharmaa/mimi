#!/usr/bin/env python3
"""Full-parameter sequence-level distillation for one ElanMT direction."""

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
from huggingface_hub import snapshot_download
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    get_linear_schedule_with_warmup,
)

from training_manifest_provenance import (
    authenticate_dataset_manifest,
    authenticate_structural_pruning_manifest,
    derive_target_provenance,
)


MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
]


def mlx_affine_dequantize(
    weight: torch.Tensor,
    *,
    group_size: int = 64,
    bits: int = 4,
) -> torch.Tensor:
    """Reproduce MLX affine quantize+dequantize without packing the integers.

    This follows the pinned MLX 0.30.6 affine kernel, including its signed scale,
    zero-aligned edge adjustment, and per-row contiguous groups. It deliberately
    quantizes weights only; MLX leaves Linear biases in floating point.
    """
    if weight.ndim < 2 or weight.shape[-1] % group_size:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} is not divisible by group size {group_size}"
        )
    if group_size not in {32, 64, 128} or bits not in {2, 3, 4, 5, 6, 8}:
        raise ValueError(f"unsupported MLX affine quantization: {bits}-bit group-{group_size}")
    # prepare_elanmt_mlx.py casts the source checkpoint to float16 before
    # quantizing. MLX computes group statistics in float, stores half-precision
    # scales/biases, and rounds the final dequantized values back to half.
    groups = weight.to(torch.float16).float().reshape(
        *weight.shape[:-1], -1, group_size
    )
    minimum = groups.amin(dim=-1, keepdim=True)
    maximum = groups.amax(dim=-1, keepdim=True)
    bins = float((1 << bits) - 1)
    scale = ((maximum - minimum) / bins).clamp_min(1e-7)
    negative_edge = minimum.abs() > maximum.abs()
    scale = torch.where(negative_edge, scale, -scale)
    edge = torch.where(negative_edge, minimum, maximum)
    edge_bin = torch.round(edge / scale)
    at_zero = edge_bin == 0
    scale = torch.where(at_zero, scale, edge / edge_bin)
    bias = torch.where(at_zero, torch.zeros_like(edge), edge)
    quantized = torch.round((groups - bias) / scale).clamp(0, bins)
    stored_scale = scale.to(torch.float16).float()
    stored_bias = bias.to(torch.float16).float()
    return (
        (quantized * stored_scale + stored_bias)
        .to(torch.float16)
        .to(weight.dtype)
        .reshape_as(weight)
    )


class MLXAffineFakeQuantization(nn.Module):
    """Straight-through view of Mimi's exact shipping weight quantizer."""

    def __init__(self, group_size: int = 64, bits: int = 4) -> None:
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self._evaluation_cache: torch.Tensor | None = None

    def train(self, mode: bool = True) -> MLXAffineFakeQuantization:
        # Training changes the raw weights. Invalidate the dequantized view on
        # every mode transition so each validation pass sees the latest update.
        self._evaluation_cache = None
        return super().train(mode)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.training and self._evaluation_cache is not None:
            return self._evaluation_cache
        dequantized = mlx_affine_dequantize(
            weight,
            group_size=self.group_size,
            bits=self.bits,
        )
        if not self.training:
            # Autoregressive generation reuses fixed weights for every token.
            # Caching here matches the shipping quantized module and avoids
            # repeatedly quantizing all 61M parameters during validation.
            self._evaluation_cache = dequantized.detach()
            return self._evaluation_cache
        return weight + (dequantized - weight).detach()


def enable_mlx_affine_fake_quantization(
    model: MarianMTModel,
    *,
    group_size: int,
    bits: int,
) -> list[str]:
    """Parametrize the exact Linear/shared-Embedding weights quantized by MLX."""
    names: list[str] = []
    for name, module in model.named_modules():
        # MarianSinusoidalPositionalEmbedding subclasses Embedding, but Mimi's
        # MLX port computes positions analytically and does not quantize them.
        if not (isinstance(module, nn.Linear) or type(module) is nn.Embedding):
            continue
        if module.weight.shape[-1] % group_size:
            raise SystemExit(
                f"MLX fake-quantized module is not group aligned: {name} / "
                f"{tuple(module.weight.shape)} / group {group_size}"
            )
        parametrize.register_parametrization(
            module,
            "weight",
            MLXAffineFakeQuantization(group_size=group_size, bits=bits),
        )
        names.append(name)
    if not names:
        raise SystemExit("no Marian weights were eligible for MLX fake quantization")
    return names


def disable_mlx_affine_fake_quantization(model: MarianMTModel) -> None:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            # Retain the learned full-precision source weight. The normal MLX
            # conversion performs the one authoritative final quantization.
            parametrize.remove_parametrizations(
                module,
                "weight",
                leave_parametrized=False,
            )


def canonical_parameter_name(name: str) -> str:
    return name.replace(".parametrizations.weight.original", ".weight")


def capture_canonical_state_dict(model: MarianMTModel) -> dict[str, torch.Tensor]:
    """Snapshot raw learned weights while fake-quant parametrizations are active."""
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        canonical = canonical_parameter_name(name)
        value = tensor.detach().cpu().clone()
        previous = state.get(canonical)
        if previous is not None and not torch.equal(previous, value):
            raise RuntimeError(f"conflicting canonical state tensor: {canonical}")
        state[canonical] = value
    return state


def load_rows(path: Path, direction: str) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit(f"dataset is empty: {path}")
    if any(row.get("source_language") == row.get("target_language") for row in rows):
        raise SystemExit(f"dataset has same-language row: {path}")
    expected = {"en-ja": ("en-US", "ja-JP"), "ja-en": ("ja-JP", "en-US")}[direction]
    if any((row.get("source_language"), row.get("target_language")) != expected for row in rows):
        raise SystemExit(f"dataset contains a row for the wrong direction: {path}")
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit(f"dataset contains duplicate IDs: {path}")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


LINEAGE_MANIFEST_NAMES = (
    "mimi_training_manifest.json",
    "mimi_checkpoint_averaging_manifest.json",
    "mimi_checkpoint_interpolation_manifest.json",
    "mimi_structural_pruning_manifest.json",
)


def checkpoint_lineage_manifests(checkpoint: Path) -> list[dict[str, str]]:
    return [
        {
            "path": str(path),
            "sha256": sha256(path),
        }
        for name in LINEAGE_MANIFEST_NAMES
        if (path := checkpoint / name).is_file()
    ]


def checkpoint_identity(checkpoint: Path) -> tuple[str, str, str] | None:
    training_manifest = checkpoint / "mimi_training_manifest.json"
    if training_manifest.is_file():
        payload = json.loads(training_manifest.read_text(encoding="utf-8"))
        return (
            payload.get("direction"),
            payload.get("student_repository"),
            payload.get("student_revision"),
        )
    averaging_manifest = checkpoint / "mimi_checkpoint_averaging_manifest.json"
    if averaging_manifest.is_file():
        payload = json.loads(averaging_manifest.read_text(encoding="utf-8"))
        identity = payload.get("identity") or {}
        return (
            identity.get("direction"),
            identity.get("student_repository"),
            identity.get("student_revision"),
        )
    return None


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


class TranslationRows(Dataset):
    def __init__(
        self,
        rows: list[dict],
        *,
        seed: int = 0,
        sample_target_variants: bool = False,
    ) -> None:
        self.rows = rows
        self.seed = seed
        self.sample_target_variants = sample_target_variants
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        variants = row.get("target_variants")
        if not self.sample_target_variants or not variants:
            return row
        digest = hashlib.sha256(
            f"{self.seed}\0{self.epoch}\0{row['id']}".encode()
        ).digest()
        selected = variants[int.from_bytes(digest[:8], "big") % len(variants)]
        return {
            **row,
            "target": selected["translation"],
            "selected_target_variant_id": selected["candidate_id"],
        }


class Collator:
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        max_source_tokens: int,
        max_target_tokens: int,
        preservation_origins: set[str],
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_tokens = max_source_tokens
        self.max_target_tokens = max_target_tokens
        self.preservation_origins = preservation_origins

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            [row["source"] for row in rows],
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
        batch["preservation_mask"] = torch.tensor(
            [row.get("origin") in self.preservation_origins for row in rows],
            dtype=torch.bool,
        )
        return batch


def weighted_sequence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    preservation_mask: torch.Tensor,
    domain_weight: float,
) -> torch.Tensor:
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    token_mask = labels.ne(-100)
    row_weights = torch.where(
        preservation_mask,
        torch.ones_like(preservation_mask, dtype=token_losses.dtype),
        torch.full_like(preservation_mask, domain_weight, dtype=token_losses.dtype),
    )
    weights = token_mask.to(token_losses.dtype) * row_weights[:, None]
    return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)


def frozen_base_kl(
    student_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    preservation_mask: torch.Tensor,
) -> torch.Tensor:
    token_mask = labels.ne(-100) & preservation_mask[:, None]
    if not bool(token_mask.any()):
        return student_logits.new_zeros((), dtype=torch.float32)
    divergences = F.kl_div(
        F.log_softmax(student_logits.float(), dim=-1),
        F.softmax(base_logits.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return divergences[token_mask].mean()


def l2_to_frozen_base(
    model: MarianMTModel,
    base_parameters: dict[str, torch.Tensor],
) -> torch.Tensor:
    squared = None
    for name, parameter in model.named_parameters():
        reference = base_parameters[canonical_parameter_name(name)]
        value = (parameter.float() - reference.float()).square().sum()
        squared = value if squared is None else squared + value
    if squared is None:
        raise RuntimeError("model has no parameters for L2-to-base")
    return squared


def curriculum_domain_weight(
    update_step: int,
    ramp_steps: int,
    start: float,
    end: float,
) -> float:
    progress = min(1.0, max(0.0, update_step / max(1, ramp_steps)))
    return start + (end - start) * progress


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


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
    losses: list[tuple[float, int]] = []
    hypotheses: list[str] = []
    for batch in loader:
        batch = move(batch, device)
        batch.pop("preservation_mask")
        outputs = model(**batch)
        batch_size = int(batch["input_ids"].shape[0])
        losses.append((float(outputs.loss.item()), batch_size))
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
        hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    synchronize(device)
    references = [row["target"] for row in rows]
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score
    loss = sum(value * count for value, count in losses) / sum(count for _, count in losses)
    slices: dict[str, dict[str, dict]] = {"origin": {}, "domain": {}}
    for field in slices:
        values = sorted({str(row.get(field, "unknown")) for row in rows})
        for value in values:
            indices = [index for index, row in enumerate(rows) if str(row.get(field, "unknown")) == value]
            slices[field][value] = {
                "cases": len(indices),
                "chrf_pp": sacrebleu.corpus_chrf(
                    [hypotheses[index] for index in indices],
                    [[references[index] for index in indices]],
                    word_order=2,
                ).score,
            }
    model.train()
    return {
        "loss": loss,
        "chrf_pp": chrf,
        "hypotheses": len(hypotheses),
        "slices": slices,
    }


def save_candidate(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    output: Path,
    metadata: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    (output / "mimi_training_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--direction", choices=("en-ja", "ja-en"), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Optional local full-precision checkpoint to continue adapting instead of the pinned base.",
    )
    parser.add_argument(
        "--preservation-checkpoint",
        type=Path,
        help=(
            "Optional frozen local teacher/reference for KL and L2. Defaults to the "
            "pinned base; set this to the current preferred checkpoint when adapting it."
        ),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("Research/translation/models/hf-cache"))
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--evaluation-steps", type=int, default=250)
    parser.add_argument(
        "--skip-initial-evaluation",
        action="store_true",
        help=(
            "Skip step-0 generation for structurally pruned recovery pilots whose "
            "known non-terminating outputs would dominate the bounded experiment."
        ),
    )
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--frozen-base-kl-weight", type=float, default=0.0)
    parser.add_argument("--l2-to-base-weight", type=float, default=0.0)
    parser.add_argument("--domain-loss-weight-start", type=float, default=1.0)
    parser.add_argument("--domain-loss-weight-end", type=float, default=1.0)
    parser.add_argument("--curriculum-ramp-steps", type=int)
    parser.add_argument(
        "--mlx-fake-quantization-bits",
        type=int,
        choices=(2, 3, 4, 5, 6, 8),
        help="Train through the pinned MLX affine weight quantizer using a straight-through estimator.",
    )
    parser.add_argument(
        "--mlx-fake-quantization-group-size",
        type=int,
        choices=(32, 64, 128),
        default=64,
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        help="Optional directory for every evaluated full-precision checkpoint used by averaging.",
    )
    parser.add_argument(
        "--preservation-origin",
        action="append",
        default=[],
        help="Repeat for each replay origin receiving frozen-base KL; defaults to human-kftt-replay.",
    )
    parser.add_argument(
        "--training-description",
        default="sequence-level distillation; reviewed targets only; no private chain-of-thought",
    )
    args = parser.parse_args()

    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output_directory}")
    if args.initial_checkpoint is not None and not (
        args.initial_checkpoint / "model.safetensors"
    ).is_file():
        raise SystemExit(f"initial checkpoint is incomplete: {args.initial_checkpoint}")
    if args.preservation_checkpoint is not None and not (
        args.preservation_checkpoint / "model.safetensors"
    ).is_file():
        raise SystemExit(
            f"preservation checkpoint is incomplete: {args.preservation_checkpoint}"
        )
    if args.mlx_fake_quantization_bits is not None and args.checkpoint_directory is not None:
        raise SystemExit(
            "fake-quantization pilot stores the validation-selected raw checkpoint only; "
            "do not combine it with checkpoint-directory"
        )
    if args.checkpoint_directory is not None:
        if args.checkpoint_directory.exists() and any(args.checkpoint_directory.iterdir()):
            raise SystemExit(
                f"refusing to overwrite non-empty checkpoints: {args.checkpoint_directory}"
            )
        args.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if min(args.batch_size, args.gradient_accumulation, args.max_steps, args.evaluation_steps) < 1:
        raise SystemExit("batch, accumulation, steps, and evaluation interval must be positive")
    if min(
        args.frozen_base_kl_weight,
        args.l2_to_base_weight,
        args.domain_loss_weight_start,
        args.domain_loss_weight_end,
    ) < 0:
        raise SystemExit("regularization and curriculum weights must be non-negative")
    if args.curriculum_ramp_steps is not None and args.curriculum_ramp_steps < 1:
        raise SystemExit("curriculum-ramp-steps must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

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

    snapshot = Path(
        snapshot_download(
            repo_id=args.repository,
            revision=args.revision,
            cache_dir=args.hf_home,
            allow_patterns=MODEL_FILES,
        )
    )
    tokenizer = MarianTokenizer.from_pretrained(snapshot)
    initial_checkpoint = args.initial_checkpoint or snapshot
    structural_pruning_manifest = authenticate_structural_pruning_manifest(
        initial_checkpoint
    )
    if args.skip_initial_evaluation and structural_pruning_manifest is None:
        raise SystemExit(
            "--skip-initial-evaluation requires an authenticated structural-pruning "
            "initial checkpoint"
        )
    initial_manifest_path = initial_checkpoint / "mimi_training_manifest.json"
    expected_identity = (args.direction, args.repository, args.revision)
    actual_identity = checkpoint_identity(initial_checkpoint)
    if actual_identity is not None:
        if actual_identity != expected_identity:
            raise SystemExit(
                f"initial checkpoint identity differs: expected {expected_identity}, "
                f"found {actual_identity}"
            )
    model = MarianMTModel.from_pretrained(initial_checkpoint).to(device)
    regularized = args.frozen_base_kl_weight > 0 or args.l2_to_base_weight > 0
    preservation_checkpoint = args.preservation_checkpoint or snapshot
    preservation_manifest_path = preservation_checkpoint / "mimi_training_manifest.json"
    actual_identity = checkpoint_identity(preservation_checkpoint)
    if actual_identity is not None:
        if actual_identity != expected_identity:
            raise SystemExit(
                f"preservation checkpoint identity differs: expected {expected_identity}, "
                f"found {actual_identity}"
            )
    base_model = None
    base_parameters: dict[str, torch.Tensor] = {}
    if regularized:
        base_model = MarianMTModel.from_pretrained(preservation_checkpoint).to(device)
        base_model.eval()
        base_model.requires_grad_(False)
        base_parameters = {
            name: parameter.detach()
            for name, parameter in base_model.named_parameters()
        }
    fake_quantized_modules: list[str] = []
    if args.mlx_fake_quantization_bits is not None:
        fake_quantized_modules = enable_mlx_affine_fake_quantization(
            model,
            group_size=args.mlx_fake_quantization_group_size,
            bits=args.mlx_fake_quantization_bits,
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    preservation_origins = set(args.preservation_origin or ["human-kftt-replay"])
    collator = Collator(
        tokenizer,
        args.max_source_tokens,
        args.max_target_tokens,
        preservation_origins,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = TranslationRows(
        train_rows,
        seed=args.seed,
        sample_target_variants=True,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, args.max_steps),
        num_training_steps=args.max_steps,
    )
    optimizer.zero_grad(set_to_none=True)
    history: list[dict] = []
    if args.skip_initial_evaluation:
        history.append(
            {
                "step": 0,
                "evaluation_skipped": True,
                "reason": "declared structural-recovery pilot",
            }
        )
        best = {
            "step": 0,
            "chrf_pp": float("-inf"),
            "loss": float("inf"),
            "hypotheses": 0,
            "evaluation_skipped": True,
        }
        best_state = None
    else:
        base_metrics = evaluate(
            model, tokenizer, valid_loader, valid_rows, device, args.max_target_tokens
        )
        history.append({"step": 0, **base_metrics})
        best = {"step": 0, **base_metrics}
        best_state = (
            capture_canonical_state_dict(model)
            if args.mlx_fake_quantization_bits is not None
            else None
        )

    common_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "direction": args.direction,
        "student_repository": args.repository,
        "student_revision": args.revision,
        "license": "CC-BY-SA-4.0",
        "training_description": target_provenance["training_description"],
        "initial_checkpoint": {
            "path": str(initial_checkpoint),
            "model_sha256": sha256(initial_checkpoint / "model.safetensors"),
            "training_manifest_sha256": (
                sha256(initial_manifest_path) if initial_manifest_path.is_file() else None
            ),
            "structural_pruning_manifest": structural_pruning_manifest,
            "lineage_manifests": checkpoint_lineage_manifests(initial_checkpoint),
        },
        "preservation_checkpoint": {
            "path": str(preservation_checkpoint),
            "model_sha256": sha256(preservation_checkpoint / "model.safetensors"),
            "training_manifest_sha256": (
                sha256(preservation_checkpoint / "mimi_training_manifest.json")
                if (preservation_checkpoint / "mimi_training_manifest.json").is_file()
                else None
            ),
            "lineage_manifests": checkpoint_lineage_manifests(
                preservation_checkpoint
            ),
        },
        "dataset": {
            "train_path": str(train_path),
            "train_sha256": sha256(train_path),
            "train_rows": len(train_rows),
            "valid_path": str(valid_path),
            "valid_sha256": sha256(valid_path),
            "valid_rows": len(valid_rows),
            "train_rows_with_reviewed_target_variants": sum(
                bool(row.get("target_variants")) for row in train_rows
            ),
        },
        "dataset_manifest": dataset_manifest_metadata,
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
            "initial_evaluation_skipped": args.skip_initial_evaluation,
            "max_source_tokens": args.max_source_tokens,
            "max_target_tokens": args.max_target_tokens,
            "gradient_checkpointing": args.gradient_checkpointing,
            "frozen_base_kl_weight": args.frozen_base_kl_weight,
            "l2_to_base_weight": args.l2_to_base_weight,
            "domain_loss_weight_start": args.domain_loss_weight_start,
            "domain_loss_weight_end": args.domain_loss_weight_end,
            "curriculum_ramp_steps": args.curriculum_ramp_steps or args.max_steps,
            "preservation_origins": sorted(preservation_origins),
            "checkpoint_directory": (
                str(args.checkpoint_directory) if args.checkpoint_directory is not None else None
            ),
            "mlx_fake_quantization_bits": args.mlx_fake_quantization_bits,
            "mlx_fake_quantization_group_size": (
                args.mlx_fake_quantization_group_size
                if args.mlx_fake_quantization_bits is not None
                else None
            ),
            "mlx_fake_quantized_modules": fake_quantized_modules,
        },
        "objective": {
            "sequence_target": target_provenance["sequence_target"],
            "preservation_teacher": (
                "frozen declared preservation checkpoint on declared replay origins"
                if regularized
                else None
            ),
            "kl_direction": "KL(frozen-base || adapted-student)",
            "l2_normalization": "sum of squared parameter displacement",
            "curriculum": "linear domain token-loss weight with replay floor 1.0",
            "quantization_aware_training": (
                "straight-through exact MLX affine weight quantize-dequantize; "
                "floating-point biases; analytic positional embeddings excluded"
                if args.mlx_fake_quantization_bits is not None
                else None
            ),
        },
    }
    if args.mlx_fake_quantization_bits is None:
        save_candidate(
            model,
            tokenizer,
            args.output_directory,
            {**common_manifest, "best": best, "history": history},
        )

    update_step = 0
    micro_step = 0
    checkpoints: list[dict] = []
    epoch = 0
    model.train()
    while update_step < args.max_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            batch = move(batch, device)
            preservation_mask = batch.pop("preservation_mask")
            labels = batch["labels"]
            outputs = model(**batch)
            domain_weight = curriculum_domain_weight(
                update_step,
                args.curriculum_ramp_steps or args.max_steps,
                args.domain_loss_weight_start,
                args.domain_loss_weight_end,
            )
            sequence_loss = weighted_sequence_cross_entropy(
                outputs.logits,
                labels,
                preservation_mask,
                domain_weight,
            )
            kl_loss = outputs.logits.new_zeros((), dtype=torch.float32)
            if args.frozen_base_kl_weight > 0 and bool(preservation_mask.any()):
                assert base_model is not None
                with torch.inference_mode():
                    base_logits = base_model(**batch).logits
                kl_loss = frozen_base_kl(
                    outputs.logits,
                    base_logits,
                    labels,
                    preservation_mask,
                )
            l2_loss = outputs.logits.new_zeros((), dtype=torch.float32)
            if args.l2_to_base_weight > 0:
                l2_loss = l2_to_frozen_base(model, base_parameters)
            combined_loss = (
                sequence_loss
                + args.frozen_base_kl_weight * kl_loss
                + args.l2_to_base_weight * l2_loss
            )
            loss = combined_loss / args.gradient_accumulation
            loss.backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
            if update_step % args.evaluation_steps == 0 or update_step == args.max_steps:
                metrics = evaluate(
                    model, tokenizer, valid_loader, valid_rows, device, args.max_target_tokens
                )
                record = {"step": update_step, **metrics}
                record["training_objective"] = {
                    "sequence_loss": float(sequence_loss.detach().cpu()),
                    "frozen_base_kl": float(kl_loss.detach().cpu()),
                    "l2_to_base": float(l2_loss.detach().cpu()),
                    "domain_loss_weight": domain_weight,
                }
                history.append(record)
                if args.checkpoint_directory is not None:
                    checkpoint_path = args.checkpoint_directory / f"step-{update_step:07d}"
                    checkpoint_record = {
                        "step": update_step,
                        "path": str(checkpoint_path),
                        "metrics": record,
                    }
                    checkpoints.append(checkpoint_record)
                    save_candidate(
                        model,
                        tokenizer,
                        checkpoint_path,
                        {
                            **common_manifest,
                            "checkpoint_step": update_step,
                            "checkpoint_metrics": record,
                            "history": history,
                        },
                    )
                if (metrics["chrf_pp"], -metrics["loss"]) > (
                    best["chrf_pp"],
                    -best["loss"],
                ):
                    best = record
                    if args.mlx_fake_quantization_bits is not None:
                        best_state = capture_canonical_state_dict(model)
                    else:
                        save_candidate(
                            model,
                            tokenizer,
                            args.output_directory,
                            {**common_manifest, "best": best, "history": history},
                        )
                print(json.dumps({"current": record, "best": best}, ensure_ascii=False))
            if update_step >= args.max_steps:
                break
        epoch += 1

    final_manifest = {
        **common_manifest,
        "best": best,
        "history": history,
        "checkpoints": checkpoints,
    }
    if args.mlx_fake_quantization_bits is not None:
        assert best_state is not None
        disable_mlx_affine_fake_quantization(model)
        model.load_state_dict(best_state, strict=True)
        save_candidate(model, tokenizer, args.output_directory, final_manifest)
    else:
        # Keep the manifest current even when the base checkpoint remained best.
        (args.output_directory / "mimi_training_manifest.json").write_text(
            json.dumps(
                final_manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output": str(args.output_directory), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
