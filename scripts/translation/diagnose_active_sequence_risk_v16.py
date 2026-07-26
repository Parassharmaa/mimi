#!/usr/bin/env python3
"""Mine active full-sequence omission/repetition risks and audit gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from build_marian_negative_space_dataset import (
    duplication_variants,
    omission_variants,
)
from filter_training_dataset_against_protected import sha256
from train_marian_distillation import hardware_name, synchronize
from transformers import MarianMTModel, MarianTokenizer

EXPERIMENT = "active-sequence-risk-v16-ja-en-diagnostic"
DATASET_MANIFEST_SHA256 = (
    "9b412e0a7d49234ab374f4e47fc71e0f70e9cb432af6f792d26e7ce56910c523"
)
PARENT_MODEL_SHA256 = "8e7f7eff76d74b343884fe9a170b6dbad55d42f20ac5f526b6e8ec71e6c94f71"
PARENT_MANIFEST_SHA256 = (
    "0d195dc163250a9fa9312fb7ad8ba3341ab65167b90926d46f0a65d76047bd38"
)


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": display_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def stable_rank(seed: int, role: str, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{seed}\0{role}\0{row.get('id', '')}\0"
            f"{row.get('source', '')}\0{row.get('target', '')}"
        ).encode()
    ).hexdigest()


def select_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    limit: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            (
                0
                if str(row["v15_stratum"]).endswith(
                    ("omission-risk", "repetition-risk", "long")
                )
                else 1
            ),
            stable_rank(seed, "parent", row),
            str(row["id"]),
        ),
    )
    return ordered[:limit]


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    pairs = []
    for row in rows:
        variants = {
            "omission": omission_variants(
                str(row["target"]),
                str(row["target_language"]),
            ),
            "repetition": duplication_variants(
                str(row["target"]),
                str(row["target_language"]),
            ),
        }
        for role, choices in variants.items():
            if not choices:
                continue
            index = int(
                stable_rank(seed, role, row)[:8],
                16,
            ) % len(choices)
            rejected, violation_type, severity = choices[index]
            if rejected == row["target"]:
                continue
            pairs.append(
                {
                    "id": f"v16-{role}:{row['id']}",
                    "parent_id": row["id"],
                    "source": row["source"],
                    "chosen": row["target"],
                    "rejected": rejected,
                    "source_language": row["source_language"],
                    "target_language": row["target_language"],
                    "source_license": row["source_license"],
                    "source_provenance": row["source_provenance"],
                    "origin": row["origin"],
                    "v15_stratum": row["v15_stratum"],
                    "risk_role": role,
                    "violation_type": violation_type,
                    "severity": severity,
                    "positive_target_source": "licensed-human-reference",
                    "negative_generation": (
                        "deterministic-reference-corruption-used-only-as-negative"
                    ),
                    "generated_strings_are_positive_targets": False,
                }
            )
    return pairs


def token_batch(
    tokenizer: MarianTokenizer,
    sources: list[str],
    targets: list[str],
    *,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        sources,
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=maximum_source_tokens,
        return_tensors="pt",
    )
    if batch["labels"].shape[1] > maximum_target_tokens:
        labels = batch["labels"][:, :maximum_target_tokens].clone()
        labels[:, -1] = int(tokenizer.eos_token_id)
        batch["labels"] = labels
    labels = batch["labels"]
    labels[labels == int(tokenizer.pad_token_id)] = -100
    return {key: value.to(device) for key, value in batch.items()}


def sequence_scores(
    model: MarianMTModel,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(**batch)
    labels = batch["labels"]
    mask = labels.ne(-100)
    selected = (
        F.log_softmax(outputs.logits.float(), dim=-1)
        .gather(
            2,
            labels.clamp_min(0).unsqueeze(2),
        )
        .squeeze(2)
    )
    totals = (selected * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp_min(1)
    return totals / lengths, lengths


@torch.inference_mode()
def score_pairs(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    pairs: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
) -> list[dict[str, Any]]:
    model.eval()
    scored = []
    for start in range(0, len(pairs), batch_size):
        batch_rows = pairs[start : start + batch_size]
        sources = [str(row["source"]) for row in batch_rows]
        chosen_batch = token_batch(
            tokenizer,
            sources,
            [str(row["chosen"]) for row in batch_rows],
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
            device=device,
        )
        rejected_batch = token_batch(
            tokenizer,
            sources,
            [str(row["rejected"]) for row in batch_rows],
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
            device=device,
        )
        chosen_scores, chosen_lengths = sequence_scores(model, chosen_batch)
        rejected_scores, rejected_lengths = sequence_scores(model, rejected_batch)
        for row, chosen, rejected, chosen_length, rejected_length in zip(
            batch_rows,
            chosen_scores.cpu(),
            rejected_scores.cpu(),
            chosen_lengths.cpu(),
            rejected_lengths.cpu(),
        ):
            scored.append(
                {
                    **row,
                    "chosen_mean_log_probability": float(chosen),
                    "rejected_mean_log_probability": float(rejected),
                    "chosen_minus_rejected_margin": float(chosen - rejected),
                    "chosen_token_count": int(chosen_length),
                    "rejected_token_count": int(rejected_length),
                }
            )
    synchronize(device)
    return scored


def objective_gradients(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    rows: list[dict[str, Any]],
    *,
    objective: str,
    target_margin: float,
    maximum_source_tokens: int,
    maximum_target_tokens: int,
    device: torch.device,
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    model.zero_grad(set_to_none=True)
    sources = [str(row["source"]) for row in rows]
    chosen_batch = token_batch(
        tokenizer,
        sources,
        [str(row["chosen"]) for row in rows],
        maximum_source_tokens=maximum_source_tokens,
        maximum_target_tokens=maximum_target_tokens,
        device=device,
    )
    chosen_scores, _ = sequence_scores(model, chosen_batch)
    if objective == "mle":
        loss = -chosen_scores.mean()
        active_fraction = 1.0
    else:
        rejected_batch = token_batch(
            tokenizer,
            sources,
            [str(row["rejected"]) for row in rows],
            maximum_source_tokens=maximum_source_tokens,
            maximum_target_tokens=maximum_target_tokens,
            device=device,
        )
        rejected_scores, _ = sequence_scores(model, rejected_batch)
        margins = chosen_scores - rejected_scores
        losses = F.relu(target_margin - margins)
        loss = losses.mean()
        active_fraction = float(losses.gt(0).float().mean().detach().cpu())
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    values = torch.autograd.grad(
        loss,
        parameters,
        allow_unused=True,
    )
    synchronize(device)
    gradients = [
        None if value is None else value.detach().float().cpu() for value in values
    ]
    squared_norm = sum(
        float(value.double().square().sum()) for value in gradients if value is not None
    )
    return gradients, {
        "loss": float(loss.detach().cpu()),
        "gradient_norm": math.sqrt(squared_norm),
        "active_fraction": active_fraction,
        "rows": float(len(rows)),
    }


def gradient_cosine(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        if left_value is None or right_value is None:
            continue
        left_double = left_value.double()
        right_double = right_value.double()
        dot += float((left_double * right_double).sum())
        left_norm += float(left_double.square().sum())
        right_norm += float(right_double.square().sum())
    denominator = math.sqrt(left_norm * right_norm)
    return dot / denominator if denominator else float("nan")


def quantiles(values: list[float]) -> dict[str, float]:
    points = np.quantile(
        np.asarray(values, dtype=np.float64),
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    )
    return {
        name: float(value)
        for name, value in zip(
            ("minimum", "p10", "p25", "median", "p75", "p90", "maximum"),
            points,
        )
    }


def scalar_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "minimum": min(values),
        "maximum": max(values),
        "negative_fraction": sum(value < 0 for value in values) / len(values),
    }


def select_gradient_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    role: str,
    limit: int,
) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_stratum.setdefault(str(row["v15_stratum"]), []).append(row)
    for stratum_rows in by_stratum.values():
        stratum_rows.sort(
            key=lambda row: (
                float(row["chosen_minus_rejected_margin"]),
                stable_rank(seed, f"gradient:{role}", row),
            )
        )
    selected = []
    while len(selected) < limit:
        progressed = False
        for stratum in sorted(by_stratum):
            if by_stratum[stratum]:
                selected.append(by_stratum[stratum].pop(0))
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("safe_parent", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--row-limit", type=int, default=1_024)
    parser.add_argument("--active-margin", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-batch-size", type=int, default=16)
    parser.add_argument("--gradient-replicates", type=int, default=4)
    parser.add_argument("--maximum-source-tokens", type=int, default=192)
    parser.add_argument("--maximum-target-tokens", type=int, default=192)
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default="mps",
    )
    args = parser.parse_args()
    if args.output_directory.exists() and any(args.output_directory.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty output: {args.output_directory}"
        )
    if (
        min(
            args.row_limit,
            args.batch_size,
            args.gradient_batch_size,
            args.gradient_replicates,
        )
        < 1
    ):
        raise SystemExit("diagnostic sizes must be positive")
    if args.active_margin <= 0:
        raise SystemExit("active margin must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    root = Path(__file__).resolve().parents[2]
    dataset_manifest_path = args.dataset_directory / "manifest.json"
    train_path = args.dataset_directory / "train.jsonl"
    dataset_manifest = load_json(dataset_manifest_path)
    if (
        sha256(dataset_manifest_path) != DATASET_MANIFEST_SHA256
        or dataset_manifest.get("experiment")
        != "canonical-constrained-recovery-v15-ja-en"
        or dataset_manifest.get("outputs", {}).get("train", {}).get("sha256")
        != sha256(train_path)
        or dataset_manifest.get("counts", {}).get("train") != 7_104
        or dataset_manifest.get("distribution_provenance", {}).get(
            "all_positive_targets_are_licensed_human_references"
        )
        is not True
        or dataset_manifest.get("decontamination", {}).get("protected_hits_in_outputs")
        != 0
    ):
        raise SystemExit("v16 diagnostic dataset safety state differs")
    model_path = args.safe_parent / "model.safetensors"
    parent_manifest_path = args.safe_parent / "mimi_training_manifest.json"
    if (
        sha256(model_path) != PARENT_MODEL_SHA256
        or sha256(parent_manifest_path) != PARENT_MANIFEST_SHA256
    ):
        raise SystemExit("v16 safe parent differs")

    rows = select_rows(
        load_jsonl(train_path),
        seed=args.seed,
        limit=args.row_limit,
    )
    pairs = build_pairs(rows, seed=args.seed)
    tokenizer = MarianTokenizer.from_pretrained(args.safe_parent)
    device = torch.device(args.device)
    model = MarianMTModel.from_pretrained(args.safe_parent).to(device)
    scored = score_pairs(
        model,
        tokenizer,
        pairs,
        device=device,
        batch_size=args.batch_size,
        maximum_source_tokens=args.maximum_source_tokens,
        maximum_target_tokens=args.maximum_target_tokens,
    )
    active = [
        row
        for row in scored
        if float(row["chosen_minus_rejected_margin"]) < args.active_margin
    ]
    active_by_role = {
        role: [row for row in active if row["risk_role"] == role]
        for role in ("omission", "repetition")
    }
    gradient_rows_required = args.gradient_batch_size * args.gradient_replicates
    if any(len(values) < gradient_rows_required for values in active_by_role.values()):
        raise SystemExit("insufficient active pairs for the frozen gradient audit")

    gradient_pools = {
        role: select_gradient_rows(
            values,
            seed=args.seed,
            role=role,
            limit=gradient_rows_required,
        )
        for role, values in active_by_role.items()
    }
    objectives = ("mle", "omission", "repetition")
    gradient_replicates = []
    for replicate in range(args.gradient_replicates):
        start = replicate * args.gradient_batch_size
        end = start + args.gradient_batch_size
        gradient_rows = {
            role: values[start:end] for role, values in gradient_pools.items()
        }
        mle_rows = [
            *gradient_rows["omission"][: args.gradient_batch_size // 2],
            *gradient_rows["repetition"][
                : args.gradient_batch_size - args.gradient_batch_size // 2
            ],
        ]
        gradient_values = {}
        gradient_summaries = {}
        for objective, objective_rows in (
            ("mle", mle_rows),
            ("omission", gradient_rows["omission"]),
            ("repetition", gradient_rows["repetition"]),
        ):
            values, summary = objective_gradients(
                model,
                tokenizer,
                objective_rows,
                objective=objective,
                target_margin=args.active_margin,
                maximum_source_tokens=args.maximum_source_tokens,
                maximum_target_tokens=args.maximum_target_tokens,
                device=device,
            )
            gradient_values[objective] = values
            gradient_summaries[objective] = summary
        cosine_matrix = {
            left: {
                right: gradient_cosine(
                    gradient_values[left],
                    gradient_values[right],
                )
                for right in objectives
            }
            for left in objectives
        }
        gradient_replicates.append(
            {
                "replicate": replicate,
                "objectives": gradient_summaries,
                "cosine_similarity": cosine_matrix,
            }
        )
        del gradient_values
    cosine_summary = {
        f"{left}_vs_{right}": scalar_summary(
            [item["cosine_similarity"][left][right] for item in gradient_replicates]
        )
        for left, right in (
            ("mle", "omission"),
            ("mle", "repetition"),
            ("omission", "repetition"),
        )
    }

    args.output_directory.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_directory / "scored.jsonl"
    active_path = args.output_directory / "active.jsonl"
    write_jsonl(scored_path, scored)
    write_jsonl(active_path, active)
    role_margins = {
        role: [
            float(row["chosen_minus_rejected_margin"])
            for row in scored
            if row["risk_role"] == role
        ]
        for role in ("omission", "repetition")
    }
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment": EXPERIMENT,
        "status": "diagnostic-complete-no-training-authorized",
        "direction": "ja-en",
        "purpose": (
            "full-sequence active-risk and gradient-conflict measurement "
            "before any v16 training contract"
        ),
        "dataset": {
            "manifest": record(dataset_manifest_path, root),
            "train": record(train_path, root),
            "selected_rows": len(rows),
            "validation_rows_used": False,
            "protected_rows_used": False,
        },
        "safe_parent": {
            "path": display_path(args.safe_parent, root),
            "model": record(model_path, root),
            "training_manifest": record(parent_manifest_path, root),
        },
        "implementation": record(Path(__file__).resolve(), root),
        "configuration": {
            "seed": args.seed,
            "row_limit": args.row_limit,
            "active_margin": args.active_margin,
            "batch_size": args.batch_size,
            "gradient_batch_size": args.gradient_batch_size,
            "gradient_replicates": args.gradient_replicates,
            "maximum_source_tokens": args.maximum_source_tokens,
            "maximum_target_tokens": args.maximum_target_tokens,
        },
        "counts": {
            "candidate_pairs": len(scored),
            "candidate_roles": dict(
                sorted(Counter(str(row["risk_role"]) for row in scored).items())
            ),
            "active_pairs": len(active),
            "active_roles": dict(
                sorted(Counter(str(row["risk_role"]) for row in active).items())
            ),
            "active_strata": dict(
                sorted(Counter(str(row["v15_stratum"]) for row in active).items())
            ),
        },
        "margins": {
            role: {
                "quantiles": quantiles(values),
                "active": len(active_by_role[role]),
                "active_fraction": len(active_by_role[role]) / len(values),
            }
            for role, values in role_margins.items()
        },
        "gradient_audit": {
            "selection": (
                "disjoint stratum-balanced low-margin active batches per role; "
                "MLE uses an even role mix"
            ),
            "replicates": gradient_replicates,
            "cosine_summary": cosine_summary,
            "optimizer_selected": False,
            "diagnostic_is_not_optimizer_authorization": True,
        },
        "outputs": {
            "scored": {
                **record(scored_path, root),
                "rows": len(scored),
            },
            "active": {
                **record(active_path, root),
                "rows": len(active),
            },
        },
        "distribution_provenance": {
            "all_positive_targets_are_licensed_human_references": all(
                row["positive_target_source"] == "licensed-human-reference"
                for row in scored
            ),
            "all_rows_have_source_license": all(
                bool(row.get("source_license")) for row in scored
            ),
            "all_rows_have_source_provenance": all(
                bool(row.get("source_provenance")) for row in scored
            ),
            "negative_strings_are_positive_targets": False,
            "private_reasoning_traces_used": False,
        },
        "hardware": hardware_name(),
        "operating_system": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": args.device,
        "optimizer_step_executed": False,
        "model_checkpoint_written": False,
        "training_authorized": False,
        "app_change_authorized": False,
        "bundle_replacement_authorized": False,
        "public_upload_authorized": False,
    }
    result_path = args.output_directory / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": display_path(result_path, root),
                "sha256": sha256(result_path),
                "status": result["status"],
                "counts": result["counts"],
                "margins": result["margins"],
                "gradient_audit": result["gradient_audit"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
