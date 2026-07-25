#!/usr/bin/env python3
"""Compress a full Marian specialist as low-rank deltas around a shared base."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


COPY_FILES = (
    "config.json",
    "generation_config.json",
    "source.spm",
    "target.spm",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def approximate_delta(
    delta: torch.Tensor,
    *,
    rank: int,
    niter: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    energy = float(torch.sum(delta.float() ** 2))
    if delta.ndim != 2 or min(delta.shape) <= rank or energy == 0:
        exact = delta.float()
        return exact, {"delta": delta.to(torch.float16)}, energy
    left, singular_values, right = torch.svd_lowrank(
        delta.float(),
        q=rank,
        niter=niter,
    )
    factor_a = left * singular_values.unsqueeze(0)
    factor_b = right.transpose(0, 1)
    approximation = factor_a @ factor_b
    captured = float(torch.sum(approximation**2))
    return (
        approximation,
        {
            "factor_a": factor_a.to(torch.float16).contiguous(),
            "factor_b": factor_b.to(torch.float16).contiguous(),
        },
        captured,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("expert_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.rank < 1 or args.power_iterations < 0:
        raise SystemExit("rank must be positive and power iterations non-negative")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory must be absent or empty: {args.output}")

    base_weights = args.base_checkpoint / "model.safetensors"
    expert_weights = args.expert_checkpoint / "model.safetensors"
    for path in (base_weights, expert_weights):
        if not path.is_file():
            raise SystemExit(f"missing checkpoint weights: {path}")
    base = load_file(base_weights, device="cpu")
    expert = load_file(expert_weights, device="cpu")
    if base.keys() != expert.keys():
        raise SystemExit("base and expert state dictionaries differ")
    for key in base:
        if base[key].shape != expert[key].shape:
            raise SystemExit(f"base and expert tensor shapes differ: {key}")

    torch.manual_seed(args.seed)
    merged: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}
    total_energy = 0.0
    captured_energy = 0.0
    low_rank_tensors = 0
    exact_tensors = 0
    for key in sorted(base):
        delta = expert[key].float() - base[key].float()
        approximation, factors, captured = approximate_delta(
            delta,
            rank=args.rank,
            niter=args.power_iterations,
        )
        total_energy += float(torch.sum(delta**2))
        captured_energy += captured
        merged[key] = (base[key].float() + approximation).to(base[key].dtype)
        if "factor_a" in factors:
            low_rank_tensors += 1
            adapter[f"{key}.factor_a"] = factors["factor_a"]
            adapter[f"{key}.factor_b"] = factors["factor_b"]
        else:
            exact_tensors += 1
            adapter[f"{key}.delta"] = factors["delta"]

    args.output.mkdir(parents=True, exist_ok=True)
    merged_path = args.output / "model.safetensors"
    adapter_path = args.output / "expert_delta.safetensors"
    save_file(adapter, adapter_path)
    save_file(merged, merged_path)
    for name in COPY_FILES:
        source = args.base_checkpoint / name
        if source.is_file():
            shutil.copy2(source, args.output / name)

    expert_training_manifest = (
        args.expert_checkpoint / "mimi_training_manifest.json"
    )
    training_manifest = (
        json.loads(expert_training_manifest.read_text(encoding="utf-8"))
        if expert_training_manifest.is_file()
        else {}
    )
    training_manifest["posthoc_low_rank_delta"] = {
        "schemaVersion": 1,
        "baseCheckpoint": str(args.base_checkpoint.resolve()),
        "baseWeightsSha256": sha256(base_weights),
        "expertCheckpoint": str(args.expert_checkpoint.resolve()),
        "expertWeightsSha256": sha256(expert_weights),
        "rank": args.rank,
        "powerIterations": args.power_iterations,
        "seed": args.seed,
        "mergedWeightsSha256": sha256(merged_path),
        "adapterSha256": sha256(adapter_path),
        "promotionEligible": False,
    }
    (args.output / "mimi_training_manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "purpose": "post-hoc shared-backbone Marian expert-delta feasibility",
        "promotionEligible": False,
        "base": {
            "path": str(args.base_checkpoint.resolve()),
            "weightsSha256": sha256(base_weights),
        },
        "expert": {
            "path": str(args.expert_checkpoint.resolve()),
            "weightsSha256": sha256(expert_weights),
        },
        "approximation": {
            "algorithm": "torch randomized truncated SVD",
            "rank": args.rank,
            "powerIterations": args.power_iterations,
            "seed": args.seed,
            "lowRankTensors": low_rank_tensors,
            "exactTensors": exact_tensors,
            "totalDeltaSquaredFrobeniusNorm": total_energy,
            "capturedSquaredFrobeniusNorm": captured_energy,
            "capturedEnergyFraction": (
                captured_energy / total_energy if total_energy else 1.0
            ),
        },
        "artifacts": {
            "mergedCheckpoint": {
                "path": str(merged_path.resolve()),
                "bytes": merged_path.stat().st_size,
                "sha256": sha256(merged_path),
            },
            "expertDelta": {
                "path": str(adapter_path.resolve()),
                "bytes": adapter_path.stat().st_size,
                "sha256": sha256(adapter_path),
                "dtype": "float16",
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "warning": (
            "The merged checkpoint is a quality-screening artifact. "
            "A native MLX/Swift adapter runtime and independent evaluation are "
            "required before this representation can reduce the shipped bundle."
        ),
    }
    report_path = args.output / "mimi_low_rank_delta_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "directoryBytes": directory_bytes(args.output),
                "adapterBytes": adapter_path.stat().st_size,
                "capturedEnergyFraction": report["approximation"][
                    "capturedEnergyFraction"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
