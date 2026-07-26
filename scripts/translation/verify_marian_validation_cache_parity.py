#!/usr/bin/env python3
"""Verify exact cached/uncached Marian validation parity on a frozen selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer

from train_bidirectional_marian import (
    Collator,
    Rows,
    evaluate,
    load_rows,
    package_versions,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite cache parity report: {args.output}")
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    rows = load_rows(args.selection)
    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).to(device)
    collator = Collator(
        tokenizer,
        args.max_source_tokens,
        args.max_target_tokens,
    )
    loader = DataLoader(
        Rows(rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    common = {
        "model": {
            "path": str(args.model),
            "weights_sha256": sha256(args.model / "model.safetensors"),
        },
        "selection": {
            "path": str(args.selection),
            "sha256": sha256(args.selection),
            "rows": len(rows),
        },
        "batch_size": args.batch_size,
        "max_source_tokens": args.max_source_tokens,
        "max_target_tokens": args.max_target_tokens,
    }
    uncached = evaluate(
        model,
        tokenizer,
        loader,
        rows,
        device,
        args.max_target_tokens,
        loss_use_cache=False,
        generation_use_cache=False,
        return_generated_token_ids=True,
    )
    split_policy = evaluate(
        model,
        tokenizer,
        loader,
        rows,
        device,
        args.max_target_tokens,
        loss_use_cache=False,
        generation_use_cache=True,
        return_generated_token_ids=True,
    )
    uncached_tokens = uncached.pop("generated_token_ids")
    split_tokens = split_policy.pop("generated_token_ids")
    exact_tokens = uncached_tokens == split_tokens
    result = {
        "schema_version": 1,
        "experiment": "shared-bidirectional-v18-validation-cache-parity",
        "verifier": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__)),
        },
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": (
            "cached-generation-parity-passed"
            if exact_tokens
            and uncached["loss"] == split_policy["loss"]
            and uncached["macro_direction_chrf_pp"]
            == split_policy["macro_direction_chrf_pp"]
            else "cached-generation-parity-failed"
        ),
        **common,
        "hardware": platform.platform(),
        "runtime": package_versions(),
        "uncached": {
            "policy": {"loss_forward": False, "greedy_generation": False},
            "metrics": uncached,
            "generated_token_ids_sha256": digest_json(uncached_tokens),
        },
        "split_cache_policy": {
            "policy": {"loss_forward": False, "greedy_generation": True},
            "metrics": split_policy,
            "generated_token_ids_sha256": digest_json(split_tokens),
        },
        "comparison": {
            "exact_generated_token_ids": exact_tokens,
            "loss_delta": split_policy["loss"] - uncached["loss"],
            "macro_direction_chrf_pp_delta": (
                split_policy["macro_direction_chrf_pp"]
                - uncached["macro_direction_chrf_pp"]
            ),
            "direction_chrf_pp_delta": {
                direction: (
                    split_policy["directions"][direction]["chrf_pp"]
                    - uncached["directions"][direction]["chrf_pp"]
                )
                for direction in ("en-ja", "ja-en")
            },
        },
        "promotion_evidence": False,
        "training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "cached-generation-parity-passed":
        raise SystemExit("cached validation generation changed the frozen selector")


if __name__ == "__main__":
    main()
