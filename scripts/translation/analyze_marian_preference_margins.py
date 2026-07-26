#!/usr/bin/env python3
"""Report per-pair preference margins for a Marian candidate and frozen parent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

from train_marian_dqo import PreferenceCollator, pair_logps


def sha256(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preferences", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--max-source-tokens", type=int, default=192)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    rows = [
        json.loads(line)
        for line in args.preferences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("preference input is empty")
    device = torch.device(args.device)
    tokenizer = MarianTokenizer.from_pretrained(args.parent)
    candidate = MarianMTModel.from_pretrained(args.candidate).to(device)
    parent = MarianMTModel.from_pretrained(args.parent).to(device)
    candidate.eval()
    parent.eval()
    candidate.requires_grad_(False)
    parent.requires_grad_(False)
    collator = PreferenceCollator(
        tokenizer, args.max_source_tokens, args.max_target_tokens
    )
    details: list[dict] = []
    with torch.inference_mode():
        for row in rows:
            pair = collator([row])
            chosen = {
                key: value.to(device) for key, value in pair["chosen"].items()
            }
            rejected = {
                key: value.to(device) for key, value in pair["rejected"].items()
            }
            candidate_chosen, candidate_rejected = pair_logps(
                candidate, chosen, rejected
            )
            parent_chosen, parent_rejected = pair_logps(
                parent, chosen, rejected
            )
            policy_margin = float((candidate_chosen - candidate_rejected).item())
            parent_margin = float((parent_chosen - parent_rejected).item())
            details.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "domain": row.get("domain", "unknown"),
                    "policy_margin": policy_margin,
                    "parent_margin": parent_margin,
                    "relative_margin": policy_margin - parent_margin,
                    "improved": policy_margin > parent_margin,
                    "source": row["source"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
            )
    margins = [value["relative_margin"] for value in details]
    payload = {
        "schema_version": 1,
        "preferences": {
            "path": str(args.preferences),
            "sha256": sha256(args.preferences),
            "pairs": len(rows),
        },
        "candidate": {
            "path": str(args.candidate),
            "model_sha256": sha256(args.candidate / "model.safetensors"),
        },
        "parent": {
            "path": str(args.parent),
            "model_sha256": sha256(args.parent / "model.safetensors"),
        },
        "summary": {
            "pairs": len(details),
            "improved": sum(value["improved"] for value in details),
            "relative_pair_accuracy": sum(value["improved"] for value in details)
            / len(details),
            "mean_relative_margin": sum(margins) / len(margins),
            "minimum_relative_margin": min(margins),
            "maximum_relative_margin": max(margins),
        },
        "pairs": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
