#!/usr/bin/env python3
"""Score source/backtranslation mutual entailment with a pinned NLI model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_REVISION = "84ccdcb62589067b29b930cff8e362e75ba0dd15"
MODEL_FILES = [
    "config.json", "model.safetensors", "spm.model", "special_tokens_map.json",
    "sentencepiece.bpe.model", "tokenizer.json", "tokenizer_config.json",
]


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def indexed_report(path: Path) -> tuple[dict, dict[str, dict]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, dict] = {}
    for row in report.get("results", []):
        identifier = str(row.get("caseID", ""))
        if not identifier or identifier in output:
            raise SystemExit(f"backtranslation report has missing or duplicate ID: {path}")
        output[identifier] = row
    return report, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("forward_suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backtranslation-report", type=Path, action="append", required=True)
    parser.add_argument("--backtranslation-name", action="append", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-license", default="Apache-2.0")
    parser.add_argument("--hf-home", type=Path, default=Path("Research/translation/models/hf-cache"))
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() and args.output.stat().st_size:
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    if len(args.backtranslation_name) != len(args.backtranslation_report):
        raise SystemExit("provide one unique backtranslation name per report")
    if len(set(args.backtranslation_name)) != len(args.backtranslation_name):
        raise SystemExit("backtranslation names must be unique")
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")

    suite = rows(args.forward_suite)
    source_by_id = {str(row["id"]): str(row["source"]) for row in suite}
    if len(source_by_id) != len(suite):
        raise SystemExit("forward suite has duplicate IDs")
    reports: dict[str, dict] = {}
    indexed: dict[str, dict[str, dict]] = {}
    for name, path in zip(
        args.backtranslation_name, args.backtranslation_report, strict=True
    ):
        reports[name], indexed[name] = indexed_report(path)
        if set(indexed[name]) != set(source_by_id):
            raise SystemExit(f"backtranslation report does not cover the exact suite: {name}")

    args.hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    snapshot = Path(snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.hf_home,
        allow_patterns=MODEL_FILES,
    ))
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    device = torch.device(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot).to(device).eval()
    labels = {str(value).casefold(): int(key) for key, value in model.config.id2label.items()}
    if set(labels) != {"contradiction", "entailment", "neutral"}:
        raise SystemExit(f"unexpected NLI labels: {model.config.id2label}")

    tasks: list[tuple[str, str, str, str]] = []
    for identifier, source in source_by_id.items():
        for name in args.backtranslation_name:
            backtranslation = str(indexed[name][identifier].get("hypothesis", "")).strip()
            if not backtranslation:
                raise SystemExit(f"empty backtranslation: {name}/{identifier}")
            tasks.append((identifier, name, source, backtranslation))

    scored: dict[str, dict[str, dict]] = {identifier: {} for identifier in source_by_id}
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start:start + args.batch_size]
            hypotheses = [text for value in batch for text in (value[3], value[2])]
            reverse_premises = [text for value in batch for text in (value[2], value[3])]
            encoded = tokenizer(
                reverse_premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)
            probabilities = model(**encoded).logits.float().softmax(dim=-1).cpu()
            for index, (identifier, name, source, backtranslation) in enumerate(batch):
                forward = probabilities[index * 2]
                reverse = probabilities[index * 2 + 1]
                scored[identifier][name] = {
                    "source": source,
                    "backtranslation": backtranslation,
                    "source_entails_backtranslation": float(forward[labels["entailment"]]),
                    "backtranslation_entails_source": float(reverse[labels["entailment"]]),
                    "source_contradicts_backtranslation": float(forward[labels["contradiction"]]),
                    "backtranslation_contradicts_source": float(reverse[labels["contradiction"]]),
                }

    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "provisional synthetic-data semantic filter; never evaluation evidence",
        "claimEligible": False,
        "model": args.model,
        "modelRevision": args.revision,
        "modelLicense": args.model_license,
        "modelBytes": directory_bytes(snapshot),
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "inputs": {
            "forwardSuite": {"path": str(args.forward_suite.resolve()), "sha256": sha256(args.forward_suite)},
            "backtranslationReports": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "engine": reports[name].get("engine"),
                }
                for name, path in zip(
                    args.backtranslation_name, args.backtranslation_report, strict=True
                )
            },
        },
        "results": [
            {"caseID": identifier, "backtranslations": scored[identifier]}
            for identifier in source_by_id
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tasks)} mutual-entailment pairs to {args.output}")


if __name__ == "__main__":
    main()
