#!/usr/bin/env python3
"""Benchmark the pinned MIT-licensed SMaLL-100 checkpoint on Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import M2M100ForConditionalGeneration


DEFAULT_REPOSITORY = "alirezamsh/small100"
DEFAULT_REVISION = "8ab680e26a596d2e3d2d2d17ae0f68df1037328c"
LANGUAGE_CODES = {"en-US": "en", "ja-JP": "ja"}
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenization_small100.py",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_suite(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("benchmark suite is empty")
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise SystemExit("benchmark suite contains duplicate IDs")
    for row in rows:
        for language in (row["sourceLanguage"], row["targetLanguage"]):
            if language not in LANGUAGE_CODES:
                raise SystemExit(f"unsupported benchmark language: {language}")
    return rows


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def hardware_name() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def load_tokenizer(snapshot: Path):
    module_path = snapshot / "tokenization_small100.py"
    spec = importlib.util.spec_from_file_location("mimi_small100_tokenizer", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load pinned SMaLL-100 tokenizer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer_class = getattr(module, "SMALL100Tokenizer", None)
    if tokenizer_class is None:
        raise SystemExit("pinned SMaLL-100 tokenizer class is missing")
    return tokenizer_class.from_pretrained(snapshot)


def translate(
    model,
    tokenizer,
    row: dict,
    device: torch.device,
    max_tokens: int,
    num_beams: int,
) -> tuple[str, list[int]]:
    tokenizer.tgt_lang = LANGUAGE_CODES[row["targetLanguage"]]
    encoded = tokenizer(row["source"], return_tensors="pt", truncation=True).to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            do_sample=False,
            num_beams=num_beams,
            use_cache=True,
            max_new_tokens=max_tokens,
        )
    sync(device)
    output_ids = output[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip(), output_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--num-beams", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--hf-home", type=Path, default=Path("Research/translation/models/hf-cache")
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()

    if args.warm_runs < 0:
        raise SystemExit("warm-runs must be non-negative")
    if args.max_tokens < 1:
        raise SystemExit("max-tokens must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; pass --device cpu for a CPU-only run")
    suite = load_suite(args.suite)
    device = torch.device(args.device)

    preparation_started = time.perf_counter()
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
    missing = [name for name in MODEL_FILES if not (snapshot / name).is_file()]
    if missing:
        raise SystemExit(f"pinned SMaLL-100 snapshot is incomplete: {missing}")
    tokenizer = load_tokenizer(snapshot)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    model = M2M100ForConditionalGeneration.from_pretrained(
        snapshot,
        dtype=dtype,
        use_safetensors=True,
    )
    adapter_record = None
    if args.adapter is not None:
        from peft import PeftModel

        adapter_manifest = args.adapter / "mimi_training_manifest.json"
        if not adapter_manifest.is_file():
            raise SystemExit("SMaLL-100 adapter lacks its training manifest")
        metadata = json.loads(adapter_manifest.read_text(encoding="utf-8"))
        base_metadata = metadata.get("base_model", {})
        if (
            base_metadata.get("repository") != args.repository
            or base_metadata.get("revision") != args.revision
            or base_metadata.get("model_sha256") != sha256(snapshot / "model.safetensors")
        ):
            raise SystemExit("SMaLL-100 adapter base identity differs")
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        adapter_record = {
            "path": str(args.adapter),
            "manifestSha256": sha256(adapter_manifest),
            "bytes": directory_bytes(args.adapter),
        }
    model = model.to(device).eval()
    sync(device)
    preparation_seconds = time.perf_counter() - preparation_started

    results = []
    for row in suite:
        started = time.perf_counter()
        hypothesis, output_ids = translate(
            model, tokenizer, row, device, args.max_tokens, args.num_beams
        )
        first_latency = time.perf_counter() - started
        warm_latencies = []
        for _ in range(args.warm_runs):
            warm_started = time.perf_counter()
            translate(model, tokenizer, row, device, args.max_tokens, args.num_beams)
            warm_latencies.append(time.perf_counter() - warm_started)
        results.append(
            {
                "caseID": row["id"],
                "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"],
                "domain": row["domain"],
                "source": row["source"],
                "references": row["references"],
                "claimEligible": bool(row["claimEligible"]),
                "hypothesis": hypothesis,
                "outputTokenIDs": output_ids,
                "latencySeconds": first_latency,
                "warmLatencySeconds": warm_latencies,
            }
        )

    files = {
        name: {
            "bytes": (snapshot / name).stat().st_size,
            "sha256": sha256(snapshot / name),
        }
        for name in MODEL_FILES
    }
    report = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": (
            f"transformers-small100:{args.repository}@{args.revision[:12]}:"
            f"{args.device}:beam{args.num_beams}:"
            f"{'base' if args.adapter is None else args.adapter.name}"
        ),
        "license": "MIT",
        "hardware": hardware_name(),
        "operatingSystem": platform.platform(),
        "preparationSeconds": preparation_seconds,
        "peakResidentBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "modelBytes": directory_bytes(snapshot)
        + (directory_bytes(args.adapter) if args.adapter is not None else 0),
        "physicalModelCount": 1,
        "modelRevision": f"huggingface-revision:{args.revision}",
        "files": files,
        "adapter": adapter_record,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(results)} cases to {args.output}")


if __name__ == "__main__":
    main()
