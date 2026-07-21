#!/usr/bin/env python3
"""Package Mimi's two generalists plus two source-routed Marian experts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ENGINE_FILES = (
    "manifest.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def validate_engine(source: Path, direction: str) -> dict:
    missing = [name for name in ENGINE_FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"{source} is missing: {', '.join(missing)}")
    manifest = load_object(source / "manifest.json")
    if (
        manifest.get("format") != "mimi-mlx-marian-v1"
        or manifest.get("direction") != direction
        or manifest.get("bits") != 4
        or manifest.get("group_size") != 64
    ):
        raise SystemExit(f"unsupported engine manifest: {source}")
    for name in ENGINE_FILES[1:]:
        record = manifest.get("files", {}).get(name)
        expected = {
            "bytes": (source / name).stat().st_size,
            "sha256": digest(source / name),
        }
        if record != expected:
            raise SystemExit(f"engine file authentication failed: {source / name}")
    return manifest


def validate_router(source: Path, direction: str) -> None:
    router = load_object(source)
    if (
        router.get("schemaVersion") != 1
        or router.get("format") != "mimi-source-expert-router-v1"
        or router.get("direction") != direction
    ):
        raise SystemExit(f"unsupported expert router: {source}")


def generalist_lineage_record(source: Path, expected_model_sha256: str) -> dict:
    if source.is_dir():
        model_path = source / "model.safetensors"
        manifest_path = source / "mimi_training_manifest.json"
        if not model_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"direct lineage is incomplete: {source}")
        if digest(model_path) != expected_model_sha256:
            raise SystemExit(f"direct lineage output differs: {source}")
        return {
            "kind": "direct-training-checkpoint",
            "path": str(manifest_path),
            "sha256": digest(manifest_path),
            "modelSha256": expected_model_sha256,
        }
    manifest = load_object(source)
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("model_sha256") != expected_model_sha256
    ):
        raise SystemExit(f"generalist lineage output differs: {source}")
    operation = manifest.get("operation")
    if operation not in {
        "linear-checkpoint-interpolation",
        "arithmetic-mean-of-best-adjacent-full-precision-checkpoints",
    }:
        raise SystemExit(f"unsupported generalist lineage operation: {operation}")
    return {
        "kind": operation,
        "path": str(source),
        "sha256": digest(source),
        "modelSha256": expected_model_sha256,
    }


def copy_engine(source: Path, destination: Path, manifest: dict) -> None:
    destination.mkdir(parents=True)
    for name in ENGINE_FILES:
        shutil.copy2(source / name, destination / name)
    copied_manifest = dict(manifest)
    copied_manifest["files"] = {
        name: {
            "bytes": (destination / name).stat().st_size,
            "sha256": digest(destination / name),
        }
        for name in ENGINE_FILES[1:]
    }
    (destination / "manifest.json").write_text(
        json.dumps(copied_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generalist_pack", type=Path)
    parser.add_argument("en_ja_expert", type=Path)
    parser.add_argument("en_ja_router", type=Path)
    parser.add_argument("ja_en_expert", type=Path)
    parser.add_argument("ja_en_router", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--en-ja-generalist-lineage", type=Path)
    parser.add_argument("--ja-en-generalist-lineage", type=Path)
    parser.add_argument("--formal-en-ja-lineage", type=Path)
    parser.add_argument("--legal-ja-en-lineage", type=Path)
    args = parser.parse_args()

    lineage_inputs = {
        "generalist-en-ja": args.en_ja_generalist_lineage,
        "generalist-ja-en": args.ja_en_generalist_lineage,
        "formal-en-ja": args.formal_en_ja_lineage,
        "legal-ja-en": args.legal_ja_en_lineage,
    }
    lineage_values = tuple(lineage_inputs.values())
    if any(lineage_values) and not all(lineage_values):
        raise SystemExit("all four engine lineage inputs must be supplied together")

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty bundle: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    inputs = {
        "generalist-en-ja": (args.generalist_pack / "en-ja", "en-ja"),
        "generalist-ja-en": (args.generalist_pack / "ja-en", "ja-en"),
        "formal-en-ja": (args.en_ja_expert, "en-ja"),
        "legal-ja-en": (args.ja_en_expert, "ja-en"),
    }
    manifests = {
        name: validate_engine(source, direction)
        for name, (source, direction) in inputs.items()
    }
    validate_router(args.en_ja_router, "en-ja")
    validate_router(args.ja_en_router, "ja-en")
    engine_lineage = {}
    if all(lineage_values):
        engine_lineage = {
            name: generalist_lineage_record(
                source,
                str(manifests[name]["source_weights_sha256"]),
            )
            for name, source in lineage_inputs.items()
        }
        for name in ("formal-en-ja", "legal-ja-en"):
            declared_manifest_sha = (
                manifests[name].get("training_data", {}).get(
                    "training_manifest_sha256"
                )
            )
            if engine_lineage[name]["sha256"] != declared_manifest_sha:
                raise SystemExit(f"expert training-manifest lineage differs: {name}")

    for name, (source, _) in inputs.items():
        copy_engine(source, args.output / "engines" / name, manifests[name])
    router_directory = args.output / "routers"
    router_directory.mkdir()
    shutil.copy2(args.en_ja_router, router_directory / "formal-en-ja.json")
    shutil.copy2(args.ja_en_router, router_directory / "legal-ja-en.json")

    files = {
        str(item.relative_to(args.output)): {
            "bytes": item.stat().st_size,
            "sha256": digest(item),
        }
        for item in sorted(args.output.rglob("*"))
        if item.is_file()
    }
    required_attributions = []
    seen_attributions = set()
    distribution_statuses = set()
    for manifest in manifests.values():
        training_data = manifest.get("training_data") or {}
        if not isinstance(training_data, dict):
            raise SystemExit("engine training-data provenance must be an object or null")
        distribution_statuses.add(training_data.get("distribution_status"))
        for attribution in training_data.get("required_attributions", []):
            key = json.dumps(attribution, ensure_ascii=False, sort_keys=True)
            if key not in seen_attributions:
                seen_attributions.add(key)
                required_attributions.append(attribution)
    root_manifest = {
        "format": "mimi-mlx-marian-moe-v1",
        "interface": "bidirectional-en-ja",
        "quantization": {"bits": 4, "groupSize": 64},
        "generalists": {
            "en-ja": "engines/generalist-en-ja",
            "ja-en": "engines/generalist-ja-en",
        },
        "experts": {
            "en-ja": {
                "engine": "engines/formal-en-ja",
                "router": "routers/formal-en-ja.json",
            },
            "ja-en": {
                "engine": "engines/legal-ja-en",
                "router": "routers/legal-ja-en.json",
            },
        },
        "routing": {
            "inputs": "source-text-only",
            "implementation": "dependency-free TF-IDF character n-grams plus ridge",
            "defaultOnRouterFailure": "generalist",
        },
        "license": "CC-BY-SA-4.0",
        "requiredAttributions": required_attributions,
        "distributionStatus": (
            "blocked-pending-share-alike-and-attribution-review"
            if "blocked-pending-share-alike-and-attribution-review"
            in distribution_statuses
            else "research-candidate-not-approved-for-distribution"
        ),
        "qualityStatus": "development-gates-passed-private-claim-suite-pending",
        "doesNotAuthorizeAppIntegration": True,
        "engines": {
            name: {
                "direction": manifests[name]["direction"],
                "sourceRevision": manifests[name]["source_revision"],
                "sourceWeightsSha256": manifests[name]["source_weights_sha256"],
                "trainingData": manifests[name].get("training_data"),
                "releaseLineage": engine_lineage.get(name),
            }
            for name in sorted(manifests)
        },
        "files": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "bytes": directory_bytes(args.output),
                "sha256": digest(args.output / "manifest.json"),
                "distributionStatus": root_manifest["distributionStatus"],
                "doesNotAuthorizeAppIntegration": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
