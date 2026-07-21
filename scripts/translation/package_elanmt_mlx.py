#!/usr/bin/env python3
"""Create the minimal two-direction Mimi MLX translation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


FILES = ("manifest.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("en_ja", type=Path)
    parser.add_argument("ja_en", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty bundle: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for direction, source in (("en-ja", args.en_ja), ("ja-en", args.ja_en)):
        missing = [name for name in FILES if not (source / name).is_file()]
        if missing:
            raise SystemExit(f"{source} is missing: {', '.join(missing)}")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("direction") != direction:
            raise SystemExit(f"direction mismatch in {source}")
        destination = args.output / direction
        destination.mkdir()
        for name in FILES:
            shutil.copy2(source / name, destination / name)
        manifest["files"] = {
            name: {
                "bytes": (destination / name).stat().st_size,
                "sha256": digest(destination / name),
            }
            for name in FILES
            if name != "manifest.json"
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifests[direction] = manifest

    quantizations = {
        (manifest.get("bits"), manifest.get("group_size"))
        for manifest in manifests.values()
    }
    if len(quantizations) != 1:
        raise SystemExit("direction manifests must use the same quantization")
    bits, group_size = quantizations.pop()
    if bits not in {4, 6, 8} or group_size not in {32, 64, 128}:
        raise SystemExit(
            f"unsupported shipping quantization: bits={bits} group_size={group_size}"
        )

    files = {
        str(item.relative_to(args.output)): {
            "bytes": item.stat().st_size,
            "sha256": digest(item),
        }
        for item in sorted(args.output.rglob("*"))
        if item.is_file()
    }
    training_data = {
        direction: manifest["training_data"]
        for direction, manifest in manifests.items()
        if manifest.get("training_data")
    }
    required_attributions = []
    seen_attributions = set()
    for provenance in training_data.values():
        for attribution in provenance.get("required_attributions", []):
            key = json.dumps(attribution, ensure_ascii=False, sort_keys=True)
            if key not in seen_attributions:
                seen_attributions.add(key)
                required_attributions.append(attribution)
    distribution_statuses = {
        provenance.get("distribution_status") for provenance in training_data.values()
    }
    root_manifest = {
        "format": "mimi-mlx-marian-pair-v1",
        "interface": "bidirectional-en-ja",
        "engines": ["en-ja", "ja-en"],
        "quantization": {"bits": bits, "group_size": group_size},
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "ElanMT by ELAN MITSUA Project / Abstract Engine",
        "source_revisions": {
            direction: manifest["source_revision"]
            for direction, manifest in manifests.items()
        },
        "training_data": training_data,
        "required_attributions": required_attributions,
        "distribution_status": (
            "blocked-pending-share-alike-and-attribution-review"
            if "blocked-pending-share-alike-and-attribution-review"
            in distribution_statuses
            else "provenance-incomplete-not-approved-for-distribution"
            if "provenance-incomplete-not-approved-for-distribution"
            in distribution_statuses
            else "research-candidate-not-approved-for-distribution"
            if training_data
            else "provenance-incomplete-not-approved-for-distribution"
        ),
        "files": files,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"bundle": str(args.output), "bytes": directory_bytes(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
