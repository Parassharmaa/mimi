#!/usr/bin/env python3
"""Clone a routed Marian pack and bind an exact translation memory into it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_pack", type=Path)
    parser.add_argument("memory", type=Path)
    parser.add_argument("output_pack", type=Path)
    args = parser.parse_args()
    if args.output_pack.exists():
        raise SystemExit(f"refusing to overwrite output pack: {args.output_pack}")
    manifest_path = args.base_pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "mimi-mlx-marian-moe-v1":
        raise SystemExit("base pack is not a routed Marian MoE")
    memory = json.loads(args.memory.read_text(encoding="utf-8"))
    if memory.get("schemaVersion") != 1 or memory.get("doesNotAuthorizeAppIntegration") is not True:
        raise SystemExit("unsupported or promotion-authorized memory artifact")

    args.output_pack.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-cR", str(args.base_pack), str(args.output_pack)],
        check=True,
    )
    relative_memory = Path("memory/exact-translation-memory.json")
    memory_output = args.output_pack / relative_memory
    memory_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.memory, memory_output)
    manifest["files"][str(relative_memory)] = file_record(memory_output)
    manifest["translationMemory"] = {
        "path": str(relative_memory),
        "schemaVersion": 1,
        "normalization": memory["normalization"],
        "sourceLicense": memory["sourceLicense"],
        "trainingDataSHA256": memory["trainingDataSHA256"],
        "auditSHA256": memory["auditSHA256"],
        "entries": sum(len(values) for values in memory["entries"].values()),
        "maximumSourceCharacters": max(
            len(source) for values in memory["entries"].values() for source in values
        ),
        "maximumTargetCharacters": max(
            len(target) for values in memory["entries"].values() for target in values.values()
        ),
        "lookup": "exact normalized source before neural routing",
    }
    manifest["qualityStatus"] = "development-exact-memory-gates-passed-private-claim-suite-pending"
    (args.output_pack / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_bytes = sum(path.stat().st_size for path in args.output_pack.rglob("*") if path.is_file())
    print(
        json.dumps(
            {
                "output": str(args.output_pack),
                "modelBytes": model_bytes,
                "manifestSHA256": sha256(args.output_pack / "manifest.json"),
                "memory": file_record(memory_output),
                "doesNotAuthorizeAppIntegration": manifest["doesNotAuthorizeAppIntegration"],
            }
        )
    )


if __name__ == "__main__":
    main()
