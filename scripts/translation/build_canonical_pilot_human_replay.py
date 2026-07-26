#!/usr/bin/env python3
"""Freeze licensed human replay rows from a canonical source pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ORIGINS = {
    "kftt": "human-kftt-replay",
    "alt": "human-alt-parallel",
    "jlt": "finalized-japanese-law-translation",
    "tatoeba": "human-tatoeba-bidirectional-agreement-filtered",
    "ui": "mimi-shipped-ui-pair",
    "alt-document": "human-alt-document-window",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_seeds", type=Path)
    parser.add_argument("pilot_contract", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    contract = json.loads(args.pilot_contract.read_text(encoding="utf-8"))
    if (
        contract.get("outputs", {}).get("pilotSeeds", {}).get("sha256")
        != sha256(args.pilot_seeds)
        or contract.get("selectionUsesPreviousTeacherOrJudgeOutputs") is not False
    ):
        raise SystemExit("pilot seeds are not bound to the canonical contract")

    output_rows = []
    seen: set[str] = set()
    for seed in rows(args.pilot_seeds):
        source_id = str(seed.get("id", ""))
        corpus = str(seed.get("source_corpus", ""))
        if not source_id or source_id in seen or corpus not in ORIGINS:
            raise SystemExit(f"invalid canonical replay seed: {source_id}")
        seen.add(source_id)
        source = str(seed.get("source", "")).strip()
        target = str(seed.get("reference_translation", "")).strip()
        license_name = str(seed.get("license", "")).strip()
        provenance = str(seed.get("provenance", "")).strip()
        if not source or not target or not license_name or not provenance:
            raise SystemExit(f"incomplete canonical replay seed: {source_id}")
        output_rows.append({
            "id": f"canonical-pilot-human-reference:{source_id}",
            "source_id": source_id,
            "source_language": seed["source_language"],
            "target_language": seed["target_language"],
            "source": source,
            "target": target,
            "domain": seed["domain"],
            "origin": ORIGINS[corpus],
            "source_license": license_name,
            "source_provenance": provenance,
            "target_license": license_name,
            "target_provenance": seed.get("reference_provenance") or provenance,
            "promotion_eligible": True,
        })
    output_rows.sort(key=lambda row: row["id"])
    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    valid_path = args.output / "valid.jsonl"
    write_jsonl(train_path, output_rows)
    write_jsonl(valid_path, [])
    manifest = {
        "schema_version": 1,
        "purpose": "distribution-matched licensed human replay for canonical distillation",
        "promotion_eligible": True,
        "private_reasoning_traces_used": False,
        "input": {
            "pilotSeeds": {
                "path": str(args.pilot_seeds),
                "sha256": sha256(args.pilot_seeds),
            },
            "pilotContract": {
                "path": str(args.pilot_contract),
                "sha256": sha256(args.pilot_contract),
            },
        },
        "counts": {
            "train": len(output_rows),
            "valid": 0,
            "directions": dict(sorted(Counter(
                f"{row['source_language']}->{row['target_language']}"
                for row in output_rows
            ).items())),
            "origins": dict(sorted(Counter(
                row["origin"] for row in output_rows
            ).items())),
            "licenses": dict(sorted(Counter(
                row["source_license"] for row in output_rows
            ).items())),
        },
        "outputs": {
            "train": {"path": str(train_path), "sha256": sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": sha256(valid_path)},
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(output_rows),
        "train_sha256": sha256(train_path),
        "manifest": str(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
