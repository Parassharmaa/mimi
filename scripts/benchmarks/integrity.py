"""Receipts bind completed run artifacts; summaries can be regenerated separately."""
import json
from pathlib import Path
from prepare_suite import sha256


def seal_run(root: Path) -> None:
    files = {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*"))
             if path.is_file() and path.name not in {"receipt.json", "summary.json", "summary.md"}}
    with (root / "receipt.json").open("x") as output:
        output.write(json.dumps({"schemaVersion": 1, "files": files}, indent=2) + "\n")


def verify_run(root: Path) -> None:
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt.get("schemaVersion") != 1 or not receipt.get("files"): raise ValueError("Invalid run receipt")
    for name, expected in receipt["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError("Run artifact hash mismatch: " + name)
    if not {"metadata.json", "attempts.jsonl"}.issubset(receipt["files"]): raise ValueError("Receipt lacks required run artifacts")
