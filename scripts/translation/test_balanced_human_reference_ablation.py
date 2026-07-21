#!/usr/bin/env python3
"""Offline contract for the all-human balanced hard-source training arm."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ALLOWED_LICENSES = [
    "Apache-2.0", "CC-BY-2.0-FR", "CC-BY-4.0", "CC-BY-SA-3.0",
    "CC0-1.0", "MIT", "project-owned",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-balanced-human-test-") as temporary:
        root = Path(temporary)
        base = root / "base"
        suite = root / "suite.jsonl"
        protected = root / "protected.jsonl"
        output = root / "output"
        write_jsonl(base / "train.jsonl", [{
            "id": "base-train", "source": "Existing sentence.", "target": "既存文です。",
            "source_language": "en-US", "target_language": "ja-JP",
            "source_license": "CC-BY-SA-3.0", "origin": "human-base",
        }])
        write_jsonl(base / "valid.jsonl", [{
            "id": "base-valid", "source": "Validation sentence.", "target": "検証文です。",
            "source_language": "en-US", "target_language": "ja-JP",
            "source_license": "CC-BY-SA-3.0", "origin": "human-base",
        }])
        write_jsonl(protected, [{
            "id": "heldout", "source": "A protected example.", "references": ["保護例です。"],
        }])
        suite_rows = [{
            "id": f"suite-{index}", "sourceLanguage": "en-US", "targetLanguage": "ja-JP",
            "domain": domain, "source": source, "references": [target],
            "claimEligible": False, "referenceExposedToTeacher": False,
            "sourceLicense": license_name, "sourceProvenance": f"source {index}",
            "referenceProvenance": f"reference {index}",
        } for index, (domain, source, target, license_name) in enumerate([
            ("conversation", "Where is the station?", "駅はどこですか。", "CC-BY-2.0-FR"),
            ("news", "The vote starts today.", "投票は今日始まる。", "CC-BY-4.0"),
        ], start=1)]
        write_jsonl(suite, suite_rows)
        manifest = {
            "purpose": "reference-hidden local Qwen teacher training suite; never evaluation evidence",
            "promotion_eligible": False,
            "reference_exposed_to_teacher": False,
            "allowed_licenses": ALLOWED_LICENSES,
            "maximum_protected_five_gram_jaccard": 0.8,
            "inputs": {
                "protected_suites": [{"path": str(protected.resolve()), "sha256": sha256(protected)}],
                "excluded_datasets": [
                    {"path": str(path.resolve()), "sha256": sha256(path)}
                    for path in (base / "train.jsonl", base / "valid.jsonl")
                ],
            },
            "outputs": {"suite": {"path": str(suite.resolve()), "sha256": sha256(suite)}},
        }
        (root / "suite.jsonl.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        subprocess.run([
            "python3", "scripts/translation/build_balanced_human_reference_ablation.py",
            str(suite), str(base), str(output), "--direction", "en-ja",
            "--protected-suite", str(protected),
        ], check=True, capture_output=True, text=True)
        train = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        selected = [row for row in train if row.get("origin") == "human-balanced-hard-reference"]
        assert len(train) == 3 and len(selected) == 2
        assert {row["target"] for row in selected} == {"駅はどこですか。", "投票は今日始まる。"}
        output_manifest = json.loads((output / "manifest.json").read_text())
        assert output_manifest["counts"]["human_reference_train"] == 2
        assert output_manifest["counts"]["synthetic_train"] == 0
        assert output_manifest["domains"] == {"conversation": 1, "news": 1}
    print("Balanced human-reference ablation contract passed.")


if __name__ == "__main__":
    main()
