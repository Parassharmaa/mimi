#!/usr/bin/env python3
"""Contract test for the law-grouped legal safety validation builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/prepare_legal_safety_validation.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(law: int, unit: int, direction: str, en: str, ja: str) -> dict:
    source_id = f"law-{law}:tu-{unit}"
    source, target = (en, ja) if direction == "en-ja" else (ja, en)
    source_language, target_language = (
        ("en-US", "ja-JP") if direction == "en-ja" else ("ja-JP", "en-US")
    )
    return {
        "id": f"jlt:{source_id}:{direction}",
        "source_id": source_id,
        "source_language": source_language,
        "target_language": target_language,
        "source": source,
        "target": target,
        "origin": "finalized-japanese-law-translation",
        "translation_status": "finalized",
        "source_license": "PDL-1.0-compatible-CC-BY-4.0",
        "training_only": True,
        "promotion_eligible": False,
        "attribution": "fixture Japanese Law attribution",
        "source_provenance": f"https://example.test/law-{law}",
        "source_normalized_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "target_normalized_sha256": hashlib.sha256(target.encode()).hexdigest(),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in rows),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-legal-safety-") as temporary:
        root = Path(temporary)
        jlt = root / "jlt"
        jlt.mkdir()
        split_pairs = {
            "train": [(1, "Training sentence.", "訓練文。")],
            "test": [
                (20, "This test must not be disclosed.", "この試験は開示してはならない。"),
                (21, "The test limit is 14 percent.", "試験上限は十四パーセントとする。"),
                (22, "(ii) Test enumeration requirement", "二 試験列挙要件"),
                (23, "A " + "long test legal sentence " * 15, "長い試験法令文" * 30),
                (24, "General test legal rule.", "一般的な試験法令。"),
            ],
            "valid": [
                (10, "This must not be disclosed.", "これは開示してはならない。"),
                (11, "The limit is 12 percent.", "上限は十二パーセントとする。"),
                (12, "(i) Enumeration requirement", "一 列挙要件"),
                (13, "A " + "long legal sentence " * 15, "長い法令文" * 30),
                (14, "General legal rule.", "一般的な法令。"),
            ],
        }
        outputs = {}
        for split, pairs in split_pairs.items():
            rows = [
                row(law, 1, direction, en, ja)
                for law, en, ja in pairs
                for direction in ("en-ja", "ja-en")
            ]
            path = jlt / f"{split}.jsonl"
            write_rows(path, rows)
            outputs[split] = {
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(rows),
                "bytes": path.stat().st_size,
            }
        (jlt / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "document_grouped_split": True,
                    "split_seed": "fixture",
                    "outputs": outputs,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        protected = root / "protected.jsonl"
        write_rows(
            protected,
            [{"id": "protected", "source": "Unrelated protected source.", "references": []}],
        )
        output = root / "suite.jsonl"
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                str(jlt),
                str(output),
                "--pairs",
                "5",
                "--protected-suite",
                str(protected),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert manifest["cases"] == 10
        assert manifest["cases_per_direction"] == 5
        assert manifest["selection_uses_model_outputs"] is False
        assert manifest["claim_eligible"] is False
        assert manifest["does_not_authorize_model_promotion"] is True
        assert manifest["contamination_controls"][
            "law_ids_disjoint_across_jlt_train_valid_test"
        ] is True
        assert manifest["selection"]["selected_by_bucket"] == {
            "critical-structure": 1,
            "general": 1,
            "legal-enumeration": 1,
            "long-form": 1,
            "negation": 1,
        }
        assert len(rows) == 10
        assert len({value["sourceID"] for value in rows}) == 5
        assert all(value["claimEligible"] is False for value in rows)
        assert manifest["output"]["sha256"] == sha256(output)

        test_output = root / "test-suite.jsonl"
        test_result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                str(jlt),
                str(test_output),
                "--pairs",
                "5",
                "--source-split",
                "test",
                "--protected-suite",
                str(protected),
                "--protected-suite",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert test_result.returncode == 0, test_result.stderr
        test_manifest = json.loads(test_output.with_suffix(".manifest.json").read_text())
        test_rows = [json.loads(line) for line in test_output.read_text().splitlines()]
        assert test_manifest["suite"] == "legal-safety-test-v1"
        assert test_manifest["source"]["test"]["available_complete_pairs"] == 5
        assert test_manifest["contamination_controls"][
            "exact_normalized_text_disjoint_from_jlt_train_and_valid"
        ] is True
        assert all(value["sourceCorpus"] == "jlt-test" for value in test_rows)
        assert all(value["split"] == "legal-safety-test" for value in test_rows)

    print("Legal safety validation contracts passed.")


if __name__ == "__main__":
    main()
