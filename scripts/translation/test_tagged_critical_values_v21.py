#!/usr/bin/env python3
"""Contract tests for V21 source-bound value tagging."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "scripts/translation/build_tagged_critical_values_v21.py"
sys.path.insert(0, str(BUILDER.parent))
from build_tagged_critical_values_v21 import tagged_row  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(identifier: str, source: str, target: str, origin: str) -> dict:
    return {
        "id": identifier,
        "source": source,
        "target": target,
        "source_language": "ja-JP",
        "target_language": "en-US",
        "origin": origin,
        "source_origin": "human-fixture",
        "domain": "fixture",
        "source_license": "CC-BY-4.0",
        "source_provenance": "fixture",
        "attribution": "fixture",
    }


def main() -> None:
    exact = tagged_row(
        row(
            "exact",
            "第74条は140人に適用される。",
            "Article 74 applies to 140 people.",
            "typed-numeric-target",
        ),
        "ja-en",
    )
    assert exact is not None
    assert exact["source"] == "第<v00>条は<v01>人に適用される。"
    assert exact["target"] == "Article <v00> applies to <v01> people."
    assert [item["canonical"] for item in exact["value_sidecar"]] == [
        "number:74",
        "number:140",
    ]

    lexical = tagged_row(
        row(
            "lexical",
            "四機のヘリコプターが到着した。",
            "Four helicopters arrived.",
            "typed-numeric-target",
        ),
        "ja-en",
    )
    assert lexical is not None
    assert lexical["source"] == "<v00>機のヘリコプターが到着した。"
    assert lexical["target"] == "<v00> helicopters arrived."
    assert lexical["value_sidecar"][0]["source_has_ascii_digits"] is False
    assert lexical["value_sidecar"][0]["target_has_ascii_digits"] is False

    duplicate = tagged_row(
        row(
            "duplicate",
            "第2条第2項",
            "Article 2, paragraph 2",
            "typed-numeric-target",
        ),
        "ja-en",
    )
    assert duplicate is None
    mismatch = tagged_row(
        row(
            "mismatch",
            "第74条",
            "Article 71",
            "typed-numeric-target",
        ),
        "ja-en",
    )
    assert mismatch is None

    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        parent = work / "parent"
        parent.mkdir()
        train_rows = [
            row(
                "tag-a",
                "第74条は140人に適用される。",
                "Article 74 applies to 140 people.",
                "typed-numeric-target",
            ),
            row(
                "tag-b",
                "四機のヘリコプターが到着した。",
                "Four helicopters arrived.",
                "typed-numeric-target",
            ),
            row(
                "replay-a",
                "報告書を開く。",
                "Open the report.",
                "base-preservation-replay",
            ),
            row(
                "replay-b",
                "窓を閉める。",
                "Close the window.",
                "base-preservation-replay",
            ),
        ]
        valid_rows = [
            row(
                "valid-tag",
                "第9節が適用される。",
                "Section nine applies.",
                "typed-numeric-target",
            ),
            row(
                "valid-plain",
                "記録を保存する。",
                "Save the record.",
                "base-preservation-replay",
            ),
        ]
        train = parent / "train.jsonl"
        valid = parent / "valid.jsonl"
        write_jsonl(train, train_rows)
        write_jsonl(valid, valid_rows)
        manifest = {
            "schema_version": 1,
            "experiment": "typed-numeric-preservation-v20-curriculum",
            "direction": "ja-en",
            "promotion_eligible": False,
            "does_not_authorize_app_integration": True,
            "does_not_authorize_public_upload": True,
            "effective_licenses": {
                "train": {"CC-BY-4.0": 4},
                "valid": {"CC-BY-4.0": 2},
            },
            "contamination_screen": {"fixture": True},
            "outputs": {
                "train": {
                    "bytes": train.stat().st_size,
                    "sha256": sha256(train),
                },
                "valid": {
                    "bytes": valid.stat().st_size,
                    "sha256": sha256(valid),
                },
            },
        }
        (parent / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        output = work / "output"
        output.mkdir()
        result = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                str(parent),
                str(output),
                "--direction",
                "ja-en",
                "--tagged-train-rows",
                "2",
                "--plain-replay-rows",
                "2",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise AssertionError(result.stderr or result.stdout)
        built = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert built["selection"]["selected_tagged_train"] == 2
        assert built["selection"]["selected_plain_replay"] == 2
        assert built["selection"]["tagged_valid"] == 1
        assert built["outputs"]["train"]["rows"] == 4
        assert built["does_not_authorize_training"] is True

    print("Mimi V21 tagged critical-value dataset contract passed.")


if __name__ == "__main__":
    main()
