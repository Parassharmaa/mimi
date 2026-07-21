#!/usr/bin/env python3
"""Contract test for local source-only teacher generation and agreement filtering."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def report(path: Path, engine: str, suite: list[dict], hypotheses: dict[str, str]) -> None:
    path.write_text(json.dumps({
        "engine": engine,
        "results": [
            {
                "caseID": row["id"],
                "source": row["source"],
                "hypothesis": hypotheses[row["id"]],
            }
            for row in suite
        ],
    }, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-local-teacher-test-") as temporary:
        root = Path(temporary)
        seeds, protected, suite = root / "seeds.jsonl", root / "protected.jsonl", root / "suite.jsonl"
        write_jsonl(seeds, [
            {
                "id": "accept", "source": "Where is the station?", "source_language": "en-US",
                "target_language": "ja-JP", "domain": "travel", "license": "CC-BY-4.0",
                "provenance": "fixture", "split": "train",
            },
            {
                "id": "reject", "source": "This output should disagree.", "source_language": "en-US",
                "target_language": "ja-JP", "domain": "travel", "license": "CC-BY-4.0",
                "provenance": "fixture", "split": "train",
            },
            {
                "id": "accept-ja-en", "source": "駅はどこですか？", "source_language": "ja-JP",
                "target_language": "en-US", "domain": "travel", "license": "CC-BY-4.0",
                "provenance": "fixture", "split": "train",
            },
        ])
        write_jsonl(protected, [{"id": "heldout", "source": "A protected source.", "references": ["保護文です。"]}])
        subprocess.run([
            "python3", "scripts/translation/prepare_local_teacher_suite.py",
            str(seeds), str(protected), str(suite),
        ], check=True, capture_output=True, text=True)
        suite_rows = [json.loads(line) for line in suite.read_text().splitlines()]
        assert len(suite_rows) == 3 and all(row["claimEligible"] is False for row in suite_rows)

        preferred, teacher, independent = root / "preferred.json", root / "teacher.json", root / "independent.json"
        report(preferred, "preferred", suite_rows, {
            "accept": "駅はどこですか？", "reject": "この出力は違うはずです。",
            "accept-ja-en": "Where is the station?",
        })
        report(teacher, "teacher", suite_rows, {
            "accept": "駅はどこにありますか？", "reject": "まったく別の文章です。",
            "accept-ja-en": "Where is the station located?",
        })
        report(independent, "independent", suite_rows, {
            "accept": "駅はどこにあるのですか？", "reject": "翻訳結果が一致しません。",
            "accept-ja-en": "Where is the station?",
        })
        teacher_back, preferred_back, independent_back = (
            root / "teacher-back.json", root / "preferred-back.json", root / "independent-back.json"
        )
        report(teacher_back, "teacher-back", suite_rows, {
            "accept": "Where is the station?", "reject": "An unrelated sentence.",
            "accept-ja-en": "駅はどこですか？",
        })
        report(preferred_back, "preferred-back", suite_rows, {
            "accept": "Where is the station?", "reject": "This is different.",
            "accept-ja-en": "駅はどこですか？",
        })
        report(independent_back, "independent-back", suite_rows, {
            "accept": "Where is the station?", "reject": "The translation differs.",
            "accept-ja-en": "駅はどこですか？",
        })
        output = root / "accepted.jsonl"
        subprocess.run([
            "python3", "scripts/translation/build_local_teacher_consensus.py",
            str(suite), str(preferred), str(teacher), str(independent), str(protected), str(output),
            "--minimum-teacher-preferred", "20",
            "--minimum-teacher-independent", "35",
            "--minimum-preferred-independent", "20",
            "--teacher-backtranslation-report", str(teacher_back),
            "--preferred-backtranslation-report", str(preferred_back),
            "--independent-backtranslation-report", str(independent_back),
            "--minimum-roundtrip-agreement", "45",
        ], check=True, capture_output=True, text=True)
        accepted = [json.loads(line) for line in output.read_text().splitlines()]
        manifest = json.loads((root / "accepted.jsonl.manifest.json").read_text())
        assert [row["source_id"] for row in accepted] == ["accept", "accept-ja-en"]
        assert all(row["promotion_eligible"] is False for row in accepted)
        assert manifest["counts"]["accepted"] == 2
        assert sum(manifest["counts"]["rejected"].values()) == 1
    print("Local teacher consensus contract passed.")


if __name__ == "__main__":
    main()
