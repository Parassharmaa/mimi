#!/usr/bin/env python3
"""Offline contract for blinded licensed-reference candidate anchoring."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        arguments,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def teacher_result(seed: dict, translations: list[str], response_id: str) -> dict:
    styles = ["natural-spoken", "concise-caption", "meaning-conservative"]
    payload = {
        "source_id": seed["id"],
        "translation_brief": {
            "register": "neutral",
            "terms": [],
            "preserve": ["Mimi"],
            "ambiguities": [],
        },
        "candidates": [
            {
                "translation": translation,
                "style": style,
                "risk_tags": [],
            }
            for style, translation in zip(styles, translations)
        ],
    }
    return {
        "custom_id": seed["id"],
        "response": {
            "body": {
                "id": response_id,
                "model": "gpt-5.6-sol",
                "system_fingerprint": "offline-fixture",
                "output_text": json.dumps(payload, ensure_ascii=False),
            }
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-reference-anchor-test-") as temporary:
        work = Path(temporary)
        seeds = [
            {
                "id": "reference-anchor-unique",
                "split": "train",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "mimi-product-ui",
                "source": "Please save the document before closing Mimi.",
                "license": "project-owned",
                "provenance": "offline fixture",
                "reference_translation": "Mimiを閉じる前に、書類を保存してください。",
                "reference_provenance": "offline licensed reference",
            },
            {
                "id": "reference-anchor-exact",
                "split": "train",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "mimi-product-ui",
                "source": "Reopen Mimi and start a new transcript.",
                "license": "project-owned",
                "provenance": "offline fixture",
                "reference_translation": "Mimiを開き直して、新しい文字起こしを開始してください。",
                "reference_provenance": "offline licensed reference",
            },
        ]
        results = [
            teacher_result(
                seeds[0],
                [
                    "Mimiを閉じる前に文書を保存してください。",
                    "閉じる前に、Mimiで書類を保存してください。",
                    "Mimiを終了する前に書類を保存してください。",
                ],
                "response-unique",
            ),
            teacher_result(
                seeds[1],
                [
                    seeds[1]["reference_translation"],
                    "Mimiを再度開き、新しい文字起こしを始めてください。",
                    "Mimiを開き直し、新規の文字起こしを開始してください。",
                ],
                "response-exact",
            ),
        ]
        seeds_path = work / "seeds.jsonl"
        results_path = work / "results.jsonl"
        protected_path = work / "protected.jsonl"
        queue_path = work / "queue.jsonl"
        judge_requests = work / "judge-requests.jsonl"
        write_jsonl(seeds_path, seeds)
        write_jsonl(results_path, results)
        write_jsonl(
            protected_path,
            [{"id": "heldout", "source": "Unrelated held-out text.", "references": ["無関係の文。"]}],
        )

        run(
            "python3",
            "scripts/translation/filter_synthetic_batch.py",
            str(seeds_path),
            str(results_path),
            str(protected_path),
            str(queue_path),
            "--include-licensed-reference-candidate",
        )
        queue = [
            json.loads(line)
            for line in queue_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in queue:
            grouped[row["source_id"]].append(row)
        assert {source_id: len(rows) for source_id, rows in grouped.items()} == {
            "reference-anchor-unique": 4,
            "reference-anchor-exact": 3,
        }
        origins = Counter(row["candidate_origin"] for row in queue)
        assert origins == {
            "teacher": 5,
            "licensed-reference": 1,
            "teacher-reference-equivalent": 1,
        }
        reference = next(
            row for row in queue if row["candidate_origin"] == "licensed-reference"
        )
        assert reference["style"] == "licensed-reference"
        assert reference["translation"] == seeds[0]["reference_translation"]

        run(
            "python3",
            "scripts/translation/prepare_distillation_judge_batch.py",
            str(queue_path),
            str(judge_requests),
            "--model",
            "offline-distinct-judge",
        )
        validation = run(
            "python3",
            "scripts/translation/run_synthetic_batch.py",
            "validate",
            str(judge_requests),
        )
        assert json.loads(validation.stdout)["request_count"] == 2
        for line in judge_requests.read_text(encoding="utf-8").splitlines():
            request = json.loads(line)
            judge_input = json.loads(request["body"]["input"][1]["content"])
            assert len(judge_input["candidates"]) in {3, 4}
            serialized = json.dumps(judge_input, ensure_ascii=False)
            assert "candidate_origin" not in serialized
            assert "licensed-reference" not in serialized
            assert "reference_provenance" not in serialized

    print("Mimi reference-anchored synthetic filter contract passed.")


if __name__ == "__main__":
    main()
