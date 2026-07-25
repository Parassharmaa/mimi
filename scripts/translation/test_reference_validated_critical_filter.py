#!/usr/bin/env python3
"""End-to-end adversarial contract for reference-validated critical admission."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILTER = ROOT / "scripts/translation/filter_synthetic_batch.py"
STYLES = (
    "natural-spoken",
    "concise-caption",
    "meaning-conservative",
)


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def result(seed: dict, translations: list[str]) -> dict:
    payload = {
        "source_id": seed["id"],
        "translation_brief": {
            "register": "neutral",
            "terms": [],
            "preserve": [],
            "ambiguities": [],
        },
        "candidates": [
            {
                "translation": translation,
                "style": style,
                "risk_tags": [],
            }
            for style, translation in zip(STYLES, translations)
        ],
    }
    return {
        "custom_id": seed["id"],
        "response": {
            "body": {
                "id": f"fixture-{seed['id']}",
                "model": "offline-teacher",
                "system_fingerprint": "offline",
                "output_text": json.dumps(payload, ensure_ascii=False),
            }
        },
    }


def run_filter(
    seeds: Path,
    results: Path,
    protected: Path,
    queue: Path,
    reference_validated: bool,
) -> dict:
    command = [
        "python3",
        str(FILTER),
        str(seeds),
        str(results),
        str(protected),
        str(queue),
        "--include-licensed-reference-candidate",
    ]
    if reference_validated:
        command.append("--reference-validated-critical-equivalence")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="mimi-reference-critical-filter-test-"
    ) as temporary:
        work = Path(temporary)
        seeds = [
            {
                "id": "safe-month",
                "split": "train",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "fixture",
                "source": "The meeting is in September.",
                "license": "project-owned",
                "provenance": "offline fixture",
                "reference_translation": "会議は9月にあります。",
                "reference_provenance": "offline fixture",
            },
            {
                "id": "wrong-month",
                "split": "train",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "fixture",
                "source": "The meeting is in September.",
                "license": "project-owned",
                "provenance": "offline fixture",
                "reference_translation": "会議は9月にあります。",
                "reference_provenance": "offline fixture",
            },
            {
                "id": "wrong-url",
                "split": "train",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "fixture",
                "source": "Open https://example.com now.",
                "license": "project-owned",
                "provenance": "offline fixture",
                "reference_translation": "今すぐhttps://example.comを開いてください。",
                "reference_provenance": "offline fixture",
            },
        ]
        results = [
            result(
                seeds[0],
                [
                    "会議は9月です。",
                    "9月に会議があります。",
                    "その会議は9月に開かれます。",
                ],
            ),
            result(
                seeds[1],
                [
                    "会議は9月です。",
                    "6月に会議があります。",
                    "その会議は9月に開かれます。",
                ],
            ),
            result(
                seeds[2],
                [
                    "今すぐhttps://example.comを開いてください。",
                    "https://example.netを今開いてください。",
                    "ただちにhttps://example.comを開いてください。",
                ],
            ),
        ]
        seeds_path = work / "seeds.jsonl"
        results_path = work / "results.jsonl"
        protected_path = work / "protected.jsonl"
        strict_queue = work / "strict-queue.jsonl"
        reference_queue = work / "reference-queue.jsonl"
        write_jsonl(seeds_path, seeds)
        write_jsonl(results_path, results)
        write_jsonl(
            protected_path,
            [
                {
                    "id": "unrelated",
                    "source": "Unrelated protected source.",
                    "references": ["無関係な保護対象。"],
                }
            ],
        )

        strict = run_filter(
            seeds_path,
            results_path,
            protected_path,
            strict_queue,
            False,
        )
        assert strict["sources"] == 0
        assert strict["rejected"] == {"protected-token-mismatch": 3}
        assert strict["reference_validated_critical_equivalence"] is False

        reference_validated = run_filter(
            seeds_path,
            results_path,
            protected_path,
            reference_queue,
            True,
        )
        assert reference_validated["sources"] == 1
        assert reference_validated["queued"] == 4
        assert reference_validated["rejected"] == {
            "protected-token-mismatch": 2
        }
        assert reference_validated[
            "reference_validated_critical_equivalence"
        ] is True
        assert reference_validated["critical_token_admission"] == {
            "reference-target-strict": 3
        }
        queue = [
            json.loads(line)
            for line in reference_queue.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert {row["source_id"] for row in queue} == {"safe-month"}
        assert sum(
            row["candidate_origin"] == "licensed-reference" for row in queue
        ) == 1

    print("Reference-validated critical filter contract passed.")


if __name__ == "__main__":
    main()
