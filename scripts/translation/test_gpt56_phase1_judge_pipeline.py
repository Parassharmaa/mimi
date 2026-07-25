#!/usr/bin/env python3
"""Contract test for blinded phase-1 judge request preparation and collection."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts/translation/prepare_gpt56_phase1_judge_batch.py"
COLLECT = ROOT / "scripts/translation/collect_gpt56_phase1_judge.py"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-phase1-judge-") as temporary:
        root = Path(temporary)
        suite_path = root / "suite.jsonl"
        candidate_path, baseline_path = root / "candidate.json", root / "baseline.json"
        requests_path, batch_path, output_path = (
            root / "requests.jsonl",
            root / "batch.jsonl",
            root / "report.json",
        )
        cases = [
            {
                "id": "en-ja-1",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "ui",
                "source": "Save changes.",
                "references": ["変更を保存します。"],
                "claimEligible": False,
            },
            {
                "id": "ja-en-1",
                "sourceLanguage": "ja-JP",
                "targetLanguage": "en-US",
                "domain": "ui",
                "source": "変更を保存します。",
                "references": ["Save changes."],
                "claimEligible": False,
            },
        ]
        suite_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
            encoding="utf-8",
        )
        candidate_hypotheses = {
            "en-ja-1": "変更を保存します。",
            "ja-en-1": "Save changes.",
        }
        baseline_hypotheses = {
            "en-ja-1": "変更。",
            "ja-en-1": "Changes.",
        }

        def engine(engine: str, hypotheses: dict[str, str]) -> dict:
            return {
                "schemaVersion": 1,
                "engine": engine,
                "modelRevision": f"{engine}-revision",
                "results": [
                    {
                        "caseID": row["id"],
                        **{
                            field: row[field]
                            for field in (
                                "sourceLanguage",
                                "targetLanguage",
                                "domain",
                                "source",
                                "references",
                                "claimEligible",
                            )
                        },
                        "hypothesis": hypotheses[row["id"]],
                    }
                    for row in cases
                ],
            }

        write_json(candidate_path, engine("candidate", candidate_hypotheses))
        write_json(baseline_path, engine("baseline", baseline_hypotheses))
        prepared = subprocess.run(
            [
                "python3",
                str(PREPARE),
                str(suite_path),
                str(candidate_path),
                str(baseline_path),
                str(requests_path),
                "--model",
                "fixture-judge-revision",
                "--model-family",
                "fixture-family",
                "--model-revision",
                "fixture-judge-revision",
                "--judge-role",
                "phase1-judge-a",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert prepared.returncode == 0, prepared.stderr
        requests = [
            json.loads(line) for line in requests_path.read_text().splitlines()
        ]
        assert len(requests) == 2
        assert all("candidate" not in row["body"]["input"][1]["content"] for row in requests)
        assert all("baseline" not in row["body"]["input"][1]["content"] for row in requests)

        batch_rows = []
        for index, request in enumerate(requests):
            visible = json.loads(request["body"]["input"][1]["content"])
            case_id = visible["case_id"]
            good = candidate_hypotheses[case_id]

            def score(text: str) -> dict:
                if text == good:
                    return {
                        "adequacy": 4,
                        "fluency": 4,
                        "terminology": 2,
                        "critical_error": False,
                        "error_tags": [],
                    }
                return {
                    "adequacy": 2,
                    "fluency": 3,
                    "terminology": 1,
                    "critical_error": False,
                    "error_tags": ["omission"],
                }

            payload = {
                "case_id": case_id,
                "output_a": score(visible["output_a"]),
                "output_b": score(visible["output_b"]),
            }
            batch_rows.append(
                {
                    "custom_id": request["custom_id"],
                    "error": None,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": f"response-{index}",
                            "status": "completed",
                            "model": "fixture-judge-revision",
                            "output_text": json.dumps(payload, ensure_ascii=False),
                        },
                    },
                }
            )
        batch_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in batch_rows),
            encoding="utf-8",
        )
        collected = subprocess.run(
            [
                "python3",
                str(COLLECT),
                str(suite_path),
                str(candidate_path),
                str(baseline_path),
                str(requests_path),
                str(batch_path),
                str(output_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert collected.returncode == 0, collected.stderr
        report = json.loads(output_path.read_text())
        assert report["blinded"] is True
        assert report["reasoningTracesStored"] is False
        assert report["judgeModelFamily"] == "fixture-family"
        assert len(report["results"]) == 2
        assert all(row["candidate"]["adequacy"] == 4 for row in report["results"])
        assert all(row["baseline"]["errorTags"] == ["omission"] for row in report["results"])

    print("GPT-5.6 phase-1 judge pipeline contract passed.")


if __name__ == "__main__":
    main()
