#!/usr/bin/env python3
"""Contract test for deterministic blinded translation comparison artifacts."""

from __future__ import annotations

import json
import stat
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts/translation/prepare_blinded_translation_comparison.py"
ANALYZE = ROOT / "scripts/translation/analyze_blinded_translation_comparison.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-blinded-comparison-") as temporary:
        work = Path(temporary)
        suite_path = work / "suite.jsonl"
        candidate_path = work / "candidate.json"
        baseline_path = work / "baseline.json"
        batch_path = work / "batch.jsonl"
        mapping_path = work / "mapping.jsonl"
        rubric_path = work / "rubric.json"

        cases = []
        candidate_results = []
        baseline_results = []
        for direction, languages in (
            ("en-ja", ("en-US", "ja-JP")),
            ("ja-en", ("ja-JP", "en-US")),
        ):
            for index in range(2):
                case_id = f"{direction}-{index}"
                case = {
                    "id": case_id,
                    "sourceLanguage": languages[0],
                    "targetLanguage": languages[1],
                    "domain": "fixture",
                    "source": f"source-{case_id}",
                    "references": [f"reference-{case_id}"],
                    "sourceUnit": "sentence",
                    "segmentCount": 1,
                }
                cases.append(case)
                shared = {
                    "caseID": case_id,
                    **{
                        key: case[key]
                        for key in (
                            "sourceLanguage",
                            "targetLanguage",
                            "domain",
                            "source",
                            "references",
                        )
                    },
                }
                candidate_results.append(
                    {**shared, "hypothesis": f"candidate-{case_id}"}
                )
                baseline_results.append({**shared, "hypothesis": f"baseline-{case_id}"})

        write_jsonl(suite_path, cases)
        candidate_path.write_text(
            json.dumps(
                {
                    "engine": "candidate",
                    "modelRevision": "candidate-revision",
                    "results": candidate_results,
                }
            ),
            encoding="utf-8",
        )
        baseline_path.write_text(
            json.dumps(
                {
                    "engine": "baseline",
                    "modelRevision": "baseline-revision",
                    "results": baseline_results,
                }
            ),
            encoding="utf-8",
        )
        prepared = subprocess.run(
            [
                "python3",
                str(PREPARE),
                str(suite_path),
                str(candidate_path),
                str(baseline_path),
                str(batch_path),
                str(mapping_path),
                str(rubric_path),
                "--seed",
                "fixture-seed",
                "--candidate-label",
                "candidate",
                "--baseline-label",
                "baseline",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert prepared.returncode == 0, prepared.stderr
        mapping_mode = stat.S_IMODE(mapping_path.stat().st_mode)
        assert mapping_mode == 0o600
        manifest = json.loads(batch_path.with_suffix(".manifest.json").read_text())
        assert manifest["directionBalance"] == {
            "en-US>ja-JP": {"cases": 2, "candidateA": 1, "candidateB": 1},
            "ja-JP>en-US": {"cases": 2, "candidateA": 1, "candidateB": 1},
        }

        mappings = [json.loads(line) for line in mapping_path.read_text().splitlines()]
        verdicts = []
        for mapping in mappings:
            winner = "A" if mapping["candidate_A_system"] == "candidate" else "B"
            verdicts.append(
                {
                    "case_id": mapping["case_id"],
                    "adequacy_winner": winner,
                    "fluency_winner": winner,
                    "overall_preference": winner,
                    "critical_error_A": False,
                    "critical_error_B": False,
                    "candidate_A_error_tags": [],
                    "candidate_B_error_tags": [],
                    "brief_justification": "The candidate preserves the fixture source.",
                }
            )
        verdicts_path = work / "verdicts.jsonl"
        summary_path = work / "summary.json"
        unblinded_path = work / "unblinded.jsonl"
        write_jsonl(verdicts_path, verdicts)
        analyzed = subprocess.run(
            [
                "python3",
                str(ANALYZE),
                str(suite_path),
                str(verdicts_path),
                str(mapping_path),
                str(summary_path),
                str(unblinded_path),
                "--candidate-label",
                "candidate",
                "--baseline-label",
                "baseline",
                "--judge-model",
                "fixture-model",
                "--judge-family",
                "fixture-family",
                "--judge-revision",
                "fixture-revision",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert analyzed.returncode == 0, analyzed.stderr
        summary = json.loads(summary_path.read_text())
        assert summary["overall"]["overallPreference"] == {
            "candidateWins": 4,
            "baselineWins": 0,
            "ties": 0,
            "decisiveCases": 4,
            "candidateDecisiveWinRate": 1.0,
            "candidateDecisiveWinRateWilson95": [
                0.5101091635454027,
                1.0,
            ],
            "twoSidedExactSignTestP": 0.125,
        }

    print("Blinded translation comparison contract passed.")


if __name__ == "__main__":
    main()
