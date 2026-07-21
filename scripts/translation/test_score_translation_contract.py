#!/usr/bin/env python3
"""Contract tests for authenticated, correctly labelled score reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/translation/score_translation.py"


def write_report(path: Path, engine: str, revision: str, hypotheses: list[str]) -> None:
    results = []
    for index, hypothesis in enumerate(hypotheses):
        results.append(
            {
                "caseID": f"case-{index}",
                "sourceLanguage": "en",
                "targetLanguage": "ja",
                "domain": "conversation",
                "source": f"source {index}",
                "references": [f"reference {index}"],
                "hypothesis": hypothesis,
                "claimEligible": False,
                "latencySeconds": 0.01 + index / 1000,
            }
        )
    path.write_text(
        json.dumps(
            {
                "engine": engine,
                "modelRevision": revision,
                "preparationSeconds": 0.1,
                "runtimeImplementation": {
                    "benchmarkScriptSha256": "a" * 64,
                    "marianRuntimeSha256": "b" * 64,
                    "pythonVersion": "3.12.0",
                    "packages": {
                        "mlx": "0.30.6",
                        "tokenizers": "0.19.1",
                        "transformers": "4.40.2",
                    },
                },
                "benchmarkConfiguration": {
                    "warmRunsPerCase": 1,
                    "maximumGeneratedTokens": 192,
                },
                "decoderSelfKVCache": {
                    "strategy": "concatenate",
                    "blockSize": None,
                    "crossAttentionImmutable": True,
                },
                "positionEmbeddings": {
                    "strategy": "dynamic-sinusoidal",
                    "tableShape": None,
                    "appliedTo": [],
                },
                "results": results,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(SCORER), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-score-contract-") as temporary:
        root = Path(temporary)
        candidate = root / "candidate.json"
        baseline = root / "baseline.json"
        write_report(candidate, "candidate-engine", "candidate-rev", ["reference 0", "x"])
        write_report(baseline, "baseline-engine", "baseline-rev", ["x", "x"])

        generic = run(
            str(candidate),
            "--compare-report",
            str(baseline),
            "--bootstrap-samples",
            "100",
            "--seed",
            "7",
        )
        assert generic.returncode == 0, generic.stderr
        payload = json.loads(generic.stdout)
        assert payload["schemaVersion"] == 2
        assert payload["candidateReport"]["sha256"] == hashlib.sha256(
            candidate.read_bytes()
        ).hexdigest()
        assert payload["baselineReport"]["sha256"] == hashlib.sha256(
            baseline.read_bytes()
        ).hexdigest()
        assert payload["candidateReport"]["suiteContentSha256"] == payload[
            "baselineReport"
        ]["suiteContentSha256"]
        assert payload["candidateReport"]["modelRevision"] == "candidate-rev"
        assert payload["baselineReport"]["modelRevision"] == "baseline-rev"
        assert payload["scoringContract"]["sacrebleuVersion"]
        assert payload["scoringContract"]["chrfPlusPlusSignature"]
        comparison = payload["directions"]["en>ja"]
        assert "versusBaseline" in comparison
        assert "versusApple" not in comparison
        assert comparison["versusBaseline"]["bootstrapSamples"] == 100
        assert comparison["versusBaseline"]["bootstrapSeed"] == 7

        matched_baseline = root / "matched-baseline.json"
        write_report(
            matched_baseline,
            "candidate-engine",
            "matched-rev",
            ["x", "x"],
        )
        matched = run(
            str(candidate),
            "--compare-report",
            str(matched_baseline),
            "--bootstrap-samples",
            "10",
        )
        assert matched.returncode == 0, matched.stderr

        mismatched_payload = json.loads(matched_baseline.read_text(encoding="utf-8"))
        mismatched_payload["benchmarkConfiguration"]["warmRunsPerCase"] = 0
        matched_baseline.write_text(
            json.dumps(mismatched_payload) + "\n", encoding="utf-8"
        )
        mismatched = run(
            str(candidate),
            "--compare-report",
            str(matched_baseline),
        )
        assert mismatched.returncode != 0
        assert "different benchmarkConfiguration" in mismatched.stderr

        mismatched_payload["benchmarkConfiguration"]["warmRunsPerCase"] = 1
        mismatched_payload["runtimeImplementation"]["marianRuntimeSha256"] = "c" * 64
        matched_baseline.write_text(
            json.dumps(mismatched_payload) + "\n", encoding="utf-8"
        )
        mismatched = run(
            str(candidate),
            "--compare-report",
            str(matched_baseline),
        )
        assert mismatched.returncode != 0
        assert "different runtimeImplementation" in mismatched.stderr

        apple = run(
            str(candidate),
            "--compare-apple",
            str(baseline),
            "--bootstrap-samples",
            "10",
        )
        assert apple.returncode == 0, apple.stderr
        assert "versusApple" in json.loads(apple.stdout)["directions"]["en>ja"]

        invalid = run(str(candidate), "--bootstrap-samples", "0")
        assert invalid.returncode != 0
        assert "must be positive" in invalid.stderr

    print("Mimi authenticated translation scoring contract passed.")


if __name__ == "__main__":
    main()
