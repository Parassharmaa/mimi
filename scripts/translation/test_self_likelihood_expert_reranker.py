#!/usr/bin/env python3
"""Offline contracts for source-routed self-likelihood reranking."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATE = ROOT / "scripts/translation/evaluate_self_likelihood_expert_reranker.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def report(results: list[dict], engine: str) -> dict:
    return {
        "schemaVersion": 1,
        "createdAt": "2026-07-20T00:00:00Z",
        "engine": engine,
        "modelRevision": f"{engine}-revision",
        "hardware": "fixture",
        "operatingSystem": "fixture",
        "preparationSeconds": 0.0,
        "peakResidentBytes": 1,
        "modelBytes": 148_000_000,
        "physicalModelCount": 4,
        "runtimeImplementation": {"fixture": True},
        "results": results,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-self-likelihood-reranker-") as temporary:
        work = Path(temporary)
        generalist_rows: list[dict] = []
        en_expert_rows: list[dict] = []
        ja_expert_rows: list[dict] = []
        routed_rows: list[dict] = []
        diagnostics: list[dict] = []
        for index in range(1, 31):
            source = f"Open case E-{index} at {index + 5}."
            reference = f"{index + 5}時に案件E-{index}を開いてください。"
            base = {
                "caseID": f"en-{index}",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "fixture",
                "source": source,
                "references": [reference],
                "claimEligible": False,
                "latencySeconds": 0.01,
                "warmLatencySeconds": [0.01],
            }
            generalist = {**base, "hypothesis": f"案件E-{index}を{index + 5}時に開く。"}
            expert = {**base, "hypothesis": reference}
            generalist_rows.append(generalist)
            en_expert_rows.append(expert)
            ja_expert_rows.append(generalist)
            routed_rows.append({**expert, "selectedEngine": "expert"})
            diagnostics.append(
                {
                    "caseID": base["caseID"],
                    "direction": "en-ja",
                    "generalist": {"meanChosenTokenNLL": 2.0},
                    "expert": {"meanChosenTokenNLL": 1.0},
                }
            )
        for index in range(1, 31):
            source = f"案件J-{index}を{index + 7}時に開いてください。"
            reference = f"Please open case J-{index} at {index + 7}."
            base = {
                "caseID": f"ja-{index}",
                "sourceLanguage": "ja-JP",
                "targetLanguage": "en-US",
                "domain": "fixture",
                "source": source,
                "references": [reference],
                "claimEligible": False,
                "latencySeconds": 0.01,
                "warmLatencySeconds": [0.01],
            }
            generalist = {**base, "hypothesis": f"Open case J-{index} at {index + 7}."}
            expert = {**base, "hypothesis": reference}
            generalist_rows.append(generalist)
            en_expert_rows.append(generalist)
            ja_expert_rows.append(expert)
            routed_rows.append({**expert, "selectedEngine": "expert"})
            diagnostics.append(
                {
                    "caseID": base["caseID"],
                    "direction": "ja-en",
                    "generalist": {"meanChosenTokenNLL": 2.0},
                    "expert": {"meanChosenTokenNLL": 1.0},
                }
            )

        generalist_path = work / "generalist.json"
        en_expert_path = work / "en-expert.json"
        ja_expert_path = work / "ja-expert.json"
        routed_path = work / "routed.json"
        diagnostic_path = work / "diagnostics.json"
        write_json(generalist_path, report(generalist_rows, "generalist"))
        write_json(en_expert_path, report(en_expert_rows, "en-expert"))
        write_json(ja_expert_path, report(ja_expert_rows, "ja-expert"))
        routed = report(routed_rows, "routed")
        write_json(routed_path, routed)
        diagnostic = {
            "schemaVersion": 1,
            "runtime": {"fixture": True},
            "inputs": {
                "generalistReport": {"sha256": sha256(generalist_path)},
                "enJAExpertReport": {"sha256": sha256(en_expert_path)},
                "jaENExpertReport": {"sha256": sha256(ja_expert_path)},
                "routedReport": {"sha256": sha256(routed_path)},
            },
            "results": diagnostics,
        }
        write_json(diagnostic_path, diagnostic)

        outputs = [
            work / "full.json",
            work / "calibration.json",
            work / "test.json",
            work / "selection.json",
        ]
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "scripts/translation")}
        command = [
            "python3",
            str(EVALUATE),
            str(generalist_path),
            str(en_expert_path),
            str(ja_expert_path),
            str(routed_path),
            str(diagnostic_path),
            *(str(path) for path in outputs),
        ]
        evaluated = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert evaluated.returncode == 0, evaluated.stderr
        full = json.loads(outputs[0].read_text())
        selection = json.loads(outputs[3].read_text())
        assert len(full["results"]) == 60
        assert full["peakResidentBytes"] is None
        assert all(
            row["selectedEngine"] == "expert-self-likelihood-reranker"
            for row in full["results"]
        )
        assert selection["margins"] == {"en-ja": 1.0, "ja-en": 1.0}
        assert json.loads(outputs[1].read_text())["results"]
        assert json.loads(outputs[2].read_text())["results"]

        diagnostic["inputs"]["generalistReport"]["sha256"] = "0" * 64
        bad_diagnostic_path = work / "bad-diagnostics.json"
        write_json(bad_diagnostic_path, diagnostic)
        bad_outputs = [work / f"bad-{name}.json" for name in ("full", "cal", "test", "selection")]
        rejected = subprocess.run(
            [
                *command[:6],
                str(bad_diagnostic_path),
                *(str(path) for path in bad_outputs),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0 and "not bound" in rejected.stderr

    print("Mimi self-likelihood expert reranker contracts passed.")


if __name__ == "__main__":
    main()
