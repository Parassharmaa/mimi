#!/usr/bin/env python3
"""Offline contracts for source-routed reverse-consistency reranking."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts/translation/prepare_roundtrip_expert_reranking_suites.py"
EVALUATE = ROOT / "scripts/translation/evaluate_roundtrip_expert_reranker.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
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
        "modelBytes": 2,
        "physicalModelCount": 4,
        "runtimeImplementation": {"fixture": True},
        "results": results,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-roundtrip-reranker-") as temporary:
        work = Path(temporary)
        generalist_rows: list[dict] = []
        en_expert_rows: list[dict] = []
        ja_expert_rows: list[dict] = []
        routed_rows: list[dict] = []
        for index in range(1, 21):
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
        for index in range(1, 21):
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
            generalist = {**base, "hypothesis": f"Open J-{index} at {index + 7}."}
            expert = {**base, "hypothesis": reference}
            generalist_rows.append(generalist)
            en_expert_rows.append(generalist)
            ja_expert_rows.append(expert)
            routed_rows.append({**expert, "selectedEngine": "expert"})

        generalist_path = work / "generalist.json"
        en_expert_path = work / "en-expert.json"
        ja_expert_path = work / "ja-expert.json"
        routed_path = work / "routed.json"
        write_json(generalist_path, report(generalist_rows, "generalist"))
        write_json(en_expert_path, report(en_expert_rows, "en-expert"))
        write_json(ja_expert_path, report(ja_expert_rows, "ja-expert"))
        routed = report(routed_rows, "routed")
        routed["inputs"] = {
            "generalistReport": {"sha256": sha256(generalist_path)},
            "enJAExpertReport": {"sha256": sha256(en_expert_path)},
            "jaENExpertReport": {"sha256": sha256(ja_expert_path)},
        }
        write_json(routed_path, routed)

        generalist_suite = work / "generalist-suite.jsonl"
        expert_suite = work / "expert-suite.jsonl"
        manifest = work / "manifest.json"
        prepared = subprocess.run(
            [
                "python3",
                str(PREPARE),
                str(generalist_path),
                str(en_expert_path),
                str(ja_expert_path),
                str(routed_path),
                str(generalist_suite),
                str(expert_suite),
                str(manifest),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert prepared.returncode == 0, prepared.stderr
        assert json.loads(manifest.read_text())["counts"]["cases"] == 40

        back_paths = []
        for kind, suite_path in (("generalist", generalist_suite), ("expert", expert_suite)):
            suite_rows = [json.loads(line) for line in suite_path.read_text().splitlines()]
            results = [
                {
                    "caseID": row["id"],
                    "sourceLanguage": row["sourceLanguage"],
                    "targetLanguage": row["targetLanguage"],
                    "domain": row["domain"],
                    "source": row["source"],
                    "references": row["references"],
                    "claimEligible": False,
                    "hypothesis": row["references"][0],
                    "latencySeconds": 0.01,
                    "warmLatencySeconds": [0.01],
                }
                for row in suite_rows
            ]
            path = work / f"{kind}-back.json"
            write_json(path, report(results, f"{kind}-back"))
            back_paths.append(path)

        outputs = [
            work / "full.json",
            work / "calibration.json",
            work / "test.json",
            work / "selection.json",
        ]
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "scripts/translation")}
        evaluated = subprocess.run(
            [
                "python3",
                str(EVALUATE),
                str(generalist_path),
                str(en_expert_path),
                str(ja_expert_path),
                str(routed_path),
                str(back_paths[0]),
                str(back_paths[1]),
                *(str(path) for path in outputs),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert evaluated.returncode == 0, evaluated.stderr
        full = json.loads(outputs[0].read_text())
        assert len(full["results"]) == 40
        assert full["peakResidentBytes"] is None
        assert all(row["selectedEngine"] == "expert-roundtrip-reranker" for row in full["results"])
        assert json.loads(outputs[3].read_text())["margins"] == {"en-ja": 0.0, "ja-en": 0.0}
        assert json.loads(outputs[1].read_text())["results"]
        assert json.loads(outputs[2].read_text())["results"]

        routed["inputs"]["generalistReport"]["sha256"] = "0" * 64
        bad_routed = work / "bad-routed.json"
        write_json(bad_routed, routed)
        rejected = subprocess.run(
            [
                "python3",
                str(PREPARE),
                str(generalist_path),
                str(en_expert_path),
                str(ja_expert_path),
                str(bad_routed),
                str(work / "bad-generalist-suite.jsonl"),
                str(work / "bad-expert-suite.jsonl"),
                str(work / "bad-manifest.json"),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert rejected.returncode != 0 and "not bound" in rejected.stderr

    print("Mimi roundtrip expert reranker contracts passed.")


if __name__ == "__main__":
    main()
