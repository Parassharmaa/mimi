#!/usr/bin/env python3
"""End-to-end contract test for independent two-checkpoint structure fallback."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/evaluate_structure_fallback.py"
BENCHMARK = ROOT / "scripts/translation/run_mlx_marian_benchmark.py"
RUNTIME = ROOT / "scripts/translation/marian_mlx.py"
AUDIT = ROOT / "scripts/translation/audit_translation_structures.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def make_model(path: Path, label: str) -> dict:
    path.mkdir()
    weights = path / "model.safetensors"
    weights.write_bytes(f"{label} q4 weights".encode())
    source_sha = hashlib.sha256(f"{label} full weights".encode()).hexdigest()
    write_json(
        path / "manifest.json",
        {
            "source_weights_sha256": source_sha,
            "files": {
                "model.safetensors": {
                    "bytes": weights.stat().st_size,
                    "sha256": sha256(weights),
                }
            },
        },
    )
    return {
        "path": str(path),
        "manifestSha256": sha256(path / "manifest.json"),
        "sourceWeightsSha256": source_sha,
        "quantizedWeightsSha256": sha256(weights),
    }


def suite_row(case_id: str, source: str, reference: str) -> dict:
    return {
        "id": case_id,
        "sourceLanguage": "en-US",
        "targetLanguage": "ja-JP",
        "domain": "legal",
        "source": source,
        "references": [reference],
        "claimEligible": False,
    }


def report_row(suite: dict, hypothesis: str) -> dict:
    return {
        "caseID": suite["id"],
        **{key: suite[key] for key in (
            "sourceLanguage", "targetLanguage", "domain", "source", "references", "claimEligible"
        )},
        "hypothesis": hypothesis,
        "latencySeconds": 0.01,
        "warmLatencySeconds": [0.01],
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-structure-fallback-") as temporary:
        root = Path(temporary)
        suite_values = [
            suite_row("negation", "It is not allowed.", "許可されない。"),
            suite_row("number", "The limit is 12.", "上限は12。"),
        ]
        suite = root / "suite.jsonl"
        suite.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in suite_values))
        suite_manifest = root / "suite.manifest.json"
        write_json(suite_manifest, {"fixture": True})

        models = {
            name: make_model(root / name, name)
            for name in ("primary", "alternate", "baseline")
        }
        runtime = {
            "benchmarkScriptSha256": sha256(BENCHMARK),
            "marianRuntimeSha256": sha256(RUNTIME),
            "packages": {"mlx": "0.30.6"},
        }

        def write_report(name: str, hypotheses: list[str]) -> Path:
            path = root / f"{name}.json"
            write_json(
                path,
                {
                    "runtimeImplementation": runtime,
                    "declaredModels": {"en-ja": models[name]},
                    "results": [
                        report_row(row, hypothesis)
                        for row, hypothesis in zip(suite_values, hypotheses, strict=True)
                    ],
                },
            )
            return path

        primary_report = write_report("primary", ["許可される。", "上限は12。"])
        alternate_report = write_report("alternate", ["許可されない。", "上限は13。"])
        baseline_report = write_report("baseline", ["許可される。", "上限は13。"])
        pack = root / "pack"
        pack.mkdir()
        write_json(pack / "manifest.json", {"fixture": True})
        contract = root / "contract.json"
        write_json(
            contract,
            {
                "suite": {
                    "path": str(suite),
                    "sha256": sha256(suite),
                    "manifestPath": str(suite_manifest),
                    "manifestSha256": sha256(suite_manifest),
                    "casesPerDirection": 2,
                },
                "models": models,
                "implementation": {
                    "evaluatorScriptSha256": sha256(SCRIPT),
                    "benchmarkScriptSha256": sha256(BENCHMARK),
                    "marianRuntimeSha256": sha256(RUNTIME),
                    "structureAuditScriptSha256": sha256(AUDIT),
                    "mlxVersion": "0.30.6",
                },
                "gates": {
                    "pairedSentenceChrFPlusPlus": {
                        "samples": 100,
                        "seed": 9,
                        "minimumLowerBound": 0.0,
                    },
                    "maximumFallbackRate": 0.6,
                    "maximumWarmP95Seconds": 0.1,
                },
                "distribution": {
                    "currentPackPath": str(pack),
                    "currentPackManifestSha256": sha256(pack / "manifest.json"),
                    "maximumBytes": 100000,
                },
                "selectionUsesReferences": False,
                "doesNotAuthorizeModelPromotion": True,
                "doesNotAuthorizeAppIntegration": True,
            },
        )
        output = root / "output.json"
        command = [
            sys.executable,
            str(SCRIPT),
            str(contract),
            str(primary_report),
            str(alternate_report),
            str(baseline_report),
            str(output),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        payload = json.loads(output.read_text())
        assert payload["status"] == "independent-legal-safety-test-passed"
        assert payload["policy"]["fallbackCases"] == 1
        assert payload["structure"]["candidate"] == {
            "exactCriticalTokenMismatches": 0,
            "negationMarkerMismatches": 0,
        }
        assert payload["doesNotAuthorizeModelPromotion"] is True

        (root / "primary" / "model.safetensors").write_bytes(b"tampered")
        tampered = subprocess.run(
            command[:-1] + [str(root / "tampered-output.json")],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert tampered.returncode != 0
        assert "model integrity differs" in tampered.stderr

    print("Structure fallback contracts passed.")


if __name__ == "__main__":
    main()
