#!/usr/bin/env python3
"""End-to-end contracts for exact-source translation-memory artifacts."""

from __future__ import annotations

import gzip
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "scripts/translation/build_exact_translation_memory.py"
PREPARE = ROOT / "scripts/translation/prepare_exact_translation_memory_suite.py"
APPLY = ROOT / "scripts/translation/apply_exact_translation_memory.py"
PACKAGE = ROOT / "scripts/translation/package_marian_translation_memory.py"


def training_row(case_id: str, document: str, source: str, target: str) -> dict:
    source_language, target_language = ("ja-JP", "en-US")
    return {
        "id": case_id,
        "source": source,
        "target": target,
        "source_id": f"{document}:tu-1",
        "source_language": source_language,
        "target_language": target_language,
        "source_license": "PDL-1.0-compatible-CC-BY-4.0",
        "source_provenance": f"https://example.test/{document}",
        "source_tmx_sha256": document.rjust(64, "0")[-64:],
        "domain": "ministry-published-legal",
    }


with tempfile.TemporaryDirectory(prefix="mimi-exact-memory-test-") as directory:
    root = Path(directory)
    train = root / "train.jsonl"
    runtime = root / "memory.json"
    audit = root / "memory-audit.json.gz"
    valid = root / "valid.jsonl"
    suite = root / "suite.jsonl"
    manifest = root / "suite-manifest.json"
    base_report = root / "base.json"
    memory_report = root / "memory-report.json"
    base_pack = root / "base-pack"
    output_pack = root / "memory-pack"
    rows = [
        training_row("a", "law-1", "（立入調査等）", "(On-Site Investigations)"),
        training_row("b", "law-2", "（立入調査等）", "(On-site Inspections)"),
        training_row("c", "law-3", "Keep 25%", "20%を維持する"),
        training_row("d", "law-4", "Keep 25%", "20%を維持する"),
        training_row("e", "law-5", "単独", "Alone"),
        training_row("f", "law-6", "競合", "Conflict A"),
        training_row("g", "law-6", "競合", "Conflict B"),
        training_row("h", "law-7", "競合", "Conflict A"),
        training_row("i", "law-8", "競合", "Conflict A"),
        training_row("j", "law-9", "Value 12.", "Value 13。"),
        training_row("k", "law-10", "Value 12.", "Value 13。"),
    ]
    train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    subprocess.run(
        ["python3", str(BUILD), str(train), str(runtime), str(audit)],
        check=True,
        capture_output=True,
        text=True,
    )
    memory = json.loads(runtime.read_text())
    selected = memory["entries"]["ja-en"]["(立入調査等)"]
    assert selected in {"(On-Site Investigations)", "(On-site Inspections)"}
    assert "Keep 25%" not in memory["entries"]["ja-en"]
    assert "Value 12." not in memory["entries"]["ja-en"]
    with gzip.open(audit, "rt", encoding="utf-8") as handle:
        audit_payload = json.load(handle)
    assert audit_payload["counts"]["entries"] == 2
    assert audit_payload["counts"]["rejected"]["critical-token-mismatch"] == 2, (
        audit_payload["counts"]["rejected"]
    )
    assert audit_payload["counts"]["rejected"]["conflicting-document-targets"] == 1

    validation = training_row("valid", "law-valid", "（立入調査等）", "(Site Inspection, etc.)")
    valid.write_text(json.dumps(validation, ensure_ascii=False) + "\n")
    subprocess.run(
        ["python3", str(PREPARE), str(runtime), str(valid), str(suite), str(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    suite_row = json.loads(suite.read_text())
    engine_row = {
        **{key: value for key, value in suite_row.items() if key != "id"},
        "caseID": suite_row["id"],
        "hypothesis": "(Interest Survey)",
        "latencySeconds": 0.02,
        "warmLatencySeconds": [0.01],
        "selectedEngine": "expert",
    }
    base_report.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "engine": "fixture",
                "modelRevision": "fixture",
                "modelBytes": 1,
                "preparationSeconds": 0.0,
                "results": [engine_row],
            }
        )
    )
    subprocess.run(
        ["python3", str(APPLY), str(runtime), str(base_report), str(memory_report)],
        check=True,
        capture_output=True,
        text=True,
    )
    applied = json.loads(memory_report.read_text())
    assert applied["translationMemory"]["hits"] == {"en-ja": 0, "ja-en": 1}
    assert applied["results"][0]["hypothesis"] == selected
    assert applied["results"][0]["selectedNeuralEngine"] == "expert"

    base_pack.mkdir()
    (base_pack / "manifest.json").write_text(
        json.dumps(
            {
                "format": "mimi-mlx-marian-moe-v1",
                "doesNotAuthorizeAppIntegration": True,
                "qualityStatus": "fixture",
                "files": {},
            }
        )
    )
    subprocess.run(
        ["python3", str(PACKAGE), str(base_pack), str(runtime), str(output_pack)],
        check=True,
        capture_output=True,
        text=True,
    )
    packaged = json.loads((output_pack / "manifest.json").read_text())
    record = packaged["files"]["memory/exact-translation-memory.json"]
    assert record["bytes"] == runtime.stat().st_size
    assert packaged["translationMemory"]["entries"] == 2
    assert packaged["doesNotAuthorizeAppIntegration"] is True
    subprocess.run(
        [
            "python3",
            str(APPLY),
            str(runtime),
            str(base_report),
            str(memory_report),
            "--pack",
            str(output_pack),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    packed_report = json.loads(memory_report.read_text())
    assert packed_report["modelBytes"] == sum(
        path.stat().st_size for path in output_pack.rglob("*") if path.is_file()
    )
    assert packed_report["translationMemory"]["pack"]["manifestSHA256"]

print("Exact translation-memory contracts passed.")
