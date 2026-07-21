#!/usr/bin/env python3
"""One-update MPS smoke for the post-SFT Marian DQO entrypoint."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "Research/translation/models/elanmt-conversational-control-en-ja"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(identifier: str, source: str, chosen: str, rejected: str) -> dict:
    return {
        "id": f"dqo:{identifier}",
        "source_id": identifier,
        "source_language": "en-US",
        "target_language": "ja-JP",
        "source": source,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_candidate_id": f"{identifier}-chosen",
        "rejected_candidate_id": f"{identifier}-rejected",
        "domain": "mimi-live-speech-smoke",
        "origin": "two-reviewer-human-preference",
        "source_license": "CC0-1.0",
        "source_provenance": "local DQO optimizer smoke fixture",
        "review_status": "two-reviewer-selected-over-unapproved-candidate",
        "reviewer_ids": ["fixture-reviewer-a", "fixture-reviewer-b"],
    }


def main() -> None:
    required = [
        CHECKPOINT / "model.safetensors",
        CHECKPOINT / "config.json",
        CHECKPOINT / "source.spm",
        CHECKPOINT / "target.spm",
    ]
    if any(not path.is_file() for path in required):
        raise SystemExit(f"missing local supervised checkpoint for smoke: {CHECKPOINT}")
    with tempfile.TemporaryDirectory(prefix="mimi-marian-dqo-smoke-") as temporary:
        work = Path(temporary)
        preferences = work / "preferences"
        output = work / "output"
        preferences.mkdir()
        train_path, valid_path = preferences / "train.jsonl", preferences / "valid.jsonl"
        write_jsonl(
            train_path,
            [
                row(
                    "train-1", "Please open Mimi's settings.",
                    "Mimiの設定を開いてください。", "Mimiが設定を閉じました。",
                ),
                row(
                    "train-2", "The microphone is still muted.",
                    "マイクはまだミュートされています。", "マイクの音量を上げました。",
                ),
            ],
        )
        write_jsonl(
            valid_path,
            [
                row(
                    "valid-1", "Translation is ready now.",
                    "翻訳の準備ができました。", "翻訳は利用できません。",
                ),
                row(
                    "valid-2", "Try again after restarting Mimi.",
                    "Mimiを再起動してからもう一度お試しください。", "Mimiを削除してください。",
                ),
            ],
        )
        preference_manifest = {
            "schema_version": 1,
            "purpose": "post-supervised-win human-preference DQO only",
            "direction": "en-ja",
            "train": {"sha256": sha256(train_path)},
            "valid": {"sha256": sha256(valid_path)},
        }
        (preferences / "manifest.json").write_text(
            json.dumps(preference_manifest), encoding="utf-8"
        )

        win_report = work / "supervised-win.json"
        win_report.write_text(
            json.dumps({
                "schemaVersion": 1,
                "status": "supervised-win-approved",
                "approved": True,
                "direction": "en-ja",
                "candidateModelRevision": "offline-dqo-optimizer-smoke-fixture",
                "supervisedCheckpoint": {
                    "modelSHA256": sha256(CHECKPOINT / "model.safetensors")
                },
                "gates": [
                    {"name": name, "passed": True, "fixtureOnly": True}
                    for name in (
                        "reviewed-development-chrf-win",
                        "blind-human-development-win",
                        "no-new-critical-errors",
                        "general-retention",
                        "exact-bundle-checkpoint-binding",
                    )
                ],
                "fixtureOnly": True,
            }),
            encoding="utf-8",
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/translation/train_marian_dqo.py",
                str(preferences),
                str(CHECKPOINT),
                str(win_report),
                str(output),
                "--direction", "en-ja",
                "--device", "mps",
                "--batch-size", "2",
                "--gradient-accumulation", "1",
                "--max-steps", "1",
                "--evaluation-steps", "1",
                "--warmup-steps", "0",
                "--max-source-tokens", "48",
                "--max-target-tokens", "48",
            ],
            cwd=ROOT,
            check=True,
        )
        manifest = json.loads(
            (output / "mimi_dqo_training_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["direction"] == "en-ja"
        assert manifest["preferences"]["train_pairs"] == 2
        assert manifest["preferences"]["valid_pairs"] == 2
        assert manifest["supervised_win_report"]["status"] == "supervised-win-approved"
        assert len(manifest["history"]) == 2
        assert manifest["history"][-1]["step"] == 1
        assert (output / "model.safetensors").is_file()
    print("Mimi Marian DQO one-update MPS smoke passed.")


if __name__ == "__main__":
    main()
