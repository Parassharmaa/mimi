#!/usr/bin/env python3
"""Offline contract for the 25%-synthetic reference-anchored dataset."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORIGINS = [
    ("human-kftt-replay", "wikipedia", "CC-BY-SA-3.0"),
    ("human-alt-parallel", "human-translated-news", "CC-BY-4.0"),
    ("finalized-japanese-law-translation", "ministry-published-legal", "PDL-1.0-compatible-CC-BY-4.0"),
    ("human-tatoeba-bidirectional-agreement-filtered", "conversational", "CC-BY-2.0-FR"),
    ("human-alt-document-window", "long-document-news", "CC-BY-4.0"),
    ("mimi-shipped-ui-pair", "mimi-product-ui", "project-owned"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_dataset(directory: Path, train: list[dict], valid: list[dict]) -> None:
    directory.mkdir()
    train_path = directory / "train.jsonl"
    valid_path = directory / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    manifest = {
        "outputs": {
            "train": {"sha256": sha256(train_path)},
            "valid": {"sha256": sha256(valid_path)},
        }
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )


def assessment(candidate_id: str, selected: bool) -> dict:
    return {
        "critical_error": False,
        "protected_tokens_preserved": True,
        "error_tags": [],
        "adequacy": 4 if selected else 3,
        "fluency": 4 if selected else 3,
        "terminology": 4 if selected else 3,
        "candidate_id": candidate_id,
    }


def approved_row(index: int) -> tuple[dict, dict]:
    source_id = f"pilot-{index}"
    candidate_id = f"teacher-candidate-{index}"
    source = f"Please save document {chr(64 + index)} before closing Mimi."
    reference = f"Mimiを閉じる前に文書{index}を保存してください。"
    translation = f"Mimiを終了する前に文書{index}を保存してください。"
    candidates = [
        candidate_id,
        f"other-a-{index}",
        f"other-b-{index}",
        f"reference-{index}",
    ]
    judgments = []
    for judge in ("judge-a", "judge-b"):
        judgments.append(
            {
                "source_id": source_id,
                "judge_model": judge,
                "assessments": {
                    value: assessment(value, value == candidate_id)
                    for value in candidates
                },
            }
        )
    policy = {
        "minimum_adequacy": 4,
        "minimum_fluency": 3,
        "minimum_terminology": 3,
        "require_no_error_tags": True,
        "require_no_critical_error": True,
        "require_protected_tokens_preserved": True,
        "require_unique_best_per_judge": True,
        "require_matching_selection": True,
        "candidate_count": 4,
        "licensed_reference_blinded_when_available": True,
        "selected_candidate_must_be_teacher_only": True,
    }
    approved = {
        "source_id": source_id,
        "candidate_id": candidate_id,
        "candidate_origin": "teacher",
        "source_language": "en-US",
        "target_language": "ja-JP",
        "source": source,
        "translation": translation,
        "licensed_reference": reference,
        "reference_provenance": "offline licensed reference",
        "source_license": "project-owned",
        "source_provenance": "offline fixture",
        "domain": "mimi-product-ui",
        "teacher_model": "gpt-5.6-sol",
        "teacher_response_id": f"response-{index}",
        "judge_model_ids": ["judge-a", "judge-b"],
        "automated_judgments": judgments,
        "automated_consensus_policy": policy,
        "review_status": "two-judge-reference-anchored",
        "promotion_eligible": True,
    }
    seed = {
        "id": source_id,
        "source_language": "en-US",
        "target_language": "ja-JP",
        "source": source,
        "reference_translation": reference,
    }
    return approved, seed


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-anchored-dataset-test-") as temporary:
        work = Path(temporary)
        approved_values, seeds = zip(*(approved_row(index) for index in (1, 2)))
        approved = work / "approved.jsonl"
        pilot = work / "pilot.jsonl"
        protected = work / "protected.jsonl"
        replay = work / "replay"
        output = work / "output"
        write_jsonl(approved, list(approved_values))
        write_jsonl(pilot, list(seeds))
        write_jsonl(
            protected,
            [{"id": "heldout", "source": "Completely unrelated.", "references": ["完全に無関係です。"]}],
        )
        train: list[dict] = []
        for origin_index, (origin, domain, license_name) in enumerate(ORIGINS):
            for row_index in range(3):
                train.append(
                    {
                        "id": f"replay-{origin_index}-{row_index}",
                        "source_id": f"replay-{origin_index}-{row_index}",
                        "source_language": "en-US",
                        "target_language": "ja-JP",
                        "source": f"Replay source {origin_index} item {row_index}.",
                        "target": f"再生資料{origin_index}項目{row_index}です。",
                        "origin": origin,
                        "domain": domain,
                        "source_license": license_name,
                        "source_provenance": "offline replay fixture",
                    }
                )
        valid = [
            {
                "id": "valid-1",
                "source_id": "valid-1",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "source": "A separate validation sentence.",
                "target": "別の検証文です。",
                "origin": "human-alt-parallel",
                "domain": "human-translated-news",
                "source_license": "CC-BY-4.0",
                "source_provenance": "offline validation fixture",
                "promotion_eligible": True,
            }
        ]
        make_dataset(replay, train, valid)
        result = subprocess.run(
            [
                "python3",
                "scripts/translation/build_reference_anchored_distillation_dataset.py",
                str(approved),
                str(pilot),
                str(replay),
                str(output),
                "--direction",
                "en-ja",
                "--replay-dataset",
                str(replay),
                "--protected-suite",
                str(protected),
                "--minimum-approved",
                "2",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["promotion_eligible"] is True
        assert manifest["counts"]["approved_teacher_targets"] == 2
        assert manifest["counts"]["same_source_human_anchors"] == 2
        assert manifest["counts"]["general_human_replay"] == 4
        assert manifest["counts"]["train"] == 8
        assert manifest["counts"]["valid_human_only"] == 1
        assert manifest["synthetic_policy"]["actual_synthetic_fraction"] == 0.25
        assert manifest["target_source"] == (
            "reviewed-canonical-teacher-and-licensed-human-reference-mixture"
        )
        assert sum(manifest["effective_licenses"]["train"].values()) == 8
        assert sum(manifest["effective_licenses"]["valid"].values()) == 1
        assert manifest["teacher_models"]["sequence_teachers"] == {
            "gpt-5.6-sol": 2
        }
        assert len(manifest["teacher_models"]["admission_judges"]) == 2
        rows = [
            json.loads(line)
            for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert sum(
            row["origin"] == "automated-gpt-teacher-reference-anchored"
            for row in rows
        ) == 2
        assert all(row["promotion_eligible"] is True for row in rows)

    print("Mimi reference-anchored distillation dataset contract passed.")


if __name__ == "__main__":
    main()
