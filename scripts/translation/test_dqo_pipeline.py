#!/usr/bin/env python3
"""Offline contracts for conservative preference data and gated Marian DQO."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "translation"))
from train_marian_dqo import dqo_loss, sequence_log_probabilities, validate_supervised_win  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {' '.join(args)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def result(case: dict, hypothesis: str) -> dict:
    return {
        "caseID": case["id"],
        "sourceLanguage": case["sourceLanguage"],
        "targetLanguage": case["targetLanguage"],
        "domain": case["domain"],
        "source": case["source"],
        "references": case["references"],
        "claimEligible": case["claimEligible"],
        "hypothesis": hypothesis,
        "latencySeconds": 0.01,
        "warmLatencySeconds": [0.01, 0.01, 0.01],
    }


def main() -> None:
    logits = torch.tensor(
        [[[5.0, 0.0], [0.0, 5.0]], [[0.0, 5.0], [5.0, 0.0]]]
    )
    labels = torch.tensor([[0, 1], [1, -100]])
    logps = sequence_log_probabilities(logits, labels)
    assert logps.shape == (2,)
    assert torch.isfinite(logps).all()
    policy_chosen = torch.tensor([-0.1, -0.2])
    policy_rejected = torch.tensor([-1.0, -0.8])
    reference_chosen = torch.tensor([-0.5, -0.5])
    reference_rejected = torch.tensor([-0.6, -0.6])
    preferred_loss, margin = dqo_loss(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected, 0.1
    )
    reversed_loss, _ = dqo_loss(
        policy_rejected, policy_chosen, reference_chosen, reference_rejected, 0.1
    )
    assert preferred_loss < reversed_loss
    assert bool((margin > 0).all())

    with tempfile.TemporaryDirectory(prefix="mimi-dqo-test-") as temporary:
        work = Path(temporary)
        queue_path = work / "queue.jsonl"
        approved_path = work / "approved.jsonl"
        protected_path = work / "protected.jsonl"
        preferences = work / "preferences"
        queue: list[dict] = []
        approved: list[dict] = []
        for index in range(20):
            source_id = f"source-{index:03d}"
            candidates = [
                (f"candidate-{index:03d}-a", f"これは正確な翻訳です {index}。"),
                (f"candidate-{index:03d}-b", f"こちらも自然な訳です {index}。"),
                (f"candidate-{index:03d}-c", f"これは不適切な候補です {index}。"),
            ]
            for candidate_id, translation in candidates:
                queue.append(
                    {
                        "candidate_id": candidate_id,
                        "source_id": source_id,
                        "source_language": "en-US",
                        "target_language": "ja-JP",
                        "source": f"A distinct supervised preference source number {index}.",
                        "translation": translation,
                        "domain": "everyday-conversation",
                        "source_license": "CC-BY-4.0",
                        "source_provenance": f"offline fixture {index}",
                        "teacher_model": "teacher-fixture",
                    }
                )
            reviews = [
                {
                    "source_id": source_id,
                    "reviewer_id": reviewer,
                    "decision": "select",
                    "selected_candidate_id": candidates[0][0],
                    "approved_alternative_candidate_id": candidates[1][0],
                    "critical_error": False,
                }
                for reviewer in ("reviewer-a", "reviewer-b")
            ]
            approved.append(
                {
                    **queue[-3],
                    "review_status": "two-reviewer-selected",
                    "reviewer_ids": ["reviewer-a", "reviewer-b"],
                    "source_level_reviews": reviews,
                }
            )
        write_jsonl(queue_path, queue)
        write_jsonl(approved_path, approved)
        write_jsonl(
            protected_path,
            [{
                "id": "unrelated-protected",
                "source": "Completely unrelated benchmark sentence.",
                "references": ["保護された無関係の文です。"],
            }],
        )
        run(
            "python3",
            "scripts/translation/build_dqo_preferences.py",
            str(queue_path),
            str(approved_path),
            str(protected_path),
            str(preferences),
            "--direction", "en-ja",
            "--validation-fraction", "0.25",
            "--minimum-pairs", "1",
        )
        train = [json.loads(line) for line in (preferences / "train.jsonl").read_text().splitlines()]
        valid = [json.loads(line) for line in (preferences / "valid.jsonl").read_text().splitlines()]
        assert train and valid
        assert not ({row["source_id"] for row in train} & {row["source_id"] for row in valid})
        assert all(row["rejected_candidate_id"].endswith("-c") for row in train + valid)
        manifest = json.loads((preferences / "manifest.json").read_text())
        assert manifest["pairs"] == 20
        assert manifest["policy"]["approved_diverse_alternatives_as_losers"] is False

        checkpoint = work / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"supervised checkpoint fixture")
        win_report = work / "supervised-win.json"
        win = {
            "schemaVersion": 1,
            "status": "supervised-win-approved",
            "approved": True,
            "direction": "en-ja",
            "candidateModelRevision": "fixture-revision",
            "supervisedCheckpoint": {
                "modelSHA256": sha256(checkpoint / "model.safetensors")
            },
            "gates": [
                {"name": name, "passed": True}
                for name in sorted({
                    "reviewed-development-chrf-win",
                    "blind-human-development-win",
                    "no-new-critical-errors",
                    "general-retention",
                    "exact-bundle-checkpoint-binding",
                })
            ],
        }
        win_report.write_text(json.dumps(win), encoding="utf-8")
        assert validate_supervised_win(win_report, checkpoint, "en-ja")["approved"] is True
        win["gates"][0]["passed"] = False
        win_report.write_text(json.dumps(win), encoding="utf-8")
        try:
            validate_supervised_win(win_report, checkpoint, "en-ja")
        except SystemExit as error:
            assert "does not pass every required" in str(error)
        else:
            raise AssertionError("DQO gate accepted a failed supervised prerequisite")

        development_suite = work / "development.jsonl"
        retention_suite = work / "retention.jsonl"
        development = [
            {
                "id": f"dev-{index}",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "everyday-conversation",
                "source": f"Development source {index}",
                "references": [f"正しい開発翻訳{index}です。"],
                "claimEligible": False,
            }
            for index in range(12)
        ]
        retention = [
            {
                "id": f"retention-{index}",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "domain": "general-retention",
                "source": f"Retention source {index}",
                "references": [f"一般翻訳{index}です。"],
                "claimEligible": False,
            }
            for index in range(6)
        ]
        write_jsonl(development_suite, development)
        write_jsonl(retention_suite, retention)

        bundle = work / "bundle"
        (bundle / "en-ja").mkdir(parents=True)
        direction_manifest = {
            "format": "mimi-mlx-marian-v1",
            "direction": "en-ja",
            "source_weights_sha256": sha256(checkpoint / "model.safetensors"),
        }
        direction_manifest_path = bundle / "en-ja" / "manifest.json"
        direction_manifest_path.write_text(json.dumps(direction_manifest), encoding="utf-8")
        root_manifest = {
            "format": "mimi-mlx-marian-pair-v1",
            "files": {
                "en-ja/manifest.json": {
                    "bytes": direction_manifest_path.stat().st_size,
                    "sha256": sha256(direction_manifest_path),
                }
            },
        }
        root_manifest_path = bundle / "manifest.json"
        root_manifest_path.write_text(json.dumps(root_manifest), encoding="utf-8")
        revision = f"pair-manifest-sha256:{sha256(root_manifest_path)}"

        def report(path: Path, engine: str, cases: list[dict], good: bool) -> None:
            rows = [
                result(
                    case,
                    case["references"][0] if good else "意味が異なる出力です。",
                )
                for case in cases
            ]
            path.write_text(
                json.dumps({
                    "schemaVersion": 1,
                    "engine": engine,
                    "modelRevision": revision if engine == "candidate-sft" else "base-fixture",
                    "results": rows,
                }),
                encoding="utf-8",
            )

        candidate_dev = work / "candidate-dev.json"
        base_dev = work / "base-dev.json"
        candidate_retention = work / "candidate-retention.json"
        base_retention = work / "base-retention.json"
        report(candidate_dev, "candidate-sft", development, True)
        report(base_dev, "base-student", development, False)
        report(candidate_retention, "candidate-sft", retention, True)
        report(base_retention, "base-student", retention, True)

        review_directory = work / "dev-review"
        run(
            "python3", "scripts/translation/prepare_engine_comparison_packets.py",
            str(candidate_dev), str(base_dev), str(review_directory),
            "--reviewer", "reviewer-a", "--reviewer", "reviewer-b",
            "--baseline-key", "base",
        )
        assignments_path = review_directory / "sealed-assignments.jsonl"
        assignment_rows = [
            json.loads(line) for line in assignments_path.read_text().splitlines()
        ]
        assert all(
            {row["outputAEngine"], row["outputBEngine"]} == {"candidate", "base"}
            for row in assignment_rows
        )
        assignment_index = {
            (row["reviewerID"], row["caseID"]): row for row in assignment_rows
        }
        response_paths: dict[str, Path] = {}
        for reviewer in ("reviewer-a", "reviewer-b"):
            responses: list[dict] = []
            for case in development:
                case_id = case["id"]
                assignment = assignment_index[(reviewer, case_id)]
                score = {
                    "candidate": {"adequacy": 4, "fluency": 4, "terminology": 2, "criticalError": False},
                    "base": {"adequacy": 0, "fluency": 1, "terminology": 0, "criticalError": True},
                }
                responses.append({
                    "reviewerID": reviewer,
                    "caseID": case_id,
                    "blinded": True,
                    "outputA": score[assignment["outputAEngine"]],
                    "outputB": score[assignment["outputBEngine"]],
                })
            response_path = review_directory / f"{reviewer}.responses.jsonl"
            write_jsonl(response_path, responses)
            response_paths[reviewer] = response_path
        review_a, review_b = response_paths["reviewer-a"], response_paths["reviewer-b"]
        evaluated_win = work / "evaluated-win.json"
        run(
            "python3", "scripts/translation/evaluate_supervised_win.py",
            str(development_suite), str(candidate_dev), str(base_dev),
            str(retention_suite), str(candidate_retention), str(base_retention),
            str(assignments_path), str(review_a), str(review_b),
            str(bundle), str(checkpoint), str(evaluated_win),
            "--direction", "en-ja", "--paired-bootstrap-samples", "1000",
        )
        evaluated = json.loads(evaluated_win.read_text())
        assert evaluated["status"] == "supervised-win-approved"
        assert evaluated["approved"] is True
        assert all(gate["passed"] for gate in evaluated["gates"])

    print("DQO preference, objective, and supervised-win gate contracts passed.")


if __name__ == "__main__":
    main()
