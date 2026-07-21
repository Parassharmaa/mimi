#!/usr/bin/env python3
"""Contract tests for the hidden-reference local Qwen teacher pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import sacrebleu

from filter_local_reference_teacher import protected_tokens, valid_translation


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_report(
    suite: Path,
    engine_report: Path,
    engine: str,
    scores: dict[str, float],
) -> dict:
    signature_value = {
        "metric": "COMET-22",
        "modelRepository": "Unbabel/wmt22-comet-da",
        "modelRevision": "371e9839ca4e213dde891b066cf3080f75ec7e72",
        "modelLicense": "Apache-2.0",
        "package": "unbabel-comet",
        "packageVersion": "2.2.7",
        "setuptoolsVersion": "80.9.0",
        "precision": "float32",
        "multipleReferenceAggregation": "mean",
    }
    return {
        **signature_value,
        "signatureSHA256": hashlib.sha256(json.dumps(
            signature_value, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "engine": engine,
        "suiteSHA256": sha256(suite),
        "engineReportSHA256": sha256(engine_report),
        "results": [{"caseID": key, "score": value} for key, value in scores.items()],
    }


def main() -> None:
    assert protected_tokens("August 15: 2.1%(183)") == protected_tokens(
        "8月15日：2.1％（183）"
    )
    assert valid_translation(
        "August 15: 2.1%(183)", "8月15日：2.1％（183）", "ja-JP"
    ) is None
    assert protected_tokens("August 15") != protected_tokens("9月15日")
    with tempfile.TemporaryDirectory(prefix="mimi-reference-teacher-test-") as temporary:
        root = Path(temporary)
        seeds = root / "seeds.jsonl"
        protected = root / "protected.jsonl"
        base = root / "base"
        suite = root / "suite.jsonl"
        baseline = root / "baseline.json"
        examples = [
            {
                "id": "en-ja-accept", "source": "The station is nearby.",
                "reference_translation": "駅は近くにあります。", "student_hypothesis": "駅があります。",
                "student_chrf_pp": sacrebleu.sentence_chrf(
                    "駅があります。", ["駅は近くにあります。"], word_order=2
                ).score,
                "source_language": "en-US", "target_language": "ja-JP", "domain": "fixture",
                "license": "CC-BY-SA-3.0", "provenance": "fixture source",
                "reference_provenance": "fixture reference", "split": "train",
            },
            {
                "id": "ja-en-accept", "source": "会議は三時に始まります。",
                "reference_translation": "The meeting starts at three.",
                "student_hypothesis": "Meeting starts.",
                "student_chrf_pp": sacrebleu.sentence_chrf(
                    "Meeting starts.", ["The meeting starts at three."], word_order=2
                ).score,
                "source_language": "ja-JP", "target_language": "en-US", "domain": "fixture",
                "license": "CC-BY-SA-3.0", "provenance": "fixture source",
                "reference_provenance": "fixture reference", "split": "train",
            },
            {
                "id": "existing", "source": "Already trained.",
                "reference_translation": "学習済みです。", "student_hypothesis": "学習しました。",
                "student_chrf_pp": 20.0, "source_language": "en-US", "target_language": "ja-JP",
                "domain": "fixture", "license": "CC-BY-SA-3.0", "provenance": "fixture",
                "reference_provenance": "fixture", "split": "train",
            },
            {
                "id": "protected", "source": "A protected source.",
                "reference_translation": "保護された参照です。", "student_hypothesis": "保護されています。",
                "student_chrf_pp": 20.0, "source_language": "en-US", "target_language": "ja-JP",
                "domain": "fixture", "license": "CC-BY-SA-3.0", "provenance": "fixture",
                "reference_provenance": "fixture", "split": "train",
            },
        ]
        write_jsonl(seeds, examples)
        write_jsonl(protected, [{
            "id": "heldout", "source": "A protected source.",
            "references": ["保護された参照です。"],
        }])
        write_jsonl(base / "train.jsonl", [{
            "id": "base-en-train", "source": "Already trained.", "target": "学習済みです。",
            "source_language": "en-US", "target_language": "ja-JP", "origin": "fixture-base",
            "source_license": "CC0-1.0",
        }])
        write_jsonl(base / "valid.jsonl", [{
            "id": "base-en-valid", "source": "Validation only.", "target": "検証のみです。",
            "source_language": "en-US", "target_language": "ja-JP", "origin": "fixture-base",
            "source_license": "CC0-1.0",
        }])
        ja_base = root / "ja-base"
        write_jsonl(ja_base / "train.jsonl", [{
            "id": "base-ja-train", "source": "既存の学習文です。", "target": "Existing training text.",
            "source_language": "ja-JP", "target_language": "en-US", "origin": "fixture-base",
            "source_license": "CC0-1.0",
        }])
        write_jsonl(ja_base / "valid.jsonl", [{
            "id": "base-ja-valid", "source": "検証文です。", "target": "Validation text.",
            "source_language": "ja-JP", "target_language": "en-US", "origin": "fixture-base",
            "source_license": "CC0-1.0",
        }])
        subprocess.run([
            "python3", "scripts/translation/prepare_local_reference_teacher_suite.py",
            str(seeds), str(suite), str(baseline),
            "--protected-suite", str(protected),
            "--exclude-dataset", str(base), "--exclude-dataset", str(ja_base),
        ], check=True, capture_output=True, text=True)
        suite_rows = [json.loads(line) for line in suite.read_text().splitlines()]
        assert [row["id"] for row in suite_rows] == ["en-ja-accept", "ja-en-accept"]
        assert all(row["referenceExposedToTeacher"] is False for row in suite_rows)
        manifest = json.loads((root / "suite.jsonl.manifest.json").read_text())
        assert manifest["counts"]["rejected"] == {
            "existing-student-source": 1,
            "near-protected-evaluation": 1,
        }

        teacher = root / "teacher.json"
        hypotheses = {
            "en-ja-accept": "駅は近くにあります。",
            "ja-en-accept": "The meeting starts at three.",
        }
        teacher.write_text(json.dumps({
            "engine": "teacher",
            "claimEligible": False,
            "referenceExposedToTeacher": False,
            "studentHypothesisExposedToTeacher": False,
            "reasoningTraceRequestedOrStored": False,
            "modelRepository": "mlx-community/Qwen3-8B-4bit",
            "modelRevision": "545dc4251c05440727734bcd94334791f6ab0192",
            "modelLicense": "Apache-2.0",
            "suite": {"sha256": sha256(suite)},
            "results": [{
                "caseID": row["id"], "sourceLanguage": row["sourceLanguage"],
                "targetLanguage": row["targetLanguage"], "domain": row["domain"],
                "source": row["source"], "references": row["references"],
                "hypothesis": hypotheses[row["id"]],
            } for row in suite_rows],
        }, ensure_ascii=False), encoding="utf-8")
        teacher_comet_path, student_comet_path = root / "teacher-comet.json", root / "student-comet.json"
        teacher_comet_path.write_text(json.dumps(metric_report(suite, teacher, "teacher", {
            "en-ja-accept": 0.92, "ja-en-accept": 0.91,
        })), encoding="utf-8")
        student_comet_path.write_text(json.dumps(metric_report(suite, baseline, "frozen-seed-student-baseline", {
            "en-ja-accept": 0.80, "ja-en-accept": 0.75,
        })), encoding="utf-8")
        output = root / "accepted.jsonl"
        subprocess.run([
            "python3", "scripts/translation/filter_local_reference_teacher.py",
            str(suite), str(teacher), str(teacher_comet_path), str(student_comet_path), str(output),
        ], check=True, capture_output=True, text=True)
        accepted = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(accepted) == 2
        assert {row["source_language"] for row in accepted} == {"en-US", "ja-JP"}
        assert all(row["promotion_eligible"] is False for row in accepted)
        floor_failure = subprocess.run([
            "python3", "scripts/translation/filter_local_reference_teacher.py",
            str(suite), str(teacher), str(teacher_comet_path), str(student_comet_path),
            str(root / "floor-failure.jsonl"),
            "--minimum-accepted-per-domain-direction", "2",
        ], capture_output=True, text=True)
        assert floor_failure.returncode != 0
        assert "domain/direction floors" in floor_failure.stderr
        assert not (root / "floor-failure.jsonl").exists()
        failure_report = json.loads(
            (root / "floor-failure.jsonl.floor-failure.json").read_text()
        )
        assert failure_report["training_rows_emitted"] is False
        assert failure_report["counts"]["potentially_accepted"] == 2

        retry_teacher = root / "retry-teacher.json"
        retry_report = json.loads(teacher.read_text())
        retry_report["engine"] = "retry-teacher"
        retry_teacher.write_text(json.dumps(retry_report), encoding="utf-8")
        retry_comet_path = root / "retry-teacher-comet.json"
        retry_comet_path.write_text(json.dumps(metric_report(
            suite,
            retry_teacher,
            "retry-teacher",
            {"en-ja-accept": 0.95, "ja-en-accept": 0.70},
        )), encoding="utf-8")
        selected_teacher = root / "selected-teacher.json"
        selected_comet = root / "selected-teacher-comet.json"
        subprocess.run([
            "python3", "scripts/translation/select_local_teacher_candidates.py",
            str(suite), str(student_comet_path), str(selected_teacher), str(selected_comet),
            "--candidate", str(teacher), str(teacher_comet_path),
            "--candidate", str(retry_teacher), str(retry_comet_path),
        ], check=True, capture_output=True, text=True)
        selected_report = json.loads(selected_teacher.read_text())
        assert selected_report["candidateSelection"]["inputs"][0]["selectedCases"] == 1
        assert selected_report["candidateSelection"]["inputs"][1]["selectedCases"] == 1
        selected_metric_rows = {
            row["caseID"]: row["score"]
            for row in json.loads(selected_comet.read_text())["results"]
        }
        assert selected_metric_rows == {"en-ja-accept": 0.95, "ja-en-accept": 0.91}
        selected_output = root / "selected-accepted.jsonl"
        subprocess.run([
            "python3", "scripts/translation/filter_local_reference_teacher.py",
            str(suite), str(selected_teacher), str(selected_comet),
            str(student_comet_path), str(selected_output),
        ], check=True, capture_output=True, text=True)
        assert len(selected_output.read_text().splitlines()) == 2

        for direction, base_path in (("en-ja", base), ("ja-en", ja_base)):
            for target_source in ("qwen", "human-reference"):
                dataset = root / f"dataset-{direction}-{target_source}"
                subprocess.run([
                    "python3", "scripts/translation/build_reference_teacher_ablation.py",
                    str(output), str(suite), str(base_path), str(dataset),
                    "--direction", direction, "--target-source", target_source,
                    "--protected-suite", str(protected),
                ], check=True, capture_output=True, text=True)
                dataset_manifest = json.loads((dataset / "manifest.json").read_text())
                assert dataset_manifest["counts"]["teacher_train"] == 1
                assert dataset_manifest["direction"] == direction
                teacher_row = next(
                    row for row in [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
                    if row.get("source_id") in {"en-ja-accept", "ja-en-accept"}
                )
                expected_target = (
                    hypotheses[teacher_row["source_id"]]
                    if target_source == "qwen"
                    else next(row for row in suite_rows if row["id"] == teacher_row["source_id"])["references"][0]
                )
                assert teacher_row["target"] == expected_target

        # A copied source with excellent fake metrics must still fail deterministic checks.
        bad_teacher = json.loads(teacher.read_text())
        bad_teacher["results"][0]["hypothesis"] = bad_teacher["results"][0]["source"]
        bad_teacher_path = root / "bad-teacher.json"
        bad_teacher_path.write_text(json.dumps(bad_teacher), encoding="utf-8")
        bad_teacher_comet_path = root / "bad-teacher-comet.json"
        bad_teacher_comet_path.write_text(json.dumps(metric_report(
            suite, bad_teacher_path, "teacher", {
                "en-ja-accept": 0.92, "ja-en-accept": 0.91,
            }
        )), encoding="utf-8")
        bad_output = root / "bad-accepted.jsonl"
        subprocess.run([
            "python3", "scripts/translation/filter_local_reference_teacher.py",
            str(suite), str(bad_teacher_path), str(bad_teacher_comet_path), str(student_comet_path), str(bad_output),
        ], check=True, capture_output=True, text=True)
        bad_manifest = json.loads((root / "bad-accepted.jsonl.manifest.json").read_text())
        assert bad_manifest["counts"]["accepted"] == 1
        assert bad_manifest["counts"]["rejected"] == {"empty-or-source-copy": 1}
    print("Local reference teacher contract passed.")


if __name__ == "__main__":
    main()
