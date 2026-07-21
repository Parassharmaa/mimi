#!/usr/bin/env python3
"""Offline contract test for the GPT candidate/filter/two-reviewer pipeline."""

from __future__ import annotations

import importlib.util
import json
import hashlib
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def load_batch_runner():
    path = ROOT / "scripts/translation/run_synthetic_batch.py"
    spec = importlib.util.spec_from_file_location("mimi_batch_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run(*arguments: str) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-synthetic-test-") as temporary:
        work = Path(temporary)
        seed = {
            "id": "synthetic-smoke-001",
            "split": "train",
            "source_language": "en-US",
            "target_language": "ja-JP",
            "domain": "macos-and-technical-ui",
            "source": "Please reopen Mimi and press Start.",
            "license": "CC0-1.0",
            "provenance": "offline pipeline contract test",
            "reference_translation": "Mimiを開き直して、「開始」を押してください。",
            "reference_provenance": "offline human reference fixture",
        }
        seeds = work / "seeds.jsonl"
        requests = work / "requests.jsonl"
        batch_output = work / "batch-output.jsonl"
        queue = work / "review-queue.jsonl"
        judge_requests = work / "judge-requests.jsonl"
        judge_output = work / "judge-output.jsonl"
        judge_priority = work / "judge-priority.jsonl"
        reviews = work / "reviews.jsonl"
        approved = work / "approved.jsonl"
        review_packets = work / "review-packets"
        approved_selections = work / "approved-selections.jsonl"
        disagreements = work / "disagreements.jsonl"
        adjudications = work / "adjudications.jsonl"
        additional_seeds = work / "additional-seeds.jsonl"
        merged_seeds = work / "merged-seeds.jsonl"
        kftt = work / "kftt"
        parallel = work / "parallel"
        parallel_alt = work / "parallel-alt"
        distilled = work / "distilled-en-ja"
        distilled_diverse = work / "distilled-en-ja-diverse"
        write_jsonl(seeds, [seed])

        write_jsonl(additional_seeds, [{
            "id": "synthetic-smoke-ja-en-001",
            "split": "train",
            "source_language": "ja-JP",
            "target_language": "en-US",
            "domain": "travel-and-service-conversation",
            "source": "窓の近くの席をお願いします。",
            "license": "CC-BY-4.0",
            "provenance": "offline source-only seed fixture",
        }])
        run(
            "python3", "scripts/translation/merge_distillation_seeds.py",
            "Research/translation/benchmark/canary.jsonl", str(merged_seeds),
            str(seeds), str(additional_seeds),
        )
        merged = [json.loads(line) for line in merged_seeds.read_text(encoding="utf-8").splitlines()]
        assert len(merged) == 2
        merged_manifest = json.loads(
            merged_seeds.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        assert merged_manifest["directions"] == {"en-ja": 1, "ja-en": 1}

        run("python3", "scripts/translation/prepare_synthetic_batch.py", str(seeds), str(requests))
        validation = subprocess.run(
            [
                "python3", "scripts/translation/run_synthetic_batch.py",
                "validate", str(requests),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        validation_result = json.loads(validation.stdout)
        assert validation_result["request_count"] == 1
        assert validation_result["request_sha256"] == hashlib.sha256(requests.read_bytes()).hexdigest()
        assert validation_result["model"] == "gpt-5.6-sol"

        refused_submission = subprocess.run(
            [
                "python3", "scripts/translation/run_synthetic_batch.py",
                "submit", str(requests), str(work / "batch-state.json"),
                "--confirm-input-sha256", "0" * 64,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert refused_submission.returncode != 0
        assert "confirmation does not match" in refused_submission.stderr
        assert not (work / "batch-state.json").exists()

        batch_runner = load_batch_runner()
        request_sha256 = validation_result["request_sha256"]

        class FakeModel:
            def __init__(self, **values):
                self.values = values
                for key, value in values.items():
                    setattr(self, key, value)

            def model_dump(self, mode="json"):
                assert mode == "json"
                return self.values

        class FakeContent:
            def __init__(self, content):
                self.content = content

            def read(self):
                return self.content

        class FakeFiles:
            def __init__(self):
                self.output = b""

            def create(self, *, file, purpose):
                assert purpose == "batch"
                assert file.read() == requests.read_bytes()
                return FakeModel(
                    id="file-offline-input", object="file", purpose="batch",
                    bytes=requests.stat().st_size,
                )

            def content(self, file_id):
                assert file_id == "file-offline-output"
                return FakeContent(self.output)

        class FakeBatches:
            def __init__(self):
                self.create_arguments = None
                self.current = None

            def list(self, limit):
                assert limit == 100
                return []

            def create(self, **arguments):
                self.create_arguments = arguments
                self.current = FakeModel(
                    id="batch-offline", object="batch", status="validating",
                    input_file_id="file-offline-input", output_file_id=None,
                    error_file_id=None, request_counts={"total": 0, "completed": 0, "failed": 0},
                    metadata=arguments["metadata"],
                )
                return self.current

            def retrieve(self, batch_id):
                assert batch_id == "batch-offline"
                return self.current

        fake_client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())
        batch_runner.api_client = lambda: fake_client
        state_path = work / "offline-batch-state.json"
        batch_runner.submit_command(SimpleNamespace(
            requests=requests,
            state=state_path,
            confirm_input_sha256=request_sha256,
        ))
        assert state_path.exists()
        assert fake_client.batches.create_arguments == {
            "input_file_id": "file-offline-input",
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": {
                "pipeline": "mimi-translation-v1",
                "request_sha256": request_sha256,
            },
        }

        fake_client.files.output = (
            json.dumps({"custom_id": seed["id"], "response": {"body": {"id": "resp-offline"}}})
            + "\n"
        ).encode()
        fake_client.batches.current = FakeModel(
            id="batch-offline", object="batch", status="completed",
            input_file_id="file-offline-input", output_file_id="file-offline-output",
            error_file_id=None, request_counts={"total": 1, "completed": 1, "failed": 0},
            metadata={"pipeline": "mimi-translation-v1", "request_sha256": request_sha256},
        )
        collected_output = work / "collected-output.jsonl"
        batch_runner.collect_command(SimpleNamespace(
            state=state_path,
            output=collected_output,
            error_output=work / "collected-errors.jsonl",
        ))
        assert collected_output.read_bytes() == fake_client.files.output
        collected_state = json.loads(state_path.read_text(encoding="utf-8"))
        assert collected_state["phase"] == "collected"
        assert collected_state["collection"]["complete_without_request_errors"] is True
        assert collected_state["collection"]["output_count"] == 1

        request = json.loads(requests.read_text(encoding="utf-8"))
        assert request["body"]["model"] == "gpt-5.6-sol"
        assert request["body"]["store"] is False
        assert request["url"] == "/v1/responses"
        teacher_input = request["body"]["input"][1]["content"]
        assert "reference_translation" not in teacher_input
        assert "offline human reference fixture" not in teacher_input
        assert "student_hypothesis" not in teacher_input

        candidates = {
            "source_id": seed["id"],
            "translation_brief": {
                "register": "polite",
                "terms": [{"source": "Start", "target": "開始"}],
                "preserve": ["Mimi"],
                "ambiguities": [],
            },
            "candidates": [
                {"translation": "Mimiを開き直して、「開始」を押してください。", "style": "natural-spoken", "risk_tags": []},
                {"translation": "Mimiを再起動し、「開始」を押してください。", "style": "concise-caption", "risk_tags": ["terminology"]},
                {"translation": "Mimiをもう一度開いて、開始ボタンを押してください。", "style": "meaning-conservative", "risk_tags": []},
            ],
        }
        write_jsonl(batch_output, [{
            "custom_id": seed["id"],
            "response": {"body": {
                "id": "resp_offline_fixture",
                "model": "gpt-5.6-sol-2026-07-01",
                "system_fingerprint": "offline-fixture",
                "output_text": json.dumps(candidates, ensure_ascii=False),
            }},
        }])
        run(
            "python3", "scripts/translation/filter_synthetic_batch.py",
            str(seeds), str(batch_output),
            "Research/translation/benchmark/canary.jsonl", str(queue),
        )
        queued = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
        assert len(queued) == 3
        assert all(row["review_status"] == "pending-two-independent-reviews" for row in queued)
        assert all(row["translation_brief"]["register"] == "polite" for row in queued)
        assert all(row["licensed_reference"] == seed["reference_translation"] for row in queued)

        duplicate_batch = work / "duplicate-batch-output.jsonl"
        duplicate_queue = work / "duplicate-review-queue.jsonl"
        duplicate_candidates = {
            **candidates,
            "candidates": [
                {**candidate, "translation": candidates["candidates"][0]["translation"]}
                for candidate in candidates["candidates"]
            ],
        }
        write_jsonl(duplicate_batch, [{
            "custom_id": seed["id"],
            "response": {"body": {
                "id": "resp_duplicate_fixture",
                "model": "gpt-5.6-sol-2026-07-01",
                "output_text": json.dumps(duplicate_candidates, ensure_ascii=False),
            }},
        }])
        run(
            "python3", "scripts/translation/filter_synthetic_batch.py",
            str(seeds), str(duplicate_batch),
            "Research/translation/benchmark/canary.jsonl", str(duplicate_queue),
        )
        assert not duplicate_queue.read_text(encoding="utf-8")

        missing_batch = work / "missing-batch-output.jsonl"
        missing_queue = work / "missing-review-queue.jsonl"
        missing_batch.write_text("", encoding="utf-8")
        missing = subprocess.run(
            [
                "python3", "scripts/translation/filter_synthetic_batch.py",
                str(seeds), str(missing_batch),
                "Research/translation/benchmark/canary.jsonl", str(missing_queue),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert missing.returncode != 0
        assert "missing 1 seed results" in missing.stderr

        run(
            "python3", "scripts/translation/prepare_distillation_judge_batch.py",
            str(queue), str(judge_requests), "--model", "offline-fast-judge-fixture",
        )
        judge_request = json.loads(judge_requests.read_text(encoding="utf-8"))
        assert judge_request["body"]["store"] is False
        assert judge_request["body"]["model"] == "offline-fast-judge-fixture"
        judge_input = json.loads(judge_request["body"]["input"][1]["content"])
        assert len(judge_input["candidates"]) == 3
        assert "teacher_model" not in judge_request["body"]["input"][1]["content"]
        judge_validation = subprocess.run(
            [
                "python3", "scripts/translation/run_synthetic_batch.py",
                "validate", str(judge_requests),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        judge_contract = json.loads(judge_validation.stdout)
        assert judge_contract["pipeline"] == "mimi-translation-judge-v1"
        assert judge_contract["reasoning_effort"] == "low"

        judge_assessments = {
            "source_id": seed["id"],
            "assessments": [
                {
                    "candidate_id": row["candidate_id"],
                    "adequacy": 4 if index == 0 else 3,
                    "fluency": 4,
                    "terminology": 4 if index != 1 else 2,
                    "protected_tokens_preserved": True,
                    "critical_error": False,
                    "error_tags": [] if index != 1 else ["terminology"],
                }
                for index, row in enumerate(queued)
            ],
        }
        write_jsonl(judge_output, [{
            "custom_id": seed["id"],
            "response": {
                "status_code": 200,
                "body": {
                    "id": "resp_offline_judge",
                    "model": "offline-fast-judge-fixture-2026-07-01",
                    "system_fingerprint": "offline-judge-fixture",
                    "output_text": json.dumps(judge_assessments, ensure_ascii=False),
                },
            },
        }])
        run(
            "python3", "scripts/translation/prioritize_distillation_judgments.py",
            str(queue), str(judge_output), str(judge_priority),
        )
        priority_row = json.loads(judge_priority.read_text(encoding="utf-8"))
        assert priority_row["priority_rank"] == 1
        assert priority_row["priority_status"] == "automated-review-order-only-not-approval"

        write_jsonl(reviews, [
            {"candidate_id": queued[0]["candidate_id"], "reviewer_id": "reviewer-a", "decision": "accept"},
            {"candidate_id": queued[0]["candidate_id"], "reviewer_id": "reviewer-b", "decision": "accept"},
            {"candidate_id": queued[1]["candidate_id"], "reviewer_id": "reviewer-a", "decision": "accept"},
            {"candidate_id": queued[2]["candidate_id"], "reviewer_id": "reviewer-a", "decision": "reject"},
        ])
        run(
            "python3", "scripts/translation/approve_synthetic_reviews.py",
            str(queue), str(reviews), str(approved),
        )
        accepted = [json.loads(line) for line in approved.read_text(encoding="utf-8").splitlines()]
        assert len(accepted) == 1
        assert accepted[0]["review_status"] == "two-reviewer-accepted"
        assert accepted[0]["reviewer_ids"] == ["reviewer-a", "reviewer-b"]

        run(
            "python3", "scripts/translation/prepare_bilingual_review_packets.py",
            str(queue), str(review_packets), "--reviewer", "reviewer-a",
            "--reviewer", "reviewer-b", "--priority", str(judge_priority),
        )
        packet = (review_packets / "reviewer-a.packet.jsonl").read_text(encoding="utf-8")
        assert "teacher_model" not in packet
        assert "judge_model" not in packet
        assert "priority_rank" not in packet
        assert "risk_tags" not in packet
        assert '"style"' not in packet
        packet_manifest = json.loads((review_packets / "manifest.json").read_text(encoding="utf-8"))
        assert packet_manifest["source_order"].startswith("automated risk priority")
        for reviewer_id in ("reviewer-a", "reviewer-b"):
            response_path = review_packets / f"{reviewer_id}.responses.jsonl"
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response.update({
                "decision": "select",
                "selected_candidate_id": queued[0]["candidate_id"],
                "approved_alternative_candidate_id": queued[1]["candidate_id"],
            })
            write_jsonl(response_path, [response])
        write_jsonl(adjudications, [])
        run(
            "python3", "scripts/translation/approve_bilingual_selections.py",
            str(queue), str(review_packets / "reviewer-a.responses.jsonl"),
            str(review_packets / "reviewer-b.responses.jsonl"),
            str(approved_selections), str(disagreements),
            "--adjudications", str(adjudications),
        )
        source_approved = [
            json.loads(line)
            for line in approved_selections.read_text(encoding="utf-8").splitlines()
        ]
        assert len(source_approved) == 1
        assert source_approved[0]["review_status"] == "two-reviewer-selected"
        assert (
            source_approved[0]["approved_alternative"]["candidate_id"]
            == queued[1]["candidate_id"]
        )
        assert not disagreements.read_text(encoding="utf-8")

        reviewer_b_path = review_packets / "reviewer-b.responses.jsonl"
        reviewer_b = json.loads(reviewer_b_path.read_text(encoding="utf-8"))
        reviewer_b["selected_candidate_id"] = queued[1]["candidate_id"]
        reviewer_b["approved_alternative_candidate_id"] = queued[0]["candidate_id"]
        write_jsonl(reviewer_b_path, [reviewer_b])
        unresolved_approved = work / "unresolved-approved.jsonl"
        unresolved = work / "unresolved.jsonl"
        run(
            "python3", "scripts/translation/approve_bilingual_selections.py",
            str(queue), str(review_packets / "reviewer-a.responses.jsonl"),
            str(reviewer_b_path), str(unresolved_approved), str(unresolved),
            "--adjudications", str(adjudications),
        )
        assert not unresolved_approved.read_text(encoding="utf-8")
        assert len(unresolved.read_text(encoding="utf-8").splitlines()) == 1

        write_jsonl(adjudications, [{
            "source_id": seed["id"],
            "reviewer_id": "reviewer-c",
            "decision": "select",
            "selected_candidate_id": queued[2]["candidate_id"],
            "critical_error": False,
        }])
        adjudicated_approved = work / "adjudicated-approved.jsonl"
        adjudicated_disagreements = work / "adjudicated-disagreements.jsonl"
        run(
            "python3", "scripts/translation/approve_bilingual_selections.py",
            str(queue), str(review_packets / "reviewer-a.responses.jsonl"),
            str(reviewer_b_path), str(adjudicated_approved),
            str(adjudicated_disagreements), "--adjudications", str(adjudications),
        )
        adjudicated = json.loads(adjudicated_approved.read_text(encoding="utf-8"))
        assert adjudicated["review_status"] == "adjudicated"
        assert adjudicated["candidate_id"] == queued[2]["candidate_id"]
        assert adjudicated["reviewer_ids"] == ["reviewer-a", "reviewer-b", "reviewer-c"]
        assert adjudicated["adjudication"]["reviewer_id"] == "reviewer-c"
        assert not adjudicated_disagreements.read_text(encoding="utf-8")

        kftt.mkdir()
        kftt_metadata = {
            "source": "Kyoto Free Translation Task 1.0",
            "license": "CC-BY-SA-3.0",
            "attribution": "offline fixture",
            "direction": "en-ja",
        }
        write_jsonl(kftt / "train.jsonl", [{
            "messages": [
                {"role": "system", "content": "Translate."},
                {"role": "user", "content": "The temple garden opens in the morning."},
                {"role": "assistant", "content": "寺院の庭園は朝に開園します。"},
            ],
            "metadata": {**kftt_metadata, "source_id": "kftt-train-offline"},
        }])
        write_jsonl(kftt / "valid.jsonl", [{
            "messages": [
                {"role": "system", "content": "Translate."},
                {"role": "user", "content": "The exhibition includes three wooden statues."},
                {"role": "assistant", "content": "展覧会には3体の木像が含まれています。"},
            ],
            "metadata": {**kftt_metadata, "source_id": "kftt-valid-offline"},
        }])
        parallel.mkdir()
        parallel_common = {
            "source_language": "en-US",
            "target_language": "ja-JP",
            "origin": "mimi-shipped-ui-pair",
            "source_license": "project-owned",
            "source_provenance": "offline fixture",
        }
        write_jsonl(parallel / "train.jsonl", [{
            **parallel_common,
            "id": "parallel-train-offline",
            "source_id": "parallel-train-offline",
            "source": "Show captions above other apps.",
            "target": "他のアプリの上に字幕を表示します。",
        }])
        write_jsonl(parallel / "valid.jsonl", [{
            **parallel_common,
            "id": "parallel-valid-offline",
            "source_id": "parallel-valid-offline",
            "source": "Mimi is ready for transcription.",
            "target": "Mimiで文字起こしを開始できます。",
        }])
        parallel_alt.mkdir()
        alt_common = {
            "source_language": "en-US",
            "target_language": "ja-JP",
            "origin": "human-alt-parallel",
            "source_license": "CC-BY-4.0",
            "source_provenance": "offline ALT fixture",
        }
        write_jsonl(parallel_alt / "train.jsonl", [{
            **alt_common,
            "id": "alt-train-offline",
            "source_id": "alt-train-offline",
            "source": "The committee published its report today.",
            "target": "委員会は本日、報告書を公表した。",
        }])
        write_jsonl(parallel_alt / "valid.jsonl", [{
            **alt_common,
            "id": "alt-valid-offline",
            "source_id": "alt-valid-offline",
            "source": "The train service resumed this morning.",
            "target": "列車の運行は今朝再開した。",
        }])
        run(
            "python3", "scripts/translation/build_distillation_dataset.py",
            str(approved_selections), str(kftt),
            "Research/translation/benchmark/canary.jsonl", str(distilled),
            "--direction", "en-ja", "--validation-fraction", "0.0001",
            "--kftt-replay-multiplier", "1", "--minimum-synthetic-train", "0",
            "--minimum-synthetic-validation", "0", "--parallel-corpus-directory",
            str(parallel), "--parallel-corpus-directory", str(parallel_alt),
            "--maximum-parallel-train-per-corpus", "1",
            "--maximum-parallel-valid-per-corpus", "1",
        )
        manifest = json.loads((distilled / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["counts"]["synthetic_train"] == 1
        assert manifest["counts"]["kftt_replay_train"] == 1
        assert manifest["counts"]["kftt_valid"] == 1
        assert manifest["counts"]["parallel_train"] == 2
        assert manifest["counts"]["parallel_valid"] == 2
        assert len(manifest["inputs"]["parallel_corpora"]) == 2
        assert manifest["private_chain_of_thought_stored"] is False

        run(
            "python3", "scripts/translation/build_distillation_dataset.py",
            str(approved_selections), str(kftt),
            "Research/translation/benchmark/canary.jsonl", str(distilled_diverse),
            "--direction", "en-ja", "--validation-fraction", "0.0001",
            "--kftt-replay-multiplier", "1", "--minimum-synthetic-train", "0",
            "--minimum-synthetic-validation", "0", "--reviewed-target-mode",
            "sample-approved-diverse",
        )
        diverse_rows = [
            json.loads(line)
            for line in (distilled_diverse / "train.jsonl").read_text().splitlines()
        ]
        diverse_synthetic = next(
            row for row in diverse_rows if row["origin"] == "reviewed-gpt-teacher"
        )
        assert len(diverse_synthetic["target_variants"]) == 2
        diverse_manifest = json.loads(
            (distilled_diverse / "manifest.json").read_text()
        )
        assert diverse_manifest["reviewed_target_mode"] == "sample-approved-diverse"
        assert diverse_manifest["counts"]["synthetic_train_with_diverse_alternative"] == 1

    print("Mimi synthetic translation pipeline smoke passed.")


if __name__ == "__main__":
    main()
