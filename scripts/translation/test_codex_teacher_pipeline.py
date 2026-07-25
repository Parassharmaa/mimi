#!/usr/bin/env python3
"""Offline contract test for the authenticated Codex teacher transport."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/run_codex_teacher.py"
SCRIPTS = SCRIPT.parent


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def load_teacher():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("run_codex_teacher", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(source: dict) -> dict:
    if source["target_language"] == "ja-JP":
        translations = [
            "Mimiを再起動してから、開始してください。",
            "Mimiを再起動し、始めてください。",
            "Mimiを再起動した後で、開始してください。",
        ]
    else:
        translations = [
            "Please restart Mimi before continuing.",
            "Restart Mimi, then continue.",
            "Please reboot Mimi before you proceed.",
        ]
    return {
        "source_id": source["source_id"],
        "translation_brief": {
            "register": "polite",
            "terms": [{"source": "Mimi", "target": "Mimi"}],
            "preserve": ["Mimi"],
            "ambiguities": [],
        },
        "candidates": [
            {
                "translation": translation,
                "style": style,
                "risk_tags": [],
            }
            for style, translation in zip(
                (
                    "natural-spoken",
                    "concise-caption",
                    "meaning-conservative",
                ),
                translations,
            )
        ],
    }


def main() -> None:
    teacher = load_teacher()
    canonical = {
        "source_id": "canonical-1",
        "canonical_translation": "これは正規の翻訳です。",
        "risk_tags": [],
    }
    assert teacher.validate_teacher_payload(canonical, "canonical-1") == canonical
    try:
        teacher.validate_teacher_payload(
            {**canonical, "risk_tags": ["not-enumerated"]},
            "canonical-1",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("canonical teacher payload accepted an invalid risk tag")
    with tempfile.TemporaryDirectory(prefix="mimi-codex-teacher-test-") as temporary:
        work = Path(temporary)
        seeds = [
            {
                "id": "codex-teacher-en-ja-1",
                "split": "train",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "mimi-product-ui",
                "source": "Please restart Mimi before beginning.",
                "license": "project-owned",
                "provenance": "offline fixture",
                "reference_translation": "Mimiを再起動してから、開始してください。",
                "reference_provenance": "offline private teacher reference",
            },
            {
                "id": "codex-teacher-ja-en-1",
                "split": "train",
                "source_language": "ja-JP",
                "target_language": "en-US",
                "domain": "mimi-product-ui",
                "source": "続行する前にMimiを再起動してください。",
                "license": "project-owned",
                "provenance": "offline fixture",
                "reference_translation": "Please restart Mimi before continuing.",
                "reference_provenance": "offline private teacher reference",
            },
            {
                "id": "codex-teacher-en-ja-2",
                "split": "train",
                "source_language": "en-US",
                "target_language": "ja-JP",
                "domain": "mimi-product-ui",
                "source": "Restart Mimi and begin.",
                "license": "project-owned",
                "provenance": "offline fixture",
                "reference_translation": "Mimiを再起動し、開始してください。",
                "reference_provenance": "offline private teacher reference",
            },
        ]
        seeds_path = work / "seeds.jsonl"
        requests_path = work / "requests.jsonl"
        run_directory = work / "run"
        output_path = work / "output.jsonl"
        protected_path = work / "protected.jsonl"
        queue_path = work / "queue.jsonl"
        write_jsonl(seeds_path, seeds)
        subprocess.run(
            [
                "python3",
                "scripts/translation/prepare_synthetic_batch.py",
                str(seeds_path),
                str(requests_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        teacher.prepare_command(
            SimpleNamespace(
                requests=requests_path,
                run_directory=run_directory,
                maximum_items=2,
                maximum_source_characters=10_000,
            )
        )
        manifest = json.loads(
            (run_directory / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["source_only"] is True
        assert manifest["reasoning_traces_stored"] is False
        assert manifest["request_count"] == 3
        assert manifest["shard_count"] == 2

        real_run = teacher.subprocess.run
        prompts: list[str] = []

        def fake_run(command, **kwargs):
            if command[-1] == "--version":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="codex-cli 0.test\n",
                    stderr="",
                )
            assert command[1] == "exec"
            assert "--ignore-user-config" in command
            assert "--ignore-rules" in command
            assert command[command.index("--sandbox") + 1] == "read-only"
            prompt = kwargs["input"]
            prompts.append(prompt)
            assert "offline private teacher reference" not in prompt
            assert "reference_translation" not in prompt
            assert "student_hypothesis" not in prompt
            sources = json.loads(prompt.split("Source-only inputs:\n", 1)[1])
            output = {"results": [payload(source) for source in sources]}
            output_path_argument = Path(
                command[command.index("--output-last-message") + 1]
            )
            output_path_argument.write_text(
                json.dumps(output, ensure_ascii=False),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="tokens used\n1,234\n",
                stderr="",
            )

        teacher.subprocess.run = fake_run
        first = teacher.run_one_shard(
            requests_path,
            run_directory,
            0,
            "codex",
        )
        second = teacher.run_one_shard(
            requests_path,
            run_directory,
            1,
            "codex",
        )
        assert first["reported_tokens_used"] == 1234
        assert second["reported_tokens_used"] == 1234
        assert len(prompts) == 2
        resumed = teacher.run_one_shard(
            requests_path,
            run_directory,
            0,
            "codex",
        )
        assert resumed["status"] == "already-complete"
        assert len(prompts) == 2
        teacher.subprocess.run = real_run

        teacher.collect_command(
            SimpleNamespace(
                requests=requests_path,
                run_directory=run_directory,
                output=output_path,
            )
        )
        collected = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["custom_id"] for row in collected] == [
            seed["id"] for seed in seeds
        ]
        assert all(
            row["response"]["body"]["model"]
            == "gpt-5.6-sol-via-codex-cli"
            for row in collected
        )
        assert all(
            row["mimi_teacher_transport"]["reasoning_trace_stored"] is False
            for row in collected
        )

        write_jsonl(
            protected_path,
            [
                {
                    "id": "unrelated-heldout",
                    "source": "A completely unrelated held-out sentence.",
                    "references": ["無関係な評価文です。"],
                }
            ],
        )
        subprocess.run(
            [
                "python3",
                "scripts/translation/filter_synthetic_batch.py",
                str(seeds_path),
                str(output_path),
                str(protected_path),
                str(queue_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        queue = [
            json.loads(line)
            for line in queue_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(queue) == 9
        assert all(
            row["teacher_model"] == "gpt-5.6-sol-via-codex-cli"
            for row in queue
        )

    print("Mimi Codex teacher pipeline contract passed.")


if __name__ == "__main__":
    main()
