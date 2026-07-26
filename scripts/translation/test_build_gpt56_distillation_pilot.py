#!/usr/bin/env python3
"""Offline fixtures for the GPT-5.6 pilot selection and cost contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    pilot = load_module(
        "scripts/translation/build_gpt56_distillation_pilot.py", "mimi_pilot"
    )
    with tempfile.TemporaryDirectory(prefix="mimi-gpt56-pilot-test-") as temporary:
        work = Path(temporary)
        protected_path = work / "protected.jsonl"
        write_jsonl(
            protected_path,
            [
                {
                    "id": "heldout-1",
                    "source": "Do not train on this sentence.",
                    "references": ["この文を学習に使用しないでください。"],
                }
            ],
        )
        protected = pilot.ProtectedIndex([protected_path], 5)
        candidates = defaultdict(list)

        def candidate(source: str, *, priority: bool = False) -> dict:
            return {
                "direction": "en-ja",
                "bucket": "kftt",
                "source": source,
                "target": f"参照: {source}",
                "source_id": hashlib.sha256(source.encode()).hexdigest()[:12],
                "license": "CC-BY-SA-3.0",
                "provenance": "offline fixture",
                "priority": priority,
                "original_id": f"fixture:{source}",
                "domain": "wikipedia",
                "selection": "fixture",
            }

        candidates[("en-ja", "kftt")].extend(
            [
                candidate("A normal training sentence."),
                candidate("A second normal training sentence."),
                candidate("Do not train on this sentence.", priority=True),
            ]
        )
        rows, summary = pilot.select_rows(
            candidates,
            protected,
            {"en-ja": {"kftt": 2}},
            "fixture-seed",
            0.8,
            5,
        )
        assert len(rows) == 2
        assert {row["source"] for row in rows} == {
            "A normal training sentence.",
            "A second normal training sentence.",
        }
        assert summary["selected_priority"] == {}
        assert summary["rejected"] == {"en-ja:kftt:protected-exact": 1}
        assert all(row["split"] == "train" for row in rows)
        assert all(row["license"] == "CC-BY-SA-3.0" for row in rows)

        fixture = {
            "id": "teacher-kftt:en-ja:fixture",
            "source_language": "en-US",
            "target_language": "ja-JP",
            "domain": "professional-wikipedia-hard",
            "source": "A priority hard sentence.",
            "license": "CC-BY-SA-3.0",
            "provenance": "offline fixture",
            "reference_translation": "難しい優先文。",
            "student_hypothesis": "学生の仮説。",
            "student_chrf_pp": 10.0,
        }
        canonical = pilot.canonical_candidate(fixture, priority=True)
        output = pilot.make_seed(canonical, "fixture-seed")
        assert output["reference_translation"] == "難しい優先文。"
        assert output["student_hypothesis"] == "学生の仮説。"
        assert output["source_corpus"] == "kftt"
        assert output["source_selection"].startswith("priority")

    print("gpt56 distillation pilot tests passed")


if __name__ == "__main__":
    main()
