#!/usr/bin/env python3
"""Offline contracts for balanced reference-teacher pool preparation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_balanced_reference_teacher_seeds.py")
SPEC = importlib.util.spec_from_file_location("balanced_reference_teacher", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def chat_row(identifier: str, direction: str, source: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "translate"},
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "source": "fixture",
            "source_id": identifier,
            "direction": direction,
            "license": "CC-BY-SA-3.0",
            "attribution": "fixture attribution",
        },
    }


def main() -> None:
    parsed = module.parse_parallel_row(
        chat_row("k1", "en-ja", "The train is late.", "列車が遅れています。"),
        "kftt",
    )
    assert parsed["direction"] == "en-ja"
    assert parsed["domain"] == "professional-wikipedia-hard"
    flat = module.parse_parallel_row({
        "source_id": "a1",
        "source_language": "ja-JP",
        "target_language": "en-US",
        "source": "列車が遅れています。",
        "target": "The train is late.",
        "source_license": "CC-BY-4.0",
        "source_provenance": "ALT fixture",
        "attribution": "ALT fixture attribution",
    }, "alt")
    assert flat["direction"] == "ja-en"
    assert flat["domain"] == "human-translated-news-hard"

    with tempfile.TemporaryDirectory(prefix="mimi-balanced-teacher-test-") as temporary:
        root = Path(temporary)
        base = root / "base"
        prior = root / "prior.jsonl"
        protected = root / "protected.jsonl"
        write_jsonl(base / "train.jsonl", [{
            "source": "Already trained.", "target": "学習済みです。",
        }])
        write_jsonl(base / "valid.jsonl", [{
            "source": "Validation sentence.", "target": "検証文です。",
        }])
        write_jsonl(prior, [{
            "source": "Prior teacher source.", "references": ["以前の教師文です。"],
        }])
        write_jsonl(protected, [{
            "source": "Protected benchmark source.", "references": ["保護された評価文です。"],
        }])
        corpus_paths: dict[str, Path] = {}
        for corpus in module.CORPORA:
            path = root / f"{corpus}.jsonl"
            rows = [
                chat_row(f"{corpus}-good-en", "en-ja", f"Novel {corpus} English.", f"新しい{corpus}日本語です。"),
                chat_row(f"{corpus}-good-ja", "ja-en", f"新しい{corpus}会話です。", f"Novel {corpus} conversation."),
            ]
            if corpus == "kftt":
                rows.extend([
                    chat_row("existing", "en-ja", "Already trained.", "学習済みです。"),
                    chat_row("prior", "en-ja", "Prior teacher source.", "以前の教師文です。"),
                    chat_row("heldout", "ja-en", "保護された評価文です。", "Protected benchmark source."),
                ])
            if corpus == "tatoeba":
                rows.extend([
                    chat_row("ambiguous", "en-ja", "One source.", "一つ目の訳です。"),
                    chat_row("ambiguous", "en-ja", "One source.", "二つ目の訳です。"),
                ])
            write_jsonl(path, rows)
            corpus_paths[corpus] = path

        excluded, validation, validation_grams, files = module.exclusion_policy([base], [prior])
        assert len(files) == 2
        eligible, rejected = module.eligible_by_corpus_direction(
            corpus_paths,
            excluded,
            validation,
            validation_grams,
            module.protected_grams([protected]),
            0.8,
        )
        for corpus in module.CORPORA:
            assert len(eligible[(corpus, "en-ja")]) == 1
            assert len(eligible[(corpus, "ja-en")]) == 1
        assert rejected["kftt:existing-student-or-prior-teacher-source"] == 2
        assert rejected["kftt:near-protected-evaluation"] == 1
        assert rejected["tatoeba:ambiguous-source-id"] == 2
        inventory = module.inventory_report(eligible, rejected)
        assert inventory["eligible"]["alt"] == {"en-ja": 1, "ja-en": 1}
    print("Balanced reference-teacher seed contracts passed.")


if __name__ == "__main__":
    main()
