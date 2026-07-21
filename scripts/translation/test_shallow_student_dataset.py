#!/usr/bin/env python3
"""Contract tests for the large licensed shallow-student dataset builder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/build_shallow_student_dataset.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def chat(source_id: str, source: str, target: str, corpus: str, license_name: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "translate"},
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "direction": "en-ja",
            "source_id": source_id,
            "source": corpus,
            "license": license_name,
            "attribution": f"attribution {source_id}",
        },
    }


def canonical(
    identifier: str,
    source: str,
    target: str,
    origin: str,
    *,
    license_name: str = "project-owned",
    domain: str = "mimi-product-ui",
) -> dict:
    return {
        "id": identifier,
        "source_id": identifier,
        "source": source,
        "target": target,
        "source_language": "en-US",
        "target_language": "ja-JP",
        "source_license": license_name,
        "source_provenance": f"fixture {identifier}",
        "attribution": f"attribution {identifier}",
        "domain": domain,
        "origin": origin,
    }


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-shallow-dataset-") as temporary:
        root = Path(temporary)
        validation = root / "validation"
        valid_rows = [
            canonical("valid-1", "Validation sentence.", "検証文です。", "human-kftt-replay")
        ]
        write_jsonl(validation / "valid.jsonl", valid_rows)
        write_jsonl(validation / "train.jsonl", [])
        (validation / "manifest.json").write_text(
            json.dumps(
                {
                    "outputs": {
                        "valid": {"sha256": sha256(validation / "valid.jsonl")}
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )

        protected = root / "protected.jsonl"
        write_jsonl(
            protected,
            [
                {
                    "id": "protected",
                    "source": "Protected source sentence.",
                    "references": ["保護された参照文です。"],
                }
            ],
        )
        kftt = root / "kftt" / "train.jsonl"
        alt = root / "alt" / "train.jsonl"
        tatoeba = root / "tatoeba" / "train.jsonl"
        jlt = root / "jlt" / "train.jsonl"
        write_jsonl(
            kftt,
            [
                chat("k1", "Kyoto has many temples.", "京都には多くの寺があります。", "KFTT", "CC-BY-SA-3.0"),
                chat("k2", "Validation sentence.", "検証文です。", "KFTT", "CC-BY-SA-3.0"),
                chat("k3", "Protected source sentence.", "保護された参照文です。", "KFTT", "CC-BY-SA-3.0"),
            ],
        )
        write_jsonl(
            alt,
            [
                chat("a1", "The cabinet met today.", "内閣は本日会合を開いた。", "ALT", "CC-BY-4.0"),
                chat("a2", "The vote starts tomorrow.", "投票は明日始まる。", "ALT", "CC-BY-4.0"),
            ],
        )
        write_jsonl(
            tatoeba,
            [
                chat("t1", "Please open the window.", "窓を開けてください。", "Tatoeba", "CC-BY-2.0-FR"),
                chat("t2", "I will call you tonight.", "今夜電話します。", "Tatoeba", "CC-BY-2.0-FR"),
            ],
        )
        write_jsonl(
            jlt,
            [
                canonical(
                    "jlt-1",
                    "The Minister may specify the procedure.",
                    "大臣は手続を定めることができる。",
                    "finalized-japanese-law-translation",
                    license_name="PDL-1.0-compatible-CC-BY-4.0",
                    domain="ministry-published-legal",
                )
            ],
        )
        for directory in (kftt.parent, alt.parent, tatoeba.parent):
            (directory / "manifest.json").write_text("{}\n", encoding="utf-8")
        (jlt.parent / "manifest.json").write_text(
            json.dumps({"outputs": {"train": {"sha256": sha256(jlt)}}}) + "\n",
            encoding="utf-8",
        )

        ui = root / "ui"
        write_jsonl(
            ui / "train.jsonl",
            [
                canonical(
                    "ui-1",
                    "Show the floating caption.",
                    "フローティング字幕を表示します。",
                    "mimi-shipped-ui-pair",
                )
            ],
        )
        (ui / "manifest.json").write_text("{}\n", encoding="utf-8")
        output = root / "output"
        command = [
            "python3",
            str(SCRIPT),
            str(validation),
            str(output),
            "--direction",
            "en-ja",
            "--corpus",
            f"kftt={kftt}",
            "--corpus",
            f"alt={alt}",
            "--corpus",
            f"tatoeba={tatoeba}",
            "--corpus",
            f"jlt={jlt}",
            "--ui-dataset",
            str(ui),
            "--protected-suite",
            str(protected),
            "--cap",
            "kftt=1",
            "--cap",
            "alt=2",
            "--cap",
            "tatoeba=2",
            "--cap",
            "jlt=1",
            "--repeat",
            "kftt=2",
            "--repeat",
            "alt=1",
            "--repeat",
            "tatoeba=2",
            "--repeat",
            "jlt=1",
            "--repeat",
            "ui=3",
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        train = rows(output / "train.jsonl")
        assert manifest["counts"] == {"train": 12, "valid": 1}
        assert manifest["selection"]["rejected"]["kftt:protected-overlap"] == 1
        assert manifest["selection"]["rejected"]["kftt:validation-overlap"] == 1
        assert len({row["id"] for row in train}) == len(train)
        assert {row["origin"] for row in train} == {
            "human-kftt-replay",
            "human-alt-parallel",
            "human-tatoeba-bidirectional-agreement-filtered",
            "finalized-japanese-law-translation",
            "mimi-shipped-ui-pair",
        }
        assert all("Protected source" not in row["source"] for row in train)
        assert manifest["promotion_eligible"] is False
        assert manifest["private_reasoning_traces_used"] is False
        assert manifest["outputs"]["train"]["sha256"] == sha256(
            output / "train.jsonl"
        )

    print("Large licensed shallow-student dataset contract passed.")


if __name__ == "__main__":
    main()
