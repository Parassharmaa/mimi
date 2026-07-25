#!/usr/bin/env python3
"""Contract test for the direct-document slice builder."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, models, processors, pre_tokenizers


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/translation/prepare_direct_within_limit_suite.py"


def write_model(path: Path, direction: str) -> None:
    path.mkdir(parents=True)
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "</s>": 1, "p": 2, "a": 3, "b": 4}))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", 1)]
    )
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "manifest.json").write_text(
        json.dumps({"direction": direction, "source_prefixes": {direction: "p "}}),
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-direct-slice-") as temporary:
        root = Path(temporary)
        suite = root / "suite.jsonl"
        rows = [
            {
                "id": "en-fit",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "source": "a",
            },
            {
                "id": "en-long",
                "sourceLanguage": "en-US",
                "targetLanguage": "ja-JP",
                "source": "a a a",
            },
            {
                "id": "ja-fit",
                "sourceLanguage": "ja-JP",
                "targetLanguage": "en-US",
                "source": "b",
            },
        ]
        suite.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        en_ja, ja_en = root / "en-ja", root / "ja-en"
        write_model(en_ja, "en-ja")
        write_model(ja_en, "ja-en")
        output, manifest = root / "direct.jsonl", root / "manifest.json"
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                str(suite),
                str(en_ja),
                str(ja_en),
                str(output),
                str(manifest),
                "--maximum-source-tokens",
                "3",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        selected = [
            json.loads(line) for line in output.read_text().splitlines() if line
        ]
        assert [row["id"] for row in selected] == ["en-fit", "ja-fit"]
        assert all(row["directSourceTokenCount"] == 3 for row in selected)
        frozen = json.loads(manifest.read_text())
        assert frozen["selection"]["manualInclusions"] == []
        assert frozen["selection"]["manualExclusions"] == []
        assert frozen["output"]["directions"] == {"en-ja": 1, "ja-en": 1}
        assert frozen["excluded"] == [
            {"caseID": "en-long", "direction": "en-ja", "sourceTokens": 5}
        ]

    print("Direct-document slice contract passed.")


if __name__ == "__main__":
    main()
