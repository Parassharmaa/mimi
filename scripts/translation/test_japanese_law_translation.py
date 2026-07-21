#!/usr/bin/env python3
"""Contracts for the finalized Japanese Law Translation data path."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download = load("download_japanese_law_translation")
prepare = load("prepare_japanese_law_translation")


SEARCH = b"""<html><body>
<input name="_csrfToken" value="token-123">
<span>Showing 1 to 2 of 2</span>
<a href="/en/laws/view/12">A</a><a href="/en/laws/view/34">B</a>
</body></html>"""
VIEW = b'<a id="tmxDownload" href="/en/laws/download/12/23/99.tmx">TMX</a>'
TMX = """<?xml version="1.0" encoding="utf-8"?>
<tmx version="1.4"><header srclang="ja-JP" creationdate="20260718T000000Z"/>
<body>
<tu tuid="1"><tuv xml:lang="ja-JP"><seg>これは法律です。</seg></tuv>
<tuv xml:lang="en-US"><seg>This is a law.</seg></tuv></tu>
<tu tuid="2"><tuv xml:lang="ja-JP"><seg>秘密の評価文です。</seg></tuv>
<tuv xml:lang="en-US"><seg>This protected evaluation sentence must stay held out.</seg></tuv></tu>
<tu tuid="3"><tuv xml:lang="ja-JP"><seg>これは法律です。</seg></tuv>
<tuv xml:lang="en-US"><seg>This is a law.</seg></tuv></tu>
<tu tuid="4"><tuv xml:lang="ja-JP"><seg>日本語だけです。</seg></tuv></tu>
</body></tmx>""".encode()


def main() -> None:
    assert download.parse_csrf(SEARCH) == "token-123"
    ids, total = download.parse_search_page(SEARCH)
    assert ids == {"12", "34"} and total == 2
    assert download.parse_tmx_href(VIEW) == "/en/laws/download/12/23/99.tmx"
    assert download.parse_tmx_href(b"\xff" + VIEW) == "/en/laws/download/12/23/99.tmx"
    assert "ia" not in download.search_form("token", "A")
    metadata = download.tmx_metadata(TMX, "fixture")
    assert metadata["translation_units"] == 4
    assert metadata["source_language"] == "ja-JP"
    try:
        download.tmx_metadata(b"not XML", "broken")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed TMX must be rejected")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tmx_path = root / "law-12.tmx"
        tmx_path.write_bytes(TMX)
        protected = root / "protected.jsonl"
        protected.write_text(
            json.dumps(
                {
                    "source": "This protected evaluation sentence must stay held out.",
                    "references": ["秘密の評価文です。"],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        output, manifest = prepare.prepare(
            [tmx_path],
            [protected],
            maximum_jaccard=0.80,
            maximum_english_characters=240,
            maximum_japanese_characters=160,
            split_seed="fixture",
        )
        rows = [row for split in output.values() for row in split]
        assert len(rows) == 2
        assert {row["source_language"] for row in rows} == {"en-US", "ja-JP"}
        assert all(row["promotion_eligible"] is False for row in rows)
        assert all(row["translation_status"] == "finalized" for row in rows)
        assert manifest["unique_pairs"] == 1
        assert manifest["rejected"] == {
            "contamination": 1,
            "duplicate": 1,
            "empty_or_language": 1,
        }
        assert manifest["tentative_translations_included"] is False
        assert manifest["private_reasoning_traces_used"] is False

    print("Japanese Law Translation data contracts passed.")


if __name__ == "__main__":
    main()
