#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from deduplicate_marian_moe_tokenizer import (
    OUTPUT_FORMAT,
    SHARED_TOKENIZER_PATH,
    file_record,
    repack,
    validate_file_table,
)


ENGINE_LAYOUT = {
    "engines/generalist-en-ja": "en-ja",
    "engines/generalist-ja-en": "ja-en",
    "engines/formal-en-ja": "en-ja",
    "engines/legal-ja-en": "ja-en",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_fixture(root: Path, mismatched: bool = False) -> None:
    tokenizer = bytes(range(256)) * 32
    for index, (relative, direction) in enumerate(ENGINE_LAYOUT.items()):
        engine = root / relative
        engine.mkdir(parents=True)
        (engine / "model.safetensors").write_bytes(f"model-{relative}".encode())
        (engine / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "T5Tokenizer", "direction": direction}),
            encoding="utf-8",
        )
        payload = tokenizer + (b"different" if mismatched and index == 3 else b"")
        (engine / "tokenizer.json").write_bytes(payload)
        files = {
            name: file_record(engine / name)
            for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json")
        }
        write_json(
            engine / "manifest.json",
            {
                "bits": 4,
                "direction": direction,
                "files": files,
                "format": "mimi-mlx-marian-v1",
                "group_size": 64,
            },
        )

    routers = root / "routers"
    routers.mkdir(parents=True)
    (routers / "formal-en-ja.json").write_text('{"direction":"en-ja"}', encoding="utf-8")
    (routers / "legal-ja-en.json").write_text('{"direction":"ja-en"}', encoding="utf-8")
    root_files = {
        path.relative_to(root).as_posix(): file_record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    write_json(
        root / "manifest.json",
        {
            "experts": {
                "en-ja": {
                    "engine": "engines/formal-en-ja",
                    "router": "routers/formal-en-ja.json",
                },
                "ja-en": {
                    "engine": "engines/legal-ja-en",
                    "router": "routers/legal-ja-en.json",
                },
            },
            "files": root_files,
            "format": "mimi-mlx-marian-moe-v1",
            "generalists": {
                "en-ja": "engines/generalist-en-ja",
                "ja-en": "engines/generalist-ja-en",
            },
            "interface": "bidirectional-en-ja",
            "quantization": {"bits": 4, "groupSize": 64},
            "routing": {
                "defaultOnRouterFailure": "generalist",
                "inputs": "source-text-only",
            },
        },
    )


def test_lossless_repack(parent: Path) -> None:
    source = parent / "source"
    output = parent / "output"
    build_fixture(source)
    result = repack(source, output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == OUTPUT_FORMAT
    assert manifest["sharedTokenizer"] == SHARED_TOKENIZER_PATH
    assert result["savedBytes"] > 0
    assert result["outputBytes"] < result["sourceBytes"]
    assert len(list(output.rglob("tokenizer.json"))) == 1
    validate_file_table(output, manifest["files"])

    source_tokenizer = source / "engines/generalist-en-ja/tokenizer.json"
    assert (output / SHARED_TOKENIZER_PATH).read_bytes() == source_tokenizer.read_bytes()
    for relative in ENGINE_LAYOUT:
        assert not (output / relative / "tokenizer.json").exists()
        engine_manifest = json.loads(
            (output / relative / "manifest.json").read_text(encoding="utf-8")
        )
        assert "tokenizer.json" not in engine_manifest["files"]
        assert engine_manifest["shared_tokenizer"] == file_record(source_tokenizer)
        for name in ("model.safetensors", "tokenizer_config.json"):
            assert (output / relative / name).read_bytes() == (source / relative / name).read_bytes()


def test_rejects_mismatched_tokenizers(parent: Path) -> None:
    source = parent / "mismatch-source"
    output = parent / "mismatch-output"
    build_fixture(source, mismatched=True)
    try:
        repack(source, output)
    except ValueError as error:
        assert "not byte-identical" in str(error)
    else:
        raise AssertionError("mismatched engine tokenizers were accepted")
    assert not output.exists()


def test_rejects_unauthenticated_source_file(parent: Path) -> None:
    source = parent / "extra-source"
    output = parent / "extra-output"
    build_fixture(source)
    (source / "unauthenticated.bin").write_bytes(b"must not be silently promoted")
    try:
        repack(source, output)
    except ValueError as error:
        assert "not exhaustive" in str(error)
        assert "unauthenticated.bin" in str(error)
    else:
        raise AssertionError("unauthenticated source file was accepted")
    assert not output.exists()


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        test_lossless_repack(parent)
        test_rejects_mismatched_tokenizers(parent)
        test_rejects_unauthenticated_source_file(parent)
    print("shared-tokenizer Marian MoE repack tests passed")


if __name__ == "__main__":
    main()
