#!/usr/bin/env python3
"""Offline contracts for licensed ALT, BTEC, Mimi UI, and Tatoeba preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "scripts/translation" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(module, arguments: list[str]) -> None:
    original = sys.argv
    try:
        sys.argv = [module.__file__, *arguments]
        module.main()
    finally:
        sys.argv = original


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-licensed-data-test-") as temporary:
        work = Path(temporary)
        protected = work / "protected.jsonl"
        protected.write_text("", encoding="utf-8")

        alt_archive = work / "alt.zip"
        alt_root = "ALT-Parallel-Corpus-20191206"
        with zipfile.ZipFile(alt_archive, "w") as archive:
            archive.writestr(f"{alt_root}/URL.txt", "URL.1\thttps://example.test/article\n")
            archive.writestr(
                f"{alt_root}/data_en.txt",
                "SNT.1.1\tThe service resumed today.\nSNT.1.2\tA second train arrived.\n",
            )
            archive.writestr(
                f"{alt_root}/data_ja.txt",
                "SNT.1.1\tサービスは本日再開した。\nSNT.1.2\t2本目の列車が到着した。\n",
            )
        alt = load_script("prepare_alt.py")
        alt.ARCHIVE_SHA256 = hashlib.sha256(alt_archive.read_bytes()).hexdigest()
        alt_output = work / "alt-output"
        run_main(alt, [str(alt_archive), str(protected), str(alt_output)])
        alt_rows = sum(
            (read_jsonl(alt_output / f"{split}.jsonl") for split in ("train", "valid", "test")),
            [],
        )
        assert len(alt_rows) == 4
        assert {row["source_license"] for row in alt_rows} == {"CC-BY-4.0"}
        assert {row["origin"] for row in alt_rows} == {"human-alt-parallel"}

        btec_archive = work / "btec.zip"
        with zipfile.ZipFile(btec_archive, "w") as archive:
            archive.writestr(
                "enBTEC20K.txt",
                "en/file:BTEC1@jpn001@00001@en@@@@Can I exchange it?|Please help me.\n"
                "en/other:BTEC1@jpn002@00002@en@@@@Can I exchange it?\n",
            )
        btec = load_script("prepare_btec_teacher_seeds.py")
        btec.ARCHIVE_SHA256 = hashlib.sha256(btec_archive.read_bytes()).hexdigest()
        btec_output = work / "btec.jsonl"
        run_main(btec, [str(btec_archive), str(protected), str(btec_output), "--maximum-seeds", "10"])
        btec_rows = read_jsonl(btec_output)
        assert sorted(row["source"] for row in btec_rows) == ["Can I exchange it?", "Please help me."]
        assert all("BTEC1@" not in row["source"] for row in btec_rows)
        assert len({row["id"] for row in btec_rows}) == 2
        btec_manifest = json.loads(btec_output.with_suffix(".manifest.json").read_text())
        assert btec_manifest["parallel_gold"] is False
        assert btec_manifest["rejected"]["duplicate"] == 1

        ui = load_script("prepare_mimi_ui_parallel.py")
        repository = work / "ui-repository"
        source_directory = repository / "Sources/Mimi"
        source_directory.mkdir(parents=True)
        split_paths: dict[str, Path] = {}
        for index in range(100):
            relative = Path(f"Sources/Mimi/Fixture{index}.swift")
            split = ui.split_for(relative, "mimi-ui-parallel-v1", 0.20)
            split_paths.setdefault(split, source_directory / relative.name)
            if split_paths.keys() == {"train", "valid"}:
                break
        assert split_paths.keys() == {"train", "valid"}
        split_paths["train"].write_text(
            'let a = t("Start recording", "録音を開始")\n'
            'let b = t("Microphone", "マイク")\n',
            encoding="utf-8",
        )
        split_paths["valid"].write_text(
            'let a = t("Stop recording", "録音を停止")\n'
            'let b = t("My microphone", "マイク")\n',
            encoding="utf-8",
        )
        ui_output = work / "ui-output"
        run_main(ui, [str(source_directory), str(protected), str(ui_output)])
        ui_manifest = json.loads((ui_output / "manifest.json").read_text())
        assert ui_manifest["rejected"]["ambiguous-source"] == 2
        assert ui_manifest["pairs"] == {"train": 1, "valid": 1}
        ui_rows = read_jsonl(ui_output / "train.jsonl") + read_jsonl(ui_output / "valid.jsonl")
        assert all(row["source"] != "マイク" for row in ui_rows)

        tatoeba = load_script("prepare_tatoeba_parallel.py")
        tatoeba_input = work / "tatoeba-input"
        tatoeba_input.mkdir()

        def tatoeba_row(source_id: str, direction: str, source: str, target: str) -> dict:
            return {
                "messages": [
                    {"role": "system", "content": "translate"},
                    {"role": "user", "content": source},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "source_id": source_id,
                    "direction": direction,
                    "license": "CC-BY-2.0-FR",
                    "attribution": f"Tatoeba #{source_id} & #ja-{source_id}",
                },
            }

        train_rows = [
            tatoeba_row("1", "en-ja", "Start recording.", "録音を開始して。"),
            tatoeba_row("1", "ja-en", "録音を開始して。", "Start recording."),
            tatoeba_row("1", "en-ja", "Start recording.", "録音を始めて。"),
            tatoeba_row("1", "ja-en", "録音を始めて。", "Start recording."),
            tatoeba_row("2", "en-ja", "Please wait.", "お待ちください。"),
            tatoeba_row("2", "ja-en", "お待ちください。", "Please wait."),
        ]
        valid_rows = [
            tatoeba_row("3", "en-ja", "It is ready.", "準備できました。"),
            tatoeba_row("3", "ja-en", "準備できました。", "It is ready."),
        ]
        for split, rows in (("train", train_rows), ("valid", valid_rows), ("test", [])):
            (tatoeba_input / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
        pairs, rejected = tatoeba.reciprocal_pairs(tatoeba_input)
        assert {row["source_id"] for row in pairs} == {"2", "3"}
        assert rejected["ambiguous-source"] == 2
        assert all(tatoeba.eligible_text(row, [], 0.8) is None for row in pairs)

    print("Mimi licensed data preparation smoke passed.")


if __name__ == "__main__":
    main()
