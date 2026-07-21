#!/usr/bin/env python3
"""Contract tests for variable-precision Marian pair packaging."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGER = ROOT / "scripts/translation/package_elanmt_mlx.py"


def direction_fixture(
    root: Path,
    direction: str,
    bits: int,
    group_size: int = 64,
    training_data: dict | None = None,
) -> Path:
    output = root / direction
    output.mkdir()
    for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        (output / name).write_text(f"{direction}:{name}\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "format": "mimi-mlx-marian-v1",
                "direction": direction,
                "source_revision": f"fixture-{direction}",
                "bits": bits,
                "group_size": group_size,
                "training_data": training_data,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def run_packager(en_ja: Path, ja_en: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(PACKAGER), str(en_ja), str(ja_en), str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mimi-quantization-packaging-") as temporary:
        work = Path(temporary)
        matched = work / "matched"
        matched.mkdir()
        training_data = {
            "dataset_manifest": {"path": "dataset/manifest.json", "sha256": "abc"},
            "target_source": "qwen",
            "effective_licenses": {"train": {"CC-BY-SA-3.0": 3}},
            "required_attributions": [
                {
                    "corpus": "KFTT",
                    "license": "CC-BY-SA-3.0",
                    "required_notice": "fixture notice",
                }
            ],
            "distribution_status": (
                "blocked-pending-share-alike-and-attribution-review"
            ),
        }
        en_ja = direction_fixture(
            matched, "en-ja", 6, group_size=32, training_data=training_data
        )
        ja_en = direction_fixture(matched, "ja-en", 6, group_size=32)
        output = work / "pair-6bit"
        result = run_packager(en_ja, ja_en, output)
        assert result.returncode == 0, result.stderr
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["quantization"] == {"bits": 6, "group_size": 32}
        assert manifest["training_data"]["en-ja"] == training_data
        assert manifest["required_attributions"] == training_data[
            "required_attributions"
        ]
        assert manifest["distribution_status"] == (
            "blocked-pending-share-alike-and-attribution-review"
        )
        for direction in ("en-ja", "ja-en"):
            child = json.loads(
                (output / direction / "manifest.json").read_text(encoding="utf-8")
            )
            assert set(child["files"]) == {
                "model.safetensors", "tokenizer.json", "tokenizer_config.json"
            }
            assert all((output / direction / name).is_file() for name in child["files"])

        incomplete = work / "incomplete"
        incomplete.mkdir()
        en_ja = direction_fixture(incomplete, "en-ja", 4)
        ja_en = direction_fixture(incomplete, "ja-en", 4)
        output = work / "pair-incomplete"
        result = run_packager(en_ja, ja_en, output)
        assert result.returncode == 0, result.stderr
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["distribution_status"] == (
            "provenance-incomplete-not-approved-for-distribution"
        )

        mismatched = work / "mismatched"
        mismatched.mkdir()
        en_ja = direction_fixture(mismatched, "en-ja", 4)
        ja_en = direction_fixture(mismatched, "ja-en", 8)
        result = run_packager(en_ja, ja_en, work / "pair-mismatched")
        assert result.returncode != 0
        assert "same quantization" in result.stderr

        unsupported = work / "unsupported"
        unsupported.mkdir()
        en_ja = direction_fixture(unsupported, "en-ja", 5)
        ja_en = direction_fixture(unsupported, "ja-en", 5)
        result = run_packager(en_ja, ja_en, work / "pair-unsupported")
        assert result.returncode != 0
        assert "unsupported shipping quantization" in result.stderr

    print("Mimi variable-precision translation packaging contract passed.")


if __name__ == "__main__":
    main()
