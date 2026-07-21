#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from compare_marian_moe_pack_smokes import compare_pair


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        neural = {
            "schemaVersion": 1,
            "direction": "en-ja",
            "expectedEngine": "generalist",
            "selectedEngine": "generalist",
            "source": "Hello",
            "hypothesis": "こんにちは",
            "outputTokenIDs": [10, 20, 0],
            "status": "passed",
        }
        source = root / "source.json"
        candidate = root / "candidate.json"
        write(source, neural)
        write(candidate, neural)
        assert compare_pair("neural", source, candidate)["exactMatch"] is True

        changed = dict(neural)
        changed["outputTokenIDs"] = [10, 21, 0]
        write(candidate, changed)
        mismatch = compare_pair("changed", source, candidate)
        assert mismatch["exactMatch"] is False
        assert mismatch["tokenExactMatch"] is False

        memory = dict(neural)
        memory.update(
            {
                "expectedEngine": "translation-memory",
                "selectedEngine": "translation-memory",
                "outputTokenIDs": None,
            }
        )
        write(source, memory)
        write(candidate, memory)
        exact_memory = compare_pair("memory", source, candidate)
        assert exact_memory["exactMatch"] is True
        assert exact_memory["tokenExactMatch"] is None
    print("Marian MoE smoke parity comparison tests passed")


if __name__ == "__main__":
    main()
