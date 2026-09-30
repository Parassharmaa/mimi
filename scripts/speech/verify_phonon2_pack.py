#!/usr/bin/env python3
"""Verify the converted native pack and every pinned file digest."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_MANIFEST_SHA256 = "d3836539e203396f1e0f32cec20bd2e319850fef2a47cb4ff43ee1ae4b5fd45c"

def verify(root):
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise SystemExit("Phonon manifest does not match the pinned native conversion")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "mimi-phonon2-two-plane-v1"
    assert manifest["repository"] == "FermionResearch/Phonon-2"
    assert manifest["revision"] == "1c388bcec35d19740bf36b0b675718223fa7904e"
    assert manifest["license"] == "CC-BY-4.0" and manifest["languages"] == ["en"]
    assert manifest["weight_equivalence"] == {"modules":264,"max_absolute_error":0.0}
    assert len(manifest["modules"]) == 264
    assert len({m["path"] for m in manifest["modules"]}) == 264
    assert {p.name for p in root.iterdir()} == {"manifest.json", "config.json", "model.safetensors"}
    for name, file in manifest["files"].items():
        assert name in {"config.json","model.safetensors"}
        path = root / name
        assert not path.is_symlink() and path.stat().st_size == file["bytes"]
        h = hashlib.sha256()
        with path.open("rb") as stream:
            while data := stream.read(1024*1024): h.update(data)
        assert h.hexdigest() == file["sha256"], f"Hash mismatch: {name}"
    size = sum(p.stat().st_size for p in root.iterdir())
    assert size < 500_000_000
    print(f"Mimi native Phonon 2 pack verified: {size} bytes, 264 exact five-value modules, English only")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=Path)
    verify(parser.parse_args().model_root)
