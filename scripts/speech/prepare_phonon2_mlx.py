#!/usr/bin/env python3
"""Convert pinned Phonon 2 into exact native MLX two-plane weights.

Developer/build tool only. Python is not shipped or used by Mimi.
Requires fermion-research==0.2.3, mlx, numpy, safetensors and zstandard.
"""
import argparse
import hashlib
import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np
import zstandard
from fermion._speech._engine_phonon2 import load

REPOSITORY = "FermionResearch/Phonon-2"
REVISION = "1c388bcec35d19740bf36b0b675718223fa7904e"
ARCHIVE_SHA = "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e"
CONTAINER_SHA = "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"
FORMAT = "mimi-phonon2-two-plane-v1"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024*1024): h.update(chunk)
    return h.hexdigest()

def native_key(key):
    key = key.replace(".pos_bias_u", ".posBiasU").replace(".pos_bias_v", ".posBiasV")
    key = key.replace("joint.joint_net.2.", "joint.joint_net.")
    mapping = {"0":"conv0", "2":"depthwise_layers.0", "3":"pointwise_layers.0", "5":"depthwise_layers.1", "6":"pointwise_layers.1"}
    match = re.match(r"encoder\.pre_encode\.conv\.(\d+)\.(.+)", key)
    if match: key = f"encoder.pre_encode.{mapping[match[1]]}.{match[2]}"
    return key

def prepare(root, source):
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "model"
    notices = root / "notices"
    model_dir.mkdir(exist_ok=True)
    notices.mkdir(exist_ok=True)
    container = source / "model.fermion"
    assert sha(container) == CONTAINER_SHA, "Phonon source container hash mismatch"
    fc, mapping, packed = load("fermion_container"), load("hf_to_mlx_parakeet"), load("packed_runtime")
    tensors, index, records = fc.read_container(container, with_raw=True)
    names = [entry["n"] for entry in index if entry["k"] == "five_value"]
    assert len(names) == 264
    other = mapping.hf_to_mlx(tensors)
    for name in names:
        other.pop(packed.five_value_hf_to_mlx(name) + ".weight")
    weights = {native_key(k): mx.array(v).astype(mx.bfloat16) for k, v in other.items()}
    modules = []
    for name in names:
        raw = records[name]
        record = packed.FiveValueRecord(name, raw["sign"], raw["is_hi"], raw["lo"], raw["hi"])
        qa, qb, lo, delta = record.planes()
        out, inp = record.shape
        groups = inp // 128
        assert inp % 128 == 0
        reconstructed = packed.expand_planes(qa, qb, lo, delta, inp)
        assert np.array_equal(reconstructed, record.dense()), f"Weight equivalence failed: {name}"
        path = native_key(packed.five_value_hf_to_mlx(name))
        kind = "conv1d" if "pointwise_conv" in path else "linear"
        shape = (out, 1, inp // 16) if kind == "conv1d" else (out, inp // 16)
        weights[path + ".weight"] = mx.array(qa.reshape(shape))
        weights[path + ".residualWeight"] = mx.array(qb.reshape(shape))
        scale = np.broadcast_to(lo[:,None], (out,groups)).copy()
        residual = np.broadcast_to(delta[:,None], (out,groups)).copy()
        for suffix, value in [("scales",scale),("biases",-scale),("residualScales",residual),("residualBiases",-residual)]:
            weights[path + "." + suffix] = mx.array(value)
        modules.append({"path": path, "kind": kind, "input_dimensions": inp, "output_dimensions": out})
    mx.save_safetensors(str(model_dir / "model.safetensors"), weights)
    shutil.copyfile(source / "config.json", model_dir / "config.json")
    for source_name, target_name in [("NOTICE","NOTICE.txt"),("LICENSE-WEIGHTS-CC-BY-4.0.txt","PHONON-2-CC-BY-4.0.txt"),("LICENSE-CODE-Apache-2.0.txt","FERMION-APACHE-2.0.txt")]:
        with urllib.request.urlopen(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{source_name}", timeout=60) as response:
            (notices / target_name).write_bytes(response.read())
    manifest = {"format": FORMAT, "repository": REPOSITORY, "revision": REVISION,
                "source_container_sha256": CONTAINER_SHA, "license": "CC-BY-4.0", "languages": ["en"],
                "conversion": "Exact five-value representation using two 2-bit affine MLX planes, FP32 scales, group size 128; remaining parameters BF16",
                "weight_equivalence": {"modules": len(modules), "max_absolute_error": 0.0}, "modules": modules,
                "files": {p.name: {"bytes":p.stat().st_size,"sha256":sha(p)} for p in [model_dir / "model.safetensors", model_dir / "config.json"]}}
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    assert sum(p.stat().st_size for p in model_dir.iterdir()) < 500_000_000
    print(json.dumps({"model_bytes":sum(p.stat().st_size for p in model_dir.iterdir()),"weight_equivalence":manifest["weight_equivalence"]}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source is None:
        archive = args.output / "source.bps.tar.zst"
        args.output.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            with urllib.request.urlopen(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/phonon-2.bps.tar.zst", timeout=120) as response, archive.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        assert sha(archive) == ARCHIVE_SHA, "Archive checksum mismatch"
        source = args.output / "source"
        source.mkdir(exist_ok=True)
        with archive.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as stream, tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if member.isfile() and Path(member.name).name in {"model.fermion", "config.json", "packed_manifest.json"}:
                    with tar.extractfile(member) as payload, (source / Path(member.name).name).open("wb") as target:
                        shutil.copyfileobj(payload, target)
        args.source = source
    prepare(args.output, args.source)
