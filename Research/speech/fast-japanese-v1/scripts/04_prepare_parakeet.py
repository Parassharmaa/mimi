#!/usr/bin/env python3
"""Fetch verified BF16 source and reproduce Mimi's selected encoder-onlyQ4 pack."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mimi', type=Path)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    validation = json.loads((ROOT / 'configs/parakeet-source-validation.json').read_text())
    a.source.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{validation['repo']}/resolve/{validation['revision']}/"
    for file in validation['files']:
        target = a.source / file['file']
        if not target.exists() or target.stat().st_size != file['bytes'] or sha(target) != file['sha256']:
            partial = target.with_suffix(target.suffix + '.partial')
            subprocess.run(['curl', '-fL', '--retry', '2', base + file['file'], '-o', str(partial)], check=True)
            if partial.stat().st_size != file['bytes'] or sha(partial) != file['sha256']:
                raise RuntimeError('Pinned source integrity failure: ' + file['file'])
            partial.replace(target)
    environment = dict(os.environ, MIMI_PARAKEET_Q4_ENCODER_ONLY="1")
    subprocess.run([str(a.mimi.resolve()), '--convert-parakeet-ja-q4', str(a.source.resolve()),
                    '--output', str(a.output.resolve())], check=True, env=environment)
    expected = {'model.safetensors': (481725023, '84163fa7b961a579a63ae491ca04ebfac3078fa1d13bc1c5d08accd0a8e50c1c'),
                'config.json': (132341, '1d850ea4e1fffa94f43251707d260127b8489b73261e09bdc5f130ce5fccc9f2')}
    for name, (size, digest) in expected.items():
        target = a.output / name
        if target.stat().st_size != size or sha(target) != digest:
            raise RuntimeError('Conversion differs from the qualified pack: ' + name)
    print('Reproduced exact qualified encoder-Q4 model/config digests.')


if __name__ == '__main__':
    main()
