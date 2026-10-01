#!/usr/bin/env python3
"""Research-only native utility replacement; never patches Mimi's dependency."""
import argparse
from pathlib import Path
import subprocess

PIN = '234f60faa0eb388b01cdf7e60aca232af37aefda'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('native_release', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sources = {
        'ort-utils.cpp': 'core/ort-utils/ort-utils.cpp',
        'ort-utils.h': 'core/ort-utils/ort-utils.h',
        'debug-utils.h': 'core/moonshine-utils/debug-utils.h',
        'onnxruntime_c_api.h': 'core/third-party/onnxruntime/include/onnxruntime_c_api.h',
        'onnxruntime_ep_c_api.h': 'core/third-party/onnxruntime/include/onnxruntime_ep_c_api.h',
    }
    for name, source in sources.items():
        subprocess.run(['curl', '-fsSL', f'https://raw.githubusercontent.com/moonshine-ai/moonshine/{PIN}/{source}',
                        '-o', str(a.output / name)], check=True)
    source = a.output / 'ort-utils.cpp'
    text = source.read_text()
    assert '"session.disable_prepacking", "1"' in text
    assert 'ort_api->DisableCpuMemArena(session_options)' in text
    source.write_text(text.replace('"session.disable_prepacking", "1"', '"session.disable_prepacking", "0"')
                     .replace('ort_api->DisableCpuMemArena(session_options)', 'ort_api->EnableCpuMemArena(session_options)'))
    subprocess.run(['clang++', '-std=c++17', '-O3', '-I', str(a.output), '-c', str(source),
                    '-o', str(a.output / 'ort-utils.o')], check=True)
    probe = Path(__file__).resolve().parents[1] / 'src/moonshine_probe.cpp'
    command = ['clang++', '-std=c++17', '-O3', str(probe), str(a.output / 'ort-utils.o'),
               '-I', str(a.native_release / 'include'), str(a.native_release / 'lib/libmoonshine.a')]
    for framework in ['Accelerate', 'Foundation', 'CoreML', 'Security', 'AudioToolbox', 'Metal', 'MetalPerformanceShaders']:
        command.extend(['-framework', framework])
    command.extend(['-o', str(a.output / 'probe')])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
