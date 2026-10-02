#!/usr/bin/env python3
"""Reject a signed Mimi bundle that cannot request microphone access."""
import argparse
from pathlib import Path
import plistlib
import subprocess
import tempfile


def verify(app: Path) -> None:
    with tempfile.TemporaryDirectory(prefix='mimi-audio-entitlements-') as directory:
        entitlements_file = Path(directory) / 'entitlements.plist'
        subprocess.run(['codesign', '--display', '--xml', '--entitlements',
                        str(entitlements_file), str(app)], check=True, capture_output=True)
        entitlements = plistlib.loads(entitlements_file.read_bytes())
    if entitlements.get('com.apple.security.device.audio-input') is not True:
        raise SystemExit('Mimi microphone verification failed: signed bundle is missing the hardened-runtime audio-input entitlement')
    info = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
    if not str(info.get('NSMicrophoneUsageDescription', '')).strip():
        raise SystemExit('Mimi microphone verification failed: missing microphone usage description')
    print('Mimi microphone signing contract passed: audio-input entitlement and usage description are present.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('app', type=Path)
    verify(parser.parse_args().app)
