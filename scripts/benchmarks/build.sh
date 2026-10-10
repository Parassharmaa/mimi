#!/bin/zsh
set -euo pipefail
MIMI_BENCH_ROOT="${0:A:h:h:h}"
cd "$MIMI_BENCH_ROOT"
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts/speech')
from evidence_identity import adaptive_segmentation_implementation_sha256
identity = adaptive_segmentation_implementation_sha256()
path = Path('Sources/Mimi/AdaptiveSegmentationBuildIdentity.swift')
text = 'enum MimiAdaptiveSegmentationBuildIdentity {\n    static let sha256 = "' + identity + '"\n}\n'
if path.read_text() != text: path.write_text(text)
PY
swift build --disable-keychain --disable-index-store -c release --product Mimi
scripts/prepare-mlx-metallib.sh .build/release release required
python3 - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
environment = json.loads(subprocess.check_output(['.build/release/Mimi', '--benchmark-environment'], text=True))
receipt = {'schemaVersion':1, 'builtAt':datetime.now(timezone.utc).isoformat(),
           'command':'swift build --disable-keychain --disable-index-store -c release --product Mimi',
           'compiler':subprocess.check_output(['swift','--version'],text=True).strip(),
           'appSha256':sha('.build/release/Mimi'), 'shaderSha256':sha('.build/release/mlx.metallib'),
           'packageResolvedSha256':sha('Package.resolved'), 'environment':environment}
Path('.build/benchmark-build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('.build/benchmark-build-receipt.json')
PY
