#!/bin/zsh
set -euo pipefail
ROOT="${0:A:h:h:h}"
DESTINATION="${1:-$ROOT/.build/phonon2-model}"
if [[ -f "$DESTINATION/model/manifest.json" ]]; then
  python3 "$ROOT/scripts/speech/verify_phonon2_pack.py" "$DESTINATION/model"
  exit 0
fi
if ! command -v uv >/dev/null; then
  print -u2 "Preparing the Phonon model requires uv on the developer's build machine. Mimi users do not need Python."
  exit 1
fi
uv run --no-project --python 3.12 \
  --with 'fermion-research==0.2.3' --with 'mlx==0.32.3' --with 'zstandard==0.25.0' \
  "$ROOT/scripts/speech/prepare_phonon2_mlx.py" "$DESTINATION"
python3 "$ROOT/scripts/speech/verify_phonon2_pack.py" "$DESTINATION/model"
