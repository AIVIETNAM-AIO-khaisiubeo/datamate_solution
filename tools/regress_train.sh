#!/usr/bin/env bash
# Inside the image (or any env with requirements installed): score the deterministic engine on the train split.
# Expected: PA = 1.0000 (3000/3000). Anything lower means a library-version difference changed behaviour.
# Usage: tools/regress_train.sh /path/to/split_root   (folder that contains train/)
set -euo pipefail
export DATA_ROOT=$(realpath "$1") CACHE_DIR=${CACHE_DIR:-/tmp/dm_cache}
mkdir -p "$CACHE_DIR"
python3 "$(dirname "$0")/../run_eval.py" train "" --rebuild
