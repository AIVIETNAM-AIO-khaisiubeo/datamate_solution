#!/usr/bin/env bash
# Download the model pair into ./models/{gen,ver} (run once, with internet, before docker build).
# Option A (default, 11.6B): Qwen2.5-Coder-7B-Instruct + Qwen3-4B
# Option B (14.8B, one model for both roles): GEN=Qwen/Qwen3-14B VER=""
set -euo pipefail
cd "$(dirname "$0")/.."
GEN=${GEN:-Qwen/Qwen2.5-Coder-7B-Instruct}
VER=${VER-Qwen/Qwen3-4B}
pip install -q "huggingface_hub[cli]"
huggingface-cli download "$GEN" --local-dir models/gen
if [ -n "$VER" ]; then huggingface-cli download "$VER" --local-dir models/ver; else mkdir -p models/ver; fi
python3 tools/count_params.py models/gen $( [ -n "$VER" ] && echo models/ver )
