#!/usr/bin/env bash
# Usage (organizers): ./predict.sh /data /result/submission.jsonl
set -euo pipefail
DATA=${1:-/data}
OUT=${2:-/result/submission.jsonl}
mkdir -p "$(dirname "$OUT")/logs"
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Model routing (total params <= 15B, see README). Defaults: Qwen2.5-Coder-7B writes code, Qwen3-4B verifies.
export DM_GEN_MODEL=${DM_GEN_MODEL:-/opt/models/gen}
export DM_VER_MODEL=${DM_VER_MODEL:-/opt/models/ver}
export DATAMATE_TP=${DATAMATE_TP:-2} DM_MODE=${DM_MODE:-hybrid}
if [ ! -f "$DM_GEN_MODEL/config.json" ]; then export DM_MODE=rules; unset DM_GEN_MODEL DM_VER_MODEL; fi
if [ -n "${DM_VER_MODEL:-}" ] && [ ! -f "$DM_VER_MODEL/config.json" ]; then export DM_VER_MODEL=$DM_GEN_MODEL; fi
python3 predict.py "$DATA" "$OUT" --logs "$(dirname "$OUT")/logs" --mode "$DM_MODE" 2>&1 | tee "$(dirname "$OUT")/logs/run.log"
