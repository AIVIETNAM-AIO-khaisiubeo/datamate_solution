#!/usr/bin/env bash
# Run the image exactly like the organizers: no network, data mounted read-only at /data, output in /result.
# Usage: tools/smoke_test.sh /path/to/public_test [/path/to/result]
set -euo pipefail
DATA=$(realpath "$1"); RES=$(realpath -m "${2:-./result_smoke}")
TAG=${TAG:-datamate:final}
mkdir -p "$RES"
start=$(date +%s)
docker run --rm --gpus all --network none --ipc=host \
  -v "$DATA":/data:ro -v "$RES":/result "$TAG" /data /result/submission.jsonl
end=$(date +%s)
echo "wall-clock: $((end-start)) s for $(wc -l < "$DATA/questions.jsonl") questions"
python3 "$(dirname "$0")/../validate_submission.py" "$DATA/questions.jsonl" "$RES/submission.jsonl"
python3 - "$RES/submission.jsonl" <<'PY'
import json,sys,collections
S=[json.loads(l) for l in open(sys.argv[1])]
print(collections.Counter(s["type"] for s in S))
PY
grep -l '"handler": "llm_fallback"' "$RES"/logs/*.json 2>/dev/null | wc -l | xargs echo "answers produced by LLM fallback:"
