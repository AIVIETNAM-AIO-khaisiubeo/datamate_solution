#!/usr/bin/env bash
# Make coarse A/B submission files from one run (no per-question probing):
#   A = full submission
#   B = same, but every 'unanswerable' turned into abstain  -> PA(A)-PA(B) = (correct_unans - wrong_unans)/N
#   C = same, but agent-only answers (rules abstained, agent answered) turned into abstain   -> PA(A)-PA(C) = net value of the LLM fallback
# Usage: tools/ablation.sh result_dir   (needs result_dir/submission.jsonl and result_dir/logs)
set -euo pipefail
R=$(realpath "$1")
python3 - "$R" <<'PY'
import json, sys, os, glob
R = sys.argv[1]
S = [json.loads(l) for l in open(os.path.join(R, "submission.jsonl"))]
llm = set()
for f in glob.glob(os.path.join(R, "logs", "*.json")):
    try:
        d = json.load(open(f))
        if d.get("handler") in ("llm_fallback", "agent"): llm.add(d["id"])
    except Exception: pass
def dump(name, rows):
    with open(os.path.join(R, name), "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, sum(r["type"] != "abstain" for r in rows), "answered")
dump("sub_A_full.jsonl", S)
dump("sub_B_no_unanswerable.jsonl", [{"id": r["id"], "type": "abstain"} if r["type"] == "unanswerable" else r for r in S])
dump("sub_C_no_llm.jsonl", [{"id": r["id"], "type": "abstain"} if r["id"] in llm else r for r in S])
PY
