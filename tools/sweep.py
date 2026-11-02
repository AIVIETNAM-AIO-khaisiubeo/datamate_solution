"""Sweep router/agent thresholds on a dump made by `eval_sealed.py --mode hybrid --dump FILE`.
Prints ONLY aggregate PA / wrong / coverage per setting (sealed protocol).
Usage: python tools/sweep.py dump.jsonl [--data-root DIR]"""
import sys, os, json, argparse, itertools, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datamate.scorer import score
from datamate.router import combine

ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/home/claude/data"))
a = ap.parse_args()
A = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(a.data_root, "train", "answers.jsonl"), encoding="utf-8")}
D = [json.loads(l) for l in open(a.dump, encoding="utf-8")]
qs = [{"id": d["id"]} for d in D]
rules = [d["rules"] for d in D]

def agent_view(min_share, tau):
    out = {}
    for d in D:
        g = d.get("agent") or {"type": "abstain"}
        if g.get("type") != "abstain" and (g.get("share", 0) < min_share or g.get("confidence", 0) < tau):
            g = {"type": "abstain"}
        out[d["id"]] = g
    return out

def pa(preds):
    c = collections.Counter(score(p, A[q["id"]]) for p, q in zip(preds, qs))
    n = len(qs)
    return (c[1] - c[-1]) / n, c[-1], (c[1] + c[-1]) / n

print(f"n={len(D)}")
n_ag = sum(1 for d in D if (d.get("agent") or {}).get("type") not in ("abstain", None))
print(f"agent produced a (non-abstain) answer for {n_ag}/{len(D)} questions")
if n_ag == 0:
    print("!! AGENT PRODUCED NOTHING -- the LLM worker failed; look for '[llm] WORKER FAILED' in the eval log. Thresholds below are meaningless.")
print("rules only        PA=%.4f wrong=%d coverage=%.3f" % pa(rules))
best = None
for ms, tau in itertools.product([0.4, 0.6, 0.8, 1.0], [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]):
    ag = agent_view(ms, tau)
    r_agent = pa(combine("agent", rules, ag, qs))
    for pt in [0.95, 0.99, 2.0]:              # 2.0 = never trust rules over a disagreeing agent
        for ta in [0.5, 0.7, 0.9]:
            r_h = pa(combine("hybrid", rules, ag, qs, p_trust=pt, tau_agent=ta))
            if best is None or r_h[0] > best[0][0]:
                best = (r_h, ms, tau, pt, ta)
    print(f"min_share={ms:.1f} tau={tau:.1f}  agent PA={r_agent[0]:.4f} wrong={r_agent[1]} cov={r_agent[2]:.3f}")
(r, ms, tau, pt, ta) = best
print(f"BEST hybrid PA={r[0]:.4f} wrong={r[1]} cov={r[2]:.3f}  -> DM_MIN_SHARE={ms} DM_TAU={tau} DM_P_TRUST={pt} DM_TAU_AGENT_ONLY={ta}")
