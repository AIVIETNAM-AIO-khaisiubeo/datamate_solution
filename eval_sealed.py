"""Aggregate-only evaluation on the sealed validation shops.
Prints PA / coverage / precision / per-type aggregates. Never prints question ids, questions, predictions or gold.
Usage: python eval_sealed.py [--data-root DIR] [--mode rules|agent|hybrid] [--perturb none|questions] [--data-variant DIR]"""
import sys, os, json, argparse, collections, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamate.model import build_model
from datamate.intents import answer
from datamate.timeparse import load_calendar
from datamate.scorer import score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/home/claude/data"))
    ap.add_argument("--data-variant", default=None, help="folder with perturbed copies of the sealed shops (same business ids)")
    ap.add_argument("--mode", default="rules")
    ap.add_argument("--perturb", default="none", choices=["none", "questions"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default=None, help="save raw rules/agent outputs (no gold) for tools/sweep.py; agent thresholds are disabled while dumping")
    ap.add_argument("--limit", type=int, default=0, help="evaluate a fixed random subset (per-shop stratified) to save GPU time")
    a = ap.parse_args()
    if a.dump:
        os.environ["DM_TAU"] = "0"; os.environ["DM_MIN_SHARE"] = "0"   # keep every agent winner + its scores
    sealed = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "split.json")))["sealed"])
    root = os.path.join(a.data_root, "train")
    Q = [json.loads(l) for l in open(os.path.join(root, "questions.jsonl"), encoding="utf-8")]
    A = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(root, "answers.jsonl"), encoding="utf-8")}
    cal = load_calendar(os.path.join(root, "calendar.csv"))
    Q = [q for q in Q if q["business_id"] in sealed]
    if a.limit:
        import random
        rnd = random.Random(1234)
        per = collections.defaultdict(list)
        for q in Q:
            per[q["business_id"]].append(q)
        frac = a.limit / len(Q)
        Q = [q for b in sorted(per) for q in rnd.sample(per[b], max(1, round(len(per[b]) * frac)))]
    if a.perturb == "questions":
        from tools_perturb import perturb_question
        import random
        rnd = random.Random(a.seed)
        Q = [dict(q, question=perturb_question(q["question"], rnd)) for q in Q]
    by = collections.defaultdict(list)
    for q in Q:
        by[q["business_id"]].append(q)
    ctx = {"rulebook": open(os.path.join(root, "rulebook.md"), encoding="utf-8").read()}
    sc = collections.Counter(); by_type = collections.defaultdict(collections.Counter); unans = collections.Counter()
    t0 = time.time()
    for b in sorted(by):
        bdir = os.path.join(a.data_variant, b) if a.data_variant else os.path.join(root, "data", b)
        try:
            M = build_model(bdir)
        except Exception:
            M = None
        if M is None:
            rs = [{"type": "abstain"}] * len(by[b])
        elif a.mode == "rules":
            rs = []
            for q in by[b]:
                try:
                    rs.append(answer(M, q["question"], q["as_of_date"], cal))
                except Exception:
                    rs.append({"type": "abstain"})
        else:
            from datamate.router import get_agent, agent_items, combine
            ctx["readme"] = open(os.path.join(bdir, "README.md"), encoding="utf-8").read()
            try:
                rules = [answer(M, q["question"], q["as_of_date"], cal) for q in by[b]]
                ag = get_agent().solve_all(agent_items(M, by[b], cal, ctx))
                if a.dump:
                    with open(a.dump, "a", encoding="utf-8") as f:
                        for q, r in zip(by[b], rules):
                            f.write(json.dumps({"id": q["id"], "rules": r, "agent": ag.get(q["id"])}, ensure_ascii=False, default=str) + "\n")
                    ag = {k: (v if v.get("type") == "abstain" or (v.get("share", 0) >= 0.6 and v.get("confidence", 0) >= 0.55)
                              else {"type": "abstain"}) for k, v in ag.items()}
                rs = combine(a.mode, rules, ag, by[b])
            except Exception:
                import traceback; traceback.print_exc()
                rs = [{"type": "abstain"}] * len(by[b])
        for q, r in zip(by[b], rs):
            s = score(r, A[q["id"]])
            sc[s] += 1
            by_type[A[q["id"]]["type"]][s] += 1
            if r.get("type") == "unanswerable":
                unans[s] += 1
    n = sum(sc.values())
    ans = sc[1] + sc[-1]
    print(f"sealed shops={len(by)} questions={n} mode={a.mode} perturb={a.perturb} variant={'yes' if a.data_variant else 'no'}")
    print(f"PA={(sc[1]-sc[-1])/n:.4f}  correct={sc[1]} wrong={sc[-1]} abstain={sc[0]}  coverage={ans/n:.3f}  precision={sc[1]/max(ans,1):.4f}")
    print(f"unanswerable emitted: right={unans[1]} wrong={unans[-1]}")
    print("by gold type (correct/wrong/abstain):", {t: (c[1], c[-1], c[0]) for t, c in sorted(by_type.items())})
    print(f"time {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
