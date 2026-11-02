"""Entry point: python predict.py <data_dir> <out_submission.jsonl> [--logs DIR] [--mode rules|hybrid|agent]

Phase 1 (CPU)  : build the canonical model of every business, answer every question with the deterministic engine,
                 write a complete submission immediately (so a crash/time-out later still leaves valid answers).
Phase 2 (GPU)  : agentic LLM (code-writer + verifier models in separate worker processes) on all questions,
                 combined with phase 1 by the router; the submission is rewritten atomically.
A global deadline (DM_DEADLINE_S, default 5h20m from start) stops LLM work early; unanswered = abstain."""
import sys, os, json, time, argparse, traceback, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamate.model import build_model
from datamate.intents import answer
from datamate.timeparse import load_calendar

def to_submission(r, qid):
    t = r.get("type")
    if t in (None, "abstain"):
        return {"id": qid, "type": "abstain"}
    if t == "unanswerable":
        return {"id": qid, "type": "unanswerable", "reason": str(r.get("reason", ""))}
    return {"id": qid, "type": t, "answer": r["answer"]}

def write_all(path, Q, res, policy):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for q in Q:
            r = res.get(q["id"], {"type": "abstain"})
            if r.get("type") == "unanswerable" and policy == "abstain":
                r = {"type": "abstain"}
            f.write(json.dumps(to_submission(r, q["id"]), ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def main():
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir"); ap.add_argument("out")
    ap.add_argument("--logs", default=None)
    ap.add_argument("--mode", default=os.environ.get("DM_MODE", "rules"), choices=["rules", "hybrid", "agent"])
    ap.add_argument("--unanswerable-policy", default=os.environ.get("DM_UNANS", "emit"), choices=["emit", "abstain"])
    a = ap.parse_args()
    deadline = T0 + float(os.environ.get("DM_DEADLINE_S", str(5 * 3600 + 20 * 60)))
    logs = a.logs or os.path.join(os.path.dirname(os.path.abspath(a.out)), "logs")
    os.makedirs(logs, exist_ok=True)
    Q = [json.loads(l) for l in open(os.path.join(a.data_dir, "questions.jsonl"), encoding="utf-8")]
    cal = load_calendar(os.path.join(a.data_dir, "calendar.csv"))
    rb_path = os.path.join(a.data_dir, "rulebook.md")
    rulebook = open(rb_path, encoding="utf-8").read() if os.path.exists(rb_path) else ""
    by = collections.defaultdict(list)
    for q in Q:
        by[q["business_id"]].append(q)
    # ---------------- phase 1: deterministic engine
    models, res, meta = {}, {}, {}
    for b in sorted(by):
        t0 = time.time()
        try:
            models[b] = build_model(os.path.join(a.data_dir, "data", b)); err = None
        except Exception:
            models[b] = None; err = traceback.format_exc()[-2000:]
        tb = time.time() - t0
        for q in by[b]:
            t = time.time()
            try:
                r = answer(models[b], q["question"], q["as_of_date"], cal) if models[b] is not None else {"type": "abstain", "reason": "model build failed"}
            except Exception as e:
                r = {"type": "abstain", "reason": "exception: " + repr(e)[:300]}
            res[q["id"]] = r
            meta[q["id"]] = {"business_id": b, "rules": dict(r), "rules_s": round(time.time() - t, 4), "model_build_s": round(tb, 2), "model_error": err}
    write_all(a.out, Q, res, a.unanswerable_policy)
    print(f"[phase1] rules done in {time.time()-T0:.0f}s", flush=True)
    # ---------------- phase 2: agentic LLM + router
    if a.mode != "rules" and (os.environ.get("DM_GEN_MODEL") or os.environ.get("DATAMATE_LLM")):
        try:
            from datamate.router import get_agent, agent_items, combine
            agent = get_agent()
            # agentic data repair: unseen order-status labels -> ask the LLM (unanimous self-consistency), re-run rules
            from datamate.model import unknown_status_labels, apply_status_map
            from datamate.agent import classify_status_labels
            for b in sorted(by):
                M = models[b]
                if M is not None and getattr(M, "unknown_status_share", 0) > 0.03 and time.time() < deadline:
                    mapping = classify_status_labels(agent.generate, agent.ver_model, unknown_status_labels(M))
                    if mapping:
                        apply_status_map(M, mapping)
                        for q in by[b]:
                            try:
                                res[q["id"]] = answer(M, q["question"], q["as_of_date"], cal)
                            except Exception:
                                pass
                        print(f"[phase2] {b}: status labels mapped {mapping}", flush=True)
            print("[phase2] building agent prompts", flush=True)
            items, qs_all = [], []
            for b in sorted(by):
                if models[b] is None:
                    continue
                ctx = {"rulebook": rulebook, "readme": open(os.path.join(a.data_dir, "data", b, "README.md"), encoding="utf-8").read()}
                items += agent_items(models[b], by[b], cal, ctx); qs_all += by[b]
            t = time.time()
            print(f"[phase2] {len(items)} questions -> agent", flush=True)
            ag = agent.solve_all(items, deadline=deadline)
            print(f"[phase2] agent answered {sum(1 for v in ag.values() if v.get('type') not in ('abstain', None))}/{len(items)}", flush=True)
            merged = combine(a.mode, [res[q["id"]] for q in qs_all], ag, qs_all)
            for q, r in zip(qs_all, merged):
                res[q["id"]] = r
                meta[q["id"]]["agent"] = ag.get(q["id"])
            write_all(a.out, Q, res, a.unanswerable_policy)
            print(f"[phase2] agent done in {time.time()-t:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()   # phase-1 submission already on disk
    # ---------------- logs
    for q in Q:
        r = res.get(q["id"], {})
        with open(os.path.join(logs, q["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(dict(id=q["id"], question=q["question"], as_of_date=q["as_of_date"], final=to_submission(r, q["id"]),
                           handler=r.get("handler"), reason=r.get("reason"), cross_check=r.get("cross_check"), **meta.get(q["id"], {})),
                      f, ensure_ascii=False, default=str)
    print(f"[done] total {time.time()-T0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
