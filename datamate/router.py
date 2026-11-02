"""Combine the deterministic engine ('rules') with the agentic LLM ('agent').

mode='rules'  : engine only
mode='agent'  : agent only (engine is used only to resolve time windows for the prompt)
mode='hybrid' : cross-check
   rules answer + agent agrees        -> answer
   rules answer + agent disagrees     -> keep rules only if that handler's precision on the DEV split >= P_TRUST
                                         and the shop's data was fully understood (no data_partial flag), else abstain
   rules answer + agent abstains      -> keep rules (high-precision path)
   rules abstain + agent confident    -> agent answer (confidence >= TAU_AGENT_ONLY)
   rules 'unanswerable' + agent answers confidently -> abstain (conflict), otherwise keep 'unanswerable'
Thresholds are tuned on the sealed validation split in aggregate only (eval_sealed.py)."""
import os, json
from .intents import answer
from .engine import Ctx
from .agent import data_dict, canon

P_TRUST = float(os.environ.get("DM_P_TRUST", "0.99"))
TAU_AGENT_ONLY = float(os.environ.get("DM_TAU_AGENT_ONLY", "0.7"))
_PREC = None
_AGENT = None

def handler_precision():
    global _PREC
    if _PREC is None:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits", "handler_precision.json")
        _PREC = json.load(open(p)) if os.path.exists(p) else {}
    return _PREC

def get_agent():
    """DM_GEN_MODEL (code writer) and DM_VER_MODEL (verifier); DATAMATE_LLM is used for both when they are unset"""
    global _AGENT
    if _AGENT is None:
        from .agent import Agent
        gen = os.environ.get("DM_GEN_MODEL") or os.environ["DATAMATE_LLM"]
        ver = os.environ.get("DM_VER_MODEL") or gen
        fake = None
        if os.environ.get("DM_FAKE_LLM") == "1":
            import importlib, sys as _s
            _s.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            fake = importlib.import_module("tests.fake_llm").generate
        _AGENT = Agent(gen, ver, generate=fake, tp=int(os.environ.get("DATAMATE_TP", "2")), thinking=os.environ.get("DM_THINKING", "0") == "1",
                       max_repairs=int(os.environ.get("DM_MAX_REPAIRS", "2")), q_budget_s=float(os.environ.get("DM_Q_BUDGET_S", "40")),
                       tau=float(os.environ.get("DM_TAU", "0.55")), min_share=float(os.environ.get("DM_MIN_SHARE", "0.6")),
                       langs={"python": int(os.environ.get("DM_K_PY", "3")), "sql": int(os.environ.get("DM_K_SQL", "2"))})
    return _AGENT

def _same(a, b):
    ca, cb = canon(dict(a, n_rows=1)), canon(dict(b, n_rows=1))
    return isinstance(ca, dict) and ca == cb

def agent_items(M, qs, cal, ctx):
    D = data_dict(M)
    return [dict(key=q["id"], M=M, rulebook=ctx.get("rulebook", ""), readme=ctx.get("readme", ""), question=q["question"],
                 as_of=q["as_of_date"], periods=Ctx(M, q["question"], q["as_of_date"], cal).periods, D=D) for q in qs]

def combine(mode, rules, agent_ans, qs, p_trust=None, tau_agent=None):
    """rules: list of engine results; agent_ans: {qid: agent result}"""
    p_trust = P_TRUST if p_trust is None else p_trust
    tau_agent = TAU_AGENT_ONLY if tau_agent is None else tau_agent
    out = []
    prec = handler_precision()
    if mode == "agent":
        return [dict(agent_ans.get(q["id"], {"type": "abstain"}), handler="agent") for q in qs]
    for i, r in enumerate(rules):
        a = agent_ans.get(qs[i]["id"], {"type": "abstain"})
        a_ok = a.get("type") not in ("abstain", None)
        t = r.get("type")
        if t not in ("abstain", "unanswerable"):
            if not a_ok or _same(r, a):
                out.append(dict(r, cross_check="agree" if a_ok else "agent_abstained"))
            elif prec.get(r.get("handler"), 0.0) >= p_trust and not r.get("data_partial"):
                out.append(dict(r, cross_check="conflict_kept_trusted_rule"))
            else:
                out.append({"type": "abstain", "reason": "rules/agent conflict", "handler": r.get("handler")})
        elif t == "unanswerable":
            if a_ok and a.get("type") != "unanswerable" and a.get("confidence", 0) >= tau_agent:
                out.append({"type": "abstain", "reason": "unanswerable conflict", "handler": r.get("handler")})
            else:
                out.append(r)
        else:
            if a_ok and a.get("type") != "unanswerable" and a.get("confidence", 0) >= tau_agent:
                out.append(dict(a, handler="agent"))
            else:
                out.append(r)
    return out

def route_batch(mode, M, qs, cal, ctx, agent=None, deadline=None):
    rules = [answer(M, q["question"], q["as_of_date"], cal) for q in qs]
    if mode == "rules":
        return rules
    agent = agent or get_agent()
    ag = agent.solve_all(agent_items(M, qs, cal, ctx), deadline=deadline)
    return combine(mode, rules, ag, qs)
