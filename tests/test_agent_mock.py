"""End-to-end test of the agent orchestration with a fake LLM: plan+code, SQL, traceback repair, voting, verifier, abstention."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dill
from datamate.agent import Agent, data_dict
from datamate.engine import Ctx
from datamate.intents import answer

PY_OK = '''```python
def solve(D):
    # bước 1: lọc đơn hợp lệ tháng 6/2025
    o = D["orders"]
    o = o[(o["date"] >= "2025-06-01") & (o["date"] <= "2025-06-30") & (~o["status"].isin(["cancel","return"]))]
    # bước 2: cộng doanh thu
    return {"type": "number", "answer": float(o["revenue"].sum()), "n_rows": len(o)}
```'''
PY_BAD = '''```python
def solve(D):
    return {"type": "number", "answer": D["orders"]["no_such_col"].sum(), "n_rows": 1}
```'''
PY_EMPTY = '''```python
def solve(D):
    o = D["orders"]; o = o[o["date"] > "2099-01-01"]
    return {"type": "number", "answer": 0, "n_rows": len(o)}
```'''
SQL_OK = '''```sql
-- bước 1: đơn hợp lệ tháng 6/2025 ; bước 2: tổng doanh thu
SELECT SUM(revenue) FROM orders WHERE date >= '2025-06-01' AND date <= '2025-06-30 23:59:59' AND status NOT IN ('cancel','return')
```
TYPE: number'''

calls = []
def fake_generate(model, jobs):
    calls.append((model, len(jobs)))
    out = {}
    for j in jobs:
        last = j["messages"][-1]["content"]
        if "CONFIDENCE" in last:
            out[j["id"]] = ["CONFIDENCE: 90"]
        elif "Traceback" in last:                       # repair round -> fixed program
            out[j["id"]] = [PY_OK]
        elif j["id"].endswith("|sql"):
            out[j["id"]] = [SQL_OK] * j["n"]
        else:
            out[j["id"]] = [PY_OK, PY_BAD, PY_EMPTY][: j["n"]]
    return out

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_cache = os.path.join(os.environ.get("CACHE_DIR", os.path.join(_root, "cache")), "biz_102.dill")
if os.path.exists(_cache):
    M = dill.load(open(_cache, "rb"))
else:
    from datamate.model import build_model
    M = build_model(os.path.join(os.environ.get("DATA_ROOT", "/home/claude/data"), "train", "data", "biz_102"))
q = "doanh thu tháng 6/2025"
D = data_dict(M)
items = [dict(key="q1", M=M, rulebook="(rulebook)", readme="(readme)", question=q, as_of="2026-06-16",
              periods=Ctx(M, q, "2026-06-16").periods, D=D)]
ag = Agent("CODER-7B", "VERIFIER-4B", generate=fake_generate)
res = ag.solve_all(items)
print("agent:", res)
print("rules:", answer(M, "doanh thu tháng 6/2025", "2026-06-16"))
print("LLM calls (model, #jobs):", calls)
