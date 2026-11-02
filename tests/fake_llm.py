"""Deterministic stand-in for the LLM workers (tests only; enabled with DM_FAKE_LLM=1)."""
import json
PY = '''```python
def solve(D):
    # bước 1: đếm đơn hợp lệ (cố tình là câu trả lời 'ngây thơ' để kiểm tra cơ chế xung đột)
    o = D["orders"]; o = o[~o["status"].isin(["cancel","return"])]
    return {"type": "number", "answer": int(len(o)), "n_rows": len(o)}
```'''
SQL = '''```sql
-- bước 1: đếm đơn hợp lệ
SELECT COUNT(*) FROM orders WHERE status NOT IN ('cancel','return')
```
TYPE: number'''
def generate(model, jobs):
    out = {}
    for j in jobs:
        last = j["messages"][-1]["content"]
        if "CONFIDENCE" in last:
            out[j["id"]] = ["CONFIDENCE: 80"]
        elif "Nhãn:" in last:
            out[j["id"]] = ["{}"] * j.get("n", 1)
        elif j["id"].endswith("|sql"):
            out[j["id"]] = [SQL] * j.get("n", 1)
        else:
            out[j["id"]] = [PY] * j.get("n", 1)
    return out
