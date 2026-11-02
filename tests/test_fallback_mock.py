"""Exercise the LLM-fallback plumbing (prompt, sandbox, repair, vote) with a fake LLM — no GPU needed."""
import sys, types
sys.path.insert(0, "/home/claude/dm")
import dill
from datamate import llm_fallback as F
from datamate.engine import Ctx

GOOD = '''```python
def solve(D):
    o = D["orders"]
    o = o[(o["date"] >= "2025-06-01") & (o["date"] <= "2025-06-30") & (~o["status"].isin(["cancel","return"]))]
    return {"type": "number", "answer": float(o["revenue"].sum())}
```'''
BAD = '''```python
def solve(D):
    return D["nope"]
```'''

class Out:
    def __init__(self, t): self.text = t
class Res:
    def __init__(self, ts): self.outputs = [Out(t) for t in ts]

class FakeLLM:
    def chat(self, msgs, sps, chat_template_kwargs=None):
        res = []
        for i, m in enumerate(msgs):
            n = getattr(sps[i] if isinstance(sps, list) else sps, "n", 1)
            res.append(Res(([GOOD, GOOD, BAD, GOOD, BAD] * 2)[:n] if n > 1 else [GOOD]))
        return res

class SP:
    def __init__(self, **kw): self.__dict__.update(kw)

fb = F.Fallback.__new__(F.Fallback)
fb.LLM, fb.SP, fb.k, fb.max_tokens, fb.min_agree, fb.seed = FakeLLM(), SP, 5, 100, 3, 1
M = dill.load(open("/home/claude/dm/cache/biz_102.dill", "rb"))
q = "doanh thu thang 6/2025"
item = dict(M=M, rulebook="(rulebook)", readme="(readme)", question=q, as_of="2026-06-16",
            periods=Ctx(M, q, "2026-06-16").periods, D=F.data_dict(M))
print("prompt chars:", len(F.build_prompt(M, "r", "rd", q, "2026-06-16", item["periods"])))
print(fb.answer_batch([item]))
from datamate.intents import answer
print(answer(M, "doanh thu tháng 6/2025", "2026-06-16"))
