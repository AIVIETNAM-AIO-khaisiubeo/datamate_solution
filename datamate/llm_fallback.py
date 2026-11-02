"""LLM fallback for questions the deterministic engine abstains on.

Design (CHASE-SQL / MAC-SQL style, adapted to pandas over the canonical model):
  1. prompt = split rulebook + business README + canonical schema with samples + resolved time window + question
  2. sample K candidate programs (Qwen3-14B via vLLM, fixed seeds -> reproducible)
  3. execute each candidate in a forked sandbox with a timeout; failures are retried once with the error (self-repair)
  4. normalise outputs to the submission format and vote; answer only on a clear majority, otherwise abstain
  ('unanswerable' is accepted only if every successful candidate says so)
Runs fully offline; the model weights are baked into the Docker image."""
import json, os, re, signal, traceback, multiprocessing as mp
from collections import Counter
import numpy as np
import pandas as pd

SYSTEM = """Bạn là chuyên gia phân tích dữ liệu bán hàng. Viết MỘT hàm Python `solve(D)` trả lời câu hỏi bằng pandas.
D là dict các DataFrame đã chuẩn hoá (mô tả bên dưới) và các tiện ích. Hàm trả về dict {"type": ..., "answer": ...}
theo đúng format rulebook mục 9 (number | percent | date | boolean | string | list | table | unanswerable).
Tuân thủ tuyệt đối các luật mặc định trong rulebook. Không đọc file, không dùng mạng. Chỉ trả về code trong ```python```."""

def schema_card(M, max_rows=3):
    parts = []
    def card(name, df, note=""):
        if df is None or len(df) == 0:
            return
        cols = [c for c in df.columns if not str(c).startswith("_")][:30]
        smp = df[cols].head(max_rows).to_dict("records")
        parts.append(f"### D['{name}'] ({len(df)} dòng) {note}\ncột: {cols}\nmẫu: {json.dumps(smp, ensure_ascii=False, default=str)[:1500]}")
    card("orders", M.orders, "mỗi dòng một đơn; status ∈ done/cancel/return/shipping; revenue = doanh thu đơn theo §1; date = ngày đặt")
    card("lines", M.lines, "mỗi dòng một dòng hàng; amount = thành tiền sau giảm dòng; qty_base = số lượng quy về đơn vị cơ bản; nonitem/is_service_fee đánh dấu dòng không phải hàng")
    card("products", pd.DataFrame(list(M.products.values())).drop(columns=["names"], errors="ignore"), "danh mục hàng (key = mã hoặc 'N:'+tên)")
    card("imports", M.imports, "dòng phiếu nhập; cancelled = phiếu huỷ")
    card("payments", M.payments, "mỗi lần khách trả tiền cho một đơn")
    card("expenses", M.expenses)
    card("deliveries", M.deliveries)
    card("lots", M.lots)
    card("lot_sales", M.lot_sales)
    for n, (df, roles, kind) in list(M.tables.items())[:25]:
        if kind in ("other", "staff", "customers", "debt_terms", "shifts", "returns", "abbrev"):
            card("raw:" + n, df, f"bảng gốc (loại {kind})")
    return "\n\n".join(parts)

def build_prompt(M, rulebook, readme, question, as_of, periods):
    win = "; ".join(f"{p.kind}: {p.start.date()}→{p.end.date()}" for p in periods) or "(không có)"
    return (f"{SYSTEM}\n\n## Rulebook\n{rulebook}\n\n## README của shop\n{readme}\n\n## Dữ liệu chuẩn hoá\n{schema_card(M)}\n\n"
            f"## Câu hỏi\nHôm nay (as_of_date) = {as_of}. Các mốc thời gian đã nhận diện: {win}\nCâu hỏi: {question}\n\n"
            "Viết hàm solve(D).")

def extract_code(text):
    m = re.search(r"```python\s*(.*?)```", text, flags=re.S)
    return m.group(1) if m else text

def _child(code, D, q):
    try:
        g = {"pd": pd, "np": np, "re": re}
        exec(code, g)
        r = g["solve"](D)
        q.put(("ok", json.loads(json.dumps(r, default=lambda x: x.item() if hasattr(x, "item") else str(x)))))
    except Exception:
        q.put(("err", traceback.format_exc()[-1500:]))

def run_program(code, D, timeout=20):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(code, D, q))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.kill(); return ("err", "timeout")
    try:
        return q.get_nowait()
    except Exception:
        return ("err", "no result")

def canon(r):
    """normalise a candidate answer for voting"""
    if not isinstance(r, dict) or "type" not in r:
        return None
    t = r["type"]; a = r.get("answer")
    try:
        if t == "number":
            a = float(a); a = int(round(a)) if abs(a - round(a)) < 1e-6 or abs(a) > 1000 else round(a, 2)
        elif t == "percent":
            a = round(float(a), 2)
        elif t == "list":
            a = [round(float(x), 2) if isinstance(x, (int, float)) else str(x) for x in a]
        elif t == "table":
            a = {"columns": list(a["columns"]), "rows": sorted([list(x) for x in a["rows"]], key=str)}
        elif t in ("string", "date"):
            a = str(a)
        elif t == "boolean":
            a = bool(a)
        elif t == "unanswerable":
            a = None
        else:
            return None
    except Exception:
        return None
    return {"type": t, "answer": a}

def vote(cands, k_total, min_agree):
    keys = [json.dumps(c, ensure_ascii=False, sort_keys=True) for c in cands if c is not None]
    if not keys:
        return None
    (best, n), = Counter(keys).most_common(1)
    if n < min_agree:
        return None
    r = json.loads(best)
    if r["type"] == "unanswerable" and n < len(keys):
        return None
    return r

class Fallback:
    def __init__(self, model_path, tp=2, k=5, max_tokens=1200, min_agree=3, seed=1234):
        from vllm import LLM, SamplingParams  # imported lazily; only inside the GPU image
        self.LLM = LLM(model=model_path, tensor_parallel_size=tp, max_model_len=32768, seed=seed, enforce_eager=False)
        self.SP = SamplingParams
        self.k, self.max_tokens, self.min_agree, self.seed = k, max_tokens, min_agree, seed

    def answer_batch(self, items):
        """items: list of dict(M, rulebook, readme, question, as_of, periods, D) -> list of answers or None"""
        prompts = [build_prompt(it["M"], it["rulebook"], it["readme"], it["question"], it["as_of"], it["periods"]) for it in items]
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        sp = [self.SP(n=self.k, temperature=0.7, top_p=0.95, max_tokens=self.max_tokens, seed=self.seed + i) for i in range(len(items))]
        outs = self.LLM.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
        results = []
        for it, o in zip(items, outs):
            cands, errs = [], []
            for comp in o.outputs:
                st, r = run_program(extract_code(comp.text), it["D"])
                if st == "ok":
                    cands.append(canon(r))
                else:
                    errs.append((comp.text, r))
            # one self-repair round for failed candidates
            if errs and len(cands) < self.min_agree:
                fix = [[{"role": "user", "content": prompts[items.index(it)]},
                        {"role": "assistant", "content": t},
                        {"role": "user", "content": f"Code lỗi:\n{e}\nSửa lại hàm solve(D)."}] for t, e in errs[:3]]
                rep = self.LLM.chat(fix, [self.SP(n=1, temperature=0.0, max_tokens=self.max_tokens, seed=self.seed)] * len(fix),
                                    chat_template_kwargs={"enable_thinking": False})
                for ro in rep:
                    st, r = run_program(extract_code(ro.outputs[0].text), it["D"])
                    if st == "ok":
                        cands.append(canon(r))
            results.append(vote(cands, self.k, self.min_agree))
        return results

def data_dict(M):
    return {"orders": M.orders, "lines": M.lines, "products": pd.DataFrame(list(M.products.values())),
            "imports": M.imports, "payments": M.payments, "expenses": M.expenses, "deliveries": M.deliveries,
            "lots": M.lots, "lot_sales": M.lot_sales,
            "raw": {n: df for n, (df, roles, kind) in M.tables.items()}}
