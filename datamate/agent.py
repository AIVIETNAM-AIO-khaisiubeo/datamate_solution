"""Agentic answering: the LLM inspects table structure (column names + sample rows) and writes its own pandas code.

Pipeline per question
  1. context  : rulebook + shop README + schema cards of the canonical tables AND of the raw tables
                (column names, dtypes, 3 sample rows) + resolved time windows + as-of date
  2. generate : K independent programs `solve(D)` (Qwen3-14B via vLLM, fixed seeds -> reproducible)
  3. execute  : each program in a forked sandbox with a timeout; it must also report how many rows its answer
                was computed from (`n_rows`)
  4. repair   : failed programs get one repair round with the traceback (MAC-SQL style refiner)
  5. vote     : normalise answers, majority vote (CHASE-SQL style selection by agreement)
  6. verify   : the LLM grades the winning program against the rulebook and returns a confidence 0-100
Abstain when: every program raised / timed out, the winning answer was computed from 0 rows (empty selection),
the vote share is below `min_share`, or vote_share * verifier_confidence is below `tau`."""
import json, re, traceback, multiprocessing as mp
from collections import Counter
import numpy as np
import pandas as pd

GEN_INSTR_SQL = """Bạn là chuyên gia SQL. Các bảng (cấu trúc và dòng mẫu bên dưới) đã được nạp vào SQLite với cùng tên
(orders, lines, products, imports, payments, ...; bảng gốc tên raw_1, raw_2, ... theo thứ tự liệt kê).
Bước 1 — KẾ HOẠCH: viết các bước tính dưới dạng comment SQL (-- bước 1: ..., -- bước 2: ...), tách câu hỏi phức tạp thành phép tính nhỏ.
Bước 2 — Viết MỘT câu lệnh SELECT (được dùng CTE WITH) trả về kết quả cuối cùng, tuân thủ luật trong rulebook.
Trả lời đúng một khối ```sql``` và ngay sau đó một dòng: TYPE: <number|percent|date|boolean|string|list|table|unanswerable>
Quy ước: number/percent/date/string/boolean = ô đầu tiên của dòng đầu; list = cột đầu theo thứ tự dòng; table = toàn bộ kết quả."""

GEN_INSTR = """Bạn là chuyên gia phân tích dữ liệu. Viết MỘT hàm Python `solve(D)` dùng pandas để trả lời câu hỏi.
- Bước 1 — KẾ HOẠCH: ở đầu hàm, viết các bước tính dưới dạng comment (# bước 1: ..., # bước 2: ...). Câu hỏi so sánh/tỉ lệ
  phải tách thành các đại lượng con (ví dụ: tính doanh thu Q1, tính doanh thu Q2, rồi (Q2-Q1)/Q1*100).
- D["orders"], D["lines"], ... là các bảng đã chuẩn hoá; D["raw"][tên] là bảng gốc. Xem cấu trúc và dòng mẫu bên dưới.
- Tự quyết định phép toán (lọc, sum, mean, groupby, sort...). Tuân thủ đúng luật mặc định trong rulebook.
- Hàm trả về dict: {"type": <number|percent|date|boolean|string|list|table|unanswerable>, "answer": <giá trị đúng format rulebook mục 9>,
  "n_rows": <số dòng dữ liệu đã dùng để tính>}.
- Nếu dữ liệu không có thông tin cần thiết thì trả {"type": "unanswerable", "answer": null, "n_rows": 0}.
- Không đọc file, không dùng mạng, không in ra. Chỉ trả lời bằng một khối ```python```."""

VERIFY_INSTR = """Bạn là người chấm. Cho câu hỏi, các luật liên quan, đoạn code và kết quả.
Code có tuân thủ đúng luật và trả lời đúng điều câu hỏi hỏi không (kỳ thời gian, bộ lọc, cách đếm, làm tròn, format)?
Chỉ trả lời một dòng: CONFIDENCE: <0-100>"""

def _card(name, df, note="", rows=3):
    if df is None or not hasattr(df, "columns") or len(df) == 0:
        return ""
    cols = [c for c in df.columns if not str(c).startswith("_")][:40]
    dtypes = {c: str(df[c].dtype) for c in cols}
    smp = df[cols].head(rows).to_dict("records")
    return (f"### {name}  ({len(df)} dòng) {note}\n"
            f"cột/kiểu: {json.dumps(dtypes, ensure_ascii=False)}\n"
            f"dòng mẫu: {json.dumps(smp, ensure_ascii=False, default=str)[:1800]}")

def schema_text(M):
    P = pd.DataFrame(list(M.products.values())).drop(columns=["names"], errors="ignore") if M.products else None
    parts = [
        _card('D["orders"]', M.orders, "một dòng một đơn. status ∈ {done,cancel,return,shipping}; revenue = doanh thu đơn (§1); date = ngày đặt; ts = giờ tạo"),
        _card('D["lines"]', M.lines, "một dòng một dòng hàng. amount = thành tiền sau giảm dòng; qty_base = SL quy về đơn vị cơ bản; nonitem/is_service_fee = dòng không phải mặt hàng"),
        _card('D["products"]', P, "danh mục hàng; key = mã hàng hoặc 'N:'+tên"),
        _card('D["imports"]', M.imports, "dòng phiếu nhập; cancelled = phiếu huỷ"),
        _card('D["payments"]', M.payments, "mỗi lần khách trả tiền cho một đơn"),
        _card('D["expenses"]', M.expenses, ""), _card('D["deliveries"]', M.deliveries, ""),
        _card('D["lots"]', M.lots, ""), _card('D["lot_sales"]', M.lot_sales, ""),
    ]
    for i, (n, (df, roles, kind)) in enumerate(list(M.tables.items())[:30], 1):
        parts.append(_card(f'D["raw"]["{n}"]  (SQL: raw_{i})', df, f"bảng gốc (đoán loại: {kind})"))
    return "\n\n".join(p for p in parts if p)

def data_dict(M):
    return {"orders": M.orders, "lines": M.lines,
            "products": pd.DataFrame(list(M.products.values())) if M.products else pd.DataFrame(),
            "imports": M.imports, "payments": M.payments, "expenses": M.expenses, "deliveries": M.deliveries,
            "lots": M.lots, "lot_sales": M.lot_sales,
            "raw": {n: df for n, (df, roles, kind) in M.tables.items()}}

def gen_prompt(M, rulebook, readme, question, as_of, periods, lang="python"):
    """Order matters for vLLM prefix caching: everything that is identical for all questions of a shop
    (rulebook, README, schema, instructions) comes first; the question comes last."""
    win = "; ".join(f"{p.kind}: {p.start.date()} → {p.end.date()}" for p in periods) or "(không nhận diện được)"
    instr = GEN_INSTR_SQL if lang == "sql" else GEN_INSTR
    return (f"## Rulebook\n{rulebook}\n\n## README của shop\n{readme}\n\n## Cấu trúc bảng\n{_schema_cached(M)}\n\n"
            f"## Nhiệm vụ\n{instr}\n\n## Câu hỏi\nHôm nay = {as_of}. Mốc thời gian gợi ý: {win}\n{question}")

def _schema_cached(M):
    if getattr(M, "_schema_txt", None) is None:
        M._schema_txt = schema_text(M)
    return M._schema_txt

def to_sqlite(D):
    """load the canonical + raw tables into an in-memory SQLite database (for SQL candidates)"""
    import sqlite3
    con = sqlite3.connect(":memory:")
    def put(name, df):
        if df is None or not hasattr(df, "columns") or len(df) == 0:
            return
        x = df.copy()
        x.columns = [str(c) for c in x.columns]
        for c in x.columns:
            if str(x[c].dtype).startswith("datetime"):
                x[c] = x[c].dt.strftime("%Y-%m-%d %H:%M:%S")
            elif x[c].dtype == object:
                x[c] = x[c].map(lambda v: v if v is None or isinstance(v, (str, int, float)) else str(v))
        x.to_sql(name, con, index=False)
    for k, v in D.items():
        if k == "raw":
            for i, (n, df) in enumerate(v.items(), 1):
                put(f"raw_{i}", df)
        else:
            put(k, v)
    return con

def extract_sql(text):
    m = re.findall(r"```sql\s*(.*?)```", text, flags=re.S)
    t = re.search(r"TYPE:\s*(\w+)", text)
    return (m[-1] if m else None), (t.group(1).lower() if t else None)

def sql_to_answer(con, sql, typ):
    cur = con.execute(sql)
    rows = cur.fetchall(); cols = [d[0] for d in cur.description or []]
    if typ == "unanswerable":
        return {"type": "unanswerable", "answer": None, "n_rows": 0}
    if not rows or rows[0][0] is None:
        return {"type": typ, "answer": None, "n_rows": 0}          # empty selection -> abstain
    if typ in ("number", "percent", "date", "string", "boolean"):
        return {"type": typ, "answer": rows[0][0], "n_rows": len(rows)}
    if typ == "list":
        return {"type": typ, "answer": [r[0] for r in rows], "n_rows": len(rows)}
    if typ == "table":
        return {"type": typ, "answer": {"columns": cols, "rows": [list(r) for r in rows]}, "n_rows": len(rows)}
    raise ValueError("unknown TYPE " + str(typ))

def extract_code(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    return m[-1] if m else text

# ------------------------------------------------------------------ sandbox
def _child(code, D, q):
    if code.startswith("--SQL--"):
        try:
            _, typ, sql = code.split("\n", 2)
            con = to_sqlite(D)
            q.put(("ok", json.loads(json.dumps(sql_to_answer(con, sql, typ), default=str))))
        except Exception:
            q.put(("err", traceback.format_exc()[-1500:]))
        return
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
        p.kill(); return "err", "timeout"
    try:
        return q.get_nowait()
    except Exception:
        return "err", "no result"

def canon(r):
    if not isinstance(r, dict) or r.get("type") not in ("number", "percent", "date", "boolean", "string", "list", "table", "unanswerable"):
        return None
    t, a = r["type"], r.get("answer")
    try:
        if t == "number":
            a = float(a); a = int(round(a)) if abs(a - round(a)) < 1e-6 else round(a, 2)
        elif t == "percent":
            a = round(float(a), 2)
        elif t == "list":
            a = [(int(x) if float(x).is_integer() else round(float(x), 2)) if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x) for x in a]
            if not a:
                return None
        elif t == "table":
            a = {"columns": [str(c) for c in a["columns"]], "rows": sorted([list(x) for x in a["rows"]], key=str)}
            if not a["rows"]:
                return None
        elif t in ("string", "date"):
            a = str(a).strip()
            if not a or a.lower() in ("nan", "none"):
                return None
        elif t == "boolean":
            a = bool(a)
        elif t == "unanswerable":
            a = None
    except Exception:
        return None
    if t not in ("unanswerable",) and int(r.get("n_rows") or 0) <= 0:
        return "EMPTY"            # computed from an empty selection -> counts as a failed candidate
    return {"type": t, "answer": a}

# ------------------------------------------------------------------ LLM worker processes
def llm_generate(model_path, jobs, tp=2, thinking=False, max_model_len=None, timeout_s=None):
    """Run text generation in a separate process (frees all GPU memory when it exits, so two different
    models -- e.g. a code model and a verifier -- never have to share the GPUs).
    jobs: [{"id", "messages", "n", "temperature", "seed", "max_tokens"}] -> {id: [texts]}"""
    import subprocess, tempfile, os, sys, time
    if not jobs:
        return {}
    max_model_len = max_model_len or int(os.environ.get("DM_MAX_LEN", "32768"))
    d = tempfile.mkdtemp(prefix="dm_llm_")
    inp, out = os.path.join(d, "in.jsonl"), os.path.join(d, "out.jsonl")
    with open(inp, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    cmd = [sys.executable, "-m", "datamate.agent_worker", "--model", model_path, "--tp", str(tp), "--in", inp, "--out", out,
           "--max-model-len", str(max_model_len)] + (["--thinking"] if thinking else []) + \
          (["--quantization", os.environ["DM_QUANT"]] if os.environ.get("DM_QUANT") else []) + \
          ["--gpu-util", os.environ.get("DM_GPU_UTIL", "0.88"), "--dtype", os.environ.get("DM_DTYPE", "auto")] + \
          (["--enforce-eager"] if os.environ.get("DM_EAGER") == "1" else [])
    # each attempt is a fresh process; later attempts use safer settings. The worker's full log is kept on disk
    # (DM_WORKER_LOG_DIR) and its tail is printed on failure, so errors are visible even when stdout is filtered.
    flip = {"VLLM_USE_V1": "1" if os.environ.get("VLLM_USE_V1") == "0" else "0"}
    attempts = [({}, []), ({}, ["--no-prefix-cache"]), (flip, ["--no-prefix-cache", "--enforce-eager"])]
    log_dir = os.environ.get("DM_WORKER_LOG_DIR") or d
    os.makedirs(log_dir, exist_ok=True)
    for i, (env_over, extra) in enumerate(attempts):
        log = os.path.join(log_dir, f"worker_{os.path.basename(model_path.rstrip('/'))}_{int(time.time())}_try{i}.log")
        t0 = time.time()
        try:
            with open(log, "w") as lf:
                p = subprocess.run(cmd + extra, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout_s,
                                   env={**os.environ, **env_over}, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            rc = p.returncode
        except Exception as e:
            rc = repr(e)
        if rc == 0 and os.path.exists(out):
            print(f"[llm] {os.path.basename(model_path.rstrip('/'))}: {len(jobs)} jobs ok in {time.time()-t0:.0f}s (try {i})", flush=True)
            break
        tail = open(log, encoding="utf-8", errors="replace").read()[-3500:] if os.path.exists(log) else ""
        print(f"[llm] WORKER FAILED (exit {rc}, try {i}, extra={extra} {env_over}); log: {log}\n----- last lines -----\n{tail}\n-----", flush=True)
    res = {}
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            r = json.loads(l); res[r["id"]] = r["texts"]
    return res

# ------------------------------------------------------------------ agent orchestration
class Agent:
    """gen_model writes pandas programs; ver_model grades the winning program (may be the same model).
    Default pairing within the 15B budget: Qwen2.5-Coder-7B-Instruct (7.6B) + Qwen3-4B (4.0B) = 11.6B,
    or a single Qwen3-14B (14.8B). Choose on the sealed validation split."""
    def __init__(self, gen_model, ver_model=None, tp=2, k=5, max_repairs=2, max_tokens=1500, min_share=0.6, tau=0.55,
                 thinking=False, seed=1234, q_budget_s=40.0, exec_timeout_s=10.0, generate=None, langs=None):
        self.gen_model, self.ver_model = gen_model, ver_model or gen_model
        self.tp, self.k, self.max_repairs, self.max_tokens = tp, k, max_repairs, max_tokens
        self.min_share, self.tau, self.thinking, self.seed = min_share, tau, thinking, seed
        self.q_budget_s, self.exec_timeout_s = q_budget_s, exec_timeout_s
        self.generate = generate or (lambda model, jobs: llm_generate(model, jobs, tp=self.tp, thinking=self.thinking))
        # candidate mix for self-consistency: pandas programs and SQL queries vote together
        self.langs = langs or {"python": 3, "sql": 2}
        self.k = sum(self.langs.values())

    def solve_all(self, items, deadline=None):
        """items: dict(key, M, rulebook, readme, question, as_of, periods, D) -> {key: result}
        result = {type, answer, confidence} or {type: 'abstain', reason}"""
        import time
        now = time.time
        spent = {it["key"]: 0.0 for it in items}            # per-question sandbox time
        byk = {it["key"]: it for it in items}
        prompts = {(it["key"], lang): gen_prompt(it["M"], it["rulebook"], it["readme"], it["question"], it["as_of"], it["periods"], lang)
                   for it in items for lang, n in self.langs.items() if n > 0}
        if self.langs.get("sql"):
            for it in items:
                if "_sqlite" not in it["D"]:
                    try:
                        it["D"]["_sqlite"] = to_sqlite(it["D"])
                    except Exception:
                        it["D"]["_sqlite"] = None
        def out_of_time():
            return deadline is not None and now() > deadline
        # round 0: independent samples per question and language (temperature > 0 -> self-consistency)
        jobs = [{"id": f"{k}|{lang}", "messages": [{"role": "user", "content": prompts[(k, lang)]}], "n": self.langs[lang],
                 "temperature": 0.7, "seed": self.seed + i, "max_tokens": self.max_tokens} for i, (k, lang) in enumerate(prompts)]
        gen = self.generate(self.gen_model, jobs) if not out_of_time() else {}
        cand = {it["key"]: [] for it in items}
        pending = []
        def to_code(text, lang):
            if lang == "sql":
                sql, typ = extract_sql(text)
                return f"--SQL--\n{typ}\n{sql}" if sql and typ else "raise ValueError('no sql block or TYPE line')"
            return extract_code(text)
        def run(k, text, lang):
            code = to_code(text, lang)
            if spent[k] > self.q_budget_s:
                return None, code, "question time budget exhausted"
            t0 = now()
            st, r = run_program(code, byk[k]["D"], timeout=self.exec_timeout_s)
            spent[k] += now() - t0
            if st == "ok":
                return canon(r), code, None
            return None, code, r
        for jid, texts in gen.items():
            k, lang = jid.rsplit("|", 1)
            for t in texts:
                a, code, err = run(k, t, lang)
                cand[k].append((a, code))
                if err and "budget" not in str(err):
                    pending.append((k, [{"role": "user", "content": prompts[(k, lang)]}], t, err, 1, lang))
        # traceback feedback loop: up to max_repairs attempts per failed program
        for attempt in range(1, self.max_repairs + 1):
            pending = [p for p in pending if spent[p[0]] <= self.q_budget_s]
            if not pending or out_of_time():
                break
            jobs = []
            for j, (k, conv, last, err, at, lang) in enumerate(pending):
                conv2 = conv + [{"role": "assistant", "content": last},
                                {"role": "user", "content": f"Chạy lỗi (lần {at}). Traceback:\n{err}\n" + ("Hãy sửa và trả lại khối ```sql``` cùng dòng TYPE." if lang == "sql" else "Hãy sửa lỗi và trả về toàn bộ hàm solve(D).")}]
                jobs.append({"id": f"{k}#r{attempt}#{j}", "messages": conv2, "n": 1, "temperature": 0.0, "seed": self.seed, "max_tokens": self.max_tokens})
            rep = self.generate(self.gen_model, jobs)
            nxt = []
            for job, (k, conv, last, err, at, lang) in zip(jobs, pending):
                texts = rep.get(job["id"]) or []
                if not texts:
                    continue
                a, code, err2 = run(k, texts[0], lang)
                cand[k].append((a, code))
                if err2 and "budget" not in str(err2):
                    nxt.append((k, job["messages"], texts[0], err2, at + 1, lang))
            pending = nxt
        # vote
        results, to_verify = {}, []
        for k, cs in cand.items():
            good = [(a, code) for a, code in cs if isinstance(a, dict)]
            if not good:
                results[k] = {"type": "abstain", "reason": "empty selection" if any(a == "EMPTY" for a, _ in cs) else
                              ("time budget" if spent[k] > self.q_budget_s else "all programs failed")}
                continue
            keys = [json.dumps(a, ensure_ascii=False, sort_keys=True) for a, _ in good]
            best, n = Counter(keys).most_common(1)[0]
            share = min(1.0, n / max(self.k, 1))
            win = json.loads(best)
            if share < self.min_share:
                results[k] = {"type": "abstain", "reason": f"low agreement {share:.2f}"}; continue
            if win["type"] == "unanswerable" and n < len(good):
                results[k] = {"type": "abstain", "reason": "split unanswerable vote"}; continue
            code = next(c for a, c in good if json.dumps(a, ensure_ascii=False, sort_keys=True) == best)
            results[k] = dict(win, share=round(share, 3)); to_verify.append((k, code))
        # verifier model grades the winning program -> confidence
        if to_verify and not out_of_time():
            jobs = []
            for k, code in to_verify:
                it = byk[k]
                body = (f"## Rulebook\n{it['rulebook']}\n\n{VERIFY_INSTR}\n\n## Câu hỏi (hôm nay {it['as_of']})\n{it['question']}\n\n"
                        f"## Code\n```python\n{code}\n```\n\n## Kết quả\n{json.dumps({x: results[k][x] for x in ('type', 'answer')}, ensure_ascii=False)}")
                jobs.append({"id": k, "messages": [{"role": "user", "content": body}], "n": 1, "temperature": 0.0, "seed": self.seed, "max_tokens": 32})
            ver = self.generate(self.ver_model, jobs)
            for k, code in to_verify:
                txt = (ver.get(k) or [""])[0]
                m = re.search(r"CONFIDENCE:\s*(\d{1,3})", txt)
                vc = min(int(m.group(1)), 100) / 100 if m else 0.0
                conf = results[k]["share"] * vc
                results[k]["confidence"] = round(conf, 3)
                if conf < self.tau:
                    results[k] = {"type": "abstain", "reason": f"low confidence {conf:.2f}", "confidence": round(conf, 3)}
        else:
            for k, _ in to_verify:
                results[k] = {"type": "abstain", "reason": "no verification (deadline)"}
        return results


# ------------------------------------------------------------------ LLM-assisted normalisation of unseen status labels
LABEL_INSTR = """Phân loại từng nhãn trạng thái đơn hàng (tiếng Việt, có thể viết tắt) vào đúng một nhóm:
done (hoàn thành/đã giao/đã lấy), cancel (huỷ, khách bỏ), return (trả hàng/hoàn), shipping (đang giao/chờ giao/đang xử lý).
Trả về JSON object {"<nhãn>": "<nhóm>"} cho mọi nhãn, không giải thích."""

def classify_status_labels(generate, model, labels, samples=3, seed=7):
    """self-consistency: a label is mapped only when all samples agree"""
    if not labels:
        return {}
    body = LABEL_INSTR + "\nNhãn: " + json.dumps(labels, ensure_ascii=False)
    out = generate(model, [{"id": "labels", "messages": [{"role": "user", "content": body}], "n": samples,
                            "temperature": 0.7, "seed": seed, "max_tokens": 400}]).get("labels", [])
    votes = []
    for t in out:
        m = re.search(r"\{.*\}", t, flags=re.S)
        try:
            votes.append(json.loads(m.group(0)) if m else {})
        except Exception:
            votes.append({})
    res = {}
    for lab in labels:
        vs = [v.get(lab) for v in votes]
        if len(votes) == samples and len(set(vs)) == 1 and vs[0] in ("done", "cancel", "return", "shipping"):
            res[lab] = vs[0]
    return res
