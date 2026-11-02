"""python validate_submission.py questions.jsonl submission.jsonl"""
import sys, json, re
qs = [json.loads(l)["id"] for l in open(sys.argv[1], encoding="utf-8")]
seen, errs = set(), []
for i, l in enumerate(open(sys.argv[2], encoding="utf-8")):
    r = json.loads(l); t = r.get("type"); a = r.get("answer")
    if r.get("id") in seen: errs.append((i, "dup id"))
    seen.add(r.get("id"))
    ok = {
        "number": isinstance(a, (int, float)) and not isinstance(a, bool),
        "percent": isinstance(a, (int, float)) and not isinstance(a, bool),
        "date": isinstance(a, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a or ""),
        "boolean": isinstance(a, bool),
        "string": isinstance(a, str) and a.strip() != "",
        "list": isinstance(a, list) and len(a) > 0,
        "table": isinstance(a, dict) and isinstance(a.get("columns"), list) and isinstance(a.get("rows"), list),
        "unanswerable": "reason" not in r or isinstance(r["reason"], str),
        "abstain": True,
    }.get(t, False)
    if not ok: errs.append((i, t, a))
missing = set(qs) - seen
print("lines", len(seen), "missing", len(missing), "errors", len(errs))
for e in errs[:20]: print(e)
