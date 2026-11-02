"""Local re-implementation of the official scorer (Evaluation.docx)."""
import math
from .norm import eval_norm

def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6 * max(1, abs(float(b)))
    except Exception:
        return False

def _str_eq(a, ref, aliases):
    if not isinstance(a, str):
        return False
    forms = [ref] + list(aliases or [])
    return eval_norm(a) in {eval_norm(f) for f in forms if isinstance(f, str)}

def _elem_eq(a, ref, aliases):
    if isinstance(ref, (int, float)) and not isinstance(ref, bool):
        return isinstance(a, (int, float)) and not isinstance(a, bool) and _num_eq(a, ref)
    return _str_eq(a, ref, aliases)

def score(pred, gold):
    """pred: dict(type, answer) ; gold: dict(type, answer, alias) -> 1 / 0 / -1"""
    if pred is None or pred.get("type") in (None, "abstain"):
        return 0
    t = pred["type"]
    if t != gold["type"]:
        return -1
    if t == "unanswerable":
        return 1
    a, g, al = pred.get("answer"), gold["answer"], gold.get("alias") or []
    ok = False
    if t == "number":
        ok = isinstance(a, (int, float)) and not isinstance(a, bool) and _num_eq(a, g)
    elif t == "percent":
        ok = isinstance(a, (int, float)) and not isinstance(a, bool) and round(float(a), 2) == round(float(g), 2)
    elif t in ("date", "boolean"):
        ok = a == g
    elif t == "string":
        ok = _str_eq(a, g, al)
    elif t == "list":
        ok = isinstance(a, list) and len(a) == len(g) and all(
            _elem_eq(x, y, al[i] if i < len(al) and isinstance(al[i], list) else []) for i, (x, y) in enumerate(zip(a, g)))
    elif t == "table":
        try:
            gc, gr = g["columns"], g["rows"]; ac, ar = a["columns"], a["rows"]
            if len(ar) != len(gr):
                ok = False
            else:
                used = [False] * len(ar)
                ok = True
                for i, row in enumerate(gr):
                    als = al[i] if i < len(al) else []
                    found = False
                    for j, prow in enumerate(ar):
                        if used[j] or len(prow) != len(row):
                            continue
                        if _str_eq(prow[0], row[0], als) and all(_num_eq(x, y) for x, y in zip(prow[1:], row[1:])):
                            used[j] = found = True; break
                    if not found:
                        ok = False; break
        except Exception:
            ok = False
    return 1 if ok else -1
