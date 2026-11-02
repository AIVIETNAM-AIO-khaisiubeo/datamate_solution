"""Intent handlers. Each handler returns (type, answer) or raises Unanswerable/Abstain.
Matchers work on the normalised question text c.s"""
import re
import numpy as np
import pandas as pd
from .engine import *
from .engine import _item_index
from .norm import nkey

K_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}

def get_k(c, default=None):
    m = re.search(r"\btop\s*(\d+|mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\b", c.s) or \
        re.search(r"(?<![/\d.,-])\b(\d{1,2}|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\s*(?:ban chay|hot nhat|chay nhat)", c.s) or \
        re.search(r"(?<![/\d.,-])\b(\d{1,2}|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\s*(?:mat hang|san pham|mon|hang|khach|nguoi|nha cung cap|loai|sp|thuoc|nhan vien)\b", c.s)
    if m:
        v = m.group(1)
        return int(v) if v.isdigit() else K_WORDS[v]
    return default

def threshold(c):
    """'tu 55.000 dong tro len' / 'tu 10 cu tro len' / 'tren 3 don' -> (value, inclusive)"""
    m = re.search(r"(?:tu|it nhat|toi thieu|>=|≥|dat)\s*([\d.,]+)\s*(trieu|tr|cu|k|nghin|ngan)?\s*(?:dong|d|vnd)?\s*(?:tro len|tro di|tro len|\b)", c.s)
    if m:
        v = _amt(m.group(1), m.group(2))
        incl = True
        return v, incl
    m = re.search(r"(?:tren|hon|qua|vuot)\s*([\d.,]+)\s*(trieu|tr|cu|k|nghin|ngan)?", c.s)
    if m:
        return _amt(m.group(1), m.group(2)), False
    return None, None

def _amt(v, u):
    if u in ("trieu", "tr", "cu"):
        return float(v.replace(".", "").replace(",", ".")) * 1e6 if "," in v else float(v) * 1e6 if v.count(".") <= 1 and len(v.split(".")[-1]) != 3 else parse_money(v) * 1e6
    if u in ("k", "nghin", "ngan"):
        return float(v.replace(",", ".")) * 1e3
    return parse_money(v)

def main_period(c, idx=0):
    ps = [p for p in c.periods if p.kind != "snapshot"]
    if len(ps) <= idx:
        if idx == 0 and any(p.kind == "snapshot" for p in c.periods):
            return [p for p in c.periods if p.kind == "snapshot"][0]
        raise Abstain("no period")
    return ps[idx]

def months_listed(c):
    """months mentioned explicitly in order; expands month ranges and 'N thang lien nhau ket thuc o thang X'"""
    ms = []
    for p in c.periods:
        if p.kind == "monthrange":
            ms += p.meta["months"]
        elif p.kind == "month":
            ms.append((p.start.year, p.start.month))
        elif p.kind == "quarter":
            pass
    m = re.search(r"(\d+|hai|ba|bon|nam|sau)\s*thang\s*(?:lien nhau|lien tiep)?\s*(?:ket thuc|cuoi cung)\s*(?:o|vao|la|tai)?\s*thang", c.s)
    if m and ms:
        n = int(m.group(1)) if m.group(1).isdigit() else K_WORDS[m.group(1)]
        y, mo = ms[-1]
        out = []
        for i in range(n - 1, -1, -1):
            yy, mm = y, mo - i
            while mm <= 0:
                mm += 12; yy -= 1
            out.append((yy, mm))
        return out
    m = re.search(r"(\d+|hai|ba|bon|nam|sau)\s*thang\s*(?:lien nhau|lien tiep|gan nhat|vua qua)?\s*(?:bat dau|tu)\s*(?:o|vao|la|tai)?\s*thang", c.s)
    if m and ms:
        n = int(m.group(1)) if m.group(1).isdigit() else K_WORDS[m.group(1)]
        y, mo = ms[0]
        out = []
        for i in range(n):
            yy, mm = y, mo + i
            while mm > 12:
                mm -= 12; yy += 1
            out.append((yy, mm))
        return out
    seen = []
    for x in ms:
        if x not in seen:
            seen.append(x)
    return seen

def mp(c, ym):
    return month_period(ym[0], ym[1], c.today)

# ============================================================== handlers
def h_subjective(c):
    if c.has(SUBJECTIVE):
        raise Unanswerable("subjective / not in data")
    if c.has(r"\bnhom (hang |mon )[a-z]") and not item_groups(c.M) and not getattr(c.M, "item_type", None) and c.has(r"doanh thu|ban duoc|chiem|ti trong|tien"):
        raise Abstain("shop has no item groups")
    if c.has(r"(khach|nha)\s*(hang\s*)?(o\s*|tai\s*)?(duong|pho)\s+[a-z]") and not any(d.get("address") for d in c.M.customers.values()):
        raise Unanswerable("no address")
    return None

def is_revenue_q(c):
    return c.has(r"doanh thu|ban duoc .*tien|ban duoc bao nhieu tien|thu ve|tien hang|tien ban|ban (ra|duoc)? ?(tong cong|tat ca)? ?(duoc)? ?bao nhieu tien|mua het bao nhieu tien|mua .*bao nhieu tien|chi .*tien|so tien mua|nhieu tien|bao nhieu tien|ban duoc .*tien|thu duoc bao nhieu|dem ve|mang ve|chot duoc bao nhieu tien|tien thu|lam ra")

def h_trend_bool(c):
    if not c.has(r"tang lien tuc|giam lien tuc|luon (cao|thap|tang|giam|hon|it)|tang deu|giam deu|co phai .*cao hon .*va|lien tuc tang|lien tuc giam"):
        return None
    ms = months_listed(c)
    if len(ms) < 2:
        raise Abstain("months")
    vals = [filtered_revenue(c, mp(c, ym), collect_filters(c, False)) for ym in ms]
    dec = c.has(r"giam lien tuc|luon (thap|giam|it)|giam deu|lien tuc giam")
    ok = all((b < a) if dec else (b > a) for a, b in zip(vals, vals[1:]))
    return "boolean", bool(ok)

def h_best_month(c):
    if not c.has(r"thang nao") or c.has(r"\bngay nao\b"):
        return None
    ms = months_listed(c)
    if len(ms) < 2:
        p = [p for p in c.periods if p.kind in ("quarter", "year", "ytd", "monthrange")]
        if not p:
            raise Abstain("months")
        a, b = clip(c, p[0])
        ms = _months_between(a.year, a.month, b.year, b.month)
    f = collect_filters(c, True)
    vals = []
    for ym in ms:
        try:
            vals.append((filtered_revenue(c, mp(c, ym), f), ym))
        except Unanswerable:
            vals.append((None, ym))
    if all(v is None for v, _ in vals):
        raise Unanswerable("none")
    vals = [(v, ym) for v, ym in vals if v is not None]
    low = c.has(r"(thap|it|kem) nhat")
    v, ym = (min if low else max)(vals, key=lambda t: (t[0], -(t[1][0] * 12 + t[1][1])) if not low else (t[0], t[1][0] * 12 + t[1][1]))
    if not low:
        best = max(v for v, _ in vals)
        ym = sorted([ym for v, ym in vals if v == best])[0]
    else:
        best = min(v for v, _ in vals)
        ym = sorted([ym for v, ym in vals if v == best])[0]
    return "date", f"{ym[0]}-{ym[1]:02d}"

def h_revenue_list(c):
    ms = months_listed(c)
    if len(ms) < 2 or not is_revenue_q(c):
        return None
    if c.has(r"phan tram|thay doi|tang hay giam|so voi|so vs|nao|cao nhat|thap nhat"):
        return None
    f = collect_filters(c, True)
    return "list", [rmoney(filtered_revenue(c, mp(c, ym), f)) for ym in ms]

def _compare_period(c, p):
    """the period to compare p against, from the question"""
    s = c.s
    if re.search(r"cung ky nam (ngoai|truoc)|cung thang nam (ngoai|truoc)|nam ngoai|so voi .*nam truoc", s) and p.kind in ("month", "mtd", "quarter", "week", "day"):
        a = p.start - pd.DateOffset(years=1); b = p.end - pd.DateOffset(years=1)
        return Period(a, b, p.kind)
    if p.kind == "mtd" and (c.has(r"cung (ky|so ngay)|thang truoc|thang roi")):
        pm_end = p.start - pd.Timedelta(days=1)
        a = pm_end.replace(day=1)
        b = min(a + pd.Timedelta(days=p.end.day - 1), pm_end)
        return Period(a, b, "range")
    if p.kind == "month":
        return prev_month_period(p, c.today)
    if p.kind == "week":
        return Period(p.start - pd.Timedelta(days=7), p.start - pd.Timedelta(days=7) + (p.end - p.start), "week")
    if p.kind == "quarter":
        a = p.start - pd.DateOffset(months=3)
        return Period(a, month_end(a.year, a.month + 2), "quarter")
    if p.kind == "day":
        d = p.start - pd.Timedelta(days=1)
        return Period(d, d, "day")
    raise Abstain("compare period")

def h_growth(c):
    if not c.has(r"phan tram|%") or not c.has(r"so voi|so vs|so thang|so cung|thay doi|tang hay giam|tang giam|hon hay kem|cao hay thap|len hay xuong|tang truong|tang bao nhieu|giam bao nhieu|so sanh|chenh"):
        return None
    if c.has(r"ca chieu.*ca sang|ca sang.*ca chieu"):
        return h_shift_compare(c)
    if c.has(r"\b(tang|doi|len) gia\b"):
        return None
    if c.has(r"\btet\b"):
        return h_tet(c)
    ps = growth_periods(c)
    if not ps:
        raise Abstain("no period")
    f = collect_filters(c, True)
    metric = "orders" if c.has(r"(so|bao nhieu|may) don") and not c.has(r"doanh thu") else "revenue"
    def val(p):
        if metric == "orders":
            return float(len(apply_order_filters(c, orders_in(c, p), f)))
        return filtered_revenue(c, p, f)
    if len(ps) >= 2:
        p1, p2 = ps[0], ps[1]
        # "T5/2026 so voi T5/2025": first is asked, second is base
        return "percent", pct(val(p1), val(p2))
    p = ps[0]
    q = _compare_period(c, p)
    return "percent", pct(val(p), val(q))

def growth_periods(c):
    ps = [p for p in c.periods if p.kind != "snapshot"]
    # a single day that lies inside another mentioned period is just its end point ('thang nay tinh toi hom nay')
    ps = [p for p in ps if not (p.kind == "day" and any(q is not p and q.kind != "day" and q.start <= p.start <= q.end for q in ps))]
    explicit = [p for p in ps if not p.meta.get("relative") and p.kind not in ("year",) ]
    s = c.s
    rel_cmp = re.search(r"(cung ky|cung so ngay|cung thang|so voi|so vs|so|hon hay kem)[^0-9]{0,25}(thang|tuan|nam) (truoc|roi|lien truoc|ngoai)", s)
    if rel_cmp:
        # comparison base is relative to the asked period -> keep only the asked period(s)
        ps = [p for p in ps if not p.meta.get("relative") and not (p.kind == "year" and re.search(r"nam (truoc|ngoai|roi)", s))]
        if not ps:
            ps = [p for p in c.periods if p.kind != "snapshot"][:1]
    return ps

def h_tet(c):
    tp = [p for p in c.periods if p.kind == "tet"]
    if not tp:
        raise Abstain("tet")
    p = tp[0]
    n = p.days()
    m = re.search(r"(\d+)\s*ngay\s*(?:ngay\s*)?truoc\s*tet", c.s)
    if m:
        n2 = int(m.group(1))
    else:
        n2 = n
    q = Period(p.start - pd.Timedelta(days=n2), p.start - pd.Timedelta(days=1), "range")
    f = collect_filters(c, True)
    a = filtered_revenue(c, p, f); b = filtered_revenue(c, q, f)
    if c.has(r"phan tram|%"):
        return "percent", pct(a, b)
    return "number", rmoney(a - b)

def h_shift_compare(c):
    p = main_period(c)
    f = collect_filters(c, True); f.pop("shift", None)
    fa = dict(f, shift="chieu"); fb = dict(f, shift="sang")
    if c.has(r"(so|bao nhieu|may) don") and not c.has("doanh thu|tien"):
        a = len(apply_order_filters(c, orders_in(c, p), fa)); b = len(apply_order_filters(c, orders_in(c, p), fb))
    else:
        a = filtered_revenue(c, p, fa); b = filtered_revenue(c, p, fb)
    if c.has(r"phan tram|%"):
        return "percent", pct(a, b)
    return "number", rmoney(a - b)

def h_cancel(c):
    if not c.has(r"\bhuy\b"):
        return None
    p = main_period(c)
    o = orders_in(c, p, valid=False)
    if c.has(r"ti le|phan tram|%|ty le"):
        if len(o) == 0:
            raise Abstain("no orders")
        return "percent", r2((o["status"] == "cancel").sum() / len(o) * 100)
    if c.has(r"khach nao|\bai\b .*huy|khach .*huy nhieu"):
        return _top_customers_by(c, o[o["status"] == "cancel"], "count")
    return "number", int((o["status"] == "cancel").sum())

def h_avg_order(c):
    if not c.has(r"trung binh|binh quan|gia tri trung binh|tb ") or not c.has(r"\bdon\b|bill|hoa don|phieu|order"):
        return None
    if c.has(r"moi ngay|1 ngay|mot ngay"):
        return None
    p = main_period(c)
    f = collect_filters(c, True)
    o = apply_order_filters(c, orders_in(c, p), f)
    if len(o) == 0:
        raise Abstain("no orders")
    if c.has(r"khach phai tra|phai tra"):
        if "payable" not in o:
            raise Unanswerable("no payable")
        return "number", rmoney(o["payable"].sum() / len(o))
    return "number", rmoney(revenue(o) / len(o))

def h_orders_threshold(c):
    if not c.has(r"\bdon\b|\bbill\b|hoa don") or not c.has(r"tro len|tren|it nhat|>=|≥|hon|vuot|qua"):
        return None
    if c.has(r"khach|mat hang|mon|san pham|ngay nao|ngay"):
        return None
    v, incl = threshold(c)
    if v is None:
        return None
    p = main_period(c)
    o = orders_in(c, p)
    r = o["revenue"].fillna(0)
    return "number", int((r >= v).sum() if incl else (r > v).sum())

def h_days_threshold(c):
    if not c.has(r"(bao nhieu|may|so|dem|nhieu) (ngay|bua)") or not c.has(r"tro len|tren|it nhat|hon|vuot|>=|≥|dat"):
        return None
    v, incl = threshold(c)
    if v is None:
        return None
    p = main_period(c)
    o = orders_in(c, p)
    if c.has(r"\bdon\b") and not c.has(r"doanh thu|tien|dong|trieu|cu\b"):
        d = o.groupby("date").size()
    else:
        d = o.groupby("date")["revenue"].sum()
    return "number", int((d >= v).sum() if incl else (d > v).sum())

def h_best_day(c):
    if not c.has(r"ngay nao|bua nao|ngay (co )?(doanh thu|ban|thu)[a-z ]* nhat|ngay (dong|dat) khach nhat") or c.has(r"thang nao|gan nhat"):
        return None
    p = main_period(c)
    o = orders_in(c, p)
    if len(o) == 0:
        raise Unanswerable("no orders")
    if c.has(r"nhieu don|dong khach|dat khach|nhieu khach|nhieu bill|nhieu hoa don"):
        d = o.groupby("date").size()
    else:
        d = o.groupby("date")["revenue"].sum()
    low = c.has(r"(it|thap|kem) nhat")
    best = d.min() if low else d.max()
    day = sorted(d[d == best].index)[0]
    return "date", day.strftime("%Y-%m-%d")


def _rank(series, k, ascending=False, key_order=None):
    """series indexed by key -> top-k keys, ties by key asc"""
    df = series.reset_index(); df.columns = ["key", "val"]
    df["ord"] = df["key"].map(key_order) if key_order else df["key"].astype(str)
    df = df.sort_values(["val", "ord"], ascending=[ascending, True])
    return list(df["key"].head(k)), list(df["val"].head(k))

def item_order(M):
    def f(k):
        p = M.products.get(k)
        if p and p.get("code"):
            return "0" + str(p["code"])
        if isinstance(k, str) and not k.startswith("N:"):
            return "0" + k
        return "1" + str(display_name(M, k))
    return f

def h_top_items(c):
    if not c.has(r"(ban chay|ban nhieu|ban duoc nhieu|hot nhat|chay nhat|ban it|ban cham|ua chuong|loi nhat|lai nhieu nhat|lai gop nhieu nhat|lai cao nhat|loi nhieu nhat|nhieu tien nhat|doanh thu cao nhat|it nhat|nhieu nhat|(lai gop|loi nhuan gop|loi gop).*(cao|lon|nhieu) nhat)") \
            or not c.has(r"(mat hang|san pham|mon|hang|sp|item|loai|thuoc|chai|lon|ly|goi|hop|cai|nao|gi|nhung)"):
        return None
    if not re.search(r"nhat|\btop\b|\d+ (mat hang|mon|san pham|sp)|liet ke", c.s.replace("gan nhat", "")):
        return None
    if c.has(r"khach|nhan vien|tho\b|nha cung cap|ncc|ngay nao|thang nao|kenh|gio"):
        return None
    p = main_period(c)
    o = orders_in(c, p)
    f = collect_filters(c, True)
    o = apply_order_filters(c, o, f)
    L = lines_of(c, o)
    f2 = {k: v for k, v in f.items() if k in ("group", "rx", "ingredient")}
    L = apply_line_filters(c, L, f2)
    if c.has(r"loi nhat|lai nhieu nhat|lai gop|lai cao nhat|loi nhieu nhat|lai nhat|loi nhuan|loi gop"):
        if L["cost_unit"].isna().all():
            raise Unanswerable("no cost")
        val = (L["amount"] - L["qty"] * L["cost_unit"]).groupby(L["item_key"]).sum()
    elif c.has(r"nhieu tien|doanh thu|thu ve nhieu"):
        val = L.groupby("item_key")["amount"].sum()
    elif c.has(r"ua chuong|nhieu don nhat|co mat trong nhieu don"):
        val = L.groupby("item_key")["order_id"].nunique()
    else:
        val = L.groupby("item_key")["qty_base"].sum()
    if len(val) == 0:
        raise Abstain("empty")
    asc = c.has(r"ban (it|cham) nhat|it nhat|cham nhat|thap nhat")
    k = get_k(c, None)
    single = k is None
    keys, vals = _rank(val, k or 1, ascending=asc, key_order=item_order(c.M))
    names = [display_name(c.M, x) for x in keys]
    if single and not c.has(r"nhung (mat hang|mon|san pham)|liet ke|cac (mat hang|mon)"):
        return "string", names[0]
    return "list", names

def _top_customers_by(c, o, how, k=None):
    o = o[o["cust_key"].notna()]
    if len(o) == 0:
        raise Unanswerable("no customers")
    if how == "count":
        val = o.groupby("cust_key").size()
    elif how == "days":
        val = o.groupby("cust_key")["date"].nunique()
    else:
        val = o.groupby("cust_key")["revenue"].sum()
    k2 = k or get_k(c, None)
    keys, vals = _rank(val, k2 or 1, key_order=lambda x: str(x))
    disp = o.drop_duplicates("cust_key").set_index("cust_key")["cust_disp"]
    names = [disp.get(x, x) for x in keys]
    if k2 is None:
        return "string", names[0]
    return "list", names

def h_top_customers(c):
    if not c.has(r"(khach|\bai\b)") or not c.has(r"nhat|top"):
        return None
    if c.has(r"\bno\b|cong no|con thieu|mua chiu"):
        return None
    if not (c.has(r"khach") or c.has(r"\bai\b")):
        return None
    p = main_period(c)
    o = orders_in(c, p, valid=not c.has(r"huy|doi y"))
    if "cust_key" not in o or o["cust_key"].isna().all():
        raise Unanswerable("no customer identity")
    f = collect_filters(c, False)
    o = apply_order_filters(c, o, f)
    if c.has(r"doi y|huy nhieu"):
        return _top_customers_by(c, o[o["status"] == "cancel"], "count")
    if c.has(r"trung thanh|nhieu ngay"):
        return _top_customers_by(c, o, "days")
    if c.has(r"sop"):
        oo = o[o["cust_key"].notna()]
        cnt = oo.groupby("cust_key").size()
        avg = oo.groupby("cust_key")["revenue"].sum() / cnt
        avg = avg[cnt >= 3]
        keys, _ = _rank(avg, get_k(c, 1), key_order=lambda x: str(x))
        disp = oo.drop_duplicates("cust_key").set_index("cust_key")["cust_disp"]
        return ("string", disp.get(keys[0])) if get_k(c) is None else ("list", [disp.get(x) for x in keys])
    if c.has(r"nhieu (don|lan|hoa don|bill)|so don|mua nhieu lan|hay mua"):
        return _top_customers_by(c, o, "count")
    return _top_customers_by(c, o, "revenue")

def h_customers_norders(c):
    if not c.has(r"khach") or not c.has(r"(\d+)\s*(don|lan|hoa don)"):
        return None
    m = re.search(r"(?:tu|it nhat|>=|≥|tren|hon|qua)\s*(\d+)\s*(?:don|lan|hoa don)", c.s)
    if not m:
        return None
    n = int(m.group(1))
    incl = not re.search(r"(tren|hon|qua)\s*" + m.group(1), c.s)
    p = main_period(c)
    o = orders_in(c, p)
    if "cust_key" not in o or o["cust_key"].isna().all():
        raise Unanswerable("no customer identity")
    cnt = o[o["cust_key"].notna()].groupby("cust_key").size()
    hit = int((cnt >= n).sum() if incl else (cnt > n).sum())
    if c.has(r"phan tram|%|ti le|chiem"):
        if len(cnt) == 0:
            raise Abstain("no buyers")
        return "percent", r2(hit / len(cnt) * 100)
    return "number", hit

def h_items_norders(c):
    m = re.search(r"(?:o|trong|duoc mua o|co mat trong|xuat hien trong|co mat o|xuat hien o)\s*(?:it nhat\s*|tu\s*|>=\s*|≥\s*)?(\d+)\s*(?:don|bill|hoa don)", c.s)
    if not m or not c.has(r"(mat hang|mon|san pham)"):
        return None
    n = int(m.group(1))
    p = main_period(c)
    L = lines_of(c, orders_in(c, p))
    cnt = L.groupby("item_key")["order_id"].nunique()
    incl = c.has(r"tro len|it nhat|>=|≥")
    return "number", int((cnt >= n).sum() if incl else (cnt > n).sum())

def h_order_count(c):
    if not c.has(r"(bao nhieu|may|so|dem|tong|chot)\s*(?:duoc\s*)?(don|bill|hoa don|order|phieu|lan ban)|co bao nhieu don|bao nhieu don|may don|so don|so luong don|bao nhieu order|may bill|bao nhieu bill"):
        return None
    if c.has(r"khach|mat hang|\bmon\b|huy"):
        if not c.has(r"co (it nhat )?mot mat hang|co mat hang|co .*nhom"):
            return None
    p = main_period(c)
    f = collect_filters(c, True)
    guard_unlinked(c, f)
    o = apply_order_filters(c, orders_in(c, p), f)
    if f.get("items") or f.get("group") or f.get("rx") is not None:
        L = lines_of(c, o)
        L = apply_line_filters(c, L, f)
        return "number", int(L["order_id"].nunique())
    return "number", int(len(o))


UNIT_WORDS = r"(ly|chai|lon|goi|hop|cai|cay|bao|bich|vi|vien|tuyp|lo|ong|kg|thung|loc|binh|cuon|tam|m2|m3|met|m|dia|to|phan|noi|con|ket|qua|cuc|bo|can|set|mieng|chiec|lan|tan|xe)"

def h_unit_count(c):
    """'bao nhieu ly size M' / 'ban ra bao nhieu cay' (all items with that base unit)"""
    m = re.search(r"(?:bao nhieu|may|so)\s*" + UNIT_WORDS + r"\b", c.s)
    if not m or c.has(r"phan tram|%|ti trong|ty trong|tang gia|doi gia|len gia|ton kho|trong kho|con lai|con bao nhieu|nhap"):
        return None
    unit = m.group(1)
    items = find_items(c)
    guard_unlinked(c, {"items": items} if items else collect_filters(c, True))
    p = main_period(c)
    o = orders_in(c, p)
    f = collect_filters(c, True)
    o = apply_order_filters(c, o, f)
    L = lines_of(c, o)
    L = nonempty(apply_line_filters(c, L, {k: v for k, v in f.items() if k in ("items", "group", "rx", "ingredient")}), "unit-count lines")
    if c.has(r"size (m|l)\b"):
        sz = re.search(r"size (m|l)\b", c.s).group(1).upper()
        nm = L["item_name"].astype(str)
        L = L[nm.str.upper().str.rstrip().str.endswith(f"({sz})")]
        return "number", num_out(float(L["qty_base"].sum()))
    if items:
        # quantity of that item expressed in asked unit
        k0 = items[0]
        q = L["qty_base"].sum()
        fac = c.M.factor(k0, unit)
        if fac != fac:
            fac = 1.0
        return "number", num_out(float(q / fac) if fac else float(q))
    # all items whose base unit is the unit
    bu = L["item_key"].map(lambda k: nkey(c.M.base_unit.get(k) or c.M.products.get(k, {}).get("base_unit") or ""))
    if (bu == "").all():
        # infer base unit from line units where qty==qty_base
        raise Abstain("no base units")
    L = L[bu == unit]
    return "number", num_out(float(L["qty_base"].sum()))

def h_share(c):
    if not c.has(r"phan tram|%|ti trong|ty trong|chiem") or not c.has(r"tong doanh thu|doanh thu|tien"):
        return None
    if c.has(r"so voi|thay doi|tang hay giam|so vs|hon hay kem"):
        return None
    p = main_period(c)
    f = collect_filters(c, True)
    total = revenue(orders_in(c, p))
    if not f:
        raise Abstain("share without filter")
    part = filtered_revenue(c, p, f)
    if total == 0:
        raise Abstain("0")
    return "percent", r2(part / total * 100)

def h_two_values(c):
    """'doanh thu ... COD la bao nhieu, ... chuyen khoan la bao nhieu' -> list"""
    pms = find_paymethods(c)
    chs = find_channels(c)
    p = main_period(c)
    if len(pms) >= 2 and is_revenue_q(c):
        vals = [rmoney(filtered_revenue(c, p, {"paymethod": pm})) for pm in pms]
    elif len(chs) >= 2 and is_revenue_q(c):
        vals = [rmoney(filtered_revenue(c, p, {"channel": ch})) for ch in chs]
    else:
        return None
    if c.has(r"ben nao|nao nhieu hon|nao cao hon"):
        return None
    return "list", vals

def h_item_qty(c):
    items = find_items(c)
    if not items:
        return None
    if not c.has(r"ban (ra|duoc)|ban bao nhieu|ban may|so luong|bao nhieu|may"):
        return None
    if c.has(r"\bton\b|con lai|trong kho|\bnhap\b|gia ban|gia nhap|don gia"):
        return None
    p = main_period(c)
    o = orders_in(c, p)
    L = lines_of(c, o)
    L = nonempty(L[L["item_key"].isin(items[:1])], "item lines")
    if is_revenue_q(c) and not re.search(r"(bao nhieu|may)\s*" + UNIT_WORDS + r"\b", c.s):
        return "number", rmoney(L["amount"].sum())
    m = re.search(r"(?:bao nhieu|may)\s*" + UNIT_WORDS + r"\b", c.s)
    q = L["qty_base"].sum()
    if m:
        fac = c.M.factor(items[0], m.group(1))
        if fac == fac and fac:
            q = q / fac
    return "number", num_out(float(q))

def guard_unlinked(c, f):
    """abstain when the question names something we could not link (unknown item/customer/attribute)"""
    if f.get("items") or f.get("group"):
        return
    extra = []
    if f.get("customer"):
        extra.append(f["customer"])
    u = unlinked_tokens(c, extra)
    if len(u) >= 2:
        raise Abstain("unlinked mention: " + " ".join(u[:5]))

def h_revenue(c):
    if not is_revenue_q(c):
        return None
    if c.has(r"\bno\b|cong no|nhap hang|tien nhap|lai gop|loi nhuan|lai rong"):
        return None
    ps = [p for p in c.periods if p.kind != "snapshot"]
    if not ps:
        raise Abstain("no period")
    f = collect_filters(c, True)
    guard_unlinked(c, f)
    if len(ps) >= 2 and c.has(r"hon|kem|chenh|so voi"):
        a, b = filtered_revenue(c, ps[0], f), filtered_revenue(c, ps[1], f)
        return "number", rmoney(a - b)
    return "number", rmoney(filtered_revenue(c, ps[0], f))

HANDLERS = [h_subjective, h_trend_bool, h_growth, h_cancel, h_best_month, h_revenue_list, h_two_values, h_avg_order,
            h_customers_norders, h_items_norders, h_orders_threshold, h_days_threshold, h_best_day,
            h_top_customers, h_top_items, h_share, h_unit_count, h_order_count, h_item_qty, h_revenue]

def sanity_problem(t, a):
    """reject answers that are malformed or implausible instead of risking -1"""
    import math
    if t in ("number", "percent"):
        if a is None or isinstance(a, bool) or not isinstance(a, (int, float)) or math.isnan(a) or math.isinf(a):
            return "non-finite number"
        if t == "percent" and abs(a) > 5000:
            return "implausible percent"
    if t == "list":
        if not a or any(x is None or (isinstance(x, float) and math.isnan(x)) for x in a):
            return "empty/invalid list"
    if t == "date" and (not isinstance(a, str) or not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a)):
        return "invalid date"
    if t == "boolean" and not isinstance(a, bool):
        return "invalid boolean"
    if t == "string" and (a is None or str(a).strip() == "" or str(a).lower() in ("nan", "none")):
        return "empty string"
    if t == "table" and (not a or not a.get("rows")):
        return "empty table"
    return None

UNKNOWN_KINDS = (None, "other", "unknown")

def data_understood(M):
    """Generic health check of the canonical model. A shop counts as 'understood' only when a sales/order table was
    recognised and no sizeable table was left unclassified. Otherwise information the engine cannot see may exist,
    so the engine must not claim 'unanswerable' (-1 if wrong) and its answers need the agent's agreement."""
    v = getattr(M, "_understood", None)
    if v is None:
        big_unknown = [n for n, (df, roles, kind) in M.tables.items() if kind in UNKNOWN_KINDS and len(df) >= 100]
        v = (M.orders is not None and len(M.orders) > 0) and not big_unknown
        try:
            M._understood = v
        except Exception:
            pass
    return v

def answer(M, question, as_of, cal=None, extra_handlers=None):
    r = _answer(M, question, as_of, cal, extra_handlers)
    if not data_understood(M):
        if r.get("type") == "unanswerable":
            r = {"type": "abstain", "reason": "unanswerable claimed on partially understood data: " + str(r.get("reason")), "handler": r.get("handler")}
        elif r.get("type") != "abstain":
            r["data_partial"] = True
    return r

def _answer(M, question, as_of, cal=None, extra_handlers=None):
    c = Ctx(M, question, as_of, cal)
    if extra_handlers is None:
        from .intents2 import EXTRA
        extra_handlers = EXTRA
    for h in [h_subjective, h_trend_bool] + list(extra_handlers) + HANDLERS:
        try:
            r = h(c)
        except Unanswerable as e:
            return {"type": "unanswerable", "reason": str(e), "handler": h.__name__}
        except Abstain as e:
            return {"type": "abstain", "reason": str(e), "handler": h.__name__}
        if r is not None:
            t, a = r
            bad = sanity_problem(t, a)
            if bad:
                return {"type": "abstain", "reason": "sanity: " + bad, "handler": h.__name__}
            return {"type": t, "answer": a, "handler": h.__name__}
    return {"type": "abstain", "reason": "no intent", "handler": None}
