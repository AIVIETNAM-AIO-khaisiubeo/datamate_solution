"""Second wave of intents: price, imports, stock, debt, profit, deliveries, pharmacy, staff."""
import re
import numpy as np
import pandas as pd
from .engine import *
from .intents import (get_k, threshold, main_period, months_listed, mp, _rank, item_order, UNIT_WORDS, is_revenue_q,
                      _top_customers_by)
from .norm import nkey, parse_money, paymethod_of
from .model import _dates
from .timeparse import Period

def _date_mentions(c):
    return [p for p in c.periods if p.kind in ("day", "snapshot")]

def _asked_unit(c, item):
    units = sorted(c.M.conv.get(item, {}).keys(), key=lambda u: -len(u))
    for u in units:
        if u and re.search(r"(?:mot|1|moi|tinh theo|theo|bao nhieu|may|so)\s*" + re.escape(u) + r"\b", c.s):
            return u
    m = re.search(r"(?:mot|1|moi|tinh theo|theo|bao nhieu|may|so)\s*" + UNIT_WORDS + r"\b", c.s)
    if m:
        return m.group(1)
    return None

def _unit_factor(c, item, unit):
    if unit is None:
        return 1.0
    f = c.M.factor(item, unit)
    return f if f == f and f else 1.0

# ------------------------------------------------------------------ prices
def h_price_effect(c):
    if not (c.has(r"\b(tang|len|doi) gia\b|\bgia (tang|len)\b") or
            (c.has(r"\blen\s*[\d.,]+\s*(k|d|dong|nghin|ngan)?\b") and c.has(r"\d+\s*ngay") and find_items(c))):
        return None
    items = find_items(c)
    if not items:
        raise Abstain("item")
    it = items[0]
    m = re.search(r"(\d+)\s*ngay\s*(?:ke tu|tu|bat dau)", c.s) or re.search(r"(\d+)\s*ngay", c.s)
    if not m:
        raise Abstain("N")
    n = int(m.group(1))
    days = _date_mentions(c)
    if days:
        d = days[0].start
    else:
        # find from price history: first date the price rose to the stated new price
        mp_ = re.search(r"(?:len|tang len|thanh|den)\s*(?:gia\s*)?([\d.,]+)\s*(k|d|dong)?", c.s)
        ph = c.M.price_hist
        if ph is None or mp_ is None:
            raise Abstain("no date")
        newp = parse_money(mp_.group(1)) * (1000 if mp_.group(2) == "k" else 1)
        h = ph[ph["item_key"] == it].sort_values("date")
        prev = None; d = None
        for dt_, pr in zip(h["date"], h["price"]):
            if prev is not None and pr > prev and abs(pr - newp) < 1:
                d = dt_; break
            prev = pr
        if d is None:
            raise Abstain("no rise found")
    a = Period(d, d + pd.Timedelta(days=n - 1), "range")
    b = Period(d - pd.Timedelta(days=n), d - pd.Timedelta(days=1), "range")
    def q(p):
        o = orders_in(c, p)
        L = lines_of(c, o)
        return float(L.loc[L["item_key"] == it, "qty_base"].sum())
    qa, qb = q(a), q(b)
    if c.has(r"phan tram|%"):
        return "percent", pct(qa, qb)
    return "number", num_out(qa - qb)

def h_price_on_date(c):
    if not c.has(r"(\bdon gia\b(?! tri)|\bgia ban\b|\bgia bao nhieu\b|\bvoi gia\b|\bgia la\b|\bban gia\b|\bgia mot\b|\bgia 1\b)"):
        return None
    if c.has(r"nhap|gia von|tang gia|len gia|doi gia"):
        return None
    items = find_items(c)
    if not items:
        raise Abstain("item")
    days = _date_mentions(c) or [p for p in c.periods if p.kind == "day"]
    if not days:
        return None
    d = days[0].start
    it = items[0]
    if d > c.today:
        raise Abstain("future")
    o = O(c)
    oids = set(o.loc[o["date"] == d, "order_id"])
    L = c.M.lines
    x = L[L["order_id"].isin(oids) & (L["item_key"] == it)]
    x = x[x["price"].notna() & (x["qty"] != 0)]
    if len(x) == 0:
        check_coverage(c, days[0])
        raise Abstain("no sale that day")
    per_base = x["price"] / (x["qty_base"] / x["qty"]).replace(0, np.nan)
    per_base = per_base.dropna()
    vals = per_base.round(4).value_counts()
    v = vals.index[0]
    unit = _asked_unit(c, it)
    fac = _unit_factor(c, it, unit)
    return "number", rmoney(v * fac)

# ------------------------------------------------------------------ imports
def _imports_in(c, p):
    I = c.M.imports
    if I is None or len(I) == 0:
        raise Unanswerable("no imports")
    check_coverage_imports(c, p)
    a, b = clip(c, p)
    x = I[(I["date"] >= a) & (I["date"] <= b) & ~I["cancelled"]]
    return x

def check_coverage_imports(c, p):
    I = c.M.imports
    a, b = clip(c, p)
    ds = min(I["date"].min(), c.M.data_start) if c.M.data_start is not None else I["date"].min()
    if b < ds or a > (c.M.data_end or I["date"].max()):
        raise Unanswerable("period outside data")

def h_import(c):
    if not c.has(r"\bnhap\b|phieu nhap|nha cung cap|\bncc\b"):
        return None
    if c.has(r"ban (ra|duoc)") and not c.has(r"nhap (ve|vao|hang)|tien nhap|gia nhap|phieu nhap|nhap tru"):
        return None
    if c.has(r"ban ra .* nhap|nhap tru ban|chenh .*nhap"):
        return None
    p = main_period(c)
    x = _imports_in(c, p)
    items = find_items(c)
    if c.has(r"nha cung cap|\bncc\b"):
        if len(x) == 0:
            raise Abstain("no imports in period")
        if c.has(r"chinh|nhieu phieu|nhieu lan|so phieu"):
            val = x.groupby("supplier")["receipt_id"].nunique()
        else:
            val = x.groupby("supplier")["amount"].sum()
        k = get_k(c, None)
        keys, _ = _rank(val, k or 1, key_order=lambda s: str(s))
        names = [c.M.sup_names.get(str(s).strip(), s) for s in keys]
        return ("string", names[0]) if k is None else ("list", names)
    if items and c.has(r"binh quan|trung binh"):
        it = items[0]
        y = x[x["item_key"] == it]
        if len(y) == 0:
            raise Unanswerable("item not imported in period")
        unit = _asked_unit(c, it)
        fac = _unit_factor(c, it, unit) if unit else None
        qb = y["qty_base"].sum()
        if fac is None:
            # in import unit
            return "number", rmoney(y["amount"].sum() / y["qty"].sum())
        return "number", rmoney(y["amount"].sum() / (qb / fac))
    if items and c.has(r"bao nhieu|may|so luong"):
        it = items[0]
        y = x[x["item_key"] == it]
        if c.has(r"tien"):
            return "number", rmoney(y["amount"].sum())
        unit = _asked_unit(c, it)
        fac = _unit_factor(c, it, unit)
        return "number", num_out(float(y["qty_base"].sum() / fac))
    if c.has(r"(bao nhieu|may|so) phieu"):
        return "number", int(x["receipt_id"].nunique())
    if c.has(r"tien|gia tri|bao nhieu|ton|het"):
        tot = x.drop_duplicates("receipt_id")["total"] if "total" in x and x["total"].notna().all() and len(x) else None
        val = x["amount"].sum()
        return "number", rmoney(val)
    return None

# ------------------------------------------------------------------ stock
def stock_at(c, it, when):
    """stock (base units) at end of day `when`"""
    M = c.M
    if M.ledger is not None and len(M.ledger):
        L = M.ledger[(M.ledger["item_key"] == it) & (M.ledger["ts"] < when + pd.Timedelta(days=1))]
        if len(L) == 0:
            return 0.0
        return float(L.sort_values("ts")["stock"].iloc[-1])
    if it not in M.stock_end:
        if M.lot_sales is not None and (M.lots is not None or (M.imports is not None and "lot" in M.imports.columns)):
            lots = _lot_stock(c, when)
            return float(lots.loc[lots["item_key"] == it, "remain"].sum())
        raise Unanswerable("no stock info")
    s = M.stock_end[it]
    end = M.data_end
    if when >= end:
        return float(s)
    # roll back: undo imports and sales after `when`
    I = M.imports
    if I is not None and len(I):
        y = I[(I["item_key"] == it) & (I["date"] > when) & ~I["cancelled"]]
        s -= float(y["qty_base"].sum())
    o = M.orders
    oo = o[(o["date"] > when) & (o["status"] != "cancel")]
    L = M.lines
    z = L[L["order_id"].isin(set(oo["order_id"])) & (L["item_key"] == it)]
    s += float(z["qty_base"].sum())
    return s

def has_stock_info(M):
    return (M.ledger is not None and len(M.ledger)) or len(M.stock_end) > 0 or (
        M.lot_sales is not None and (M.lots is not None or (M.imports is not None and "lot" in M.imports.columns)))

def find_big_unit(c, it):
    units = sorted(c.M.conv.get(it, {}).items(), key=lambda t: -len(t[0]))
    bu = nkey(c.M.base_unit.get(it) or "")
    for u, f in units:
        if u and u != bu and re.search(r"\b" + re.escape(u) + r"\s*(chan|nguyen)\b", c.s):
            return u, f
    for u, f in units:
        if u and u != bu and re.search(r"\b" + re.escape(u) + r"\b", c.s):
            return u, f
    return None, None

def h_stock(c):
    if not (c.has(r"ton kho|trong kho|con ton|ton cuoi|het hang|con hang|gia tri hang ton|\bhang ton\b|ton bao nhieu|\bcon du\b|chay hang|dut hang") or
            (c.has(r"con (lai )?(bao nhieu|may)") and c.has(r"cuoi ngay|het hom nay|den het|tinh den|toi het|het ngay|cuoi bua"))):
        return None
    if c.has(r"\bno\b|cong no|con thieu|\bhet han\b|han dung|han su dung"):
        return None
    if not has_stock_info(c.M):
        raise Unanswerable("no stock data")
    snaps = [p for p in c.periods if p.kind in ("snapshot", "day")]
    when = snaps[0].start if snaps else c.today
    when = min(when, c.today)
    if c.has(r"gia tri|gia von|dang bao nhieu tien|bao nhieu tien"):
        tot = 0.0
        keys = set(c.M.stock_end) | (set(c.M.ledger["item_key"]) if c.M.ledger is not None else set())
        for it in keys:
            q = stock_at(c, it, when)
            cost = c.M.products.get(it, {}).get("cost")
            if q > 0:
                if cost is None or cost != cost:
                    raise Unanswerable("no cost")
                tot += q * cost
        return "number", rmoney(tot)
    if c.has(r"het hang|chay hang|dut hang"):
        keys = set(c.M.stock_end) | (set(c.M.ledger["item_key"]) if c.M.ledger is not None else set())
        out = sorted([it for it in keys if stock_at(c, it, when) <= 0], key=item_order(c.M))
        if c.has(r"bao nhieu|may|so"):
            return "number", len(out)
        return "list", [display_name(c.M, it) for it in out]
    items = find_items(c)
    if not items:
        raise Abstain("item")
    it = items[0]
    q = stock_at(c, it, when)
    if c.has(r"\ble\b|chan|nguyen"):
        big, fac = find_big_unit(c, it)
        if big is None:
            raise Unanswerable("no conversion")
        return "list", [int(q // fac), num_out(float(q - (q // fac) * fac))]
    unit = _asked_unit(c, it)
    fac = _unit_factor(c, it, unit)
    return "number", num_out(float(q / fac))

# ------------------------------------------------------------------ debt
def _debt_orders(c):
    o = O(c)
    if c.M.payments is None and "debt" not in o.columns:
        for n, (df, roles, kind) in c.M.tables.items():
            if any(nkey(cc) in ("no", "con no", "cong no", "da tra", "tra gop", "da gop", "no cu") for cc in df.columns):
                raise Abstain("debt book present but not modelled")
        if getattr(c.M, "debt_terms", None) is not None and c.has(r"\bhan\b|den han|qua han|ngay phai tra|phai tra ngay|khi nao tra"):
            raise Abstain("asks about due dates and a due-date table exists (not modelled by rules)")
        raise Unanswerable("no debt info (only due dates or nothing recorded)")
    if "payable" not in o.columns and "revenue" not in o.columns:
        raise Unanswerable("no totals")
    o = o[o["status"] != "cancel"]
    if "paymethod" in o.columns:
        o = o[o["paymethod"] != "cod"]
    if "channel" in o.columns:
        o = o[o["channel"] != "ecommerce"]
    return o

def debt_as_of(c, when):
    """Series order_id -> remaining debt at end of `when` (only >0), plus order frame"""
    o = _debt_orders(c)
    o = o[o["date"] <= when]
    pay = c.M.payments
    total = o["payable"] if "payable" in o.columns else o["revenue"]
    if pay is not None:
        p = pay[pay["ts"].dt.normalize() <= when].groupby("order_id")["amount"].sum()
        paid = o["order_id"].map(p).fillna(0)
    else:
        if when < c.M.data_end - pd.Timedelta(days=0):
            raise Unanswerable("debt history unavailable")
        paid = total - o["debt"].fillna(0)
    rem = (total.fillna(0) - paid).round(6)
    o = o.assign(remaining=rem.values)
    return o[o["remaining"] > 0.5]

def h_debt(c):
    if not c.has(r"\bno\b|cong no|con thieu|mua chiu|ban chiu|ghi no|chua tra het|tra bot|tra no|thu hoi|ban thieu|mua thieu"):
        return None
    if c.has(r"\bnha cung cap\b|\bncc\b"):
        return None
    snaps = [p for p in c.periods if p.kind in ("snapshot", "day")]
    # flows within a period
    if c.has(r"no them|ghi no them|mua chiu them|tra bot|tra no|thu hoi|thu duoc .* no|khach tra .*no|tra duoc"):
        p = main_period(c)
        check_coverage(c, p)
        a, b = clip(c, p)
        o = _debt_orders(c)
        if "paymethod" in o.columns and (o["paymethod"] == "debt").any():
            o = o[o["paymethod"] == "debt"]
        pay = c.M.payments
        if pay is None:
            raise Unanswerable("no payment history")
        cust = find_customer(c)
        if cust is not None:
            o = o[o["cust_key"] == cust]
        total = o["payable"] if "payable" in o.columns else o["revenue"]
        pj = pay.merge(o[["order_id", "date"]], on="order_id")
        pj["pday"] = pj["ts"].dt.normalize()
        same = pj[pj["pday"] <= pj["date"]].groupby("order_id")["amount"].sum()
        later = pj[(pj["pday"] > pj["date"]) & (pj["pday"] >= a) & (pj["pday"] <= b)]["amount"].sum()
        inp = o[(o["date"] >= a) & (o["date"] <= b)]
        added = (inp["payable"] if "payable" in inp.columns else inp["revenue"]).fillna(0) - inp["order_id"].map(same).fillna(0)
        added = float(added.clip(lower=0).sum())
        if cust is not None or c.has(r"no them.*tra|hai so"):
            return "list", [rmoney(added), rmoney(later)]
        if c.has(r"no them|mua chiu them|ghi no them"):
            return "number", rmoney(added)
        return "number", rmoney(later)
    when = snaps[0].start if snaps else c.today
    when = min(when, c.today)
    d = debt_as_of(c, when)
    if c.has(r"tuoi no|nhom tuoi|0\s*[-–]\s*7"):
        age = (when - d["date"]).dt.days
        return "list", [rmoney(d.loc[age <= 7, "remaining"].sum()), rmoney(d.loc[(age >= 8) & (age <= 15), "remaining"].sum()),
                        rmoney(d.loc[age > 15, "remaining"].sum())]
    m = re.search(r"(?:qua|hon|tren)\s*(\d+)\s*ngay", c.s)
    if m:
        n = int(m.group(1))
        age = (when - d["date"]).dt.days
        return "number", rmoney(d.loc[age > n, "remaining"].sum())
    if c.has(r"(bao nhieu|may|so) (khach|nguoi)"):
        return "number", int(d["cust_key"].dropna().nunique())
    k = get_k(c, None)
    if k or c.has(r"khach nao|ai .*no nhieu|no nhieu nhat"):
        dd = d[d["cust_key"].notna()]
        val = dd.groupby("cust_key")["remaining"].sum()
        keys, vals = _rank(val, k or 1, key_order=lambda x: str(x))
        disp = dd.drop_duplicates("cust_key").set_index("cust_key")["cust_disp"]
        rows = [[disp.get(x, x), rmoney(v)] for x, v in zip(keys, vals)]
        if k is None and not c.has(r"bao nhieu|kem"):
            return "string", rows[0][0]
        return "table", {"columns": ["khach_hang", "so_tien_no"], "rows": rows}
    cust = find_customer(c)
    if cust is not None:
        return "number", rmoney(d.loc[d["cust_key"] == cust, "remaining"].sum())
    return "number", rmoney(d["remaining"].sum())

# ------------------------------------------------------------------ profit
def gross_profit(c, p, f=None):
    o = orders_in(c, p)
    L = lines_of(c, o, items_only=False)
    L = L[~L["nonitem"]]
    if f:
        L = apply_line_filters(c, L, f)
    if L["cost_unit"].isna().all():
        raise Unanswerable("no cost at sale")
    return float((L["amount"] - L["qty"] * L["cost_unit"].fillna(0)).sum())

def h_profit(c):
    if not re.search(r"lai gop|loi gop|loi nhuan|lai rong|loi that|lai that|tien loi|\blai bao nhieu|loi bao nhieu|\blai duoc", c.s.replace("mang lai", "mang_lai")):
        return None
    if c.has(r"mat hang nao|mon nao|san pham nao|hang nao|(san pham|mat hang|mon).*nhat|la gi"):
        return None
    p = main_period(c)
    f = {k: v for k, v in collect_filters(c, True).items() if k in ("items", "group")}
    gp = gross_profit(c, p, f)
    if c.has(r"lai rong|loi nhuan rong|tru chi phi|loi that|lai that|sau chi phi|tru het chi phi"):
        E = c.M.expenses
        if E is None:
            raise Unanswerable("no expenses")
        a, b = clip(c, p)
        e = E[(E["date"] >= p.start) & (E["date"] <= p.end)]["amount"].sum()
        return "number", rmoney(gp - e)
    return "number", rmoney(gp)

# ------------------------------------------------------------------ staff
def h_staff(c):
    if not c.has(r"nhan vien|nhan dien|\bnv\b|thu ngan|nguoi ban"):
        return None
    if not c.has(r"\bnao\b|\bai\b"):
        return None
    p = main_period(c)
    o = orders_in(c, p)
    if "staff" not in o or o["staff"].isna().all():
        raise Unanswerable("no staff")
    if c.has(r"cham chi|nhieu ngay"):
        val = o.groupby("staff")["date"].nunique()
    elif c.has(r"nhieu don"):
        val = o.groupby("staff").size()
    else:
        val = o.groupby("staff")["revenue"].sum()
    k = get_k(c, None)
    keys, _ = _rank(val, k or 1, key_order=lambda x: str(x))
    names = list(keys)
    return ("string", names[0]) if k is None else ("list", names)

# ------------------------------------------------------------------ deliveries (building materials)
def h_delivery(c):
    if not c.has(r"dot giao|chuyen giao|luot giao|cho giao|giao .*dot|xe (da )?giao|da giao|chua giao|giao hang|dot hang|xe cho|chuyen hang"):
        return None
    D = c.M.deliveries
    if D is None:
        raise Unanswerable("no delivery data")
    if c.has(r"(bao nhieu|may|so) (dot|chuyen|luot)"):
        p = main_period(c)
        a, b = clip(c, p)
        x = D[(D["date"] >= a) & (D["date"] <= b)]
        return "number", int(x.drop_duplicates(["order_id", "batch"]).shape[0])
    items = find_items(c)
    if items:
        p = main_period(c)
        o = O(c)
        oids = set(o.loc[(o["date"] >= p.start) & (o["date"] <= p.end) & (o["status"] != "cancel"), "order_id"])
        x = D[D["order_id"].isin(oids) & (D["item_key"] == items[0]) & (D["date"] <= c.today)]
        unit = _asked_unit(c, items[0])
        fac = _unit_factor(c, items[0], unit)
        return "number", num_out(float(x["qty_base"].sum() / fac))
    return None

# ------------------------------------------------------------------ pharmacy lots
def _lot_stock(c, when):
    key = ("_lotstock", when)
    cache = c.M.__dict__.setdefault("_cache", {})
    if key in cache:
        return cache[key]
    r = _lot_stock_impl(c, when)
    cache[key] = r
    return r

def _lot_stock_impl(c, when):
    M = c.M
    lots = M.lots
    if lots is None:
        I = M.imports
        if I is None or "lot" not in I.columns:
            raise Unanswerable("no lot data")
        lots = I[~I["cancelled"]].rename(columns={"qty_base": "qty_in"})[["lot", "item_key", "expiry", "qty_in", "price", "date", "unit", "qty"]].copy()
        lots["lot"] = lots["lot"].astype(str).str.strip()
        lots["price_base"] = lots["price"] / (lots["qty_in"] / lots["qty"])
    else:
        lots = lots.copy()
        I = M.imports
        if I is not None and "lot" in I.columns:
            imp = I.assign(lot=I["lot"].astype(str).str.strip()).drop_duplicates("lot").set_index("lot")
            lots["date"] = lots["lot"].map(imp["date"])
            if lots["price"].isna().all():
                lots["price_base"] = lots["lot"].map(imp["price"] / (imp["qty_base"] / imp["qty"]))
            else:
                lots["price_base"] = lots["price"]
        else:
            lots["price_base"] = lots["price"]
        if "date" not in lots.columns:
            if lots.get("receipt_id") is not None and getattr(M, "import_headers", None):
                lots["date"] = lots["receipt_id"].map(lambda r: M.import_headers.get(r, {}).get("date"))
                lots["date"] = pd.to_datetime(lots["date"], errors="coerce")
            else:
                lots["date"] = pd.NaT
    lots = lots[(lots["date"].isna()) | (lots["date"] <= when)]
    S = M.lot_sales
    if S is None:
        raise Unanswerable("no lot sales")
    o = M.orders
    od = o.set_index("order_id")["date"]
    S = S.assign(date=S["order_id"].map(od), status=S["order_id"].map(o.set_index("order_id")["status"]))
    S = S[(S["date"] <= when) & (S["status"] != "cancel")]
    sold = S.groupby("lot")["qty"].sum() - S.groupby("lot")["qty_ret"].sum()
    lots["remain"] = lots["qty_in"] - lots["lot"].map(sold).fillna(0)
    return lots

def h_expiry(c):
    if not c.has(r"\bhet han\b|\bhan dung\b|\bhan su dung\b|\bhsd\b"):
        return None
    when = c.today
    lots = _lot_stock(c, when)
    live = lots[lots["remain"] > 0]
    m = re.search(r"(\d+)\s*ngay\s*(toi|nua|sap toi|ke tiep)", c.s)
    if m or c.has(r"tu .* den het"):
        if m:
            n = int(m.group(1)); end = when + pd.Timedelta(days=n)
        else:
            ds = [p for p in c.periods if p.kind in ("day", "snapshot", "range")]
            end = ds[-1].end if ds else None
            if end is None:
                raise Abstain("window")
        x = live[(live["expiry"] >= when) & (live["expiry"] <= end)]
        if c.has(r"tien von|gia tri|bao nhieu tien"):
            return "number", rmoney((x["remain"] * x["price_base"]).sum())
        if c.has(r"bao nhieu|may|so"):
            if c.has(r"\blo\b"):
                return "number", int(len(x))
            return "number", int(x["item_key"].nunique())
        first = x.groupby("item_key")["expiry"].min().reset_index()
        first["ord"] = first["item_key"].map(item_order(c.M))
        first = first.sort_values(["expiry", "ord"])
        return "list", [display_name(c.M, k) for k in first["item_key"]]
    items = find_items(c)
    if items and c.has(r"\blo\b"):
        snaps = [p for p in c.periods if p.kind in ("snapshot", "day")]
        when2 = min(snaps[0].start, c.today) if snaps else when
        lots2 = _lot_stock(c, when2)
        live2 = lots2[lots2["remain"] > 0]
        x = live2[(live2["item_key"] == items[0]) & (live2["expiry"] >= when2)].sort_values(["expiry", "lot"])
        if len(x) == 0:
            raise Abstain("no lot")
        if c.has(r"ngay nao|han (dung|su dung) (la )?(ngay )?(nao|bao gio)|het han (vao )?(ngay nao|khi nao|bao gio)|co han"):
            return "date", x["expiry"].iloc[0].strftime("%Y-%m-%d")
        return "string", x["lot"].iloc[0]
    return None

# ------------------------------------------------------------------ unsold / slow items
def h_unsold(c):
    if not c.has(r"khong ban (duoc|ra)|chua ban duoc|hang e\b|mat hang e\b|\bbi e\b|ban cham den muc|khong co don|khong ai mua|khong ban duoc"):
        return None
    p = main_period(c)
    a, b = clip(c, p)
    check_coverage(c, p)
    o = O(c)
    ov = o[valid_mask(o)]
    L = c.M.lines
    Lv = L[L["order_id"].isin(set(ov["order_id"])) & ~L["nonitem"] & ~L["is_service_fee"].fillna(False) & L["item_key"].notna()]
    before = set(Lv.loc[Lv["date"] < a, "item_key"])
    during = set(Lv.loc[(Lv["date"] >= a) & (Lv["date"] <= b), "item_key"])
    if c.has(r"dang ban|van dang ban|con kinh doanh"):
        act = {k for k, pr in c.M.products.items() if str(pr.get("active")).lower() in ("true", "1", "1.0", "yes")}
        if any("active" in pr for pr in c.M.products.values()):
            before = before & act
    res = sorted(before - during, key=item_order(c.M))
    if c.has(r"bao nhieu|may|so|dem"):
        return "number", len(res)
    return "list", [display_name(c.M, k) for k in res]

# ------------------------------------------------------------------ misc order-level money (ship fee, discounts, trade-in)

def h_import_minus_sales(c):
    if not c.has(r"nhap tru (so )?ban|nhap tru di ban|chenh (lech )?(giua )?nhap va ban|nhap - ban|so nhap tru so ban|nhap\b.{0,25}\btru (di )?(so )?.{0,12}\bban"):
        return None
    items = find_items(c)
    if not items:
        raise Abstain("item")
    it = items[0]
    p = main_period(c)
    x = _imports_in(c, p)
    qi = float(x.loc[x["item_key"] == it, "qty_base"].sum())
    L = lines_of(c, orders_in(c, p))
    qs = float(L.loc[L["item_key"] == it, "qty_base"].sum())
    unit = _asked_unit(c, it)
    fac = _unit_factor(c, it, unit)
    return "number", num_out((qi - qs) / fac)

def h_tet_yoy(c):
    if not c.has(r"\btet\b") or not c.has(r"tet (nam )?(nay|truoc|ngoai|roi)|nam truoc|nam ngoai"):
        return None
    from .timeparse import lunar_tet
    y = c.today.year
    w = lunar_tet(y, c.cal)
    if w is None or w[0] > c.today:
        y -= 1
    w1, w0 = lunar_tet(y, c.cal), lunar_tet(y - 1, c.cal)
    if not w1 or not w0:
        raise Abstain("calendar")
    f = collect_filters(c, True)
    a = filtered_revenue(c, Period(w1[0], w1[1], "tet"), f); b = filtered_revenue(c, Period(w0[0], w0[1], "tet"), f)
    if c.has(r"phan tram|%"):
        return "percent", pct(a, b)
    return "number", rmoney(a - b)


def h_shift_cmp_early(c):
    if c.has(r"ca chieu") and c.has(r"ca sang") and c.has(r"phan tram|%|hon|kem|chenh"):
        from .intents import h_shift_compare
        return h_shift_compare(c)
    return None


EXTRA = [h_shift_cmp_early, h_tet_yoy, h_import_minus_sales, h_price_effect, h_debt, h_expiry, h_stock, h_import, h_price_on_date, h_profit, h_staff, h_delivery, h_unsold]

WEEKDAYS = {"thu hai": 0, "thu 2": 0, "thu ba": 1, "thu 3": 1, "thu tu": 2, "thu 4": 2, "thu nam": 3, "thu 5": 3,
            "thu sau": 4, "thu 6": 4, "thu bay": 5, "thu 7": 5, "chu nhat": 6, "cn": 6}

def weekday_in(c):
    for k, v in WEEKDAYS.items():
        if re.search(r"\b(moi|ngay|cac ngay|cac|1)\s*" + k + r"\b", c.s) or re.search(r"\b" + k + r"\b", c.s) and k.startswith(("thu", "chu")) and c.has(r"trung binh|binh quan"):
            return v
    return None

def h_weekday_avg(c):
    wd = weekday_in(c)
    if wd is None or not c.has(r"trung binh|binh quan|moi ngay|1 ngay|mot ngay"):
        return None
    p = main_period(c)
    a, b = clip(c, p)
    check_coverage(c, p)
    nd = int((pd.date_range(a, b).weekday == wd).sum())
    if nd == 0:
        raise Abstain("no weekday")
    o = orders_in(c, p)
    o = o[o["date"].dt.weekday == wd]
    if c.has(r"(so|bao nhieu|may) (don|bill|hoa don|luot)") and not c.has(r"doanh thu|tien"):
        return "number", r2(len(o) / nd)
    return "number", rmoney(revenue(o) / nd)

# ------------------------------------------------------------------ service-ticket ledgers (e.g. repair shops)




EXTRA.insert(1, h_weekday_avg)

# ------------------------------------------------------------------ device shops (abbreviation sheet with item types, IMEI)







# ------------------------------------------------------------------ catalog attributes (brand / size / number), order adjustments







