"""Build the canonical business model from raw tables (generic, role-driven)."""
import re, os, unicodedata
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from .loader import load_business
from .semantic import parse_readme, assign_roles
from .norm import (nkey, parse_money, parse_qty, parse_date, parse_time_of_day, status_of,
                   paymethod_of, channel_of, is_walkin, qty_unit_suffix, eval_norm)

MONEY_ROLES = ["line_amount", "price", "old_price", "disc", "order_disc", "cost", "goods_total", "ship_fee",
               "payable", "paid", "debt", "money", "tradein", "pay_amount"]
QTY_ROLES = ["qty", "qty_base", "qty_return_base", "qty_return_base2", "factor", "stock", "guests", "terms_days", "delivery_no"]
HEADER_ROLES = ["order_id", "date", "status", "cust_name", "cust_code", "paymethod", "channel", "staff", "payable",
                "paid", "debt", "goods_total", "order_disc", "ship_fee", "note", "table", "area", "guests", "phone", "address", "cust_group"]

def item_key(name):
    """item identity key from a name: case/space/accents-insensitive"""
    return nkey(name)

def cust_key(name):
    s = str(name)
    s = re.sub(r"[\s\-–(]*\+?\d[\d .]{6,}\)?\s*$", "", s)  # strip trailing phone
    return nkey(s)

def _first(roles, r):
    v = roles.get(r)
    return v[0] if v else None

def _year_from(fname, sheet):
    for t in (str(sheet), str(fname)):
        m = re.search(r"(20\d\d)", t)
        if m:
            return int(m.group(1))
    return None

def _month_from_sheet(sheet):
    m = re.search(r"(?:^|[^0-9])(?:T|Th|Thang|Tháng)?\s*(\d{1,2})(?:[^0-9]|$)", str(sheet))
    return int(m.group(1)) if m else None

class Model:
    pass

def classify(roles, df):
    has = lambda *rs: all(r in roles for r in rs)
    anyof = lambda *rs: any(r in roles for r in rs)
    item = anyof("item_name", "item_code")
    if anyof("abbrev"):
        return "abbrev"
    if anyof("return_id"):
        return "returns"
    if has("stock") and anyof("ts") and anyof("order_id", "receipt_id"):
        return "ledger"
    if has("lot") and anyof("order_id"):
        return "lot_sales"
    if has("delivery_no") or has("plate"):
        return "deliveries"
    if anyof("terms_days", "due_date") and has("order_id") and not item:
        return "debt_terms"
    if has("lot") and item and anyof("qty_base") and not anyof("qty", "unit", "line_amount"):
        return "lots"
    if anyof("receipt_id") and item:
        return "import_lines"
    if anyof("receipt_id") and not item and not anyof("order_id"):
        return "import_headers"
    if has("lot") and anyof("expiry") and item:
        return "lots"
    if anyof("ingredient", "rx"):
        return "drug_info"
    if has("factor") and (anyof("big_unit", "unit")) and item:
        return "units"
    if has("stock") and item:
        return "stock"
    if has("order_id") and anyof("money") and not item:
        return "payments"
    if anyof("shift") and anyof("money"):
        return "shifts"
    if item and anyof("price") and anyof("date") and not anyof("order_id", "qty", "line_amount"):
        return "price_hist"
    if item and (anyof("qty", "line_amount", "qty_base")) and (anyof("order_id", "date")):
        return "sales_lines"
    if has("order_id") and anyof("payable", "goods_total", "status", "date", "debt") and not item:
        return "orders"
    if item and not anyof("order_id", "receipt_id"):
        return "products"
    if anyof("cust_code", "cust_name") and anyof("phone", "cust_group", "address") and not anyof("order_id"):
        return "customers"
    if anyof("expense_cat") or (anyof("money", "payable") and anyof("date") and not anyof("order_id")):
        return "expenses"
    if anyof("staff", "name"):
        return "staff"
    return "other"

def _clean_rows(df):
    """drop repeated header rows / note rows"""
    cols = [c for c in df.columns if not c.startswith("_")]
    if not cols:
        return df
    hdr_hits = np.zeros(len(df), dtype=int)
    for c in cols:
        hdr_hits += (df[c].astype(str).str.strip() == c).values
    keep = hdr_hits < max(2, len(cols) // 2)
    nn = df[cols].notna().sum(axis=1).values
    # free-text note rows ('Ghi chú: số liệu ghi tay') may sit in any column
    note = np.zeros(len(df), dtype=bool)
    for c in cols:
        note |= np.array([isinstance(v, str) and v.strip().lower().startswith(("ghi chú:", "ghi chu:", "note:", "lưu ý:", "luu y:", "tổng cộng", "tong cong")) for v in df[c].values])
    keep &= ~((nn <= 2) & note)
    # embedded subtotal rows: a cell 'TỔNG NGÀY ..', 'Tổng cộng', 'Cộng tháng' with most other cells empty
    sub = np.zeros(len(df), dtype=bool)
    for c in cols:
        v = df[c]
        if v.dtype == object or str(v.dtype).startswith("str"):
            k = v.astype(str).map(nkey)
            sub |= k.str.match(r"^(tong cong|tong ngay|tong thang|tong ky|tong so|tong tien|cong thang|cong ngay|cong ky|total|subtotal|sum|tong|cong)( [0-9 ]*)?$").values & v.notna().values
    keep &= ~(sub & (nn <= max(3, len(cols) // 2)))
    # rows where only one cell has a note-like long text
    return df[keep].reset_index(drop=True)

def apply_status_map(M, mapping):
    """re-label order statuses with a mapping raw_label -> done|cancel|return|shipping (e.g. proposed by an LLM)"""
    o = M.orders
    if "status_raw" not in o.columns:
        return
    from .norm import nkey as _nk
    mp = {_nk(k): v for k, v in mapping.items() if v in ("done", "cancel", "return", "shipping")}
    new = o["status_raw"].map(lambda v: mp.get(_nk(v)) if isinstance(v, str) else None)
    o["status"] = new.where(new.notna(), o["status"])
    o.loc[o["order_id"].isin(M.return_evidence) & (o["status"] != "cancel"), "status"] = "return"
    raw_present = o["status_raw"].notna()
    known = o["status_raw"].map(lambda v: status_of(v) is not None or (_nk(v) in mp if isinstance(v, str) else False))
    M.unknown_status_share = float((raw_present & ~known).mean())
    if M.lines is not None and len(M.lines):
        M.lines["status"] = M.lines["order_id"].map(o.set_index("order_id")["status"])

def unknown_status_labels(M):
    o = M.orders
    if "status_raw" not in o.columns:
        return []
    raw = o["status_raw"].dropna().astype(str)
    return sorted({v for v in raw.unique() if status_of(v) is None})

def build_model(bdir, readme_text=None):
    M = Model()
    M.bdir = bdir
    readme_text = readme_text if readme_text is not None else open(os.path.join(bdir, "README.md"), encoding="utf-8").read()
    info = parse_readme(readme_text)
    M.info = info
    raw = load_business(bdir)
    M.raw = raw
    M.tables = {}
    M.kinds = defaultdict(list)
    for name, df in raw.items():
        df = _clean_rows(df)
        fname = df["_file"].iloc[0] if len(df) else name
        sheet = df["_sheet"].iloc[0] if len(df) else None
        roles = assign_roles(df, info, fname, sheet or None)
        # contextual fixes
        if "name" in roles:
            if "cust_code" in roles and "cust_name" not in roles:
                roles["cust_name"] = roles.pop("name")
            elif "item_code" in roles and "item_name" not in roles:
                roles["item_name"] = roles.pop("name")
            elif "supplier" in roles:
                roles["supplier_name"] = roles.pop("name")
            elif "staff" in roles:
                roles["staff_name"] = roles.pop("name")
        if "cust_code" in roles and "item_name" in roles and "item_code" not in roles and "order_id" not in roles:
            roles["cust_group"] = roles.pop("item_name")  # 'Hạng' in customer lists
        for okc in info.get("order_key_cols", []):
            if okc in df.columns and "order_id" not in roles:
                for r_ in list(roles):
                    if okc in roles[r_]:
                        roles[r_] = [x for x in roles[r_] if x != okc]
                        if not roles[r_]:
                            del roles[r_]
                roles["order_id"] = [okc]
        kind = classify(roles, df)
        if kind == "orders" and "line_amount" in roles and "payable" not in roles:
            roles["payable"] = roles.pop("line_amount")
        M.tables[name] = (df, roles, kind)
        M.kinds[kind].append(name)
    _build(M)
    return M

# ------------------------------------------------------------------ helpers
def _num(series, money=True):
    f = parse_money if money else parse_qty
    return series.map(f).astype(float)

def _dates(series, df):
    yrs = [_year_from(f, s) for f, s in zip(df["_file"], df["_sheet"])]
    out = [parse_date(v, y) for v, y in zip(series.values, yrs)]
    return pd.to_datetime(pd.Series(out, index=series.index), errors="coerce")

def _col(df, roles, r):
    c = _first(roles, r)
    return df[c] if c is not None else None

def _build(M):
    info = M.info
    T = M.tables
    # ---------------- product catalog
    prod_rows = []
    code2name, name2code = {}, {}
    for kind in ("products", "stock", "drug_info", "units", "price_hist", "import_lines", "lots"):
        for n in M.kinds.get(kind, []):
            df, roles, _ = T[n]
            cc, nc = _first(roles, "item_code"), _first(roles, "item_name")
            if cc and nc:
                for c, nm in zip(df[cc], df[nc]):
                    if pd.notna(c) and pd.notna(nm):
                        code2name.setdefault(str(c).strip(), str(nm).strip())
                        name2code.setdefault(item_key(nm), str(c).strip())
    # abbreviations (e.g. sheet 'Ghi tắt')
    M.abbrev = {}
    M.item_type = {}
    for n in M.kinds.get("abbrev", []):
        df, roles, _ = T[n]
        ac, nc, tc = _first(roles, "abbrev"), _first(roles, "item_name"), _first(roles, "item_type")
        for _, r in df.iterrows():
            if pd.notna(r[ac]) and pd.notna(r[nc]):
                M.abbrev[item_key(r[ac])] = str(r[nc]).strip()
                M.abbrev.setdefault(item_key(r[nc]), str(r[nc]).strip())
                if tc and pd.notna(r[tc]):
                    M.item_type[item_key(r[nc])] = nkey(r[tc])
    M.code2name, M.name2code = code2name, name2code

    def resolve_item(code, name):
        """-> (key, display_name, code)"""
        code = str(code).strip() if code is not None and pd.notna(code) and str(code).strip() else None
        nm = str(name).strip() if name is not None and pd.notna(name) and str(name).strip() else None
        if nm and item_key(nm) in M.abbrev:
            nm = M.abbrev[item_key(nm)]
        if code is None and nm is not None:
            code = name2code.get(item_key(nm))
        if code is not None:
            disp = code2name.get(code, nm)
            return code, disp, code
        if nm is None:
            return None, None, None
        return "N:" + item_key(nm), nm, None
    M.resolve_item = resolve_item

    products = {}
    def prod(key, disp, code):
        p = products.get(key)
        if p is None:
            p = products[key] = {"key": key, "name": disp, "code": code, "names": Counter()}
        if disp:
            p["names"][disp] += 0
        return p

    for n in M.kinds.get("products", []) + M.kinds.get("stock", []):
        df, roles, _ = T[n]
        for _, r in df.iterrows():
            key, disp, code = resolve_item(r.get(_first(roles, "item_code")) if _first(roles, "item_code") else None,
                                           r.get(_first(roles, "item_name")) if _first(roles, "item_name") else None)
            if key is None:
                continue
            p = prod(key, disp, code)
            for role, fld, conv in [("item_group", "group", str), ("base_unit", "base_unit", str), ("unit", "base_unit", str),
                                    ("cost", "cost", parse_money), ("price", "price", parse_money), ("active", "active", str)]:
                c = _first(roles, role)
                if c and pd.notna(r[c]) and fld not in p:
                    p[fld] = conv(r[c]) if conv is not str else str(r[c]).strip()
            for c in df.columns:
                if c.startswith("_"):
                    continue
                k = nkey(c)
                if k in ("ten goi khac",) and pd.notna(r[c]):
                    p["alt_names"] = [x.strip() for x in str(r[c]).split(",") if x.strip()]
                if k in ("so size", "size", "so") and pd.notna(r[c]):
                    p["size"] = str(r[c]).strip()
                if k in ("hang", "thuong hieu") and "item_name" in roles and c not in roles.get("item_name", [])[:1] and pd.notna(r[c]):
                    p["brand"] = str(r[c]).strip()
                if k in ("quy doi",) and pd.notna(r[c]) and isinstance(r[c], str):
                    p["conv_text"] = r[c]
    for n in M.kinds.get("drug_info", []):
        df, roles, _ = T[n]
        for _, r in df.iterrows():
            key, disp, code = resolve_item(r.get(_first(roles, "item_code")) if _first(roles, "item_code") else None,
                                           r.get(_first(roles, "item_name")) if _first(roles, "item_name") else None)
            if key is None:
                continue
            p = prod(key, disp, code)
            if _first(roles, "ingredient"):
                v = r[_first(roles, "ingredient")]
                p["ingredient"] = str(v).strip() if pd.notna(v) else None
            if _first(roles, "rx"):
                v = str(r[_first(roles, "rx")]).strip().lower()
                p["rx"] = v in ("true", "1", "1.0", "x", "có", "co", "yes")
    M.products = products

    # ---------------- unit conversions: factor to base per (item_key, unit_key)
    conv = defaultdict(dict)    # key -> {unit: factor_to_base}
    base_unit = {}
    raw_conv = []  # (key, big, factor, small_or_None)
    for n in M.kinds.get("units", []):
        df, roles, _ = T[n]
        bu = _first(roles, "big_unit") or _first(roles, "unit")
        for _, r in df.iterrows():
            key, disp, code = resolve_item(r.get(_first(roles, "item_code")) if _first(roles, "item_code") else None,
                                           r.get(_first(roles, "item_name")) if _first(roles, "item_name") else None)
            if key is None or pd.isna(r[bu]):
                continue
            prod(key, disp, code)
            fv = r[_first(roles, "factor")]
            small = None
            if isinstance(fv, str):
                m = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*(.*)$", fv)
                f = float(m.group(1).replace(",", ".")) if m else np.nan
                small = m.group(2).strip() or None if m else None
            else:
                f = float(fv) if pd.notna(fv) else np.nan
            if _first(roles, "base_unit") and pd.notna(r[_first(roles, "base_unit")]):
                base_unit.setdefault(key, str(r[_first(roles, "base_unit")]).strip())
            raw_conv.append((key, str(r[bu]).strip(), f, small))
    for it, big, f, small in info["conversions"]:
        key, disp, code = resolve_item(None, it)
        if key is None:
            continue
        prod(key, disp, code)
        raw_conv.append((key, big, f, small))
        base_unit.setdefault(key, small)
    # products with 'Quy đổi' text e.g. "1 thùng = 4 bịch"
    for key, p in products.items():
        t = p.get("conv_text")
        if t:
            for m in re.finditer(r"1\s*([^=,;]+?)\s*=\s*(\d+(?:[.,]\d+)?)\s*([^,;]+)", t):
                raw_conv.append((key, m.group(1).strip(), float(m.group(2).replace(",", ".")), m.group(3).strip()))
    for key, p in products.items():
        if p.get("base_unit"):
            base_unit.setdefault(key, p["base_unit"])
    # resolve multi-level (small unit given = intermediate unit)
    per_item = defaultdict(list)
    for key, big, f, small in raw_conv:
        per_item[key].append((nkey(big), f, nkey(small) if small else None))
    for key, lst in per_item.items():
        bu = nkey(base_unit.get(key)) if base_unit.get(key) else None
        direct = {}
        for big, f, small in lst:
            if small is None or small == bu:
                direct[big] = f
        changed = True
        while changed:
            changed = False
            for big, f, small in lst:
                if big not in direct and small in direct:
                    direct[big] = f * direct[small]; changed = True
        for big, f, small in lst:
            direct.setdefault(big, f)
        if bu:
            direct[bu] = 1.0
        conv[key] = direct
    M.conv = conv
    M.base_unit = base_unit

    def factor(key, unit):
        if unit is None or (isinstance(unit, float) and np.isnan(unit)):
            return 1.0
        u = nkey(unit)
        d = conv.get(key, {})
        if u in d:
            return d[u]
        bu = base_unit.get(key)
        if bu is None or nkey(bu) == u:
            return 1.0
        return np.nan
    M.factor = factor

    # ---------------- orders table(s)
    orders = None
    for n in M.kinds.get("orders", []):
        df, roles, _ = T[n]
        o = pd.DataFrame({"order_id": df[_first(roles, "order_id")].astype(str).str.strip()})
        _fill_header(o, df, roles)
        orders = o if orders is None else pd.concat([orders, o], ignore_index=True)
    # ---------------- sales lines
    parts = []
    for n in M.kinds.get("sales_lines", []):
        df, roles, _ = T[n]
        parts.append(_lines_from(M, df, roles))
    lines = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    M.has_lines = len(lines) > 0 and lines["item_key"].notna().any()
    if orders is None and len(lines):
        hdr = [c for c in lines.columns if c in HEADER_ROLES + ["ts", "status_raw", "date"] and c != "order_id"]
        orders = lines.groupby("order_id", sort=False)[hdr].first().reset_index()
    elif orders is not None and len(lines):
        # bring header fields that exist only on lines
        extra = [c for c in lines.columns if c in HEADER_ROLES + ["ts", "status_raw"] and c not in orders.columns and c != "order_id"]
        if extra:
            orders = orders.merge(lines.groupby("order_id")[extra].first().reset_index(), on="order_id", how="left")
        missing_ids = set(lines["order_id"].dropna()) - set(orders["order_id"])
        if missing_ids:
            pass
    if orders is None:
        orders = pd.DataFrame(columns=["order_id"])
    orders = orders.drop_duplicates("order_id", keep="first").reset_index(drop=True)
    if len(orders) == 0:
        # no sales/order table recognised: keep an empty canonical model (rule handlers abstain);
        # the agent still sees every raw table through data_dict()
        M.orders = orders; M.lines = pd.DataFrame(columns=["order_id"]); M.has_lines = False
        M.payments = None; M.customers = {}; M.cust_by_code = False; M.tickets = None
        M.data_start = M.data_end = None
        _imports(M); _stock(M); _misc(M)
        return
    # ---------------- statuses
    if "status_raw" not in orders.columns and "note" in orders.columns:
        orders["status_raw"] = orders["note"]
    orders["status"] = orders["status_raw"].map(status_of) if "status_raw" in orders.columns else None
    if "status_raw" in orders.columns and "note" in orders.columns:
        miss = orders["status"].isna()
        orders.loc[miss, "status"] = orders.loc[miss, "note"].map(status_of)
    raw_present = orders["status_raw"].notna() & (orders["status_raw"].astype(str).str.strip() != "") if "status_raw" in orders.columns else pd.Series(False, index=orders.index)
    M.unknown_status_share = float((orders["status"].isna() & raw_present).mean()) if len(orders) else 0.0
    if "status_raw" in orders.columns and "note" not in (orders.columns if False else []):
        pass
    orders["status"] = orders["status"].fillna("done")
    # returns evidence: negative quantities, lot returns, returns table
    ret_ids = set()
    if len(lines) and "qty" in lines:
        ret_ids |= set(lines.loc[lines["qty"] < 0, "order_id"].dropna())
    for n in M.kinds.get("lot_sales", []):
        df, roles, _ = T[n]
        rc = _first(roles, "qty_return_base") or _first(roles, "qty_return_base2")
        if rc:
            q = _num(df[rc], money=False)
            oid = df[_first(roles, "order_id")].astype(str).str.strip()
            ret_ids |= set(oid[q > 0])
    for n in M.kinds.get("returns", []):
        df, roles, _ = T[n]
        ret_ids |= set(df[_first(roles, "order_id")].astype(str).str.strip())
    M.return_evidence = ret_ids
    orders.loc[orders["order_id"].isin(ret_ids) & (orders["status"] != "cancel"), "status"] = "return"
    # dates
    if "date" not in orders.columns or orders["date"].isna().all():
        if "ts" in orders.columns:
            orders["date"] = orders["ts"]
    orders["date"] = pd.to_datetime(orders.get("date"), errors="coerce").dt.normalize()
    if "ts" in orders.columns:
        orders["ts"] = pd.to_datetime(orders["ts"], errors="coerce")
        bad = orders["date"].isna() & orders["ts"].notna()
        orders.loc[bad, "date"] = orders.loc[bad, "ts"].dt.normalize()
    # debt written in order notes (e.g. 'đưa trước 87k, còn thiếu 73 nghìn', 'nợ cả đơn', 'ghi sổ, khách trả hết rồi')
    if "debt" not in orders.columns and "note" in orders.columns and orders["note"].notna().sum() > 5:
        tot = orders["payable"] if "payable" in orders.columns else orders.get("revenue")
        dn = [parse_debt_note(v, t) for v, t in zip(orders["note"], tot if tot is not None else [np.nan] * len(orders))]
        dn = pd.Series(dn, index=orders.index, dtype=float)
        if dn.notna().sum() > 5:
            orders["debt"] = dn.fillna(0)
    # customers
    _customers(M, orders)
    # revenue per order from lines
    if len(lines):
        lines = lines.merge(orders[["order_id", "status", "date"] + [c for c in ("ts",) if c in orders.columns]].rename(columns={"date": "o_date", "ts": "o_ts"}),
                            on="order_id", how="left")
        lines["date"] = lines["o_date"].where(lines["o_date"].notna(), lines.get("date"))
        if "o_ts" in lines:
            lines["ts"] = lines["o_ts"].where(lines["o_ts"].notna(), lines.get("ts"))
        rev = lines.loc[~lines["nonitem"], :].groupby("order_id")["amount"].sum()
        orders["revenue"] = orders["order_id"].map(rev)
        nonitem = lines.loc[lines["nonitem"]]
        if len(nonitem):
            for kind in ("voucher", "points", "ship"):
                s = nonitem.loc[nonitem["nonitem_kind"] == kind].groupby("order_id")["amount"].sum()
                orders["adj_" + kind] = orders["order_id"].map(s).fillna(0)
    if "revenue" not in orders.columns or orders["revenue"].isna().all():
        if "goods_total" in orders.columns:
            orders["revenue"] = orders["goods_total"]
        elif "payable" in orders.columns:
            orders["revenue"] = orders["payable"]
    M.orders = orders
    M.lines = lines
    M.line_consistency = 1.0
    if len(lines) and lines["price"].notna().any() and lines["qty"].notna().any():
        x = lines[~lines["nonitem"] & lines["price"].notna() & lines["qty"].notna() & lines["amount"].notna() & ~lines.get("is_topping", pd.Series(False, index=lines.index)).fillna(False).astype(bool)]
        if len(x) > 50:
            exp = x["qty"] * x["price"] - x["disc"].fillna(0)
            ok = (abs(x["amount"] - exp) <= 1) | (abs(x["amount"] - x["qty"] * x["price"]) <= 1) | (x["amount"] <= x["qty"] * x["price"] + 1)
            M.line_consistency = float(ok.mean())
    # ---------------- payments
    pays = []
    for n in M.kinds.get("payments", []):
        df, roles, _ = T[n]
        p = pd.DataFrame({"order_id": df[_first(roles, "order_id")].astype(str).str.strip(),
                          "amount": _num(df[_first(roles, "money")])})
        tc = _first(roles, "ts") or _first(roles, "date")
        p["ts"] = _dates(df[tc], df) if tc else pd.NaT
        if _first(roles, "paymethod"):
            p["method"] = df[_first(roles, "paymethod")].map(paymethod_of)
        pays.append(p)
    M.payments = pd.concat(pays, ignore_index=True) if pays else None
    # ---------------- imports
    _imports(M)
    _stock(M)
    _misc(M)

_AMT = r"([\d][\d.,]*\s*(?:k|nghin|ngan|tr|trieu|d)?)\b"

def _amt(t):
    t = t.strip().replace(" ", "")
    m = re.match(r"([\d.,]+)(k|nghin|ngan|tr|trieu|d)?$", t)
    if not m:
        return np.nan
    v, u = m.groups()
    base = parse_money(v)
    if u in ("k", "nghin", "ngan"):
        return float(v.replace(",", ".")) * 1000 if v.count(".") <= 1 and not re.search(r"\.\d{3}$", v) else base * 1000
    if u in ("tr", "trieu"):
        return float(v.replace(",", ".")) * 1e6
    return base

def parse_debt_note(note, total):
    """remaining debt stated in a free-text order note; NaN when the note says nothing about debt"""
    if not isinstance(note, str) or not note.strip():
        return np.nan
    from .norm import strip_acc
    k = re.sub(r"[^a-z0-9.,\s]", " ", strip_acc(note).lower())
    k = re.sub(r"\s+", " ", k).strip()
    if not re.search(r"\b(no|ghi so|thieu|chua tra|con lai)\b", k):
        return np.nan
    if re.search(r"\b(du roi|tra du|da tra du|tra het|het roi|ck du|thanh toan du)\b", k):
        return 0.0
    m = re.search(r"(?:con thieu|thieu lai|thieu|ghi no|ghi so|\bno)\s*" + _AMT, k)
    if m and re.search(r"\d", m.group(1)):
        return _amt(m.group(1))
    m = re.search(r"(?:dua truoc|tra truoc|ck|tra|dua)\s*" + _AMT + r".*con lai", k)
    if m and total == total and total is not None:
        return float(total) - _amt(m.group(1))
    if re.search(r"no ca don|chua tra", k) and total == total and total is not None:
        return float(total)
    return np.nan

def _fill_header(o, df, roles):
    for role in ["date", "ts", "cust_name", "cust_code", "paymethod", "channel", "staff", "payable", "paid", "debt",
                 "goods_total", "order_disc", "ship_fee", "note", "table", "area", "guests", "phone", "address", "cust_group", "disc"]:
        c = _first(roles, role)
        if c is None:
            continue
        v = df[c]
        if role in ("date", "ts"):
            o[role] = _dates(v, df).values
        elif role in ("payable", "paid", "debt", "goods_total", "order_disc", "ship_fee", "disc"):
            o["order_disc" if role == "disc" and "order_disc" not in roles else role] = _num(v).values
        elif role == "paymethod":
            o[role] = v.map(paymethod_of).values
        elif role == "channel":
            o[role] = v.map(channel_of).values
        elif role == "guests":
            o[role] = _num(v, False).values
        else:
            o[role] = v.values
    c = _first(roles, "status")
    if c:
        o["status_raw"] = df[c].values

def _lines_from(M, df, roles):
    df = df.copy()
    oc = _first(roles, "order_id")
    has_oid = oc is not None
    if oc is None:
        oc = "_rowid"; df[oc] = ["R%07d" % i for i in range(len(df))]; roles = dict(roles); roles["order_id"] = [oc]
    # forward fill group columns (rulebook §8)
    oid = df[oc].where(df[oc].notna() & (df[oc].astype(str).str.strip() != ""))
    has_item = pd.Series(False, index=df.index)
    for r in ("item_name", "item_code"):
        c = _first(roles, r)
        if c:
            has_item |= df[c].notna()
    if oid.isna().any() and oid.notna().any():
        df[oc] = oid.ffill()
    df[oc] = df[oc].astype(str).str.strip()
    grp_cols = [c for r in HEADER_ROLES + ["status", "ts"] for c in roles.get(r, [])[:1] if r != "order_id"]
    # only ffill within the order; never across orders
    for c in set(grp_cols):
        if c in df.columns and df[c].isna().any():
            df[c] = df.groupby(oc, sort=False)[c].ffill()
    # export-error duplicates: identical rows of the same order
    real = [c for c in df.columns if not c.startswith("_")]
    if has_oid:
        # compare cells by meaning, not spelling: '199.000đ' == '199,000' == 199000 ; ' Coca ' == 'coca'
        def _norm_cell(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return None
            if isinstance(v, (int, float, np.integer, np.floating)):
                return round(float(v), 6)
            if hasattr(v, "year"):
                return str(pd.Timestamp(v))
            t = str(v).strip()
            if re.fullmatch(r"-?[\d.,\s]+(đ|d|k|tr|vnd)?", t.lower()) and re.search(r"\d", t):
                m = parse_money(t)
                if m == m:
                    return round(float(m), 6)
            d = parse_date(t) if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|thg", t) else pd.NaT
            if d is not pd.NaT and not pd.isna(d):
                return str(d)
            return nkey(t)
        normed = pd.DataFrame({c: df[c].map(_norm_cell) for c in real}, index=df.index)
        df = df.loc[~normed.astype(str).duplicated(keep="first")]
    L = pd.DataFrame({"order_id": df[oc].values})
    ic, nc = _first(roles, "item_code"), _first(roles, "item_name")
    keys, disps, codes = [], [], []
    for c, nm in zip(df[ic] if ic else [None] * len(df), df[nc] if nc else [None] * len(df)):
        k, d, cd = M.resolve_item(c, nm)
        keys.append(k); disps.append(d); codes.append(cd)
    L["item_key"], L["item_name"], L["item_code"] = keys, disps, codes
    L["raw_name"] = df[nc].values if nc else None
    uc = _first(roles, "unit")
    qc = _first(roles, "qty")
    L["qty"] = _num(df[qc], False).values if qc else np.nan
    if uc:
        L["unit"] = df[uc].astype(str).str.strip().where(df[uc].notna(), None).values
    elif qc:
        L["unit"] = df[qc].map(qty_unit_suffix).values
    else:
        L["unit"] = None
    L["price"] = _num(df[_first(roles, "price")]).values if _first(roles, "price") else np.nan
    L["disc"] = _num(df[_first(roles, "disc")]).fillna(0).values if _first(roles, "disc") else 0.0
    if _first(roles, "line_amount"):
        L["amount"] = _num(df[_first(roles, "line_amount")]).values
    else:
        L["amount"] = L["qty"] * L["price"] - L["disc"]
    L["cost_unit"] = _num(df[_first(roles, "cost")]).values if _first(roles, "cost") else np.nan
    if _first(roles, "qty_base"):
        L["qty_base"] = _num(df[_first(roles, "qty_base")], False).values
    else:
        L["qty_base"] = [q * M.factor(k, u) if k is not None else np.nan for q, k, u in zip(L["qty"], L["item_key"], L["unit"])]
    if _first(roles, "line_no"):
        L["line_no"] = df[_first(roles, "line_no")].values
    for role in ("imei", "tradein", "item_group"):
        c = _first(roles, role)
        if c:
            L[role] = df[c].values if role != "tradein" else _num(df[c]).values
    # header info on lines
    o = pd.DataFrame(index=range(len(df)))
    _fill_header(o, df, roles)
    for c in o.columns:
        L[c] = o[c].values
    L["_sheet"] = df["_sheet"].values
    L["_file"] = df["_file"].values
    # non-item rows (voucher / points / ship / service rows without qty)
    nm = L["raw_name"].map(lambda x: nkey(x) if x is not None else "")
    kind = np.where(nm.str.startswith("voucher"), "voucher",
           np.where(nm.str.match(r"^dung \d+ diem"), "points",
           np.where(nm.str.match(r"^phi ship|^phi giao"), "ship", "")))
    L["nonitem_kind"] = kind
    L["nonitem"] = (kind != "")
    nm2 = L["item_name"].map(lambda x: nkey(x) if isinstance(x, str) else "")
    L["is_service_fee"] = nm.str.match(r"^phi phuc vu") | nm2.str.match(r"^phi phuc vu")
    # topping split "Món + 2 Trân châu (10.000)"
    if L["raw_name"].astype(str).str.contains(r" \+ ", regex=True).any() and not L["raw_name"].astype(str).str.contains(r"^\s*[^+]+\+[^+]+$").all():
        L = _split_toppings(M, L)
    return L

_top_re = re.compile(r"^\s*(?:(\d+)\s*(?:x\s*)?)?(.+?)\s*\(([^)]*)\)\s*$")

def _split_toppings(M, L):
    rows = []
    for r in L.to_dict("records"):
        nmv = r.get("raw_name")
        if not isinstance(nmv, str) or " + " not in nmv:
            rows.append(r); continue
        parts = [p.strip() for p in nmv.split(" + ")]
        base = parts[0]
        # only treat as topping split when every tail part has a bracketed price
        tails = []
        ok = True
        for p in parts[1:]:
            m = _top_re.match(p)
            if not m:
                ok = False; break
            tails.append((float(m.group(1)) if m.group(1) else 1.0, m.group(2).strip(), parse_money(m.group(3))))
        if not ok:
            rows.append(r); continue
        k, d, cd = M.resolve_item(None, base)
        tot_top = 0.0
        for n_, tname, tmoney in tails:
            cand = [tname, "Topping " + tname, "topping " + tname]
            tk = None
            for cnd in cand:
                kk, dd, cc = M.resolve_item(None, cnd)
                if kk is not None and (kk in M.products or not kk.startswith("N:")):
                    tk, td, tc = kk, dd, cc; break
            if tk is None:
                tk, td, tc = M.resolve_item(None, "Topping " + tname)
            tr = dict(r)
            # the number before a topping is the total number of portions on the line (not per drink)
            tr.update(item_key=tk, item_name=td, item_code=tc, raw_name=tname, qty=n_, qty_base=n_,
                      price=np.nan, disc=0.0, amount=tmoney if tmoney == tmoney else np.nan, cost_unit=np.nan, unit=None,
                      is_topping=True)
            tot_top += tr["amount"] if tr["amount"] == tr["amount"] else 0.0
            rows.append(tr)
        br = dict(r)
        br.update(item_key=k, item_name=d, item_code=cd, raw_name=base, amount=r["amount"] - tot_top)
        rows.append(br)
    return pd.DataFrame(rows)

def _customers(M, orders):
    T = M.tables
    cust = {}
    groups = {}
    for n in M.kinds.get("customers", []):
        df, roles, _ = T[n]
        cc, nc = _first(roles, "cust_code"), _first(roles, "cust_name")
        gc, ph, ad = _first(roles, "cust_group"), _first(roles, "phone"), _first(roles, "address")
        for _, r in df.iterrows():
            code = str(r[cc]).strip() if cc and pd.notna(r[cc]) else None
            name = str(r[nc]).strip() if nc and pd.notna(r[nc]) else None
            key = code or ("N:" + cust_key(name) if name else None)
            if not key:
                continue
            d = cust.setdefault(key, {"code": code, "name": name})
            if gc and pd.notna(r[gc]):
                d["group"] = str(r[gc]).strip()
            if ph and pd.notna(r[ph]):
                d["phone"] = str(r[ph]).strip()
            if ad and pd.notna(r[ad]):
                d["address"] = str(r[ad]).strip()
            for c in df.columns:
                k = nkey(c)
                if k in ("ngay dang ky", "ngay tao") and pd.notna(r[c]):
                    d["registered"] = parse_date(r[c])
    M.customers = cust
    has_codes = "cust_code" in orders.columns and orders["cust_code"].notna().any()
    M.cust_by_code = bool(has_codes)
    if has_codes:
        orders["cust_key"] = orders["cust_code"].where(orders["cust_code"].notna() & (orders["cust_code"].astype(str).str.strip() != ""), None)
        orders["cust_key"] = orders["cust_key"].map(lambda x: str(x).strip() if x is not None and pd.notna(x) else None)
        def disp(k, nm):
            if k is None:
                return None
            if k in cust and cust[k].get("name"):
                return cust[k]["name"]
            return nm if isinstance(nm, str) else k
        orders["cust_disp"] = [disp(k, nm) for k, nm in zip(orders["cust_key"], orders.get("cust_name", pd.Series([None] * len(orders))))]
    elif "cust_name" in orders.columns:
        keys = orders["cust_name"].map(lambda x: None if (x is None or (isinstance(x, float) and np.isnan(x)) or is_walkin(x)) else "N:" + cust_key(x))
        orders["cust_key"] = keys
        # display: most frequent spelling per key, prefer with diacritics
        best = {}
        for k, nm in zip(keys, orders["cust_name"]):
            if k is None:
                continue
            best.setdefault(k, Counter())[str(nm).strip()] += 1
        dmap = {k: sorted(c.items(), key=lambda kv: (-(kv[0] != unicodedata.normalize("NFC", kv[0]).encode("ascii", "ignore").decode()), -kv[1]))[0][0] for k, c in best.items()}
        orders["cust_disp"] = keys.map(lambda k: dmap.get(k) if k else None)
    else:
        orders["cust_key"] = None
        orders["cust_disp"] = None
    # customer group
    if "cust_group" not in orders.columns or orders["cust_group"].isna().all():
        orders["cust_group"] = orders["cust_key"].map(lambda k: cust.get(k, {}).get("group") if k else None)

def _imports(M):
    T = M.tables
    hdr = {}
    for n in M.kinds.get("import_headers", []):
        df, roles, _ = T[n]
        rc = _first(roles, "receipt_id")
        for _, r in df.iterrows():
            d = hdr.setdefault(str(r[rc]).strip(), {})
            if _first(roles, "date"):
                d["date"] = parse_date(r[_first(roles, "date")])
            if _first(roles, "supplier"):
                d["supplier"] = r[_first(roles, "supplier")]
            sc = _first(roles, "import_status") or _first(roles, "status")
            if sc:
                d["status_raw"] = r[sc]
            if _first(roles, "payable"):
                d["total"] = parse_money(r[_first(roles, "payable")])
            if _first(roles, "note"):
                d["note"] = r[_first(roles, "note")]
    parts = []
    for n in M.kinds.get("import_lines", []):
        df, roles, _ = T[n]
        rc = _first(roles, "receipt_id")
        I = pd.DataFrame({"receipt_id": df[rc].astype(str).str.strip().values})
        ic, nc = _first(roles, "item_code"), _first(roles, "item_name")
        res = [M.resolve_item(c, nm) for c, nm in zip(df[ic] if ic else [None] * len(df), df[nc] if nc else [None] * len(df))]
        I["item_key"] = [r[0] for r in res]; I["item_name"] = [r[1] for r in res]
        I["qty"] = _num(df[_first(roles, "qty")], False).values if _first(roles, "qty") else (_num(df[_first(roles, "qty_base")], False).values if _first(roles, "qty_base") else np.nan)
        I["unit"] = df[_first(roles, "unit")].values if _first(roles, "unit") else None
        I["price"] = _num(df[_first(roles, "price")]).values if _first(roles, "price") else np.nan
        I["amount"] = _num(df[_first(roles, "line_amount")]).values if _first(roles, "line_amount") else (I["qty"] * I["price"]).values
        I["date"] = _dates(df[_first(roles, "date")], df).values if _first(roles, "date") else pd.NaT
        I["supplier"] = df[_first(roles, "supplier")].values if _first(roles, "supplier") else None
        sc = _first(roles, "import_status") or _first(roles, "status")
        I["status_raw"] = df[sc].values if sc else None
        I["note"] = df[_first(roles, "note")].values if _first(roles, "note") else None
        I["total"] = _num(df[_first(roles, "payable")]).values if _first(roles, "payable") else np.nan
        if _first(roles, "lot"):
            I["lot"] = df[_first(roles, "lot")].astype(str).values
        if _first(roles, "expiry"):
            I["expiry"] = _dates(df[_first(roles, "expiry")], df).values
        if _first(roles, "qty_base") and _first(roles, "qty"):
            I["qty_base"] = _num(df[_first(roles, "qty_base")], False).values
        parts.append(I)
    I = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["receipt_id", "item_key", "item_name", "qty", "unit", "price", "amount", "date", "supplier", "status_raw", "note", "total"])
    for fld in ("date", "supplier", "status_raw", "total", "note"):
        if fld not in I.columns or I[fld].isna().all():
            I[fld] = I["receipt_id"].map(lambda r: hdr.get(r, {}).get(fld))
        else:
            fill = I["receipt_id"].map(lambda r: hdr.get(r, {}).get(fld))
            I[fld] = I[fld].where(I[fld].notna(), fill)
    I["date"] = pd.to_datetime(I["date"], errors="coerce").dt.normalize()
    I["cancelled"] = I["status_raw"].map(lambda v: bool(re.search(r"huy|cancel", nkey(v))) if v is not None else False)
    if "qty_base" not in I.columns:
        I["qty_base"] = [q * M.factor(k, u) if k is not None else np.nan for q, k, u in zip(I["qty"], I["item_key"], I["unit"])]
    if len(I) and I["amount"].isna().all():
        I["amount"] = I["qty"] * I["price"]
    # supplier display names
    sup_names = {}
    for n, (df, roles, kind) in M.tables.items():
        if "supplier" in roles and "supplier_name" in roles:
            for c, nm in zip(df[_first(roles, "supplier")], df[_first(roles, "supplier_name")]):
                sup_names[str(c).strip()] = str(nm).strip()
    M.sup_names = sup_names
    M.imports = I
    M.import_headers = hdr

def _stock(M):
    T = M.tables
    st = {}
    for n in M.kinds.get("stock", []):
        df, roles, _ = T[n]
        sc = _first(roles, "stock")
        for _, r in df.iterrows():
            key, disp, code = M.resolve_item(r.get(_first(roles, "item_code")) if _first(roles, "item_code") else None,
                                             r.get(_first(roles, "item_name")) if _first(roles, "item_name") else None)
            if key is not None:
                st[key] = parse_qty(r[sc])
    for key, p in M.products.items():
        pass
    M.stock_end = st
    M.ledger = None
    for n in M.kinds.get("ledger", []):
        df, roles, _ = T[n]
        L = pd.DataFrame({"item_key": [M.resolve_item(c, None)[0] for c in df[_first(roles, "item_code")]],
                          "ts": _dates(df[_first(roles, "ts")], df).values,
                          "stock": _num(df[_first(roles, "stock")], False).values})
        M.ledger = L

def _misc(M):
    T = M.tables
    # price history
    ph = []
    for n in M.kinds.get("price_hist", []):
        df, roles, _ = T[n]
        for _, r in df.iterrows():
            key, disp, code = M.resolve_item(r.get(_first(roles, "item_code")) if _first(roles, "item_code") else None,
                                             r.get(_first(roles, "item_name")) if _first(roles, "item_name") else None)
            ph.append((key, parse_date(r[_first(roles, "date")], _year_from(r["_file"], r["_sheet"])), parse_money(r[_first(roles, "price")])))
    M.price_hist = pd.DataFrame(ph, columns=["item_key", "date", "price"]) if ph else None
    # lots
    lots = []
    for n in M.kinds.get("lots", []):
        df, roles, _ = T[n]
        for _, r in df.iterrows():
            key, disp, code = M.resolve_item(r.get(_first(roles, "item_code")) if _first(roles, "item_code") else None,
                                             r.get(_first(roles, "item_name")) if _first(roles, "item_name") else None)
            lots.append({"lot": str(r[_first(roles, "lot")]).strip(), "item_key": key,
                         "expiry": parse_date(r[_first(roles, "expiry")]) if _first(roles, "expiry") else pd.NaT,
                         "qty_in": parse_qty(r[_first(roles, "qty_base")]) if _first(roles, "qty_base") else np.nan,
                         "price": parse_money(r[_first(roles, "price")]) if _first(roles, "price") else np.nan,
                         "receipt_id": str(r[_first(roles, "receipt_id")]).strip() if _first(roles, "receipt_id") else None})
    M.lots = pd.DataFrame(lots) if lots else None
    ls = []
    for n in M.kinds.get("lot_sales", []):
        df, roles, _ = T[n]
        q = _first(roles, "qty_base") or _first(roles, "qty")
        rc = _first(roles, "qty_return_base") or _first(roles, "qty_return_base2")
        ls.append(pd.DataFrame({"order_id": df[_first(roles, "order_id")].astype(str).str.strip().values,
                                "lot": df[_first(roles, "lot")].astype(str).str.strip().values,
                                "qty": _num(df[q], False).values if q else np.nan,
                                "qty_ret": _num(df[rc], False).fillna(0).values if rc else 0.0}))
    M.lot_sales = pd.concat(ls, ignore_index=True) if ls else None
    # deliveries
    dl = []
    for n in M.kinds.get("deliveries", []):
        df, roles, _ = T[n]
        D = pd.DataFrame({"order_id": df[_first(roles, "order_id")].astype(str).str.strip().values,
                          "date": _dates(df[_first(roles, "date")], df).values,
                          "batch": _num(df[_first(roles, "delivery_no")], False).values if _first(roles, "delivery_no") else np.nan})
        res = [M.resolve_item(None, nm) for nm in df[_first(roles, "item_name")]] if _first(roles, "item_name") else [(None, None, None)] * len(df)
        D["item_key"] = [r[0] for r in res]
        qc = _first(roles, "qty_base") or _first(roles, "qty")
        D["qty_base"] = _num(df[qc], False).values if qc else np.nan
        D["plate"] = df[_first(roles, "plate")].values if _first(roles, "plate") else None
        D["driver"] = df[_first(roles, "driver")].values if _first(roles, "driver") else None
        dl.append(D)
    M.deliveries = pd.concat(dl, ignore_index=True) if dl else None
    # debt terms
    dt_ = []
    for n in M.kinds.get("debt_terms", []):
        df, roles, _ = T[n]
        dt_.append(pd.DataFrame({"order_id": df[_first(roles, "order_id")].astype(str).str.strip().values,
                                 "due": _dates(df[_first(roles, "due_date")], df).values if _first(roles, "due_date") else pd.NaT,
                                 "days": _num(df[_first(roles, "terms_days")], False).values if _first(roles, "terms_days") else np.nan}))
    M.debt_terms = pd.concat(dt_, ignore_index=True) if dt_ else None
    # expenses
    ex = []
    for n in M.kinds.get("expenses", []):
        df, roles, _ = T[n]
        mc = _first(roles, "money") or _first(roles, "payable")
        if not mc:
            continue
        E = pd.DataFrame({"amount": _num(df[mc]).values,
                          "date": _dates(df[_first(roles, "date")], df).values if _first(roles, "date") else pd.NaT})
        cat = _first(roles, "expense_cat") or _first(roles, "item_group") or _first(roles, "cust_group")
        E["cat"] = df[cat].values if cat else None
        ex.append(E)
    M.expenses = pd.concat(ex, ignore_index=True) if ex else None
    # staff names
    M.staff_names = {}
    for n, (df, roles, kind) in T.items():
        if "staff" in roles and ("staff_name" in roles or "name" in roles):
            nc = _first(roles, "staff_name") or _first(roles, "name")
            for c_, nm in zip(df[_first(roles, "staff")], df[nc]):
                if pd.notna(c_) and pd.notna(nm):
                    M.staff_names[str(c_).strip()] = str(nm).strip()
    # shifts
    sh = []
    for n in M.kinds.get("shifts", []):
        df, roles, _ = T[n]
        sh.append(df)
    M.shifts = sh
    # data range
    d = M.orders["date"] if len(M.orders) else pd.Series(dtype="datetime64[ns]")
    M.data_start = d.min() if len(d) else None
    M.data_end = M.info.get("data_end") or (d.max() if len(d) else None)
