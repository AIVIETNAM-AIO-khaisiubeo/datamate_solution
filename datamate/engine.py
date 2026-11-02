"""Deterministic question engine: intent+slot parsing and rulebook-faithful execution."""
import re, math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from .norm import nkey, eval_norm, paymethod_of, channel_of, parse_money, strip_acc
from .qtext import qnorm
from .timeparse import parse_periods, Period, month_period, month_end, lunar_tet, _months_between, prev_month_period

class Unanswerable(Exception):
    pass

class Abstain(Exception):
    pass

def r2(x):
    """round half up to 2 decimals (percent)"""
    return float(np.floor(x * 100 + 0.5 + 1e-9) / 100) if x >= 0 else -float(np.floor(-x * 100 + 0.5 + 1e-9) / 100)

def rmoney(x):
    return int(np.floor(x + 0.5 + 1e-9)) if x >= 0 else -int(np.floor(-x + 0.5 + 1e-9))

def pct(a, b):
    if b == 0 or b is None or (isinstance(b, float) and np.isnan(b)):
        raise Abstain("div0")
    return r2((a - b) / b * 100)

def num_out(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    return x

# ------------------------------------------------------------------ vocabulary
SUBJECTIVE = r"\b(chat luong|ba bau|co thai|dung duoc khong|uong duoc khong|an toan khong|lanh nghe|tay nghe|tan tam|le phep|chu dao|khen|quy men|than thien|de thuong|dep trai|xinh|nen uong|lieu dung|cach dung|khach nu|khach nam|gioi tinh|giao tre|vi loi|thoi tiet|ngon nhat|dang tin|uy tin|nhiet tinh)\b"
NUM_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}

def parse_amount(txt):
    """'55.000 dong' '10 cu' '1,8 trieu' '40.000.000' -> float"""
    t = txt.strip()
    m = re.match(r"([\d.,]+)\s*(trieu|tr|cu|cu|k|nghin|ngan)?", t)
    if not m:
        return None
    v, u = m.groups()
    if u in ("trieu", "tr", "cu"):
        return float(v.replace(",", ".")) * 1e6 if v.count(".") <= 1 and "," not in v[-3:] or u else None
    if u in ("k", "nghin", "ngan"):
        return float(v.replace(",", ".")) * 1e3
    return parse_money(v)

def _pct_names(M):
    names = set()
    for p in M.products.values():
        if p.get("name") and "%" in p["name"]:
            names.add(p["name"])
    if M.lines is not None and len(M.lines):
        for n in M.lines["item_name"].dropna().unique():
            if isinstance(n, str) and "%" in n:
                names.add(n)
    M._pct_names = sorted(names, key=len, reverse=True)
    return M._pct_names

import os as _os
try:
    GENERIC = set(open(_os.path.join(_os.path.dirname(__file__), "generic_words.txt"), encoding="utf-8").read().split())
except Exception:
    GENERIC = set()
GENERIC |= set("""ban chay nhat ngay thang nam quy tuan hom qua nay truoc roi doanh thu tien don hang khach mat san pham nhom
bao nhieu may tong cong duoc cua hang tiem quan shop minh vay nhi ha a z ne nhe the do oi gium giup xem cho hoi biet coi check
la co khong cac nhung moi mot hai ba bon nam sau bay tam chin muoi tu den het toi va voi so sanh hon kem tang giam phan tram
gi nao ai dau trong ngoai tren duoi it nhieu cao thap lon nho dong trieu cu k nghin ngan d vnd xin ad chi em anh ban oi
ly chai lon goi hop cai cay bao bich vi vien tuyp lo ong kg thung loc binh cuon tam m2 m3 met m dia to phan noi con ket qua cuc bo can set mieng chiec lan tan xe
size tat ca cong cac mon bac si ke don hoat chat chuyen khoan dien tu tien mat cod zalo oa san tmdt thuong mai qua quay ghi no tra bang
ca sang chieu nhan vien ma thu ngan khach le si than thiet vip tong so luong dau tuan cuoi gan nhat lien tiep lien nhau ket thuc
tang gia len gia doi gia giam gia phi ship voucher diem thanh vien nhap kho ton lai gop rong chi phi
dip tet am lich nguyen dan loai do uong an mon nuoc hang hoa tinh tat moi deu gom gop ke ca nhung""".split())

def unlinked_tokens(c, extra_names=()):
    """content tokens of the question that are neither generic vocabulary nor part of a linked entity/time expression"""
    k = " " + c.k2 + " "
    for nk, key in _item_index(c.M):
        if " " + nk + " " in k:
            k = k.replace(" " + nk + " ", " ")
    for g in item_groups(c.M):
        k = k.replace(" " + nkey(g) + " ", " ")
    names = list(extra_names)
    for p in c.M.products.values():
        if p.get("ingredient"):
            names += [p["ingredient"]] + [x for x in re.split(r"\s*\+\s*", p["ingredient"])]
    names += list(getattr(c.M, "staff_names", {}).values()) + list(getattr(c.M, "sup_names", {}).values())
    for d in c.M.customers.values():
        if d.get("name"):
            names.append(re.sub(r"[\s\-–(]*\+?\d[\d .]{6,}\)?\s*$", "", d["name"]))
    o = c.M.orders
    if "cust_disp" in o.columns and len(o):
        names += [str(x) for x in o["cust_disp"].dropna().unique()[:5000]]
    for n in sorted(set(names), key=len, reverse=True):
        nk = nkey(n)
        if nk and " " + nk + " " in k:
            k = k.replace(" " + nk + " ", " ")
    toks = [t for t in k.split() if t not in GENERIC and not re.fullmatch(r"\d+[a-z]{0,3}|[a-z]\d+|nv\d+", t) and len(t) > 1]
    return toks

class Ctx:
    def __init__(self, M, question, as_of, cal=None):
        self.M = M
        self.q = question
        qq = question
        if "%" in qq:
            for pn in getattr(M, "_pct_names", None) or _pct_names(M):
                i = strip_acc(qq).lower().find(strip_acc(pn).lower())
                if i >= 0:
                    qq = qq[:i] + qq[i:i + len(pn)].replace("%", "") + qq[i + len(pn):]
        self.s = qnorm(qq)
        self.k = nkey(question)
        self.k2 = nkey(self.s)
        self.today = pd.Timestamp(as_of).normalize()
        self.cal = cal
        self.periods = parse_periods(self.s, self.today, cal)

    def has(self, pat):
        return re.search(pat, self.s) is not None

# ------------------------------------------------------------------ data access
def model_safety(M):
    """raise Abstain when the canonical model is not trustworthy for order-level computations"""
    if getattr(M, "unknown_status_share", 0) > 0.03:
        raise Abstain("unrecognised order status labels")
    o_ = M.orders
    if len(o_) and "date" in o_ and o_["date"].isna().mean() > 0.05:
        raise Abstain("order dates could not be parsed")
    if getattr(M, "line_consistency", 1.0) < 0.6:
        raise Abstain("line amounts inconsistent with qty x price")

def O(c):
    model_safety(c.M)
    return c.M.orders

def valid_mask(o):
    return ~o["status"].isin(["cancel", "return"])

def clip(c, p):
    """records after today do not exist yet"""
    end = min(p.end, c.today)
    return p.start, end

def check_coverage(c, p):
    M = c.M
    ds, de = M.data_start, M.data_end
    if ds is None:
        raise Unanswerable("no data")
    a, b = clip(c, p)
    if b < ds or a > de or a > b:
        raise Unanswerable("period outside data")

def orders_in(c, p, valid=True, check=True):
    model_safety(c.M)
    if check:
        check_coverage(c, p)
    a, b = clip(c, p)
    o = O(c)
    m = (o["date"] >= a) & (o["date"] <= b)
    if valid:
        m &= valid_mask(o)
    return o[m]

def lines_of(c, orders, items_only=True):
    L = c.M.lines
    if L is None or len(L) == 0:
        raise Unanswerable("no line data")
    x = L[L["order_id"].isin(set(orders["order_id"]))]
    if items_only:
        x = x[~x["nonitem"] & ~x["is_service_fee"].fillna(False) & x["item_key"].notna()]
    return x

# ------------------------------------------------------------------ entity linking
def _item_index(M):
    if hasattr(M, "_item_idx"):
        return M._item_idx
    idx = []
    names = defaultdict(set)
    for key, p in M.products.items():
        if p.get("name"):
            names[key].add(p["name"])
        for a in p.get("alt_names", []) or []:
            names[key].add(a)
    if M.lines is not None and len(M.lines):
        for key, nm in M.lines[["item_key", "item_name"]].drop_duplicates().itertuples(index=False):
            if key is not None and isinstance(nm, str):
                names[key].add(nm)
        if "raw_name" in M.lines:
            for key, nm in M.lines[["item_key", "raw_name"]].drop_duplicates().itertuples(index=False):
                if key is not None and isinstance(nm, str):
                    names[key].add(nm)
    if M.imports is not None and len(M.imports):
        for key, nm in M.imports[["item_key", "item_name"]].drop_duplicates().itertuples(index=False):
            if key is not None and isinstance(nm, str):
                names[key].add(nm)
    for ab, full in M.abbrev.items():
        k, d, cd = M.resolve_item(None, full)
        if k:
            names[k].add(full)
    for key, ns in names.items():
        for n in ns:
            nk = nkey(n)
            if len(nk) >= 3:
                idx.append((nk, key))
    idx.sort(key=lambda t: -len(t[0]))
    M._item_idx = idx
    return idx

def display_name(M, key):
    p = M.products.get(key)
    if p and p.get("name"):
        return p["name"]
    if M.lines is not None and len(M.lines):
        x = M.lines.loc[M.lines["item_key"] == key, "item_name"]
        if len(x):
            return x.mode().iloc[0]
    if isinstance(key, str) and key.startswith("N:"):
        return key[2:]
    return key

def find_items(c, max_n=3):
    r = _find_items(c, c.k, max_n)
    if not r:
        r = _find_items(c, c.k2, max_n)
    if not r:
        r = _fuzzy_items(c)
    return r

def _fuzzy_items(c):
    """one item whose name is within a small edit distance of a question span (typos like 'coc sac' for 'cu sac')"""
    from rapidfuzz.distance import Levenshtein
    toks = c.k2.split()
    best = []
    for nk, key in _item_index(c.M):
        n = len(nk.split())
        if n < 2:
            continue
        for i in range(0, len(toks) - n + 1):
            sp = " ".join(toks[i:i + n])
            d = Levenshtein.distance(sp, nk)
            if d <= max(1, int(len(nk) * 0.12)):
                best.append((d, -len(nk), key))
    if not best:
        return []
    best.sort()
    keys = {k for d, l, k in best if (d, l) == best[0][:2]}
    return [best[0][2]] if len(keys) == 1 else []

def _find_items(c, kk, max_n=3):
    """longest non-overlapping product-name matches in the question"""
    k = " " + kk + " "
    hits = []
    taken = []
    for nk, key in _item_index(c.M):
        i = k.find(" " + nk + " ")
        if i < 0:
            continue
        a, b = i, i + len(nk)
        if any(a < y and b > x for x, y in taken):
            continue
        taken.append((a, b)); hits.append((a, key))
    hits.sort()
    out = []
    for _, key in hits:
        if key not in out:
            out.append(key)
    return out[:max_n]

def item_groups(M):
    gs = set()
    for p in M.products.values():
        if p.get("group"):
            gs.add(p["group"])
    if M.lines is not None and "item_group" in M.lines:
        gs |= set(M.lines["item_group"].dropna().astype(str))
    return gs

def find_group(c):
    r = _find_group(c, c.k)
    return r if r else _find_group(c, c.k2)

def _find_group(c, kk):
    groups = sorted(item_groups(c.M), key=lambda g: -len(nkey(g)))
    k = " " + kk + " "
    for g in groups:
        gk = nkey(g)
        if gk and (" nhom " + gk + " " in k or " nhom hang " + gk + " " in k or " nhom mon " + gk + " " in k or " loai " + gk + " " in k
                   or " danh muc " + gk + " " in k):
            return g
    for g in groups:
        gk = nkey(g)
        if gk and len(gk) > 3 and " " + gk + " " in k and re.search(r"\bnhom\b", c.s):
            return g
    return None

def group_of(M, key):
    p = M.products.get(key)
    if p and p.get("group"):
        return p["group"]
    return None

def line_groups(c, L):
    if "item_group" in L.columns and L["item_group"].notna().any():
        g = L["item_group"]
        g2 = L["item_key"].map(lambda k: group_of(c.M, k))
        return g.where(g.notna(), g2)
    return L["item_key"].map(lambda k: group_of(c.M, k))

PAY_WORDS = [("cash", r"tien mat|\btm\b|tien tuoi"), ("transfer", r"chuyen khoan"), ("ewallet", r"vi dien tu|vi\b"), ("cod", r"\bcod\b|thu ho|xi o di|ship cod"), ("debt", r"ghi no|mua chiu|ban chiu")]
CHAN_WORDS = [("zalo", r"zalo"), ("phone", r"dien thoai|goi dien|qua phone"), ("store", r"tai quay|tai cua hang|tai tiem|tai quan|truc tiep|tai cho"), ("ecommerce", r"san tmdt|san thuong mai|thuong mai dien tu|tmdt|shopee|lazada")]

def find_paymethods(c):
    out = []
    for m in re.finditer(r"(tra|thanh toan|tt|bang|qua)\s*(?:bang\s*)?(tien mat|chuyen khoan|vi dien tu|cod|xi o di|ghi no|tm|tien tuoi|ck)\b", c.s):
        pm = {"tm": "cash", "tien tuoi": "cash", "ck": "transfer", "xi o di": "cod"}.get(m.group(2)) or paymethod_of(m.group(2))
        if pm not in out:
            out.append(pm)
    if not out:
        for pm, pat in PAY_WORDS[:4]:
            if re.search(r"(don|tra|thanh toan)[^.]{0,20}(" + pat + ")", c.s):
                out.append(pm)
    return out

def find_channels(c):
    out = []
    pos = []
    for ch, pat in CHAN_WORDS:
        m = re.search(pat, c.s)
        if m and re.search(r"(ban|dat|kenh|don|qua|tai|nguon)", c.s):
            pos.append((m.start(), ch))
    pos.sort()
    return [ch for _, ch in pos]

def find_cust_group(c):
    m = re.search(r"khach\s*(?:hang\s*)?(?:nhom|hang|loai)?\s*(than thiet|si|le|vip|thuong)\b", c.s)
    if m:
        return m.group(1)
    m = re.search(r"nhom\s*(than thiet|si|vip)\b", c.s)
    return m.group(1) if m else None

def find_staff(c):
    m = re.search(r"\bnv\s*0*(\d+)\b", c.s)
    if m:
        return "NV" + m.group(1).zfill(3)
    # by staff name
    names = getattr(c.M, "staff_names", {})
    for code, nm in names.items():
        if nkey(nm) and nkey(nm) in c.k:
            return code
    return None

def find_customer(c):
    """customer mentioned by name -> cust_key"""
    o = O(c)
    if "cust_disp" not in o:
        return None
    cands = {}
    for key, nm in o[["cust_key", "cust_disp"]].dropna().drop_duplicates().itertuples(index=False):
        cands.setdefault(nkey(re.sub(r"[\s\-–]*\d[\d .]{6,}$", "", str(nm))), key)
    for key, d in c.M.customers.items():
        if d.get("name"):
            cands.setdefault(nkey(re.sub(r"[\s\-–]*\d[\d .]{6,}$", "", d["name"])), key)
    k = " " + c.k + " "
    best = None
    for nk, key in sorted(cands.items(), key=lambda t: -len(t[0])):
        if len(nk.split()) >= 2 and " " + nk + " " in k:
            return key
    return best

# ------------------------------------------------------------------ revenue computations
def revenue(o):
    return float(o["revenue"].fillna(0).sum())

def nonempty(df, what):
    """abstention rule: never answer from an empty selection"""
    if df is None or len(df) == 0:
        raise Abstain("empty selection: " + what)
    return df

def filtered_revenue(c, p, f):
    """revenue for period with filters dict -> float"""
    o = nonempty(orders_in(c, p), "orders in period")
    o = apply_order_filters(c, o, f)
    if f.get("items") or f.get("group") or f.get("rx") is not None or f.get("ingredient"):
        L = c.M.lines
        x = L[L["order_id"].isin(set(o["order_id"])) & ~L["nonitem"]]
        x = nonempty(apply_line_filters(c, x, f), "lines after item/group filter")
        return float(x["amount"].fillna(0).sum())
    nonempty(o, "orders after filters")
    return revenue(o)

def apply_order_filters(c, o, f):
    if f.get("paymethod"):
        if "paymethod" not in o or o["paymethod"].isna().all():
            raise Unanswerable("no paymethod")
        if f["paymethod"] not in set(o["paymethod"].dropna()):
            raise Unanswerable("paymethod absent")
        o = o[o["paymethod"] == f["paymethod"]]
    if f.get("channel"):
        if "channel" not in o or o["channel"].isna().all():
            raise Unanswerable("no channel")
        if f["channel"] not in set(o["channel"].dropna()):
            raise Unanswerable("channel absent")
        o = o[o["channel"] == f["channel"]]
    if f.get("cust_group"):
        if "cust_group" not in o or o["cust_group"].isna().all():
            raise Unanswerable("no cust group")
        g = o["cust_group"].map(lambda x: nkey(x) if isinstance(x, str) else None)
        o = o[g == f["cust_group"]]
    if f.get("staff"):
        if "staff" not in o or o["staff"].isna().all():
            raise Unanswerable("no staff")
        o = o[o["staff"].astype(str).str.upper() == f["staff"]]
    if f.get("shift"):
        o = shift_filter(c, o, f["shift"])
    if f.get("customer"):
        o = o[o["cust_key"] == f["customer"]]
    return o

def shift_filter(c, o, shift):
    if "ts" not in o or o["ts"].isna().all():
        raise Unanswerable("no time of day")
    t = o["ts"].dt.hour
    if shift == "sang":
        return o[(t >= 6) & (t < 14)]
    return o[(t >= 14) & (t < 22)]

def apply_line_filters(c, x, f):
    if f.get("items"):
        x = x[x["item_key"].isin(f["items"])]
    if f.get("group"):
        g = line_groups(c, x)
        x = x[g.map(lambda v: nkey(v) if isinstance(v, str) else "") == nkey(f["group"])]
    if f.get("rx") is not None:
        rx = x["item_key"].map(lambda k: bool(c.M.products.get(k, {}).get("rx")))
        if not any(p.get("rx") is not None for p in c.M.products.values()):
            raise Unanswerable("no rx info")
        x = x[rx == f["rx"]]
    if f.get("ingredient"):
        ing = x["item_key"].map(lambda k: nkey(c.M.products.get(k, {}).get("ingredient") or ""))
        x = x[ing == nkey(f["ingredient"])]
    return x

def find_ingredient(c):
    ings = set(p.get("ingredient") for p in c.M.products.values() if p.get("ingredient"))
    k = " " + c.k + " "
    for ing in sorted(ings, key=lambda s: -len(s)):
        ik = nkey(ing)
        if ik and " hoat chat " + ik + " " in k:
            return ing
    return None

def collect_filters(c, allow_items=True):
    f = {}
    if allow_items:
        its = find_items(c)
        if its:
            f["items"] = its
    g = find_group(c)
    if g:
        f["group"] = g
    else:
        def _strip_items(t):
            t = " " + t + " "
            for nk, key in _item_index(c.M):
                if " " + nk + " " in t:
                    t = t.replace(" " + nk + " ", " ")
            return t
        pat = r"(?<!khach )(?<!khach hang )(?<!thuoc )\bnhom (?:hang |mon |san pham )?([a-z]+)"
        m = re.search(pat, _strip_items(c.k2))
        if m and not re.search(pat, _strip_items(c.k)):
            m = None
        if m and find_cust_group(c) is None and m.group(1) not in ("khach", "tuoi", "nao", "gi", "nay", "do", "co", "la", "thi", "ma", "bao", "duoc", "tren", "sau", "truoc", "ban", "hang", "mon"):
            if not (getattr(c.M, "item_type", None) and m.group(1) in ("may", "phu")):
                raise Abstain("group mentioned but not linked: " + m.group(0))
    pm = find_paymethods(c)
    if len(pm) == 1:
        f["paymethod"] = pm[0]
    ch = find_channels(c)
    if len(ch) == 1 and re.search(r"(kenh|ban qua|dat qua|ban tai|don ban|qua zalo|qua dien thoai|tai quay|tmdt|san)", c.s):
        f["channel"] = ch[0]
    cg = find_cust_group(c)
    if cg:
        f["cust_group"] = {"le": "le", "si": "si", "than thiet": "than thiet", "vip": "vip"}.get(cg, cg)
    st = find_staff(c)
    if st:
        f["staff"] = st
    if c.has(r"\bca sang\b") and not c.has(r"ca chieu"):
        f["shift"] = "sang"
    elif c.has(r"\bca chieu\b") and not c.has(r"ca sang"):
        f["shift"] = "chieu"
    if c.has(r"(tru|khong tinh|ngoai|bo) thuoc ke don|khong ke don|khong phai thuoc ke don|hang khong ke don|khong can don|khong can ke don|khong theo don"):
        f["rx"] = False
    elif c.has(r"thuoc ke don|ke don"):
        f["rx"] = True
    ing = find_ingredient(c)
    if ing:
        f["ingredient"] = ing
    return f
