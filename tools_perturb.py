"""Label-preserving perturbations used for leakage-free robustness tests on the sealed shops.
- perturb_question: diacritics removal, teencode, telex-style typos on function words, casing/punctuation noise
- perturb_shop: rewrites a shop folder (headers renamed + README updated, money/date formats changed,
  status labels replaced with unseen synonyms, columns shuffled, big files split)."""
import os, re, random, shutil, unicodedata, json, sqlite3
import numpy as np
import pandas as pd

def _strip(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")

TEEN = [("không", "ko"), ("được", "dc"), ("bao nhiêu", "bnhiu"), ("doanh thu", "dthu"), ("tháng", "th"), ("hôm nay", "hnay"),
        ("hôm qua", "hqua"), ("trước", "trc"), ("nhiều", "nhìu"), ("vậy", "z"), ("cửa hàng", "shop"), ("sản phẩm", "sp")]
TELEX = {"bán": "bans", "nhiêu": "nhieeu", "tháng": "thangs", "được": "dduwowcj", "hàng": "hangf", "tiền": "tieenf", "của": "cuar"}

def perturb_question(q, rnd):
    s = q
    if rnd.random() < 0.5:
        for a, b in TEEN:
            if a in s and rnd.random() < 0.5:
                s = s.replace(a, b)
    if rnd.random() < 0.3:
        words = [w for w in TELEX if w in s]
        if words:
            w = rnd.choice(words); s = s.replace(w, TELEX[w], 1)
    if rnd.random() < 0.4:
        s = _strip(s)
    if rnd.random() < 0.3:
        s = s.lower().rstrip("?.! ")
    if rnd.random() < 0.2:
        s = rnd.choice(["cho mình hỏi ", "ad ơi ", "check giúp ", "e hỏi xíu "]) + s
    return s

# ---------------------------------------------------------------- data perturbation
HEADER_ALT = {  # README description prefix -> new header names (realistic, chosen without regard to the engine's lists)
    "mã đơn": ["Số chứng từ bán", "Invoice No", "Mã phiếu xuất"],
    "ngày bán": ["Ngày phát sinh", "Sale date", "Ngày chứng từ"],
    "trạng thái đơn": ["Tình hình đơn", "Order status", "TT xử lý đơn"],
    "tên khách": ["Người mua hàng", "Customer", "Khách mua"],
    "tên hàng": ["Diễn giải hàng", "Item name", "Mô tả mặt hàng"],
    "số lượng": ["Lượng", "Qty", "SL xuất"],
    "đơn giá": ["Giá bán ra", "Unit price", "Giá/đv"],
    "thành tiền sau giảm": ["Tiền sau CK", "Net amount", "Thực thu dòng"],
    "đơn vị tính": ["ĐV", "UoM", "Đơn vị"],
    "số lượng đã đổi ra đơn vị cơ bản": ["SL gốc", "Base qty", "Quy đổi lẻ"],
    "mã hàng": ["Item code", "Mã SP hàng", "Code"],
}
STATUS_ALT = {"done": "Đã chốt sổ", "cancel": "Bỏ, không lấy", "return": "Hoàn đơn về kho", "shipping": "Chờ lấy hàng"}

def _fmt_money(v, rnd):
    try:
        x = float(v)
    except Exception:
        return v
    if x != x or x < 1000 or abs(x - round(x)) > 1e-9:
        return v
    x = int(round(x))
    k = rnd.random()
    if k < 0.33:
        return f"{x:,}".replace(",", ".") + "đ"
    if k < 0.66 and x % 1000 == 0:
        return f"{x // 1000}k"
    return f"{x:,}"

def _fmt_date(v, rnd):
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "year"):
        try:
            t = pd.Timestamp(v)
        except Exception:
            return v
        if t.hour or t.minute or t.second:
            return v
        return t.strftime("%d-%m-%Y") if rnd.random() < 0.5 else f"{t.day} thg {t.month} {t.year}"
    return v

def perturb_shop(src, dst, kinds, seed=0):
    """kinds: subset of {'headers','values','status','layout'}"""
    from datamate.loader import _read_xlsx, _read_csv
    from datamate.norm import status_of
    rnd = random.Random(seed)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    readme = open(os.path.join(src, "README.md"), encoding="utf-8").read()
    readme0 = readme
    chosen = {}
    for fn in sorted(os.listdir(src)):
        p = os.path.join(src, fn)
        if fn == "README.md":
            continue
        if fn.lower().endswith((".sqlite", ".db")):
            shutil.copy(p, os.path.join(dst, fn)); continue
        sheets = _read_xlsx(p) if fn.lower().endswith(".xlsx") else {None: _read_csv(p)}
        out = {}
        for sh, df in sheets.items():
            df = df.copy()
            if "headers" in kinds:
                ren = {}
                for line in re.finditer(r"-\s*\*\*(.+?)\*\*:\s*(.+)", readme0):
                    col, desc = line.group(1).strip(), line.group(2).strip()
                    for key, alts in HEADER_ALT.items():
                        if desc.lower().startswith(key) and col in df.columns and col not in ren:
                            new = chosen.setdefault(col, rnd.choice(alts))   # one new name per column for the whole shop
                            if new not in df.columns and new not in ren.values():
                                ren[col] = new
                df = df.rename(columns=ren)
                for a_, b_ in ren.items():
                    readme = readme.replace(f"**{a_}**", f"**{b_}**")
            if "values" in kinds:
                for c in df.columns:
                    col = df[c]
                    num = pd.to_numeric(col, errors="coerce")
                    serial_dates = num.notna().mean() > 0.9 and num.dropna().between(20000, 80000).mean() > 0.9 and (num.dropna() % 1 == 0).all() and num.dropna().nunique() < 2000
                    if num.notna().mean() > 0.9 and (num.dropna() >= 1000).mean() > 0.8 and not serial_dates:
                        df[c] = [_fmt_money(v, rnd) for v in col]
                    elif col.map(lambda v: hasattr(v, "year")).mean() > 0.9:
                        df[c] = [_fmt_date(v, rnd) for v in col]
            if "status" in kinds:
                for c in df.columns:
                    vals = df[c].dropna().astype(str)
                    if len(vals) and vals.map(status_of).notna().mean() > 0.95 and vals.nunique() < 40:
                        df[c] = df[c].map(lambda v: STATUS_ALT.get(status_of(v), v) if isinstance(v, str) else v)
            if "layout" in kinds:
                cols = list(df.columns); rnd.shuffle(cols); df = df[cols]
            out[sh] = df
        stem, ext = os.path.splitext(fn)
        if "layout" in kinds and ext.lower() == ".csv" and len(out[None]) > 20000:
            d = out[None]; h = len(d) // 2
            d.iloc[:h].to_csv(os.path.join(dst, f"{stem}_p1.csv"), index=False)
            d.iloc[h:].to_csv(os.path.join(dst, f"{stem}_p2.csv"), index=False)
            readme = readme.replace(f"`{fn}`", f"`{stem}_p1.csv`, `{stem}_p2.csv`")
            continue
        if ext.lower() == ".csv":
            out[None].to_csv(os.path.join(dst, fn), index=False)
        else:
            with pd.ExcelWriter(os.path.join(dst, fn)) as w:
                for sh, d in out.items():
                    d.to_excel(w, sheet_name=str(sh)[:31] or "Sheet1", index=False)
    open(os.path.join(dst, "README.md"), "w", encoding="utf-8").write(readme)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sealed = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "split.json")))["sealed"]
    kinds = set(sys.argv[2].split(","))
    for b in sealed:
        perturb_shop(os.path.join(sys.argv[1], "train", "data", b), os.path.join(sys.argv[3], b), kinds, seed=hash(b) % 1000)
        print("perturbed", b, kinds)
