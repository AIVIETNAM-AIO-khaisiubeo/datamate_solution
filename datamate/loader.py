"""Generic raw-data loader: csv / xlsx (all sheets) / sqlite -> dict of DataFrames.
No business-specific logic: header-row detection, 2-row merged headers, month-sheet
concatenation and file-split concatenation are all driven by structure only."""
import os, re, sqlite3, unicodedata, warnings
import pandas as pd
warnings.filterwarnings("ignore")

def nfc(s):
    return unicodedata.normalize("NFC", str(s)) if s is not None else s

def _read_csv(path):
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            with open(path, encoding=enc) as f:
                head = f.read(4096)
            sep = max([",", ";", "\t", "|"], key=head.count)
            return pd.read_csv(path, sep=sep, encoding=enc, dtype=str, keep_default_na=False, na_values=[""])
        except Exception:
            continue
    raise RuntimeError(path)

def _is_texty(v):
    return isinstance(v, str) and v.strip() != "" and not re.fullmatch(r"[\d.,\-/: ]+", v.strip())

def _fix_header(raw: pd.DataFrame):
    """raw: sheet read with header=None. Detect header row (and a 2-row merged header)."""
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return raw
    best, best_i = -1, 0
    for i in range(min(10, len(raw))):
        row = raw.iloc[i].tolist()
        tex = [str(v).strip() for v in row if _is_texty(v)]
        empties = sum(1 for v in row if v is None or (isinstance(v, float) and pd.isna(v)))
        # distinct labels: a merged title/group row repeats one value and must not win
        score = len(set(tex)) - 0.5 * empties - 0.5 * (len(tex) - len(set(tex)))
        if score > best + 0.5:
            best, best_i = score, i
    hdr = raw.iloc[best_i].tolist()
    data = raw.iloc[best_i + 1:]
    # 2-row header: next row also mostly text and the header row has gaps (merged group row)
    if len(data) > 1:
        nxt = data.iloc[0].tolist()
        n_txt = sum(_is_texty(v) for v in nxt)
        gaps = sum(1 for v in hdr if v is None or (isinstance(v, float) and pd.isna(v)))
        if n_txt >= max(2, 0.6 * len(nxt)) and (gaps > 0 or best_i > 0 and False):
            # group row on top -> use lower row, fill missing lower cells from upper
            new = []
            for up, lo in zip(hdr, nxt):
                lo_ok = isinstance(lo, str) and lo.strip()
                new.append(lo if lo_ok else up)
            hdr, data = new, data.iloc[1:]
    cols, seen = [], {}
    for j, c in enumerate(hdr):
        c = nfc(c).strip() if isinstance(c, str) and c.strip() else f"col_{j}"
        if c in seen:
            seen[c] += 1; c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    data = data.copy(); data.columns = cols
    # drop trailing total/note rows that are entirely empty in most columns
    return data.reset_index(drop=True)

def _col_idx(letters):
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1

def _merged_ranges(path):
    """sheet name -> list of (r0, c0, r1, c1) merged ranges, read straight from the xlsx XML (fast, no openpyxl load)"""
    import zipfile
    out = {}
    try:
        z = zipfile.ZipFile(path)
        wb = z.read("xl/workbook.xml").decode("utf8", "ignore")
        rels = z.read("xl/_rels/workbook.xml.rels").decode("utf8", "ignore")
        rid2target = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels)) | {a: b for b, a in re.findall(r'Target="([^"]+)"[^>]*Id="([^"]+)"', rels)}
        for name, rid in re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb):
            tgt = rid2target.get(rid, "")
            tgt = tgt.lstrip("/")
            tgt = tgt if tgt.startswith("xl/") else "xl/" + tgt
            x = z.read(tgt).decode("utf8", "ignore")
            rng = []
            for a, r0, b, r1 in re.findall(r'<mergeCell ref="([A-Z]+)(\d+):([A-Z]+)(\d+)"', x):
                rng.append((int(r0) - 1, _col_idx(a), int(r1) - 1, _col_idx(b)))
            if rng:
                import html
                out[nfc(html.unescape(name))] = rng
    except Exception:
        pass
    return out

def _unmerge(raw, ranges):
    """copy the top-left value of every merged block into all its cells (assumes the frame starts at A1)"""
    raw = raw.copy()
    for r0, c0, r1, c1 in ranges:
        if r0 < raw.shape[0] and c0 < raw.shape[1]:
            v = raw.iat[r0, c0]
            raw.iloc[r0:min(r1, raw.shape[0] - 1) + 1, c0:min(c1, raw.shape[1] - 1) + 1] = v
    return raw

def _read_xlsx(path):
    out = {}
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine="calamine")
    except Exception:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    merged = _merged_ranges(path)
    for sh, raw in sheets.items():
        if merged.get(nfc(sh)):
            raw = _unmerge(raw, merged[nfc(sh)])
        df = _fix_header(raw)
        if df is not None and not df.empty:
            out[nfc(sh)] = df
    return out

def _read_sqlite(path):
    out = {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    for (t,) in con.execute("select name from sqlite_master where type='table'"):
        out[t] = pd.read_sql_query(f'select * from "{t}"', con)
    con.close()
    return out

def _stem_group(name):
    """Group file-split names: X_1, X_2, X_2026, X_T1_2025, X_Thang01_2025 -> X"""
    s = re.sub(r"\.(xlsx|csv|sqlite|db)$", "", name, flags=re.I)
    s2 = re.sub(r"(_T(hang)?\d{1,2})?(_\d{4})$", "", s)
    s2 = re.sub(r"[_\- ]?(p|part|phan|phần)?[_\- ]?\d{1,2}$", "", s2, flags=re.I)
    s2 = re.sub(r"\s*\(\d+\)$", "", s2)          # 'file (2).xlsx'
    return s2

def load_business(bdir):
    """Return dict table_name -> DataFrame, with meta columns _file, _sheet."""
    tables = {}
    for fn in sorted(os.listdir(bdir)):
        p = os.path.join(bdir, fn); fnn = nfc(fn)
        low = fn.lower()
        if low.endswith(".csv"):
            df = _read_csv(p); df.columns = [nfc(c).strip() for c in df.columns]
            tables[(fnn, None)] = df
        elif low.endswith((".xlsx", ".xlsm", ".xls")):
            for sh, df in _read_xlsx(p).items():
                tables[(fnn, sh)] = df
        elif low.endswith((".sqlite", ".db", ".sqlite3")):
            for t, df in _read_sqlite(p).items():
                tables[(fnn, t)] = df
    # concatenate structurally-identical splits (same column set) within same stem group
    groups = {}
    for (fn, sh), df in tables.items():
        key = (_stem_group(fn), tuple(sorted(df.columns)))
        groups.setdefault(key, []).append((fn, sh, df))
    out = {}
    for (stem, cols), items in groups.items():
        # also merge month sheets of one workbook having identical columns
        parts = []
        for fn, sh, df in items:
            d = df.copy(); d["_file"] = fn; d["_sheet"] = sh if sh is not None else ""
            parts.append(d)
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        if len(items) == 1:
            fn, sh, _ = items[0]
            name = f"{fn}" + (f"::{sh}" if sh is not None else "")
        else:
            shs = {sh for _, sh, _ in items}; fns = {fn for fn, _, _ in items}
            name = (f"{stem}*" if len(fns) > 1 else list(fns)[0]) + (f"::{list(shs)[0]}" if len(shs) == 1 and None not in shs else ("::*" if None not in shs else ""))
        while name in out: name += "'"
        out[name] = df
    return out

if __name__ == "__main__":
    import sys
    for b in sys.argv[1:]:
        T = load_business(b)
        print("=====", b)
        for n, df in T.items():
            cols = [c for c in df.columns if not c.startswith("_")]
            print(f"  [{n}] {df.shape}  sheets={df['_sheet'].nunique()}")
            print("    cols:", cols)
            r = df.iloc[min(1, len(df)-1)]
            print("    row:", {c: r[c] for c in cols[:40]})
