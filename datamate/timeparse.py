"""Resolve Vietnamese time expressions (normalised text) against an as-of date."""
import re
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Period:
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str            # day | week | month | mtd | quarter | year | range | lastN | ytd | tet | snapshot
    label: str = ""
    pos: int = 0
    meta: dict = field(default_factory=dict)

    def days(self):
        return (self.end - self.start).days + 1

def month_end(y, m):
    return (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)).normalize()

def _year(y):
    y = int(y)
    return y + 2000 if y < 100 else y

def _month_no_year(m, today):
    y = today.year
    if pd.Timestamp(y, m, 1) > today:
        y -= 1
    return y

def month_period(y, m, today, pos=0, label=""):
    return Period(pd.Timestamp(y, m, 1), month_end(y, m), "month", label or f"{y}-{m:02d}", pos, {"y": y, "m": m})

def lunar_tet(year, cal):
    """window 23 tháng Chạp (lunar year-1) .. mùng 10 tháng Giêng (lunar year)"""
    if cal is None:
        return None
    try:
        a = cal[(cal["lunar_year"] == year - 1) & (cal["lunar_month"] == 12) & (cal["lunar_day"] == 23) & (~cal["leap"])]["solar"].iloc[0]
        b = cal[(cal["lunar_year"] == year) & (cal["lunar_month"] == 1) & (cal["lunar_day"] == 10) & (~cal["leap"])]["solar"].iloc[0]
        return pd.Timestamp(a), pd.Timestamp(b)
    except Exception:
        return None

def load_calendar(path):
    try:
        c = pd.read_csv(path)
    except Exception:
        return None
    cols = {k.lower(): k for k in c.columns}
    def pick(*names):
        for n in names:
            for k, v in cols.items():
                if k == n:
                    return c[v]
        for n in names:
            for k, v in cols.items():
                if n in k:
                    return c[v]
        return None
    solar = pick("solar_date", "duong_lich", "date", "solar", "ngay_duong")
    ly, lm, ld = pick("lunar_year", "nam_am"), pick("lunar_month", "thang_am"), pick("lunar_day", "ngay_am")
    leap = pick("is_leap", "leap", "nhuan")
    if solar is None or ly is None:
        return None
    out = pd.DataFrame({"solar": pd.to_datetime(solar), "lunar_year": ly.astype(int), "lunar_month": lm.astype(int), "lunar_day": ld.astype(int)})
    out["leap"] = leap.astype(str).str.lower().isin(["true", "1", "yes"]) if leap is not None else False
    return out

def parse_periods(s, today, cal=None):
    """Return list[Period] in textual order. `s` is normalised (no accents) text."""
    today = pd.Timestamp(today).normalize()
    found = []
    used = []

    def add(p, a, b):
        for (x, y) in used:
            if a < y and b > x:
                return
        used.append((a, b)); p.pos = a; found.append(p)

    # explicit date ranges: tu dd/mm/yyyy den (het) dd/mm/yyyy
    for m in re.finditer(r"(?:tu|giai doan tu|trong giai doan tu|tu ngay)\s*(?:ngay\s*)?(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*(?:den|toi|-|–)\s*(?:het\s*)?(?:ngay\s*)?(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s):
        d1, m1, y1, d2, m2, y2 = m.groups()
        add(Period(pd.Timestamp(_year(y1), int(m1), int(d1)), pd.Timestamp(_year(y2), int(m2), int(d2)), "range"), m.start(), m.end())
    # month range: tu thang a/yyyy den thang b/yyyy
    for m in re.finditer(r"(?:(?:tu|giua)\s*)?thang\s*(\d{1,2})[/-](\d{2,4})\s*(?:den|toi|-)\s*(?:het\s*)?thang\s*(\d{1,2})[/-](\d{2,4})", s):
        m1, y1, m2, y2 = m.groups()
        p = Period(pd.Timestamp(_year(y1), int(m1), 1), month_end(_year(y2), int(m2)), "monthrange")
        p.meta["months"] = _months_between(_year(y1), int(m1), _year(y2), int(m2))
        add(p, m.start(), m.end())
    # explicit full dates dd/mm/yyyy
    for m in re.finditer(r"(?<![\d/])(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})(?![\d/])", s):
        d, mo, y = m.groups()
        try:
            day = pd.Timestamp(_year(y), int(mo), int(d))
        except ValueError:
            continue
        pre = s[max(0, m.start() - 25):m.start()]
        kind = "snapshot" if re.search(r"(den het|tinh den|toi het|cuoi ngay|den ngay|tinh toi|toi ngay|het ngay|den cuoi)\s*(ngay)?\s*$", pre) else "day"
        add(Period(day, day, kind), m.start(), m.end())
    # quarters
    for m in re.finditer(r"quy\s*([1-4]|i{1,3}|iv)\s*(?:/|nam|-)?\s*(\d{4}|\d{2})?", s):
        q = m.group(1); q = {"i": 1, "ii": 2, "iii": 3, "iv": 4}.get(q, q); q = int(q)
        if m.group(2):
            y = _year(m.group(2))
        else:
            y = today.year if pd.Timestamp(today.year, 3 * q - 2, 1) <= today else today.year - 1
        add(Period(pd.Timestamp(y, 3 * q - 2, 1), month_end(y, 3 * q), "quarter", f"Q{q}/{y}", meta={"y": y, "q": q}), m.start(), m.end())
    # months with year: thang m/yyyy, thang m nam yyyy, m/yyyy, m-yyyy, m/yy
    for m in re.finditer(r"(?:thang\s*)?(?<![\d/.])(\d{1,2})\s*(?:/|-|\s+nam\s+)\s*(\d{4}|\d{2})(?![\d/.])", s):
        mo, y = int(m.group(1)), m.group(2)
        if not 1 <= mo <= 12:
            continue
        txt = m.group(0)
        if len(y) == 2 and not txt.startswith("thang"):
            # "3/26" -> month 3/2026 (only when 2-digit year plausible)
            if not (20 <= int(y) <= 30):
                continue
        add(month_period(_year(y), mo, today), m.start(), m.end())
    # 'thang m' without year
    for m in re.finditer(r"thang\s*(\d{1,2})(?![\d/])(?!\s*(?:nam|/|-)\s*\d)", s):
        mo = int(m.group(1))
        if not 1 <= mo <= 12:
            continue
        # skip "3 thang" counts (number before 'thang') handled by regex order; skip 'thang 1 lan'
        add(month_period(_month_no_year(mo, today), mo, today), m.start(), m.end())
    # month written in words: thang ba, thang gieng, thang chap ...
    WM = {"mot": 1, "gieng": 1, "hai": 2, "ba": 3, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10,
          "muoi mot": 11, "muoi hai": 12, "chap": 12}
    for m in re.finditer(r"\bthang (muoi mot|muoi hai|gieng|chap|mot|hai|ba|bay|tam|chin|muoi)\b(?!\s*(?:nay|truoc|roi|ngay|thang|lien|sau|toi))", s):
        mo = WM[m.group(1)]
        if m.group(1) == "nam" and re.match(r"\s*\d{4}", s[m.end():]):
            continue
        add(month_period(_month_no_year(mo, today), mo, today), m.start(), m.end())
    # relative expressions
    rel = [
        (r"(?:tu\s*)?(?:dau|mung 1|ngay 1|ngay mung 1)\s*thang\s*(?:nay)?\s*(?:den|toi)\s*(?:het\s*)?(?:ngay\s*)?(?:hom nay|bua nay|nay|bay gio|gio)", "mtd"),
        (r"\bthang nay\s*(?:tinh\s*)?(?:den|toi)\s*(?:het\s*)?(?:hom nay|bua nay|nay|gio)", "mtd"),
        (r"(?:tu\s*)?dau\s*nam\s*(?:nay)?\s*(?:den|toi)?\s*(?:het\s*)?(?:hom nay|nay|bay gio|gio)?", "ytd"),
        (r"tu\s*(?:ngay\s*)?0?1/0?1(?![/\d])\s*(?:den|toi)?\s*(?:het\s*)?(?:hom nay|bua nay|nay|bay gio|gio)?", "ytd"),
        (r"(\d+)\s*ngay\s*(?:gan\s*(?:nhat|day)|qua|vua qua|cuoi|gan nay)", "lastN"),
        (r"\b(?:thang|th)\s*(?:nay|hien tai)\b", "this_month"),
        (r"\b(?:thang|th)\s*(?:truoc|roi|vua roi|vua qua)\b", "last_month"),
        (r"\btuan\s*(?:nay|hien tai)\b", "this_week"),
        (r"\btuan\s*(?:truoc|roi|vua roi|vua qua)\b", "last_week"),
        (r"\b(?:hom|bua|ngay)\s*(?:nay|ni)\b", "today"),
        (r"\b(?:hom|bua|ngay)\s*qua\b", "yesterday"),
        (r"\bnam\s*(?:nay)\b", "this_year"),
        (r"\bnam\s*(?:ngoai|truoc|roi)\b", "last_year"),
        (r"\bnam\s*(20\d\d)\b", "year"),
        (r"\btet\s*(?:am lich\s*|nguyen dan\s*)?(?:nam\s*)?(20\d\d)?", "tet"),
    ]
    for pat, kind in rel:
        for m in re.finditer(pat, s):
            a, b = m.start(), m.end()
            if kind == "mtd" or kind == "this_month":
                p = Period(today.replace(day=1), today, "mtd", "this_month")
            elif kind == "ytd":
                p = Period(pd.Timestamp(today.year, 1, 1), today, "ytd")
            elif kind == "lastN":
                n = int(m.group(1)); p = Period(today - pd.Timedelta(days=n - 1), today, "lastN", meta={"n": n})
            elif kind == "last_month":
                pm = today.replace(day=1) - pd.Timedelta(days=1)
                p = month_period(pm.year, pm.month, today); p.meta["relative"] = True
            elif kind == "this_week":
                p = Period(today - pd.Timedelta(days=today.weekday()), today, "week", "this_week")
            elif kind == "last_week":
                st = today - pd.Timedelta(days=today.weekday() + 7)
                p = Period(st, st + pd.Timedelta(days=6), "week", "last_week")
            elif kind == "today":
                p = Period(today, today, "day", "today")
            elif kind == "yesterday":
                d = today - pd.Timedelta(days=1); p = Period(d, d, "day", "yesterday")
            elif kind == "this_year":
                p = Period(pd.Timestamp(today.year, 1, 1), today, "ytd")
            elif kind == "last_year":
                p = Period(pd.Timestamp(today.year - 1, 1, 1), pd.Timestamp(today.year - 1, 12, 31), "year")
            elif kind == "year":
                y = int(m.group(1)); p = Period(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31), "year")
            elif kind == "tet":
                y = int(m.group(1)) if m.group(1) else None
                if y is None:
                    # nearest Tết not after today
                    y = today.year
                    w = lunar_tet(y, cal)
                    if w and w[0] > today:
                        y -= 1
                w = lunar_tet(y, cal)
                if not w:
                    continue
                p = Period(w[0], w[1], "tet", meta={"y": y})
            add(p, a, b)
    # snapshot phrases without explicit date: "den het hom nay", "cuoi ngay hom nay", "tinh den hom nay"
    found.sort(key=lambda p: p.pos)
    return found

def _months_between(y1, m1, y2, m2):
    out = []
    y, m = y1, m1
    while (y, m) <= (y2, m2):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1; y += 1
    return out

def prev_month_period(p, today):
    y, m = p.start.year, p.start.month
    pm = pd.Timestamp(y, m, 1) - pd.Timedelta(days=1)
    return month_period(pm.year, pm.month, today)
