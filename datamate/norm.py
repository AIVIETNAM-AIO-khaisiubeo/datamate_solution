"""Value normalisation helpers (text, money, quantity, date, status, payment)."""
import re, unicodedata, datetime as dt
import numpy as np
import pandas as pd

def strip_acc(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")

def nkey(s) -> str:
    """normalised key: no accents, lowercase, alnum + single spaces"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = "".join((" " + ch) if (i and ch.isupper() and s[i-1].islower()) else ch for i, ch in enumerate(s))  # camelCase split
    s = strip_acc(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()

def nfc_lower(s):
    return unicodedata.normalize("NFC", str(s)).lower().strip()

def eval_norm(s):
    """normalisation used by the official string matcher"""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------------------------------------------- money / numbers
_num_re = re.compile(r"-?\d[\d.,\s]*")

def parse_money(v):
    """'25.000đ' '25k' '4,000' '35,000.0000' '5tr520' '1,01tr' '78,361tr' '524 000' -> float"""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "-"):
        return np.nan
    neg = s.startswith("-") or s.startswith("(")
    s = s.replace("vnđ", "").replace("vnd", "").replace("đồng", "").replace("đ", "").replace("d", "").strip()
    s = s.strip("()-+ ")
    m = re.fullmatch(r"([\d.,\s]*\d)\s*(tr|triệu|trieu|m)\s*(\d*)", s)
    if m:  # 5tr520 => 5,520,000 ; 1,01tr => 1,010,000
        a, _, b = m.groups()
        a = a.replace(" ", "")
        if "," in a or "." in a:
            a = a.replace(",", ".")
            if a.count(".") > 1:
                a = a.replace(".", "", a.count(".") - 1)
            val = float(a) * 1e6
        else:
            val = float(a) * 1e6
            if b:
                val += float(b) * 10 ** (6 - len(b)) if len(b) <= 3 else float(b)
        return -val if neg else val
    m = re.fullmatch(r"([\d.,\s]*\d)\s*(k|nghìn|nghin|ngàn|ngan)", s)
    if m:
        a = _plain_number(m.group(1))
        return -a * 1000 if neg else a * 1000
    try:
        a = _plain_number(s)
    except Exception:
        return np.nan
    return -a if neg else a

def _plain_number(s):
    s = s.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"\d+", s):
        return float(s)
    # 35,000.0000 / 4,000 / 30.500 / 1.234.567 / 12,5
    if "," in s and "." in s:
        if s.rfind(",") < s.rfind("."):   # 35,000.0000
            return float(s.replace(",", ""))
        return float(s.replace(".", "").replace(",", "."))
    if "," in s:
        parts = s.split(",")
        if all(len(p) == 3 for p in parts[1:]):
            return float(s.replace(",", ""))
        return float(s.replace(",", "."))
    if "." in s:
        parts = s.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            return float(s.replace(".", ""))
        return float(s)
    return float(s)

def parse_qty(v):
    """'2 lon' -> 2 ; numeric passthrough"""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    m = re.match(r"^\s*(-?\d+(?:[.,]\d+)?)", s)
    if not m:
        return np.nan
    return float(m.group(1).replace(",", "."))

def qty_unit_suffix(v):
    if v is None or isinstance(v, (int, float)):
        return None
    m = re.match(r"^\s*-?\d+(?:[.,]\d+)?\s*(.+)$", str(v))
    return m.group(1).strip() if m else None

# ---------------------------------------------------------------- dates
_VN_MON = r"thg|tháng|th"
def parse_date(v, year_hint=None):
    """returns pandas Timestamp or NaT. Handles datetime, excel serial, dd/mm/yyyy,
    yyyy-mm-dd, '1 thg 6 2025', 'd/m' (needs year_hint)."""
    if v is None:
        return pd.NaT
    if isinstance(v, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        if np.isnan(v):
            return pd.NaT
        if 20000 < v < 80000:
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))
        return pd.NaT
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return pd.NaT
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        return parse_date(float(s))
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?)?", s)
    if m:
        y, mo, d, hh, mi, ss = m.groups()
        return pd.Timestamp(int(y), int(mo), int(d), int(hh or 0), int(mi or 0), int(ss or 0))
    m = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?", s)
    if m:
        d, mo, y, hh, mi, ss = m.groups()
        y = int(y); y = y + 2000 if y < 100 else y
        try:
            return pd.Timestamp(y, int(mo), int(d), int(hh or 0), int(mi or 0), int(ss or 0))
        except ValueError:
            return pd.NaT
    m = re.fullmatch(rf"(\d{{1,2}})\s*(?:{_VN_MON})\s*(\d{{1,2}})[ ,]+(\d{{4}})", s, flags=re.I)
    if m:
        d, mo, y = m.groups()
        return pd.Timestamp(int(y), int(mo), int(d))
    m = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})", s)
    if m and year_hint:
        d, mo = m.groups()
        try:
            return pd.Timestamp(int(year_hint), int(mo), int(d))
        except ValueError:
            return pd.NaT
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        return pd.NaT
    try:
        return pd.Timestamp(s)
    except Exception:
        return pd.NaT

def parse_time_of_day(v):
    """returns seconds since midnight or nan"""
    if v is None:
        return np.nan
    if isinstance(v, (pd.Timestamp, dt.datetime)):
        return v.hour * 3600 + v.minute * 60 + v.second
    if isinstance(v, dt.time):
        return v.hour * 3600 + v.minute * 60 + v.second
    if isinstance(v, (float, int, np.floating)) and not np.isnan(v) and 0 <= v < 1:
        return round(v * 86400)
    s = str(v).strip()
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)
    return np.nan

# ---------------------------------------------------------------- status / methods
def status_of(v):
    """-> 'done' | 'cancel' | 'return' | 'shipping' | None"""
    k = nkey(v)
    if not k:
        return None
    if re.search(r"\b(huy|cancel|cancelled|canceled|khong giao duoc|bom hang|bom|khach bo|bo don|khong nhan)\b", k):
        return "cancel"
    if re.search(r"\b(tra hang|tra lai|tra ve|hoan tra|hoan hang|doi tra|returned|return|khach tra|mang tra)\b", k):
        return "return"
    if re.search(r"\b(dang|chua giao|cho giao|shipping|ship|van chuyen|dang gui|xu ly)\b", k):
        return "shipping"
    if re.search(r"\b(xong|hoan thanh|thanh cong|ok|da lay|da giao|completed|complete|done|da thanh toan|da nhan|giao thanh cong|paid|delivered)\b", k):
        return "done"
    return None

def paymethod_of(v):
    k = nkey(v)
    if not k:
        return None
    if "cod" in k or "thu ho" in k:
        return "cod"
    if "tien mat" in k or k == "cash" or "cash" in k:
        return "cash"
    if "chuyen khoan" in k or "transfer" in k or k in ("ck", "bank"):
        return "transfer"
    if "vi" in k.split() or "ewallet" in k or "e wallet" in k or "momo" in k:
        return "ewallet"
    if "no" in k.split() or "debt" in k or "ghi no" in k or "credit" in k:
        return "debt"
    return k

def channel_of(v):
    k = nkey(v)
    if not k:
        return None
    if "zalo" in k:
        return "zalo"
    if "phone" in k or "dien thoai" in k:
        return "phone"
    if "san" in k.split() or "tmdt" in k or "shopee" in k or "lazada" in k or "tiki" in k or "ecommerce" in k:
        return "ecommerce"
    if "store" in k or "tai tiem" in k or "tai quan" in k or "tai cua hang" in k or "offline" in k:
        return "store"
    if "web" in k:
        return "web"
    return k

WALKIN_KEYS = {"", "khach le", "le", "kl", "khach vang lai", "vang lai", "khach", "nan", "none", "k le", "kh le"}

def is_walkin(name):
    return nkey(name) in WALKIN_KEYS
