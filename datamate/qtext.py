"""Question text normalisation: accents stripped, teencode/telex residue repaired."""
import re
from .norm import strip_acc

# token-level replacements applied after accent stripping + lowercasing
TEEN = {
    "dthu": "doanh thu", "doanhthu": "doanh thu", "dt": "doanh thu", "bnhiu": "bao nhieu", "bnhieu": "bao nhieu", "bn": "bao nhieu",
    "bao nhiu": "bao nhieu", "nhiu": "nhieu", "nhieu": "nhieu", "nhiuu": "nhieu", "nhieeu": "nhieu", "nhiue": "nhieu",
    "hnay": "hom nay", "hqua": "hom qua", "homnay": "hom nay", "homqua": "hom qua",
    "dc": "duoc", "dk": "duoc", "duoc": "duoc", "duowc": "duoc", "dduowc": "duoc", "duoic": "duoc", "duowcj": "duoc", "dduwowcj": "duoc", "duwowcj": "duoc", "dduoc": "duoc", "duuoc": "duoc", "duocwj": "duoc", "duowc": "duoc", "duoc": "duoc", "đc": "duoc",
    "k": "khong", "ko": "khong", "hok": "khong", "hem": "khong", "kh": "khach hang", "kg": "kg",
    "trc": "truoc", "truoc": "truoc", "trk": "truoc", "sp": "san pham", "sph": "san pham", "dh": "don hang", "đh": "don hang",
    "z": "vay", "v": "vay", "zay": "vay", "dzay": "vay", "dza": "vay", "dzạ": "vay", "rua": "vay", "ri": "vay",
    "mik": "minh", "mjk": "minh", "j": "gi", "ji": "gi", "gj": "gi", "nhut": "nhat", "nhứt": "nhat", "nhat": "nhat", "nhta": "nhat",
    "mo": "nao", "nao": "nao", "naof": "nao", "nàoo": "nao", "naoo": "nao",
    "ntn": "nhu the nao", "tg": "thoi gian", "ng": "nguoi", "nguoi": "nguoi", "ngui": "nguoi", "ngoi": "nguoi",
    "bua": "bua", "nay": "nay", "ni": "nay", "hum": "hom", "hum nay": "hom nay",
    "thag": "thang", "thg": "thang", "thangs": "thang", "thnag": "thang", "thanng": "thang",
    "ngaynao": "ngay nao", "tiem": "tiem", "tieng": "tien", "tieenf": "tien", "tien": "tien",
    "tuan": "tuan", "tuần": "tuan", "tuann": "tuan", "tuaan": "tuan",
    "cuar": "cua", "kua": "cua", "cuaa": "cua", "nhaf": "nha", "trar": "tra", "nowj": "no", "noj": "no",
    "chayj": "chay", "laix": "lai", "loi": "loi", "vaayj": "vay", "luwowtj": "luot", "lan luwowtj": "lan luot",
    "heets": "het", "hoom": "hom", "laij": "lai", "cuoois": "cuoi", "soos": "so", "ddoof": "do", "coongj": "cong", "thuoocj": "thuoc",
    "chiuj": "chiu", "dduwowcj": "duoc", "nhapj": "nhap", "hagn": "hang", "hagn": "hang", "bna": "ban", "bná": "ban",
    "dag": "dang", "kon": "con", "kòn": "con", "tih": "tinh", "tíh": "tinh", "nhìu": "nhieu", "nghen": "nhe",
    "bít": "biet", "bit": "biet", "jùm": "gium", "jum": "gium", "gium": "gium", "dum": "gium", "giup": "giup",
    "trog": "trong", "wan": "quan", "wán": "quan", "wa": "qua", "cko": "cho", "kủa": "cua", "kua": "cua", "lon": "lon",
    "bnhiêu": "bao nhieu", "nhieu": "nhieu", "mấy": "may", "may": "may", "o đờ": "order", "chong": "trong",
    "dảm": "giam", "dam": "giam", "xo": "so", "lựot": "luot", "luot": "luot", "lênn": "len", "lenn": "len",
    "tóp": "top", "ddown": "don", "doen": "don", "ddon": "don", "gaanf": "gan", "ees": "e", "nawm": "nam", "cos": "co",
    "nhoms": "nhom", "nheiu": "nhieu", "kac": "cac", "zia": "vao", "vo": "vao", "tog": "tong", "cog": "cong", "trung bih": "trung binh", "bih": "binh", "dthu": "doanh thu", "dtu": "doanh thu", "top": "top", "hàngf": "hang", "hangf": "hang", "mặt hangf": "mat hang", "naof": "nao",
    "cuwar": "cua", "doanhthu": "doanh thu", "nhât": "nhat", "đuowc": "duoc", "duowc": "duoc", "đượcc": "duoc",
    "tháng": "thang", "bnhieu": "bao nhieu", "sx": "san xuat", "cu": "cu", "tr": "trieu", "ty": "ty",
}

TELEX_TAIL = re.compile(r"^([a-z]*[aeiouy][a-z]*?)([sfrxj])$")

def _telex_fix(tok, vocab):
    if tok in vocab or len(tok) < 3:
        return tok
    t = tok
    for a, b in (("dd", "d"), ("aa", "a"), ("ee", "e"), ("oo", "o"), ("ow", "o"), ("uw", "u"), ("aw", "a")):
        t = t.replace(a, b)
    m = TELEX_TAIL.match(t)
    if m and m.group(1) in vocab:
        return m.group(1)
    if t in vocab:
        return t
    return tok

_VOCAB = set("""thang nam ngay tuan quy hom nay qua bua truoc roi nay sau doanh thu ban duoc bao nhieu tien don hang khach mat
san pham nhom huy tra no con ton kho nhap gia von lai gop top nhat chay mua may ly cai chai lon goi hop thung tang giam
phan tram ti le trong cua tiem quan cua hang tinh den het cuoi dau tu nao gi ai the nhu nhieu it hon kem so voi cung ky
lien tuc luon trung binh moi tren mot binh quan ca sang chieu nhan vien thuoc ke ngoai hoat chat ngay giao dot xe cho
size tong cong mang ve dat tro len duoi qua khong co la thi da chua dang thu chi phi ship voucher diem hang hoa
loi gop that rong lon ban tien mat chuyen khoan vi dien tu cod ghi no""".split())

import os, unicodedata
try:
    _VOCAB |= set(open(os.path.join(os.path.dirname(__file__), "vocab.txt"), encoding="utf-8").read().split())
except Exception:
    pass

EN = {"o đờ": "đơn", "ô đờ": "đơn", "revenue": "doanh thu", "sales": "doanh thu", "best seller": "ban chay nhat", "best sellers": "ban chay nhat",
      "total": "tong cong", "order": "don", "orders": "don", "top seller": "ban chay nhat", "customer": "khach", "item": "mat hang"}

def qnorm(q: str) -> str:
    q = unicodedata.normalize("NFC", q)
    ql = q.lower()
    # 'nhiêu' (with circumflex) alone means 'bao nhiêu'
    ql = re.sub(r"(?<!bao )(?<!bao)\bnhiêu\b", "bao nhiêu", ql)
    ql = re.sub(r"\b(hông|hôn)\b", "không", ql)
    for a, b in EN.items():
        ql = re.sub(r"\b" + a + r"\b", b, ql)
    s = ql.replace("đ", "d").replace("Đ", "D")
    s = strip_acc(s).lower()
    s = s.replace("%", " phan tram ")
    # keep digits separators for dates/money; separate other punctuation
    s = re.sub(r"[?!,;:\"“”…()\[\]{}]+", " ", s)
    s = re.sub(r"(\d)\s*k\b", r"\1k", s)
    toks = s.split()
    out = []
    for t in toks:
        tt = t.strip(".")
        if tt in TEEN:
            out.append(TEEN[tt]); continue
        fx = _telex_fix(tt, _VOCAB)
        out.append(TEEN.get(fx, fx))
    s = " ".join(out)
    s = re.sub(r"\s+", " ", s).strip()
    # common multiword fixes
    s = s.replace("bao nhieu tien", "bao nhieu tien")
    s = re.sub(r"\bth\s*(\d)", r"thang \1", s)
    s = re.sub(r"\bth\b", "thang", s)
    s = re.sub(r"\bt\s*(\d{1,2})\s*/", r"thang \1/", s)
    s = re.sub(r"\bthang(\d)", r"thang \1", s)
    s = re.sub(r"\bngay(\w)", r"ngay \1", s) if re.search(r"\bngay(nao)\b", s) else s
    return s
