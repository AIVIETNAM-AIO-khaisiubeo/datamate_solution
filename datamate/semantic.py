"""Generic semantic layer: raw tables -> canonical business model.

Every decision is driven by column names, README column descriptions and value
profiles; nothing is keyed on business identifiers."""
import re, os
import numpy as np
import pandas as pd
from .norm import (nkey, parse_money, parse_qty, parse_date, parse_time_of_day, status_of,
                   paymethod_of, channel_of, is_walkin, qty_unit_suffix)
from .loader import load_business

# ------------------------------------------------------------------ README parsing
def parse_readme(text):
    """-> dict: col_descs {(file_or_None, colname): desc}, data_end Timestamp, conversions [(item, big, factor, small)], raw"""
    info = {"col_descs": {}, "data_end": None, "conversions": [], "raw": text, "bundle": [], "order_key_cols": []}
    for m in re.finditer(r"các dòng cùng \*\*(.+?)\*\* là một đơn", text):
        info["order_key_cols"].append(m.group(1).strip())
    cur_file = None
    for line in text.splitlines():
        m = re.match(r"##\s*Cột trong `([^`]+)`(?:,\s*(?:sheet|bảng)\s*`([^`]+)`)?", line)
        if m:
            cur_file = (m.group(1), m.group(2)); continue
        if line.startswith("## "):
            cur_file = None
        m = re.match(r"-\s*\*\*(.+?)\*\*:\s*(.+)$", line.strip())
        if m and cur_file is not None:
            info["col_descs"][(cur_file, m.group(1).strip())] = m.group(2).strip()
        m = re.search(r"Dữ liệu có đến hết (\d{1,2}/\d{1,2}/\d{4})", line)
        if m:
            info["data_end"] = pd.to_datetime(m.group(1), dayfirst=True)
        m = re.match(r"\s*-\s*(.+?):\s*1\s+(.+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s*$", line)
        if m and not line.strip().startswith("- **"):
            info["conversions"].append((m.group(1).strip(), m.group(2).strip(), float(m.group(3).replace(",", ".")), m.group(4).strip()))
        m = re.search(r"\*\*Hàng theo (\w+)\*\*:\s*(.+?) bán theo \w+, 1 (\w+) = (\d+) (\w+)", line)
        if m:
            for it in m.group(2).split(","):
                info["conversions"].append((it.strip(), m.group(3), float(m.group(4)), m.group(5)))
    return info

# ------------------------------------------------------------------ role patterns
# (role, regex over nkey(colname)); order matters: first match wins per column
NAME_ROLES = [
    ("receipt_id", r"^(ma |so )?(phieu nhap|nhap hang|don nhap|pn)$|^ma phieu nhap$|^ma nhap hang$|^ma don nhap$|^ma phieu nhap$"),
    ("pay_id", r"^(ma )?(phieu thu|ma tt|ma thanh toan|ma gd|ma phieu)$|^ma phieu thu$|^phieu thu$|^ma tt$"),
    ("lot", r"^(ma )?lo( hang)?$|^ma lo$|^so lo$"),
    ("expiry", r"^(han su dung|hsd|han dung)$"),
    ("qty_return_base", r"so luong tra|sl tra|sltraquydoi|sl tra quy doi|sltra quy doi"),
    ("qty_return_base2", r"^sltra quy doi$"),
    ("qty_base", r"slquy doi|slnhap quy doi|so luong quy doi|sl quy doi|quy ra dv le|so luong le|sl le|sl co ban|sl dv co ban|slquydoi|sl quy ra le|so luong dvt co ban|so luong co ban|sl nhap quy doi|sl nhap dvt co ban|slnhapquydoi"),
    ("order_id", r"^(ma )?(hoa don|don hang|don|bill|chung tu|ma bill|order id|order)$|^(so|ma) (ct|hd|don|phieu|chung tu|bill)$|^don so$|^hd$|^ma gd$|^ma giao dich$|^so hoa don$|^ma hoa don$|^bill$|^ma don$"),
    ("line_no", r"^(stt|so dong|thu tu|no|tt dong|dong|so thu tu)$"),
    ("ts", r"^(thoi gian tao|gio tao don|gio ban|thoi gian lap|thoigianlap|ngay gio thu|thoi gian thu|thoi diem|ngay tt|ngay thu|ngay gio phat sinh|ngay gio tra|gio)$"),
    ("date", r"^(ngay ban|ngay|ngay dat hang|ngay thang|ngay giao dich|thoi gian|ngay hd|ngay lap|ngay xuat|date|ngay mua|ngay chung tu|ngay nhap|ngay nhap kho|ngay ap dung|ngay chi|ngay giao hang|ngay giao|ngay cho|thoi gian giao|ngay sua|ngay nhan)$"),
    ("status", r"^(trang thai.*|status|ket qua|xu ly|tinh trang( don)?|tien do|tt don|ghi chu tt|trang thai giao)$"),
    ("cust_code", r"^(ma kh( mua)?|ma khach( hang| mua| le si)?|customer id|ma the|ma nguoi mua|so kh|ma khach hang|ma kh mua|ma khach le si)$"),
    ("item_code", r"^(ma hang( hoa| ban)?|ma sp|sku|ma sku|ma vach|barcode|ma h hoa|id sp|ma mat hang|ma san pham( ban)?|ma hang hoa|ma hh|ma mon)$"),
    ("cust_name", r"^(khach hang|ten khach hang|khach|doi tac|ben mua|ten kh|khach hang ten|ho ten khach|khach hang mua|khach le si|kh|ten khach|tenkh)$"),
    ("item_name", r"^(ten hang( hoa)?|ten san pham|san pham|noi dung|hang|mat hang|hang ban|hang hoa|ten sp|chi tiet hang|mon|item|product|ten mon|ten mat hang|ten hang|ten_hang|may)$"),
    ("name", r"^(ten goi|ho ten|ten)$"),
    ("abbrev", r"^(ghi tat|viet tat)$"),
    ("item_type", r"^(loai|loai hang)$"),
    ("imei", r"^imei$"),
    ("tradein", r"^(thu may cu|gia thu)$"),
    ("shift", r"^(ca|ca lam|ma ca lam)$"),
    ("unit", r"^(don vi tinh|dvt|don vi|unit|dv|dv tinh|donvi|d vi|dvt nhap|don vi nhap|dv nhap)$"),
    ("qty", r"^(so luong( san pham| ban| nhap| hang)?|sl( ban| nhap)?|slg|sluong|so_luong|soluong|s l|sl nhap)$"),
    ("price", r"^(don gia( ban| d| nhap)?|gia ban( le)?|gia|gia 1 dv|gia le|gia san pham|gia don vi|giadonvi|gia niem yet|price|don_gia|gia nhap|dongianhap|gia moi)$"),
    ("old_price", r"^gia cu$"),
    ("line_amount", r"^(thanh tien( d)?|tri gia|amount|tong|cong|tien ban|t tien|trigia|thanh_tien|tong tien)$"),
    ("disc", r"^(giam gia|giam d|ck|khuyen mai|chiet khau|tien giam|giam)$"),
    ("order_disc", r"^(giam gia hd|giam gia hoa don|bot)$"),
    ("cost", r"^(gia von( luc ban| hien tai| hang)?|giavonlucban|giavonhang)$"),
    ("goods_total", r"^(tong tien hang|tien hang( truoc giam)?|gia tri hang|tongtienhang)$"),
    ("ship_fee", r"^(phi ship|phi van chuyen|phivanchuyen|phi giao hang)$"),
    ("payable", r"^(tong sau giam gia|tong thanh toan|tong bill|phai thu|tong don|tongthanhtoan|tongtien|tong tien phieu|can tra ncc|tong phieu|tonggiatrinhap|tong gia tri nhap)$"),
    ("paid", r"^(khach da tra|khach tra|da tra|da thu|khachdatra|da thanh toan)$"),
    ("debt", r"^(con no|no|chua tra|con lai|du no|cong no|conphaithu|con phai thu)$"),
    ("paymethod", r"^(phuong thuc.*|hinh thuc thanh toan|pttt|tra bang|thanh toan bang|httt|loai tt|phuongthuctt|phuongthuc)$"),
    ("channel", r"^(kenh.*|nguon( don)?|ban qua|loai ban|kenhbanhang)$"),
    ("staff", r"^(nguoi ban|nhan vien( ban)?|ma nv( ban)?|manhanvien|nhan vien ban)$"),
    ("cust_group", r"^(nhom khach( hang)?|loai khach( hang)?|hang|kieu khach|nhom|loaikhachhang)$"),
    ("item_group", r"^(nhom hang( hoa)?( 3 cap)?|loai san pham|phan loai|danh muc|loai hang|nhom|nhom hang)$"),
    ("supplier", r"^(nha cung cap|ma ncc|ma nha cung cap|ncc|manhacungcap)$"),
    ("note", r"^(ghi chu|note|luu y|dien giai|mo ta|noi dung)$"),
    ("stock", r"^(ton cuoi( ky)?|ton kho|ton thuc|toncuoi|ton)$"),
    ("phone", r"^(so dt|dien thoai|so dien thoai|sdt|phone|di dong|sodienthoai)$"),
    ("address", r"^(dia chi|addr|diachilienhe)$"),
    ("table", r"^(ban|phong ban)$"),
    ("area", r"^(khu vuc)$"),
    ("guests", r"^(so khach)$"),
    ("active", r"^(dang kinh doanh|conkinhdoanh|con kinh doanh)$"),
    ("rx", r"^(thuoc ke don|ke don|bantheodon|ban theo don)$"),
    ("ingredient", r"^(hoat chat|thanhphanchinh|thanh phan chinh)$"),
    ("factor", r"^(ty le quy doi|ti le quy doi|he so( qd)?|quy doi|bang|hesoquydoi|he so quy doi)$"),
    ("big_unit", r"^(don vi quy doi|loai dv|dv lon|quy cach)$"),
    ("base_unit", r"^(don vi co ban|dv goc|don vi goc|dv nho|donvicoso|don vi co so)$"),
    ("delivery_no", r"^(lan giao|luot giao)$"),
    ("plate", r"^(bien so xe)$"),
    ("driver", r"^(nguoi giao hang|lai xe|nguoi cho)$"),
    ("terms_days", r"^(so ngay duoc no|ky han|han no ngay|han no)$"),
    ("due_date", r"^(han thanh toan|ngay phai tra|han tra)$"),
    ("expense_cat", r"^(muc|loai chi|khoan chi)$"),
    ("money", r"^(so tien( thu)?|tien( thu| chi)?|gia tri|sotienthu|so tien|tien mat thu|so tien thu)$"),
    ("pay_date", r"^(ngay tra|ngay gop)$"),
    ("pay_amount", r"^(da tra|da gop)$"),
]

# README description phrase -> role
DESC_ROLES = [
    ("qty_return_base", r"số lượng khách trả lại"),
    ("qty_base", r"số lượng (đã đổi ra|bán lấy từ lô|nhập \(đơn vị cơ bản)|đã đổi ra đơn vị cơ bản"),
    ("receipt_id", r"^mã phiếu nhập"),
    ("pay_id", r"^mã phiếu thu"),
    ("order_id", r"^(mã đơn|số hoá đơn|số thứ tự đơn)"),
    ("line_no", r"^số thứ tự dòng"),
    ("ts", r"^(thời điểm đặt đơn|giờ đặt đơn|thời điểm thu)"),
    ("date", r"^(ngày bán|ngày nhập|ngày giao|ngày bắt đầu áp dụng|ngày sửa)"),
    ("status", r"^trạng thái đơn"),
    ("import_status", r"^trạng thái phiếu nhập"),
    ("cust_code", r"^mã khách"),
    ("cust_name", r"^tên khách"),
    ("item_code", r"^mã hàng"),
    ("item_name", r"^(tên hàng|hàng bán)"),
    ("unit", r"^(đơn vị tính|đơn vị nhập)"),
    ("base_unit", r"^đơn vị cơ bản"),
    ("big_unit", r"^đơn vị lớn"),
    ("factor", r"^hệ số"),
    ("qty", r"^số lượng( nhập)?$|^số lượng \(|^số lượng$"),
    ("price", r"^(đơn giá|giá nhập một đơn vị|giá bán áp dụng|giá bán hiện tại)"),
    ("old_price", r"^giá trước lần đổi"),
    ("line_amount", r"^(thành tiền sau giảm|thành tiền của dòng|tiền nhập của cả dòng)"),
    ("disc", r"^(giảm giá$|tiền giảm của cả dòng)"),
    ("order_disc", r"^(giảm giá của cả đơn|tiền bớt cho cả đơn)"),
    ("cost", r"^giá vốn"),
    ("goods_total", r"^(tổng tiền hàng|tiền hàng của đơn)"),
    ("ship_fee", r"^phí giao hàng"),
    ("payable", r"^(tổng tiền đơn|số khách phải trả|khách cần trả|tổng tiền phiếu nhập|tổng tiền của cả phiếu|tổng tiền cả phiếu)"),
    ("paid", r"^(đã thanh toán|khách đã trả|số tiền khách trả cho cả đơn)"),
    ("debt", r"^(còn nợ|số khách còn nợ|khách còn nợ)"),
    ("paymethod", r"^hình thức thanh toán"),
    ("channel", r"^kênh bán"),
    ("staff", r"^mã nhân viên"),
    ("cust_group", r"^nhóm khách"),
    ("item_group", r"^nhóm hàng"),
    ("supplier", r"^(nhà cung cấp|mã nhà cung cấp)"),
    ("note", r"^ghi chú"),
    ("stock", r"^tồn kho"),
    ("phone", r"^số điện thoại"),
    ("address", r"^địa chỉ"),
    ("table", r"^bàn$"),
    ("area", r"^khu vực"),
    ("guests", r"^số khách của đơn"),
    ("active", r"^còn kinh doanh"),
    ("rx", r"^thuốc kê đơn"),
    ("ingredient", r"^hoạt chất"),
    ("lot", r"^số lô"),
    ("expiry", r"^hạn sử dụng"),
    ("delivery_no", r"^đợt giao"),
    ("plate", r"^biển số"),
    ("driver", r"^tài xế"),
    ("terms_days", r"^số ngày được nợ"),
    ("due_date", r"^hạn thanh toán"),
    ("money", r"^số tiền"),
    ("empty", r"^cột trống"),
]

ID_PATTERNS = {
    "order_id": r"^ORD\d+$", "receipt_id": r"^PN\d+$", "pay_id": r"^(PAY|PT)\d+$", "cust_code": r"^KH\d+$",
    "item_code": r"^SP\d+$", "supplier": r"^NCC\d+$", "staff": r"^NV\d+$", "return_id": r"^TH\d+$",
}

def _id_role(series):
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) < 3:
        return None
    sample = s.sample(min(300, len(s)), random_state=0)
    for role, pat in ID_PATTERNS.items():
        if sample.str.match(pat).mean() > 0.9:
            return role
    return None

def _file_stem(name):
    from .loader import _stem_group
    n = re.sub(r"<[^>]+>", "", str(name))
    return _stem_group(n).lower().strip("_ -")

def _desc_for(info, fname, sheet, col):
    for (f, sh), c in [((fname, sheet), col), ((fname, None), col)]:
        d = info["col_descs"].get(((f, sh), c))
        if d:
            return d
    # README section written for a pattern ('BanHang_T<tháng>_<năm>.xlsx') or for the unsplit file
    # ('ban_hang.csv' documented, data shipped as 'ban_hang_p1.csv' + 'ban_hang_p2.csv')
    st = _file_stem(fname)
    for ((f, sh), c), d in info["col_descs"].items():
        if c == col and (sh is None or sh == sheet) and (_file_stem(f) == st or ("<" in f and st.startswith(_file_stem(f)))):
            return d
    return None

def assign_roles(df, info, fname, sheet):
    roles = {}
    for c in df.columns:
        if c.startswith("_"):
            continue
        r = None
        idr = _id_role(df[c])
        d = _desc_for(info, fname, sheet, c)
        if d:
            for role, pat in DESC_ROLES:
                if re.search(pat, d, flags=re.I):
                    r = role; break
        if r is None:
            k = nkey(c)
            for role, pat in NAME_ROLES:
                if re.search(pat, k):
                    r = role; break
        if idr and r not in ("supplier",):
            if idr in ("order_id", "receipt_id", "pay_id", "cust_code", "item_code", "return_id") or r is None:
                if not (idr == "order_id" and r in ("order_id",)):
                    r = idr if r not in ("staff",) or idr == "staff" else r
        if r == "staff" and idr is None:
            pass
        if r == "empty":
            r = None
        if r:
            roles.setdefault(r, []).append(c)
    return roles
