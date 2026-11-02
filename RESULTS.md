# Kết quả đánh giá (01/11/2026, engine tất định — mode `rules`)

Tập niêm phong: 6 shop, 631 câu (21 % train), chia theo business. Chỉ chấm tổng hợp.

| Thí nghiệm | PA | Sai | Bỏ trống | Ghi chú |
|---|---|---|---|---|
| Dev (24 shop, 2.369 câu) | 1.000 | 0 | 0 | trong mẫu — luật được viết trên các câu này |
| Niêm phong, nguyên bản | 1.000 | 0 | 0 | **lạc quan**: bản luật đầu tiên đã thấy cả 3.000 câu |
| Niêm phong + nhiễu câu hỏi (425/631 câu đổi: bỏ dấu, teencode, telex) | 1.000 | 0 | 0 | |
| Niêm phong + đổi tên cột (README cập nhật) | 0.995 | 0 | 3 | đo ở phiên bản trước các bản sửa sau |
| Niêm phong + đổi định dạng tiền/ngày | 1.000 | 0 | 0 | **lần đầu 0.778, 70 câu sai** → sửa lỗi so trùng dòng theo chuỗi thô |
| Niêm phong + nhãn trạng thái chưa từng gặp | 0.260 | 0 | 467 | an toàn: từ chối thay vì sai. Với LLM gán nhãn (mô phỏng): khôi phục hoàn toàn trên shop dev |
| Niêm phong + xáo cột + tách file | 1.000 | 0 | 0 | |
| Niêm phong + đổi tên cột + định dạng + bố cục (kết hợp) | 0.975 | 2 | 12 | **lần đầu 0.772, 63 câu sai** → gộp file tách phần, khớp README theo gốc tên file, dòng ghi chú ở cột bất kỳ |

Các lỗi tìm được bằng kiểm thử biến đổi đều được chẩn đoán qua **thống kê tổng hợp của mô hình chuẩn**
(số dòng, tổng doanh thu, số ô không parse được) và tái hiện/sửa trên shop dev, không xem câu hỏi hay đáp án niêm phong.

## Sau khi gỡ phần viết dựa trên public (01/11/2026)
* Train (3.000 câu): 3.000 đúng, 0 sai — không đổi, vì các handler bị gỡ chưa từng trả lời câu train nào.
* Niêm phong: nguyên bản 1.000; nhiễu câu hỏi 1.000; kết hợp tên cột/định dạng/bố cục 0.975 (2 sai, 12 bỏ trống); nhãn trạng thái lạ 0.260 (0 sai). Không đổi.
* Public (chỉ rules): 853/1000 câu có trả lời (trước đó 954); `unanswerable` giảm từ 21 xuống 16. Phần chênh được nhường cho agent.

## Chưa đo (cần GPU)
* `--mode agent` và `--mode hybrid`; chọn cấu hình model A (Coder-7B + Qwen3-4B) hay B (Qwen3-14B);
  chỉnh `DM_TAU`, `DM_MIN_SHARE`, `DM_P_TRUST` bằng `eval_sealed.py` (chỉ số tổng hợp).
* Thời gian chạy thực tế trên 2×RTX 5090.
