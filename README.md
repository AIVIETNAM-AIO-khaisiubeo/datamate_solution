# DataMate solution — Zalo AI Challenge 2026

Hệ thống trả lời câu hỏi tiếng Việt trên dữ liệu bảng (CSV / XLSX / SQLite) của doanh nghiệp nhỏ, tối ưu cho
Penalized Accuracy (đúng +1, sai −1, bỏ trống 0). Nguyên tắc số một: **không chắc thì từ chối**.

## Kiến trúc

```
raw files ─► tiền xử lý bảng: dò dòng tiêu đề thật, gỡ ô gộp (đọc mergeCell từ XML), header 2 tầng,
             forward-fill trong nhóm dòng, bỏ dòng tổng/ghi chú/header lặp, gộp file tách tháng/tách phần
          ─► lớp ngữ nghĩa: vai trò cột từ README + tên cột + mẫu giá trị ─► mô hình chuẩn
             (orders · lines · products · units · payments · imports · stock/ledger · lots · deliveries · debt terms · expenses · tickets)
question  ─► chuẩn hoá tiếng Việt (dấu, teencode, telex, English) ─► thời gian (tuần/tháng/quý/Tết âm lịch)
          ─► liên kết thực thể (± fuzzy) ─► ~50 handler tất định theo rulebook          [Phase 1, CPU]
          ─► Agent (LLM tự viết code từ cấu trúc bảng):                                   [Phase 2, GPU]
               schema + 3 dòng mẫu (.head()) → lập kế hoạch → K chương trình (pandas + SQL in-memory)
               → sandbox (timeout) → traceback feedback ≤ 2 vòng → bỏ phiếu → model chấm cho điểm tin cậy
          ─► router: đối chiếu chéo rules ↔ agent ─► đáp án | unanswerable | abstain
```

### Cơ chế từ chối (abstention)
Hệ thống trả `{"type": "abstain"}` (0 điểm) thay vì đoán khi:
* code agent lỗi / hết giờ ở mọi lần thử (kể cả sau 2 vòng sửa theo traceback);
* lựa chọn dữ liệu rỗng (agent báo `n_rows = 0`, engine gặp selection rỗng sau khi lọc);
* tỉ lệ đồng thuận giữa các mẫu < `DM_MIN_SHARE` (0.6) hoặc `đồng thuận × tin cậy của model chấm` < `DM_TAU` (0.55);
* rules và agent mâu thuẫn mà handler không thuộc nhóm tin cậy (precision dev ≥ `DM_P_TRUST`);
* câu hỏi nhắc tới thực thể không liên kết được (mặt hàng/nhóm lạ);
* mô hình dữ liệu không đáng tin: nhãn trạng thái lạ > 3 % đơn, ngày không parse được > 5 %, thành tiền lệch SL×giá;
* đáp án sai định dạng (NaN, list rỗng, ngày sai mẫu, % vô lý).
Nhãn trạng thái lạ được LLM phân loại lại (3 mẫu, phải nhất trí) rồi chạy lại rules.

### Ngân sách thời gian
* mỗi chương trình chạy trong sandbox tối đa `exec_timeout_s` (10 s); mỗi câu tối đa `DM_Q_BUDGET_S` (40 s) thời gian thực thi;
* deadline toàn cục `DM_DEADLINE_S` (mặc định 5 h 20 m): quá hạn thì dừng LLM, giữ đáp án rules;
* submission được ghi ngay sau Phase 1 và ghi đè nguyên tử sau Phase 2 → dừng giữa chừng vẫn có file hợp lệ.

## Tuân thủ (Rule 4 / Rule 6) và tổng quát hoá

* **Không gắn cứng ID**: luồng suy luận không chứa ID câu hỏi hay ID doanh nghiệp. `business_id` chỉ dùng để chọn thư mục dữ liệu của câu hỏi.
  `splits/split.json` chỉ dùng cho đánh giá, không dùng khi suy luận.
* **Không học thuộc đáp án**: không có bảng tra đáp án. Mọi đáp án đều được tính lại từ dữ liệu khi chạy; đổi tên cột, đổi định dạng hay xáo bố cục thì đáp án vẫn đúng (xem các kiểm thử biến đổi).
* **Đã gỡ phần viết dựa trên public test (01/11/2026)**: 11 handler chưa từng trả lời câu nào trong train nhưng lại trả lời 120 câu public, nên đã bị xoá hẳn khỏi mã nguồn.
  Danh sách: sổ sửa xe, shop điện thoại/thu máy cũ/trả góp, thuộc tính bỉm–sữa, hạn nợ, điều chỉnh voucher/điểm, phí ship, lý do trả hàng, giờ cao điểm, đếm khách, địa chỉ, thành viên mới.
  Bộ dựng sổ phiếu dịch vụ (`special.py`) cũng bị xoá. Các câu hỏi loại này giờ do agent xử lý.
* **Chốt chặn "dữ liệu chưa hiểu hết"** (`data_understood`): nếu không nhận ra bảng bán hàng, hoặc còn bảng ≥ 100 dòng chưa phân loại, thì:
  - engine không được trả `unanswerable` (sai bị −1) mà chuyển thành abstain;
  - đáp án của engine bị gắn cờ `data_partial`, và router chỉ giữ khi agent đồng ý.
  Mục đích là để engine không tự tin quá mức trên loại doanh nghiệp mới ở private.
* Vẫn còn **vùng xám, đã công khai**: một số sửa lỗi đọc định dạng mang tính tổng quát được làm khi quan sát file public (chưa bao giờ có đáp án public). Cụ thể: dò dòng tiêu đề, khoảng tháng trong câu hỏi, gợi ý "các dòng cùng X là một đơn" trong README, xử lý shop không có bảng đơn, một số tên cột ngày.
  Các sửa lỗi này không phụ thuộc shop nào và không đổi kết quả train.

## Mô hình (Rule 2 — tổng ≤ 15B)

| Cấu hình | Sinh code | Chấm tin cậy | Tổng |
|---|---|---|---|
| A (mặc định) | Qwen/Qwen2.5-Coder-7B-Instruct (7.6B, Apache-2.0, 09/2024) | Qwen/Qwen3-4B (4.0B, Apache-2.0, 04/2025) | 11.6B |
| B | Qwen/Qwen3-14B (14.8B, Apache-2.0, 04/2025) | (cùng model) | 14.8B |

Kiểm tra bằng `python tools/count_params.py models/gen models/ver`. Mỗi model chạy trong một tiến trình worker riêng
(vLLM, tensor parallel = 2, prefix caching, tuỳ chọn `DM_QUANT=fp8|awq|gptq`); tiến trình thoát là giải phóng GPU.
Chọn A hay B bằng `eval_sealed.py --mode hybrid` trên tập niêm phong.

## Đánh giá không rò rỉ (leakage-aware)

* `splits/split.json`: 6 shop (631 câu, 21 %) được **niêm phong** theo business (không chia ngẫu nhiên theo câu, vì câu cùng shop
  dùng chung mẫu và dữ liệu). `run_eval.py` tự loại các shop này; `eval_sealed.py` chỉ in số tổng hợp, không in câu/đáp án.
* Từ vựng sinh từ dữ liệu (`vocab.txt`, `generic_words.txt`) và bảng precision handler chỉ tính trên shop dev.
* **Lưu ý trung thực**: phiên bản đầu của luật được viết khi đã thấy toàn bộ 3.000 câu train, nên điểm "sạch" trên shop niêm phong
  vẫn lạc quan. Thước đo đáng tin hơn là các **kiểm thử biến đổi (metamorphic)** dưới đây — nhãn giữ nguyên, dữ liệu/câu hỏi đổi dạng
  mà hệ thống chưa từng thấy — và điểm leaderboard public.

| Tập niêm phong (631 câu) | PA | Sai | Bỏ trống |
|---|---|---|---|
| Nguyên bản (lạc quan) | 1.000 | 0 | 0 |
| + nhiễu câu hỏi | 1.000 | 0 | 0 |
| + đổi định dạng tiền/ngày | 1.000 | 0 | 0 |
| + nhãn trạng thái lạ | 0.260 | 0 | 467 |
| + kết hợp tên cột/định dạng/bố cục | 0.975 | 2 | 12 |

Chi tiết: RESULTS.md

## Chạy

```bash
python3 predict.py /data /result/submission.jsonl --mode rules     # chỉ engine (CPU)
DM_GEN_MODEL=models/gen DM_VER_MODEL=models/ver python3 predict.py /data /result/submission.jsonl --mode hybrid
./predict.sh /data /result/submission.jsonl                          # entrypoint Docker (tự chọn hybrid nếu có weights)
python3 eval_sealed.py [--perturb questions] [--data-variant DIR] [--mode rules|agent|hybrid]
python3 tools_perturb.py <data_root> headers,values,status,layout <out_dir>   # tạo biến thể dữ liệu cho shop niêm phong
```

Colab (1 GPU): mở `DataMate_colab.ipynb`. Notebook tự chọn AWQ+fp16 cho T4, bf16 cho L4/A100 và đặt `DATAMATE_TP=1`.
Sau đó chạy `eval_sealed.py --mode hybrid --limit N --dump F` rồi `tools/sweep.py F` để chọn `DM_MIN_SHARE / DM_TAU / DM_P_TRUST / DM_TAU_AGENT_ONLY`.
Biến môi trường cho GPU nhỏ: `DM_DTYPE=float16`, `DM_EAGER=1`, `DM_MAX_LEN`, `DM_GPU_UTIL`, `DM_K_PY`, `DM_K_SQL`.

Mỗi câu ghi log `/result/logs/<id>.json`: đáp án rules, đáp án agent (độ tin cậy), kết quả đối chiếu, đáp án cuối, thời gian.

## Cấu trúc mã
* `datamate/loader.py` — đọc file thô, gỡ ô gộp, dò tiêu đề.  `semantic.py` — vai trò cột.  `model.py` — mô hình chuẩn.
* `special.py` — sổ phiếu dịch vụ.  `qtext.py`, `timeparse.py` — câu hỏi, thời gian.  `engine.py`, `intents*.py` — handler.
* `agent.py` — agent sinh code (pandas + SQL), sandbox, traceback loop, bỏ phiếu, chấm.  `agent_worker.py` — worker vLLM.
* `router.py` — kết hợp rules ↔ agent.  `scorer.py` — chấm cục bộ theo Evaluation.docx.
