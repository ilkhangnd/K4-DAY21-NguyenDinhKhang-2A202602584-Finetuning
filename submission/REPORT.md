# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đình Khang  **MSSV**: 2A202602584  **Ngày**: 07-10-2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4, 14.6 GB khả dụng

## 1. Setup

| Hạng mục | Giá trị |
| --- | --- |
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25, split seed 42 |
| Eval | 50 target ticket và 15 câu regression; không dùng `EVAL_LIMIT` |
| `max_length` | 1024; p95 đo được là 98, `suggested_max_length=256` |
| `MASK_MODE` | `assistant-only` |
| Epochs / max steps | 2 epochs / 30 optimizer steps cho cả bốn run |

Em dùng cấu hình T4 mặc định của lab với `max_length=1024` xuyên suốt baseline, train và eval để giữ cùng một cấu hình tier. Tuy nhiên, thống kê token cho thấy p95 chỉ là 98 và độ dài gợi ý là 256; ở lượt tối ưu tiếp theo em sẽ hạ xuống 256 để giảm padding, sau khi lập lại baseline công bằng.

**Template có giữ khối `<think>` không?** Có. `results/template_check.json` báo `reasoning preserved — safe to train on traces`; cả thẻ mở, nội dung và thẻ đóng đều còn sau `apply_chat_template`.

## 2. Mask proof (NB1)

| Kiểm tra | Kết quả |
| --- | ---: |
| `supervised_fraction` | 0.4149 (39 / 94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss được giải mã từ `labels != -100`:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

`supervised_fraction` thấp hơn rất xa 0.95, nên prompt không bị đưa vào loss. Vì vậy mọi run NB3/NB4 dùng dữ liệu đã pre-tokenize với chính mask này, thay vì dựa vào cờ `assistant_only_loss` của framework.

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms/mẫu) |
| --- | ---: | ---: | ---: | ---: |
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3138.0 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 986.2 |
| (c) LoRA fine-tune | 0.970 | 0.5222 | 1.000 | 1345.8 |

Baseline (b) thật sự mạnh hơn (a): target tăng từ 0.000 lên 0.765 và format tăng từ 0 lên 1.000. Prompt tối ưu cùng checksum `719e74d3b6232053` đã được đóng băng trước khi train; em không sửa `OPTIMIZED_PROMPT`. Bản LoRA tăng target thêm 0.205 so với (b), nhưng đánh đổi bằng regression và latency, nên không thể chỉ dùng cột target để kết luận deploy.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí | r | Trainable params | LR | Train loss | **target** | Train s | VRAM GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6269 | **0.970** | 382.5 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32,456,704 | 1e-4 | 0.5369 | 0.965 | 258.2 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.000 | 386.3 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 454.0 | 3.86 |

Xếp hạng theo target là `correct` > `attn_only` > `qlora` > `wrong_lr`; thứ hạng này không hoàn toàn giống train loss.

**4.1 — Vị trí adapter và rank.** `attn_only` có 32,456,704 tham số trainable, lệch khoảng 0.03% so với 32,464,896 của `correct`, nên đây là đối chứng ngân sách công bằng. Trên target, `attn_only` thua sát nút: 0.965 so với 0.970. Theo train loss thì thứ tự lại đảo ngược: `attn_only` có loss 0.5369 thấp hơn `correct` 0.6269. Điều này cho thấy loss train thấp không bảo đảm điểm tác vụ cao hơn; ở bài toán triage hẹp này, all-linear chỉ nhỉnh hơn nhẹ, còn rank lớn ở q,v không tạo ưu thế rõ ràng.

**4.2 — Sai thang learning rate.** `wrong_lr` chỉ đổi LR từ 1e-4 xuống 1e-5, nhưng train loss cuối tăng từ 0.6269 lên 1.5702. Khi đánh giá target, run này đạt 0.000 và format cũng là 0.000, trái ngược hoàn toàn với `correct`. Nếu chỉ nhìn một vài điểm loss mà không biết LR, em có thể quy lỗi cho rank hoặc vị trí adapter; đối chứng một-biến này cho thấy nguyên nhân chính là learning rate full fine-tuning quá nhỏ cho LoRA.

**4.3 — Trade-off QLoRA.** QLoRA giảm peak VRAM từ 8.78 GB xuống 3.86 GB, tiết kiệm 4.92 GB, tương đương khoảng 56%. Đổi lại, target giảm 0.030 (0.970 xuống 0.940), train loss tăng lên 0.7058, thời gian train tăng từ 382.5 s lên 454.0 s và latency tăng từ 1345.8 lên 1722.8 ms/mẫu. Vì format vẫn đạt 1.000, đây không phải lỗi định dạng mà là trade-off chất lượng/tốc độ để đổi lấy VRAM. Trên model và GPU này, số đo ủng hộ khuyến nghị không dùng QLoRA nếu bộ nhớ 16-bit LoRA vẫn đủ.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = -0.269` · `valid_trace_rate = 0.000`

Fine-tune thắng baseline prompt mạnh trên target triage, từ 0.765 lên 0.970, đồng thời duy trì format JSON hoàn chỉnh. Tuy nhiên, regression giảm từ 0.7911 xuống 0.5222, tức mất 0.269; mức giảm này vượt rất xa tolerance 0.020 của regression gate. Vì thế verdict FAILED không phải do mask, format hay target không học được, mà là do catastrophic forgetting ở năng lực phổ thông. Latency của LoRA cũng tăng 359.6 ms/mẫu, khoảng 36% so với baseline (b). Kết quả này trả lời đúng câu hỏi lab: bản fine-tune thắng ở tác vụ hẹp nhưng không thắng một cách đủ an toàn trên toàn bộ tiêu chí. Hướng thử tiếp theo là trộn 1–5% dữ liệu replay phổ thông vào train, sau đó đóng băng một baseline mới và đánh giá lại bốn nhóm bằng cùng giao thức.

## 6. Định tính — có cả ca đúng và ca sai

`qualitative.json` lưu điểm theo mẫu và dự đoán fine-tune đã rút gọn; pipeline chỉ lưu aggregate của baseline (b), không lưu toàn văn prediction (b) theo từng ticket. Vì vậy bảng dưới đây chỉ khẳng định những gì artefact ghi nhận, không bịa lại output của baseline.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
| ---: | --- | --- | --- | --- | --- |
| 1 | Chuột không dây VN232232, muốn trả lại gấp | `doi_tra`, `cao`, chuột không dây, `tich_cuc` | Không lưu per-case; aggregate target = 0.765 | 4/4 trường đúng | ✅ FT đúng hoàn toàn |
| 2 | Đèn bàn LED VN339109, vỡ khi nhận, gấp | `san_pham_loi`, `cao`, đèn bàn LED, `trung_tinh` | Không lưu per-case; aggregate target = 0.765 | 4/4 trường đúng | ✅ FT đúng hoàn toàn |
| 3 | Bình giữ nhiệt VN804124, chưa thấy tiền, khi nào tiện | `hoan_tien`, `thap`, bình giữ nhiệt, `tich_cuc` | Không lưu per-case; aggregate target = 0.765 | 3/4; dự đoán `urgency=trung_binh` | ❌ FT sai urgency |
| 4 | Nồi chiên không dầu DH249548, thiếu phụ kiện, khi nào tiện | `san_pham_loi`, `thap`, nồi chiên không dầu, `trung_tinh` | Không lưu per-case; aggregate target = 0.765 | 3/4; dự đoán `urgency=trung_binh` | ❌ FT sai urgency |
| 5 | Áo khoác gió VN613097, bị lỗi, khi nào tiện | `san_pham_loi`, `thap`, áo khoác gió, `tich_cuc` | Không lưu per-case; aggregate target = 0.765 | 3/4; dự đoán `urgency=trung_binh` | ❌ FT sai urgency |

Mẫu chung ở các ca FT sai là ticket chứa tín hiệu “khi nào tiện”, có nhãn urgency thấp, nhưng model lại dự đoán urgency trung bình. Đây là lỗi nhất quán trên một trường, không phải lỗi JSON hay nhận diện sản phẩm/intent.

## 7. Kết luận & điều em học được

Em không nên deploy bản fine-tune này nguyên trạng. Bản LoRA cho thấy nó học được tác vụ triage rất tốt: target tăng 0.205 tuyệt đối so với base model đã prompt tối ưu và tỉ lệ JSON hợp lệ vẫn là 1.000. Tuy nhiên, đây không phải là một chiến thắng tổng thể vì regression giảm 0.269 và vượt xa ngưỡng an toàn. Nếu chỉ báo cáo target hoặc train loss, em có thể kết luận nhầm rằng model đã tốt hơn. Kết quả này cho thấy đòn bẩy quan trọng nhất trước hết là dữ liệu và cách đánh giá: loss mask đúng làm kết quả train có nghĩa, còn regression set phát hiện trade-off mà target set không thể thấy. Về cấu hình, LR cũng là đòn bẩy mạnh: `wrong_lr` làm target và format về 0. Trong khi đó, khác biệt giữa all-linear và attention-only ở bộ dữ liệu nhỏ này chỉ 0.005 target, nên em không nên khái quát rằng một vị trí adapter luôn thắng tuyệt đối. Lần tiếp theo, em sẽ thêm replay data phổ thông, dùng `max_length=256` sau khi freeze lại baseline, rồi đánh giá lại trước khi cân nhắc deploy.

**Ba điều em học được:**

1. Loss mask phải được giải mã và kiểm chứng; chỉ bật cờ framework không đủ để chứng minh token nào đang chịu loss.
2. Train loss thấp không phải metric để xếp hạng model: `attn_only` có loss thấp hơn nhưng target thấp hơn `correct`.
3. Một baseline prompting mạnh là đối thủ thật của fine-tuning; kết quả deploy phải qua regression, format và latency chứ không chỉ target.

**Nếu có thêm 2 giờ nữa, em sẽ thử:** trộn 1–5% replay data phổ thông để giảm catastrophic forgetting, chạy lại full pipeline với `max_length=256`, rồi so sánh verdict mới với run hiện tại.

## Phụ lục — Thưởng đã làm

- [x] B1 NB6 merge + hot-swap
- [x] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [x] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [x] B4 quét rank có kiểm soát
- [x] B5 HuggingFace Hub — link: <https://huggingface.co/2khangnd/qwen35-4b-vi-customer-triage-lora>

### B1 — Merge & Phục vụ nhiều adapter (+3 điểm · deck §23)

- **Trạng thái**: Hoàn thành (`results/merge_check.json`, chạy qua `notebooks/06_merge_and_serve.py`).
- **Kết quả đo đạc**:
  - Điểm target trước merge: **0.970**
  - Điểm target sau merge (`model.merge_and_unload()`): **0.970**
  - Độ lệch $\Delta$: **0.000** (thỏa mãn ngưỡng an toàn $\Delta \ge -0.010$).
  - Đã thực hiện hot-swap thành công cả 3 adapter (`correct`, `attn_only`, `qlora`) trên cùng một base model đang nạp trong VRAM.
- **Trả lời câu hỏi lý thuyết deck §23**:
  - *Merge cho overhead suy luận bằng 0, nhưng bạn mất gì?*
    Khi merge vĩnh viễn trọng số adapter vào base (`W = W₀ + (α/r)·BA`), ta mất đi tính linh hoạt đa tác vụ (multi-tenancy). Mỗi mô hình sau merge trở thành một checkpoint cồng kềnh độc lập (hàng GB), không thể chia sẻ chung GPU VRAM cho các tác vụ/khách hàng khác nhau. Ngoài ra, việc merge cố định làm mất khả năng rollback nhanh khi adapter gặp sự cố hallucination hoặc trôi phân phối.
  - *Khi nào nên giữ adapter riêng dù chậm hơn một chút?*
    Nên giữ adapter riêng trong kiến trúc phục vụ Multi-LoRA Serving (như vLLM / SGLang): khi một hệ thống cần phục vụ đồng thời hàng chục nghiệp vụ hoặc khách hàng doanh nghiệp khác nhau (mỗi nghiệp vụ là một adapter 20–30 MB) trên cùng một base model nạp cố định trong VRAM. Tốc độ suy luận chỉ giảm nhẹ ở bước nhân ma trận phụ nhưng tiết kiệm hàng trăm GB VRAM và chi phí hạ tầng.

---

### B2 — Dataset miền riêng (+3 điểm · deck §17)

- **Trạng thái**: Hoàn thành (`data/custom_dataset.jsonl` gồm 250 mẫu, tài liệu đặc tả `data/CUSTOM_DATASET.md`).
- **Tóm tắt đặc tả**:
  - **Miền bài toán**: CSKH Ngân hàng số & Fintech Việt Nam (Digital Banking Triage: khóa thẻ khẩn cấp, khiếu nại giao dịch NAPAS/POS, hoàn phí thường niên, lỗi ứng dụng, xác thực sinh trắc học CCCD gắn chip NFC theo QĐ 2345/QĐ-NHNN).
  - **Quy trình khử nhiễm 3 lớp**:
    1. Khử trùng lặp chuỗi tuyệt đối bằng mã băm SHA-256.
    2. Kiểm soát độ tương đồng 8-gram MinHash / Jaccard similarity giữa train và eval dưới 0.15.
    3. Phân tách theo phiên giao dịch / thực thể khách hàng (`txn_id`), loại trừ hoàn toàn nguy cơ data leakage.
  - **Độ mới về phân phối (deck §3.3)**: Dữ liệu chứa các thực thể mới xuất hiện gần đây trong hệ thống tài chính Việt Nam (xác thực sinh trắc học NFC CCCD, hạn mức chuyển khoản trên 10 triệu) và ontology 4 trường chặt chẽ mà dữ liệu web tiền huấn luyện của base model chưa từng tiếp xúc.

---

### B3 — Reasoning-Trace Collapse (+4 điểm · deck §17.5)

- **Trạng thái**: Hoàn thành (`results/b3_reasoning_collapse.json`).
- **Bảng đối chứng giữa hai chế độ mask (chạy thực nghiệm trên Colab GPU T4)**:

| MASK_MODE | Target accuracy | valid_trace_rate | Regression accuracy |
| --- | ---: | ---: | ---: |
| `assistant-only` | **0.9700** | 0.0000 | 0.5222 |
| `response-only` | **0.9700** | 0.0000 | **0.6111** |

- **Phân tích hiện tượng & Trả lời câu hỏi deck §17.5 & §21**:
  - Khi huấn luyện với `MASK_MODE=response-only`, gradient loss chỉ phạt tập trung vào các token câu trả lời cuối cùng sau thẻ đóng `</think>`. Do tập seed không có reasoning trace mẫu, `valid_trace_rate` ở cả hai chế độ đều bằng 0.0000 (sụp đổ hoàn toàn thói quen suy luận trung gian).
  - Điểm target trên tập triage hẹp ở cả hai chế độ đều đạt mức đỉnh 0.9700 (97%). Nếu người kỹ sư chỉ nhìn vào `target`, họ sẽ ngộ nhận rằng hai cách mask này tương đương nhau.
  - Tuy nhiên, trên tập `regression` tổng quát, `response-only` đạt 0.6111 cao hơn so với 0.5222 của `assistant-only` (+8.89%). Nguyên nhân là do `response-only` không gò ép model phải học cả các token tiền xử lý/cấu trúc của trợ lý, giúp các trọng số tổng quát ít bị can thiệp quá mức.
  - **Ý nghĩa thực tiễn**: Đây là minh chứng rõ ràng cho kết luận ở Deck §21: nếu chỉ nhìn vào `target` hoặc `perplexity`, ta hoàn toàn mù trước sự biến đổi năng lực suy luận nền tảng của mô hình. Cổng hồi quy 4 nhóm là công cụ bắt buộc để phát hiện rủi ro này.

---

### B4 — Quét rank có kiểm soát (+3 điểm · deck §11)

- **Trạng thái**: Hoàn thành (`results/b4_rank_sweep.json`). Cố định vị trí `target_modules="text-linear"`, cùng LR $1\times 10^{-4}$, cùng 30 optimizer steps.
- **Bảng so sánh quét rank (chạy thực nghiệm trên Colab GPU T4)**:

| Rank ($r$) | Tham số trainable | LR | Steps | Target accuracy | Thời gian train (s) | Nhận xét |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 16,232,448 | 1e-4 | 30 | 0.8700 | 437.2 | Dung lượng nhỏ (16M), chưa đủ bao phủ ontology phức tạp |
| **16** | **32,464,896** | **1e-4** | **30** | **0.9700** | **382.5** | **Baseline `correct` tối ưu (vùng không hối tiếc)** |
| 64 | 129,859,584 | 1e-4 | 30 | 1.0000 | 414.5 | Dung lượng gấp 4 (130M), overfit 100% target, tiệm cận trần VRAM |

- **So sánh 3 nút vặn (Knob Hierarchy Ranking)**:
  1. **Hạng 1: Learning Rate (LR)** — $\Delta \text{target} = 0.9700$ (từ 0.970 ở 1e-4 sập về 0.000 ở 1e-5 trong run `wrong_lr`). Đây là nút vặn mang tính sống còn nhất: lệch thang LR thì mô hình không thể học.
  2. **Hạng 2: Rank (Dung lượng LoRA)** — $\Delta \text{target} = 0.1300$ ($r=8$ chỉ đạt 0.8700 do thiếu dung lượng; $r=16$ đạt 0.9700; $r=64$ bão hòa đạt 1.0000).
  3. **Hạng 3: Vị trí adapter (Placement)** — $\Delta \text{target} = 0.0050$ (all-linear 0.970 vs attn_only 0.965 khi đã khớp ngân sách tham số).
- **Kết luận bản chất của Rank (deck §11)**:
  Rank biểu thị **dung lượng biểu diễn so với lượng thông tin có trong dữ liệu huấn luyện**, hoàn toàn không phải là "nút vặn tăng chất lượng". Với tập dữ liệu triage 250 mẫu, rank $r=8$ bị thiếu tham số để học trọn vẹn cả 4 trường JSON dẫn đến target rơi về 0.87. Nâng lên $r=16$ là điểm ngọt tối ưu (0.970). Nâng lên $r=64$ làm tăng tham số lên 130 triệu (gấp 4 lần), tuy khớp 100% target tập mẫu nhưng ngốn sạch 14.5 GB VRAM GPU T4 và tiềm ẩn rủi ro overfit cao.

---

### B5 — HuggingFace Hub (+2 điểm)

- **Adapter công khai**: [Qwen3.5-4B Vietnamese customer-triage LoRA adapter](https://huggingface.co/2khangnd/qwen35-4b-vi-customer-triage-lora)
- **Checkpoints**: Đầy đủ `adapter_model.safetensors`, `adapter_config.json`, README hướng dẫn tải và nạp adapter qua thư viện `peft`.
