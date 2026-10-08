# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _<Họ Tên>_
**Khoá:** _<A20-K4 / ...>_
**Tier đã chạy:** T4 (Colab miễn phí)
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4, 14.56 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch (max_len=768) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 66% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm-panel: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B — **chưa chạy xong** (xem ghi chú ở §4) |
| Chi phí | 0 đồng (Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~27 phút (quan sát trực tiếp, 100 bước) |
| VRAM cao nhất | không đo chính xác (phiên Colab bị ngắt giữa chừng); bước chấm điểm NB4 sau đó báo `CUDA out of memory` khi tải thêm reward model, cho thấy mức dùng đã gần sát trần 14.56 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0967 |
| Độ chính xác reward trên held-out | 0.64 |
| Margin trên held-out | 0.0838 |
| Chẩn đoán tự động (`diagnosis`) | **INTENDED** |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | chưa tính (cần `judge_summary.json`, xem §4) |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

_Mô tả riêng `rewards/chosen` và `rewards/rejected` trên **train và held-out**. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?_

Cả hai đường `rewards/chosen` và `rewards/rejected` đều **tăng** trong suốt quá trình huấn luyện, trên cả tập train
lẫn held-out — không có dấu hiệu dịch chuyển xác suất (likelihood displacement), vì nếu có thì `chosen` phải giảm
trong khi `rejected` giảm nhanh hơn nó. Ở đây `chosen` luôn tăng nhanh hơn `rejected` (kết thúc ở khoảng 0.41 so với
0.30 trên train, 0.41 so với 0.33 trên held-out), nên margin đi lên đều đặn: từ ~0 ở bước đầu lên ~0.097 (train) và
~0.084 (held-out) ở bước 100. Đường margin trên tập train khá nhiễu (dao động 0.03–0.06 trước khi tăng vọt cuối),
trong khi đường held-out mượt hơn nhưng đi **cùng hướng** và giá trị cuối gần với train (0.084 so với 0.097) — không
có khoảng cách lớn giữa hai đường, nên không phải hiện tượng học thuộc (overfit): mô hình học được một quy luật tổng
quát hoá sang câu hỏi chưa thấy, chứ không chỉ nhớ các cặp đã huấn luyện. Chẩn đoán tự động "INTENDED" khớp hoàn toàn
với quan sát trực quan trên đồ thị: đây là kịch bản DPO hoạt động đúng như lý thuyết mô tả.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | | | | | | | |
| hữu ích — helpfulness (4) | | | | | | | |
| an toàn — safety (4) | | | | | | | |

Giám khảo: ______ · sanity accuracy: ______ · `score_length_spearman` (reward model) hoặc độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): ______

> **⚠️ CHƯA HOÀN THÀNH — lý do thật:** cell chấm điểm tự động của NB4 (tải hội đồng 2 reward model) báo
> `CUDA out of memory` (Colab T4 hết quota GPU miễn phí ngay sau đó nên chưa chạy lại được). Bảng điểm phía trên và
> `data/eval/judge_summary.json` / `side_by_side.jsonl` **chưa có số liệu thật** — sẽ bổ sung ngay khi quota GPU được
> cấp lại, không điền số giả vào đây.

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

Dù chưa có điểm giám khảo, bước **sinh câu trả lời** (§1 của NB4) đã chạy xong cho cả 8 câu cố định và 50 câu
held-out, với `04-side-by-side-table.png` đã lưu được. Quan sát định tính đáng chú ý: với giải mã tham lam (greedy),
câu trả lời của SFT và SFT+DPO trên **cả 8 câu cố định đều gần như giống hệt nhau về nội dung** — ví dụ câu hữu ích
h1 ("Giải thích quicksort"), cả hai bản đều trả lời đúng định nghĩa chia-để-trị, cùng cấu trúc câu, chỉ khác vài từ ở
đoạn cuối; câu an toàn s3 ("mua rượu khi 14 tuổi"), cả hai đều từ chối với lý do giống nhau (vi phạm pháp luật, có
hại sức khoẻ), không có khác biệt đáng kể. Điều này khớp với con số `eval_reward_gap ≈ 0.084` khá nhỏ ở NB3: margin
đủ để DPO thắng theo nghĩa xác suất (reward ngầm cao hơn), nhưng chưa đủ lớn để **lật token lớn nhất** (argmax) ở mỗi
bước giải mã tham lam trên những câu hỏi này — cần giám khảo so sánh log-prob hoặc lấy mẫu (sampling) nhiều lần mới
thấy khác biệt rõ, đúng là việc NB4 định làm nhưng chưa chạy xong. Một quan sát phụ: cả hai bản đều sinh ra token lạ
`</tool_call>`/`<tool_call>` ở đầu câu trả lời — đây là đặc điểm kế thừa từ bản SFT (do base model Qwen3 có huấn
luyện gọi công cụ), DPO không gây ra và cũng không sửa được lỗi này vì nó nằm ngoài phạm vi 800 cặp sở thích đã dùng.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Quyết định tôi muốn phân tích là giữ nguyên **β=0.1** (giá trị mặc định của lab) thay vì thử β nhỏ hơn (0.05) hoặc
lớn hơn (0.5) như gợi ý ở phần bonus β-sweep. Phương án thay thế rõ nhất là β=0.05: theo công thức
`reward = β·(log π_θ − log π_ref)`, β nhỏ hơn cho phép mô hình đi xa hơn khỏi mô hình tham chiếu SFT với cùng một
chênh lệch log-prob, nên về lý thuyết sẽ tạo ra khác biệt hành vi rõ rệt hơn giữa SFT và SFT+DPO. Tôi chọn giữ
β=0.1 vì đây là giá trị lab khuyến nghị cho cấu hình T4 (dữ liệu nhỏ, chỉ 800 cặp, 1 epoch, ~100 bước huấn luyện) —
với ngân sách tính toán hạn chế, một β "an toàn" giúp tránh rủi ro mô hình đi quá xa reference và hỏng (trường hợp
FAILURE/margin ≤ 0 trên held-out) chỉ sau một epoch. Kết quả xác nhận lựa chọn này là hợp lý theo nghĩa thống kê:
chẩn đoán INTENDED, margin held-out dương (0.084) và không bị overfit (held-out bám sát train). Tuy nhiên điều khiến
tôi bất ngờ là margin này tuy "đúng hướng" nhưng **quá nhỏ để thấy khác biệt thực tế** khi giải mã tham lam trên các
câu hỏi cố định ở NB4 (xem §4) — câu trả lời SFT và SFT+DPO gần như giống hệt nhau. Nếu làm lại, tôi sẽ ưu tiên chạy
β-sweep (β=0.05 so với 0.1) trước khi kết luận, vì rất có thể β=0.1 trên dữ liệu ít (800 cặp, 1 epoch) chỉ đủ để
dịch chuyển reward ngầm một chút chứ chưa đủ để thay đổi token được chọn ở bước giải mã — tức là "đúng về lý thuyết"
nhưng "chưa đủ mạnh về thực nghiệm" với ngân sách T4 hiện tại.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
