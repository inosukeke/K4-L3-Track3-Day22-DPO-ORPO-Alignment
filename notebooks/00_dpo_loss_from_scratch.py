# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB0 — DPO loss tự cài từ đầu (CPU, ~10 phút)
#
# **Không cần GPU.** Trước khi gọi `DPOTrainer`, bạn tự viết loss và kiểm tra nó
# trên số liệu đồ chơi. Phần này lấy từ lab K3 (tự cài DPO) và là nền để đọc
# đường cong reward ở NB3.
#
# Bạn sẽ thấy:
# 1. Tại bước 0 (mô hình đang học (policy) = reference) loss luôn bằng `log 2 ≈ 0.693`.
# 2. Gradient của DPO bị nhân với `sigmoid(-margin)`: cặp đã phân biệt tốt gần như không còn được học.
# 3. **Likelihood displacement**: loss vẫn giảm khi log-prob của *chosen* giảm, miễn rejected giảm nhanh hơn.
# 4. IPO, RPO, SimPO, ORPO khác DPO ở đâu, trên cùng một bộ số.

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

import math

import torch

from lab22 import dpo_math as M

torch.manual_seed(0)

# %% [markdown]
# ## 1. Log-prob của một câu trả lời
#
# `log π(y|x) = Σ_t log π(y_t | x, y_<t)`, chỉ cộng trên token của câu trả lời
# (mask = 1), không cộng trên câu hỏi.

# %%
vocab, length = 8, 5
logits = torch.randn(1, length, vocab)
labels = torch.randint(0, vocab, (1, length))
mask = torch.tensor([[0, 0, 1, 1, 1]])  # 2 token prompt, 3 token trả lời
total, mean = M.sequence_logps(logits, labels, mask)
print(f"sum log p = {total.item():.3f}   mean log p = {mean.item():.3f}")

# %% [markdown]
# ## 2. Bài tập: tự viết DPO loss
#
# Công thức (Rafailov et al. 2023):
#
# $$\mathcal{L} = -\log\sigma\Big(\beta\big[(\log\pi_\theta(y_w) - \log\pi_{ref}(y_w)) - (\log\pi_\theta(y_l) - \log\pi_{ref}(y_l))\big]\Big)$$
#
# Điền hàm dưới đây. Ô kiểm tra sẽ so với bản tham chiếu trong `lab22/dpo_math.py`.


# %%
def my_dpo_loss(pc, pr, rc, rr, beta=0.1):
    """pc/pr: policy log-prob chosen/rejected; rc/rr: reference. Trả về loss trung bình."""
    chosen_reward = beta * (pc - rc)
    rejected_reward = beta * (pr - rr)
    loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward)
    return loss.mean()


# %%
pc, pr = torch.tensor([-12.0, -30.0]), torch.tensor([-15.0, -28.0])
rc, rr = torch.tensor([-13.0, -29.0]), torch.tensor([-14.0, -29.0])
ref_loss, _, _ = M.dpo_loss(pc, pr, rc, rr, beta=0.1)
mine = my_dpo_loss(pc, pr, rc, rr, beta=0.1)
if mine is None:
    print(f"Chưa cài my_dpo_loss. Đáp số tham chiếu: {ref_loss.item():.4f}")
else:
    assert torch.allclose(torch.as_tensor(mine), ref_loss, atol=1e-6), (mine, ref_loss)
    print(f"✓ Khớp tham chiếu: {ref_loss.item():.4f}")

# %% [markdown]
# ## 3. Bước 0: mô hình đang học (policy) = reference ⇒ loss = log 2
#
# NB3 khởi tạo mô hình đang học (policy) bằng chính mô hình SFT (LoRA mới có trọng số B = 0), nên
# reward ngầm định ban đầu bằng 0 và loss bắt đầu ở 0.693. Nếu log của bạn
# không bắt đầu gần 0.693, reference đang không phải mô hình SFT.

# %%
same = torch.tensor([-20.0, -35.0])
loss0, cr0, rr0 = M.dpo_loss(same, same - 3, same, same - 3)
print(f"loss at init = {loss0.item():.4f}   log 2 = {math.log(2):.4f}   rewards = {cr0.tolist()}, {rr0.tolist()}")

# %% [markdown]
# ## 4. Trọng số gradient = sigmoid(−margin)

# %%
for margin in (-2.0, 0.0, 2.0, 5.0):
    m = torch.tensor(margin, requires_grad=True)
    loss = -torch.nn.functional.logsigmoid(m)
    loss.backward()
    print(f"margin {margin:+.1f}: loss {loss.item():.3f}   |dL/dmargin| {abs(m.grad.item()):.3f}")

# %% [markdown]
# ## 5. Likelihood displacement bằng số
#
# Hai kịch bản đều làm margin tăng 2 nat. Loss giống hệt nhau, nhưng ở kịch
# bản B log-prob của câu *được chọn* lại giảm. DPO không phân biệt được hai
# trường hợp này; chỉ đường cong `rewards/chosen` ở NB3 cho bạn biết.

# %%
ref_c, ref_r = torch.tensor([-20.0]), torch.tensor([-22.0])
scenarios = {
    "A: chosen ↑, rejected ↓": (ref_c + 1, ref_r - 1),
    "B: chosen ↓, rejected ↓↓": (ref_c - 3, ref_r - 5),
}
for name, (pc_, pr_) in scenarios.items():
    loss, cr, rj = M.dpo_loss(pc_, pr_, ref_c, ref_r, beta=1.0)
    print(f"{name:28s} loss {loss.item():.3f}  reward chosen {cr.item():+.1f}  rejected {rj.item():+.1f}")

# %% [markdown]
# **RPO** thêm NLL của câu chosen vào loss: kịch bản B bị phạt vì chosen bị đẩy xuống.

# %%
for name, (pc_, pr_) in scenarios.items():
    nll = -pc_ / 10  # NLL trung bình trên 10 token
    print(f"{name:28s} RPO loss {M.rpo_loss(pc_, pr_, ref_c, ref_r, nll, beta=1.0).item():.3f}")

# %% [markdown]
# ### Trả lời: vì sao margin có thể tăng trong khi log-prob của `chosen` lại giảm?
#
# Margin = `rewards/chosen - rewards/rejected`, với mỗi reward là `beta * (log π_θ(y) - log π_ref(y))`.
# Loss `-log σ(margin)` chỉ "nhìn" vào **hiệu số** này, không ràng buộc riêng từng số hạng. Vì vậy có hai
# cách để margin tăng:
#
# 1. **Đúng kỳ vọng (INTENDED):** `chosen` tăng, `rejected` giảm.
# 2. **Dịch chuyển xác suất (likelihood displacement):** cả hai cùng giảm, nhưng `rejected` giảm *nhanh
#    hơn* `chosen`. Ví dụ kịch bản B ở trên: `chosen` giảm 3 nat, `rejected` giảm 5 nat — margin vẫn tăng
#    2 nat giống kịch bản A (chosen tăng 1, rejected giảm 1), loss giống hệt nhau.
#
# Nguyên nhân: gradient của DPO đẩy xác suất **tương đối** giữa hai câu, không có lực nào giữ xác suất
# tuyệt đối của `chosen` không giảm. Nếu hai câu có nhiều token chung ở đầu (ví dụ cùng mở đầu câu), giảm
# xác suất của token chung đó làm giảm log-prob của *cả hai* câu, nhưng nếu `rejected` nhạy hơn với thay
# đổi đó (ví dụ dài hơn, nhiều token "đặc trưng cho rejected" hơn) thì nó giảm nhanh hơn — margin vẫn tăng
# dù `chosen` không được mô hình "ưa" hơn một cách tuyệt đối. Đây là lý do NB3 phải vẽ riêng đường
# `rewards/chosen` thay vì chỉ nhìn margin: margin tăng không tự động nghĩa là mô hình trả lời tốt hơn.

# %% [markdown]
# ## 6. Bốn biến thể trên cùng một cặp
#
# | Loss | Cần mô hình tham chiếu (reference)? | Chuẩn hoá độ dài? | Ghi chú |
# |---|---|---|---|
# | DPO (sigmoid) | có | không | mức cơ sở (baseline) |
# | IPO | có | có (TRL chia theo số token) | hồi quy margin về 1/(2β), chống quá khớp khi dữ liệu gần như tất định |
# | RPO | có | không | DPO + NLL(chosen), giảm likelihood displacement |
# | SimPO | không | có | log-prob trung bình + margin γ |
# | ORPO | không | có | NLL(chosen) + λ·log-odds-ratio, gộp SFT và sở thích vào một bước |
#
# NB3b huấn luyện thật các biến thể này (TRL `loss_type` và `trl.experimental.orpo`).

# %%
n_tokens_c, n_tokens_r = 40, 120  # chosen ngắn, rejected dài
pc_, pr_ = torch.tensor([-48.0]), torch.tensor([-130.0])
rc_, rr_ = torch.tensor([-50.0]), torch.tensor([-128.0])
avg_c, avg_r = pc_ / n_tokens_c, pr_ / n_tokens_r
print(f"DPO   {M.dpo_loss(pc_, pr_, rc_, rr_)[0].item():.4f}")
print(f"IPO   {M.ipo_loss(pc_, pr_, rc_, rr_, n_tokens_c, n_tokens_r).item():.4f}")
print(f"SimPO {M.simpo_loss(avg_c, avg_r).item():.4f}")
print(f"ORPO  {M.orpo_loss(avg_c, avg_r, -avg_c).item():.4f}")

# %% [markdown]
# **Câu hỏi cho REFLECTION §3:** tổng log-prob của câu dài luôn âm hơn câu ngắn.
# Vì sao điều đó khiến DPO gốc dễ thiên vị độ dài, và SimPO/ORPO xử lý bằng cách nào?
# Gợi ý: NB2 in ra tỉ lệ cặp có chosen dài hơn rejected trong dữ liệu tiếng Việt.
#
# **Trả lời:** `log π(y) = Σ_t log π(y_t | ...)` là **tổng** trên toàn bộ token của câu trả lời, và mỗi
# số hạng đều âm (log của một xác suất < 1). Câu dài hơn cộng nhiều số hạng âm hơn nên tổng log-prob của
# nó luôn âm hơn (nhỏ hơn) một câu ngắn, bất kể câu đó "hay" hay "dở" theo nghĩa con người. DPO gốc dùng
# trực tiếp tổng này làm reward (`beta * (log π_θ - log π_ref)`), nên nếu dữ liệu có xu hướng `chosen` dài
# hơn `rejected` (NB2 đo tỉ lệ này), mô hình có một đường tắt: **giảm xác suất câu ngắn** (`rejected`,
# ít token để "chia đều" sự sụt giảm) thì margin cũng tăng mà không cần thực sự viết hay hơn — đây chính
# là dịch chuyển xác suất (likelihood displacement) ở trên, và nó tương quan với độ dài.
#
# SimPO và ORPO tránh lỗi này bằng cách dùng **log-prob trung bình trên token** (`avg_logp = log π(y) / |y|`)
# thay vì tổng — chia cho số token khử đi phần chênh lệch chỉ do độ dài, nên reward phản ánh "mức tự tin
# trung bình mỗi token" chứ không phải "tổng điểm cộng dồn". Đổi lại, cả hai không cần mô hình tham chiếu
# (reference-free): SimPO thêm một margin cố định `gamma` để thay vai trò neo mà `log π_ref` từng giữ,
# còn ORPO cộng thêm NLL(chosen) để vẫn giữ mô hình bắt chước tốt câu `chosen` (giống SFT) song song với
# việc tối đa hoá log-odds-ratio.
