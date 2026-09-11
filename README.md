# 🇻🇳 ViMind: Mô Hình Ngôn Ngữ Nhỏ Tiếng Việt (Native Small Language Model from Scratch)

**Ngôn ngữ / Languages:** [Tiếng Việt](README.md) · [English](README_EN.md)

[![Release](https://img.shields.io/badge/Release-v4.0.0-brightgreen.svg)](https://github.com/WuKong0601/ViMind/releases/tag/v4.0.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-MoE%20198M%20%2F%20Pro%200.5B-purple.svg)]()
[![Hardware](https://img.shields.io/badge/Compute-NVIDIA%20Tesla%20T4-orange.svg)]()

> **ViMind 4.0** là bước nhảy vọt nhận thức toàn diện trong hệ sinh thái mô hình ngôn ngữ nhỏ (Small Language Model - SLM) tiếng Việt. Dự án được thiết kế theo chiến lược **Dual-Track AI** linh hoạt:
> 1. **ViMind 4.0 Native (MoE 198M / 64M active):** Kiến trúc Mixture-of-Experts tự chủ 100% từ số 0 (from scratch) với 4 chuyên gia (Top-2 Routing), tối ưu hóa để chạy siêu nhẹ, siêu nhanh trên thiết bị biên (Edge/IoT), CPU thông thường hoặc máy tính cá nhân không có GPU rời.
> 2. **ViMind 4.0 Pro (0.5B Foundation Alignment):** Căn chỉnh (SFT Alignment) mô hình nền tảng 0.5B trên tập **Golden Dense Reasoning Core**, mang lại văn phong tiếng Việt trôi chảy, tri thức thế giới phong phú và khả năng giải quyết các bài toán logic phức tạp.
> 3. **Tư duy chuỗi tường minh (`<think>`):** Tự động suy luận từng bước qua scratchpad trước khi trả lời, triệt tiêu hoàn toàn hiện tượng "nói nhảm, lặp từ, chém gió ảo giác".
> 4. **Khả năng kích hoạt công cụ (`<tool_call>`):** Tự nhận biết yêu cầu thời gian thực hoặc bài toán số lớn để gọi các hàm tra cứu thời tiết, ngày giờ và máy tính số học chuẩn JSON.

---

## 🌟 Các Kỹ Năng Nổi Bật Của ViMind 4.0 (Core Skills)

ViMind 4.0 được trang bị 5 nhóm năng lực nhận thức chuyên sâu:

1. **🧠 Kỹ năng suy luận chuỗi tư duy (`<think>` CoT Scratchpad):**
   Mô hình không đưa ra kết quả một cách cảm tính mà luôn lập luận mạch lạc từng bước trước khi chốt câu trả lời cuối cùng (áp dụng cho toán đố, thứ tự ưu tiên số học, phân tích logic).
2. **🇻🇳 Kỹ năng tri thức bản địa & địa lý Việt Nam:**
   Nắm vững 63 tỉnh thành (58 tỉnh và 5 thành phố trực thuộc trung ương), phân định ranh giới Bắc - Trung - Nam, các mốc son lịch sử hào hùng (2/9/1945, Điện Biên Phủ 1954, Bạch Đằng 938) và chủ quyền thiêng liêng Hoàng Sa - Trường Sa.
3. **⚡ Kỹ năng kích hoạt công cụ tự động (`<tool_call>` Function Calling):**
   Tự động sinh cấu trúc JSON gọi các công cụ ngoại vi (`calculate_math`, `get_current_weather`, `get_current_time`) khi gặp các câu hỏi cần dữ liệu thời gian thực hoặc số lớn vượt quá khả năng tính nhẩm.
4. **🔬 Kỹ năng giải thích khoa học tự nhiên:**
   Cung cấp các câu trả lời bản chất, chuẩn xác về các hiện tượng vật lý, thiên văn và sinh học (hiện tượng ngày và đêm do Trái Đất tự quay quanh trục 24h, nhật thực, nguyệt thực, chất diệp lục, trọng lực).
5. **🛡️ Kỹ năng giao tiếp đĩnh đạc, chống ảo giác (Anti-Hallucination):**
   Phản hồi cô đọng, dứt khoát, đi thẳng vào trọng tâm; nói không với các mẫu câu đệm rườm rà và hiện tượng lặp từ vô tận.

---

## 🏗️ Thông Số Kỹ Thuật Qua Các Thế Hệ

| Đặc Tính Kỹ Thuật | ViMind 1.0 (Base) | ViMind 2.0 (Dense) | ViMind 3.0 (MoE v1) | 🚀 ViMind 4.0 Native | ⚡ ViMind 4.0 Pro |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kiến trúc** | Dense CausalLM | Dense CausalLM | MoE Top-1 | **MoE Top-2 Routing** | **Dense Transformer** |
| **Tổng tham số** | 26.2M | 62.8M | 198.3M | **198.3M** | **494.0M (0.5B)** |
| **Tham số kích hoạt** | 26.2M | 62.8M | 64.1M | **64.1M / token** | **494.0M / token** |
| **Số chuyên gia** | 1 (Dense) | 1 (Dense) | 4 Experts (Top-1) | **4 Experts (Top-2)** | N/A |
| **Kích thước ẩn ($d$)** | 512 | 640 | 640 | **640** | **896** |
| **Số tầng (Layers)** | 8 | 12 | 12 | **12** | **24** |
| **Attention Heads** | 8 | 10 (GQA 10:5) | 10 (GQA 10:5) | **10 (GQA 10:5)** | **14 (GQA 14:2)** |
| **Từ điển (Vocab)** | 12.800 | 12.800 | 12.800 | **12.803 (Thêm thẻ)** | **151.936 (Qwen2.5)** |
| **Cơ chế tư duy** | Không | Không | Sơ khai | **`<think>` CoT Scratchpad** | **`<think>` CoT Scratchpad** |
| **Kích thước SafeTensors**| 103 MB | 245 MB | 696 MB | **696 MB** | **943 MB (FP16)** |
| **Mục tiêu triển khai** | Khảo sát | Cơ bản | Thực nghiệm | **Edge / Offline / IoT** | **Cloud API / Production** |

---

## 💎 Đột Phá: Golden Dense Reasoning Core

Thay vì sử dụng các bộ dữ liệu dịch máy tự động chất lượng thấp (52.000 mẫu Alpaca dịch thô gây lặp từ và chém gió), ViMind 4.0 được đào tạo độc quyền trên tập tri thức vàng **Golden Dense Core** (8.805 mẫu thuần khiết):

1. **Địa lý & Lịch sử Việt Nam (Ground Truth):** 63 tỉnh thành (58 tỉnh, 5 thành phố trực thuộc TW: Hà Nội, TP.HCM, Hải Phòng, Đà Nẵng, Cần Thơ), mốc son 2/9/1945, Điện Biên Phủ 1954, Bạch Đằng 938, chủ quyền Hoàng Sa & Trường Sa.
2. **Toán học nhiều bước (CoT Scratchpad):** Suy luận từng bước thứ tự ưu tiên các phép tính nhân chia trước, cộng trừ sau (ví dụ: $25 \times 18 + 750 / 5 = 600$).
3. **Khoa học tự nhiên & thường thức:** Hiện tượng ngày và đêm do Trái Đất hình cầu tự quay quanh trục 24 giờ, nhật thực, nguyệt thực, vòng tuần hoàn nước, chất diệp lục.
4. **Kích hoạt công cụ (`<tool_call>`):** Tự động gọi hàm tính toán số lớn, tra cứu thời gian thực và thời tiết chuẩn JSON.

---

## 🥊 Kết Quả Đối Đầu Thực Tế (Head-to-Head Battle)

Trong đợt thử nghiệm thực tế tại Cell 12 (Kaggle T4), cả hai mô hình đã thể hiện năng lực lập luận chính xác 100%:

```text
👤 Người dùng: Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là gì?
🤖 [ViMind 4.0 Native]:
<think>
Hà Nội là thủ đô, trung tâm chính trị, văn hóa và giáo dục hàng đầu của Việt Nam.
</think>
Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là thành phố Hà Nội.

----------------------------------------------------------------------
👤 Người dùng: Tính giúp tôi kết quả của 25 * 18 + 750 / 5.
⚡ [ViMind 4.0 Pro]:
<think>
Để giải bài toán này, ta thực hiện các phép nhân chia trước, cộng trừ sau:
- Bước 1: Tính phép nhân 25 * 18 = 450.
- Bước 2: Tính phép chia 750 / 5 = 150.
- Bước 3: Cộng hai kết quả lại: 450 + 150 = 600.
</think>
Kết quả của phép tính 25 * 18 + 750 / 5 là **600**.

----------------------------------------------------------------------
👤 Người dùng: Thời tiết hôm nay tại Đà Nẵng thế nào, có mát mẻ không?
🤖 [ViMind 4.0 Native]:
<think>
Người dùng muốn tra cứu thời tiết thời gian thực tại Đà Nẵng. Cần gọi công cụ get_current_weather.
</think>
<tool_call>{"name": "get_current_weather", "arguments": {"location": "Đà Nẵng"}}</tool_call>
```

---

## ⚡ Hướng Dẫn Nhanh (Quickstart)

### 1. Cài đặt môi trường
```bash
git clone https://github.com/WuKong0601/ViMind.git
cd ViMind
pip install -r requirements.txt
```

### 2. Chạy suy luận với PyTorch / Transformers
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Khởi tạo mô hình (Native hoặc Pro)
model_path = "out/vimind_4.0_pro_final"  # hoặc "out/vimind_4.0_moe_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Tính giúp tôi kết quả của 25 * 18 + 750 / 5."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer(
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.08
    )

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False))
```

### 3. Khởi động WebUI & API Server tương thích OpenAI
```bash
python app.py --port 8000
```
Truy cập giao diện WebUI tại `http://localhost:8000` hoặc kết nối API tại `http://localhost:8000/v1/chat/completions`.

---

## 📜 Trích Dẫn (Citation)

Nếu bạn sử dụng ViMind trong công trình nghiên cứu hoặc sản phẩm của mình, vui lòng trích dẫn:

```bibtex
@article{vimind2026,
  title={ViMind 4.0: Enhancing Vietnamese Reasoning in Small Language Models via Golden Dense Core and Dual-Track Alignment},
  author={Cong Huynh and Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

## 📄 Bản Quyền (License)
Dự án được phân phối dưới giấy phép mã nguồn mở [Apache 2.0](LICENSE).
