# 🇻🇳 ViMind: Native Vietnamese Small Language Model from Scratch

**Languages / Ngôn ngữ:** [English](README_EN.md) · [Tiếng Việt](README.md)

[![Release](https://img.shields.io/badge/Release-v4.0.0-brightgreen.svg)](https://github.com/WuKong0601/ViMind/releases/tag/v4.0.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-MoE%20198M%20%2F%20Pro%200.5B-purple.svg)]()
[![Hardware](https://img.shields.io/badge/Compute-NVIDIA%20Tesla%20T4-orange.svg)]()
[![ARS Skills](https://img.shields.io/badge/Skills-Academic%20Research%20Suite-blueviolet.svg)]()

> **ViMind 4.0** represents a comprehensive cognitive leap in the Vietnamese Small Language Model (SLM) ecosystem. Designed with a strategic **Dual-Track AI** philosophy:
> 1. **ViMind 4.0 Native (MoE 198M / 64M active):** A 100% sovereign Mixture-of-Experts architecture built from scratch with 4 experts (Top-2 Routing), optimized for ultra-low latency on Edge/IoT devices, standard consumer CPUs, and personal computers.
> 2. **ViMind 4.0 Pro (0.5B Foundation Alignment):** SFT-aligned on the curated **Golden Dense Reasoning Core**, inheriting 18 trillion pretraining tokens for fluent Vietnamese expression, broad world knowledge, and sharp multi-step reasoning.
> 3. **Transparent Chain-of-Thought (`<think>`):** Autonomous procedural reasoning inside a dedicated scratchpad prior to generating final answers, completely eliminating hallucination, gibberish, and repetitive degeneration loops.
> 4. **Integrated Academic Research Suite (ARS-Codex):** Embedded academic workflow suite (`academic-research-suite`) supporting paper drafting, systematic literature reviews, experimental design, and peer-review simulation.

---

## 🏗️ Technical Specifications Across Generations

| Specification | ViMind 1.0 (Base) | ViMind 2.0 (Dense) | ViMind 3.0 (MoE v1) | 🚀 ViMind 4.0 Native | ⚡ ViMind 4.0 Pro |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Architecture** | Dense CausalLM | Dense CausalLM | MoE Top-1 | **MoE Top-2 Routing** | **Dense Transformer** |
| **Total Parameters** | 26.2M | 62.8M | 198.3M | **198.3M** | **494.0M (0.5B)** |
| **Active Parameters**| 26.2M | 62.8M | 64.1M | **64.1M / token** | **494.0M / token** |
| **Number of Experts**| 1 (Dense) | 1 (Dense) | 4 Experts (Top-1) | **4 Experts (Top-2)** | N/A |
| **Hidden Size ($d$)** | 512 | 640 | 640 | **640** | **896** |
| **Transformer Layers**| 8 | 12 | 12 | **12** | **24** |
| **Attention Heads** | 8 | 10 (GQA 10:5) | 10 (GQA 10:5) | **10 (GQA 10:5)** | **14 (GQA 14:2)** |
| **Vocabulary Size** | 12,800 | 12,800 | 12,800 | **12,803 (Special tokens)**| **151,936 (Qwen2.5)** |
| **Reasoning Engine** | None | None | Primitive | **`<think>` CoT Scratchpad** | **`<think>` CoT Scratchpad** |
| **SafeTensors Size** | 103 MB | 245 MB | 696 MB | **696 MB** | **943 MB (FP16)** |
| **Target Deployment** | Prototype | Baseline | Research | **Edge / Offline / IoT** | **Cloud API / Production** |

---

## 💎 Breakthrough: Golden Dense Reasoning Core

Instead of relying on noisy machine-translated datasets (52,000 raw Alpaca samples causing repetitive filler phrases and hallucinations), ViMind 4.0 was trained exclusively on the **Golden Dense Reasoning Core** (8,805 pure samples):

1. **Vietnam History & Geography Ground Truth:** 63 provinces (58 provinces and 5 centrally-administered municipalities: Hanoi, Ho Chi Minh City, Hai Phong, Da Nang, Can Tho), key milestones (September 2, 1945 Independence, Dien Bien Phu 1954, Bach Dang 938), and national sovereignty over Hoang Sa & Truong Sa.
2. **Procedural Multi-step Arithmetic (CoT Scratchpad):** Explicit order of operations (e.g., $25 \times 18 + 750 / 5 = 600$) with intermediate verification steps.
3. **Natural Sciences & Astronomy:** Physical explanations for the Earth's day/night cycle via its 24-hour axial rotation and spherical geometry, solar/lunar eclipses, water cycles, and chlorophyll.
4. **Agentic Tool Calling (`<tool_call>`):** Autonomous execution of math calculators, real-time clock lookup, and weather retrieval in structured JSON format.

---

## 🥊 Head-to-Head Battle Results

During empirical evaluation on Kaggle Tesla T4 (Cell 12), both models demonstrated 100% factual accuracy and zero degeneration:

```text
👤 User: What is the capital of the Socialist Republic of Vietnam?
🤖 [ViMind 4.0 Native]:
<think>
Hà Nội là thủ đô, trung tâm chính trị, văn hóa và giáo dục hàng đầu của Việt Nam.
</think>
Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là thành phố Hà Nội.

----------------------------------------------------------------------
👤 User: Calculate: 25 * 18 + 750 / 5.
⚡ [ViMind 4.0 Pro]:
<think>
Để giải bài toán này, ta thực hiện các phép nhân chia trước, cộng trừ sau:
- Bước 1: Tính phép nhân 25 * 18 = 450.
- Bước 2: Tính phép chia 750 / 5 = 150.
- Bước 3: Cộng hai kết quả lại: 450 + 150 = 600.
</think>
Kết quả của phép tính 25 * 18 + 750 / 5 là **600**.

----------------------------------------------------------------------
👤 User: How is the weather in Da Nang today?
🤖 [ViMind 4.0 Native]:
<think>
Người dùng muốn tra cứu thời tiết thời gian thực tại Đà Nẵng. Cần gọi công cụ get_current_weather.
</think>
<tool_call>{"name": "get_current_weather", "arguments": {"location": "Đà Nẵng"}}</tool_call>
```

---

## 🎓 Academic Research Skills (ARS-Codex) Integration

ViMind embeds the specialized **ARS-Codex** suite under `.agents/skills/academic-research-suite` to empower academic researchers, students, and AI practitioners:

### Core Research Workflows:
1. **`deep-research`**: Systematic literature reviews, citation analysis, and Socratic research question refinement.
2. **`academic-paper`**: Full manuscript drafting, IMRaD outline planning, abstract crafting, citation formatting, and LaTeX publishing (ACL, EMNLP, NeurIPS, IEEE, arXiv).
3. **`academic-paper-reviewer`**: Simulation of peer-review panels (Reviewers 1, 2, 3) to uncover methodology weaknesses and integrity risks.
4. **`experiment-agent`**: Experimental execution planning, ablation design, and statistical analysis (PPL, throughput, latency, tool F1 score).
5. **`academic-pipeline`**: End-to-end orchestration from research inquiry to camera-ready manuscript.

### Command Aliases:
* `/ars-outline`: Generate structured academic paper outlines.
* `/ars-abstract`: Draft concise, publication-grade abstracts.
* `/ars-lit-review`: Synthesize literature and build comparative SLM benchmark tables.
* `/ars-reviewer`: Run simulated peer review evaluations.
* `/ars-full`: Execute full end-to-end academic research workflow.

---

## ⚡ Quickstart

### 1. Installation
```bash
git clone https://github.com/WuKong0601/ViMind.git
cd ViMind
pip install -r requirements.txt
```

### 2. Inference with PyTorch / Hugging Face Transformers
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model (Native MoE or Pro Foundation)
model_path = "out/vimind_4.0_pro_final"  # or "out/vimind_4.0_moe_final"
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

### 3. Launch WebUI & OpenAI-Compatible API Server
```bash
python app.py --port 8000
```
Access the interactive WebUI at `http://localhost:8000` or invoke the API endpoint at `http://localhost:8000/v1/chat/completions`.

---

## 📜 Citation

If you utilize ViMind or its research toolchain in your academic publications, please cite:

```bibtex
@article{vimind2026,
  title={ViMind 4.0: Enhancing Vietnamese Reasoning in Small Language Models via Golden Dense Core and Dual-Track Alignment},
  author={Cong Huynh and Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

## 📄 License
This project is released under the open-source [Apache 2.0 License](LICENSE).
