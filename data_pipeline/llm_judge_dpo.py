import os
import sys
import re
import json
import time
import argparse
from typing import Dict, Any, List, Optional
import torch

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from transformers import AutoTokenizer, AutoModelForCausalLM


JUDGE_SYSTEM_PROMPT = """Bạn là một chuyên gia ngôn ngữ học tiếng Việt và là một giám khảo thẩm định AI khách quan, công tâm bậc nhất.
Nhiệm vụ của bạn là đọc một câu hỏi và so sánh hai phương án trả lời (A và B), sau đó lựa chọn phương án nào xuất sắc và đáng tin cậy hơn.

Các tiêu chí thẩm định:
1. ĐỘ CHÍNH XÁC THỰC TẾ (Factual Accuracy): Tuyệt đối không chứa ảo giác, sai lệch thông tin địa lý, lịch sử, con người hoặc khoa học (ví dụ: Thủ đô Việt Nam phải là Hà Nội). Nếu một câu trả lời sai sự thật, câu đó PHẢI BỊ XẾP KÉM HƠN.
2. TÍNH HỮU ÍCH & MẠCH LẠC: Câu trả lời đi thẳng vào vấn đề, phân tích logic, định dạng danh sách rõ ràng, văn phong tiếng Việt tự nhiên và chuẩn mực.

Quy tắc bắt buộc:
Hãy đưa ra nhận xét ngắn gọn và ở dòng cuối cùng, PHẢI ghi chính xác:
KẾT QUẢ: [[A]] nếu phương án A tốt hơn
hoặc
KẾT QUẢ: [[B]] nếu phương án B tốt hơn"""


def parse_judgment(judge_text: str) -> Optional[str]:
    """Extracts [[A]] or [[B]] judgment using robust regex."""
    match = re.search(r"KẾT QUẢ:\s*\[\[([AB])\]\]", judge_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\[\[([AB])\]\]", judge_text)
    if match:
        return match.group(1).upper()
    if "phương án a tốt hơn" in judge_text.lower() or "chọn a" in judge_text.lower():
        return "A"
    if "phương án b tốt hơn" in judge_text.lower() or "chọn b" in judge_text.lower():
        return "B"
    return None


def run_judge_pipeline(args):
    print("=" * 70)
    print("🧑‍⚖️ VIMIND 3.0: LLM-AS-A-JUDGE DPO DATASET GENERATION")
    print("=" * 70)
    print(f"   Judge Model: {args.model_name_or_path}")
    print(f"   4-bit Quantization: {args.load_in_4bit}")
    print(f"   Device: {args.device}")
    print(f"   Input Candidates: {args.input_data}")
    print(f"   Output DPO: {args.output_path}")
    print("=" * 70)

    # 1. Load Tokenizer & Model
    print(f"[1/3] Nạp Tokenizer & Mô hình Giám khảo từ: {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    model_kwargs = {"trust_remote_code": True}
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            print("   ⚡ Đã kích hoạt 4-bit NF4 Quantization (Tiết kiệm VRAM, chạy mượt mà trên T4/RTX 3050)!")
        except ImportError:
            print("   ⚠️ bitsandbytes chưa cài đặt. Chuyển sang nạp float16 thông thường.")
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = args.device
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if "cuda" in args.device:
            model_kwargs["device_map"] = args.device

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs).eval()
    print("   ✅ Mô hình Giám khảo đã sẵn sàng!")

    # 2. Read Candidate Pairs
    print(f"[2/3] Đọc dữ liệu ứng cử viên từ: {args.input_data}...")
    candidate_samples = []
    if os.path.exists(args.input_data):
        with open(args.input_data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    candidate_samples.append(data)
                except Exception:
                    continue
    else:
        print(f"   ⚠️ Tệp {args.input_data} không tồn tại. Tạo 5 mẫu minh họa kiểm thử...")
        candidate_samples = [
            {
                "prompt": "Thủ đô của Việt Nam là gì?",
                "response_a": "Thủ đô của Việt Nam là thành phố Hà Nội. Đây là trung tâm chính trị, văn hóa và kinh tế quan trọng của cả nước.",
                "response_b": "Thủ đô của Pháp nằm ở Tây Ban Nha, nằm ở khu vực trung tâm của Pháp.",
            },
            {
                "prompt": "Giải thích ngắn gọn Trí tuệ nhân tạo (AI) là gì?",
                "response_a": "AI (Artificial Intelligence) là một nhánh của khoa học máy tính nhằm tạo ra các hệ thống có khả năng thực hiện các tác vụ đòi hỏi trí tuệ của con người như học hỏi, suy luận, nhận dạng hình ảnh và giải quyết vấn đề.",
                "response_b": "Trí tuệ nhân tạo là một lĩnh vực trí tuệ nhân tạo sử dụng trí tuệ nhân tạo để làm những việc trí tuệ nhân tạo.",
            },
            {
                "prompt": "Việt Nam có bao nhiêu tỉnh thành?",
                "response_a": "Tính đến hiện tại, Việt Nam có tổng cộng 63 tỉnh và thành phố trực thuộc Trung ương (gồm 58 tỉnh và 5 thành phố trực thuộc Trung ương).",
                "response_b": "Việt Nam có khoảng 500 tỉnh thành trên khắp thế giới.",
            },
        ]

    total_candidates = min(len(candidate_samples), args.max_samples)
    print(f"   Tổng số cặp cần thẩm định: {total_candidates}")

    # 3. Judge & Output DPO Dataset
    print(f"[3/3] Bắt đầu chấm điểm và phân loại chosen / rejected...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out_file = open(args.output_path, "w", encoding="utf-8")
    
    labeled_count = 0
    start_time = time.time()

    for i, item in enumerate(candidate_samples[:total_candidates]):
        prompt = item.get("prompt") or item.get("query")
        resp_a = item.get("response_a") or item.get("candidate_1")
        resp_b = item.get("response_b") or item.get("candidate_2")

        if not prompt or not resp_a or not resp_b:
            continue

        user_content = f"""[CÂU HỎI]: {prompt}

[PHƯƠNG ÁN A]:
{resp_a}

[PHƯƠNG ÁN B]:
{resp_b}"""

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                top_p=0.8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        judge_reply = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        verdict = parse_judgment(judge_reply)

        if verdict == "A":
            chosen = resp_a
            rejected = resp_b
        elif verdict == "B":
            chosen = resp_b
            rejected = resp_a
        else:
            # Fallback to A if ambiguous
            chosen = resp_a
            rejected = resp_b

        dpo_entry = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "judge": "Qwen2.5-7B-Instruct",
            "verdict": verdict,
            "judge_reasoning": judge_reply,
        }
        out_file.write(json.dumps(dpo_entry, ensure_ascii=False) + "\n")
        out_file.flush()
        labeled_count += 1

        if (i + 1) % args.log_interval == 0 or (i + 1) == total_candidates:
            elapsed = time.time() - start_time
            speed = labeled_count / max(1e-4, elapsed)
            print(f"   [{labeled_count}/{total_candidates}] Đã thẩm định: Prompt: '{prompt[:40]}...' -> Quyết định: [[{verdict}]] ({speed:.2f} pairs/s)")

    out_file.close()
    print("=" * 70)
    print(f"🎉 Hoàn thành thẩm định! Đã ghi {labeled_count} mẫu DPO sạch ra: {args.output_path}")
    print("=" * 70)


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind 3.0: LLM-as-a-Judge Offline DPO Generation")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path or HuggingFace repo for Judge")
    parser.add_argument("--input_data", type=str, default="dataset/candidate_pairs.jsonl", help="Input candidate pairs jsonl")
    parser.add_argument("--output_path", type=str, default="dataset/dpo_qwen_judged.jsonl", help="Output clean DPO jsonl")
    parser.add_argument("--max_samples", type=int, default=10000, help="Max pairs to evaluate")
    parser.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit BitsAndBytes quantization")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--log_interval", type=int, default=10, help="Log step interval")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run_judge_pipeline(args)
