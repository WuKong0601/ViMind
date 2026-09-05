import os
import sys
import time
import torch

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from transformers import AutoTokenizer
from model.model import ViMindForCausalLM

from model.model import ViMindForCausalLM

DEFAULT_MODEL_PATH = "out/dpo/vimind_dpo_final" if os.path.exists("out/dpo/vimind_dpo_final") else "out/sft/vimind_sft_final"
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_benchmark():
    print("=" * 70)
    print(f"🚀 BẮT ĐẦU KIỂM THỬ TOÀN DIỆN MÔ HÌNH VIMIND ({'DPO FINAL - 1.0' if 'dpo' in MODEL_PATH else 'SFT FINAL'})")
    print("=" * 70)
    print(f"📍 Đường dẫn mô hình: {MODEL_PATH}")
    print(f"⚡ Thiết bị phần cứng: {DEVICE}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_start = torch.cuda.memory_allocated(0) / (1024 ** 2)
        print(f"🎮 GPU: {gpu_name}")

    start_load = time.time()
    tokenizer_dir = MODEL_PATH if os.path.exists(os.path.join(MODEL_PATH, "tokenizer_config.json")) else "model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    model = ViMindForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    load_time = time.time() - start_load

    num_params = sum(p.numel() for p in model.parameters())
    print(f"📦 Số lượng tham số: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"⏱️ Thời gian nạp mô hình: {load_time:.2f}s")
    
    if torch.cuda.is_available():
        vram_model = torch.cuda.memory_allocated(0) / (1024 ** 2) - vram_start
        print(f"💾 VRAM tiêu thụ: {vram_model:.1f} MB")
    print("=" * 70)

    test_categories = [
        {
            "category": "1. Khả năng Đọc hiểu / RAG (Context Extraction)",
            "tests": [
                {
                    "title": "Trích xuất năm thành lập",
                    "prompt": "Dựa vào đoạn văn sau: 'Trường Đại học Bách khoa Hà Nội được thành lập vào năm 1956 tại Hà Nội.'\nHỏi: Trường Đại học Bách khoa Hà Nội được thành lập vào năm nào?"
                },
                {
                    "title": "Trích xuất tên tác giả",
                    "prompt": "Dựa vào đoạn văn sau: 'Nguyễn Du là đại thi hào của dân tộc Việt Nam, tác giả của tác phẩm Truyện Kiều.'\nHỏi: Tác giả của tác phẩm Truyện Kiều là ai?"
                }
            ]
        },
        {
            "category": "2. Khả năng Giao tiếp & An ủi (Empathy & Role-play)",
            "tests": [
                {
                    "title": "Chia sẻ cảm xúc buồn",
                    "prompt": "Hôm nay tôi gặp nhiều chuyện buồn và mệt mỏi quá, bạn có thể cho tôi một lời khuyên hay động viên được không?"
                },
                {
                    "title": "Khả năng tự nhận diện",
                    "prompt": "Bạn có thể giới thiệu về bản thân và những điều bạn có thể làm được không?"
                }
            ]
        },
        {
            "category": "3. Tóm tắt & Rút trích ý chính (Summarization)",
            "tests": [
                {
                    "title": "Tóm tắt đoạn văn ngắn",
                    "prompt": "Hãy tóm tắt ngắn gọn ý chính của đoạn sau: 'Trí tuệ nhân tạo (AI) đang phát triển nhanh chóng và làm thay đổi mạnh mẽ nhiều ngành nghề, từ y tế, giáo dục cho đến tài chính và sản xuất.'"
                }
            ]
        },
        {
            "category": "4. Dịch thuật & Ngôn ngữ (Translation & Language)",
            "tests": [
                {
                    "title": "Dịch Việt -> Anh",
                    "prompt": "Hãy dịch câu sau sang tiếng Anh: 'Tôi yêu đất nước và con người Việt Nam.'"
                }
            ]
        }
    ]

    total_tokens_generated = 0
    total_generation_time = 0.0

    for cat in test_categories:
        print(f"\n{'#' * 70}")
        print(f"📂 PHÂN NHÓM: {cat['category']}")
        print(f"{'#' * 70}")

        for item in cat["tests"]:
            title = item["title"]
            prompt = item["prompt"]

            messages = [
                {"role": "system", "content": "Bạn là ViMind, một trợ lý AI thông minh và hữu ích được phát triển riêng cho tiếng Việt."},
                {"role": "user", "content": prompt}
            ]

            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=96,
                    temperature=0.6,
                    top_k=20,
                    top_p=0.85,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_time = time.time() - t0

            new_tokens = outputs[0][inputs.shape[1]:]
            num_tokens = len(new_tokens)
            speed = num_tokens / max(1e-4, gen_time)

            total_tokens_generated += num_tokens
            total_generation_time += gen_time

            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            print(f"\n📌 Bài test: {title}")
            print(f"❓ Câu hỏi:\n{prompt}")
            print(f"💬 ViMind phản hồi:\n{response_text}")
            print(f"⚡ Thống kê: {num_tokens} tokens | {gen_time:.2f}s ({speed:.1f} tokens/s)")
            print("-" * 60)

    avg_speed = total_tokens_generated / max(1e-4, total_generation_time)
    print("\n" + "=" * 70)
    print("🏁 TỔNG KẾT HIỆU NĂNG TOÀN DIỆN:")
    print(f"   • Tổng số token đã sinh: {total_tokens_generated}")
    print(f"   • Tổng thời gian sinh chữ: {total_generation_time:.2f}s")
    print(f"   • Tốc độ sinh chữ trung bình: {avg_speed:.1f} tokens/giây")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
