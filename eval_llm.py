import os
import sys
import time
import argparse
import random
import warnings
import torch

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from transformers import AutoTokenizer, TextStreamer
from model.model import ViMindConfig, ViMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings("ignore")


def load_model_and_tokenizer(args):
    print(f"Loading tokenizer from: {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    print(f"Loading ViMind model from: {args.model_path}...")
    if os.path.exists(args.model_path):
        model = ViMindForCausalLM.from_pretrained(args.model_path)
    else:
        print(f"[WARN] Path {args.model_path} not found. Loading ViMind 64M preset for testing...")
        cfg = ViMindConfig.get_config_64m(vocab_size=len(tokenizer))
        model = ViMindForCausalLM(cfg)

    # Optional LoRA adapter injection
    if args.lora_path and args.lora_path != "None" and os.path.exists(args.lora_path):
        print(f"Injecting and loading LoRA adapter from: {args.lora_path}...")
        apply_lora(model, rank=args.lora_rank)
        load_lora(model, args.lora_path)

    device = torch.device(args.device if "cuda" in args.device and torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model successfully loaded on {device}! Total parameters: {total_params / 1e6:.2f}M")
    return model, tokenizer, device


def main():
    parser = argparse.ArgumentParser(description="ViMind Model Evaluation & Interactive Inference")
    parser.add_argument("--model_path", default="out/vimind_64m_final", type=str, help="Path to model directory")
    parser.add_argument("--tokenizer_dir", default="model", type=str, help="Path to tokenizer directory")
    parser.add_argument("--lora_path", default="None", type=str, help="Path to LoRA adapter weights (.pth)")
    parser.add_argument("--lora_rank", default=16, type=int, help="Rank of LoRA adapter")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="Maximum generated tokens")
    parser.add_argument("--temperature", default=0.7, type=float, help="Sampling temperature")
    parser.add_argument("--top_p", default=0.85, type=float, help="Nucleus sampling top-p")
    parser.add_argument("--top_k", default=40, type=int, help="Top-k sampling")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--stream", action="store_true", default=True, help="Stream generated tokens in real time")
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args)

    default_prompts = [
        "Xin chào! Bạn là ai?",
        "Thủ đô của Việt Nam là gì?",
        "Giải thích ngắn gọn học máy (Machine Learning) là gì?",
        "Hãy viết một hàm Python để tính giai thừa của một số nguyên.",
        "Tại sao bầu trời lại có màu xanh?",
        "Việt Nam có bao nhiêu tỉnh thành?",
    ]

    print("\n" + "=" * 60)
    print("🤖 CHƯƠNG TRÌNH ĐÁNH GIÁ MÔ HÌNH VIMIND")
    print("   [0] Chạy bộ câu hỏi chuẩn đánh giá tự động (Benchmark Prompts)")
    print("   [1] Trò chuyện tương tác trực tiếp (Interactive Chat CLI)")
    print("=" * 60)

    try:
        mode_input = input("Chọn chế độ [0/1] (Mặc định 0): ").strip()
        mode = int(mode_input) if mode_input in ["0", "1"] else 0
    except Exception:
        mode = 0

    prompts = default_prompts if mode == 0 else iter(lambda: input("\n👤 Người dùng: "), "")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if args.stream else None

    conversation_history = []

    for prompt in prompts:
        if not prompt or not prompt.strip():
            break

        if mode == 0:
            print(f"\n👤 Câu hỏi: {prompt}")

        conversation_history.append({"role": "user", "content": prompt.strip()})
        chat_prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

        print("🤖 ViMind: ", end="", flush=True)
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                streamer=streamer,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - start_time
        gen_tokens = outputs.size(1) - inputs["input_ids"].size(1)
        speed = gen_tokens / max(1e-4, elapsed)

        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True).strip()
        if not args.stream:
            print(response_text)

        print(f"\n⚡ Tốc độ sinh: {speed:.1f} tokens/s ({gen_tokens} tokens trong {elapsed:.2f}s)")
        print("-" * 60)

        conversation_history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
