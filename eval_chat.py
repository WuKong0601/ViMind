import os
import sys
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="ViMind Interactive CLI Chat Assistant")
    parser.add_argument("--model_path", type=str, default="out/sft/vimind_sft_final", help="Path to trained model directory")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cuda:0 or cpu)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate per response")
    parser.add_argument("--system_prompt", type=str, default="Bạn là ViMind, một trợ lý AI thông minh và hữu ích được phát triển riêng cho tiếng Việt.", help="System prompt")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 65)
    print("🇻🇳 ViMind: Trợ Lý Trí Tuệ Nhân Tạo Tiếng Việt (Interactive CLI)")
    print("=" * 65)

    # 1. Load Tokenizer
    tokenizer_dir = args.model_path if os.path.exists(os.path.join(args.model_path, "tokenizer_config.json")) else "model"
    print(f"📦 Đang nạp tokenizer từ: {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 2. Load Model
    if os.path.exists(args.model_path):
        print(f"🧠 Đang nạp trọng số mô hình từ: {args.model_path}...")
        model = ViMindForCausalLM.from_pretrained(args.model_path).to(device)
    else:
        print(f"⚠️ Không tìm thấy checkpoint tại '{args.model_path}'.")
        print(f"   Khởi tạo mô hình kiến trúc ViMind 26M thử nghiệm...")
        config = ViMindConfig(vocab_size=len(tokenizer))
        model = ViMindForCausalLM(config).to(device)

    model.eval()
    print(f"✅ Mô hình sẵn sàng trên {args.device}! Bắt đầu trò chuyện.")
    print("💡 Mẹo: Gõ '/clear' để làm mới cuộc trò chuyện, '/exit' để thoát.\n")

    history = [{"role": "system", "content": args.system_prompt}]

    while True:
        try:
            user_input = input("\n👤 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Tạm biệt!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
            print("\n👋 Tạm biệt bạn!")
            break

        if user_input.lower() in ["/clear", "clear"]:
            history = [{"role": "system", "content": args.system_prompt}]
            print("🧹 Đã làm mới lịch sử hội thoại.")
            continue

        # Append user message
        history.append({"role": "user", "content": user_input})

        # Format prompt with chat template
        prompt_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        print("🤖 ViMind: ", end="", flush=True)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

        # Extract generated response text
        new_tokens = output_ids[0][input_ids.shape[1] :]
        response_text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True).strip()

        # Update history
        history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
