import os
import sys
import torch

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from transformers import AutoTokenizer
from model.model import ViMindForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ckpt_sft = "out/sft/vimind_sft_final"
ckpt_dpo = "out/dpo/vimind_dpo_final"

tokenizer = AutoTokenizer.from_pretrained("model")

test_prompts = [
    "Xin chào, bạn là ai?",
    "Thủ đô của Việt Nam là gì?",
    "Học máy là gì?"
]

for ckpt_path, label in [(ckpt_sft, "SFT FINAL"), (ckpt_dpo, "VIMIND 1.0 (DPO FINAL)")]:
    if not os.path.exists(ckpt_path):
        print(f"Skipping {label}, path not found: {ckpt_path}")
        continue
    
    print("\n" + "=" * 60)
    print(f"🧪 Testing Checkpoint: {label} ({ckpt_path})")
    print("=" * 60)
    
    model = ViMindForCausalLM.from_pretrained(ckpt_path).to(device)
    model.eval()

    for p in test_prompts:
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=60,
                temperature=0.6,
                top_k=20,
                top_p=0.85,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        reply = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"\n👤 Người dùng: {p}")
        print(f"🤖 ViMind ({label}): {reply.strip()}")
        print("-" * 50)
