import sys
import os
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from model.model import ViMindConfig, ViMindForCausalLM

def test_agent_bounds():
    print("Testing Agent Rollout & Token Bounds...")
    tokenizer = AutoTokenizer.from_pretrained("model")
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")

    config = ViMindConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = ViMindForCausalLM(config)
    model.eval()

    # Simulate prompt with toolcall special tokens (index 12800..12802)
    sample_text = "<s>user\nTính 15 + 20<tool_call>\n"
    tokens = tokenizer(sample_text, return_tensors="pt")
    input_ids = tokens["input_ids"]
    print("Max token in input:", input_ids.max().item(), "Vocab size:", config.vocab_size)
    assert input_ids.max().item() < config.vocab_size, f"Token out of bounds! {input_ids.max().item()} >= {config.vocab_size}"

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.7,
            top_p=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    print("Generated token count:", out.shape[1])
    print("Max token in output:", out.max().item())
    assert out.max().item() < config.vocab_size, f"Generated token out of bounds! {out.max().item()} >= {config.vocab_size}"
    print("[PASS] Agent Rollout & Token Bounds test passed 100%!")

if __name__ == "__main__":
    test_agent_bounds()
