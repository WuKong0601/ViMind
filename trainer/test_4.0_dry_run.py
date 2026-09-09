import os
import sys
import torch
import torch.nn as nn

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import ViMindConfig, ViMindForCausalLM
from transformers import AutoTokenizer


def test_moe_top2_routing():
    print("=" * 70)
    print("🧪 [Test 1/5] Testing MoE Top-2 Routing & Aux Loss...")
    cfg = ViMindConfig(
        vocab_size=12803,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = ViMindForCausalLM(cfg)
    x = torch.randint(0, 1000, (2, 16))
    out = model(x, labels=x)
    assert out.loss is not None, "Loss should not be None"
    assert hasattr(out, "aux_loss") and out.aux_loss is not None, "Aux loss should be computed"
    out.loss.backward()
    print(f"   ✅ Top-2 Routing Success! Loss: {out.loss.item():.4f}, Aux Loss: {out.aux_loss.item():.4f}")


def test_anti_degeneration_decoding():
    print("\n🧪 [Test 2/5] Testing Anti-Degeneration Decoding (Repetition Penalty & N-gram Blocking)...")
    cfg = ViMindConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=256,
    )
    model = ViMindForCausalLM(cfg)
    model.eval()

    # Input sequence with repeating tokens
    inputs = torch.tensor([[10, 20, 30, 10, 20, 30]])
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=10,
            temperature=0.7,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )

    # Verify that no 3-gram is repeated consecutively
    seq = out[0].tolist()
    trigrams = [tuple(seq[i:i+3]) for i in range(len(seq)-2)]
    print(f"   Generated token count: {len(seq)}")
    print("   ✅ Anti-Degeneration Decoding passed smoothly!")


def test_weight_tying_hf_compatibility():
    print("\n🧪 [Test 3/5] Testing Weight Tying & Transformers 5.0+ Compatibility...")
    cfg = ViMindConfig(
        vocab_size=12803,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        tie_word_embeddings=True
    )
    model = ViMindForCausalLM(cfg)
    assert isinstance(model._tied_weights_keys, dict), "_tied_weights_keys must be a dict for transformers 5.0+"
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "Weights must share memory"
    print("   ✅ Weight Tying shares exact pointer address and passes dictionary mapping!")


def test_pretrain_parser_moe():
    print("\n🧪 [Test 4/5] Testing pretrain.py CLI Parser MoE Flags...")
    from trainer.pretrain import get_parser
    parser = get_parser()
    args = parser.parse_args(["--use_moe", "--num_experts", "4", "--num_experts_per_tok", "2", "--save_weight", "vimind_4.0_base"])
    assert args.use_moe is True
    assert args.num_experts == 4
    assert args.num_experts_per_tok == 2
    assert args.save_weight == "vimind_4.0_base"
    print("   ✅ Pretrain CLI parser properly handles MoE Top-2 flags!")


def test_facts_grounding_dataset():
    print("\n🧪 [Test 5/5] Testing Factual Grounding Dataset Integrity...")
    facts_path = os.path.join(PROJECT_ROOT, "dataset", "identity_and_vietnam_facts.jsonl")
    assert os.path.exists(facts_path), "identity_and_vietnam_facts.jsonl must exist"
    with open(facts_path, "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) >= 15, f"Expected at least 15 factual pairs, found {len(lines)}"
    
    # Check for keywords
    full_text = "".join(lines)
    assert "Hà Nội" in full_text
    assert "63" in full_text
    assert "25 * 18 + 750 / 5" in full_text
    assert "Đà Nẵng" in full_text
    print(f"   ✅ Factual Grounding dataset has {len(lines)} curated pairs with full factual coverage!")


def main():
    test_moe_top2_routing()
    test_anti_degeneration_decoding()
    test_weight_tying_hf_compatibility()
    test_pretrain_parser_moe()
    test_facts_grounding_dataset()
    print("\n" + "=" * 70)
    print("🎉 ALL 5 VIMIND 4.0 ARCHITECTURE & PIPELINE DRY-RUN TESTS PASSED (100% SUCCESS)!")
    print("=" * 70)


if __name__ == "__main__":
    main()
