import os
import sys
import math
import torch
import torch.nn as nn

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import (
    ViMindConfig,
    ViMindModel,
    ViMindForCausalLM,
    precompute_freqs_cis,
    MOEFeedForward,
    FeedForward,
)


def log_test(num, name):
    print(f"\n[{num}/5] 🧪 Testing: {name}...")


def test_config():
    log_test(1, "ViMindConfig Dense & MoE Presets")
    cfg_dense = ViMindConfig.get_config_64m(vocab_size=12800)
    assert not cfg_dense.use_moe
    assert cfg_dense.rope_theta == 1e6

    cfg_moe = ViMindConfig.get_config_moe_198m(vocab_size=12800)
    assert cfg_moe.use_moe
    assert cfg_moe.num_experts == 4
    assert cfg_moe.num_experts_per_tok == 1
    assert cfg_moe.rope_theta == 1e6
    print("   ✅ Dense and MoE configuration presets verified!")


def test_yarn_rope():
    log_test(2, "YaRN RoPE Context Extrapolation (up to 32k)")
    dim = 64
    # Test standard RoPE
    cos_std, sin_std = precompute_freqs_cis(dim=dim, end=2048, theta=1e6)
    assert cos_std.shape == (2048, dim)
    assert sin_std.shape == (2048, dim)

    # Test YaRN scaled RoPE up to 32,768 tokens
    rope_scaling = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 2048,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "attention_factor": 1.0,
    }
    cos_yarn, sin_yarn = precompute_freqs_cis(
        dim=dim, end=32768, theta=1e6, rope_scaling=rope_scaling
    )
    assert cos_yarn.shape == (32768, dim)
    assert sin_yarn.shape == (32768, dim)
    assert not torch.isnan(cos_yarn).any()
    assert not torch.isnan(sin_yarn).any()
    print("   ✅ YaRN RoPE tables successfully generated for 32,768 context length!")


def test_moe_feedforward():
    log_test(3, "MOEFeedForward Routing & Aux Loss")
    cfg = ViMindConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=1,
        router_aux_loss_coef=0.01,
    )
    moe_layer = MOEFeedForward(cfg)
    moe_layer.train()

    x = torch.randn(2, 16, 256)
    out = moe_layer(x)
    assert out.shape == (2, 16, 256)
    assert not torch.isnan(out).any()
    assert hasattr(moe_layer, "aux_loss")
    assert moe_layer.aux_loss.item() > 0.0

    # Verify backward pass works without error
    loss = out.sum() + moe_layer.aux_loss
    loss.backward()
    print("   ✅ MOEFeedForward forward, aux_loss, and backward pass verified!")


def test_causal_lm_moe():
    log_test(4, "ViMindForCausalLM MoE Architecture & Causal Loss")
    cfg = ViMindConfig(
        vocab_size=500,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=1,
    )
    model = ViMindForCausalLM(cfg)
    model.train()

    input_ids = torch.randint(0, 500, (2, 32))
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)

    assert outputs.loss is not None
    assert not torch.isnan(outputs.loss)
    assert hasattr(outputs, "aux_loss")
    assert outputs.logits.shape == (2, 32, 500)

    # Test backward pass
    outputs.loss.backward()
    print("   ✅ ViMindForCausalLM MoE end-to-end forward/backward verified!")


def test_generation():
    log_test(5, "ViMind Generation with KV Cache")
    cfg = ViMindConfig(
        vocab_size=500,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=1,
    )
    model = ViMindForCausalLM(cfg).eval()
    prompt = torch.tensor([[10, 20, 30]])
    gen = model.generate(prompt, max_new_tokens=10, temperature=0.7, top_p=0.9)
    assert gen.shape == (1, 13)
    print("   ✅ Generation with KV Cache and MoE routing verified!")


if __name__ == "__main__":
    print("=" * 65)
    print("🚀 VIMIND 3.0 CORE ARCHITECTURE DRY RUN TEST SUITE")
    print("=" * 65)
    test_config()
    test_yarn_rope()
    test_moe_feedforward()
    test_causal_lm_moe()
    test_generation()
    print("\n" + "=" * 65)
    print("🎉 ALL 5 TESTS PASSED FOR VIMIND 3.0 CORE ARCHITECTURE!")
    print("=" * 65)
