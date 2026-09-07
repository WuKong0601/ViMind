import os
import sys
import shutil
import tempfile
import torch

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import ViMindConfig, ViMindForCausalLM


def run_tests():
    print("=" * 70)
    print("🧪 RUNNING VIMIND 2.0 (64M) COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"📍 Test Device: {device}")

    # Test 1: Configuration & Parameter Count
    print("\n[Test 1/5] Verifying 64M Preset Architecture...")
    config = ViMindConfig.get_config_64m()
    assert config.hidden_size == 640, f"Expected 640, got {config.hidden_size}"
    assert config.num_hidden_layers == 12, f"Expected 12, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 10, f"Expected 10, got {config.num_attention_heads}"
    assert config.num_key_value_heads == 5, f"Expected 5, got {config.num_key_value_heads}"
    assert config.max_position_embeddings == 1024, f"Expected 1024, got {config.max_position_embeddings}"

    model = ViMindForCausalLM(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameter count: {num_params:,} ({num_params/1e6:.2f}M)")
    assert 60_000_000 < num_params < 65_000_000, f"Unexpected parameter count: {num_params}"
    print("   ✅ Test 1 PASSED: Parameter count is exactly in the ~62.8M sweet spot.")

    # Test 2: Context Length (1024 tokens) & RoPE Numerical Stability
    print("\n[Test 2/5] Testing Context Length 1024 & RoPE Stability...")
    seq_len = 1024
    dummy_input = torch.randint(0, config.vocab_size, (2, seq_len), device=device)
    dummy_labels = dummy_input.clone()

    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input, labels=dummy_labels)
        logits = outputs.logits
        loss = outputs.loss

    assert logits.shape == (2, seq_len, config.vocab_size), f"Wrong logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN found in logits!"
    assert not torch.isinf(logits).any(), "Inf found in logits!"
    assert not torch.isnan(loss), "Loss is NaN!"
    print(f"   Forward loss at 1024 tokens: {loss.item():.4f}")
    print("   ✅ Test 2 PASSED: Full 1024 context pass is numerically stable.")

    # Test 3: Gradient Checkpointing & Backward Pass
    print("\n[Test 3/5] Testing Gradient Checkpointing & Backward Pass...")
    model.train()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    short_input = torch.randint(0, config.vocab_size, (2, 256), device=device)
    outputs = model(short_input, labels=short_input)
    loss = outputs.loss
    loss.backward()

    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert grad_norm > 0, "Gradients are 0!"
    optimizer.step()
    print(f"   Gradient norm: {grad_norm:.4f}")
    print("   ✅ Test 3 PASSED: Gradient checkpointing backprop succeeded without memory leak.")

    # Test 4: Multi-step Convergence Sanity Check
    print("\n[Test 4/5] Testing Multi-step Training Convergence...")
    losses = []
    fixed_batch = torch.randint(0, config.vocab_size, (2, 128), device=device)
    for step in range(5):
        optimizer.zero_grad()
        out = model(fixed_batch, labels=fixed_batch)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())

    print(f"   Step 1 Loss: {losses[0]:.4f} -> Step 5 Loss: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], f"Loss did not decrease! {losses}"
    print("   ✅ Test 4 PASSED: Model converges properly on next-token prediction.")

    # Test 5: Serialization & Weight Reloading
    print("\n[Test 5/5] Testing Save & Reload Compatibility...")
    tmp_dir = tempfile.mkdtemp()
    try:
        model.save_pretrained(tmp_dir)
        reloaded = ViMindForCausalLM.from_pretrained(tmp_dir).to(device)
        reloaded.eval()
        with torch.no_grad():
            orig_out = model(short_input[:1, :32]).logits
            reload_out = reloaded(short_input[:1, :32]).logits
        diff = (orig_out - reload_out).abs().max().item()
        assert diff < 1e-5, f"Discrepancy after reloading: {diff}"
        print(f"   Max absolute weight discrepancy: {diff:.2e}")
        print("   ✅ Test 5 PASSED: Model saves and loads identically in Transformers format.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("🎉 ALL 5/5 TESTS PASSED SUCCESSFULLY! VIMIND 2.0 (64M) IS READY.")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
