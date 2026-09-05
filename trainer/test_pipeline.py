import os
import sys
import shutil
import tempfile
import torch
import torch.nn as nn
from contextlib import nullcontext

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from model.model import ViMindConfig, ViMindForCausalLM
from trainer.pretrain import configure_optimizers


def log_test(step_num: int, title: str):
    print("\n" + "=" * 65)
    print(f"🧪 [TEST {step_num}/7] {title}")
    print("=" * 65)


def run_pipeline_test():
    print("=" * 65)
    print("🇻🇳 VIMIND FULL PIPELINE VERIFICATION TEST SUITE")
    print("=" * 65)

    # -------------------------------------------------------------
    # Test 1: Device & CUDA Kernel Execution
    # -------------------------------------------------------------
    log_test(1, "Device & CUDA Kernel Compatibility")
    device_str = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"Found CUDA GPU: {gpu_name} (Compute Capability {capability[0]}.{capability[1]})")
        try:
            # Test actual CUDA kernel execution (GEMM & Embedding)
            test_emb = nn.Embedding(100, 32).cuda()
            test_x = torch.randint(0, 100, (4, 16), device="cuda")
            test_out = test_emb(test_x)
            test_loss = test_out.sum()
            test_loss.backward()
            device_str = "cuda:0"
            print("✅ CUDA kernel execution PASSED without errors!")
        except Exception as e:
            print(f"⚠️ GPU execution failed ({e}). Falling back to CPU.")
            device_str = "cpu"
    else:
        print("CUDA not available. Running tests on CPU.")

    device = torch.device(device_str)
    print(f"Using test device: {device}")

    # -------------------------------------------------------------
    # Test 2: Tokenizer Loading & Vietnamese Roundtrip
    # -------------------------------------------------------------
    log_test(2, "Tokenizer Loading & Vietnamese Special Tokens")
    tokenizer_path = os.path.join(PROJECT_ROOT, "model")
    assert os.path.exists(tokenizer_path), f"Tokenizer dir not found: {tokenizer_path}"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size:,}")
    assert vocab_size == 12800, f"Expected 12,800 vocab, got {vocab_size}"

    test_vi_text = "ViMind là mô hình ngôn ngữ tiếng Việt 26M tham số siêu nhẹ."
    encoded = tokenizer.encode(test_vi_text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_vi_text}")
    print(f"Decoded:  {decoded}")
    assert test_vi_text in decoded, "Tokenizer decode did not match original text!"

    # Test Chat Template
    messages = [
        {"role": "user", "content": "Xin chào ViMind!"},
        {"role": "assistant", "content": "Chào bạn! Tôi có thể giúp gì?"}
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    assert "<s>user\n" in chat_prompt, "Missing user turn in chat template!"
    assert "<s>assistant\n" in chat_prompt, "Missing assistant turn in chat template!"
    assert "</s>" in chat_prompt, "Missing end-of-turn token in chat template!"
    print("✅ Tokenizer & Chat Template verified successfully!")

    # -------------------------------------------------------------
    # Test 3: ViMind Model Architecture & Parameter Count
    # -------------------------------------------------------------
    log_test(3, "ViMind Model Architecture & Weight Tying")
    config = ViMindConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1088,
        max_position_embeddings=512,
        dropout=0.0,
    )
    model = ViMindForCausalLM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    assert total_params == 26_224_128, f"Expected exactly 26,224,128 parameters, got {total_params}"

    # Verify weight tying
    assert model.lm_head.weight is model.model.embed_tokens.weight, "Weight tying failed: lm_head and embed_tokens do not share memory!"
    print("✅ Model architecture and weight tying verified successfully!")

    # -------------------------------------------------------------
    # Test 4: Forward & Backward Gradient Flow
    # -------------------------------------------------------------
    log_test(4, "Forward & Backward Gradient Flow with Optimizer")
    model.train()
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=5e-4)

    batch_size = 2
    seq_len = 64
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dummy_labels = dummy_input_ids.clone()

    optimizer.zero_grad()
    outputs = model(dummy_input_ids, labels=dummy_labels)
    loss = outputs.loss
    logits = outputs.logits

    print(f"Forward pass loss: {loss.item():.4f}")
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf!"
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Invalid logits shape: {logits.shape}"

    loss.backward()

    # Check gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient is None for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient has NaN for {name}"
            grad_count += 1

    optimizer.step()
    print(f"✅ Verified gradients for all {grad_count} trainable parameter tensors!")

    # -------------------------------------------------------------
    # Test 5: SFT Conversational Loss Masking Logic
    # -------------------------------------------------------------
    log_test(5, "SFT Conversational Loss Masking")
    # Simulate a user + assistant conversation
    user_prompt = "<s>user\nThủ đô Việt Nam là gì?</s>\n<s>assistant\n"
    assistant_resp = "Thủ đô của Việt Nam là Hà Nội.</s>"
    full_text = user_prompt + assistant_resp

    full_ids = tokenizer.encode(full_text)
    prompt_ids = tokenizer.encode(user_prompt)

    input_ids = torch.tensor([full_ids], device=device)
    labels = input_ids.clone()
    # Mask prompt
    labels[:, :len(prompt_ids)] = -100

    outputs_masked = model(input_ids, labels=labels)
    masked_loss = outputs_masked.loss
    print(f"Masked conversational loss: {masked_loss.item():.4f}")
    assert not torch.isnan(masked_loss), "Masked loss is NaN!"
    print("✅ SFT Conversational Loss Masking verified successfully!")

    # -------------------------------------------------------------
    # Test 6: Autoregressive Text Generation (KV Cache & Greedy)
    # -------------------------------------------------------------
    log_test(6, "Autoregressive Generation (KV Cache)")
    model.eval()
    prompt_text = "<s>user\nXin chào!</s>\n<s>assistant\n"
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generated = model.generate(
            prompt_tokens,
            max_new_tokens=15,
            temperature=0.7,
            top_k=20,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = generated[0, prompt_tokens.shape[1]:]
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"Generated tokens count: {len(new_tokens)}")
    print(f"Generated sample text:  '{gen_text}'")
    assert len(new_tokens) > 0, "No tokens generated!"
    print("✅ Autoregressive text generation verified successfully!")

    # -------------------------------------------------------------
    # Test 7: HuggingFace Checkpoint Save & Reload
    # -------------------------------------------------------------
    log_test(7, "Hugging Face Checkpoint Save & Reload")
    temp_dir = tempfile.mkdtemp(prefix="vimind_test_ckpt_")
    try:
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        print(f"Saved test checkpoint to: {temp_dir}")

        reloaded_model = ViMindForCausalLM.from_pretrained(temp_dir).to(device)
        reloaded_model.eval()

        with torch.no_grad():
            orig_logits = model(prompt_tokens).logits
            reloaded_logits = reloaded_model(prompt_tokens).logits

        diff = (orig_logits - reloaded_logits).abs().max().item()
        print(f"Logits maximum difference after reload: {diff:.8f}")
        assert diff < 1e-4, f"Reloaded model outputs differ too much: {diff}"
        print("✅ Checkpoint save and reload verified successfully!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # Test 8: DPO Preference Loss & Policy Optimization
    # -------------------------------------------------------------
    log_test(8, "DPO Preference Loss & Policy Optimization")
    from trainer.train_dpo import compute_logps, dpo_loss

    c_ids = torch.randint(10, 1000, (2, 32), device=device)
    c_lbl = c_ids.clone()
    c_lbl[:, :16] = -100

    r_ids = torch.randint(10, 1000, (2, 32), device=device)
    r_lbl = r_ids.clone()
    r_lbl[:, :16] = -100

    ref_model = ViMindForCausalLM(config).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    p_c_logps = compute_logps(model, c_ids, c_lbl)
    p_r_logps = compute_logps(model, r_ids, r_lbl)
    with torch.no_grad():
        r_c_logps = compute_logps(ref_model, c_ids, c_lbl)
        r_r_logps = compute_logps(ref_model, r_ids, r_lbl)

    dpo_l, c_rew, r_rew, acc, margin = dpo_loss(p_c_logps, p_r_logps, r_c_logps, r_r_logps, beta=0.1)
    assert not (torch.isnan(dpo_l) or torch.isinf(dpo_l)), "DPO loss is NaN or Inf!"
    dpo_l.backward()
    print(f"DPO Test Loss: {dpo_l.item():.4f} | Margin: {margin.item():+.4f}")
    print("✅ DPO preference loss and backward pass verified successfully!")

    print("\n" + "=" * 65)
    print("🎉 ALL 8 PIPELINE TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_pipeline_test()
    if not success:
        sys.exit(1)
