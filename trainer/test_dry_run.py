import os
import sys
import shutil
import torch

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from model.model import ViMindConfig, ViMindForCausalLM
from trainer.pretrain import configure_optimizers, save_checkpoint


def run_dry_run():
    print("=" * 60)
    print("🧪 RUNNING VIMIND PRE-TRAINING DRY RUN TEST")
    print("=" * 60)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            test_x = torch.zeros(1, device="cuda") + 1
            _ = test_x.cpu()
            device = torch.device("cuda:0")
        except Exception as e:
            print(f"⚠️ CUDA test failed ({e}). Falling back to CPU.")
    print(f"Device: {device}")

    # 1. Load Tokenizer
    print("\n[Step 1] Loading tokenizer from model/...")
    tokenizer = AutoTokenizer.from_pretrained("model")
    print(f"Tokenizer loaded successfully! Vocab: {len(tokenizer):,}")

    # 2. Initialize Model
    print("\n[Step 2] Initializing ViMind 26M architecture...")
    config = ViMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1088,
        max_position_embeddings=256,
        dropout=0.0,
    )
    model = ViMindForCausalLM(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    assert 25_000_000 <= total_params <= 27_000_000, f"Expected ~26M params, got {total_params}"
    print("✅ Parameter count verified (~26.2M)")

    # 3. Test Optimizer Configuration
    print("\n[Step 3] Configuring AdamW optimizer (weight decay decoupling)...")
    optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=5e-4)
    print(f"Optimizer configured with {len(optimizer.param_groups)} parameter groups.")

    # 4. Test 5 Training Steps with Synthetic Vietnamese Batch
    print("\n[Step 4] Running 5 forward-backward training steps...")
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Create synthetic batch
    sample_text = "Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á với bề dày lịch sử và văn hoá phong phú."
    sample_tokens = tokenizer(sample_text, return_tensors="pt").input_ids.to(device)
    labels = sample_tokens.clone()

    losses = []
    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32):
            outputs = model(sample_tokens, labels=labels)
            loss = outputs.loss

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        print(f"   Step {step}/5: Loss = {loss.item():.4f}")

    assert not torch.isnan(torch.tensor(losses)).any(), "Loss contains NaN!"
    print(f"✅ Forward & Backward passes verified! Initial loss: {losses[0]:.4f} -> Final step loss: {losses[-1]:.4f}")

    # 5. Test Checkpoint Saving & Loading
    print("\n[Step 5] Testing checkpoint saving & loading...")
    test_save_dir = "out/test_checkpoints"
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        config=config,
        optimizer=optimizer,
        scaler=scaler,
        epoch=1,
        step=5,
        save_dir=test_save_dir,
        prefix="vimind_dryrun",
    )

    ckpt_path = os.path.join(test_save_dir, "vimind_dryrun_step_5")
    assert os.path.exists(os.path.join(ckpt_path, "config.json")), "Missing config.json in checkpoint"
    assert os.path.exists(os.path.join(ckpt_path, "model.safetensors")) or os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")), "Missing model weights in checkpoint"
    print("✅ Checkpoint files verified!")

    # Test loading checkpoint
    print("\n[Step 6] Testing loading model from saved checkpoint...")
    loaded_model = ViMindForCausalLM.from_pretrained(ckpt_path).to(device)
    print("✅ Model loaded successfully from checkpoint!")

    # 6. Test Autoregressive Generation
    print("\n[Step 7] Testing autoregressive generation with loaded model...")
    loaded_model.eval()
    prompt = "Xin chào"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_tokens = loaded_model.generate(
        input_ids=prompt_ids,
        max_new_tokens=20,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.decode(gen_tokens[0].tolist(), skip_special_tokens=True)
    print(f"Generated text for prompt '{prompt}': {gen_text}")
    print("✅ Autoregressive text generation verified!")

    # Cleanup test dir
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
        print("\nCleaned up test checkpoints directory.")

    print("\n" + "=" * 60)
    print("🎉 ALL PHASE 3 VERIFICATION CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_dry_run()
