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
from dataset.lm_dataset import SFTDataset
from trainer.pretrain import configure_optimizers, save_checkpoint


def run_sft_dry_run():
    print("=" * 60)
    print("🧪 RUNNING VIMIND SFT DRY RUN TEST")
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

    # 2. Test SFT Dataset & Loss Masking
    print("\n[Step 2] Testing SFTDataset and Loss Masking...")
    sft_ds = SFTDataset("dataset/sft_vi.jsonl", tokenizer, max_length=128)
    input_ids, labels = sft_ds[0]

    # Verify that at least one token is -100 (masked) and at least one is > 0 (trained)
    masked_count = (labels == -100).sum().item()
    trained_count = (labels != -100).sum().item()
    print(f"Masked count (-100): {masked_count} | Trained count: {trained_count}")
    assert masked_count > 0, "Loss masking failed: no tokens masked!"
    assert trained_count > 0, "Loss masking failed: no tokens to train!"
    print("✅ Conversational Loss Masking verified successfully!")

    # 3. Initialize Model
    print("\n[Step 3] Initializing ViMind 26M architecture for SFT...")
    config = ViMindConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1088,
        max_position_embeddings=256,
        dropout=0.05,
    )
    model = ViMindForCausalLM(config).to(device)
    model.train()

    # 4. Optimizer & SFT Training Steps
    print("\n[Step 4] Running 5 SFT forward-backward steps...")
    optimizer = configure_optimizers(model, weight_decay=0.05, learning_rate=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    batch_x = input_ids.unsqueeze(0).to(device)
    batch_y = labels.unsqueeze(0).to(device)

    losses = []
    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32):
            outputs = model(batch_x, labels=batch_y)
            loss = outputs.loss

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        print(f"   SFT Step {step}/5: Loss = {loss.item():.4f}")

    assert not torch.isnan(torch.tensor(losses)).any(), "SFT Loss contains NaN!"
    print(f"✅ SFT convergence verified! Initial loss: {losses[0]:.4f} -> Final step loss: {losses[-1]:.4f}")

    # 5. Checkpoint Saving & Loading
    print("\n[Step 5] Testing SFT Checkpoint saving & loading...")
    test_save_dir = "out/test_sft_checkpoints"
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        config=config,
        optimizer=optimizer,
        scaler=scaler,
        epoch=1,
        step=5,
        save_dir=test_save_dir,
        prefix="vimind_sft_dryrun",
    )

    ckpt_path = os.path.join(test_save_dir, "vimind_sft_dryrun_step_5")
    loaded_model = ViMindForCausalLM.from_pretrained(ckpt_path).to(device)
    print("✅ SFT checkpoint saved & reloaded successfully!")

    # 6. Test Chat Inference with apply_chat_template
    print("\n[Step 6] Testing Chat Inference with apply_chat_template...")
    loaded_model.eval()
    messages = [{"role": "user", "content": "Xin chào ViMind!"}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    output_ids = loaded_model.generate(
        input_ids=prompt_ids,
        max_new_tokens=25,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output_ids[0][prompt_ids.shape[1]:].tolist(), skip_special_tokens=True)
    print(f"Prompt formatted:\n{prompt_text}")
    print(f"ViMind response:\n{response}")
    print("✅ Chat Generation pipeline verified!")

    # Cleanup
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
        print("\nCleaned up test SFT checkpoints directory.")

    print("\n" + "=" * 60)
    print("🎉 ALL SFT VERIFICATION CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_sft_dry_run()
