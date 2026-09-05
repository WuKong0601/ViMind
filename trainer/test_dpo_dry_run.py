import os
import sys
import shutil
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import ViMindConfig, ViMindForCausalLM
from dataset.lm_dataset import DPODataset
from trainer.train_dpo import compute_logps, dpo_loss


def run_dpo_dry_run():
    print("=" * 65)
    print("🧪 RUNNING VIMIND DPO DRY RUN TEST")
    print("=" * 65)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Tokenizer
    print("\n[Step 1] Loading tokenizer from model/...")
    tokenizer = AutoTokenizer.from_pretrained("model")
    print(f"✅ Tokenizer loaded successfully! Vocab: {len(tokenizer):,}")

    # 2. Test DPODataset
    print("\n[Step 2] Testing DPODataset and loss masking on dataset/dpo_vi.jsonl...")
    dataset = DPODataset(
        data_path="dataset/dpo_vi.jsonl",
        tokenizer=tokenizer,
        max_length=256,
    )
    print(f"   Total dataset samples: {len(dataset):,}")
    c_in, c_lbl, r_in, r_lbl = dataset[0]

    c_masked = (c_lbl == -100).sum().item()
    c_trained = (c_lbl != -100).sum().item()
    r_masked = (r_lbl == -100).sum().item()
    r_trained = (r_lbl != -100).sum().item()

    print(f"   Chosen: Masked (-100): {c_masked} | Trained: {c_trained}")
    print(f"   Rejected: Masked (-100): {r_masked} | Trained: {r_trained}")
    assert c_trained > 0, "Error: Chosen response has 0 trained tokens!"
    assert r_trained > 0, "Error: Rejected response has 0 trained tokens!"
    print("✅ DPODataset loss masking verified successfully!")

    # 3. Initialize Models
    print("\n[Step 3] Initializing Policy and Reference models (ViMind 26M)...")
    if os.path.exists("out/sft/vimind_sft_final"):
        print("   Loading weights from out/sft/vimind_sft_final...")
        policy_model = ViMindForCausalLM.from_pretrained("out/sft/vimind_sft_final").to(device)
        ref_model = ViMindForCausalLM.from_pretrained("out/sft/vimind_sft_final").to(device)
    else:
        print("   Initializing fresh ViMind config...")
        config = ViMindConfig(vocab_size=len(tokenizer))
        policy_model = ViMindForCausalLM(config).to(device)
        ref_model = ViMindForCausalLM(config).to(device)

    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    policy_model.train()

    # 4. Test DPO Forward and Backward
    print("\n[Step 4] Running 5 DPO forward-backward optimization steps...")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-4)

    initial_loss = None
    final_loss = None

    for step, (c_ids, c_labels, r_ids, r_labels) in enumerate(loader, start=1):
        if step > 5:
            break

        c_ids, c_labels = c_ids.to(device), c_labels.to(device)
        r_ids, r_labels = r_ids.to(device), r_labels.to(device)

        policy_c_logps = compute_logps(policy_model, c_ids, c_labels)
        policy_r_logps = compute_logps(policy_model, r_ids, r_labels)

        with torch.no_grad():
            ref_c_logps = compute_logps(ref_model, c_ids, c_labels)
            ref_r_logps = compute_logps(ref_model, r_ids, r_labels)

        loss, c_rew, r_rew, acc, margin = dpo_loss(
            policy_c_logps,
            policy_r_logps,
            ref_c_logps,
            ref_r_logps,
            beta=0.1,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1:
            initial_loss = loss.item()
        final_loss = loss.item()

        print(f"   DPO Step {step}/5: Loss = {loss.item():.4f} | Acc = {acc.item()*100:.1f}% | Margin = {margin.item():+.4f}")

    # Verify gradients on ref_model are None
    for n, p in ref_model.named_parameters():
        assert p.grad is None, f"Error: Reference model parameter {n} received gradient!"
    print("✅ Frozen Reference model verified (0 gradients received)!")

    # Verify policy model gradients exist
    has_policy_grad = any(p.grad is not None and p.grad.norm().item() > 0 for p in policy_model.parameters())
    assert has_policy_grad, "Error: Policy model received no gradients!"
    print("✅ Policy model gradients verified successfully!")

    # 5. Checkpoint Saving Test
    print("\n[Step 5] Testing DPO checkpoint saving...")
    test_dir = "out/test_dpo_checkpoints"
    os.makedirs(test_dir, exist_ok=True)
    policy_model.save_pretrained(test_dir)
    tokenizer.save_pretrained(test_dir)
    assert os.path.exists(os.path.join(test_dir, "model.safetensors")), "model.safetensors missing!"
    print(f"✅ DPO checkpoint saved successfully in: {test_dir}")
    shutil.rmtree(test_dir, ignore_errors=True)

    print("\n" + "=" * 65)
    print("🎉 ALL DPO VERIFICATION CHECKS PASSED WITH ZERO ERRORS!")
    print("=" * 65)


if __name__ == "__main__":
    run_dpo_dry_run()
