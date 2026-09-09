import os
import sys
import time
import math
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Force UTF-8 encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import ViMindConfig, ViMindForCausalLM
from dataset.lm_dataset import DPODataset


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_lr(current_step: int, total_steps: int, lr: float, min_lr: float, warmup_iters: int) -> float:
    if current_step < warmup_iters:
        return lr * current_step / max(1, warmup_iters)
    if current_step > total_steps:
        return min_lr
    decay_ratio = (current_step - warmup_iters) / max(1, (total_steps - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def compute_logps(model, input_ids, labels):
    """
    Computes sum of log probabilities of the response tokens given the prompt.
    Tokens where labels == -100 are ignored in the summation.
    """
    outputs = model(input_ids)
    logits = outputs.logits  # [B, L, V]

    # Shift so tokens predict the next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_mask = shift_labels != -100

    labels_for_gather = shift_labels.clone()
    labels_for_gather[shift_labels == -100] = 0

    per_token_logps = torch.gather(
        shift_logits.log_softmax(dim=-1),
        dim=2,
        index=labels_for_gather.unsqueeze(2),
    ).squeeze(2)

    return (per_token_logps * loss_mask).sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Direct Preference Optimization (DPO) Loss:
    L_DPO = -E[log sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))]
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    reward_acc = (chosen_rewards > rejected_rewards).float().mean()
    reward_margin = (chosen_rewards - rejected_rewards).mean()

    return loss, chosen_rewards.mean(), rejected_rewards.mean(), reward_acc, reward_margin


def save_checkpoint(model, tokenizer, config, optimizer, scaler, epoch, step, save_dir, prefix="vimind_dpo"):
    ckpt_dir = os.path.join(save_dir, f"{prefix}_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"\n[INFO] Saving DPO checkpoint to: {ckpt_dir} ...")

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    config.save_pretrained(ckpt_dir)

    state = {
        "epoch": epoch,
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": config.to_dict(),
    }
    torch.save(state, os.path.join(ckpt_dir, "trainer_state.pt"))
    print(f"[INFO] Checkpoint saved successfully to: {ckpt_dir}\n")


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind Direct Preference Optimization (DPO) Trainer")
    parser.add_argument("--data_path", type=str, default="dataset/dpo_vi.jsonl", help="Path to DPO jsonl dataset")
    parser.add_argument("--model_path", type=str, default="out/sft/vimind_sft_final", help="Path to SFT model directory")
    parser.add_argument("--save_dir", type=str, default="out/dpo", help="Directory to save DPO checkpoints")
    parser.add_argument("--save_weight", type=str, default="vimind_dpo", help="Prefix name for saved checkpoints")
    parser.add_argument("--epochs", type=int, default=1, help="Number of DPO training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU step")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Peak learning rate for DPO")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--warmup_iters", type=int, default=50, help="Number of warmup iterations")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta parameter")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length for chosen/rejected pairs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--log_interval", type=int, default=10, help="Steps between log prints")
    parser.add_argument("--save_interval", type=int, default=500, help="Steps between checkpoint saves")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision training")
    return parser


def train_dpo(args):
    print("=" * 70)
    print("🇻🇳 ViMind: Direct Preference Optimization (DPO) Training Pipeline")
    print("=" * 70)

    # 1. Device Selection & Defensive CUDA Capability Check
    device_name = args.device
    if "cuda" in device_name and torch.cuda.is_available():
        try:
            test_x = torch.zeros(1, 1, device=device_name)
            _ = torch.nn.Linear(1, 1).to(device_name)(test_x)
            device = torch.device(device_name)
            gpu_name = torch.cuda.get_device_name(device)
            print(f"✅ Active Compute Device: {device} ({gpu_name})")
        except Exception as e:
            print(f"⚠️ GPU incompatible with PyTorch CUDA kernel: {e}")
            print("🔄 Falling back safely to CPU execution.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("ℹ️ Active Compute Device: CPU")

    use_amp = args.fp16 and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"⚙️ Precision: {'FP16 Mixed Precision' if use_amp else 'Full Float32'}")

    # 2. Load Tokenizer
    tokenizer_dir = args.model_path if os.path.exists(os.path.join(args.model_path, "tokenizer_config.json")) else "model"
    print(f"[1/5] Loading tokenizer from: {tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print(f"      Tokenizer vocabulary size: {len(tokenizer):,}")

    # 3. Load DPO Dataset
    print(f"[2/5] Loading DPO preference dataset from: {args.data_path}...")
    dpo_dataset = DPODataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )
    num_samples = len(dpo_dataset)
    print(f"      Total DPO pairs: {num_samples:,}")

    dpo_loader = DataLoader(
        dpo_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # 4. Load Models: Policy Model & Frozen Reference Model
    print(f"[3/5] Loading Policy Model and Reference Model from: {args.model_path}...")
    actual_model_path = args.model_path
    if os.path.isdir(actual_model_path):
        if not os.path.exists(os.path.join(actual_model_path, "config.json")):
            subfolders = [os.path.join(actual_model_path, d) for d in os.listdir(actual_model_path) if os.path.isdir(os.path.join(actual_model_path, d))]
            for sf in subfolders:
                if os.path.exists(os.path.join(sf, "config.json")):
                    actual_model_path = sf
                    break

    policy_model = ViMindForCausalLM.from_pretrained(actual_model_path).to(device)
    ref_model = ViMindForCausalLM.from_pretrained(actual_model_path).to(device)

    # Freeze reference model
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("      Reference model loaded and frozen (0 trainable parameters).")

    # Policy model config & parameters
    config = policy_model.config
    policy_model.train()
    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"      Policy model trainable parameters: {trainable_params:,}")

    # 5. Optimizer & LR Scheduler
    print(f"[4/5] Configuring AdamW Optimizer (LR: {args.learning_rate:.1e}, Beta: {args.beta})...")
    decay_params = [p for n, p in policy_model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in policy_model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.05},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    num_batches_per_epoch = len(dpo_loader)
    total_steps = (num_batches_per_epoch // args.accumulation_steps) * args.epochs
    print(f"      Batches per epoch: {num_batches_per_epoch:,} | Total update steps: {total_steps:,}")

    # Pre-flight sanity check on 1 batch
    policy_model.eval()
    with torch.no_grad():
        c_in, c_lbl, r_in, r_lbl = next(iter(dpo_loader))
        c_in, c_lbl = c_in[:2].to(device), c_lbl[:2].to(device)
        r_in, r_lbl = r_in[:2].to(device), r_lbl[:2].to(device)
        with autocast_ctx:
            p_c_logps = compute_logps(policy_model, c_in, c_lbl)
            p_r_logps = compute_logps(policy_model, r_in, r_lbl)
            r_c_logps = compute_logps(ref_model, c_in, c_lbl)
            r_r_logps = compute_logps(ref_model, r_in, r_lbl)
            test_loss, _, _, test_acc, test_margin = dpo_loss(p_c_logps, p_r_logps, r_c_logps, r_r_logps, beta=args.beta)
        print(f"      Pre-flight validation: Loss: {test_loss.item():.4f} | Acc: {test_acc.item()*100:.1f}% | Margin: {test_margin.item():.4f}")
        assert not (torch.isnan(test_loss) or torch.isinf(test_loss)), "Pre-flight validation produced NaN/Inf loss!"
    policy_model.train()

    # 6. Training Loop
    print(f"\n[5/5] Commencing DPO training for {args.epochs} epoch(s)...")
    global_step = 0
    running_loss = 0.0
    running_acc = 0.0
    running_margin = 0.0
    start_time = time.time()
    interval_start_time = time.time()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- DPO Epoch {epoch}/{args.epochs} ---")
        for step, (chosen_ids, chosen_labels, rejected_ids, rejected_labels) in enumerate(dpo_loader, start=1):
            chosen_ids = chosen_ids.to(device, non_blocking=True)
            chosen_labels = chosen_labels.to(device, non_blocking=True)
            rejected_ids = rejected_ids.to(device, non_blocking=True)
            rejected_labels = rejected_labels.to(device, non_blocking=True)

            # Compute Policy Model logps (with grad)
            with autocast_ctx:
                policy_chosen_logps = compute_logps(policy_model, chosen_ids, chosen_labels)
                policy_rejected_logps = compute_logps(policy_model, rejected_ids, rejected_labels)

            # Compute Reference Model logps (frozen, no grad)
            with torch.no_grad(), autocast_ctx:
                ref_chosen_logps = compute_logps(ref_model, chosen_ids, chosen_labels)
                ref_rejected_logps = compute_logps(ref_model, rejected_ids, rejected_labels)

            # DPO Loss
            with autocast_ctx:
                loss, c_rew, r_rew, acc, margin = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=args.beta,
                )
                scaled_loss = loss / args.accumulation_steps

            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                print(f"⚠️ Step {step}: Loss is NaN/Inf, skipping update.", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running_loss += loss.item()
            running_acc += acc.item()
            running_margin += margin.item()

            # Optimizer Step on Accumulation Boundary
            if step % args.accumulation_steps == 0 or step == num_batches_per_epoch:
                global_step += 1

                # Update Learning Rate
                lr = get_lr(
                    current_step=global_step,
                    total_steps=total_steps,
                    lr=args.learning_rate,
                    min_lr=args.min_lr,
                    warmup_iters=args.warmup_iters,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                # Logging to Terminal
                if global_step % args.log_interval == 0:
                    elapsed = time.time() - interval_start_time
                    log_steps = args.log_interval * args.accumulation_steps
                    avg_loss = running_loss / log_steps
                    avg_acc = (running_acc / log_steps) * 100
                    avg_margin = running_margin / log_steps

                    steps_remaining = total_steps - global_step
                    speed = args.log_interval / max(1e-4, elapsed)
                    eta_seconds = steps_remaining / max(1e-4, speed)
                    percent = (global_step / total_steps) * 100

                    print(
                        f"[{percent:5.1f}%] DPO [{epoch}/{args.epochs}] Step [{global_step}/{total_steps}] "
                        f"| Loss: {avg_loss:.4f} | Acc: {avg_acc:5.1f}% | Margin: {avg_margin:+.4f} "
                        f"| LR: {lr:.2e} | Speed: {speed:.1f} it/s | ETA: {format_time(eta_seconds)}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_acc = 0.0
                    running_margin = 0.0
                    interval_start_time = time.time()

                # Periodic Checkpoint Saving
                if global_step % args.save_interval == 0:
                    save_checkpoint(
                        model=policy_model,
                        tokenizer=tokenizer,
                        config=config,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        step=global_step,
                        save_dir=args.save_dir,
                        prefix=args.save_weight,
                    )
                    policy_model.train()

    # Save Final DPO Model (ViMind 1.0 Final)
    final_save_dir = os.path.join(args.save_dir, f"{args.save_weight}_final")
    print(f"\n🎉 DPO Training complete! Total training time: {format_time(time.time() - start_time)}")
    print(f"💾 Saving final ViMind 1.0 (DPO Final) model to: {final_save_dir}...")
    os.makedirs(final_save_dir, exist_ok=True)
    policy_model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    config.save_pretrained(final_save_dir)
    policy_model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    config.save_pretrained(args.save_dir)
    pth_path = os.path.join(args.save_dir, f"{args.save_weight}.pth")
    torch.save(policy_model.state_dict(), pth_path)
    print(f"✅ ViMind Model weights saved successfully to {final_save_dir} and {pth_path}!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train_dpo(args)
