import os
import sys
import math
import time
import argparse
from datetime import timedelta
from contextlib import nullcontext

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.model import ViMindConfig, ViMindForCausalLM
from dataset.lm_dataset import SFTDataset


def get_lr(current_step: int, total_steps: int, lr: float, min_lr: float, warmup_iters: int) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if current_step < warmup_iters:
        return lr * (current_step + 1) / max(1, warmup_iters)
    if current_step > total_steps:
        return min_lr
    decay_ratio = (current_step - warmup_iters) / max(1, total_steps - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float, betas=(0.9, 0.95)):
    """Decouple parameters: apply weight decay to 2D matrices, disable for 1D biases and norms."""
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name or "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8)


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def distillation_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.5) -> torch.Tensor:
    """Computes Kullback-Leibler (KL) divergence loss scaled by T^2."""
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )
    return (temperature ** 2) * kl


def save_checkpoint(model, tokenizer, config, optimizer, scaler, epoch: int, step: int, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = os.path.join(save_dir, f"{prefix}_step_{step}" if "epoch" not in prefix else prefix)
    os.makedirs(step_save_dir, exist_ok=True)

    model.save_pretrained(step_save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(step_save_dir)

    checkpoint_state = {
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config.to_dict(),
    }
    torch.save(checkpoint_state, os.path.join(step_save_dir, "trainer_state.pt"))
    print(f"\n[INFO] Checkpoint saved successfully to: {step_save_dir}")


def train(args):
    # Device setup
    if "cuda" in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    device_type = "cuda" if "cuda" in str(device) else "cpu"

    # Precision setup
    if args.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif args.dtype == "float16" and "cuda" in args.device:
        amp_dtype = torch.float16
        use_scaler = True
    else:
        amp_dtype = torch.float32
        use_scaler = False

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
        if device_type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    print("=" * 70)
    print("🎓 VIMIND KNOWLEDGE DISTILLATION ENGINE (WHITE-BOX LOGITS KD)")
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Alpha (CE weight): {args.alpha} | Distill weight (1-alpha): {1.0 - args.alpha:.2f}")
    print(f"   Distillation Temperature (T): {args.temperature}")
    print(f"   Batch Size: {args.batch_size} (Grad Accum: {args.accumulation_steps}) -> Effective: {args.batch_size * args.accumulation_steps}")
    print("=" * 70)

    # 1. Load Tokenizer
    print(f"[1/5] Loading tokenizer from: {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    vocab_size = len(tokenizer)

    # 2. Load Student Model
    print(f"[2/5] Initializing Student Model from: {args.student_model_path}...")
    if os.path.exists(args.student_model_path):
        student_model = ViMindForCausalLM.from_pretrained(args.student_model_path)
    else:
        print(f"      Checkpoint not found at {args.student_model_path}. Initializing ViMind 64M preset...")
        cfg = ViMindConfig.get_config_64m(vocab_size=vocab_size)
        student_model = ViMindForCausalLM(cfg)

    student_model = student_model.to(device)
    student_model.train()
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"      Student Parameters: {student_params:,} ({student_params / 1e6:.2f}M)")

    # 3. Load Teacher Model (Frozen)
    print(f"[3/5] Initializing Teacher Model from: {args.teacher_model_path}...")
    if os.path.exists(args.teacher_model_path):
        try:
            teacher_model = ViMindForCausalLM.from_pretrained(args.teacher_model_path)
        except Exception:
            teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path)
    else:
        print(f"      Teacher path not found locally. Initializing teacher model from ViMind 104M preset...")
        t_cfg = ViMindConfig.get_config_104m(vocab_size=vocab_size)
        teacher_model = ViMindForCausalLM(t_cfg)

    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print(f"      Teacher Parameters (Frozen): {teacher_params:,} ({teacher_params / 1e6:.2f}M)")

    # 4. Load SFT Dataset with Conversational Loss Masking
    print(f"[4/5] Loading SFT dataset for distillation: {args.data_path}...")
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
    )
    num_batches_per_epoch = len(loader)
    total_steps = num_batches_per_epoch * args.epochs
    print(f"      Total samples: {len(dataset):,} | Batches per epoch: {num_batches_per_epoch:,} | Total Steps: {total_steps:,}")

    # 5. Optimizer & Training Loop
    print(f"[5/5] Commencing Knowledge Distillation loop for {args.epochs} epoch(s)...")
    optimizer = configure_optimizers(student_model, weight_decay=args.weight_decay, learning_rate=args.learning_rate)

    global_step = 0
    running_total_loss = 0.0
    running_ce_loss = 0.0
    running_kd_loss = 0.0
    start_time = time.time()
    interval_start_time = time.time()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Distillation Epoch {epoch}/{args.epochs} ---")
        for step, (input_ids, labels) in enumerate(loader, start=1):
            global_step += 1

            lr = get_lr(global_step, total_steps, args.learning_rate, args.min_lr, args.warmup_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Target mask where label != -100 (only backprop on assistant responses)
            loss_mask = (labels[..., 1:] != -100)

            # Forward pass: Student Model
            with autocast_ctx:
                student_outputs = student_model(input_ids)
                student_logits = student_outputs.logits[..., :-1, :].contiguous()

                # Forward pass: Teacher Model (strictly no_grad)
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids)
                    teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous()
                    # Align vocabulary if teacher vocab is larger
                    vocab_student = student_logits.size(-1)
                    if teacher_logits.size(-1) > vocab_student:
                        teacher_logits = teacher_logits[..., :vocab_student]

                # 1) Ground-Truth Cross-Entropy Loss
                shift_labels = labels[..., 1:].contiguous()
                ce_loss = F.cross_entropy(
                    student_logits.view(-1, vocab_student),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

                # 2) Soft Knowledge Distillation Loss (KL Divergence on assistant response tokens)
                mask_flat = loss_mask.view(-1)
                if mask_flat.sum() > 0:
                    student_target_logits = student_logits.view(-1, vocab_student)[mask_flat]
                    teacher_target_logits = teacher_logits.view(-1, vocab_student)[mask_flat]
                    kd_loss = distillation_kl_loss(
                        student_target_logits,
                        teacher_target_logits,
                        temperature=args.temperature,
                    )
                else:
                    kd_loss = torch.tensor(0.0, device=device)

                # 3) Dual Loss: alpha * CE + (1 - alpha) * KD
                total_loss = (args.alpha * ce_loss + (1.0 - args.alpha) * kd_loss) / args.accumulation_steps

            # Backward pass
            if use_scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            running_total_loss += total_loss.item() * args.accumulation_steps
            running_ce_loss += ce_loss.item()
            running_kd_loss += kd_loss.item()

            if step % args.accumulation_steps == 0 or step == num_batches_per_epoch:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            # Logging to Terminal
            if global_step % args.log_interval == 0:
                elapsed = time.time() - interval_start_time
                avg_total = running_total_loss / args.log_interval
                avg_ce = running_ce_loss / args.log_interval
                avg_kd = running_kd_loss / args.log_interval

                steps_done = global_step
                steps_rem = total_steps - steps_done
                speed = args.log_interval / max(1e-4, elapsed)
                eta = steps_rem / max(1e-4, speed)

                percent = (steps_done / total_steps) * 100
                print(
                    f"[{percent:5.1f}%] Distill [{epoch}/{args.epochs}] Step [{step}/{num_batches_per_epoch}] "
                    f"| Loss: {avg_total:.4f} (CE: {avg_ce:.4f}, KD: {avg_kd:.4f}) | LR: {lr:.2e} "
                    f"| Speed: {speed:.1f} it/s | ETA: {format_time(eta)}"
                )
                running_total_loss = 0.0
                running_ce_loss = 0.0
                running_kd_loss = 0.0
                interval_start_time = time.time()

            # Save Checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoint(
                    model=student_model,
                    tokenizer=tokenizer,
                    config=student_model.config,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    save_dir=args.save_dir,
                    prefix=args.save_weight,
                )
                student_model.train()

    # Save Final Distilled Model
    final_save_dir = os.path.join(args.save_dir, f"{args.save_weight}_final")
    print(f"\n🎉 Distillation complete! Total time: {format_time(time.time() - start_time)}")
    print(f"💾 Saving final distilled student model to: {final_save_dir}...")
    os.makedirs(final_save_dir, exist_ok=True)
    student_model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print("✅ Distilled model weights and tokenizer saved successfully!")


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind White-Box Knowledge Distillation")

    # Data and Model Paths
    parser.add_argument("--data_path", type=str, default="dataset/sft_vi.jsonl", help="Path to SFT dialogue dataset")
    parser.add_argument("--tokenizer_dir", type=str, default="model", help="Directory of trained tokenizer")
    parser.add_argument("--student_model_path", type=str, default="out/vimind_64m_final", help="Path to student model checkpoint")
    parser.add_argument("--teacher_model_path", type=str, default="out/teacher_model", help="Path to teacher model checkpoint")
    parser.add_argument("--save_dir", type=str, default="out/distill", help="Directory to save checkpoints")
    parser.add_argument("--save_weight", type=str, default="vimind_distill", help="Prefix for checkpoint filenames")

    # Distillation Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for CE loss (loss = alpha*CE + (1-alpha)*KD)")
    parser.add_argument("--temperature", type=float, default=1.5, help="Softmax softening temperature T")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")

    # Optimization Hyperparameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per forward step")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Linear warmup steps")

    # Hardware & System
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cuda:0 or cpu)")
    parser.add_argument("--dtype", type=str, default="float16" if torch.cuda.is_available() else "float32", help="Precision (bfloat16, float16, float32)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--log_interval", type=int, default=25, help="Logging steps interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Saving steps interval")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
