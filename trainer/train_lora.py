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
from transformers import AutoTokenizer

from model.model import ViMindConfig, ViMindForCausalLM
from model.model_lora import apply_lora, mark_only_lora_as_trainable, save_lora, merge_lora
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


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


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
    print("⚡ VIMIND NATIVE LORA FINE-TUNING ENGINE")
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   LoRA Rank (r): {args.lora_rank} | Alpha (α): {args.lora_alpha} | Dropout: {args.lora_dropout}")
    print(f"   Batch Size: {args.batch_size} (Grad Accum: {args.accumulation_steps}) -> Effective: {args.batch_size * args.accumulation_steps}")
    print("=" * 70)

    # 1. Load Tokenizer
    print(f"[1/5] Loading tokenizer from: {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    vocab_size = len(tokenizer)

    # 2. Load Base Model
    print(f"[2/5] Initializing Base Model from: {args.model_path}...")
    if os.path.exists(args.model_path):
        model = ViMindForCausalLM.from_pretrained(args.model_path)
    else:
        print(f"      Checkpoint not found at {args.model_path}. Initializing ViMind 64M preset...")
        cfg = ViMindConfig.get_config_64m(vocab_size=vocab_size)
        model = ViMindForCausalLM(cfg)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # 3. Apply LoRA Modules and Freeze Base Parameters
    print(f"[3/5] Injecting LoRA adapter into target modules...")
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    num_injected = apply_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    lora_params = mark_only_lora_as_trainable(model)
    lora_param_count = sum(p.numel() for p in lora_params)

    print(f"      Injected LoRA layers: {num_injected}")
    print(f"      Base Model Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"      Trainable LoRA Parameters: {lora_param_count:,} ({lora_param_count / 1e6:.3f}M)")
    print(f"      Trainable Ratio: {lora_param_count / total_params * 100:.3f}%")

    # 4. Load SFT Dataset
    print(f"[4/5] Loading dataset: {args.data_path}...")
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
    print(f"[5/5] Commencing LoRA training loop for {args.epochs} epoch(s)...")
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0
    running_loss = 0.0
    start_time = time.time()
    interval_start_time = time.time()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"\n--- LoRA Epoch {epoch}/{args.epochs} ---")
        for step, (input_ids, labels) in enumerate(loader, start=1):
            global_step += 1

            lr = get_lr(global_step, total_steps, args.learning_rate, args.min_lr, args.warmup_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_ctx:
                outputs = model(input_ids)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / args.accumulation_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.accumulation_steps

            if step % args.accumulation_steps == 0 or step == num_batches_per_epoch:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            # Logging to Terminal
            if global_step % args.log_interval == 0:
                elapsed = time.time() - interval_start_time
                avg_loss = running_loss / args.log_interval

                steps_done = global_step
                steps_rem = total_steps - steps_done
                speed = args.log_interval / max(1e-4, elapsed)
                eta = steps_rem / max(1e-4, speed)

                percent = (steps_done / total_steps) * 100
                print(
                    f"[{percent:5.1f}%] LoRA [{epoch}/{args.epochs}] Step [{step}/{num_batches_per_epoch}] "
                    f"| Loss: {avg_loss:.4f} | LR: {lr:.2e} "
                    f"| Speed: {speed:.1f} it/s | ETA: {format_time(eta)}"
                )
                running_loss = 0.0
                interval_start_time = time.time()

            # Save LoRA Adapter Checkpoint
            if global_step % args.save_interval == 0:
                adapter_path = os.path.join(args.save_dir, f"{args.lora_name}_step_{global_step}.pth")
                save_lora(model, adapter_path)
                print(f"[INFO] Saved intermediate LoRA adapter checkpoint to: {adapter_path}")

    # Save Final LoRA Adapter
    final_adapter_path = os.path.join(args.save_dir, f"{args.lora_name}_final.pth")
    save_lora(model, final_adapter_path)
    print(f"\n🎉 LoRA Fine-Tuning complete! Total time: {format_time(time.time() - start_time)}")
    print(f"💾 Final LoRA adapter saved to: {final_adapter_path}")

    # Merge LoRA and save standalone model if requested
    if args.merge_and_save:
        merged_save_dir = os.path.join(args.save_dir, f"{args.lora_name}_merged")
        print(f"🔄 Merging LoRA weights into base model and saving to: {merged_save_dir}...")
        merge_lora(model)
        os.makedirs(merged_save_dir, exist_ok=True)
        model.save_pretrained(merged_save_dir)
        tokenizer.save_pretrained(merged_save_dir)
        print("✅ Standalone merged model weights and tokenizer saved successfully!")


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind Native LoRA Fine-Tuning")

    # Data and Model Paths
    parser.add_argument("--data_path", type=str, default="dataset/sft_vi.jsonl", help="Path to fine-tuning dataset")
    parser.add_argument("--tokenizer_dir", type=str, default="model", help="Directory of trained tokenizer")
    parser.add_argument("--model_path", type=str, default="out/vimind_64m_final", help="Path to base model checkpoint")
    parser.add_argument("--save_dir", type=str, default="out/lora", help="Directory to save LoRA adapters")
    parser.add_argument("--lora_name", type=str, default="vimind_lora", help="Prefix for LoRA adapter filenames")

    # LoRA Hyperparameters
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of LoRA decomposition matrices")
    parser.add_argument("--lora_alpha", type=float, default=32.0, help="Scaling factor alpha for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout rate inside LoRA layers")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated target Linear layer names")
    parser.add_argument("--merge_and_save", action="store_true", help="Merge LoRA into base model and export at end of training")

    # Optimization Hyperparameters
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per forward step")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup_iters", type=int, default=50, help="Linear warmup steps")

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
