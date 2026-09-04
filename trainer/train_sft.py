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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model import ViMindConfig, ViMindForCausalLM
from dataset.lm_dataset import SFTDataset
from trainer.pretrain import configure_optimizers, get_lr, save_checkpoint, format_time


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind Supervised Fine-Tuning (SFT)")

    # Data and model directories
    parser.add_argument("--data_path", type=str, default="dataset/sft_vi.jsonl", help="Path to SFT conversational dataset")
    parser.add_argument("--tokenizer_dir", type=str, default="model", help="Directory of trained tokenizer")
    parser.add_argument("--from_pretrained", type=str, default="none", help="Path to base pretrained model (or 'none')")
    parser.add_argument("--save_dir", type=str, default="out/sft", help="Directory to save SFT checkpoints")
    parser.add_argument("--save_weight", type=str, default="vimind_sft", help="Prefix for checkpoint filenames")

    # Architecture Hyperparameters (used if training from scratch)
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden dimension size")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Number of query attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=4, help="Number of key/value heads for GQA")
    parser.add_argument("--intermediate_size", type=int, default=1088, help="SwiGLU intermediate dimension (~26M total params)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout probability for SFT")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of SFT training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size per forward step")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate for SFT")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate at end of cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Number of linear warmup steps")

    # System & Optimization
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16", help="Precision: bfloat16, float16, float32")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--log_interval", type=int, default=25, help="Interval steps for printing terminal logs")
    parser.add_argument("--save_interval", type=int, default=500, help="Interval steps for saving model checkpoints")

    return parser


def train_sft(args):
    # Device setup
    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"

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
    print(f"🎯 Starting ViMind Supervised Fine-Tuning (SFT)")
    print(f"   Device: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Precision: {args.dtype} (Scaler: {use_scaler})")
    print(f"   Batch Size: {args.batch_size} (Grad Accum: {args.accumulation_steps}) -> Effective: {args.batch_size * args.accumulation_steps}")
    print(f"   Max Seq Len: {args.max_seq_len}")
    print(f"   Base Model: {args.from_pretrained}")
    print("=" * 70)

    # 1. Load Tokenizer
    print(f"[1/5] Loading tokenizer from: {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"      Tokenizer vocab size: {len(tokenizer):,}")

    # 2. Build or Load Model
    print(f"[2/5] Initializing / Loading Model...")
    if args.from_pretrained != "none" and os.path.exists(args.from_pretrained):
        print(f"      Loading weights from pre-trained checkpoint: {args.from_pretrained}")
        model = ViMindForCausalLM.from_pretrained(args.from_pretrained)
        config = model.config
    else:
        print(f"      Training SFT directly from fresh architecture (~26.2M)...")
        config = ViMindConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            num_key_value_heads=args.num_key_value_heads,
            intermediate_size=args.intermediate_size,
            max_position_embeddings=args.max_seq_len,
            dropout=args.dropout,
        )
        model = ViMindForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    model = model.to(device)
    model.train()

    # 3. Load SFT Dataset & DataLoader
    print(f"[3/5] Loading SFT dataset from: {args.data_path}...")
    sft_dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )
    sft_loader = DataLoader(
        sft_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
    )
    num_batches_per_epoch = len(sft_loader)
    total_steps = num_batches_per_epoch * args.epochs
    print(f"      Total conversations: {len(sft_dataset):,}")
    print(f"      Batches per epoch: {num_batches_per_epoch:,} | Total SFT Steps: {total_steps:,}")

    # 4. Optimizer & Schedule
    print(f"[4/5] Setting up AdamW optimizer and Cosine LR schedule...")
    optimizer = configure_optimizers(
        model,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
    )

    # 5. Training Loop
    print(f"[5/5] Commencing SFT training loop for {args.epochs} epoch(s)...")
    global_step = 0
    running_loss = 0.0
    start_time = time.time()
    interval_start_time = time.time()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- SFT Epoch {epoch}/{args.epochs} ---")
        for step, (input_ids, labels) in enumerate(sft_loader, start=1):
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

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            with autocast_ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / args.accumulation_steps

            # Backward pass
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.accumulation_steps

            # Gradient accumulation step
            if step % args.accumulation_steps == 0 or step == num_batches_per_epoch:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            # Logging to Terminal
            if global_step % args.log_interval == 0:
                elapsed = time.time() - interval_start_time
                avg_loss = running_loss / args.log_interval
                ppl = math.exp(min(avg_loss, 20.0))

                steps_done = global_step
                steps_remaining = total_steps - steps_done
                speed = args.log_interval / elapsed
                eta_seconds = steps_remaining / max(1e-4, speed)

                percent = (steps_done / total_steps) * 100
                print(
                    f"[{percent:5.1f}%] SFT [{epoch}/{args.epochs}] Step [{step}/{num_batches_per_epoch}] "
                    f"| Loss: {avg_loss:.4f} (PPL: {ppl:.1f}) | LR: {lr:.2e} "
                    f"| Speed: {speed:.1f} it/s | ETA: {format_time(eta_seconds)}"
                )
                running_loss = 0.0
                interval_start_time = time.time()

            # Periodic Checkpoint Saving
            if global_step % args.save_interval == 0:
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    save_dir=args.save_dir,
                    prefix=args.save_weight,
                )
                model.train()

    # Save Final SFT Model
    final_save_dir = os.path.join(args.save_dir, f"{args.save_weight}_final")
    print(f"\n🎉 SFT Training complete! Total training time: {format_time(time.time() - start_time)}")
    print(f"💾 Saving final SFT model to: {final_save_dir}...")
    os.makedirs(final_save_dir, exist_ok=True)
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print("✅ SFT Model weights and tokenizer saved successfully!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train_sft(args)
