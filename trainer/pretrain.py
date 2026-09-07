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
from dataset.lm_dataset import PretrainDataset


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
        # 1D params (norms, biases) get 0 weight decay; 2D weight matrices get weight decay
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


def save_checkpoint(model, tokenizer, config, optimizer, scaler, epoch: int, step: int, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = os.path.join(save_dir, f"{prefix}_step_{step}" if "epoch" not in prefix else prefix)
    os.makedirs(step_save_dir, exist_ok=True)

    # Save Hugging Face compatible format
    model.save_pretrained(step_save_dir)
    tokenizer.save_pretrained(step_save_dir)

    # Save training state for resumption
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
    # Apply architecture presets
    if args.model_size == "64m":
        args.hidden_size = 640
        args.num_hidden_layers = 12
        args.num_attention_heads = 10
        args.num_key_value_heads = 5
        args.intermediate_size = 1728
        args.max_seq_len = 1024
        args.save_weight = "vimind_64m"
    elif args.model_size == "26m":
        args.hidden_size = 512
        args.num_hidden_layers = 8
        args.num_attention_heads = 8
        args.num_key_value_heads = 4
        args.intermediate_size = 1088
        args.max_seq_len = 512
        args.save_weight = "vimind_26m"
    elif args.model_size == "104m":
        args.hidden_size = 768
        args.num_hidden_layers = 16
        args.num_attention_heads = 12
        args.num_key_value_heads = 4
        args.intermediate_size = 2048
        args.max_seq_len = 1024
        args.save_weight = "vimind_104m"

    # Device setup
    if "cuda" in args.device and torch.cuda.is_available():
        try:
            test_x = torch.zeros(1, device=args.device) + 1
            _ = test_x.cpu()
            device = torch.device(args.device)
        except Exception as e:
            print(f"⚠️ CUDA kernel execution failed ({e}). Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
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
    print(f"🚀 Starting ViMind Pre-training ({args.model_size.upper()})")
    print(f"   Device: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Precision: {args.dtype} (Scaler: {use_scaler})")
    print(f"   Batch Size: {args.batch_size} (Grad Accum: {args.accumulation_steps}) -> Effective: {args.batch_size * args.accumulation_steps}")
    print(f"   Max Seq Len: {args.max_seq_len} | Epochs: {args.epochs}")
    print(f"   Gradient Checkpointing: {args.gradient_checkpointing}")
    print("=" * 70)

    # 1. Load Tokenizer
    print(f"[1/5] Loading tokenizer from: {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    vocab_size = len(tokenizer)
    print(f"      Tokenizer vocab size: {vocab_size:,}")

    # 2. Build Model
    print(f"[2/5] Initializing ViMind Model ({args.model_size})...")
    config = ViMindConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_seq_len,
        dropout=args.dropout,
    )

    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"      Resuming model weights from checkpoint: {args.resume_from_checkpoint}...")
        model = ViMindForCausalLM.from_pretrained(args.resume_from_checkpoint)
    else:
        model = ViMindForCausalLM(config)

    if args.gradient_checkpointing:
        print("      ⚡ Gradient Checkpointing enabled.")
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Total Parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"      Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    model = model.to(device)
    model.train()

    # 3. Load Dataset & DataLoader
    print(f"[3/5] Loading pretraining dataset from: {args.data_path}...")
    train_dataset = PretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
    )
    num_batches_per_epoch = len(train_loader)
    total_steps = num_batches_per_epoch * args.epochs
    print(f"      Total samples: {len(train_dataset):,}")
    print(f"      Batches per epoch: {num_batches_per_epoch:,} | Total Steps across {args.epochs} epochs: {total_steps:,}")

    # 4. Optimizer & Schedule
    print(f"[4/5] Setting up AdamW optimizer and Cosine LR schedule...")
    optimizer = configure_optimizers(
        model,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
    )

    # Resume trainer state if provided
    start_epoch = 1
    global_step = 0
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        state_file = os.path.join(args.resume_from_checkpoint, "trainer_state.pt")
        if os.path.exists(state_file):
            print(f"      Restoring optimizer & scaler state from {state_file}...")
            state = torch.load(state_file, map_location=device)
            start_epoch = state.get("epoch", 1)
            global_step = state.get("step", 0)
            if "optimizer_state_dict" in state and state["optimizer_state_dict"]:
                optimizer.load_state_dict(state["optimizer_state_dict"])
            if use_scaler and "scaler_state_dict" in state and state["scaler_state_dict"]:
                scaler.load_state_dict(state["scaler_state_dict"])
            print(f"      Resuming from Epoch {start_epoch}, Step {global_step}...")

    # 5. Training Loop
    print(f"[5/5] Commencing training loop for {args.epochs} epoch(s)...")
    running_loss = 0.0
    accumulated_loss = 0.0
    start_time = time.time()
    interval_start_time = time.time()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        for step, (input_ids, labels) in enumerate(train_loader, start=1):
            global_step += 1

            # Update Learning Rate across entire multi-epoch budget
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

            # Forward pass with mixed precision
            with autocast_ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / args.accumulation_steps

            # Backward pass
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item() * args.accumulation_steps
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
                ppl = math.exp(min(avg_loss, 20.0))  # guard overflow

                steps_done = global_step
                steps_remaining = total_steps - steps_done
                speed = args.log_interval / max(1e-4, elapsed)
                eta_seconds = steps_remaining / max(1e-4, speed)

                percent = (steps_done / total_steps) * 100
                print(
                    f"[{percent:5.1f}%] Epoch [{epoch}/{args.epochs}] Step [{step}/{num_batches_per_epoch}] "
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

        # Save End-of-Epoch Checkpoint
        print(f"\n✅ Completed Epoch {epoch}/{args.epochs}. Saving epoch checkpoint...")
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            config=config,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            save_dir=args.save_dir,
            prefix=f"{args.save_weight}_epoch_{epoch}",
        )
        model.train()

    # Save Final Model
    final_save_dir = os.path.join(args.save_dir, f"{args.save_weight}_final")
    print(f"\n🎉 Multi-epoch Training complete! Total training time: {format_time(time.time() - start_time)}")
    print(f"💾 Saving final model to: {final_save_dir}...")
    os.makedirs(final_save_dir, exist_ok=True)
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print("✅ Model weights and tokenizer saved successfully!")


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind Multi-Epoch Pretraining Script")

    # Data and model directories
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_vi.jsonl", help="Path to pretrain jsonl dataset")
    parser.add_argument("--tokenizer_dir", type=str, default="model", help="Directory of trained tokenizer")
    parser.add_argument("--save_dir", type=str, default="out", help="Directory to save checkpoints")
    parser.add_argument("--save_weight", type=str, default="vimind_64m", help="Prefix for checkpoint filenames")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume")

    # Architecture Hyperparameters
    parser.add_argument("--model_size", type=str, default="64m", choices=["26m", "64m", "104m", "custom"], help="Architecture preset")
    parser.add_argument("--hidden_size", type=int, default=640, help="Hidden dimension size")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=10, help="Number of query attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=5, help="Number of key/value heads for GQA")
    parser.add_argument("--intermediate_size", type=int, default=1728, help="SwiGLU intermediate dimension")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size per forward step")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=5e-5, help="Minimum learning rate at end of cosine decay")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--warmup_iters", type=int, default=300, help="Number of linear warmup steps")

    # System & Optimization
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use (cuda:0 or cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16", help="Precision: bfloat16, float16, float32")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--log_interval", type=int, default=50, help="Interval steps for printing terminal logs")
    parser.add_argument("--save_interval", type=int, default=1000, help="Interval steps for saving model checkpoints")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
