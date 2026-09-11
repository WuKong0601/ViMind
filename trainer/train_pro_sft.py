"""
ViMind 4.0 Pro - Foundation Model Alignment & SFT Trainer.
Trains/Fine-tunes a compact, ultra-smart foundation model (Qwen2.5-0.5B)
on the ViMind Golden Dense Reasoning Core dataset.
"""

import os
import sys
import math
import time
import json
import argparse
from contextlib import nullcontext

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup


class DenseCoreDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        if not os.path.exists(data_path):
            print(f"[Warning] {data_path} not found.")
            return

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.samples.append(json.loads(line))
                    except Exception:
                        pass

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        convs = item.get("conversations", [])
        
        # Apply chat template
        try:
            formatted = self.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)
        except Exception:
            formatted = ""
            for c in convs:
                role = c.get("role", "user")
                content = c.get("content", "")
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        enc = self.tokenizer(
            formatted,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, labels, attention_mask


def train_pro(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("🚀 VIMIND 4.0 PRO: FOUNDATION TRANSFER & DENSE SFT")
    print("=" * 70)
    print(f"Base Model:    {args.model_name_or_path}")
    print(f"Dataset:       {args.data_path}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Device:        {device}")
    print(f"Batch Size:    {args.batch_size} (Grad Accum: {args.accumulation_steps})")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs:        {args.epochs}")
    print("=" * 70)

    # 1. Load Tokenizer & Model
    try:
        print(f"[1/4] Loading Tokenizer & Foundation Model ({args.model_name_or_path})...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Master weights kept in float32 for AdamW numerical stability and AMP autocast
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=None
        ).to(device)
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize foundation model {args.model_name_or_path}: {e}")
        return

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Model Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # 2. Dataset
    print(f"[2/4] Preparing Dense Reasoning Dataset from {args.data_path}...")
    dataset = DenseCoreDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    print(f"      Total samples: {len(dataset):,}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 3. Optimizer & Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = ("cuda" in str(device) and args.fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # 4. Training Loop
    print(f"[3/4] Commencing Alignment & SFT Loop ({total_steps} steps)...")
    model.train()
    global_step = 0
    running_loss = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        for step, (input_ids, labels, attention_mask) in enumerate(loader, start=1):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Warning: NaN/Inf loss detected at step {step}, skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.accumulation_steps

            if step % args.accumulation_steps == 0 or step == len(loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_interval == 0:
                    avg_loss = running_loss / (args.log_interval * args.accumulation_steps)
                    cur_lr = scheduler.get_last_lr()[0]
                    print(f"[{global_step}/{total_steps}] Epoch {epoch}/{args.epochs} Step {step}/{len(loader)} | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e}")
                    running_loss = 0.0

            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    # Save ViMind Pro in FP16 to minimize disk space & fast inference
    print(f"\n[4/4] Saving ViMind Pro model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.half().save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Sanitize tokenizer_config.json if extra_special_tokens is saved as a list
    tok_cfg_path = os.path.join(args.output_dir, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        try:
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tcfg = json.load(f)
            if isinstance(tcfg.get("extra_special_tokens"), list):
                tcfg["extra_special_tokens"] = {}
                with open(tok_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(tcfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    print(f"✅ ViMind Pro export complete! Total training time: {(time.time() - start_time)/60:.1f} minutes.")
    print("=" * 70)


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind Pro Foundation Model Alignment Trainer")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face foundation model")
    parser.add_argument("--data_path", type=str, default="dataset/vimind_4.5_master.jsonl", help="Path to master reasoning dataset")
    parser.add_argument("--output_dir", type=str, default="out/vimind_4.0_pro_final", help="Directory to save model")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable half precision")
    parser.add_argument("--log_interval", type=int, default=25, help="Log step interval")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train_pro(args)
