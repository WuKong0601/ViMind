import os
import sys
import math
import time
import re
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.model import ViMindConfig, ViMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def repetition_penalty_score(text: str, n: int = 3, cap: float = 0.5) -> float:
    """Computes repetition penalty based on n-gram uniqueness."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    unique_ratio = len(set(ngrams)) / len(ngrams)
    penalty = (1.0 - unique_ratio) * 2.0 * cap
    return min(cap, penalty)


def calculate_rewards(
    prompts: list,
    completions: list,
    ground_truths: list,
    num_generations: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Computes rule-based rewards: CoT reasoning format, answer length,
    repetition penalty, and ground-truth answer alignment.
    """
    rewards = torch.zeros(len(completions), device=device)

    for i in range(len(prompts)):
        gt = ground_truths[i] if i < len(ground_truths) else ""
        for j in range(num_generations):
            idx = i * num_generations + j
            resp = completions[idx].strip()
            answer = resp

            # 1. Length reward: penalize empty/too short or runaway generations
            if 20 <= len(resp) <= 800:
                rewards[idx] += 0.5
            elif len(resp) < 15:
                rewards[idx] -= 0.5
            elif len(resp) > 1200:
                rewards[idx] -= 0.5

            # 2. Chain-of-Thought (<think> ... </think>) format reward
            if "</think>" in resp:
                parts = resp.split("</think>", 1)
                thinking = parts[0].replace("<think>", "").strip()
                answer = parts[1].strip()

                # Reward coherent thinking length
                if 20 <= len(thinking) <= 400:
                    rewards[idx] += 1.0
                elif len(thinking) < 10:
                    rewards[idx] -= 0.3

                # Reward exactly one closing tag
                if resp.count("</think>") == 1:
                    rewards[idx] += 0.25
                else:
                    rewards[idx] -= 0.5
            elif "<think>" in resp:
                # Started thinking but didn't close
                rewards[idx] -= 0.5

            # 3. Repetition penalty
            rewards[idx] -= repetition_penalty_score(answer)

            # 4. Ground-truth matching (if available)
            if gt and len(gt.strip()) > 0:
                if gt.strip().lower() in answer.lower():
                    rewards[idx] += 1.5

    return rewards


def save_checkpoint(model, tokenizer, config, optimizer, step: int, save_dir: str, prefix: str):
    os.makedirs(save_dir, exist_ok=True)
    step_save_dir = os.path.join(save_dir, f"{prefix}_step_{step}")
    os.makedirs(step_save_dir, exist_ok=True)

    raw_model = getattr(model, "_orig_mod", model)
    raw_model.save_pretrained(step_save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(step_save_dir)

    checkpoint_state = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
    }
    torch.save(checkpoint_state, os.path.join(step_save_dir, "trainer_state.pt"))
    print(f"\n[INFO] GRPO Checkpoint saved successfully to: {step_save_dir}")


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
    print("🧠 VIMIND GRPO REINFORCEMENT LEARNING ENGINE")
    print(f"   Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"   Num Generations (G): {args.num_generations} per prompt")
    print(f"   KL Penalty Beta (β): {args.beta} | Loss Type: {args.loss_type}")
    print(f"   Batch Size: {args.batch_size} (Grad Accum: {args.accumulation_steps})")
    print("=" * 70)

    # 1. Load Tokenizer
    print(f"[1/5] Loading tokenizer from: {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    vocab_size = len(tokenizer)

    # 2. Load Policy Model
    print(f"[2/5] Initializing Policy Model from: {args.model_path}...")
    if os.path.exists(args.model_path):
        model = ViMindForCausalLM.from_pretrained(args.model_path)
    else:
        print(f"      Model checkpoint not found at {args.model_path}. Initializing ViMind 64M preset...")
        cfg = ViMindConfig.get_config_64m(vocab_size=vocab_size)
        model = ViMindForCausalLM(cfg)

    model = model.to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Policy Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 3. Load Reference Model (Frozen for KL Divergence)
    print(f"[3/5] Initializing Frozen Reference Model from: {args.model_path}...")
    if os.path.exists(args.model_path):
        ref_model = ViMindForCausalLM.from_pretrained(args.model_path)
    else:
        cfg = ViMindConfig.get_config_64m(vocab_size=vocab_size)
        ref_model = ViMindForCausalLM(cfg)

    ref_model = ref_model.to(device)
    ref_model.eval()
    ref_model.requires_grad_(False)

    # 4. Rollout Engine and Dataset
    print(f"[4/5] Setting up Rollout Engine and RLAIF Dataset...")
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=str(device),
        autocast_ctx=autocast_ctx,
    )

    dataset = RLAIFDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        thinking_ratio=args.thinking_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    num_batches_per_epoch = len(loader)
    total_steps = num_batches_per_epoch * args.epochs
    print(f"      Total prompts: {len(dataset):,} | Batches per epoch: {num_batches_per_epoch:,} | Total Steps: {total_steps:,}")

    # 5. Optimizer & GRPO Training Loop
    print(f"[5/5] Commencing GRPO training loop for {args.epochs} epoch(s)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    global_step = 0
    running_policy_loss = 0.0
    running_reward = 0.0
    start_time = time.time()
    interval_start_time = time.time()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- GRPO Epoch {epoch}/{args.epochs} ---")
        for step, batch in enumerate(loader, start=1):
            global_step += 1

            prompts = batch["prompt"]
            ground_truths = batch.get("ground_truth", [""] * len(prompts))

            # Left pad prompts for generation
            prompt_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_seq_len,
            ).to(device)

            # Rollout policy completions
            rollout_result = rollout_engine.rollout(
                prompt_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs["attention_mask"],
                num_generations=args.num_generations,
                max_new_tokens=args.max_gen_len,
                temperature=args.temperature,
            )

            outputs = rollout_result.output_ids
            completion_ids = rollout_result.completion_ids
            completions = rollout_result.completions
            old_per_token_logps = rollout_result.per_token_logps.detach()

            # Compute rewards
            rewards = calculate_rewards(
                prompts=prompts,
                completions=completions,
                ground_truths=ground_truths,
                num_generations=args.num_generations,
                device=device,
            )

            # Group relative advantage normalization
            grouped_rewards = rewards.view(-1, args.num_generations)
            mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
            std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
            advantages = (rewards - mean_r) / (std_r + 1e-4)

            # Mask out padding and tokens after first EOS
            full_mask = (outputs != tokenizer.pad_token_id).long()
            is_eos = (completion_ids == tokenizer.eos_token_id)
            eos_pos = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            eos_indices = torch.where(has_eos, eos_pos, torch.tensor(completion_ids.size(1) - 1, device=device))
            seq_range = torch.arange(completion_ids.size(1), device=device).unsqueeze(0)
            valid_mask = (seq_range <= eos_indices.unsqueeze(1)) & (completion_ids != tokenizer.pad_token_id)

            # Forward pass: current policy model
            with autocast_ctx:
                per_token_logps = compute_per_token_logps(
                    model,
                    outputs,
                    completion_ids.size(1),
                    attention_mask=full_mask,
                )

                # Forward pass: reference model (frozen)
                with torch.no_grad():
                    ref_per_token_logps = compute_per_token_logps(
                        ref_model,
                        outputs,
                        completion_ids.size(1),
                        attention_mask=full_mask,
                    )

                # Approximate KL divergence per token: exp(log p_ref - log p) - (log p_ref - log p) - 1
                kl_div = ref_per_token_logps - per_token_logps
                per_token_kl = torch.exp(kl_div) - kl_div - 1.0

                # Importance sampling ratio
                ratio = torch.exp(per_token_logps - old_per_token_logps)

                if args.loss_type == "cispo":
                    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
                    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
                else:
                    clipped_ratio = torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon)
                    surr1 = ratio * advantages.unsqueeze(1)
                    surr2 = clipped_ratio * advantages.unsqueeze(1)
                    per_token_loss = -(torch.min(surr1, surr2) - args.beta * per_token_kl)

                # Masked policy loss
                masked_loss = per_token_loss * valid_mask.float()
                policy_loss = (masked_loss.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1).float()).mean()
                total_loss = policy_loss / args.accumulation_steps

            # Backward pass
            if use_scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            running_policy_loss += policy_loss.item()
            running_reward += rewards.mean().item()

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
                rollout_engine.update_policy(model)

            # Logging to Terminal
            if global_step % args.log_interval == 0:
                elapsed = time.time() - interval_start_time
                avg_loss = running_policy_loss / args.log_interval
                avg_rew = running_reward / args.log_interval

                steps_done = global_step
                steps_rem = total_steps - steps_done
                speed = args.log_interval / max(1e-4, elapsed)
                eta = steps_rem / max(1e-4, speed)

                percent = (steps_done / total_steps) * 100
                print(
                    f"[{percent:5.1f}%] GRPO [{epoch}/{args.epochs}] Step [{step}/{num_batches_per_epoch}] "
                    f"| Loss: {avg_loss:.4f} | Avg Reward: {avg_rew:.3f} "
                    f"| Speed: {speed:.1f} it/s | ETA: {format_time(eta)}"
                )
                running_policy_loss = 0.0
                running_reward = 0.0
                interval_start_time = time.time()

            # Save Checkpoint
            if global_step % args.save_interval == 0:
                save_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    config=model.config,
                    optimizer=optimizer,
                    step=global_step,
                    save_dir=args.save_dir,
                    prefix=args.save_weight,
                )

    # Save Final GRPO Policy Model
    final_save_dir = os.path.join(args.save_dir, f"{args.save_weight}_final")
    print(f"\n🎉 GRPO Training complete! Total time: {format_time(time.time() - start_time)}")
    print(f"💾 Saving final aligned GRPO model to: {final_save_dir}...")
    os.makedirs(final_save_dir, exist_ok=True)
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print("✅ GRPO aligned model weights and tokenizer saved successfully!")


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind GRPO (Group Relative Policy Optimization)")

    # Data and Model Paths
    parser.add_argument("--data_path", type=str, default="dataset/sft_vi.jsonl", help="Path to prompt dataset")
    parser.add_argument("--tokenizer_dir", type=str, default="model", help="Directory of trained tokenizer")
    parser.add_argument("--model_path", type=str, default="out/vimind_64m_final", help="Path to base/SFT model checkpoint")
    parser.add_argument("--save_dir", type=str, default="out/grpo", help="Directory to save GRPO models")
    parser.add_argument("--save_weight", type=str, default="vimind_grpo", help="Prefix for checkpoint filenames")

    # GRPO Hyperparameters
    parser.add_argument("--num_generations", type=int, default=4, help="Number of sampled completions per prompt (G)")
    parser.add_argument("--beta", type=float, default=0.05, help="KL divergence penalty coefficient")
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "cispo"], help="Loss type: standard grpo or cispo")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clipping epsilon for GRPO")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO ratio clipping upper bound")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for rollout")
    parser.add_argument("--thinking_ratio", type=float, default=0.5, help="Ratio of prompts initialized with <think> tag")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch"], help="Rollout engine backend")

    # Sequence Lengths & Optimization
    parser.add_argument("--max_seq_len", type=int, default=384, help="Max prompt sequence length")
    parser.add_argument("--max_gen_len", type=int, default=128, help="Max generation tokens")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size of prompts per step")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Policy learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")

    # Hardware & System
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cuda:0 or cpu)")
    parser.add_argument("--dtype", type=str, default="float16" if torch.cuda.is_available() else "float32", help="Precision (bfloat16, float16, float32)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--log_interval", type=int, default=5, help="Logging steps interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Saving steps interval")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train(args)
