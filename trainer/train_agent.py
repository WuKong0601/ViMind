import os
import sys
import re
import json
import math
import time
import random
import argparse
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import ViMindConfig, ViMindForCausalLM
from trainer.trainer_utils import Logger, setup_seed, get_lr, LMForRewardModel


# ==============================================================================
# 1. Tools Definition & Mock Environment (Cross-Platform Safe)
# ==============================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Thực hiện tính toán biểu thức toán học chính xác.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Biểu thức toán học (ví dụ: '125 * 8 + 50')"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Chuyển đổi giữa các đơn vị đo lường (dặm sang km, kg sang pound, v.v.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Tra cứu thông tin thời tiết tại một thành phố.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "Tên thành phố (ví dụ: 'Hà Nội')"}},
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Tra cứu ngày giờ hiện tại theo múi giờ.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "default": "Asia/Ho_Chi_Minh"}},
                "required": []
            }
        }
    }
]

WEATHER_DB = {
    "hà nội": {"temp": "26°C", "condition": "Nắng nhẹ, gió mát", "humidity": "68%"},
    "thành phố hồ chí minh": {"temp": "32°C", "condition": "Nắng ấm", "humidity": "75%"},
    "đà nẵng": {"temp": "29°C", "condition": "Mát mẻ, quang đãng", "humidity": "70%"},
    "hải phòng": {"temp": "25°C", "condition": "Mát mẻ", "humidity": "72%"},
    "cần thơ": {"temp": "31°C", "condition": "Nhiều mây", "humidity": "80%"}
}


def mock_execute_tool(name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Safe execution of tools without platform-dependent signal alarms."""
    try:
        if name == "calculate_math":
            expr = str(args.get("expression", "0")).replace("x", "*").replace("^", "**")
            # Safe eval with empty builtins
            res = eval(expr, {"__builtins__": {}, "math": math})
            return {"result": str(res)}
        elif name == "unit_converter":
            val = float(args.get("value", 0))
            f_u = str(args.get("from_unit", "")).lower()
            t_u = str(args.get("to_unit", "")).lower()
            if "mile" in f_u and "km" in t_u:
                return {"result": round(val * 1.60934, 4)}
            elif "km" in f_u and "mile" in t_u:
                return {"result": round(val / 1.60934, 4)}
            elif "kg" in f_u and "pound" in t_u:
                return {"result": round(val * 2.20462, 4)}
            return {"result": val}
        elif name == "get_current_weather":
            loc = str(args.get("location", "")).lower()
            for city, data in WEATHER_DB.items():
                if city in loc or loc in city:
                    return {"location": args.get("location"), **data}
            return {"location": args.get("location"), "temp": "27°C", "condition": "Quang đãng", "humidity": "65%"}
        elif name == "get_current_time":
            return {"datetime": time.strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Ho_Chi_Minh")}
    except Exception:
        return None
    return None


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extracts JSON tool calls embedded inside <tool_call> ... </tool_call> tags."""
    calls = []
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    for m in matches:
        try:
            call_obj = json.loads(m.strip())
            if isinstance(call_obj, dict) and "name" in call_obj:
                calls.append(call_obj)
        except Exception:
            pass
    return calls


# ==============================================================================
# 2. Reward Scoring Function
# ==============================================================================
def compute_agent_reward(prompt: str, generated_text: str, executed_tools: int, reward_model: Optional[LMForRewardModel] = None) -> float:
    reward = 0.0

    # 1. Format validity
    if "<tool_call>" in generated_text:
        if "</tool_call>" in generated_text:
            reward += 1.0  # Proper closing tag
            calls = parse_tool_calls(generated_text)
            if calls:
                reward += 1.0  # Successfully parsed valid JSON
                if executed_tools > 0:
                    reward += 1.5  # Tool successfully executed in environment
        else:
            reward -= 1.0  # Broken unclosed tag

    # 2. Reasoning tag validity
    if "<think>" in generated_text:
        if "</think>" in generated_text:
            reward += 0.5
        else:
            reward -= 0.5

    # 3. Repetition penalty
    tokens = re.findall(r"\w+", generated_text.lower())
    if len(tokens) > 10:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.4:
            reward -= 2.0  # Harsh penalty for degenerate loops

    # 4. Teacher Reward Model evaluation (if available)
    if reward_model is not None:
        ai_score = reward_model.get_score(prompt, generated_text)
        reward += 0.5 * ai_score

    return reward


# ==============================================================================
# 3. Agentic Dataset
# ==============================================================================
class AgentQueryDataset(Dataset):
    def __init__(self, data_path: str):
        self.queries = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if "conversations" in d and len(d["conversations"]) > 0:
                            self.queries.append(d["conversations"][0]["content"])
                        elif "prompt" in d:
                            self.queries.append(d["prompt"])
                    except Exception:
                        continue

        if not self.queries:
            self.queries = [
                "Tính giúp tôi biểu thức 250 * 4 + 1500 / 3.",
                "Thời tiết ở Hà Nội hôm nay thế nào?",
                "50 dặm bằng bao nhiêu km?",
                "Bây giờ là mấy giờ ở Việt Nam?",
                "Tính 12 * 12 + 100 bằng bao nhiêu?"
            ]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]


# ==============================================================================
# 4. Agentic RL Training Loop
# ==============================================================================
def train_agent(args):
    setup_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("🤖 VIMIND 3.0: AGENTIC REINFORCEMENT LEARNING (GRPO TOOL-USE)")
    print("=" * 70)
    print(f"   Model Checkpoint: {args.model_path}")
    print(f"   Reward Model: {args.reward_model_path}")
    print(f"   Device: {device}")
    print(f"   Learning Rate: {args.learning_rate}")
    print(f"   Epochs: {args.epochs}")
    print("=" * 70)

    # 1. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    print(f"📦 Loading Policy Model from: {args.model_path}...")
    actual_model_path = args.model_path
    if os.path.isdir(actual_model_path):
        if not os.path.exists(os.path.join(actual_model_path, "config.json")):
            # Look inside subfolders
            subfolders = [os.path.join(actual_model_path, d) for d in os.listdir(actual_model_path) if os.path.isdir(os.path.join(actual_model_path, d))]
            for sf in subfolders:
                if os.path.exists(os.path.join(sf, "config.json")):
                    actual_model_path = sf
                    break

    if os.path.exists(actual_model_path) and (os.path.isdir(actual_model_path) or actual_model_path.endswith((".safetensors", ".pth", ".bin"))):
        if os.path.isdir(actual_model_path):
            model = ViMindForCausalLM.from_pretrained(actual_model_path).to(device)
        else:
            config = ViMindConfig(use_moe=True, num_experts=4)
            model = ViMindForCausalLM(config)
            st = torch.load(actual_model_path, map_location="cpu") if not actual_model_path.endswith(".safetensors") else None
            if st:
                model.load_state_dict(st, strict=False)
            model = model.to(device)
    else:
        print(f"⚠️ Model path {args.model_path} not found directly, initializing fresh MoE architecture...")
        config = ViMindConfig(use_moe=True, num_experts=4)
        model = ViMindForCausalLM(config).to(device)

    model.train()

    reward_model = LMForRewardModel(args.reward_model_path, device=str(device)) if args.reward_model_path != "none" else None

    # 2. Dataset & Optimizer
    dataset = AgentQueryDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(loader) * args.epochs
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Bắt đầu Epoch {epoch}/{args.epochs} ---")
        for step, queries in enumerate(loader, start=1):
            global_step += 1
            lr = get_lr(global_step, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_rewards = []

            for query in queries:
                # Format prompt with tool system definition
                messages = [{"role": "user", "content": query}]
                context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=TOOLS)
                inputs = tokenizer(context, return_tensors="pt").to(device)

                # Generate tool call rollout
                model.eval()
                with torch.no_grad():
                    gen_tokens = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        top_p=0.85,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                model.train()

                gen_text = tokenizer.decode(gen_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

                # Parse & Execute tools
                calls = parse_tool_calls(gen_text)
                executed_count = 0
                for call in calls:
                    fn_name = call.get("name")
                    fn_args = call.get("arguments", {})
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            fn_args = {}
                    res = mock_execute_tool(fn_name, fn_args)
                    if res is not None:
                        executed_count += 1

                # Compute reward
                R = compute_agent_reward(query, gen_text, executed_count, reward_model)
                batch_rewards.append(R)

                # Policy gradient loss step (GRPO-style advantage weighting)
                full_ids = gen_tokens.clone().detach()
                outputs = model(full_ids, labels=full_ids)
                # Maximize reward -> minimize -R * log_prob
                loss = outputs.loss * (-R if R < 0 else 1.0 / (R + 1.0))
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / max(1, len(queries))
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step % args.log_interval == 0 or step == len(loader):
                avg_r = sum(batch_rewards) / max(1, len(batch_rewards))
                print(f"[{global_step}/{total_steps}] Step {step}/{len(loader)} | Loss: {batch_loss.item():.4f} | Avg Reward: {avg_r:.2f} | LR: {lr:.2e}")

    # Save final agent model
    os.makedirs(args.save_dir, exist_ok=True)
    final_save_dir = os.path.join(args.save_dir, f"{args.save_weight}_final")
    pth_save_path = os.path.join(args.save_dir, f"{args.save_weight}.pth")
    print(f"\n🎉 Agentic RL Hoàn thành! Đang lưu mô hình ra: {final_save_dir} & {pth_save_path}...")
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    torch.save(model.state_dict(), pth_save_path)
    print("✅ Mô hình Agent ViMind 3.0 đã sẵn sàng xuất xưởng!")


def get_parser():
    parser = argparse.ArgumentParser(description="ViMind 3.0 Agentic RL Trainer")
    parser.add_argument("--model_path", type=str, default="out/dpo/vimind_64m_dpo_final", help="Base model checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, default="model", help="Tokenizer directory")
    parser.add_argument("--reward_model_path", type=str, default="none", help="Path to AI reward model")
    parser.add_argument("--data_path", type=str, default="dataset/sft_toolcall_vi.jsonl", help="Path to queries dataset")
    parser.add_argument("--save_dir", type=str, default="out/agent", help="Directory to save agent models")
    parser.add_argument("--save_weight", type=str, default="vimind_3.0_agent", help="Prefix for save weight")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_rollouts", type=int, default=4, help="Number of rollouts per sample")
    parser.add_argument("--fp16", action="store_true", help="Enable half precision training")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--log_interval", type=int, default=5, help="Logging step interval")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    train_agent(args)
