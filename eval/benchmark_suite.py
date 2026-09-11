"""
ViMind 4.5 - Standard Benchmark & Evaluation Suite.
Quantitative Assessment across 4 Key Cognitive Pillars:
  1. Factual Grounding (Vietnam Geography, History, Science)
  2. Mathematical Reasoning with <think> Chain-of-Thought
  3. Real-Time Tool Calling (<tool_call> XML/JSON Schema)
  4. Multi-Turn Conversational Resilience (Topic Switching & Anti-Anchoring)

Generates quantitative scores and an academic results table for paper publication.
"""

import os
import sys
import re
import json
import time
import argparse

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.model import ViMindConfig, ViMindForCausalLM


# -------------------------------------------------------------
# Benchmark Test Battery
# -------------------------------------------------------------

BENCHMARK_TASKS = {
    "factual": [
        {
            "id": "fact_01",
            "prompt": "Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là gì?",
            "expected_keywords": ["Hà Nội", "hà nội", "Hanoi"]
        },
        {
            "id": "fact_02",
            "prompt": "Việt Nam hiện có tất cả bao nhiêu tỉnh và thành phố trực thuộc Trung ương?",
            "expected_keywords": ["63", "sáu mươi ba", "58 tỉnh"]
        },
        {
            "id": "fact_03",
            "prompt": "Chủ tịch Hồ Chí Minh đọc bản Tuyên ngôn Độc lập vào ngày tháng năm nào và ở đâu?",
            "expected_keywords": ["2/9/1945", "2 tháng 9 năm 1945", "Ba Đình"]
        },
        {
            "id": "fact_04",
            "prompt": "Đỉnh núi nào cao nhất Việt Nam và Đông Dương?",
            "expected_keywords": ["Fansipan", "Phan Xi Păng", "Phan-xi-păng"]
        },
        {
            "id": "fact_05",
            "prompt": "Tại sao Trái Đất lại có hiện tượng ngày và đêm luân phiên?",
            "expected_keywords": ["tự quay", "hình cầu", "quanh trục"]
        }
    ],
    "math_cot": [
        {
            "id": "math_01",
            "prompt": "Tính giúp tôi kết quả của 25 * 18 + 750 / 5.",
            "expected_number": 600,
            "intermediate": [450, 150]
        },
        {
            "id": "math_02",
            "prompt": "Kết quả của (95 + 22) * 7 là bao nhiêu?",
            "expected_number": 819,
            "intermediate": [117]
        },
        {
            "id": "math_03",
            "prompt": "Tính giúp tôi kết quả của 41 * 26 + 920 / 10.",
            "expected_number": 1158,
            "intermediate": [1066, 92]
        },
        {
            "id": "math_04",
            "prompt": "Một cửa hàng có 11 bao gạo, mỗi bao nặng 25 kg. Cửa hàng bán đi 64 kg. Hỏi còn lại bao nhiêu kg gạo?",
            "expected_number": 211,
            "intermediate": [275]
        }
    ],
    "tool_call": [
        {
            "id": "tool_01",
            "prompt": "Thời tiết hiện tại ở Đà Nẵng thế nào?",
            "expected_tool": "get_current_weather",
            "expected_arg": "Đà Nẵng"
        },
        {
            "id": "tool_02",
            "prompt": "Cho tôi biết thời tiết hôm nay tại Hà Nội.",
            "expected_tool": "get_current_weather",
            "expected_arg": "Hà Nội"
        },
        {
            "id": "tool_03",
            "prompt": "Bây giờ là mấy giờ rồi?",
            "expected_tool": "get_current_time"
        }
    ],
    "multiturn": [
        {
            "id": "multi_01",
            "turn_1": "Thủ đô của Việt Nam là gì?",
            "turn_2": "Cảm ơn bạn. Bây giờ hãy tính giúp tôi kết quả của 25 * 18 + 750 / 5.",
            "unwanted_in_turn_2": ["Hà Nội", "hà nội"],
            "expected_in_turn_2": ["600", "450"]
        },
        {
            "id": "multi_02",
            "turn_1": "Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập vào ngày nào?",
            "turn_2": "Tiếp theo, thời tiết hiện tại ở Đà Nẵng thế nào?",
            "unwanted_in_turn_2": ["1945", "Tuyên ngôn"],
            "expected_in_turn_2": ["get_current_weather", "Đà Nẵng"]
        }
    ]
}


def load_tested_model(model_path: str, device: str):
    config_path = os.path.join(model_path, "config.json")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_qwen = False
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("model_type") == "qwen2":
                is_qwen = True
        except Exception:
            pass

    if is_qwen:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None
        )
        if "cuda" not in device:
            model.to(device)
        model_type = "pro"
    else:
        config = ViMindConfig.from_pretrained(model_path)
        model = ViMindForCausalLM(config)
        sf_path = os.path.join(model_path, "model.safetensors")
        if os.path.exists(sf_path):
            model.load_state_dict(load_file(sf_path), strict=False)
        model.to(device)
        model_type = "native"

    model.eval()
    return model, tokenizer, model_type


def generate_response(model, tokenizer, prompt_or_msgs, model_type, device, max_new_tokens=256):
    if isinstance(prompt_or_msgs, str):
        messages = [{"role": "user", "content": prompt_or_msgs}]
    else:
        messages = prompt_or_msgs

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.08,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0
    new_tokens = out[0][inputs.input_ids.shape[1] :]
    resp = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=False).strip()
    tok_count = len(new_tokens)
    tok_per_sec = tok_count / max(gen_time, 1e-4)
    return resp, gen_time, tok_per_sec


def run_benchmark(args):
    device = "cuda:0" if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"
    print("=" * 75)
    print(f"🔬 VIMIND STANDARD BENCHMARK SUITE")
    print(f"📍 Model Path: {args.model_path}")
    print(f"⚡ Device:     {device}")
    print("=" * 75)

    model, tokenizer, model_type = load_tested_model(args.model_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📦 Model Architecture: {'Qwen2.5 Foundation (Pro)' if model_type == 'pro' else 'Native MoE 198M/64M'}")
    print(f"📊 Parameter Count:    {total_params/1e6:.1f}M")
    print("-" * 75)

    results = {
        "model_path": args.model_path,
        "model_type": model_type,
        "parameters_m": total_params / 1e6,
        "device": device,
        "scores": {}
    }

    # 1. Evaluate Factual Grounding
    print("\n[1/4] Evaluating Factual Grounding...")
    factual_correct = 0
    fact_details = []
    for item in BENCHMARK_TASKS["factual"]:
        resp, _, _ = generate_response(model, tokenizer, item["prompt"], model_type, device)
        clean = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL)
        is_pass = any(kw.lower() in clean.lower() for kw in item["expected_keywords"])
        if is_pass:
            factual_correct += 1
        fact_details.append({"id": item["id"], "pass": is_pass, "resp": clean[:100]})
        print(f"   - {item['id']}: {'✅ PASS' if is_pass else '❌ FAIL'}")

    fact_acc = (factual_correct / len(BENCHMARK_TASKS["factual"])) * 100
    results["scores"]["factual_accuracy"] = fact_acc

    # 2. Evaluate Mathematical Reasoning (CoT)
    print("\n[2/4] Evaluating Mathematical Reasoning (CoT)...")
    math_correct = 0
    cot_presence = 0
    for item in BENCHMARK_TASKS["math_cot"]:
        resp, _, _ = generate_response(model, tokenizer, item["prompt"], model_type, device)
        has_think = "<think>" in resp and "</think>" in resp
        if has_think:
            cot_presence += 1
        
        # Check numerical answer in output
        exp_str = str(item["expected_number"])
        clean = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL)
        is_pass = exp_str in clean or exp_str in resp
        if is_pass:
            math_correct += 1
        print(f"   - {item['id']}: {'✅ PASS' if is_pass else '❌ FAIL'} (CoT: {'Yes' if has_think else 'No'})")

    math_acc = (math_correct / len(BENCHMARK_TASKS["math_cot"])) * 100
    cot_rate = (cot_presence / len(BENCHMARK_TASKS["math_cot"])) * 100
    results["scores"]["math_accuracy"] = math_acc
    results["scores"]["cot_rate"] = cot_rate

    # 3. Evaluate Tool Calling
    print("\n[3/4] Evaluating Real-time Tool Calling...")
    tool_correct = 0
    for item in BENCHMARK_TASKS["tool_call"]:
        resp, _, _ = generate_response(model, tokenizer, item["prompt"], model_type, device)
        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", resp, re.DOTALL)
        is_pass = False
        if tc_match:
            try:
                data = json.loads(tc_match.group(1).strip())
                if data.get("name") == item["expected_tool"]:
                    if "expected_arg" in item:
                        args_str = json.dumps(data.get("arguments", {}), ensure_ascii=False)
                        is_pass = item["expected_arg"] in args_str
                    else:
                        is_pass = True
            except Exception:
                pass
        if is_pass:
            tool_correct += 1
        print(f"   - {item['id']}: {'✅ PASS' if is_pass else '❌ FAIL'}")

    tool_acc = (tool_correct / len(BENCHMARK_TASKS["tool_call"])) * 100
    results["scores"]["tool_calling_accuracy"] = tool_acc

    # 4. Evaluate Multi-Turn Conversational Resilience
    print("\n[4/4] Evaluating Multi-Turn Conversational Resilience...")
    multi_correct = 0
    for item in BENCHMARK_TASKS["multiturn"]:
        # Run Turn 1
        resp_t1, _, _ = generate_response(model, tokenizer, item["turn_1"], model_type, device)
        clean_t1 = re.sub(r"<think>.*?</think>", "", resp_t1, flags=re.DOTALL).strip()
        
        # Build Turn 2 dialog history
        history = [
            {"role": "user", "content": item["turn_1"]},
            {"role": "assistant", "content": clean_t1},
            {"role": "user", "content": item["turn_2"]}
        ]
        resp_t2, _, _ = generate_response(model, tokenizer, history, model_type, device)
        clean_t2 = re.sub(r"<think>.*?</think>", "", resp_t2, flags=re.DOTALL)

        # Pass condition: must NOT anchor on unwanted previous keywords, and must contain expected new target
        no_drift = not any(unw.lower() in clean_t2.lower() for unw in item["unwanted_in_turn_2"])
        has_target = any(exp.lower() in resp_t2.lower() for exp in item["expected_in_turn_2"])
        is_pass = no_drift and has_target
        if is_pass:
            multi_correct += 1
        print(f"   - {item['id']}: {'✅ PASS (No Anchoring Drift)' if is_pass else '❌ FAIL (Context Anchoring Drifted)'}")

    multi_acc = (multi_correct / len(BENCHMARK_TASKS["multiturn"])) * 100
    results["scores"]["multiturn_resilience"] = multi_acc

    # ---------------------------------------------------------
    # Benchmark Summary Table
    # ---------------------------------------------------------
    overall_score = (fact_acc + math_acc + tool_acc + multi_acc) / 4.0
    results["scores"]["overall_score"] = overall_score

    print("\n" + "=" * 75)
    print("📊 VIMIND BENCHMARK SCORECARD")
    print("=" * 75)
    print(f"🏛️ Factual Grounding (Vietnam Knowledge): {fact_acc:6.1f}%")
    print(f"🧮 Mathematical Reasoning (Step-by-Step): {math_acc:6.1f}%  (CoT: {cot_rate:.0f}%)")
    print(f"🔧 Real-Time Tool Calling (<tool_call>): {tool_acc:6.1f}%")
    print(f"💬 Multi-Turn Resilience (No Anchoring): {multi_acc:6.1f}%")
    print("-" * 75)
    print(f"🏆 OVERALL COGNITIVE SCORE:              {overall_score:6.1f}%")
    print("=" * 75)

    # Save to JSON
    out_file = args.output_report
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📁 Benchmark report written to: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViMind Benchmark Suite")
    parser.add_argument("--model_path", type=str, default="out/vimind_4.0_pro_final", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_report", type=str, default="benchmark_report.json", help="Path to save output JSON report")
    args = parser.parse_args()
    run_benchmark(args)
