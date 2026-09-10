"""
ViMind 3.0 - Tool Calling & Function Calling Benchmark Evaluation Script.
Evaluates:
  1. Tool Trigger Precision & Recall (Tool selection accuracy vs. direct answering)
  2. JSON Syntactic Validity within <tool_call>...</tool_call> tags
  3. Function Name and Parameter Extraction Correctness
"""

import os
import sys
import re
import json
import argparse
import time
from typing import List, Dict, Any

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoTokenizer
from model.model import ViMindConfig, ViMindForCausalLM


EVAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Thực hiện các phép tính toán học số học.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Biểu thức toán"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Tra cứu thông tin thời tiết.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Tên địa điểm"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Tra cứu ngày giờ hiện tại theo múi giờ Việt Nam.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Tìm kiếm thông tin trong cơ sở dữ liệu nội bộ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Từ khóa tìm kiếm"}
                },
                "required": ["query"]
            }
        }
    }
]

# Benchmark Test Suite: Mixture of tool-required queries and direct conversational queries
BENCHMARK_CASES = [
    {
        "id": "TC-01",
        "prompt": "Tính giúp tôi kết quả của 458 * 12 + 950.",
        "should_call_tool": True,
        "expected_tool": "calculate_math",
        "expected_arg_key": "expression"
    },
    {
        "id": "TC-02",
        "prompt": "Bây giờ là mấy giờ rồi bạn?",
        "should_call_tool": True,
        "expected_tool": "get_current_time",
        "expected_arg_key": None
    },
    {
        "id": "TC-03",
        "prompt": "Thời tiết hôm nay ở Đà Lạt thế nào, có mưa không?",
        "should_call_tool": True,
        "expected_tool": "get_current_weather",
        "expected_arg_key": "location",
        "expected_arg_val": "Đà Lạt"
    },
    {
        "id": "TC-04",
        "prompt": "Tìm thông tin báo cáo tài chính quý 3 năm 2024 trong hệ thống.",
        "should_call_tool": True,
        "expected_tool": "search_database",
        "expected_arg_key": "query"
    },
    {
        "id": "TC-05",
        "prompt": "Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là gì?",
        "should_call_tool": False,
        "expected_tool": None,
        "expected_arg_key": None
    },
    {
        "id": "TC-06",
        "prompt": "Hãy viết cho tôi một đoạn thơ 4 câu về mùa thu Hà Nội.",
        "should_call_tool": False,
        "expected_tool": None,
        "expected_arg_key": None
    },
    {
        "id": "TC-07",
        "prompt": "Nhiệt độ hiện tại ở TP. Hồ Chí Minh là bao nhiêu?",
        "should_call_tool": True,
        "expected_tool": "get_current_weather",
        "expected_arg_key": "location"
    },
    {
        "id": "TC-08",
        "prompt": "Giải thích cho tôi khái niệm Mixture-of-Experts trong AI.",
        "should_call_tool": False,
        "expected_tool": None,
        "expected_arg_key": None
    }
]


def parse_tool_call_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract and validate JSON tool calls from raw response."""
    calls = []
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        cleaned = m.strip()
        try:
            parsed = json.loads(cleaned)
            calls.append({"raw": cleaned, "valid_json": True, "data": parsed})
        except json.JSONDecodeError:
            calls.append({"raw": cleaned, "valid_json": False, "data": None})
    return calls


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 256) -> str:
    """Generate model response with tools system prompt."""
    messages = [
        {"role": "system", "content": "Bạn là ViMind 3.0, trợ lý AI tiếng Việt có khả năng sử dụng công cụ."},
        {"role": "user", "content": prompt}
    ]

    try:
        input_prompt = tokenizer.apply_chat_template(
            messages,
            tools=EVAL_TOOLS,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback manual formatting
        tools_str = json.dumps(EVAL_TOOLS, ensure_ascii=False)
        input_prompt = f"<|im_start|>system\nBạn có các công cụ sau:\n{tools_str}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.85,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    full_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
    return full_output


def run_benchmark(model_path: str, device_str: str = "cuda" if torch.cuda.is_available() else "cpu"):
    print("=" * 70)
    print("      ViMind 3.0 - Tool Calling Benchmark Suite")
    print("=" * 70)
    print(f"Model Path: {model_path}")
    print(f"Device: {device_str}\n")

    tokenizer_path = os.path.join(PROJECT_ROOT, "model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if checkpoint exists
    has_weights = os.path.exists(model_path) and (
        os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or
        os.path.exists(os.path.join(model_path, "model.safetensors")) or
        os.path.exists(os.path.join(model_path, "vimind_4.0_agent.pth")) or
        os.path.exists(os.path.join(model_path, "vimind_4.0_dpo.pth")) or
        os.path.exists(os.path.join(model_path, "vimind_4.0_moe.pth")) or
        os.path.exists(os.path.join(model_path, "vimind_sft.pth")) or
        os.path.exists(os.path.join(model_path, "vimind_agent_rl.pth")) or
        model_path.endswith((".pth", ".safetensors", ".bin"))
    )

    if has_weights:
        print(f"Loading ViMind weights from {model_path}...")
        config = ViMindConfig.from_pretrained(model_path) if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")) else ViMindConfig(vocab_size=len(tokenizer), use_moe=True, num_experts=4, num_experts_per_tok=2)
        model = ViMindForCausalLM(config)
        # Load weights
        weight_file = None
        if os.path.isfile(model_path) and model_path.endswith((".pth", ".safetensors", ".bin")):
            weight_file = model_path
        else:
            for wf in ["model.safetensors", "vimind_4.0_agent.pth", "vimind_4.0_dpo.pth", "vimind_4.0_moe.pth", "vimind_3.0_agent.pth", "pytorch_model.bin", "vimind_sft.pth", "vimind_agent_rl.pth"]:
                candidate = os.path.join(model_path, wf)
                if os.path.exists(candidate):
                    weight_file = candidate
                    break
        if weight_file:
            if weight_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(weight_file)
            else:
                state = torch.load(weight_file, map_location="cpu")

            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            state = {k.replace("module.", ""): v for k, v in state.items()}

            embed_key = next((k for k in ["model.embed_tokens.weight", "embed_tokens.weight"] if k in state), None)
            if embed_key and state[embed_key].shape[1] != model.config.hidden_size:
                ckpt_h = state[embed_key].shape[1]
                if ckpt_h == 640:
                    config = ViMindConfig(
                        vocab_size=state[embed_key].shape[0],
                        hidden_size=640,
                        num_hidden_layers=12,
                        num_attention_heads=10,
                        num_key_value_heads=5,
                        intermediate_size=1728,
                        max_seq_len=2048,
                        use_moe=True,
                        num_experts=4,
                        num_experts_per_tok=2
                    )
                    model = ViMindForCausalLM(config)

            model.load_state_dict(state, strict=False)
        model.to(device_str)
        model.eval()
    else:
        print(f"[Notice] No trained weights found at {model_path}. Running schema & template dry-run evaluation mode...")
        model = None

    results = []
    total_tool_cases = sum(1 for c in BENCHMARK_CASES if c["should_call_tool"])
    total_direct_cases = sum(1 for c in BENCHMARK_CASES if not c["should_call_tool"])

    correct_triggers = 0
    correct_tool_selections = 0
    valid_json_count = 0
    total_tool_calls_emitted = 0
    correct_non_triggers = 0

    for case in BENCHMARK_CASES:
        print(f"\nEvaluating [{case['id']}]: '{case['prompt']}'")
        if model is not None:
            raw_output = generate_response(model, tokenizer, case["prompt"], device_str)
        else:
            # Simulated golden test response for pipeline verification
            if case["should_call_tool"]:
                if case["expected_tool"] == "calculate_math":
                    raw_output = '<think>Cần tính toán số học.</think><tool_call>{"name": "calculate_math", "arguments": {"expression": "458 * 12 + 950"}}</tool_call>'
                elif case["expected_tool"] == "get_current_time":
                    raw_output = '<think>Người dùng hỏi giờ hiện tại.</think><tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>'
                elif case["expected_tool"] == "get_current_weather":
                    raw_output = f'<think>Tra cứu thời tiết.</think><tool_call>{{"name": "get_current_weather", "arguments": {{"location": "{case.get("expected_arg_val", "Hồ Chí Minh")}"}}}}</tool_call>'
                elif case["expected_tool"] == "search_database":
                    raw_output = '<think>Tìm báo cáo.</think><tool_call>{"name": "search_database", "arguments": {"query": "báo cáo tài chính quý 3 2024"}}</tool_call>'
            else:
                raw_output = '<think>Câu hỏi kiến thức tổng quát, không cần công cụ.</think>Hà Nội là thủ đô của Việt Nam.'

        parsed_calls = parse_tool_call_from_text(raw_output)
        called_tool = len(parsed_calls) > 0
        print(f"  [Model Output]: {raw_output.strip()[:140]}")

        case_res = {
            "id": case["id"],
            "prompt": case["prompt"],
            "expected_call": case["should_call_tool"],
            "actual_call": called_tool,
            "tool_match": False,
            "valid_json": False,
            "raw_output": raw_output[:120] + "..." if len(raw_output) > 120 else raw_output
        }

        if case["should_call_tool"]:
            if called_tool:
                correct_triggers += 1
                first_call = parsed_calls[0]
                total_tool_calls_emitted += 1
                if first_call["valid_json"]:
                    valid_json_count += 1
                    func_name = first_call["data"].get("name")
                    if func_name == case["expected_tool"]:
                        correct_tool_selections += 1
                        case_res["tool_match"] = True
                    case_res["valid_json"] = True
                print(f"  -> Triggered Tool: {parsed_calls[0].get('data', {}).get('name', 'INVALID_JSON')} (Expected: {case['expected_tool']})")
            else:
                print(f"  -> Failed to trigger tool! (Expected: {case['expected_tool']})")
        else:
            if not called_tool:
                correct_non_triggers += 1
                case_res["tool_match"] = True
                print("  -> Correctly answered directly without calling tools.")
            else:
                print("  -> False positive tool trigger!")

        results.append(case_res)

    # Metrics computation
    trigger_recall = (correct_triggers / total_tool_cases) * 100 if total_tool_cases > 0 else 0
    trigger_precision = (correct_triggers / (correct_triggers + (total_direct_cases - correct_non_triggers))) * 100 if (correct_triggers + (total_direct_cases - correct_non_triggers)) > 0 else 0
    tool_acc = (correct_tool_selections / total_tool_cases) * 100 if total_tool_cases > 0 else 0
    json_validity_rate = (valid_json_count / total_tool_calls_emitted) * 100 if total_tool_calls_emitted > 0 else 100.0

    print("\n" + "=" * 70)
    print("                    BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  * Tool Trigger Recall:          {trigger_recall:.1f}% ({correct_triggers}/{total_tool_cases})")
    print(f"  * Direct Answer Precision:       {(correct_non_triggers / total_direct_cases) * 100:.1f}% ({correct_non_triggers}/{total_direct_cases})")
    print(f"  * Correct Tool Selection Acc:   {tool_acc:.1f}% ({correct_tool_selections}/{total_tool_cases})")
    print(f"  * JSON Syntax Validity Rate:     {json_validity_rate:.1f}% ({valid_json_count}/{total_tool_calls_emitted})")
    print("=" * 70)

    # Save benchmark report to json
    report_file = os.path.join(PROJECT_ROOT, "benchmark_toolcall_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "trigger_recall": trigger_recall,
                "tool_selection_accuracy": tool_acc,
                "json_validity_rate": json_validity_rate,
                "cases_evaluated": len(BENCHMARK_CASES)
            },
            "cases": results
        }, f, ensure_ascii=False, indent=2)

    print(f"Report successfully saved to: {report_file}")
    return report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViMind 3.0 Tool Calling Benchmark")
    parser.add_argument("--model_path", type=str, default="out/sft_moe", help="Path to trained model weights directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_benchmark(args.model_path, args.device)
