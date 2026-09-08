"""
ViMind 3.0 - OpenAI Compatible Client Demonstration & Interactive Chat CLI.
Supports both the official `openai` Python SDK and built-in `requests` SSE streaming.
Demonstrates:
  1. Standard streaming chat
  2. Live `<think>` reasoning block display
  3. Automatic function calling / tool execution loop
"""

import os
import sys
import json
import argparse
import datetime

# Force UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import requests
except ImportError:
    print("[Error] 'requests' library is required. Run: pip install requests")
    sys.exit(1)


# ANSI Color Codes for beautiful terminal styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Thực hiện phép tính toán học an toàn (số học, lũy thừa, căn bậc hai).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Biểu thức toán học, ví dụ: '25 * 4 + 100 / 2'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Tra cứu thời tiết thời gian thực tại một thành phố hoặc tỉnh thành ở Việt Nam.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Tên thành phố, ví dụ: 'Hà Nội', 'TP. Hồ Chí Minh', 'Đà Nẵng'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Lấy ngày giờ hiện tại theo múi giờ Việt Nam (Asia/Ho_Chi_Minh).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def execute_local_tool(func_name: str, args: dict) -> str:
    """Executes a local tool call returned by ViMind."""
    if func_name == "calculate_math":
        expr = args.get("expression", "")
        allowed = set("0123456789+-*/(). %^")
        if all(c in allowed for c in expr):
            try:
                res = eval(expr, {"__builtins__": None}, {})
                return json.dumps({"result": res, "status": "success"}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e), "status": "failed"}, ensure_ascii=False)
        return json.dumps({"error": "Biểu thức chứa ký tự không an toàn", "status": "failed"}, ensure_ascii=False)

    elif func_name == "get_current_weather":
        loc = args.get("location", "Hà Nội")
        # Mock weather provider
        return json.dumps({
            "location": loc,
            "temperature_c": 28.5,
            "condition": "Nắng nhẹ, có mây rải rác",
            "humidity": "68%",
            "source": "ViMind Weather API Mock"
        }, ensure_ascii=False)

    elif func_name == "get_current_time":
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S (GMT+7)")
        return json.dumps({"current_time": now_str, "timezone": "Asia/Ho_Chi_Minh"}, ensure_ascii=False)

    return json.dumps({"error": f"Unknown tool: {func_name}"}, ensure_ascii=False)


def stream_chat_completion(
    base_url: str,
    messages: list,
    tools: list = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    open_thinking: bool = True
):
    """
    Streams completion from ViMind OpenAI-compatible server using Server-Sent Events (SSE).
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "vimind-3.0",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "open_thinking": open_thinking
    }
    if tools:
        payload["tools"] = tools

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
    except requests.exceptions.ConnectionError:
        print(f"\n{YELLOW}[Lỗi kết nối]{RESET} Không thể kết nối tới máy chủ ViMind tại {url}.")
        print("Hãy chắc chắn bạn đã khởi chạy máy chủ:")
        print(f"  python scripts/serve_openai_api.py --model_path out/sft_moe --port 8000")
        return None

    if response.status_code != 200:
        print(f"\n{YELLOW}[HTTP Error {response.status_code}]{RESET} {response.text}")
        return None

    full_content = ""
    full_reasoning = ""
    tool_calls = []
    in_reasoning = False

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0]["delta"]

                # Extract reasoning content (DeepSeek-R1 / Qwen style)
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    if not in_reasoning:
                        print(f"\n{DIM}{MAGENTA}[Suy nghĩ / Reasoning]:{RESET}\n{DIM}", end="", flush=True)
                        in_reasoning = True
                    print(reasoning, end="", flush=True)
                    full_reasoning += reasoning

                # Extract standard content
                content = delta.get("content")
                if content:
                    if in_reasoning:
                        print(f"{RESET}\n\n{BOLD}{GREEN}[Trả lời]:{RESET}\n", end="", flush=True)
                        in_reasoning = False
                    print(content, end="", flush=True)
                    full_content += content

                # Extract tool calls
                if "tool_calls" in delta and delta["tool_calls"]:
                    tool_calls.extend(delta["tool_calls"])

            except Exception:
                pass

    if in_reasoning:
        print(f"{RESET}\n")

    return {
        "content": full_content,
        "reasoning_content": full_reasoning,
        "tool_calls": tool_calls
    }


def main():
    parser = argparse.ArgumentParser(description="ViMind 3.0 Client & Interactive Chat CLI")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible server URL")
    parser.add_argument("--prompt", type=str, default=None, help="Single query mode (non-interactive)")
    parser.add_argument("--enable_tools", action="store_true", help="Enable tool calling demonstration")
    parser.add_argument("--no_thinking", action="store_true", help="Disable <think> reasoning tokens")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max output tokens")
    args = parser.parse_args()

    print(f"\n{CYAN}{BOLD}======================================================{RESET}")
    print(f"{CYAN}{BOLD}   ViMind 3.0 - AI Chat & Function Calling Client   {RESET}")
    print(f"{CYAN}{BOLD}======================================================{RESET}")
    print(f"Target Server: {args.base_url}")
    print(f"Tools Enabled: {args.enable_tools}")
    print(f"Reasoning Enabled: {not args.no_thinking}\n")

    tools = SAMPLE_TOOLS if args.enable_tools else None
    messages = [
        {"role": "system", "content": "Bạn là ViMind 3.0, trợ lý AI tiếng Việt thông minh, trung thực và hữu ích."}
    ]

    # Non-interactive single prompt mode
    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        print(f"{BOLD}User:{RESET} {args.prompt}")
        res = stream_chat_completion(
            args.base_url, messages, tools=tools,
            temperature=args.temp, max_tokens=args.max_tokens,
            open_thinking=not args.no_thinking
        )
        if res and res.get("tool_calls"):
            print(f"\n\n{YELLOW}[ViMind yêu cầu gọi công cụ / Tool Call]:{RESET}")
            for tc in res["tool_calls"]:
                fn = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                print(f"  -> Công cụ: {BOLD}{fn}{RESET} với tham số: {fn_args}")
                tool_out = execute_local_tool(fn, fn_args)
                print(f"  -> Kết quả thực thi: {tool_out}")
        print("\n")
        return

    # Interactive CLI loop
    print("Nhập tin nhắn của bạn (Gõ 'exit' hoặc 'quit' để thoát, 'clear' để xóa lịch sử):")
    while True:
        try:
            user_input = input(f"\n{BOLD}{CYAN}Bạn > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTạm biệt!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Tạm biệt!")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": "Bạn là ViMind 3.0, trợ lý AI tiếng Việt thông minh, trung thực và hữu ích."}]
            print(f"{YELLOW}[Đã làm mới đoạn hội thoại]{RESET}")
            continue

        messages.append({"role": "user", "content": user_input})

        res = stream_chat_completion(
            args.base_url,
            messages,
            tools=tools,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            open_thinking=not args.no_thinking
        )

        if res is None:
            continue

        # Handle tool call loop if any
        if res.get("tool_calls"):
            assistant_msg = {"role": "assistant", "content": res["content"]}
            if res.get("reasoning_content"):
                assistant_msg["reasoning_content"] = res["reasoning_content"]
            assistant_msg["tool_calls"] = res["tool_calls"]
            messages.append(assistant_msg)

            print(f"\n{YELLOW}[Thực thi công cụ được yêu cầu]:{RESET}")
            for tc in res["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args_raw = tc["function"]["arguments"]
                fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                print(f" -> Đang gọi: {BOLD}{fn_name}{RESET}({json.dumps(fn_args, ensure_ascii=False)})")

                tool_result_str = execute_local_tool(fn_name, fn_args)
                print(f" -> Trả về: {tool_result_str}")

                messages.append({
                    "role": "tool",
                    "content": tool_result_str
                })

            print(f"\n{BOLD}{GREEN}[ViMind tổng hợp sau khi gọi công cụ]:{RESET}\n")
            follow_up = stream_chat_completion(
                args.base_url,
                messages,
                tools=tools,
                temperature=args.temp,
                max_tokens=args.max_tokens,
                open_thinking=not args.no_thinking
            )
            if follow_up:
                messages.append({"role": "assistant", "content": follow_up["content"]})
        else:
            messages.append({"role": "assistant", "content": res["content"]})


if __name__ == "__main__":
    main()
