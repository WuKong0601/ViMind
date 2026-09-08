import os
import sys
import re
import json
import time
import uuid
import argparse
from typing import List, Dict, Any, Optional
from threading import Thread
from queue import Queue

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, TextIteratorStreamer

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


app = FastAPI(title="ViMind 3.0 OpenAI Compatible API Server")

# Enable CORS for all chat web clients (Open-WebUI, Cherry Studio, NextChat, LibreChat)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
server_device = None


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = ""
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    model: str = "vimind-3.0"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.85
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    open_thinking: Optional[bool] = True


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "vimind-3.0",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vimind",
            },
            {
                "id": "vimind-3.0-moe",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vimind",
            }
        ],
    }


def parse_response_tool_calls(text: str) -> List[Dict[str, Any]]:
    tool_calls = []
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    for m in matches:
        try:
            call_obj = json.loads(m.strip())
            if isinstance(call_obj, dict) and "name" in call_obj:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": call_obj.get("name"),
                        "arguments": json.dumps(call_obj.get("arguments", {}), ensure_ascii=False)
                    }
                })
        except Exception:
            pass
    return tool_calls


def extract_thinking_and_content(text: str):
    reasoning_content = ""
    content = text
    if "</think>" in text:
        parts = text.split("</think>")
        reasoning_content = parts[0].replace("<think>", "").strip()
        content = parts[1].strip()
    elif "<think>" in text:
        reasoning_content = text.replace("<think>", "").strip()
        content = ""
    return reasoning_content, content


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global model, tokenizer, server_device
    if model is None:
        raise HTTPException(status_code=500, detail="Mô hình ViMind chưa được khởi tạo!")

    # Format messages
    raw_msgs = []
    for m in req.messages:
        msg_dict = {"role": m.role, "content": m.content or ""}
        if m.reasoning_content:
            msg_dict["reasoning_content"] = m.reasoning_content
        if m.tool_calls:
            msg_dict["tool_calls"] = m.tool_calls
        raw_msgs.append(msg_dict)

    prompt = tokenizer.apply_chat_template(
        raw_msgs,
        tokenize=False,
        add_generation_prompt=True,
        tools=req.tools,
        open_thinking=req.open_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(server_device)
    prompt_len = inputs["input_ids"].shape[1]
    req_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())

    # ---------------- Non-Streaming Response ----------------
    if not req.stream:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_tokens = outputs[0][prompt_len:]
        full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

        reasoning, content = extract_thinking_and_content(raw_text)
        tool_calls = parse_response_tool_calls(raw_text)

        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }
        if reasoning:
            choice["message"]["reasoning_content"] = reasoning
        if tool_calls:
            choice["message"]["tool_calls"] = tool_calls

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": len(generated_tokens),
                "total_tokens": prompt_len + len(generated_tokens),
            },
        }

    # ---------------- Streaming SSE Response ----------------
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def sse_generator():
        in_thinking = False
        for new_text in streamer:
            if not new_text:
                continue

            delta_dict = {}
            if "<think>" in new_text:
                in_thinking = True
                new_text = new_text.replace("<think>", "")
            if "</think>" in new_text:
                in_thinking = False
                parts = new_text.split("</think>")
                if parts[0]:
                    chunk = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [{"index": 0, "delta": {"reasoning_content": parts[0]}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                new_text = parts[1] if len(parts) > 1 else ""

            if in_thinking:
                delta_dict["reasoning_content"] = new_text
            else:
                if new_text and new_text != "</s>":
                    delta_dict["content"] = new_text

            if delta_dict:
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": delta_dict, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Final termination chunk
        final_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


def start_server():
    global model, tokenizer, server_device
    parser = argparse.ArgumentParser(description="ViMind 3.0 OpenAI Compatible API Server")
    parser.add_argument("--model_path", type=str, default="out/dpo/vimind_64m_dpo_final", help="Path to model checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, default="model", help="Path to tokenizer directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    server_device = torch.device(args.device)

    print("=" * 65)
    print("🚀 VIMIND 3.0: OPENAI API COMPATIBLE SERVER")
    print("=" * 65)
    print(f"📦 Model: {args.model_path}")
    print(f"📦 Tokenizer: {args.tokenizer_dir}")
    print(f"🌐 Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"💡 Tương thích trực tiếp với: Open-WebUI, Cherry Studio, NextChat, Dify!")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if os.path.exists(args.model_path):
        model = ViMindForCausalLM.from_pretrained(args.model_path).to(server_device)
    else:
        print(f"⚠️ Checkpoint {args.model_path} chưa có sẵn. Khởi tạo ViMind 64M testing preset...")
        cfg = ViMindConfig.get_config_64m(vocab_size=len(tokenizer))
        model = ViMindForCausalLM(cfg).to(server_device)

    model.eval()
    print(f"✅ ViMind 3.0 đã sẵn sàng phục vụ trên {server_device}!")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    start_server()
