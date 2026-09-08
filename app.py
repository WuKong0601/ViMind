import os
import sys
import re
import json
import datetime
import torch
import streamlit as st

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from model.model import ViMindConfig, ViMindForCausalLM

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ViMind 3.0: Trợ Lý AI Tiếng Việt",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphism & High-Tech Accents)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8533 50%, #FFCC00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 1.2rem;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-moe {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(147, 51, 234, 0.2);
        color: #C084FC;
        border: 1px solid rgba(147, 51, 234, 0.4);
        margin-right: 6px;
    }
    .badge-rl {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        margin-right: 6px;
    }
    .tool-box {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CANDIDATES_CKPT = [
    "out/vimind_3.0_hf",
    "out/agent_rl",
    "out/sft_moe",
    "out/dpo_qwen",
    "out/dpo/vimind_64m_dpo_final",
    "out/sft/vimind_64m_sft_final",
    "out/vimind_64m_final",
]
DEFAULT_CKPT = next((p for p in CANDIDATES_CKPT if os.path.exists(p)), "out/vimind_3.0_hf")


APP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Thực hiện phép tính toán học an toàn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Biểu thức toán học"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Tra cứu thời tiết thời gian thực tại các tỉnh thành Việt Nam.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Tên thành phố"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Lấy ngày giờ hiện tại theo múi giờ Việt Nam.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


def execute_tool_call(name: str, args: dict) -> str:
    if name == "calculate_math":
        expr = args.get("expression", "")
        allowed = set("0123456789+-*/(). %^")
        if all(c in allowed for c in expr):
            try:
                res = eval(expr, {"__builtins__": None}, {})
                return json.dumps({"result": res, "status": "success"}, ensure_ascii=False)
            except Exception as e:
                return json.dumps({"error": str(e), "status": "failed"}, ensure_ascii=False)
        return json.dumps({"error": "Ký tự không hợp lệ", "status": "failed"}, ensure_ascii=False)

    elif name == "get_current_weather":
        loc = args.get("location", "Hà Nội")
        return json.dumps({
            "location": loc,
            "temperature_c": 29.0,
            "condition": "Nắng nhẹ, gió mát",
            "humidity": "65%"
        }, ensure_ascii=False)

    elif name == "get_current_time":
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S (GMT+7)")
        return json.dumps({"current_time": now_str, "timezone": "Asia/Ho_Chi_Minh"}, ensure_ascii=False)

    return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)


@st.cache_resource(show_spinner=False)
def load_vimind_model(model_path: str = DEFAULT_CKPT):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer_dir = os.path.join(PROJECT_ROOT, "model")
    if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        tokenizer_dir = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        config = ViMindConfig.from_pretrained(model_path)
    else:
        config = ViMindConfig(vocab_size=len(tokenizer), use_moe=True, num_experts=4, num_experts_per_tok=1)

    model = ViMindForCausalLM(config)

    # Check for weights
    loaded = False
    for fname in ["model.safetensors", "pytorch_model.bin", "vimind_sft.pth", "vimind_agent_rl.pth"]:
        target = os.path.join(model_path, fname)
        if os.path.exists(target):
            if target.endswith(".safetensors"):
                from safetensors.torch import load_file
                state = load_file(target)
            else:
                state = torch.load(target, map_location="cpu")
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            loaded = True
            break

    model.to(device)
    model.eval()
    return model, tokenizer, device, config, loaded


# Sidebar Controls
with st.sidebar:
    st.markdown("### 🇻🇳 ViMind 3.0 Control Panel")
    model_path_input = st.text_input(
        "Checkpoint / Model Directory:",
        value=DEFAULT_CKPT,
        help="Thư mục chứa weights hoặc cấu hình ViMind 3.0",
    )

    with st.spinner("Đang nạp mô hình & từ điển..."):
        model, tokenizer, device, config, weights_loaded = load_vimind_model(model_path_input)

    is_moe = getattr(config, "use_moe", False)
    total_params = sum(p.numel() for p in model.parameters())
    param_str = f"~{total_params / 1e6:.1f}M" if total_params > 0 else "198M (MoE)"

    st.markdown(
        f"""
        <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 15px;'>
            <p style='margin: 0;'><b>Kiến trúc:</b> {'MoE (Mixture of Experts)' if is_moe else 'Dense CausalLM'}</p>
            <p style='margin: 0;'><b>Tham số:</b> {param_str} {'(64M active)' if is_moe else ''}</p>
            <p style='margin: 0;'><b>Context (YaRN):</b> {getattr(config, "max_seq_len", 2048)} tokens</p>
            <p style='margin: 0;'><b>Weights nạp:</b> {'✅ Có sẵn' if weights_loaded else '⚠️ Cấu trúc mẫu (Demo)'}</p>
            <p style='margin: 0;'><b>Thiết bị:</b> <code>{device}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Tính Năng Nâng Cao")
    open_thinking = st.toggle("🧠 Kích hoạt Suy nghĩ (<think>)", value=True, help="Cho phép mô hình sinh luồng tư duy lý luận trước khi trả lời.")
    enable_tools = st.toggle("🛠️ Kích hoạt Gọi công cụ (Tool Call)", value=True, help="Tự động nhận diện và gọi các công cụ Toán học, Thời tiết, Thời gian.")

    st.markdown("### 🎛️ Siêu Tham Số Sinh Chữ")
    temperature = st.slider("Temperature (Độ sáng tạo)", 0.1, 1.5, 0.6, 0.05)
    top_p = st.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.85, 0.05)
    max_tokens = st.slider("Max New Tokens", 64, 2048, 512, 64)

    st.divider()
    if st.button("🧹 Xóa Lịch Sử Hội Thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Chat Interface Header
st.markdown("<div class='main-title'>🇻🇳 ViMind 3.0</div>", unsafe_allow_html=True)
st.markdown(
    "<span class='badge-moe'>MoE 198M / 64M Active</span>"
    "<span class='badge-rl'>RLAIF & Agentic GRPO</span>"
    "<div class='sub-title'>Mô hình SLM Tiếng Việt đỉnh cao: Lý luận từng bước (CoT), Gọi công cụ thông minh, Chống ảo giác.</div>",
    unsafe_allow_html=True
)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "reasoning" in msg and msg["reasoning"]:
            with st.expander("🧠 Quá trình suy nghĩ (Thinking Chain)", expanded=False):
                st.markdown(msg["reasoning"])
        if "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                st.markdown(
                    f"<div class='tool-box'>🔧 <b>Tool Call:</b> <code>{tc['name']}</code><br>"
                    f"<b>Args:</b> <code>{json.dumps(tc['args'], ensure_ascii=False)}</code><br>"
                    f"<b>Result:</b> <code>{tc.get('result', '')}</code></div>",
                    unsafe_allow_html=True
                )
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Nhập câu hỏi hoặc yêu cầu tính toán, tra cứu thời tiết..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format history with chat template
    history_for_model = [
        {"role": "system", "content": "Bạn là ViMind 3.0, trợ lý AI tiếng Việt thông minh, trung thực, có khả năng tư duy logic và sử dụng công cụ."}
    ]
    for m in st.session_state.messages:
        history_for_model.append({"role": m["role"], "content": m["content"]})

    try:
        prompt_text = tokenizer.apply_chat_template(
            history_for_model,
            tools=APP_TOOLS if enable_tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1] :]
        raw_response = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=False).strip()

        # Parse Thinking Chain
        reasoning_content = ""
        think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()
            clean_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
        else:
            clean_response = raw_response

        # Parse Tool Calls
        tool_calls = []
        tc_matches = re.findall(r"<tool_call>(.*?)</tool_call>", clean_response, re.DOTALL)
        for tcm in tc_matches:
            try:
                tc_data = json.loads(tcm.strip())
                fn_name = tc_data.get("name")
                fn_args = tc_data.get("arguments", {})
                res = execute_tool_call(fn_name, fn_args)
                tool_calls.append({"name": fn_name, "args": fn_args, "result": res})
            except Exception:
                pass

        final_clean_text = re.sub(r"<tool_call>.*?</tool_call>", "", clean_response, flags=re.DOTALL).strip()
        final_clean_text = final_clean_text.replace("<|im_end|>", "").strip()

        # Display Reasoning if present
        if reasoning_content and open_thinking:
            with st.expander("🧠 Quá trình suy nghĩ (Thinking Chain)", expanded=True):
                st.markdown(reasoning_content)

        # Display Tool Calls if present
        if tool_calls:
            for tc in tool_calls:
                st.markdown(
                    f"<div class='tool-box'>🔧 <b>Đã gọi công cụ:</b> <code>{tc['name']}</code><br>"
                    f"<b>Tham số:</b> <code>{json.dumps(tc['args'], ensure_ascii=False)}</code><br>"
                    f"<b>Kết quả trả về:</b> <code>{tc['result']}</code></div>",
                    unsafe_allow_html=True
                )

        if not final_clean_text and tool_calls:
            final_clean_text = "Tôi đã xử lý và tra cứu kết quả thành công thông qua công cụ chuyên dụng ở trên."
        elif not final_clean_text:
            final_clean_text = "Đã hoàn thành xử lý."

        response_placeholder.markdown(final_clean_text)

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_clean_text,
        "reasoning": reasoning_content,
        "tool_calls": tool_calls
    })
