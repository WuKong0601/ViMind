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

from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import ViMindConfig, ViMindForCausalLM
from safetensors.torch import load_file

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ViMind 4.0: Trợ Lý AI Tiếng Việt",
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
    .badge-pro {
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

AVAILABLE_MODELS = {
    "ViMind 4.0 Native (MoE 198M / 64M active)": "out/vimind_4.0_moe_final",
    "ViMind 4.0 Pro (0.5B Foundation Alignment)": "out/vimind_4.0_pro_final",
}


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
            "temperature_c": 29.5,
            "condition": "Nắng nhẹ, gió mát",
            "humidity": "65%"
        }, ensure_ascii=False)

    elif name == "get_current_time":
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S (GMT+7)")
        return json.dumps({"current_time": now_str, "timezone": "Asia/Ho_Chi_Minh"}, ensure_ascii=False)

    return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_file = os.path.join(model_path, "config.json")
    is_qwen = False
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            if cfg_dict.get("model_type") == "qwen2":
                is_qwen = True
        except Exception:
            pass

    if is_qwen:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        if not torch.cuda.is_available():
            model.to(device)
        config = model.config
        loaded = True
        model_type = "pro"
    else:
        if os.path.exists(config_file):
            config = ViMindConfig.from_pretrained(model_path)
        else:
            config = ViMindConfig(vocab_size=len(tokenizer), use_moe=True, num_experts=4, num_experts_per_tok=2)
        model = ViMindForCausalLM(config)
        sf_path = os.path.join(model_path, "model.safetensors")
        loaded = False
        if os.path.exists(sf_path):
            st_dict = load_file(sf_path)
            model.load_state_dict(st_dict, strict=False)
            loaded = True
        model.to(device)
        model_type = "native"

    model.eval()
    return model, tokenizer, device, config, loaded, model_type


# Sidebar Controls
with st.sidebar:
    st.markdown("### 🇻🇳 ViMind 4.0 Control Panel")
    selected_name = st.selectbox(
        "Lựa chọn phiên bản mô hình:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        help="Chọn mô hình ViMind 4.0 Native (MoE 198M) hoặc ViMind 4.0 Pro (0.5B Foundation)"
    )
    model_path = AVAILABLE_MODELS[selected_name]

    with st.spinner("Đang nạp mô hình & từ điển..."):
        model, tokenizer, device, config, weights_loaded, model_type = load_model(model_path)

    is_moe = getattr(config, "use_moe", False)
    total_params = sum(p.numel() for p in model.parameters())

    st.markdown(
        f"""
        <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 15px;'>
            <p style='margin: 0;'><b>Phiên bản:</b> {selected_name.split('(')[0].strip()}</p>
            <p style='margin: 0;'><b>Kiến trúc:</b> {'MoE (Top-2 Routing)' if is_moe else 'Dense Transformer'}</p>
            <p style='margin: 0;'><b>Tham số:</b> ~{total_params / 1e6:.1f}M {'(64M active)' if is_moe else ''}</p>
            <p style='margin: 0;'><b>Weights nạp:</b> {'✅ Trọng số sẵn sàng' if weights_loaded else '⚠️ Cấu trúc mẫu'}</p>
            <p style='margin: 0;'><b>Thiết bị tính toán:</b> <code>{device}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Tính Năng Nâng Cao")
    open_thinking = st.toggle("🧠 Hiển thị Suy nghĩ (<think>)", value=True, help="Cho phép mở rộng luồng tư duy scratchpad.")

    st.markdown("### 🎛️ Siêu Tham Số Sinh Chữ")
    temperature = st.slider("Temperature (Độ sáng tạo)", 0.05, 1.0, 0.2, 0.05, help="0.2 tối ưu cho toán và sự kiện chính xác.")
    top_p = st.slider("Top-P (Nucleus Sampling)", 0.5, 1.0, 0.9, 0.05)
    repetition_penalty = st.slider("Repetition Penalty (Chống lặp từ)", 1.0, 1.3, 1.08, 0.01, help="1.08 là chuẩn mực loại bỏ lặp từ mà không làm biến dạng câu.")
    max_tokens = st.slider("Max New Tokens", 64, 1024, 256, 32)

    st.divider()
    if st.button("🧹 Xóa Lịch Sử Hội Thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Chat Interface Header
st.markdown("<div class='main-title'>🇻🇳 ViMind 4.0: The Cognitive Sovereign</div>", unsafe_allow_html=True)
st.markdown(
    "<span class='badge-moe'>Dual-Track AI: Native MoE 198M</span>"
    "<span class='badge-pro'>Pro 0.5B Foundation Alignment</span>"
    "<div class='sub-title'>Mô hình SLM Tiếng Việt: Tư duy chuỗi logic (&lt;think&gt;), Gọi công cụ tự động (&lt;tool_call&gt;), Không ảo giác & không lặp từ.</div>",
    unsafe_allow_html=True
)

# Quick Benchmark Prompts
col1, col2, col3 = st.columns(3)
quick_prompt = None
with col1:
    if st.button("🏛️ Thủ đô của Việt Nam là gì?", use_container_width=True):
        quick_prompt = "Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là gì?"
with col2:
    if st.button("🧮 Tính 25 * 18 + 750 / 5", use_container_width=True):
        quick_prompt = "Tính giúp tôi kết quả của 25 * 18 + 750 / 5."
with col3:
    if st.button("🌦️ Thời tiết Đà Nẵng thế nào?", use_container_width=True):
        quick_prompt = "Thời tiết hiện tại ở Đà Nẵng thế nào?"

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
                    f"<b>Tham số:</b> <code>{json.dumps(tc['args'], ensure_ascii=False)}</code><br>"
                    f"<b>Kết quả:</b> <code>{tc.get('result', '')}</code></div>",
                    unsafe_allow_html=True
                )
        st.markdown(msg["content"])

# User Input (either from chat input or quick prompt button)
chat_val = st.chat_input("Nhập câu hỏi, bài toán hoặc yêu cầu tra cứu...")
prompt = quick_prompt if quick_prompt else chat_val

if prompt:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format history with native chat template (Clean, native formatting without out-of-distribution prompts)
    if model_type == "native":
        # Native MoE (64M active) was trained on single-turn task reasoning.
        # Keeping focus on the active user prompt eliminates attention anchoring onto previous turns.
        history_for_model = [{"role": "user", "content": prompt}]
    else:
        # ViMind Pro (0.5B Foundation) has full multi-turn conversational alignment.
        history_for_model = []
        for m in st.session_state.messages:
            history_for_model.append({"role": m["role"], "content": m["content"]})

    try:
        prompt_text = tokenizer.apply_chat_template(
            history_for_model,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    with st.chat_message("assistant"):
        with st.spinner("Đang tư duy suy luận..."):
            with torch.no_grad():
                if model_type == "pro":
                    output_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                else:
                    output_ids = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=0,
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
            final_clean_text = final_clean_text.replace("<|im_end|>", "").replace("</s>", "").strip()

            # Display Reasoning first if present
            if reasoning_content and open_thinking:
                with st.expander("🧠 Quá trình suy nghĩ (Thinking Chain)", expanded=True):
                    st.markdown(reasoning_content)

            # Display Tool Calls if present
            if tool_calls:
                for tc in tool_calls:
                    st.markdown(
                        f"<div class='tool-box'>🔧 <b>Đã kích hoạt công cụ:</b> <code>{tc['name']}</code><br>"
                        f"<b>Tham số:</b> <code>{json.dumps(tc['args'], ensure_ascii=False)}</code><br>"
                        f"<b>Kết quả tra cứu:</b> <code>{tc['result']}</code></div>",
                        unsafe_allow_html=True
                    )

            if not final_clean_text and tool_calls:
                final_clean_text = "Tôi đã phân tích yêu cầu và kích hoạt công cụ tra cứu thời gian thực ở trên."
            elif not final_clean_text:
                final_clean_text = "Đã hoàn tất phản hồi."

            st.markdown(final_clean_text)

    # Save to session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": final_clean_text,
        "reasoning": reasoning_content,
        "tool_calls": tool_calls
    })
