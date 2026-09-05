import os
import sys
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
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from transformers import AutoTokenizer
from model.model import ViMindConfig, ViMindForCausalLM

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ViMind: Trợ Lý Tiếng Việt",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphism)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff4b4b, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
        margin-right: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_CKPT = "out/dpo/vimind_dpo_final" if os.path.exists("out/dpo/vimind_dpo_final") else "out/sft/vimind_sft_final"


@st.cache_resource(show_spinner=False)
def load_vimind_model(model_path: str = DEFAULT_CKPT):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer_dir = model_path if os.path.exists(os.path.join(model_path, "tokenizer_config.json")) else "model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    if os.path.exists(model_path):
        model = ViMindForCausalLM.from_pretrained(model_path).to(device)
    else:
        # Fallback to fresh/untrained model for demonstration if training hasn't finished yet
        config = ViMindConfig(vocab_size=len(tokenizer))
        model = ViMindForCausalLM(config).to(device)

    model.eval()
    return model, tokenizer, device


# Sidebar Controls
with st.sidebar:
    st.markdown("### ⚙️ Cấu Hình Mô Hình")
    model_path_input = st.text_input(
        "Đường dẫn Checkpoint:",
        value=DEFAULT_CKPT,
        help="Đường dẫn đến thư mục chứa model sau khi huấn luyện (DPO hoặc SFT).",
    )

    with st.spinner("Đang nạp mô hình..."):
        model, tokenizer, device = load_vimind_model(model_path_input)

    st.markdown(
        f"""
        <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 15px;'>
            <p style='margin: 0;'><b>Kiến trúc:</b> LLaMA-3 Style</p>
            <p style='margin: 0;'><b>Tham số:</b> ~26.2 Triệu</p>
            <p style='margin: 0;'><b>Từ vựng:</b> 12.800 tokens</p>
            <p style='margin: 0;'><b>Thiết bị:</b> <code>{device}</code></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎛️ Siêu Tham Số Sinh Chữ")
    temperature = st.slider("Temperature (Độ sáng tạo)", 0.1, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-P (Nucleus Sampling)", 0.1, 1.0, 0.9, 0.05)
    max_tokens = st.slider("Max New Tokens", 64, 1024, 384, 32)

    st.divider()
    if st.button("🧹 Xóa Lịch Sử Hội Thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface Header
st.markdown("<div class='main-title'>🇻🇳 ViMind Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Mô hình ngôn ngữ tiếng Việt siêu nhẹ ~26M tham số được huấn luyện từ đầu.</div>", unsafe_allow_html=True)

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Hỏi ViMind bất cứ điều gì..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Format history with chat template
    history_for_model = [
        {"role": "system", "content": "Bạn là ViMind, một trợ lý AI thông minh và hữu ích được phát triển riêng cho tiếng Việt."}
    ] + st.session_state.messages

    prompt_text = tokenizer.apply_chat_template(
        history_for_model,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

    # Generate Response Stream
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1] :]
        full_response = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True).strip()
        response_placeholder.markdown(full_response)

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
