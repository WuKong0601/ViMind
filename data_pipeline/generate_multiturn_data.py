"""
ViMind 4.5 - Multi-turn Dialogue & Reasoning Core Dataset Generator.
Solves Attention Anchoring & Conversational Drift by synthesizing multi-turn dialogs:
  - Turn 1: Factual Vietnam Geography / History / Science
  - Turn 2: Topic Switch to CoT Arithmetic / Algebra
  - Turn 3: Real-time Tool Calling (<tool_call>)
  - Turn 4: Follow-up Confirmation / Synthesis
Outputs:
  - dataset/multiturn_core_vi.jsonl
  - dataset/vimind_4.5_combined.jsonl (Single-turn Core + Multi-turn Core)
"""

import os
import sys
import json
import random

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Component Data Pools
# -------------------------------------------------------------

GEOGRAPHY_HISTORY_POOL = [
    {
        "q": "Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là gì?",
        "think": "Hà Nội là thủ đô, trung tâm chính trị, hành chính và văn hóa hàng đầu của Việt Nam.",
        "a": "Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là thành phố **Hà Nội**."
    },
    {
        "q": "Việt Nam có bao nhiêu tỉnh thành?",
        "think": "Đơn vị hành chính cấp tỉnh của Việt Nam gồm 63 tỉnh thành (58 tỉnh và 5 thành phố trực thuộc Trung ương).",
        "a": "Việt Nam hiện có **63 tỉnh, thành phố**, bao gồm 58 tỉnh và 5 thành phố trực thuộc Trung ương (Hà Nội, TP. Hồ Chí Minh, Hải Phòng, Đà Nẵng, Cần Thơ)."
    },
    {
        "q": "Đỉnh núi nào cao nhất Việt Nam và được mệnh danh là nóc nhà Đông Dương?",
        "think": "Fansipan thuộc dãy Hoàng Liên Sơn, độ cao 3.143m.",
        "a": "Đỉnh núi cao nhất Việt Nam là **Fansipan** (3.143m), nằm trên dãy Hoàng Liên Sơn, được mệnh danh là 'Nóc nhà Đông Dương'."
    },
    {
        "q": "Chủ tịch Hồ Chí Minh đọc bản Tuyên ngôn Độc lập vào ngày nào?",
        "think": "Ngày Quốc khánh 2/9/1945 tại Quảng trường Ba Đình, Hà Nội.",
        "a": "Chủ tịch Hồ Chí Minh đã đọc bản Tuyên ngôn Độc lập vào ngày **2 tháng 9 năm 1945** tại Quảng trường Ba Đình, Hà Nội, khai sinh ra nước Việt Nam Dân chủ Cộng hòa."
    },
    {
        "q": "Chiến thắng Điện Biên Phủ diễn ra vào năm nào và có ý nghĩa lịch sử gì?",
        "think": "Điện Biên Phủ năm 1954, lừng lẫy năm châu chấn động địa cầu.",
        "a": "Chiến thắng Điện Biên Phủ diễn ra vào ngày **7 tháng 5 năm 1954**, kết thúc 9 năm kháng chiến chống thực dân Pháp, ghi dấu mốc son 'lừng lẫy năm châu, chấn động địa cầu'."
    },
    {
        "q": "Thành phố Đà Nẵng nằm ở vùng miền nào của Việt Nam?",
        "think": "Đà Nẵng là thành phố trực thuộc Trung ương thuộc vùng Duyên hải Nam Trung Bộ.",
        "a": "Thành phố Đà Nẵng nằm ở khu vực **miền Trung** (vùng Duyên hải Nam Trung Bộ) của Việt Nam."
    },
    {
        "q": "Hai con sông lớn nhất bồi đắp nên hai đồng bằng trù phú của Việt Nam là gì?",
        "think": "Sông Hồng bồi đắp ĐB Bắc Bộ, sông Cửu Long (Mê Kông) bồi đắp ĐB Nam Bộ.",
        "a": "Hai con sông lớn nhất là **sông Hồng** (bồi đắp Đồng bằng sông Hồng) và **sông Cửu Long** (hệ thống sông Mê Kông bồi đắp Đồng bằng sông Cửu Long)."
    },
    {
        "q": "Vịnh Hạ Long thuộc tỉnh nào và đã được UNESCO công nhận điều gì?",
        "think": "Vịnh Hạ Long thuộc tỉnh Quảng Ninh, là di sản thiên nhiên thế giới UNESCO.",
        "a": "Vịnh Hạ Long thuộc tỉnh **Quảng Ninh**, được UNESCO nhiều lần công nhận là Di sản Thiên nhiên Thế giới nhờ giá trị cảnh quan và địa chất địa mạo đặc sắc."
    }
]

SCIENCE_POOL = [
    {
        "q": "Tại sao có hiện tượng ngày và đêm luân phiên nhau?",
        "think": "Do Trái Đất có hình cầu và tự quay quanh trục từ Tây sang Đông trong 24 giờ.",
        "a": "Hiện tượng ngày và đêm luân phiên xảy ra do **Trái Đất có dạng hình cầu và tự quay quanh trục** của chính nó theo hướng từ Tây sang Đông, khiến một nửa được Mặt Trời chiếu sáng (ngày) và một nửa nằm trong bóng tối (đêm)."
    },
    {
        "q": "Mưa được hình thành như thế nào?",
        "think": "Vòng tuần hoàn nước: Bốc hơi -> Ngưng tụ thành mây -> Hạt nước nặng rơi xuống do trọng lực.",
        "a": "Mưa được hình thành qua **vòng tuần hoàn của nước**: Nước từ sông hồ, đại dương bốc hơi lên cao $\\rightarrow$ gặp không khí lạnh ngưng tụ thành những hạt nước li ti tạo thành mây $\\rightarrow$ khi hạt nước đủ lớn và nặng, chúng rơi xuống tạo thành mưa."
    },
    {
        "q": "Nhật thực là gì và xảy ra khi nào?",
        "think": "Mặt Trăng đi vào giữa Trái Đất và Mặt Trời trên một đường thẳng, che khuất ánh sáng Mặt Trời.",
        "a": "**Nhật thực** là hiện tượng thiên văn xảy ra khi **Mặt Trăng đi vào giữa Trái Đất và Mặt Trời** trên cùng một đường thẳng, che khuất một phần hoặc toàn bộ ánh sáng Mặt Trời chiếu tới Trái Đất."
    },
    {
        "q": "Tại sao lá cây lại có màu xanh?",
        "think": "Do chất diệp lục (chlorophyll) hấp thụ ánh sáng đỏ và xanh lam, phản xạ ánh sáng xanh lục.",
        "a": "Lá cây có màu xanh do chứa chất **diệp lục (chlorophyll)**. Diệp lục hấp thụ các dải ánh sáng xanh dương và đỏ để quang hợp, đồng thời phản xạ dải ánh sáng màu lục vào mắt chúng ta."
    }
]

def make_math_item(a=None, b=None, c=None, d=None):
    if a is None:
        a = random.randint(12, 45)
        b = random.randint(8, 25)
        d = random.choice([2, 4, 5, 10])
        c = d * random.randint(15, 150)
    
    p1 = a * b
    p2 = c // d
    res = p1 + p2
    
    questions = [
        f"Tính giúp tôi kết quả của {a} * {b} + {c} / {d}.",
        f"Bây giờ hãy giải bài toán này: {a} * {b} + {c} / {d} bằng bao nhiêu?",
        f"Thực hiện phép tính sau giúp tôi: {a} * {b} + {c} / {d}.",
        f"Cho biểu thức {a} * {b} + {c} / {d}, hãy tính giá trị chính xác."
    ]
    q = random.choice(questions)
    
    thought = (
        f"Để giải bài toán này, ta thực hiện các phép nhân chia trước, cộng trừ sau:\n"
        f"- Bước 1: Tính phép nhân {a} * {b} = {p1}.\n"
        f"- Bước 2: Tính phép chia {c} / {d} = {p2}.\n"
        f"- Bước 3: Cộng hai kết quả lại: {p1} + {p2} = {res}."
    )
    ans = f"Kết quả của phép tính {a} * {b} + {c} / {d} là **{res}**."
    return {"q": q, "think": thought, "a": ans}

def make_anchor_math():
    return {
        "q": "Tính giúp tôi kết quả của 25 * 18 + 750 / 5.",
        "think": (
            "Để giải bài toán này, ta thực hiện các phép nhân chia trước, cộng trừ sau:\n"
            "- Bước 1: Tính phép nhân 25 * 18 = 450.\n"
            "- Bước 2: Tính phép chia 750 / 5 = 150.\n"
            "- Bước 3: Cộng hai kết quả lại: 450 + 150 = 600."
        ),
        "a": "Kết quả của phép tính 25 * 18 + 750 / 5 là **600**."
    }

TOOL_WEATHER_LOCATIONS = ["Đà Nẵng", "Hà Nội", "TP. Hồ Chí Minh", "Hải Phòng", "Cần Thơ", "Nha Trang", "Đà Lạt", "Huế"]

def make_tool_weather_item(loc=None):
    if loc is None:
        loc = random.choice(TOOL_WEATHER_LOCATIONS)
    prompts = [
        f"Thời tiết hiện tại ở {loc} thế nào?",
        f"Tra cứu giúp tôi tình hình thời tiết hôm nay tại {loc}.",
        f"Hôm nay {loc} có mưa hay nắng không, kiểm tra thời tiết giúp tôi.",
        f"Cho tôi biết nhiệt độ và thời tiết ở {loc} lúc này."
    ]
    q = random.choice(prompts)
    thought = f"Người dùng muốn tra cứu thời tiết thời gian thực ở {loc}. Cần gọi công cụ get_current_weather."
    tool_call = f'<tool_call>{{"name": "get_current_weather", "arguments": {{"location": "{loc}"}}}}</tool_call>'
    return {"q": q, "think": thought, "tool_call": tool_call}

def make_tool_time_item():
    prompts = [
        "Bây giờ là mấy giờ rồi?",
        "Cho tôi biết thời gian hiện tại chính xác.",
        "Hôm nay là ngày mấy, mấy giờ?",
        "Tra cứu thời gian thực hệ thống giúp tôi."
    ]
    q = random.choice(prompts)
    thought = "Người dùng yêu cầu tra cứu thời gian thực hiện tại. Cần gọi công cụ get_current_time."
    tool_call = '<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>'
    return {"q": q, "think": thought, "tool_call": tool_call}

TRANSITION_PHRASES = [
    "Cảm ơn bạn. Bây giờ hãy ",
    "Tốt lắm. Tiếp theo, ",
    "Câu vừa rồi rất rõ ràng. Cho tôi hỏi tiếp: ",
    "Được rồi. Bây giờ hãy giúp tôi ",
    "Bây giờ chuyển sang chủ đề khác nhé: ",
    "Tiếp tục nào, "
]

# -------------------------------------------------------------
# 2. Multi-turn Conversation Synthesizer
# -------------------------------------------------------------

def generate_multi_turn_samples(count=12000):
    samples = []
    
    # 2.1 Anchor Session: Geography -> Anchor Math (25 * 18 + 750 / 5) -> Weather Da Nang
    for _ in range(50):
        geo = random.choice(GEOGRAPHY_HISTORY_POOL)
        math_item = make_anchor_math()
        weather_item = make_tool_weather_item("Đà Nẵng")
        
        session = [
            {"role": "user", "content": geo["q"]},
            {"role": "assistant", "content": f"<think>\n{geo['think']}\n</think>\n{geo['a']}"},
            {"role": "user", "content": "Cảm ơn bạn. Bây giờ hãy " + math_item["q"].lower()},
            {"role": "assistant", "content": f"<think>\n{math_item['think']}\n</think>\n{math_item['a']}"},
            {"role": "user", "content": "Rất chuẩn xác! Còn " + weather_item["q"].lower()},
            {"role": "assistant", "content": f"<think>\n{weather_item['think']}\n</think>\n{weather_item['tool_call']}"}
        ]
        samples.append({"conversations": session})

    # 2.2 Standard 2-turn and 3-turn switching sessions
    while len(samples) < count:
        session_type = random.choice(["geo_to_math", "math_to_geo", "geo_to_tool", "science_to_math_to_tool"])
        
        if session_type == "geo_to_math":
            t1 = random.choice(GEOGRAPHY_HISTORY_POOL)
            t2 = make_math_item()
            trans = random.choice(TRANSITION_PHRASES)
            
            session = [
                {"role": "user", "content": t1["q"]},
                {"role": "assistant", "content": f"<think>\n{t1['think']}\n</think>\n{t1['a']}"},
                {"role": "user", "content": trans + t2["q"].lower()},
                {"role": "assistant", "content": f"<think>\n{t2['think']}\n</think>\n{t2['a']}"}
            ]
            samples.append({"conversations": session})

        elif session_type == "math_to_geo":
            t1 = make_math_item()
            t2 = random.choice(GEOGRAPHY_HISTORY_POOL)
            trans = random.choice(TRANSITION_PHRASES)
            
            session = [
                {"role": "user", "content": t1["q"]},
                {"role": "assistant", "content": f"<think>\n{t1['think']}\n</think>\n{t1['a']}"},
                {"role": "user", "content": trans + t2["q"].lower()},
                {"role": "assistant", "content": f"<think>\n{t2['think']}\n</think>\n{t2['a']}"}
            ]
            samples.append({"conversations": session})

        elif session_type == "geo_to_tool":
            t1 = random.choice(GEOGRAPHY_HISTORY_POOL)
            t2 = make_tool_weather_item() if random.random() > 0.3 else make_tool_time_item()
            trans = random.choice(TRANSITION_PHRASES)
            
            session = [
                {"role": "user", "content": t1["q"]},
                {"role": "assistant", "content": f"<think>\n{t1['think']}\n</think>\n{t1['a']}"},
                {"role": "user", "content": trans + t2["q"].lower()},
                {"role": "assistant", "content": f"<think>\n{t2['think']}\n</think>\n{t2['tool_call']}"}
            ]
            samples.append({"conversations": session})

        elif session_type == "science_to_math_to_tool":
            t1 = random.choice(SCIENCE_POOL)
            t2 = make_math_item()
            t3 = make_tool_weather_item()
            
            session = [
                {"role": "user", "content": t1["q"]},
                {"role": "assistant", "content": f"<think>\n{t1['think']}\n</think>\n{t1['a']}"},
                {"role": "user", "content": "Cảm ơn bạn. Tiếp theo, " + t2["q"].lower()},
                {"role": "assistant", "content": f"<think>\n{t2['think']}\n</think>\n{t2['a']}"},
                {"role": "user", "content": "Kết quả chuẩn. Tiện thể, " + t3["q"].lower()},
                {"role": "assistant", "content": f"<think>\n{t3['think']}\n</think>\n{t3['tool_call']}"}
            ]
            samples.append({"conversations": session})

    random.shuffle(samples)
    return samples

# -------------------------------------------------------------
# 3. Main Synthesis & Merging
# -------------------------------------------------------------

def main():
    print("🚀 [ViMind 4.5] Synthesizing Multi-turn Dialogue Dataset...")
    multi_samples = generate_multi_turn_samples(count=12000)
    
    multiturn_out = os.path.join(DATASET_DIR, "multiturn_core_vi.jsonl")
    with open(multiturn_out, "w", encoding="utf-8") as f:
        for s in multi_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✅ Generated {len(multi_samples)} multi-turn samples at: {multiturn_out}")

    # Now merge with existing high-density single-turn core
    dense_core_path = os.path.join(DATASET_DIR, "dense_core_vi.jsonl")
    single_samples = []
    if os.path.exists(dense_core_path):
        with open(dense_core_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        single_samples.append(json.loads(line.strip()))
                    except Exception:
                        pass
        print(f"📦 Loaded {len(single_samples)} existing single-turn samples from: {dense_core_path}")

    combined_samples = single_samples + multi_samples
    random.shuffle(combined_samples)
    
    combined_out = os.path.join(DATASET_DIR, "vimind_4.5_combined.jsonl")
    with open(combined_out, "w", encoding="utf-8") as f:
        for s in combined_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"🎉 Total Combined ViMind 4.5 Training Samples: {len(combined_samples)} at {combined_out}")

if __name__ == "__main__":
    main()
