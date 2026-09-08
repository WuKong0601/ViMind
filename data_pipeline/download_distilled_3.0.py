import os
import sys
import json
import argparse

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


VIETNAM_FACTS_DATA = [
    {
        "conversations": [
            {"role": "user", "content": "Xin chào! Bạn là ai?"},
            {"role": "assistant", "content": "Xin chào! Tôi là ViMind (phiên bản 3.0), mô hình ngôn ngữ lớn tiếng Việt siêu nhẹ được phát triển hoàn toàn từ đầu. Tôi có thể hỗ trợ bạn giải đáp thắc mắc, tóm tắt thông tin, viết mã lập trình, suy luận logic và thực hiện các tác vụ hữu ích khác."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Thủ đô của Việt Nam là gì?"},
            {"role": "assistant", "content": "Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là thành phố Hà Nội. Đây là trung tâm chính trị, văn hóa và giáo dục quan trọng hàng đầu của đất nước với lịch sử nghìn năm văn hiến."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Việt Nam có bao nhiêu tỉnh và thành phố trực thuộc Trung ương?"},
            {"role": "assistant", "content": "Việt Nam có tổng cộng 63 đơn vị hành chính cấp tỉnh, bao gồm 58 tỉnh và 5 thành phố trực thuộc Trung ương (gồm: Hà Nội, Thành phố Hồ Chí Minh, Hải Phòng, Đà Nẵng và Cần Thơ)."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Đỉnh núi cao nhất Việt Nam là đỉnh núi nào?"},
            {"role": "assistant", "content": "Đỉnh núi cao nhất Việt Nam là đỉnh Fansipan (Phan Xi Păng) với độ cao 3.143 mét so với mực nước biển, thuộc dãy núi Hoàng Liên Sơn, nằm tại tỉnh Lào Cai. Fansipan còn được mệnh danh là 'Nóc nhà Đông Dương'."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Chiến thắng Điện Biên Phủ diễn ra vào năm nào?"},
            {"role": "assistant", "content": "Chiến thắng Điện Biên Phủ lịch sử diễn ra vào ngày 7 tháng 5 năm 1954 dưới sự lãnh đạo của Đại tướng Võ Nguyên Giáp, kết thúc thắng lợi cuộc kháng chiến chống thực dân Pháp của quân và dân Việt Nam."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Ai là tác giả của bài Quốc ca Việt Nam?"},
            {"role": "assistant", "content": "Tác giả của bài Tiến quân ca (Quốc ca của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam) là cố nhạc sĩ Văn Cao, được sáng tác vào năm 1944."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Sông Mê Kông chảy qua những quốc gia nào trước khi đổ ra Biển Đông tại Việt Nam?"},
            {"role": "assistant", "content": "Sông Mê Kông bắt nguồn từ cao nguyên Tây Tạng (Trung Quốc), chảy qua 6 quốc gia bao gồm: Trung Quốc, Myanmar, Lào, Thái Lan, Campuchia và cuối cùng đổ ra Biển Đông qua hệ thống sông Cửu Long tại miền Nam Việt Nam."}
        ]
    }
]

REASONING_COT_DATA = [
    {
        "conversations": [
            {"role": "user", "content": "Một người nông dân có 17 con cừu. Trừ 9 con ra, tất cả đều chết hết. Hỏi người nông dân còn lại bao nhiêu con cừu sống?"},
            {"role": "assistant", "content": "<think>\n1. Phân tích câu hỏi: 'Trừ 9 con ra, tất cả đều chết hết'.\n2. 'Trừ 9 con ra' nghĩa là 9 con cừu này không chết.\n3. Các con cừu còn lại (17 - 9 = 8 con) đã chết.\n4. Do đó, số con cừu còn sống chính xác là 9 con.\n</think>\n\nNgười nông dân còn lại chính xác 9 con cừu sống (vì câu hỏi đã nêu rõ 'trừ 9 con ra, tất cả đều chết hết')."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Tìm số tiếp theo trong dãy số: 2, 4, 8, 16, 32, ...?"},
            {"role": "assistant", "content": "<think>\n1. Quan sát quy luật giữa các số hạng:\n   - 4 = 2 * 2\n   - 8 = 4 * 2\n   - 16 = 8 * 2\n   - 32 = 16 * 2\n2. Đây là một cấp số nhân với công bội q = 2.\n3. Số tiếp theo sẽ là 32 * 2 = 64.\n</think>\n\nSố tiếp theo trong dãy số là 64 (quy luật: mỗi số hạng sau gấp đôi số hạng đứng liền trước nó)."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Nếu 3 người thợ xây một bức tường mất 3 ngày, thì 1 người thợ xây bức tường đó mất bao nhiêu ngày (giả định năng suất như nhau)?"},
            {"role": "assistant", "content": "<think>\n1. Tính khối lượng công việc theo đơn vị công-ngày:\n   - 3 người * 3 ngày = 9 công-ngày để hoàn thành bức tường.\n2. Nếu chỉ có 1 người làm việc:\n   - Thời gian cần thiết = 9 công-ngày / 1 người = 9 ngày.\n</think>\n\nMột người thợ sẽ mất 9 ngày để hoàn thành bức tường đó."}
        ]
    }
]

TOOLCALL_DATA = [
    {
        "conversations": [
            {"role": "user", "content": "Tính giúp tôi biểu thức: 125 * 8 + 450 / 9"},
            {"role": "assistant", "content": "<tool_call>\n{\"name\": \"calculate_math\", \"arguments\": {\"expression\": \"125 * 8 + 450 / 9\"}}\n</tool_call>"},
            {"role": "tool", "content": "{\"result\": \"1050.0\"}"},
            {"role": "assistant", "content": "Kết quả của biểu thức `125 * 8 + 450 / 9` là **1.050**."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Thời tiết hiện tại ở Hà Nội như thế nào?"},
            {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_current_weather\", \"arguments\": {\"location\": \"Hà Nội\"}}\n</tool_call>"},
            {"role": "tool", "content": "{\"city\": \"Hà Nội\", \"temperature\": \"26°C\", \"humidity\": \"70%\", \"condition\": \"Mát mẻ, nắng nhẹ\"}"},
            {"role": "assistant", "content": "Thời tiết hiện tại ở **Hà Nội** khá dễ chịu: nhiệt độ khoảng **26°C**, độ ẩm 70%, trời mát mẻ và có nắng nhẹ."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "Đổi 50 dặm (miles) sang ki-lô-mét (km) giúp tôi."},
            {"role": "assistant", "content": "<tool_call>\n{\"name\": \"unit_converter\", \"arguments\": {\"value\": 50, \"from_unit\": \"miles\", \"to_unit\": \"km\"}}\n</tool_call>"},
            {"role": "tool", "content": "{\"result\": 80.467}"},
            {"role": "assistant", "content": "50 dặm (miles) tương đương với khoảng **80.47 km** (chính xác là 80.467 km)."}
        ]
    }
]


def build_v3_datasets(output_dir="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 65)
    print("📚 VIMIND 3.0: XÂY DỰNG DATASET CHƯNG CẤT CHỐNG ẢO GIÁC")
    print("=" * 65)

    # 1. Facts dataset
    facts_file = os.path.join(output_dir, "identity_and_vietnam_facts.jsonl")
    with open(facts_file, "w", encoding="utf-8") as f:
        for item in VIETNAM_FACTS_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Đã tạo tập tri thức chuẩn Việt Nam: {facts_file} ({len(VIETNAM_FACTS_DATA)} mẫu)")

    # 2. Reasoning CoT dataset
    reasoning_file = os.path.join(output_dir, "sft_reasoning_vi.jsonl")
    with open(reasoning_file, "w", encoding="utf-8") as f:
        for item in REASONING_COT_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Đã tạo tập suy luận CoT (<think>): {reasoning_file} ({len(REASONING_COT_DATA)} mẫu)")

    # 3. Toolcall dataset
    toolcall_file = os.path.join(output_dir, "sft_toolcall_vi.jsonl")
    with open(toolcall_file, "w", encoding="utf-8") as f:
        for item in TOOLCALL_DATA:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Đã tạo tập công cụ Function Calling (<tool_call>): {toolcall_file} ({len(TOOLCALL_DATA)} mẫu)")

    # 4. Master SFT v3 dataset: combines existing SFT with the facts, reasoning, and toolcall data
    sft_v3_file = os.path.join(output_dir, "sft_vi_v3.jsonl")
    total_samples = 0
    with open(sft_v3_file, "w", encoding="utf-8") as f_out:
        # High-priority seeds first
        for item in VIETNAM_FACTS_DATA * 5:  # Oversample facts to guarantee factual anchoring
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_samples += 1

        for item in REASONING_COT_DATA * 3:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_samples += 1

        for item in TOOLCALL_DATA * 3:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_samples += 1

        # Existing SFT if available, or automatically download base Alpaca SFT
        existing_sft = os.path.join(output_dir, "sft_vi.jsonl")
        if not os.path.exists(existing_sft):
            print("⚡ Chưa tìm thấy dataset/sft_vi.jsonl. Tự động tải 5CD-AI/Vietnamese-alpaca-gpt4-gg-translated...")
            try:
                from download_sft import main as run_download_sft
                run_download_sft()
            except Exception as e:
                print(f"⚠️ Lỗi khi tải dataset Alpaca ({e}). Thử gọi qua datasets thư viện trực tiếp...")
                try:
                    from datasets import load_dataset
                    ds = load_dataset("5CD-AI/Vietnamese-alpaca-gpt4-gg-translated", split="train")
                    with open(existing_sft, "w", encoding="utf-8") as f_sft:
                        for item in ds:
                            ins = item.get("instruction_vi", "")
                            inp = item.get("input_vi", "")
                            out = item.get("output_vi", "")
                            p = f"{ins}\n\n{inp}" if inp else ins
                            if p and out:
                                f_sft.write(json.dumps({"conversations": [{"role": "user", "content": p}, {"role": "assistant", "content": out}]}, ensure_ascii=False) + "\n")
                except Exception as ex:
                    print(f"⚠️ Không thể tải Alpaca tự động: {ex}")

        if os.path.exists(existing_sft):
            with open(existing_sft, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if line:
                        f_out.write(line + "\n")
                        total_samples += 1

    print(f"🎉 Đã xây dựng hoàn tất tập SFT 3.0 tổng hợp: {sft_v3_file} ({total_samples:,} mẫu)!")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViMind 3.0 Distilled Dataset Builder")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory")
    args = parser.parse_args()
    build_v3_datasets(args.output_dir)
