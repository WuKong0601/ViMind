"""
ViMind 4.5 - Community Dataset Ingestion & Master Dataset Compiler.
Integrates gold-standard Vietnamese datasets from HuggingFace Hub:
  1. namfam/gsm8k-vietnamese: 6,700+ Vietnamese GSM8K math word problems with step-by-step CoT
  2. taidng/UIT-ViQuAD2.0: 10,000+ factual reading comprehension & QA pairs
  3. 5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca: 8,000+ realistic multi-turn conversations
Combines with ViMind's 20,805 Anchor samples (tool calling, math CoT, geography) into:
  -> dataset/vimind_4.5_master.jsonl (~45,000 high-quality samples)
"""

import os
import sys
import json
import random
from datasets import load_dataset

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


def process_gsm8k():
    print("📚 [1/3] Ingesting GSM8K Vietnamese (Math CoT)...")
    samples = []
    try:
        ds = load_dataset("namfam/gsm8k-vietnamese", split="train")
        for row in ds:
            prob = row.get("problem", "").strip()
            sol = row.get("solution", "").strip()
            ans = row.get("answer", "").strip()
            if not prob or not sol:
                continue

            # Clean solution text
            thought = sol.replace("####", "Đáp số cuối cùng là:")
            content = f"Đáp số của bài toán là **{ans}**." if ans else sol.split("\n")[-1]

            samples.append({
                "conversations": [
                    {"role": "user", "content": prob},
                    {"role": "assistant", "content": f"<think>\n{thought}\n</think>\n{content}"}
                ]
            })
        print(f"   -> Successfully processed {len(samples)} GSM8K Vietnamese reasoning samples.")
    except Exception as e:
        print(f"   ⚠️ Warning loading GSM8K-Vi: {e}")
    return samples


def process_viquad(max_samples=10000):
    print("🇻🇳 [2/3] Ingesting UIT-ViQuAD 2.0 (Factual QA)...")
    samples = []
    try:
        ds = load_dataset("taidng/UIT-ViQuAD2.0", split="train")
        for row in ds:
            q = row.get("question", "").strip()
            answers = row.get("answers", {}).get("text", [])
            context = row.get("context", "").strip()
            if not q or not answers:
                continue

            ans = answers[0].strip()
            if not ans:
                continue

            # Generate concise thinking chain from context
            ctx_snip = context[:120].replace("\n", " ").strip() + "..."
            thought = f"Dựa vào ngữ cảnh tài liệu: '{ctx_snip}'. Câu trả lời chính xác là: {ans}."

            samples.append({
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": f"<think>\n{thought}\n</think>\n{ans}"}
                ]
            })
            if len(samples) >= max_samples:
                break
        print(f"   -> Successfully processed {len(samples)} UIT-ViQuAD 2.0 factual samples.")
    except Exception as e:
        print(f"   ⚠️ Warning loading UIT-ViQuAD: {e}")
    return samples


def process_multiturn(max_samples=8000):
    print("💬 [3/3] Ingesting Vietnamese Multi-turn Chat...")
    samples = []
    try:
        ds = load_dataset("5CD-AI/Vietnamese-Multi-turn-Chat-Alpaca", split="train")
        for row in ds:
            convs = row.get("conversations", [])
            if not convs or len(convs) < 2:
                continue

            formatted = []
            for m in convs:
                role = m.get("from", "")
                val = m.get("value", "").strip()
                if not val:
                    continue
                if role in ["human", "user"]:
                    formatted.append({"role": "user", "content": val})
                elif role in ["gpt", "assistant"]:
                    # Ensure assistant message has valid formatting
                    if "<think>" not in val:
                        val = f"<think>\nPhân tích yêu cầu và phản hồi bằng tiếng Việt tự nhiên, chính xác.\n</think>\n{val}"
                    formatted.append({"role": "assistant", "content": val})

            if len(formatted) >= 2 and formatted[0]["role"] == "user":
                samples.append({"conversations": formatted})
                if len(samples) >= max_samples:
                    break
        print(f"   -> Successfully processed {len(samples)} Vietnamese multi-turn dialogue samples.")
    except Exception as e:
        print(f"   ⚠️ Warning loading Multi-turn Chat: {e}")
    return samples


def main():
    print("=" * 70)
    print("🚀 VIMIND 4.5: MASTER DATASET COMPILER (COMMUNITY + SYNTHETIC ANCHOR)")
    print("=" * 70)

    # 1. Community Data
    gsm8k_samples = process_gsm8k()
    viquad_samples = process_viquad(max_samples=10000)
    multiturn_samples = process_multiturn(max_samples=8000)
    community_total = gsm8k_samples + viquad_samples + multiturn_samples

    # 2. Existing Synthetic Anchor Data (20,805 samples from Stage 1)
    anchor_file = os.path.join(DATASET_DIR, "vimind_4.5_combined.jsonl")
    if not os.path.exists(anchor_file):
        anchor_file = os.path.join(DATASET_DIR, "dense_core_vi.jsonl")

    anchor_samples = []
    if os.path.exists(anchor_file):
        with open(anchor_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        anchor_samples.append(json.loads(line.strip()))
                    except Exception:
                        pass
        print(f"⚓ Loaded {len(anchor_samples)} ViMind Synthetic Anchor samples from {anchor_file}")

    # 3. Master Merge & Shuffle
    master_samples = community_total + anchor_samples
    random.seed(42)
    random.shuffle(master_samples)

    master_out = os.path.join(DATASET_DIR, "vimind_4.5_master.jsonl")
    with open(master_out, "w", encoding="utf-8") as f:
        for s in master_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("=" * 70)
    print(f"🎉 MASTER DATASET CREATED: {len(master_samples):,} total samples!")
    print(f"📁 Output file: {master_out}")
    print("   - Community GSM8K Math CoT:        {:>6,}".format(len(gsm8k_samples)))
    print("   - Community UIT-ViQuAD Factual QA: {:>6,}".format(len(viquad_samples)))
    print("   - Community Multi-turn Dialogues:  {:>6,}".format(len(multiturn_samples)))
    print("   - ViMind Synthetic Behavioral:     {:>6,}".format(len(anchor_samples)))
    print("=" * 70)


if __name__ == "__main__":
    main()
