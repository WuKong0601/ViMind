import os
import sys
import json
import argparse

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def build_v4_datasets(output_dir="dataset"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("🚀 VIMIND 4.0: COMPILING MASTER SFT & GROUNDING DATASET")
    print("=" * 70)

    facts_file = os.path.join(output_dir, "identity_and_vietnam_facts.jsonl")
    facts_data = []
    if os.path.exists(facts_file):
        with open(facts_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        facts_data.append(json.loads(line))
                    except Exception:
                        pass
    print(f"✅ Loaded {len(facts_data)} Core Factual & Grounding pairs from {facts_file}")

    sft_v4_file = os.path.join(output_dir, "sft_vi_v4.jsonl")
    total_samples = 0

    with open(sft_v4_file, "w", encoding="utf-8") as f_out:
        # 1. Heavily oversample grounding facts (x25) so the model firmly memorizes them
        for _ in range(25):
            for item in facts_data:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_samples += 1
        print(f"   [1/3] Injected {len(facts_data) * 25:,} oversampled factual grounding samples.")

        # 2. Existing SFT reasoning & tool calls
        for subfile, mult in [("sft_reasoning_vi.jsonl", 10), ("sft_toolcall_vi.jsonl", 10)]:
            p = os.path.join(output_dir, subfile)
            if os.path.exists(p):
                count = 0
                with open(p, "r", encoding="utf-8") as f_sub:
                    lines = [json.loads(l) for l in f_sub if l.strip()]
                for _ in range(mult):
                    for obj in lines:
                        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        total_samples += 1
                        count += 1
                print(f"   [2/3] Injected {count:,} samples from {subfile}.")

        # 3. Base Conversational SFT (52k Alpaca)
        existing_sft = os.path.join(output_dir, "sft_vi.jsonl")
        if not os.path.exists(existing_sft):
            print("   📥 Downloading 5CD-AI/Vietnamese-alpaca-gpt4-gg-translated...")
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
                print(f"   ⚠️ Warning: Could not download Alpaca: {ex}")

        if os.path.exists(existing_sft):
            with open(existing_sft, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if line:
                        f_out.write(line + "\n")
                        total_samples += 1

    print(f"\n🎉 ViMind 4.0 SFT Master Dataset Ready: {sft_v4_file} ({total_samples:,} samples)!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViMind 4.0 Distilled Dataset Builder")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory")
    args = parser.parse_args()
    build_v4_datasets(args.output_dir)
