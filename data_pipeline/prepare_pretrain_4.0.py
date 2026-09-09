import os
import sys
import json
import argparse
from tqdm import tqdm

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

from data_pipeline.clean_and_normalize import normalize_vietnamese_text, is_valid_text, format_textbook_qa


def load_facts_passages(facts_file: str, repeat_count: int = 100) -> list:
    """Extract factual QA into natural descriptive passages and duplicate for high embedding density."""
    passages = []
    if not os.path.exists(facts_file):
        return passages

    with open(facts_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                convs = data.get("conversations", [])
                if len(convs) >= 2:
                    q = convs[0].get("content", "").strip()
                    a = convs[1].get("content", "").strip()
                    # Convert to textbook-style factual passage
                    passage = f"Tri thức bách khoa / Câu hỏi: {q}\n\nGiải đáp sự thật: {a}"
                    passages.append(passage)
            except Exception:
                pass

    # Multiply to reinforce memorization in base model
    repeated = []
    for _ in range(repeat_count):
        repeated.extend(passages)
    return repeated


def main():
    parser = argparse.ArgumentParser(description="ViMind 4.0 Pre-training Corpus Preparer")
    parser.add_argument("--output_path", type=str, default="dataset/pretrain_vi_v4.jsonl", help="Output path")
    parser.add_argument("--max_wiki", type=int, default=80000, help="Max Wikipedia articles")
    parser.add_argument("--max_math", type=int, default=20000, help="Max math textbook samples")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print("=" * 70)
    print("📚 VIMIND 4.0: TARGETED PRE-TRAINING CORPUS COMPILER")
    print(f"   Target Output: {args.output_path}")
    print("=" * 70)

    total_saved = 0
    with open(args.output_path, "w", encoding="utf-8") as f_out:
        # 1. High-Density Grounding Facts (Memorize Vietnam Capital, 63 provinces, Math, Identity)
        facts_path = os.path.join(PROJECT_ROOT, "dataset", "identity_and_vietnam_facts.jsonl")
        print("\n🇻🇳 [1/3] Ingesting Core Vietnam Grounding Facts & Axioms...")
        fact_passages = load_facts_passages(facts_path, repeat_count=50)
        for p in fact_passages:
            f_out.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
            total_saved += 1
        print(f"   ✅ Injected {len(fact_passages):,} high-priority factual grounding passages.")

        # 2. Wikipedia Corpus (Local or Streamed from HF)
        local_wiki = os.path.join(PROJECT_ROOT, "dataset", "pretrain_vi.jsonl")
        print(f"\n🌐 [2/3] Ingesting Vietnamese Wikipedia Articles (Limit: {args.max_wiki:,})...")
        wiki_count = 0
        if os.path.exists(local_wiki) and os.path.getsize(local_wiki) > 10 * (1024 ** 2):
            print(f"   Using existing local Wikipedia corpus: {local_wiki}...")
            with open(local_wiki, "r", encoding="utf-8") as f_in:
                for line in tqdm(f_in, desc="Streaming local Wiki", total=args.max_wiki):
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        txt = normalize_vietnamese_text(obj.get("text", ""))
                        if is_valid_text(txt, min_words=40):
                            f_out.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                            wiki_count += 1
                            total_saved += 1
                            if wiki_count >= args.max_wiki:
                                break
                    except Exception:
                        continue
        else:
            print("   Streaming Wikipedia from HuggingFace (wikimedia/wikipedia 20231101.vi)...")
            try:
                from datasets import load_dataset
                ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
                for item in tqdm(ds, desc="Downloading Wiki", total=args.max_wiki):
                    txt = normalize_vietnamese_text(item.get("text", ""))
                    if is_valid_text(txt, min_words=40):
                        f_out.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                        wiki_count += 1
                        total_saved += 1
                        if wiki_count >= args.max_wiki:
                            break
            except Exception as e:
                print(f"   ⚠️ Warning: Could not stream Wikipedia: {e}")
        print(f"   ✅ Saved {wiki_count:,} Wikipedia knowledge passages.")

        # 3. Math & Science Textbooks
        print(f"\n📐 [3/3] Ingesting Math & Science Textbook QA (Limit: {args.max_math:,})...")
        math_count = 0
        try:
            from datasets import load_dataset
            ds_math = load_dataset("LuminarAI/vietnamese-ms-hs-textbook-math-300K", split="train", streaming=True)
            for item in tqdm(ds_math, desc="Processing Math QA", total=args.max_math):
                q = item.get("question", "")
                a = item.get("answer", "")
                passage = format_textbook_qa(q, a)
                if is_valid_text(passage, min_words=20):
                    f_out.write(json.dumps({"text": passage}, ensure_ascii=False) + "\n")
                    math_count += 1
                    total_saved += 1
                    if math_count >= args.max_math:
                        break
        except Exception as e:
            print(f"   ⚠️ Notice: Math textbook download skipped or not available ({e}).")
        print(f"   ✅ Saved {math_count:,} Math & Science textbook passages.")

    print("\n" + "=" * 70)
    print(f"🎉 Pre-training Corpus Compiled Successfully! Total Passages: {total_saved:,}")
    if os.path.exists(args.output_path):
        size_mb = os.path.getsize(args.output_path) / (1024 ** 2)
        print(f"📦 File Size: {size_mb:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
