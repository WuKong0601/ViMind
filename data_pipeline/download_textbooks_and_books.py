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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clean_and_normalize import normalize_vietnamese_text, format_textbook_qa, is_valid_text

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library is required. Run 'pip install datasets'.")
    sys.exit(1)


def ingest_math_textbooks(output_file, max_samples=150000, sample_only=False):
    """Ingests and formats Vietnamese middle and high school textbook QA dataset."""
    limit = 50 if sample_only else max_samples
    print(f"\n📚 [1/3] Downloading School Textbooks: LuminarAI/vietnamese-ms-hs-textbook-math-300K (limit: {limit})...")
    saved = 0
    try:
        ds = load_dataset("LuminarAI/vietnamese-ms-hs-textbook-math-300K", split="train", streaming=True)
        for item in tqdm(ds, desc="Processing Textbooks", total=limit):
            q = item.get("question", "")
            a = item.get("answer", "")
            text = format_textbook_qa(q, a)
            if is_valid_text(text, min_words=20):
                output_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                saved += 1
                if saved >= limit:
                    break
        print(f"✅ Textbooks saved: {saved:,} educational passages.")
    except Exception as e:
        print(f"⚠️ Warning: Could not ingest textbook dataset: {e}")
    return saved


def ingest_vietnamese_books(output_file, max_samples=5000, sample_only=False):
    """Ingests curated Vietnamese books (literature, science, geography, history)."""
    limit = 20 if sample_only else max_samples
    print(f"\n📖 [2/3] Downloading Vietnamese Books: thailevann/10000_Vietnamese_Books (limit: {limit})...")
    saved = 0
    try:
        ds = load_dataset("thailevann/10000_Vietnamese_Books", split="train", streaming=True)
        for item in tqdm(ds, desc="Processing Books", total=limit):
            raw_text = item.get("text", "")
            # Books are large texts; we chunk them into coherent passages (~300 - 600 words)
            paragraphs = raw_text.split("\n\n")
            current_chunk = []
            current_len = 0
            for para in paragraphs:
                p_clean = normalize_vietnamese_text(para)
                if not p_clean:
                    continue
                w_count = len(p_clean.split())
                current_chunk.append(p_clean)
                current_len += w_count

                if current_len >= 300:
                    chunk_text = "\n\n".join(current_chunk)
                    if is_valid_text(chunk_text, min_words=50):
                        output_file.write(json.dumps({"text": chunk_text}, ensure_ascii=False) + "\n")
                        saved += 1
                    current_chunk = []
                    current_len = 0
                    if sample_only and saved >= limit:
                        break

            if current_chunk and not (sample_only and saved >= limit):
                chunk_text = "\n\n".join(current_chunk)
                if is_valid_text(chunk_text, min_words=50):
                    output_file.write(json.dumps({"text": chunk_text}, ensure_ascii=False) + "\n")
                    saved += 1

            if sample_only and saved >= limit:
                break

        print(f"✅ Books passages saved: {saved:,} passages.")
    except Exception as e:
        print(f"⚠️ Warning: Could not ingest books dataset: {e}")
    return saved


def ingest_wikipedia(output_file, existing_wiki_path="dataset/pretrain_vi.jsonl", max_samples=150000, sample_only=False):
    """Ingests clean Wikipedia articles (from local cache if available, or Hugging Face)."""
    limit = 50 if sample_only else max_samples
    print(f"\n🌐 [3/3] Ingesting Wikipedia Articles (limit: {limit})...")
    saved = 0

    if os.path.exists(existing_wiki_path):
        print(f"Loading from existing local corpus: {existing_wiki_path}...")
        with open(existing_wiki_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Ingesting local Wiki", total=limit):
                if not line.strip():
                    continue
                output_file.write(line.strip() + "\n")
                saved += 1
                if saved >= limit:
                    break
    else:
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
            for item in tqdm(ds, desc="Downloading Wiki", total=limit):
                text = normalize_vietnamese_text(item.get("text", ""))
                if is_valid_text(text, min_words=50):
                    output_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    saved += 1
                    if saved >= limit:
                        break
        except Exception as e:
            print(f"⚠️ Warning: Could not download Wikipedia: {e}")

    print(f"✅ Wikipedia passages saved: {saved:,} articles.")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Multi-source Vietnamese Pre-training Ingestion Pipeline")
    parser.add_argument("--output_path", type=str, default="dataset/pretrain_vi_v2.jsonl", help="Output JSONL path")
    parser.add_argument("--sample_only", action="store_true", help="Quick sample test mode")
    parser.add_argument("--max_textbook", type=int, default=150000, help="Max textbook samples")
    parser.add_argument("--max_books", type=int, default=20000, help="Max book chunks")
    parser.add_argument("--max_wiki", type=int, default=150000, help="Max wikipedia samples")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print("=" * 70)
    print("🚀 VIMIND 2.0 KNOWLEDGE ENRICHMENT DATA PIPELINE")
    print(f"📍 Target Output: {args.output_path}")
    print(f"🧪 Mode: {'SAMPLE / TEST' if args.sample_only else 'FULL EXTRACTION'}")
    print("=" * 70)

    total_saved = 0
    with open(args.output_path, "w", encoding="utf-8") as f_out:
        total_saved += ingest_math_textbooks(f_out, max_samples=args.max_textbook, sample_only=args.sample_only)
        total_saved += ingest_vietnamese_books(f_out, max_samples=args.max_books, sample_only=args.sample_only)
        total_saved += ingest_wikipedia(f_out, max_samples=args.max_wiki, sample_only=args.sample_only)

    print("\n" + "=" * 70)
    print(f"🎉 Pipeline Completed! Total knowledge documents: {total_saved:,}")
    if os.path.exists(args.output_path):
        size_mb = os.path.getsize(args.output_path) / (1024 ** 2)
        print(f"📦 Output file size: {size_mb:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
