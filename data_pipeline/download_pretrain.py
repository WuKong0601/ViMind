import os
import sys
import json
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clean_and_normalize import normalize_vietnamese_text, is_valid_text

def main():
    print("Downloading Vietnamese Wikipedia dataset...")
    # Load wikipedia dataset (20231101.vi)
    dataset = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", trust_remote_code=True)
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pretrain_vi.jsonl")
    
    print(f"Total documents downloaded: {len(dataset)}")
    print(f"Cleaning and saving to {output_path}...")
    
    saved_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(dataset):
            text = normalize_vietnamese_text(item["text"])
            # Require at least 50 words to avoid garbage/stubs
            if is_valid_text(text, min_words=50):
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write("\n")
                saved_count += 1
                
    print(f"Done! Saved {saved_count} valid documents for pretraining.")

if __name__ == "__main__":
    main()
