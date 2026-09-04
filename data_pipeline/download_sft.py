import os
import sys
import json
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clean_and_normalize import normalize_vietnamese_text

def process_alpaca(item):
    """Convert Alpaca format to multi-turn dialogue format."""
    instruction = item.get("instruction_vi", "")
    input_text = item.get("input_vi", "")
    output = item.get("output_vi", "")
    
    if input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction
        
    prompt = normalize_vietnamese_text(prompt)
    output = normalize_vietnamese_text(output)
    
    if not prompt or not output:
        return None
        
    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]
    }

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sft_vi.jsonl")
    
    print("Downloading 5CD-AI/Vietnamese-alpaca-gpt4-gg-translated...")
    alpaca_dataset = load_dataset("5CD-AI/Vietnamese-alpaca-gpt4-gg-translated", split="train")
    
    saved_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        # Process Alpaca
        print("Processing Alpaca dataset...")
        for item in tqdm(alpaca_dataset):
            conv = process_alpaca(item)
            if conv:
                json.dump(conv, f, ensure_ascii=False)
                f.write("\n")
                saved_count += 1
                
    print(f"Done! Saved {saved_count} multi-turn conversations to {output_path}.")

if __name__ == "__main__":
    main()
