import os
import json
from datasets import load_dataset
from tqdm import tqdm
from clean_and_normalize import normalize_vietnamese_text

def process_dpo_pair(item):
    """
    Given a dataset item containing 'prompt', 'chosen', and 'rejected',
    convert it to ViMind DPO format.
    """
    # Adjust field names if your dataset differs
    prompt = item.get("question_vi", "")
    chosen = item.get("chosen_vi", "")
    rejected = item.get("rejected_vi", "")
    
    if not prompt and "question" in item:
        prompt = item["question"]
        
    prompt = normalize_vietnamese_text(prompt)
    
    # Handle list-based format vs string format
    if isinstance(chosen, str):
        chosen_content = normalize_vietnamese_text(chosen)
    elif isinstance(chosen, list) and len(chosen) > 0 and 'content' in chosen[-1]:
        chosen_content = normalize_vietnamese_text(chosen[-1]['content'])
    else:
        chosen_content = ""
        
    if isinstance(rejected, str):
        rejected_content = normalize_vietnamese_text(rejected)
    elif isinstance(rejected, list) and len(rejected) > 0 and 'content' in rejected[-1]:
        rejected_content = normalize_vietnamese_text(rejected[-1]['content'])
    else:
        rejected_content = ""
        
    if not prompt or not chosen_content or not rejected_content:
        return None
        
    return {
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_content}
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_content}
        ]
    }

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dpo_vi.jsonl")
    
    # We will use an open Vietnamese Ultrafeedback dataset
    # You can change this to any valid DPO dataset available on HuggingFace
    dataset_name = "5CD-AI/Vietnamese-Intel-orca_dpo_pairs-gg-translated"
    
    print(f"Attempting to download DPO dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Error loading '{dataset_name}': {e}")
        print("Please verify the dataset name on Hugging Face.")
        return
        
    saved_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        print("Processing DPO pairs...")
        for item in tqdm(dataset):
            dpo_item = process_dpo_pair(item)
            if dpo_item:
                json.dump(dpo_item, f, ensure_ascii=False)
                f.write("\n")
                saved_count += 1
                
    print(f"Done! Saved {saved_count} DPO pairs to {output_path}.")

if __name__ == "__main__":
    main()
