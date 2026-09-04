import os
import json
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

def batch_iterator(dataset_path, batch_size=10000):
    """Yields batches of text from the JSONL dataset"""
    batch = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get("text", "")
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset", "pretrain_vi.jsonl")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "model")
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_size = 12800
    
    # Define special tokens
    special_tokens = [
        "<pad>", 
        "<s>", 
        "</s>", 
        "<unk>", 
        "<think>", 
        "</think>", 
        "<tool_call>"
    ]
    
    print(f"Training tokenizer on {dataset_path} with vocab size {vocab_size}...")
    tokenizer = ByteLevelBPETokenizer()
    
    # Train the tokenizer
    tokenizer.train_from_iterator(
        batch_iterator(dataset_path),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens
    )
    
    print(f"Saving tokenizer to {output_dir}...")
    # Save the raw tokenizer format
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    
    # Wrap in transformers' PreTrainedTokenizerFast for easy loading later
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>"
    )
    fast_tokenizer.save_pretrained(output_dir)
    
    print("Tokenizer training complete!")
    
    # Test
    test_text = "ViMind là một dự án trí tuệ nhân tạo được huấn luyện từ đầu."
    encoded = fast_tokenizer.encode(test_text)
    decoded = fast_tokenizer.convert_ids_to_tokens(encoded)
    print("\n--- TEST TOKENIZER ---")
    print(f"Input: {test_text}")
    print(f"Tokens: {decoded}")
    print(f"IDs: {encoded}")
    print(f"Total tokens: {len(encoded)}")

if __name__ == "__main__":
    main()
