import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    Dataset for Pretraining ViMind on raw Vietnamese texts.
    Pads and truncates sequences to max_length, masking pad tokens in labels with -100.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = str(sample.get("text", ""))

        # Tokenize without adding special tokens automatically
        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids

        # Add <s> (BOS) and </s> (EOS)
        bos_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        eos_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids = [bos_id] + token_ids + [eos_id]

        # Pad to max_length
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [pad_id] * padding_len

        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels = input_tensor.clone()
        # Mask out pad tokens so loss is not computed on padding
        labels[input_tensor == pad_id] = -100

        return input_tensor, labels
