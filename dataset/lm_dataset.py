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


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) ViMind on conversational data.
    Applies Loss Masking: only computes loss on assistant responses (user prompt and pad tokens are -100).
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")
        self.bos_asst_ids = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.samples)

    def _generate_labels(self, input_ids: list) -> list:
        labels = [-100] * len(input_ids)
        asst_len = len(self.bos_asst_ids)
        i = 0
        while i <= len(input_ids) - asst_len:
            if input_ids[i : i + asst_len] == self.bos_asst_ids:
                start = i + asst_len
                end = start
                while end < len(input_ids) and input_ids[end] != self.eos_id:
                    end += 1
                # Include EOS token in training so model learns when to terminate generation
                if end < len(input_ids):
                    end += 1
                for j in range(start, min(end, len(input_ids))):
                    labels[j] = input_ids[j]
                i = end
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample.get("conversations", [])

        # Format conversation via chat template
        full_text = self.tokenizer.apply_chat_template(conversations, tokenize=False)

        # Tokenize and truncate to max_length (keeping prompt and beginning of response)
        input_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids[: self.max_length]

        labels = self._generate_labels(input_ids)

        # Safety fallback: if no assistant tokens were found within max_length, treat as standard autoregressive
        if not any(lbl != -100 for lbl in labels):
            labels = list(input_ids)

        # Padding
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [self.pad_id] * padding_len
            labels = labels + [-100] * padding_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """
    Dataset for Direct Preference Optimization (DPO) on pair-wise preference data.
    Each sample contains 'chosen' and 'rejected' conversation turns.
    Applies Loss Masking so log probabilities are computed only on the assistant's responses.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")
        self.bos_asst_ids = tokenizer("<s>assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __len__(self):
        return len(self.samples)

    def _generate_labels(self, input_ids: list) -> list:
        labels = [-100] * len(input_ids)
        asst_len = len(self.bos_asst_ids)
        i = 0
        while i <= len(input_ids) - asst_len:
            if input_ids[i : i + asst_len] == self.bos_asst_ids:
                start = i + asst_len
                end = start
                while end < len(input_ids) and input_ids[end] != self.eos_id:
                    end += 1
                if end < len(input_ids):
                    end += 1
                for j in range(start, min(end, len(input_ids))):
                    labels[j] = input_ids[j]
                i = end
            else:
                i += 1
        return labels

    def _process_item(self, conversations: list):
        full_text = self.tokenizer.apply_chat_template(conversations, tokenize=False)
        input_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids[: self.max_length]
        labels = self._generate_labels(input_ids)

        if not any(lbl != -100 for lbl in labels):
            labels = list(input_ids)

        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids = input_ids + [self.pad_id] * padding_len
            labels = labels + [-100] * padding_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen_conv = sample.get("chosen", [])
        rejected_conv = sample.get("rejected", [])

        chosen_input_ids, chosen_labels = self._process_item(chosen_conv)
        rejected_input_ids, rejected_labels = self._process_item(rejected_conv)

        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


class RLAIFDataset(Dataset):
    """
    Dataset for Reinforcement Learning / GRPO training.
    Provides formatted prompts for rollout exploration with optional Chain-of-Thought (<think>) prefixing.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512, thinking_ratio: float = 0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def _create_prompt(self, sample: dict) -> str:
        if "conversations" in sample:
            conv = sample["conversations"]
            # Exclude last assistant message if present so model generates it
            if len(conv) > 0 and conv[-1].get("role") == "assistant":
                prompt_conv = conv[:-1]
            else:
                prompt_conv = conv

            prompt = self.tokenizer.apply_chat_template(
                prompt_conv, tokenize=False, add_generation_prompt=True
            )
        elif "prompt" in sample:
            prompt = str(sample["prompt"])
        else:
            prompt = str(sample.get("text", ""))

        import random
        if self.thinking_ratio > 0 and random.random() < self.thinking_ratio:
            if not prompt.endswith("<think>\n") and not prompt.endswith("<think>"):
                prompt = prompt + "<think>\n"

        return prompt

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self._create_prompt(sample)
        gt = sample.get("ground_truth", sample.get("answer", ""))
        return {"prompt": prompt, "ground_truth": str(gt)}
