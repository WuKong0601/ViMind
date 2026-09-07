import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


def compute_per_token_logps(
    model: torch.nn.Module,
    input_ids: Tensor,
    n_keep: int,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Computes log probabilities for the last `n_keep` tokens in `input_ids`.
    Vectorized and optimized for GPU execution.
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(unwrapped, "_orig_mod", unwrapped)

    # Forward pass through model to get logits
    logits = raw_model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]

    # Sliced logits corresponding to target tokens
    target_ids = input_ids[:, -n_keep:]
    target_logits = logits[:, -n_keep:, :]

    # Vectorized gather over vocabulary dimension
    log_probs = target_logits.log_softmax(dim=-1)
    per_token_logps = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    return per_token_logps


@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


class RolloutEngine(ABC):
    @abstractmethod
    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


class TorchRolloutEngine(RolloutEngine):
    """
    Pure PyTorch Rollout Engine.
    Executes batched generation and computes exact per-token reference logprobs.
    """
    def __init__(
        self,
        policy_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        autocast_ctx=None,
    ):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx if autocast_ctx is not None else nullcontext()

    def rollout(
        self,
        prompt_ids: Tensor,
        attention_mask: Tensor,
        num_generations: int,
        max_new_tokens: int,
        temperature: float = 0.8,
    ) -> RolloutResult:
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        raw_model = getattr(model, "_orig_mod", model)

        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)

        # Expand inputs by num_generations
        expanded_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)

        with torch.no_grad(), self.autocast_ctx:
            output_ids = raw_model.generate(
                input_ids=expanded_prompt_ids,
                attention_mask=expanded_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()

            total_len = output_ids.size(1)
            completion_len = total_len - prompt_len
            completion_ids = output_ids[:, prompt_len:]

            full_mask = (output_ids != self.tokenizer.pad_token_id).long()
            per_token_logps = compute_per_token_logps(
                raw_model,
                output_ids,
                completion_len,
                attention_mask=full_mask,
            )

        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        prompt_lens = prompt_ids.new_full((output_ids.size(0),), prompt_len)
        completion_mask = (completion_ids != self.tokenizer.pad_token_id).long()

        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=per_token_logps,
            completions=completions,
            prompt_lens=prompt_lens,
            completion_mask=completion_mask,
        )

    def update_policy(self, model: torch.nn.Module):
        self.policy_model = model


def create_rollout_engine(
    engine_type: str,
    policy_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    autocast_ctx=None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(
            policy_model=policy_model,
            tokenizer=tokenizer,
            device=device,
            autocast_ctx=autocast_ctx,
        )
    else:
        raise ValueError(f"Unsupported rollout engine type: {engine_type}")
