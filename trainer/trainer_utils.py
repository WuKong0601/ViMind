import os
import sys
import math
import random
import torch
import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModel

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content: str):
    if is_main_process():
        print(content, flush=True)


def get_lr(current_step: int, total_steps: int, lr: float, min_lr: float = 1e-6) -> float:
    """Cosine learning rate decay schedule."""
    if current_step >= total_steps:
        return min_lr
    ratio = current_step / max(1, total_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def setup_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_distributed_mode() -> int:
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


class LMForRewardModel:
    """
    Reward Model interface for AI-Feedback in RLAIF & GRPO.
    Supports sequence-classification reward models or teacher scoring models.
    """

    def __init__(self, model_path: str, device: str = "cuda:0", dtype=torch.float16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"📦 Nạp Reward Model từ: {model_path} trên {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
            self.model = self.model.to(self.device).eval()
            self.has_model = True
        except Exception as e:
            print(f"⚠️ Không thể nạp Reward Model ({e}). Sử dụng chế độ heuristic fallback reward.")
            self.has_model = False

    @torch.no_grad()
    def get_score(self, prompt: str, response: str) -> float:
        if not self.has_model:
            # Fallback heuristic reward: length, absence of repetitive patterns, formatting
            score = 0.0
            if len(response.strip()) > 20:
                score += 0.5
            if "<think>" in response and "</think>" in response:
                score += 1.0
            if "<tool_call>" in response and "</tool_call>" in response:
                score += 1.0
            return min(max(score, -3.0), 3.0)

        eval_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if hasattr(self.model, "get_score"):
            score = self.model.get_score(self.tokenizer, eval_messages)
            return float(max(min(score, 3.0), -3.0))

        # Default sequence classification logit extraction
        inputs = self.tokenizer(prompt + "\n" + response, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        outputs = self.model(**inputs)
        if hasattr(outputs, "logits"):
            return float(outputs.logits[0, 0].item())
        return 0.5
