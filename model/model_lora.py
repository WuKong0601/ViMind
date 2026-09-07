import os
import sys
import torch
import torch.nn as nn
from typing import List, Optional


class LoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module.
    Decomposes weight updates: ΔW = (α / r) * (B @ A)
    where A ~ N(0, 0.02), B = 0.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / float(rank) if rank > 0 else 1.0

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize: A ~ Normal(0, 0.02), B = 0 (ensures ΔW = 0 at start)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.B(self.A(x))) * self.scaling


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
):
    """
    Injects LoRA modules into specified Linear layers.
    Default targets: ["q_proj", "k_proj", "v_proj", "o_proj"].
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Target device and dtype from model
    first_param = next(model.parameters())
    device = first_param.device
    dtype = first_param.dtype

    applied_count = 0
    # Snapshot module list before attaching any new submodules
    for name, module in list(model.named_modules()):
        if "lora" in name or hasattr(module, "lora"):
            continue
        if isinstance(module, nn.Linear):
            is_target = any(target in name for target in target_modules)
            # Exclude lm_head from automatic matching
            if "lm_head" in name:
                is_target = False

            if is_target:
                lora = LoRA(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ).to(device=device, dtype=dtype)

                setattr(module, "lora", lora)
                original_forward = module.forward
                setattr(module, "_orig_forward", original_forward)

                def make_forward(orig_fwd, lora_layer):
                    def forward_with_lora(x, *args, **kwargs):
                        return orig_fwd(x, *args, **kwargs) + lora_layer(x)
                    return forward_with_lora

                module.forward = make_forward(original_forward, lora)
                applied_count += 1

    return applied_count


def mark_only_lora_as_trainable(model: nn.Module) -> List[nn.Parameter]:
    """Freezes all parameters except those belonging to LoRA modules."""
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    return trainable_params


def save_lora(model: nn.Module, path: str):
    """Saves only the LoRA adapter state dictionary."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}

    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            for k, v in module.lora.state_dict().items():
                state_dict[f"{clean_name}.lora.{k}"] = v.cpu()

    torch.save(state_dict, path)


def load_lora(model: nn.Module, path: str, device: Optional[torch.device] = None):
    """Loads LoRA weights into existing injected LoRA layers."""
    if device is None:
        device = next(model.parameters()).device

    state_dict = torch.load(path, map_location=device)
    state_dict = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    raw_model = getattr(model, "_orig_mod", model)
    loaded_layers = 0
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora."
            lora_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if lora_state:
                module.lora.load_state_dict(lora_state)
                loaded_layers += 1

    return loaded_layers


def merge_lora(model: nn.Module, lora_path: Optional[str] = None):
    """
    Merges LoRA weights back into the original linear weights:
    W_new = W_old + scaling * (B @ A)
    Removes LoRA modules and restores original forwards.
    """
    if lora_path is not None:
        load_lora(model, lora_path)

    raw_model = getattr(model, "_orig_mod", model)
    for name, module in list(raw_model.named_modules()):
        if hasattr(module, "lora"):
            lora = module.lora
            delta_w = (lora.B.weight.data @ lora.A.weight.data) * lora.scaling
            module.weight.data.add_(delta_w.to(module.weight.dtype))

            # Restore original forward
            if hasattr(module, "_orig_forward"):
                module.forward = module._orig_forward
                delattr(module, "_orig_forward")
            delattr(module, "lora")
