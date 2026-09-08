import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


# ==============================================================================
# ViMind Configuration
# ==============================================================================
class ViMindConfig(PretrainedConfig):
    model_type = "vimind"

    def __init__(
        self,
        vocab_size: int = 12800,
        hidden_size: int = 512,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        intermediate_size: int = 1088,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        hidden_act: str = "silu",
        flash_attn: bool = True,
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: Optional[int] = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        inference_rope_scaling: bool = False,
        rope_scaling: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.flash_attn = flash_attn

        # MoE configurations
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size or intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        # Long-context YaRN RoPE scaling configurations
        self.inference_rope_scaling = inference_rope_scaling
        if self.inference_rope_scaling and rope_scaling is None:
            self.rope_scaling = {
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.0,
            }
        else:
            self.rope_scaling = rope_scaling

    @classmethod
    def get_config_26m(cls, vocab_size: int = 12800, **kwargs):
        """ViMind 1.0 (26.2M parameter baseline, 8 layers, 512 dim)"""
        return cls(
            vocab_size=vocab_size,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=1088,
            max_position_embeddings=1024,
            **kwargs,
        )

    @classmethod
    def get_config_64m(cls, vocab_size: int = 12800, **kwargs):
        """ViMind 2.0 / 3.0 Dense (62.8M parameters, 12 layers, 640 dim, 2048 context, rope_theta 1e6)"""
        return cls(
            vocab_size=vocab_size,
            hidden_size=640,
            num_hidden_layers=12,
            num_attention_heads=10,
            num_key_value_heads=5,
            intermediate_size=1728,
            max_position_embeddings=2048,
            rope_theta=1e6,
            **kwargs,
        )

    @classmethod
    def get_config_moe_198m(cls, vocab_size: int = 12800, **kwargs):
        """ViMind 3.0 MoE (198M total params, 64M active per token, 4 experts, 12 layers, 640 dim)"""
        return cls(
            vocab_size=vocab_size,
            hidden_size=640,
            num_hidden_layers=12,
            num_attention_heads=10,
            num_key_value_heads=5,
            intermediate_size=1728,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=1,
            max_position_embeddings=2048,
            rope_theta=1e6,
            **kwargs,
        )

    @classmethod
    def get_config_104m(cls, vocab_size: int = 12800, **kwargs):
        """ViMind 2.0 Plus (104M parameters, 16 layers, 768 dim, 2048 context)"""
        return cls(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=16,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2048,
            max_position_embeddings=2048,
            rope_theta=1e6,
            **kwargs,
        )


# ==============================================================================
# Model Architecture Components
# ==============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) for fast & stable convergence."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    dim: int,
    end: int = 2048,
    theta: float = 1000000.0,
    rope_scaling: Optional[dict] = None,
):
    """Precomputes Rotary Position Embeddings (RoPE) frequencies with optional YaRN scaling."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None and rope_scaling.get("type") == "yarn":
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(theta))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0.0,
                1.0,
            )
            freqs = freqs * (1.0 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Applies RoPE to query and key tensors."""
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats Key/Value heads to match Query heads for Grouped Query Attention (GQA)."""
    if n_rep == 1:
        return x
    bsz, slen, num_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bsz, slen, num_kv_heads, n_rep, head_dim)
        .reshape(bsz, slen, num_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-Head Attention with Grouped Query Attention (GQA) & RoPE."""

    def __init__(self, config: ViMindConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.flash_attn = config.flash_attn and hasattr(F, "scaled_dot_product_attention")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Q-K Normalization (improves stability in deep architectures)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape

        # Projections
        xq = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Q-K norm
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV Cache for fast inference
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # Repeat KV for GQA
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose to [bsz, num_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled Dot-Product Attention
        if self.flash_attn and (attention_mask is None or torch.all(attention_mask == 1)):
            is_causal = (seq_len > 1) and (past_key_value is None)
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if seq_len > 1 and past_key_value is None:
                causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
                scores[:, :, :, -seq_len:] += causal_mask
            if attention_mask is not None:
                # attention_mask shape: [bsz, seq_len] -> [bsz, 1, 1, seq_len]
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """SwiGLU Multi-Layer Perceptron (as used in LLaMA-3 / Mistral / Qwen)."""

    def __init__(self, config: ViMindConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        inter_dim = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class MOEFeedForward(nn.Module):
    """Mixture-of-Experts (MoE) FeedForward module with top-k gating & load balancing loss."""

    def __init__(self, config: ViMindConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_aux_loss_coef = config.router_aux_loss_coef
        inter_dim = config.moe_intermediate_size or config.intermediate_size

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=inter_dim)
            for _ in range(self.num_experts)
        ])
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)

        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # Keep parameter nodes in computational graph for DDP gradient sync
                y[0, 0] = y[0, 0] + 0.0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.num_experts * self.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(bsz, seq_len, hidden_dim)


class ViMindBlock(nn.Module):
    """Transformer Decoder Block with Pre-RMSNorm and Dense/MoE MLP."""

    def __init__(self, layer_id: int, config: ViMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MOEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self Attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_out

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, present_kv


# ==============================================================================
# ViMind Core Model
# ==============================================================================
class ViMindModel(nn.Module):
    def __init__(self, config: ViMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([ViMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Precompute RoPE tables
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _init_rope(self, device: torch.device, max_pos: Optional[int] = None):
        """Recomputes RoPE tables on the target device to guarantee finite, valid tables."""
        end = max(max_pos or 0, getattr(self.config, "max_position_embeddings", 2048), 2048)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.config.head_dim,
            end=end,
            theta=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
        )
        self.freqs_cos = freqs_cos.to(device)
        self.freqs_sin = freqs_sin.to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, seq_len = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Embedding lookup
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # Robust self-healing & dynamic-extending RoPE tables: guarantee valid, sufficient length
        required_len = start_pos + seq_len
        if (
            not hasattr(self, "freqs_cos")
            or self.freqs_cos is None
            or self.freqs_cos.device != hidden_states.device
            or self.freqs_cos.device.type == "meta"
            or self.freqs_cos.numel() == 0
            or self.freqs_cos.shape[0] < required_len
            or self.freqs_cos[0, 0] != 1.0
            or torch.isnan(self.freqs_cos[0, 0])
        ):
            target_len = max(required_len, getattr(self.config, "max_position_embeddings", 2048), 2048)
            self._init_rope(hidden_states.device, max_pos=target_len)

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, present_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    position_embeddings,
                    past_kv,
                    use_cache,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
            presents.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


# ==============================================================================
# ViMind Causal LM (HuggingFace PreTrainedModel compatible)
# ==============================================================================
class ViMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ViMindConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Optional[ViMindConfig] = None):
        config = config or ViMindConfig()
        super().__init__(config)
        self.model = ViMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie word embeddings with LM Head
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def tie_weights(self, *args, **kwargs):
        super().tie_weights(*args, **kwargs)
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, presents = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        aux_loss = None
        if getattr(self.config, "use_moe", False):
            aux_loss = sum(getattr(layer.mlp, "aux_loss", 0.0) for layer in self.model.layers if hasattr(layer, "mlp"))

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_tokens = (shift_labels != -100).sum()
            if valid_tokens == 0:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            if aux_loss is not None:
                loss = loss + aux_loss

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents if use_cache else None,
            hidden_states=hidden_states,
        )
        if aux_loss is not None:
            output.aux_loss = aux_loss
        return output

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        eos_token_id: Optional[int] = 2,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
        streamer=None,
        **kwargs,
    ) -> torch.Tensor:
        """Lightweight native autoregressive text generation."""
        if input_ids is None:
            input_ids = inputs

        bsz = input_ids.shape[0]
        finished = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        past_key_values = None

        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            if past_key_values is not None:
                current_input = input_ids[:, -1:]
            else:
                current_input = input_ids

            outputs = self.forward(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = outputs.past_key_values

            logits = outputs.logits[:, -1, :].clone()

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(bsz):
                    unique_tokens = torch.unique(input_ids[b])
                    logits[b, unique_tokens] = torch.where(
                        logits[b, unique_tokens] > 0,
                        logits[b, unique_tokens] / repetition_penalty,
                        logits[b, unique_tokens] * repetition_penalty,
                    )

            # Top-K filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_top_k = top_k_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_top_k, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype), logits)

            # Top-P (Nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits.float(), dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -1e9)

            # Sampling or greedy (strictly computed in float32 for numerical stability)
            if temperature > 0:
                probs = F.softmax(logits.float(), dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                prob_sum = probs.sum(dim=-1, keepdim=True)
                probs = torch.where(prob_sum > 0, probs / prob_sum, torch.ones_like(probs) / probs.size(-1))
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Check EOS
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), torch.tensor(eos_token_id, device=input_ids.device), next_token)
                finished |= next_token.squeeze(-1).eq(eos_token_id)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if streamer:
                streamer.put(next_token.cpu())

            if finished.all():
                break

        if streamer:
            streamer.end()

        return input_ids
