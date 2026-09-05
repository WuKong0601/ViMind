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


def precompute_freqs_cis(dim: int, end: int = 2048, theta: float = 10000.0):
    """Precomputes Rotary Position Embeddings (RoPE) frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
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
    """SwiGLU Multi-Layer Perceptron (as used in LLaMA-3 / Mistral)."""

    def __init__(self, config: ViMindConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class ViMindBlock(nn.Module):
    """Transformer Decoder Block with Pre-RMSNorm."""

    def __init__(self, layer_id: int, config: ViMindConfig):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

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

        # Precompute RoPE tables
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _init_rope(self, device: torch.device):
        """Recomputes RoPE tables on the target device to guarantee finite, valid tables."""
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.config.head_dim,
            end=self.config.max_position_embeddings,
            theta=self.config.rope_theta,
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

        # Robust self-healing RoPE tables: guarantee valid, non-meta, non-zero values on correct device
        if (
            not hasattr(self, "freqs_cos")
            or self.freqs_cos is None
            or self.freqs_cos.device != hidden_states.device
            or self.freqs_cos.device.type == "meta"
            or self.freqs_cos.numel() == 0
            or self.freqs_cos[0, 0] != 1.0
            or torch.isnan(self.freqs_cos[0, 0])
        ):
            self._init_rope(hidden_states.device)

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []
        for layer, past_kv in zip(self.layers, past_key_values):
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
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Optional[ViMindConfig] = None):
        config = config or ViMindConfig()
        super().__init__(config)
        self.model = ViMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie word embeddings with LM Head
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def tie_weights(self):
        super().tie_weights()
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents if use_cache else None,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
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
