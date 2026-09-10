"""
ViMind 4.0 - Model Conversion & Weight Packaging Tool.
Supports:
  1. Converting PyTorch (.pth / .bin) checkpoints to HuggingFace SafeTensors (model.safetensors)
  2. Auto-detecting model architecture from checkpoint directory or state dict shapes
  3. Merging LoRA adapters into base model weights
  4. Exporting complete Hugging Face repository package with tokenizers & chat template
"""

import os
import sys
import shutil
import json
import argparse
import torch

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from model.model import ViMindConfig, ViMindForCausalLM


def convert_and_export(
    input_checkpoint: str,
    output_dir: str,
    lora_dir: str = None,
    is_moe: bool = True,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    model_size: str = "64m"
):
    print("=" * 70)
    print("       ViMind 4.0 - Model Packaging & Export Tool")
    print("=" * 70)
    print(f"Input Checkpoint: {input_checkpoint}")
    print(f"Output Directory: {output_dir}")
    print(f"LoRA Adapter Dir: {lora_dir}")
    print(f"Model Preset:     {model_size}")
    print(f"MoE Architecture: {is_moe} (Experts: {num_experts}, Top-{num_experts_per_tok})\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Locate checkpoint
    resolved_path = input_checkpoint
    if not os.path.exists(resolved_path):
        candidates = [
            resolved_path + ".pth",
            resolved_path,
            os.path.join(os.path.dirname(resolved_path) if os.path.dirname(resolved_path) else ".", "vimind_4.0_agent_final", "model.safetensors"),
            os.path.join(os.path.dirname(resolved_path) if os.path.dirname(resolved_path) else ".", "vimind_4.0_dpo_final", "model.safetensors"),
            os.path.join(os.path.dirname(resolved_path) if os.path.dirname(resolved_path) else ".", "vimind_4.0_moe_final", "model.safetensors"),
            os.path.join("out", "agent_rl", "vimind_4.0_agent_final"),
            os.path.join("out", "agent_rl", "vimind_4.0_agent.pth"),
            os.path.join("out", "agent_rl", "vimind_4.0_agent_final", "model.safetensors"),
            os.path.join("out", "dpo", "vimind_4.0_dpo_final"),
            os.path.join("out", "dpo", "vimind_4.0_dpo.pth"),
            os.path.join("out", "dpo", "vimind_4.0_dpo_final", "model.safetensors"),
            os.path.join("out", "sft_moe", "vimind_4.0_moe_final"),
            os.path.join("out", "sft_moe", "vimind_4.0_moe.pth"),
            os.path.join("out", "sft_moe", "vimind_4.0_moe_final", "model.safetensors"),
            os.path.join("out", "pretrain", "vimind_4.0_base_final"),
            os.path.join("out", "pretrain", "vimind_4.0_base.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    # 2. Check if a config.json already exists in checkpoint directory
    config = None
    cfg_dirs = []
    if os.path.isdir(resolved_path):
        cfg_dirs.append(resolved_path)
    elif os.path.isfile(resolved_path):
        cfg_dirs.append(os.path.dirname(resolved_path))
    cfg_dirs.extend([
        "out/agent_rl/vimind_4.0_agent_final",
        "out/dpo/vimind_4.0_dpo_final",
        "out/sft_moe/vimind_4.0_moe_final",
        "out/pretrain/vimind_4.0_base_final"
    ])

    for cdir in cfg_dirs:
        if cdir and os.path.exists(os.path.join(cdir, "config.json")):
            try:
                print(f"[Info] Found existing config.json at: {cdir}")
                config = ViMindConfig.from_pretrained(cdir)
                break
            except Exception as e:
                print(f"[Warning] Could not load config from {cdir}: {e}")

    # Fallback config builder if no config.json was found
    if config is None:
        vocab_size = 12803
        if os.path.exists("model/tokenizer_config.json") or os.path.exists("model/tokenizer.json"):
            try:
                tok = AutoTokenizer.from_pretrained("model")
                vocab_size = max(len(tok), 12803)
            except Exception:
                pass

        if model_size == "64m":
            config = ViMindConfig(
                vocab_size=vocab_size,
                hidden_size=640,
                num_hidden_layers=12,
                num_attention_heads=10,
                num_key_value_heads=5,
                intermediate_size=1728,
                max_seq_len=2048,
                use_moe=is_moe,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                rope_theta=1e6
            )
        else:
            config = ViMindConfig(
                vocab_size=vocab_size,
                hidden_size=512,
                num_hidden_layers=8,
                num_attention_heads=8,
                num_key_value_heads=4,
                intermediate_size=1088,
                max_seq_len=2048,
                use_moe=is_moe,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                rope_theta=1e6
            )

    # 3. Load state_dict
    state_dict = {}
    actual_file = resolved_path
    if os.path.isdir(resolved_path):
        if os.path.exists(os.path.join(resolved_path, "model.safetensors")):
            actual_file = os.path.join(resolved_path, "model.safetensors")
        elif os.path.exists(os.path.join(resolved_path, "pytorch_model.bin")):
            actual_file = os.path.join(resolved_path, "pytorch_model.bin")
        else:
            for f in os.listdir(resolved_path):
                if f.endswith((".safetensors", ".pth", ".bin")):
                    actual_file = os.path.join(resolved_path, f)
                    break

    if os.path.exists(actual_file) and not os.path.isdir(actual_file):
        print(f"Loading weights from: {actual_file}...")
        if actual_file.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(actual_file)
            except ImportError:
                print("[Error] safetensors is required: pip install safetensors")
                return
        else:
            state_dict = torch.load(actual_file, map_location="cpu")

        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Auto-align config to tensor shapes directly to prevent ANY size mismatch
        embed_key = next((k for k in ["model.embed_tokens.weight", "embed_tokens.weight"] if k in state_dict), None)
        if embed_key is not None:
            ckpt_vocab, ckpt_hidden = state_dict[embed_key].shape
            config.vocab_size = ckpt_vocab
            config.hidden_size = ckpt_hidden
            if ckpt_hidden == 640:
                config.num_hidden_layers = 12
                config.num_attention_heads = 10
                config.num_key_value_heads = 5
                config.intermediate_size = 1728
            elif ckpt_hidden == 512:
                config.num_hidden_layers = 8
                config.num_attention_heads = 8
                config.num_key_value_heads = 4
                config.intermediate_size = 1088

        gate_key = next((k for k in ["model.layers.0.mlp.gate.weight", "layers.0.mlp.gate.weight"] if k in state_dict), None)
        if gate_key is not None:
            config.use_moe = True
            config.num_experts = state_dict[gate_key].shape[0]
            config.num_experts_per_tok = num_experts_per_tok

    # 4. Instantiate Model & load weights
    print(f"Instantiating ViMind architecture: hidden={config.hidden_size}, layers={config.num_hidden_layers}, experts={getattr(config, 'num_experts', 0)} (MoE: {getattr(config, 'use_moe', False)})...")
    model = ViMindForCausalLM(config)

    if state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Base weights loaded successfully. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # 5. Merge LoRA if provided
    if lora_dir and os.path.exists(lora_dir):
        print(f"Loading LoRA weights from {lora_dir}...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir)
            print("Merging LoRA adapters into base weights...")
            model = model.merge_and_unload()
            print("LoRA successfully merged.")
        except Exception as e:
            print(f"[Warning] Failed to merge LoRA: {e}. Skipping merge.")

    # 6. Save Hugging Face SafeTensors
    print(f"\nExporting Hugging Face SafeTensors to: {output_dir}...")
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # 7. Package Tokenizer
    tokenizer_source = "model"
    if not os.path.exists(os.path.join(tokenizer_source, "tokenizer.json")):
        tokenizer_source = output_dir

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        tokenizer.save_pretrained(output_dir)
        print("Tokenizer successfully packaged.")
    except Exception as e:
        print(f"[Warning] AutoTokenizer packaging encountered: {e}")

    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja"
    ]:
        src = os.path.join(tokenizer_source, fname)
        if os.path.exists(src):
            dst = os.path.join(output_dir, fname)
            shutil.copyfile(src, dst)
            print(f"Copied {fname} -> {dst}")

    # 8. Write GGUF Export Instructions
    readme_info = f"""# ViMind 4.0 Export Artifact

- Architecture: {'Mixture-of-Experts (MoE)' if config.use_moe else 'Dense'}
- Total Parameters: {'198M (64M active)' if config.use_moe else '64M'}
- Max Context Length: 32,768 (YaRN RoPE)
- Special Tokens: `<think>`, `</think>`, `<tool_call>`, `</tool_call>`

### GGUF Quantization:
To convert this model to GGUF format for llama.cpp / Ollama / LM Studio:
```bash
python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile vimind-4.0-q4_k_m.gguf --outtype q8_0
./llama-quantize vimind-4.0-q8_0.gguf vimind-4.0-q4_k_m.gguf Q4_K_M
```
"""
    with open(os.path.join(output_dir, "EXPORT_INFO.md"), "w", encoding="utf-8") as f:
        f.write(readme_info)

    sf_path = os.path.join(output_dir, "model.safetensors")
    if os.path.exists(sf_path) and os.path.getsize(sf_path) > 1000000:
        print(f"✅ Model packaging successfully completed! SafeTensors size: {os.path.getsize(sf_path)/(1024*1024):.1f} MB")
    else:
        print(f"⚠️ Warning: {sf_path} is missing or small.")
    print(f"Ready for deployment in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ViMind 4.0 Weight Converter & Packager")
    parser.add_argument("--input", type=str, default="out/agent_rl/vimind_4.0_agent_final", help="Input checkpoint file or dir")
    parser.add_argument("--output", type=str, default="out/vimind_4.0_hf", help="Output export directory")
    parser.add_argument("--lora_dir", type=str, default=None, help="Optional LoRA adapter directory to merge")
    parser.add_argument("--model_size", type=str, default="64m", choices=["26m", "64m", "104m", "custom"], help="Architecture preset")
    parser.add_argument("--moe", action="store_true", default=True, help="Set flag if model is MoE")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts if MoE")
    parser.add_argument("--num_experts_per_tok", type=int, default=2, help="Active experts per token if MoE")
    args = parser.parse_args()

    convert_and_export(
        input_checkpoint=args.input,
        output_dir=args.output,
        lora_dir=args.lora_dir,
        is_moe=args.moe,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        model_size=args.model_size
    )


if __name__ == "__main__":
    main()
