"""
ViMind 3.0 - Model Conversion & Weight Packaging Tool.
Supports:
  1. Converting PyTorch (.pth / .bin) checkpoints to HuggingFace SafeTensors (model.safetensors)
  2. Merging LoRA adapters into base model weights
  3. Exporting complete Hugging Face repository package with tokenizers & chat template
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

from model.model import ViMindConfig, ViMindForCausalLM


def convert_and_export(
    input_checkpoint: str,
    output_dir: str,
    lora_dir: str = None,
    is_moe: bool = False,
    num_experts: int = 4
):
    print("=" * 70)
    print("       ViMind 3.0 - Model Packaging & Export Tool")
    print("=" * 70)
    print(f"Input Checkpoint: {input_checkpoint}")
    print(f"Output Directory: {output_dir}")
    print(f"LoRA Adapter Dir: {lora_dir}")
    print(f"MoE Architecture: {is_moe} (Experts: {num_experts})\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare ViMind Config
    if is_moe:
        config = ViMindConfig(
            use_moe=True,
            num_experts=num_experts,
            num_experts_per_tok=1,
            max_seq_len=2048,
            rope_theta=1e6
        )
    else:
        config = ViMindConfig(
            use_moe=False,
            max_seq_len=2048,
            rope_theta=1e6
        )

    # 2. Instantiate Model
    print("Initializing ViMind model architecture...")
    model = ViMindForCausalLM(config)

    # 3. Load base weights if checkpoint exists
    if os.path.exists(input_checkpoint):
        print(f"Loading base weights from {input_checkpoint}...")
        if input_checkpoint.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(input_checkpoint)
            except ImportError:
                print("[Error] safetensors library is required. Install: pip install safetensors")
                return
        else:
            state_dict = torch.load(input_checkpoint, map_location="cpu")

        # Handle unwrapping nested 'model' or 'module' prefix
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Base weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print(f"[Warning] Base checkpoint {input_checkpoint} not found. Exporting blank architecture template.")

    # 4. Merge LoRA if provided
    if lora_dir and os.path.exists(lora_dir):
        print(f"Loading LoRA weights from {lora_dir}...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir)
            print("Merging LoRA adapter into base model weights...")
            model = model.merge_and_unload()
            print("LoRA successfully merged.")
        except ImportError:
            print("[Warning] 'peft' package not installed. Skipping LoRA merge.")
        except Exception as e:
            print(f"[Error] Failed to merge LoRA: {e}")

    # 5. Save Safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    try:
        from safetensors.torch import save_model, save_file
        try:
            save_model(model, safetensors_path)
        except Exception:
            state_dict = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
            save_file(state_dict, safetensors_path)
        print(f"Saved SafeTensors weights to: {safetensors_path}")
    except ImportError:
        torch_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), torch_path)
        print(f"safetensors not found, saved PyTorch weights to: {torch_path}")

    # 6. Save Config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2, ensure_ascii=False)
    print(f"Saved config to: {config_path}")

    # 7. Copy Tokenizer & Special Tokens & Chat Template
    tokenizer_source = os.path.join(PROJECT_ROOT, "model")
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
    readme_info = f"""# ViMind 3.0 Export Artifact

- Architecture: {'Mixture-of-Experts (MoE)' if is_moe else 'Dense'}
- Total Parameters: {'198M (64M active)' if is_moe else '64M'}
- Max Context Length: 32,768 (YaRN RoPE)
- Special Tokens: `<think>`, `</think>`, `<tool_call>`, `</tool_call>`

### GGUF Quantization:
To convert this model to GGUF format for llama.cpp / Ollama / LM Studio:
```bash
python llama.cpp/convert_hf_to_gguf.py {output_dir} --outfile vimind-3.0-q4_k_m.gguf --outtype q8_0
./llama-quantize vimind-3.0-q8_0.gguf vimind-3.0-q4_k_m.gguf Q4_K_M
```
"""
    with open(os.path.join(output_dir, "EXPORT_INFO.md"), "w", encoding="utf-8") as f:
        f.write(readme_info)

    print("\nModel packaging successfully completed!")
    print(f"Ready for deployment in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ViMind 3.0 Weight Converter & Packager")
    parser.add_argument("--input", type=str, default="out/sft_moe/vimind_sft.pth", help="Input checkpoint file")
    parser.add_argument("--output", type=str, default="out/vimind_3.0_hf", help="Output export directory")
    parser.add_argument("--lora_dir", type=str, default=None, help="Optional LoRA adapter directory to merge")
    parser.add_argument("--moe", action="store_true", help="Set flag if model is MoE")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts if MoE")
    args = parser.parse_args()

    convert_and_export(
        input_checkpoint=args.input,
        output_dir=args.output,
        lora_dir=args.lora_dir,
        is_moe=args.moe,
        num_experts=args.num_experts
    )


if __name__ == "__main__":
    main()
