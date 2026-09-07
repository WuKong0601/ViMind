import os
import sys
import unittest
import tempfile
import shutil
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import ViMindConfig, ViMindForCausalLM
from model.model_lora import apply_lora, mark_only_lora_as_trainable, save_lora, load_lora, merge_lora


class TestLoRAPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.vocab_size = 1000
        self.max_seq_len = 64

        self.cfg = ViMindConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=384,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
        )

        self.model = ViMindForCausalLM(self.cfg).to(self.device)

    def test_lora_initialization_identity(self):
        """At initialization, LoRA B=0 so output must exactly equal original model output."""
        input_ids = torch.randint(0, self.vocab_size, (2, 16), dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            orig_logits = self.model(input_ids).logits

        # Inject LoRA
        injected = apply_lora(self.model, rank=8, alpha=16.0)
        self.assertGreater(injected, 0)

        with torch.no_grad():
            lora_init_logits = self.model(input_ids).logits

        # Difference must be numerically zero
        self.assertTrue(torch.allclose(orig_logits, lora_init_logits, atol=1e-5))

    def test_trainable_parameters_isolation(self):
        """Only LoRA parameters should have gradients; base model weights must remain frozen."""
        apply_lora(self.model, rank=8, alpha=16.0)
        trainable_params = mark_only_lora_as_trainable(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        lora_params_count = sum(p.numel() for p in trainable_params)

        self.assertLess(lora_params_count, total_params)

        # Record initial weights of base linear layer and LoRA layer
        first_layer = self.model.model.layers[0].self_attn.q_proj
        base_weight_before = first_layer.weight.clone()
        lora_b_before = first_layer.lora.B.weight.clone()

        # Step optimizer
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-2)
        input_ids = torch.randint(0, self.vocab_size, (2, 16), dtype=torch.long)
        out = self.model(input_ids)
        loss = out.logits.sum()
        loss.backward()

        self.assertIsNone(first_layer.weight.grad)
        self.assertIsNotNone(first_layer.lora.B.weight.grad)

        optimizer.step()

        # Base weight must not have changed at all
        self.assertTrue(torch.equal(base_weight_before, first_layer.weight))
        # LoRA weight must have updated
        self.assertFalse(torch.equal(lora_b_before, first_layer.lora.B.weight))

    def test_lora_save_load_and_merge(self):
        """Test adapter saving, reload, and seamless weight merging into base model."""
        temp_dir = tempfile.mkdtemp()
        try:
            apply_lora(self.model, rank=8, alpha=16.0)
            trainable_params = mark_only_lora_as_trainable(self.model)

            # Perform a synthetic optimization step so LoRA weights are non-zero
            optimizer = torch.optim.AdamW(trainable_params, lr=1e-2)
            input_ids = torch.randint(0, self.vocab_size, (2, 16), dtype=torch.long)
            self.model(input_ids).logits.sum().backward()
            optimizer.step()

            # Save adapter
            adapter_path = os.path.join(temp_dir, "test_lora.pth")
            save_lora(self.model, adapter_path)
            self.assertTrue(os.path.exists(adapter_path))

            # Forward pass before merging
            self.model.eval()
            with torch.no_grad():
                unmerged_logits = self.model(input_ids).logits

            # Test loading into a new model
            new_model = ViMindForCausalLM(self.cfg).to(self.device)
            apply_lora(new_model, rank=8, alpha=16.0)
            # Copy base weights to match
            new_model.load_state_dict(self.model.state_dict(), strict=False)
            load_lora(new_model, adapter_path)

            new_model.eval()
            with torch.no_grad():
                loaded_logits = new_model(input_ids).logits
            self.assertTrue(torch.allclose(unmerged_logits, loaded_logits, atol=1e-5))

            # Merge LoRA into new_model
            merge_lora(new_model)

            # After merge, module should no longer have .lora attribute
            first_layer = new_model.model.layers[0].self_attn.q_proj
            self.assertFalse(hasattr(first_layer, "lora"))

            # Logits of merged model must match logits with LoRA active
            with torch.no_grad():
                merged_logits = new_model(input_ids).logits
            self.assertTrue(torch.allclose(unmerged_logits, merged_logits, atol=1e-4))
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
