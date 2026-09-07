import os
import sys
import unittest
import torch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import ViMindConfig, ViMindForCausalLM
from trainer.rollout_engine import compute_per_token_logps
from trainer.train_grpo import repetition_penalty_score, calculate_rewards


class TestGRPOPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.vocab_size = 500
        self.max_seq_len = 64

        self.cfg = ViMindConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=192,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
        )

        self.policy = ViMindForCausalLM(self.cfg).to(self.device)
        self.ref = ViMindForCausalLM(self.cfg).to(self.device)
        self.ref.eval()
        self.ref.requires_grad_(False)

    def test_repetition_penalty(self):
        """Test repetition detector on looping vs diverse text."""
        repeating_text = "Hà Nội là thủ đô của Việt Nam. Hà Nội là thủ đô của Việt Nam. Hà Nội là thủ đô của Việt Nam."
        diverse_text = "Việt Nam có thủ đô là Hà Nội, nằm ở đồng bằng sông Hồng với bề dày nghìn năm lịch sử."

        rep_score = repetition_penalty_score(repeating_text, n=3)
        div_score = repetition_penalty_score(diverse_text, n=3)

        self.assertGreater(rep_score, div_score)

    def test_calculate_rewards_and_cot(self):
        """Test format rewards for Chain-of-Thought <think> tags and ground-truth matching."""
        prompts = ["Thủ đô của Việt Nam là gì?"]
        completions = [
            "<think> Đây là câu hỏi địa lý cơ bản về Việt Nam. Thủ đô là Hà Nội. </think> Thủ đô của Việt Nam là Hà Nội.",
            "Hà Nội",  # short, no thinking
        ]
        gts = ["Hà Nội"]

        rewards = calculate_rewards(prompts, completions, gts, num_generations=2, device=self.device)
        self.assertEqual(rewards.shape, (2,))
        # CoT reasoning + length + ground truth should give higher reward to completion 0
        self.assertGreater(rewards[0].item(), rewards[1].item())

    def test_vectorized_per_token_logps(self):
        """Test vectorized log prob extraction for completion tokens."""
        batch_size = 2
        seq_len = 16
        n_keep = 6

        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        logps = compute_per_token_logps(self.policy, input_ids, n_keep, attention_mask=attention_mask)
        self.assertEqual(logps.shape, (batch_size, n_keep))
        # Log probabilities should be negative numbers
        self.assertTrue((logps <= 0.0).all())

    def test_grpo_objective_and_backward(self):
        """Test full GRPO loss computation and backward pass on policy parameters."""
        num_prompts = 2
        num_gen = 2
        total_samples = num_prompts * num_gen
        prompt_len = 8
        gen_len = 8
        total_len = prompt_len + gen_len

        outputs = torch.randint(0, self.vocab_size, (total_samples, total_len), dtype=torch.long)
        completion_ids = outputs[:, prompt_len:]
        full_mask = torch.ones_like(outputs)

        # 1. Compute policy & ref logps
        policy_logps = compute_per_token_logps(self.policy, outputs, gen_len, attention_mask=full_mask)
        with torch.no_grad():
            ref_logps = compute_per_token_logps(self.ref, outputs, gen_len, attention_mask=full_mask)

        old_logps = policy_logps.detach().clone()

        # 2. Rewards and advantages
        raw_rewards = torch.tensor([1.5, 0.5, -0.2, 1.0], device=self.device)
        grouped_rewards = raw_rewards.view(num_prompts, num_gen)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_gen)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_gen)
        advantages = (raw_rewards - mean_r) / (std_r + 1e-4)

        # 3. GRPO Loss
        ratio = torch.exp(policy_logps - old_logps)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        surr1 = ratio * advantages.unsqueeze(1)
        surr2 = clipped_ratio * advantages.unsqueeze(1)

        kl_div = ref_logps - policy_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1.0

        per_token_loss = -(torch.min(surr1, surr2) - 0.05 * per_token_kl)
        policy_loss = per_token_loss.mean()

        # 4. Backward
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-3)
        policy_loss.backward()

        # Verify gradients exist on policy and step works
        first_param = next(self.policy.parameters())
        self.assertIsNotNone(first_param.grad)
        optimizer.step()
        optimizer.zero_grad()

        # Verify ref parameters were never touched
        ref_first_param = next(self.ref.parameters())
        self.assertIsNone(ref_first_param.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
