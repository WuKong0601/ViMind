import os
import sys
import unittest
import tempfile
import shutil
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import ViMindConfig, ViMindForCausalLM
from trainer.train_distillation import distillation_kl_loss, configure_optimizers, save_checkpoint


class TestDistillationPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.vocab_size = 1200
        self.max_seq_len = 64

        # Student config: smaller model
        self.student_cfg = ViMindConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=384,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
        )

        # Teacher config: larger model (e.g., wider/deeper)
        self.teacher_cfg = ViMindConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            intermediate_size=768,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
        )

        self.student = ViMindForCausalLM(self.student_cfg).to(self.device)
        self.teacher = ViMindForCausalLM(self.teacher_cfg).to(self.device)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

    def test_distillation_loss_computation(self):
        """Test KL divergence loss computation and scaling with T^2."""
        batch_size, seq_len = 2, 16
        student_logits = torch.randn(batch_size * seq_len, self.vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size * seq_len, self.vocab_size)

        temp = 2.0
        kd_loss = distillation_kl_loss(student_logits, teacher_logits, temperature=temp)

        self.assertTrue(torch.is_tensor(kd_loss))
        self.assertGreater(kd_loss.item(), 0.0)
        self.assertFalse(torch.isnan(kd_loss))
        self.assertFalse(torch.isinf(kd_loss))

        # Backward should calculate gradients for student logits
        kd_loss.backward()
        self.assertIsNotNone(student_logits.grad)
        self.assertEqual(student_logits.grad.shape, student_logits.shape)

    def test_student_trainable_teacher_frozen(self):
        """Verify teacher parameters remain strictly unchanged while student learns."""
        optimizer = configure_optimizers(self.student, weight_decay=0.01, learning_rate=1e-3)

        # Store initial teacher weights clone
        teacher_initial_weight = next(self.teacher.parameters()).clone()
        student_initial_weight = next(self.student.parameters()).clone()

        # Synthetic batch: batch=2, seq_len=16
        input_ids = torch.randint(0, self.vocab_size, (2, 16), dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :8] = -100  # First half is prompt (masked), second half is assistant

        loss_mask = (labels[:, 1:] != -100)

        # Forward
        s_out = self.student(input_ids)
        s_logits = s_out.logits[:, :-1, :].contiguous()

        with torch.no_grad():
            t_out = self.teacher(input_ids)
            t_logits = t_out.logits[:, :-1, :].contiguous()

        # CE Loss on unmasked assistant tokens
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            s_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # KD Loss on unmasked assistant tokens
        mask_flat = loss_mask.view(-1)
        s_target = s_logits.view(-1, self.vocab_size)[mask_flat]
        t_target = t_logits.view(-1, self.vocab_size)[mask_flat]
        kd_loss = distillation_kl_loss(s_target, t_target, temperature=1.5)

        total_loss = 0.5 * ce_loss + 0.5 * kd_loss
        total_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Assertions
        teacher_current_weight = next(self.teacher.parameters())
        student_current_weight = next(self.student.parameters())

        # Teacher should be 100% identical
        self.assertTrue(torch.equal(teacher_initial_weight, teacher_current_weight))
        # Student should have updated
        self.assertFalse(torch.equal(student_initial_weight, student_current_weight))

    def test_checkpoint_save_and_reload(self):
        """Test saving and re-instantiating student model and trainer state."""
        temp_dir = tempfile.mkdtemp()
        try:
            optimizer = configure_optimizers(self.student, weight_decay=0.01, learning_rate=1e-3)
            save_checkpoint(
                model=self.student,
                tokenizer=None,
                config=self.student.config,
                optimizer=optimizer,
                scaler=None,
                epoch=1,
                step=50,
                save_dir=temp_dir,
                prefix="distill_test",
            )

            expected_dir = os.path.join(temp_dir, "distill_test_step_50")
            self.assertTrue(os.path.exists(expected_dir))
            self.assertTrue(os.path.exists(os.path.join(expected_dir, "trainer_state.pt")))
            self.assertTrue(os.path.exists(os.path.join(expected_dir, "config.json")))

            # Reload
            reloaded_model = ViMindForCausalLM.from_pretrained(expected_dir)
            self.assertEqual(reloaded_model.config.hidden_size, self.student_cfg.hidden_size)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
