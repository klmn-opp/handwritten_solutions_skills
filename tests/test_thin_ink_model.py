"""Optional isolated Torch contracts, not quality or real-training evidence."""

import importlib.util
import unittest


class ThinInkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Torch is isolated")
        import torch

        cls.t = torch
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(2)

    def test_zero_local_projection_preserves_old_output_and_can_learn(self):
        from research.corpus.latent_ink_model import LatentInk
        from research.corpus.thin_ink_model import LocalLatentInk

        t = self.t
        t.manual_seed(2112)
        old = LatentInk().to(self.device).eval()
        new = LocalLatentInk(local=True).to(self.device).eval()
        new.load_state_dict(old.state_dict(), strict=False)
        x = t.randn(1, 8, 16, 64, device=self.device)
        reference = t.randn_like(x)
        tokens = t.zeros(1, 32, dtype=t.long, device=self.device)
        tokens[:, :2] = t.tensor([106, 103], device=self.device)
        time = t.ones(1, device=self.device)
        positions = t.zeros(1, 32, 2, device=self.device)
        positions[:, :2] = t.tensor([[0.3, 0.1], [0.6, 0.2]], device=self.device)
        predicted = new(x, reference, tokens, time, positions)
        self.assertTrue(t.equal(predicted, old(x, reference, tokens, time)))
        predicted.square().mean().backward()
        self.assertGreater(float(new.local_projection.weight.grad.abs().sum()), 0)
        self.assertTrue(bool(t.isfinite(new.local_projection.weight.grad).all()))

    def test_detail_loss_penalizes_missing_dot_and_extra_connector(self):
        from research.corpus.thin_ink_model import ink_detail_loss

        t = self.t
        target = t.zeros(1, 1, 128, 512, device=self.device)
        target[:, :, 20:24, 20:24] = 1
        target[:, :, 40:95, 20:24] = 1
        weights = target * 2
        correct = target * 20 - 10
        missing = correct.clone()
        missing[:, :, 20:24, 20:24] = -10
        bridge = correct.clone()
        bridge[:, :, 24:40, 20:24] = 10
        loss = ink_detail_loss(correct, target, weights)
        self.assertLess(loss, ink_detail_loss(missing, target, weights))
        self.assertLess(loss, ink_detail_loss(bridge, target, weights))
        with self.assertRaises(ValueError):
            ink_detail_loss(correct, target, weights[..., :-1])


if __name__ == "__main__":
    unittest.main()
