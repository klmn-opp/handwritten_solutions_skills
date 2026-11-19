import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains optional")
class RealSupport(unittest.TestCase):
    def test_complete_variant_beats_blank_and_average(self):
        import torch

        from research.corpus.real_ink_support import nearest_real_loss

        observed = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
        valid = torch.tensor([6.0, 6.0, -6.0, -6.0], requires_grad=True)
        gray = torch.zeros(4)
        blank = torch.full((4,), -6.0)
        a = nearest_real_loss(valid, observed)
        self.assertLess(float(a.detach()), float(nearest_real_loss(gray, observed)))
        self.assertLess(float(a.detach()), float(nearest_real_loss(blank, observed)))
        a.backward()
        self.assertGreater(float(valid.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            nearest_real_loss(valid, observed[:0])
