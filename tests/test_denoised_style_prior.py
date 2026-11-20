import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains isolated")
class DenoisedStyleTest(unittest.TestCase):
    def test_denoising_changes_with_noise_and_respects_frozen_basis(self):
        import torch

        from research.corpus.denoised_style_prior import DenoisedStyle

        model = DenoisedStyle()
        tokens = torch.tensor([[102, 0, 114], [113, 0, 0]])
        noise = torch.ones(2, 8)
        support = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
        loss = model(tokens, noise, torch.ones(2, 1), support).square().sum()
        loss.backward()
        self.assertTrue(bool(model.network[-1].weight.grad.abs().sum() > 0))
        sampled = model.generate(tokens, noise, support)
        self.assertTrue(torch.equal(sampled, model.generate(tokens, noise, support)))
        self.assertTrue(torch.equal(sampled * support, sampled))
        self.assertFalse(torch.equal(sampled, model.generate(tokens, -noise, support)))
