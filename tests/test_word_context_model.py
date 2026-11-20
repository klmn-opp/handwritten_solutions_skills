import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains isolated")
class WordContextTest(unittest.TestCase):
    def test_identity_endpoints_bounds_and_trainable_update(self):
        import torch

        from research.corpus.word_context_model import WordContextInk

        torch.manual_seed(4522)
        model = WordContextInk()
        base = torch.rand(2, 1, 24, 40)
        base[..., 0, 0] = 0
        base[..., 1, 1] = 1
        self.assertTrue(torch.equal(model(base), base))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = (model(base) - (base > 0.5).float()).square().mean()
        loss.backward()
        self.assertGreater(float(model.field[-1].weight.grad.abs().sum()), 0)
        optimizer.step()
        changed = model(base)
        self.assertFalse(torch.equal(changed, base))
        self.assertEqual(float(changed[..., 0, 0].sum()), 0)
        self.assertEqual(float(changed[..., 1, 1].sum()), 2)
        self.assertTrue(bool(((changed >= 0) & (changed <= 1)).all()))
        with torch.no_grad():
            model.field[-1].bias.fill_(-100)
        self.assertTrue(bool(((model(base) >= -1e-6) & (model(base) <= 1)).all()))

    def test_context_effect_tiling_and_finite_receptive_field(self):
        import torch

        from research.corpus.word_context_model import WordContextInk

        torch.manual_seed(4522)
        model = WordContextInk().eval()
        with torch.no_grad():
            model.field[-1].weight.normal_(std=0.1)
        base = torch.rand(1, 1, 24, 100) * 0.8 + 0.1
        whole = model.image(base, tile_width=100)
        tiled = model.image(base, tile_width=19)
        self.assertTrue(torch.allclose(whole, tiled, atol=2e-6, rtol=2e-5))
        changed = base.clone()
        changed[..., 9, 51] = 1
        output = model.image(changed)
        self.assertGreater(float((output[..., 9, 50] - whole[..., 9, 50]).abs()), 1e-7)
        self.assertTrue(torch.equal(output[..., :45], whole[..., :45]))
