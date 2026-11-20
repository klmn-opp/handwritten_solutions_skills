import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains isolated")
class HierarchicalInkStyleTest(unittest.TestCase):
    def test_semantic_context_shared_and_local_latents_affect_generated_conditions(self):
        import torch

        from research.corpus.hierarchical_ink_style import HierarchicalInkStyle

        torch.manual_seed(4622)
        model = HierarchicalInkStyle()
        t = torch.tensor([[[102, 0, 111], [111, 102, 114], [114, 111, 0], [0, 0, 0]]])
        w, z = torch.zeros(1, 8), torch.zeros(1, 4, 4)
        base = model(t, w, z)
        self.assertEqual(float(base[:, -1].abs().sum()), 0)
        self.assertGreater(float((model(t, w + 1, z) - base)[:, :3].abs().min()), 0)
        changed = z.clone()
        changed[:, 1] = 1
        output = model(t, w, changed)
        self.assertTrue(torch.equal(output[:, 0], base[:, 0]))
        self.assertGreater(float((output[:, 1] - base[:, 1]).abs().sum()), 0)
        changed = t.clone()
        changed[:, 1, 1] = 97
        self.assertFalse(torch.equal(model(changed, w, z), base))
        model.context = False
        self.assertTrue(torch.equal(model(changed, w, z), model(t, w, z)))

    def test_unverified_positions_do_not_enter_word_posterior_and_image_gradient_flows(self):
        import torch

        from research.corpus.canonical_ink_model import CanonicalInk
        from research.corpus.hierarchical_ink_style import HierarchicalInkStyle

        torch.manual_seed(4622)
        model = HierarchicalInkStyle()
        ink = CanonicalInk().requires_grad_(False)
        t = torch.tensor([[[102, 0, 111], [111, 102, 114], [114, 111, 0]]])
        codes = torch.randn(1, 3, 8)
        mask = torch.tensor([[True, False, True]])
        p = model.posterior(t, codes, mask)
        codes[:, 1] += 100
        changed = model.posterior(t, codes, mask)
        self.assertTrue(torch.equal(p[0], changed[0]))
        self.assertTrue(torch.equal(p[1], changed[1]))
        prediction = model(t, p[0], p[2])
        loss = ink(t[..., 0][mask], prediction[mask]).sigmoid().mean()
        loss.backward()
        self.assertGreater(float(model.decode[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(v.grad is None for v in ink.parameters()))
