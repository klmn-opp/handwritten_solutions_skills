import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains isolated")
class CanonicalInkTest(unittest.TestCase):
    def test_shape_condition_and_writer_gradients_without_input_images(self):
        import torch

        from research.corpus.canonical_ink_model import CanonicalInk

        torch.manual_seed(4210)
        model = CanonicalInk()
        characters = torch.tensor([102, 116])
        local = torch.randn(2, 8)
        logits = model(characters, local)
        self.assertEqual(tuple(logits.shape), (2, 96, 96))
        logits.square().mean().backward()
        for p in [
            model.character.weight,
            model.writer,
            model.expand[0].weight,
            *[m.weight for m in model.modulation],
        ]:
            self.assertGreater(float(p.grad.abs().sum()), 0)
        with torch.no_grad():
            changed = model(characters, local + 1)
        self.assertGreater(float((changed - logits.detach()).abs().max()), 1e-6)

    def test_rejects_missing_identity_or_wrong_condition_shape(self):
        import torch

        from research.corpus.canonical_ink_model import CanonicalInk

        model = CanonicalInk()
        with self.assertRaises(ValueError):
            model(torch.tensor([0]), torch.zeros(1, 8))
        with self.assertRaises(ValueError):
            model(torch.tensor([102]), torch.zeros(1, 10))
