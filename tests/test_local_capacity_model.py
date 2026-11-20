import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains optional")
class LocalCapacity(unittest.TestCase):
    def test_widening_preserves_function_and_shared_conditions(self):
        import torch

        from research.corpus.local_capacity_model import widen_local_field
        from research.corpus.semantic_ink_model import ContextIndependentLocalInk

        torch.manual_seed(3719)
        base = ContextIndependentLocalInk()
        with torch.no_grad():
            base.field_modulation.weight.normal_(std=0.02)
        wide = widen_local_field(base)
        t = torch.tensor([[[102, 0, 0], [116, 0, 0]]])
        b = torch.tensor([[[8, 8, 48, 70], [60, 8, 90, 70]]]).float()
        points = torch.rand(1, 24, 2) * 90
        style = torch.randn(1, 2, 10)
        expected = base(t, b, points, style=style)
        actual = wide(t, b, points, style=style)
        for x, y in zip(expected, actual, strict=True):
            self.assertTrue(torch.allclose(x, y, atol=2e-6, rtol=2e-5))
        self.assertTrue(torch.equal(base.writer, wide.writer))
        self.assertTrue(torch.equal(base.character.weight, wide.character.weight))
        self.assertGreater(
            sum(p.numel() for p in wide.parameters()), 3 * sum(p.numel() for p in base.parameters())
        )

    def test_widened_units_can_receive_distinct_learning_gradients(self):
        import torch

        from research.corpus.local_capacity_model import widen_local_field
        from research.corpus.semantic_ink_model import ContextIndependentLocalInk

        torch.manual_seed(3719)
        model = widen_local_field(ContextIndependentLocalInk())
        t = torch.tensor([[[102, 0, 0]]])
        b = torch.tensor([[[16, 16, 80, 80]]]).float()
        points = torch.tensor([[[25.0, 34.0], [45.0, 65.0]]])
        model(t, b, points, style=torch.ones(1, 1, 10))[0].sum().backward()
        gradient = model.field[0].weight.grad
        self.assertGreater(float((gradient[0::2] - gradient[1::2]).abs().sum()), 0)
