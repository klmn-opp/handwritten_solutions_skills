import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains optional")
class AlignmentModel(unittest.TestCase):
    def test_coordinates_start_exactly_and_preserve_positive_dimensions(self):
        import torch

        from research.corpus.whole_word_alignment import adjusted_boxes, geometry_penalty

        base = torch.tensor([[[10.0, 10.0, 20.0, 30.0], [25.0, 10.0, 40.0, 45.0]]])
        d = torch.zeros_like(base, requires_grad=True)
        original = adjusted_boxes(base, d)
        self.assertTrue(torch.equal(original, base))
        original.sum().backward()
        self.assertGreater(float(d.grad.abs().sum()), 0)
        boxes = adjusted_boxes(base, d + 100)
        self.assertTrue(bool((boxes[..., 2:] > boxes[..., :2]).all()))
        t = torch.tensor([[[97, 0, 98], [98, 97, 0]]])
        good = geometry_penalty(base, t, d)
        bad = geometry_penalty(base.flip(1), t, d)
        self.assertGreater(float(bad.detach()), float(good.detach()))
