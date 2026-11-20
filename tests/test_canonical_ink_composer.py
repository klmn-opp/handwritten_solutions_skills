import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains isolated")
class CanonicalCompositionTest(unittest.TestCase):
    def test_real_decoder_tiling_and_padding_do_not_change_output(self):
        import torch

        from research.corpus.canonical_ink_composer import compose
        from research.corpus.canonical_ink_model import CanonicalInk

        torch.manual_seed(4210)
        model = CanonicalInk().eval()
        t = torch.tensor([[102, 0, 0], [116, 0, 0], [0, 0, 0]])
        b = torch.tensor([[16, 10, 40, 60], [58, 10, 82, 60], [0, 0, 0, 0]]).float()
        style = torch.randn(3, 8)
        full = compose(model, t, b, style, 80, 128, tile_width=128)
        tiled = compose(model, t, b, style, 80, 128, tile_width=31)
        unpadded = compose(model, t[:2], b[:2], style[:2], 80, 128)
        self.assertTrue(torch.equal(full, tiled))
        self.assertTrue(torch.equal(full, unpadded))
        self.assertEqual(tuple(full.shape), (80, 128))
        self.assertEqual(float(full[:, 120:].abs().sum()), 0)
        b[0, 2] = b[0, 0]
        with self.assertRaises(ValueError):
            compose(model, t, b, style, 80, 128)
