import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains isolated")
class SemanticCanvas(unittest.TestCase):
    def test_semantics_style_and_shared_writer_learn_without_reference_pixels(self):
        import torch

        from research.corpus.semantic_canvas_model import SemanticCanvasInk

        torch.manual_seed(4020)
        model = SemanticCanvasInk()
        with torch.no_grad():
            model.local.weight.normal_(std=0.02)
        t = torch.tensor([[[102, 0, 0], [116, 0, 0]]])
        b = torch.tensor([[[4, 4, 18, 28], [22, 4, 36, 28]]]).float()
        style = torch.randn(1, 2, 8)
        image = model(t, b, 32, 40, style=style)
        image.square().mean().backward()
        for parameter in [
            model.character.weight,
            model.local.weight,
            model.writer,
            model.blocks[0][0].weight,
        ]:
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        changed = model(t, b, 32, 40, style=style + 1)
        self.assertGreater(float((image.detach() - changed.detach()).abs().max()), 1e-5)

    def test_long_canvas_tiling_preserves_context_across_seams(self):
        import torch

        from research.corpus.semantic_canvas_model import SemanticCanvasInk

        torch.manual_seed(4020)
        model = SemanticCanvasInk().eval()
        t = torch.tensor([[[102, 0, 0], [116, 0, 0], [110, 0, 0]]])
        b = torch.tensor([[[8, 4, 25, 25], [36, 4, 50, 25], [63, 4, 82, 25]]]).float()
        style = torch.randn(1, 3, 8)
        whole = model.image(t, b, 32, 96, style=style, tile_width=96)
        tiled = model.image(t, b, 32, 96, style=style, tile_width=24)
        self.assertTrue(torch.allclose(whole, tiled, atol=2e-6, rtol=2e-5))

    def test_nearby_character_can_influence_ink_without_distant_word_coupling(self):
        import torch

        from research.corpus.semantic_canvas_model import SemanticCanvasInk

        torch.manual_seed(4020)
        model = SemanticCanvasInk().eval()
        t = torch.tensor([[[102, 0, 0], [116, 0, 0], [110, 0, 0]]])
        b = torch.tensor([[[8, 4, 25, 25], [32, 4, 50, 25], [145, 4, 165, 25]]]).float()
        base = model.image(t, b, 32, 180)
        near = t.clone()
        near[0, 1, 0] = 105
        edited = model.image(near, b, 32, 180)
        self.assertGreater(float((base[:, :, 8:25] - edited[:, :, 8:25]).abs().max()), 1e-6)
        far = t.clone()
        far[0, 2, 0] = 105
        edited_far = model.image(far, b, 32, 180)
        self.assertTrue(torch.equal(base[:, :, :50], edited_far[:, :, :50]))
