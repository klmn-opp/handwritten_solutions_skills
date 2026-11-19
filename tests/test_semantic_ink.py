import importlib.util
import unittest

import numpy as np

from research.corpus.semantic_ink_data import conditions, pad_conditions


class SemanticGeometry(unittest.TestCase):
    def test_preserves_descenders_and_whitespace_context(self):
        trace = {
            "crop_from_pdf": [10, 20, 70, 80],
            "glyph_provenance": [
                {"semantic": c, "bounds_px": b}
                for c, b in zip(
                    "apy", [[12, 30, 20, 40], [22, 30, 30, 47], [40, 30, 50, 48]], strict=True
                )
            ],
        }
        t, b = conditions(
            "ap y", trace, {"scale": 2, "bbox": [2, 3, 50, 60], "translation": [5, 7]}
        )
        self.assertEqual(t.tolist(), [[97, 0, 112], [112, 97, 0], [121, 0, 0]])
        self.assertEqual((b[:, 3] - b[:, 1]).tolist(), [20, 34, 36])
        self.assertEqual(b[0].tolist(), [5, 21, 21, 41])
        padded, _ = pad_conditions([(t[:1], b[:1]), (t, b)])
        self.assertTrue(np.all(padded[0, 1:] == 0))
        with self.assertRaises(ValueError):
            conditions("apyx", trace, {"scale": 1})


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains optional")
class SemanticLearning(unittest.TestCase):
    def test_content_and_context_are_learnable_without_any_reference_pixels(self):
        import torch

        from research.corpus.semantic_ink_model import SemanticInk

        torch.set_num_threads(2)
        torch.manual_seed(15)
        model = SemanticInk()
        tokens = torch.tensor([[[97, 0, 98], [98, 97, 0]]])
        boxes = torch.tensor([[[10, 10, 30, 40], [35, 10, 55, 40]]]).float()
        points = torch.tensor([[[20, 20], [40, 20], [100, 90]]]).float()
        a = model(tokens, boxes, points)[0]
        changed = tokens.clone()
        changed[0, 0, 0] = 99
        b = model(changed, boxes, points)[0]
        self.assertFalse(torch.equal(a, b))
        self.assertEqual(float(a[0, -1]), -15)
        a.sum().backward()
        for param in [model.character.weight, model.neighbour.weight, model.writer]:
            self.assertGreater(float(param.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            model(torch.rand(1, 1, 128, 512), boxes, points)

    def test_chunking_and_padding_do_not_change_generated_text(self):
        import torch

        from research.corpus.semantic_ink_model import SemanticInk

        torch.set_num_threads(2)
        model = SemanticInk().eval()
        t = torch.tensor([[[112, 0, 0]]])
        b = torch.tensor([[[10, 8, 24, 30]]]).float()
        a = model.image(t, b, 40, 50, 2000)
        padded_t = torch.cat([t, torch.zeros_like(t)], 1)
        padded_b = torch.cat([b, torch.zeros_like(b)], 1)
        c = model.image(padded_t, padded_b, 40, 50, 137)
        self.assertTrue(torch.allclose(a, c, atol=2e-7, rtol=1e-5))

    def test_identity_capacity_disables_only_neighbour_effect(self):
        import torch

        from research.corpus.semantic_ink_model import SemanticInk

        model = SemanticInk(context_scale=0)
        t = torch.tensor([[[102, 97, 98]]])
        boxes = torch.tensor([[[10, 10, 25, 45]]]).float()
        points = torch.tensor([[[15, 20]]]).float()
        a = model(t, boxes, points)[0]
        changed = t.clone()
        changed[..., 1:] = 99
        self.assertTrue(torch.equal(a, model(changed, boxes, points)[0]))
        changed[..., 0] = 100
        self.assertFalse(torch.equal(a, model(changed, boxes, points)[0]))

    def test_real_context_starts_at_exact_identity_and_has_a_learning_gradient(self):
        import torch

        from research.corpus.semantic_ink_model import ContextualSemanticInk, SemanticInk

        base = SemanticInk(context_scale=0)
        model = ContextualSemanticInk()
        model.load_identity(base.state_dict())
        tokens = torch.tensor([[[102, 97, 98], [98, 102, 0]]])
        boxes = torch.tensor([[[10, 10, 30, 40], [35, 10, 55, 40]]]).float()
        points = torch.tensor([[[20, 20], [40, 20]]]).float()
        result = model(tokens, boxes, points)[0]
        self.assertTrue(torch.equal(base(tokens, boxes, points)[0], result))
        result.sum().backward()
        self.assertGreater(float(model.context_gate.grad.abs()), 0)
        with self.assertRaises(ValueError):
            model.load_identity({})

    def test_word_latent_has_separate_posterior_and_image_free_generation(self):
        import torch

        from research.corpus.semantic_ink_model import SemanticInk, WordLatentSemanticInk

        device = "cuda" if torch.cuda.is_available() else "cpu"
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        self.addCleanup(torch.use_deterministic_algorithms, previous)
        base = SemanticInk(context_scale=0).to(device)
        model = WordLatentSemanticInk().to(device)
        model.load_identity(base.state_dict())
        tokens = torch.tensor([[[102, 0, 0]]], device=device)
        boxes = torch.tensor([[[10, 10, 30, 40]]], device=device).float()
        points = torch.tensor([[[20, 20]]], device=device).float()
        z = torch.randn(1, 8, device=device)
        initial = model(tokens, boxes, points, style=z)[0]
        self.assertTrue(torch.equal(base(tokens, boxes, points)[0], initial))
        initial.sum().backward()
        self.assertGreater(float(model.style_projection.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.style_projection.weight.normal_(0, 0.2)
        self.assertFalse(
            torch.equal(
                model(tokens, boxes, points, style=z)[0],
                model(tokens, boxes, points, style=-z)[0],
            )
        )
        with self.assertRaises(ValueError):
            model(tokens, boxes, points)
        mu, lv = model.posterior(
            torch.rand(1, 1, 128, 512, device=device), torch.rand(1, 1, 128, 512, device=device)
        )
        self.assertEqual(tuple(mu.shape), (1, 8))
        (mu.square().mean() + lv.square().mean()).backward()
        self.assertTrue(all(p.grad is not None for p in model.posterior_encoder.parameters()))
