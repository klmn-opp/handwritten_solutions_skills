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
    def test_local_diagnostic_is_invariant_to_neighbours_but_learns_morphology(self):
        import torch

        from research.corpus.semantic_ink_model import ContextIndependentLocalInk

        model = ContextIndependentLocalInk()
        with torch.no_grad():
            model.context_gate.fill_(1)
        t = torch.tensor([[[102, 0, 0], [0, 0, 0]]])
        changed = t.clone()
        changed[0, 0, 1:] = torch.tensor([110, 105])
        b = torch.tensor([[[16, 16, 80, 80], [0, 0, 0, 0]]]).float()
        points = torch.tensor([[[25, 34], [45, 65]]]).float()
        style = torch.ones(1, 2, 10)
        before = model(t, b, points, style=style)[0]
        self.assertTrue(torch.equal(before, model(changed, b, points, style=style)[0]))
        before.sum().backward()
        self.assertIsNone(model.neighbour.weight.grad)
        self.assertIsNone(model.context_gate.grad)
        self.assertGreater(float(model.local_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.field_modulation.weight.grad.abs().sum()), 0)

    def test_zero_modulation_preserves_field_and_all_hidden_layers_receive_gradients(self):
        import torch

        from research.corpus.semantic_ink_model import HierarchicalSemanticInk, ModulatedSemanticInk

        base = HierarchicalSemanticInk()
        model = ModulatedSemanticInk()
        result = model.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(
            set(result.missing_keys), {"field_modulation.weight", "field_modulation.bias"}
        )
        t = torch.tensor([[[102, 0, 0], [0, 0, 0]]])
        b = torch.tensor([[[16, 16, 80, 80], [0, 0, 0, 0]]]).float()
        points = torch.tensor([[[25, 34], [45, 65]]]).float()
        style = torch.ones(1, 2, 10)
        self.assertTrue(
            torch.equal(model(t, b, points, style=style)[0], base(t, b, points, style=style)[0])
        )
        model(t, b, points, style=style)[0].sum().backward()
        gradients = model.field_modulation.weight.grad.reshape(3, 2, 128, 10)
        for j in range(3):
            self.assertGreater(float(gradients[j].abs().sum()), 0)

    def test_local_code_changes_its_own_field_and_retains_shared_word_gradient(self):
        import torch

        from research.corpus.semantic_ink_model import HierarchicalSemanticInk, SemanticInk

        model = HierarchicalSemanticInk()
        model.load_identity(SemanticInk(context_scale=0).state_dict())
        t = torch.tensor([[[97, 0, 98], [98, 97, 0]]])
        b = torch.tensor([[[10, 10, 30, 40], [100, 10, 120, 40]]]).float()
        points = torch.tensor([[[20, 20], [110, 20]]]).float()
        style = torch.zeros(1, 2, 10)
        first = model(t, b, points, style=style)[0]
        style[0, 0, 2:] = 1
        second = model(t, b, points, style=style)[0]
        self.assertNotEqual(float(first[0, 0].detach()), float(second[0, 0].detach()))
        self.assertEqual(float(first[0, 1].detach()), float(second[0, 1].detach()))
        style[:, :, :2] = 1
        model(t, b, points, style=style)[0].sum().backward()
        self.assertGreater(float(model.local_projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.style_projection.weight.grad.abs().sum()), 0)

    def test_coupled_distance_head_has_one_contour_and_a_learning_gradient(self):
        import torch

        from research.corpus.semantic_ink_model import CoupledDistanceInk

        model = CoupledDistanceInk()
        t = torch.tensor([[[112, 0, 0]]])
        b = torch.tensor([[[10, 10, 30, 40]]]).float()
        points = torch.tensor([[[20, 20], [150, 100]]]).float()
        logits, sdf = model(t, b, points, style=torch.zeros(1, 2))
        self.assertTrue(torch.equal(logits, 16 * sdf))
        self.assertEqual(float(sdf[0, 1].detach()), -1)
        logits[0, 0].backward()
        self.assertGreater(float(model.field[-1].weight.grad[1].abs().sum()), 0)
        self.assertEqual(float(model.field[-1].weight.grad[0].abs().sum()), 0)

    def test_metric_conditioning_exposes_scale_without_reference_pixels(self):
        import torch

        from research.corpus.semantic_ink_model import MetricSemanticInk, SemanticInk

        base = SemanticInk(context_scale=0)
        model = MetricSemanticInk()
        model.load_identity(base.state_dict())
        tokens = torch.tensor([[[112, 0, 0]]])
        boxes = torch.tensor([[[10, 10, 30, 40]]]).float()
        points = torch.tensor([[[18, 20]]]).float()
        a = model(tokens, boxes, points)[0]
        self.assertTrue(torch.equal(a, base(tokens, boxes, points)[0]))
        a.sum().backward()
        self.assertGreater(float(model.metric_projection.weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.metric_projection.weight.fill_(0.1)
        small = model(tokens, boxes, points)[0]
        large = model(tokens, boxes * 2, points * 2)[0]
        self.assertFalse(torch.equal(small, large))

    def test_optimized_training_codes_are_not_required_by_decoder_at_generation(self):
        import torch

        from research.corpus.semantic_ink_model import AutoDecodedSemanticInk, SemanticInk

        device = "cuda" if torch.cuda.is_available() else "cpu"
        base = SemanticInk(context_scale=0).to(device)
        model = AutoDecodedSemanticInk().to(device)
        model.load_identity(base.state_dict())
        codes = torch.nn.Parameter(torch.randn(2, 2, device=device))
        tokens = torch.tensor([[[102, 0, 0]], [[102, 0, 0]]], device=device)
        boxes = torch.tensor([[[10, 10, 30, 40]], [[10, 10, 30, 40]]], device=device).float()
        points = torch.tensor([[[20, 22]], [[20, 22]]], device=device).float()
        initial = model(tokens, boxes, points, style=codes)[0]
        self.assertTrue(torch.equal(initial, base(tokens, boxes, points)[0]))
        opt = torch.optim.Adam([*model.parameters(), codes], lr=0.01)
        old = codes.detach().clone()
        for _ in range(3):
            value = model(tokens, boxes, points, style=codes)[0]
            loss = (value - torch.tensor([[1.0], [-1.0]], device=device)).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertGreater(float((codes - old).abs().sum().detach()), 0)
        self.assertFalse(any("code" in k for k in model.state_dict()))
        z = torch.randn(2, 2, device=device)
        a = model(tokens, boxes, points, style=z)[0]
        b = model(tokens, boxes, points, style=-z)[0]
        self.assertFalse(torch.equal(a, b))

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

    def test_spatial_word_latent_starts_exactly_and_learns_coordinate_changes(self):
        import torch

        from research.corpus.semantic_ink_model import SemanticInk, SpatialWordLatentInk

        base = SemanticInk(context_scale=0)
        model = SpatialWordLatentInk()
        model.load_identity(base.state_dict())
        tokens = torch.tensor([[[102, 97, 98]]])
        boxes = torch.tensor([[[10, 10, 30, 40]]]).float()
        points = torch.tensor([[[18, 21], [22, 32]]]).float()
        z = torch.randn(1, 8)
        value = model(tokens, boxes, points, style=z)[0]
        self.assertTrue(torch.equal(base(tokens, boxes, points)[0], value))
        value.sum().backward()
        self.assertGreater(float(model.geometry[-1].weight.grad.abs().sum()), 0)
