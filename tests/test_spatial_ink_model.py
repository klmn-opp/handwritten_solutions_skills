import importlib.util
import unittest


class SpatialInkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Torch is isolated")
        import torch

        cls.torch = torch
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(2)

    def inputs(self):
        t = self.torch
        tokens = t.zeros(2, 32, dtype=t.long, device=self.device)
        tokens[:, :3] = t.tensor([97, 98, 99], device=self.device)
        positions = t.zeros(2, 32, 2, device=self.device)
        positions[:, :3, 0] = t.tensor([0.25, 0.5, 0.75], device=self.device)
        positions[:, :3, 1] = 0.12
        x = t.randn(2, 8, 16, 64, device=self.device)
        return x, x * 0.1, tokens, t.ones(2, device=self.device), positions

    def test_local_tokens_and_writer_have_gradients_without_global_word_mlp(self):
        from research.corpus.spatial_ink_model import SpatialInk

        model = SpatialInk().to(self.device)
        model.context.forward = lambda _: (_ for _ in ()).throw(
            AssertionError("global word shortcut")
        )
        inputs = self.inputs()
        out = model(*inputs)
        out.square().mean().backward()
        self.assertEqual(out.shape, inputs[0].shape)
        for parameter in [model.style, model.embedding.weight, model.enc[0].first.weight]:
            self.assertTrue(self.torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.context.parameters()))

    def test_local_order_and_reference_positions_change_field(self):
        from research.corpus.spatial_ink_model import SpatialInk

        model = SpatialInk().to(self.device)
        _, _, tokens, _, positions = self.inputs()
        a = model.local_field(tokens, positions)
        tokens2 = tokens.clone()
        tokens2[:, :3] = tokens2[:, :3].flip(1)
        self.assertFalse(self.torch.equal(a, model.local_field(tokens2, positions)))
        with self.assertRaises(ValueError):
            model.local_field(tokens * 0, positions)

    def test_warm_start_preserves_existing_convolutions_and_has_fixed_shared_style(self):
        from research.corpus.latent_ink_model import LatentInk
        from research.corpus.spatial_ink_model import SpatialInk

        base = LatentInk().to(self.device)
        model = SpatialInk().to(self.device)
        inputs = self.inputs()
        model.warm_start(base.state_dict(), inputs[2])
        t = self.torch
        self.assertTrue(t.equal(model.enc[0].first.weight[:, :16], base.enc[0].first.weight))
        self.assertEqual(int(t.count_nonzero(model.enc[0].first.weight[:, 16:])), 0)
        expected = base.context(base.embedding(inputs[2]).flatten(1)).mean(0)
        self.assertTrue(t.equal(model.style, expected))
        self.assertEqual(tuple(model.style.shape), (64,))
