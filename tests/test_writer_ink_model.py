"""Optional GPU contracts; synthetic inputs are not training-quality evidence."""

import importlib.util
import unittest


class WriterInkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Torch belongs to isolated research")
        import torch

        cls.t = torch
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(2)

    def inputs(self):
        t = self.t
        t.manual_seed(2313)
        x = t.randn(1, 8, 16, 64, device=self.device).expand(2, -1, -1, -1).clone()
        ref = t.zeros_like(x)
        tokens = t.zeros(2, 32, dtype=t.long, device=self.device)
        tokens[:, :2] = t.tensor([106, 103], device=self.device)
        pos = t.zeros(2, 32, 2, device=self.device)
        pos[:, :2] = t.tensor([[0.45, 0.04], [0.55, 0.08]], device=self.device)
        return x, ref, tokens, t.ones(2, device=self.device), pos

    def test_shared_writer_and_attention_actually_receive_gradients(self):
        from research.corpus.writer_style_model import WriterInk

        t = self.t
        model = WriterInk().to(self.device)
        out = model(*self.inputs())
        out.square().mean().backward()
        for p in [
            model.writer,
            model.writer_projection.weight,
            model.text_attention.in_proj_weight,
        ]:
            self.assertTrue(bool(t.isfinite(p.grad).all()))
            self.assertGreater(float(p.grad.abs().sum()), 0)
        self.assertEqual(tuple(model.writer.shape), (32,))

    def test_fixed_personal_vector_is_shared_and_has_switchable_effect(self):
        from research.corpus.writer_style_model import WriterInk

        t = self.t
        model = WriterInk().to(self.device).eval()
        args = self.inputs()
        with t.inference_mode():
            active = model(*args)
            no_writer = model(*args, writer_enabled=False)
            no_attention = model(*args, text_enabled=False)
        self.assertTrue(t.equal(active[0], active[1]))
        self.assertGreater(float((active - no_writer).abs().max()), 0)
        self.assertGreater(float((active - no_attention).abs().max()), 0)

    def test_empty_text_is_rejected_before_masked_attention(self):
        from research.corpus.writer_style_model import WriterInk

        model = WriterInk().to(self.device)
        args = list(self.inputs())
        args[2].zero_()
        with self.assertRaises(ValueError):
            model(*args)


if __name__ == "__main__":
    unittest.main()
