import importlib.util
import unittest


class LocalInkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Torch is isolated")
        import torch

        cls.torch = torch
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(2)

    def test_distant_word_content_cannot_modify_local_output(self):
        from research.corpus.local_ink_model import LocalInk

        t = self.torch
        model = LocalInk().to(self.device)
        a = t.rand(1, 1, 80, 200, device=self.device)
        b = a.clone()
        b[:, :, :, :60] = 1 - b[:, :, :, :60]
        with t.inference_mode():
            x, y = model(a), model(b)
        self.assertTrue(t.equal(x[:, :, :, 100:], y[:, :, :, 100:]))
        self.assertFalse(t.equal(x[:, :, :, :50], y[:, :, :, :50]))

    def test_real_supervision_can_update_entire_local_predictor(self):
        from research.corpus.local_ink_model import LocalInk

        t = self.torch
        model = LocalInk().to(self.device)
        a = t.rand(2, 1, 64, 128, device=self.device)
        model(a).square().mean().backward()
        self.assertTrue(
            all(p.grad is not None and bool(t.isfinite(p.grad).all()) for p in model.parameters())
        )
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)
