"""Optional neural contracts; synthetic fixtures are not handwriting-quality evidence."""

import importlib.util
import unittest


class LatentInkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Torch belongs to the isolated research environment")
        import torch

        cls.torch = torch
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(2)

    def test_missing_ink_and_background_haze_are_both_penalized(self):
        from research.corpus.latent_ink_model import reconstruction_loss

        t = self.torch
        target = t.zeros(1, 1, 128, 512, device=self.device)
        target[:, :, 20:22, 40:42] = 1  # Dot.
        target[:, :, 40:115, 40:42] = 1  # Thin body and descender.
        correct = target * 20 - 10
        good = reconstruction_loss(correct, target)
        self.assertLess(good, reconstruction_loss(t.full_like(correct, -10), target))
        self.assertLess(good, reconstruction_loss(correct.clamp(min=-1), target))

    def test_frozen_decoder_transmits_gradient_only_to_generator(self):
        from research.corpus.latent_ink_model import InkCodec, LatentInk, reconstruction_loss

        t = self.torch
        t.manual_seed(2011)
        codec = InkCodec().to(self.device).requires_grad_(False)
        model = LatentInk().to(self.device)
        target = t.rand(1, 1, 128, 512, device=self.device)
        reference = codec.encode(target).detach()
        noise = t.randn_like(reference)
        pred = model(
            noise,
            reference,
            t.zeros(1, 32, dtype=t.long, device=self.device),
            t.ones(1, device=self.device),
        )
        reconstruction_loss(codec.decode(pred), target).backward()
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        self.assertGreater(
            sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0
        )
        self.assertTrue(
            all(p.grad is None or bool(t.isfinite(p.grad).all()) for p in model.parameters())
        )

    def test_noise_endpoint_inversion_and_code_bounds(self):
        from research.corpus.ink_diffusion_model import schedule
        from research.corpus.latent_ink_model import bounded_code

        t = self.torch
        real = t.randn(3, 8, 16, 64, device=self.device)
        noise = t.randn_like(real)
        a, s = schedule(t.tensor([0.0, 0.5, 1.0], device=self.device))
        self.assertTrue(
            t.allclose(a * (a * real + s * noise) - s * (a * noise - s * real), real, atol=1e-5)
        )
        mean = t.full((1, 8, 1, 1), 0.2, device=self.device)
        std = t.full_like(mean, 0.3)
        reconstructed = bounded_code(real * 20, mean, std) * std + mean
        self.assertLessEqual(float(reconstructed.max()), 1.000001)
        self.assertGreaterEqual(float(reconstructed.min()), -1.000001)

    def test_target_free_sampling_replays_and_rejects_wrong_shapes(self):
        from research.corpus.latent_ink_model import InkCodec, LatentInk, generate

        t = self.torch
        t.manual_seed(2011)
        codec = InkCodec().to(self.device).eval()
        model = LatentInk().to(self.device).eval()
        reference = t.zeros(1, 8, 16, 64, device=self.device)
        noise = t.randn_like(reference)
        tokens = t.zeros(1, 32, dtype=t.long, device=self.device)
        mean = t.zeros(1, 8, 1, 1, device=self.device)
        std = t.ones_like(mean)
        args = (model, codec, reference, tokens, noise, mean, std)
        first, _ = generate(*args, steps=2)
        second, _ = generate(*args, steps=2)
        self.assertTrue(t.equal(first, second))
        self.assertEqual(tuple(first.shape), (1, 1, 128, 512))
        self.assertTrue(bool(t.isfinite(first).all()))
        with self.assertRaises(ValueError):
            generate(*args, steps=1)
        with self.assertRaises(ValueError):
            codec.encode(t.zeros(1, 1, 64, 512, device=self.device))


if __name__ == "__main__":
    unittest.main()
