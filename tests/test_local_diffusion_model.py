import importlib.util
import unittest


class LocalDiffusionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Torch is isolated")
        import torch

        cls.t = torch
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(2)

    def test_velocity_reconstruction_and_noise_are_invertible(self):
        from research.corpus.local_diffusion_model import noise_schedule

        t = self.t
        clean = t.rand(3, 1, 32, 64, device=self.device) * 2 - 1
        noise = t.randn_like(clean)
        alpha, sigma = noise_schedule(t.tensor([0.0, 0.5, 1.0], device=self.device))
        mixed, velocity = alpha * clean + sigma * noise, alpha * noise - sigma * clean
        self.assertTrue(t.allclose(alpha * mixed - sigma * velocity, clean, atol=1e-6))
        self.assertTrue(t.allclose(sigma * mixed + alpha * velocity, noise, atol=1e-6))

    def test_real_targets_update_model_and_seed_affects_generated_ink(self):
        from research.corpus.local_diffusion_model import LocalInkDiffusion, generate_diffusion

        t = self.t
        model = LocalInkDiffusion().to(self.device)
        ref = t.rand(2, 1, 32, 64, device=self.device)
        noise = t.randn_like(ref)
        model(noise, ref, t.ones(2, device=self.device)).square().mean().backward()
        self.assertTrue(
            all(p.grad is not None and bool(t.isfinite(p.grad).all()) for p in model.parameters())
        )
        a = generate_diffusion(model, ref, noise, 4)
        b = generate_diffusion(model, ref, -noise, 4)
        self.assertTrue(bool(t.isfinite(a).all()))
        self.assertGreater(float((a - b).abs().sum()), 0)
        self.assertTrue(t.equal(a, generate_diffusion(model, ref, noise, 4)))
