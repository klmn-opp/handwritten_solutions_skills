import importlib.util
import unittest


class AdversarialInkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch") is None:
            raise unittest.SkipTest("Torch is isolated")
        import torch

        cls.t = torch
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(2)

    def test_blank_and_unsupported_ink_are_penalized(self):
        from research.corpus.adversarial_ink_model import structure_loss

        t = self.t
        ref = t.zeros(1, 1, 64, 64, device=self.device)
        ref[..., 20:44, 30:34] = 1
        perfect, _ = structure_loss(ref, ref)
        blank, _ = structure_loss(t.zeros_like(ref), ref)
        added = ref.clone()
        added[..., 5:12, 5:12] = 1
        extra, parts = structure_loss(added, ref)
        self.assertEqual(float(perfect), 0)
        self.assertGreater(float(blank), 0)
        self.assertGreater(float(extra), 0)
        self.assertGreater(float(parts["outside"]), 0)

    def test_frozen_critic_updates_generator_without_critic_gradients(self):
        from research.corpus.adversarial_ink_model import InkPatchCritic
        from research.corpus.local_ink_model import LocalInk

        t = self.t
        model = LocalInk().to(self.device)
        critic = InkPatchCritic().to(self.device).requires_grad_(False)
        ink = model(t.rand(2, 1, 64, 64, device=self.device)).sigmoid()
        (critic(ink) - 1).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0)

    def test_patch_prediction_agrees_with_full_padded_inference(self):
        from research.corpus.adversarial_ink_model import generate_local, word_patches
        from research.corpus.local_ink_model import LocalInk

        t = self.t
        model = LocalInk().to(self.device).eval()
        ref = t.rand(1, 1, 128, 192, device=self.device)
        centers = [(0, 64, 96)]
        patch = word_patches(ref, centers, 64)
        # Different convolution sizes select different TF32 kernels. Test the
        # mathematical locality contract in float32, independent of that rounding.
        with t.inference_mode(), t.backends.cudnn.flags(allow_tf32=False):
            crop = model(patch).sigmoid()[..., 32:96, 32:96]
            full = generate_local(model, ref)[..., 32:96, 64:128]
        self.assertTrue(t.allclose(crop, full, atol=1e-6, rtol=1e-6))
