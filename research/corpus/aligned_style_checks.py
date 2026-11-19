"""Optional CUDA property tests, run with the isolated research Python only."""

import unittest

import torch

from research.corpus.aligned_style_contract import text_tokens
from research.corpus.aligned_style_experiment import setup, topology
from research.corpus.aligned_style_model import AlignedStyle, dense_flow, monotone_controls, render


class GeometryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup(1707)

    def test_no_fold_even_at_extreme_inputs_and_finite_gradient(self):
        for scale in [0, 1, 10, 100]:
            raw = (torch.randn(16, 3, 17, device="cuda") * scale).requires_grad_()
            controls = monotone_controls(raw)
            self.assertTrue((topology(dense_flow(controls)) > 0).all())
            self.assertLess(float(controls[:, 0].abs().max()), 64.0001)
            self.assertLess(float(controls[:, 1].abs().max()), 24)
            controls.square().mean().backward()
            self.assertTrue(torch.isfinite(raw.grad).all())

    def test_identity_preserves_all_coverage_not_just_binary_core(self):
        reference = torch.rand(2, 1, 128, 512, device="cuda")
        controls = monotone_controls(torch.zeros(2, 3, 17, device="cuda"))
        self.assertTrue(torch.equal(render(reference, controls), reference))

    def test_generation_deterministic_for_fixed_seed_and_coordinate_gradient(self):
        reference = torch.rand(2, 1, 128, 512, device="cuda")
        tokens = torch.tensor(text_tokens(["the", "assignment"]), device="cuda")
        model = AlignedStyle(monotone=True).cuda()
        epsilon = torch.zeros(2, 4, device="cuda")
        a, control = model.generate(reference, tokens, epsilon)
        b, _ = model.generate(reference, tokens, epsilon)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue((topology(dense_flow(control)) > 0).all())
        a.mean().backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(q).all() for q in gradients))
        self.assertGreater(sum(float(q.abs().sum()) for q in gradients), 0)


if __name__ == "__main__":
    unittest.main()
