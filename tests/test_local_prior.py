import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch remains optional")
class LocalPrior(unittest.TestCase):
    def test_reproducible_joint_neighbourhood_preserves_unsupported_axes(self):
        import torch

        from research.corpus.local_prior import sample_neighborhood

        centers = torch.zeros(3, 10, requires_grad=True)
        with torch.no_grad():
            centers[:, :2] = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
        a, i = sample_neighborhood(centers, 0.15, generator=torch.Generator().manual_seed(2012))
        b, j = sample_neighborhood(centers, 0.15, generator=torch.Generator().manual_seed(2012))
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(i, j)
        self.assertFalse(a.requires_grad)
        self.assertTrue(torch.equal(a[2:], centers[i, 2:]))
        self.assertFalse(torch.equal(a[:2], centers[i, :2]))
        same, _ = sample_neighborhood(centers[:1], 0.15)
        self.assertTrue(torch.equal(same, centers[0]))

    def test_center_control_and_invalid_sampling_configuration(self):
        import torch

        from research.corpus.local_prior import sample_neighborhood

        centers = torch.randn(4, 10)
        value, index = sample_neighborhood(centers, 0)
        self.assertTrue(torch.equal(value, centers[index]))
        for sigma in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                sample_neighborhood(centers, sigma)
        with self.assertRaises(ValueError):
            sample_neighborhood(centers[:0], 0.15)
