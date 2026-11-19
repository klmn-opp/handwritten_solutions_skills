import unittest

import numpy as np

from research.corpus.local_morphology import residual_coordinates


class MorphologyDataTests(unittest.TestCase):
    def test_only_within_class_variation_enters_coordinates(self):
        rng = np.random.default_rng(22)
        x = rng.random((12, 4, 4))
        labels = np.repeat([97, 98, 99], 4)
        original = x.copy()
        codes, priors, _ = residual_coordinates(x, labels, dimensions=3)
        np.testing.assert_array_equal(original, x)
        for c in np.unique(labels):
            np.testing.assert_allclose(codes[labels == c].mean(0), 0, atol=1e-6)
            self.assertEqual(priors[str(c)]["count"], 4)
        shifted = x + (labels - 97)[:, None, None] * 100
        shifted_codes, _, _ = residual_coordinates(shifted, labels, dimensions=3)
        np.testing.assert_allclose(codes, shifted_codes, atol=1e-6)

    def test_singleton_does_not_claim_variation(self):
        rng = np.random.default_rng(23)
        x = rng.random((6, 4, 4))
        codes, priors, _ = residual_coordinates(x, np.array([97] * 5 + [98]), dimensions=2)
        np.testing.assert_allclose(codes[-1], 0, atol=1e-6)
        self.assertFalse(priors["98"]["variation_supported"])
