import unittest

import numpy as np

from research.corpus.ink_tone_data import normalize_density


class InkDensityTests(unittest.TestCase):
    def test_faint_complete_stroke_is_darkened_without_adding_or_moving_ink(self):
        im = np.zeros((20, 20), np.float32)
        im[3:17, 8:10] = 0.55
        original = im.copy()
        out, gain = normalize_density(im)
        self.assertGreater(gain, 1)
        np.testing.assert_array_equal(out > 0, im > 0)
        np.testing.assert_array_equal(im, original)
        self.assertAlmostEqual(float(out.max()), 1)

    def test_dark_sample_unchanged_and_blank_rejected(self):
        im = np.eye(20, dtype=np.float32)
        out, gain = normalize_density(im)
        self.assertEqual(gain, 1)
        np.testing.assert_array_equal(out, im)
        with self.assertRaises(ValueError):
            normalize_density(im * 0)
