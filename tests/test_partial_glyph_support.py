import unittest

import numpy as np

from research.corpus.partial_glyph_support import validate_partial


class PartialSupportTests(unittest.TestCase):
    def setUp(self):
        self.ink = np.zeros((128, 512), np.float32)
        self.ink[20:50, 20:25] = 1
        self.ink[10:13, 20:25] = 1  # Complete detached dot is allowed.
        self.ink[20:50, 40:45] = 1
        self.review = {
            "text": "in", "certain_glyphs": [
                {"position": 0, "char": "i", "bbox": [19, 9, 26, 51]}
            ], "unresolved_positions": [1],
        }

    def test_partial_word_preserves_uncertain_neighbour(self):
        before = self.ink.copy()
        self.assertEqual(len(validate_partial(self.review, "in", self.ink)), 1)
        np.testing.assert_array_equal(before, self.ink)

    def test_connected_stroke_cannot_be_cut(self):
        self.ink[30, 20:45] = 1
        with self.assertRaisesRegex(ValueError, "connected"):
            validate_partial(self.review, "in", self.ink)

    def test_identity_and_unresolved_accounting(self):
        with self.assertRaisesRegex(ValueError, "remaining"):
            validate_partial({**self.review, "unresolved_positions": []}, "in", self.ink)
        self.review["certain_glyphs"][0]["char"] = "l"
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_partial(self.review, "in", self.ink)

    def test_overlapping_claims_cannot_duplicate_ink(self):
        self.review["certain_glyphs"].append(
            {"position": 1, "char": "n", "bbox": [19, 9, 46, 51]}
        )
        self.review["unresolved_positions"] = []
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_partial(self.review, "in", self.ink)
