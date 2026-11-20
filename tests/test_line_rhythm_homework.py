import unittest

import numpy as np

from research.corpus.line_rhythm_homework import relax


class WordRhythmTest(unittest.TestCase):
    def test_translates_only_rightward_and_stops_at_fixed_items_or_margin(self):
        boxes = np.array(
            [
                [10, 10, 19, 30],
                [22, 10, 31, 30],
                [33, 10, 42, 30],
                [44, 10, 53, 30],
                [56, 10, 65, 30],
            ],
            dtype=float,
        )
        words = [{"word_id": i, "items": [i]} for i in [0, 1, 3, 4]]
        cores = {i: boxes[i].copy() for i in [0, 1, 3, 4]}
        old = boxes.copy()
        moves, pairs = relax(words, boxes, cores, 0.67, (72, 50), 66)
        self.assertEqual(moves, {1: 1, 4: 1})
        self.assertTrue(np.array_equal(boxes[[0, 2, 3]], old[[0, 2, 3]]))
        self.assertTrue(all(q["after"] > q["before"] for q in pairs))
        self.assertEqual({(q["left_word"], q["right_word"]) for q in pairs}, {(0, 1), (3, 4)})
