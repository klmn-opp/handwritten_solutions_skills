import unittest

import numpy as np

from research.corpus.context_metric_layout import adjust_boxes


class ContextMetricsTest(unittest.TestCase):
    metrics = {"t_median": 1.2, "t_upper": 1.4, "l_left_upper": 0.2, "r_right_upper": 0.2}

    def test_internal_spacing_preserves_word_starts_and_never_crosses_whitespace(self):
        tokens = np.array(
            [[97, 0, 108], [108, 97, 114], [114, 108, 97], [97, 114, 0], [108, 0, 97], [97, 108, 0]]
        )
        boxes = np.array(
            [
                [0, 0, 10, 10],
                [17, 0, 19, 10],
                [22, 0, 28, 10],
                [34, 0, 44, 10],
                [60, 0, 62, 10],
                [66, 0, 76, 10],
            ],
            np.float32,
        )
        out = adjust_boxes(tokens, boxes, self.metrics)
        self.assertTrue(np.array_equal(out[[0, 4, 5]], boxes[[0, 4, 5]]))
        self.assertAlmostEqual(float(out[1, 0] - out[0, 2]), 2)
        self.assertAlmostEqual(float(out[3, 0] - out[2, 2]), 2)
        self.assertTrue(np.array_equal(out[:, [1, 3]], boxes[:, [1, 3]]))
        self.assertTrue(np.array_equal(boxes[1], [17, 0, 19, 10]))
        reflow = adjust_boxes(tokens, boxes, self.metrics, preserve_word_gaps=True)
        self.assertAlmostEqual(float(reflow[4, 0] - reflow[3, 2]), 16)
        self.assertAlmostEqual(float(reflow[5, 0] - reflow[4, 2]), 4)

    def test_only_oversized_lowercase_t_changes_and_keeps_baseline_and_shape_ratio(self):
        tokens = np.array(
            [[97, 0, 116], [116, 97, 116], [116, 116, 84], [84, 116, 103], [103, 84, 0]]
        )
        boxes = np.array(
            [[0, 10, 8, 20], [10, 4, 18, 20], [20, 7, 26.5, 20], [30, 2, 40, 20], [43, 10, 51, 26]],
            np.float32,
        )
        out = adjust_boxes(tokens, boxes, self.metrics)
        self.assertTrue(np.array_equal(out[[0, 2, 3, 4]], boxes[[0, 2, 3, 4]]))
        self.assertEqual(out[1, 3], 20)
        self.assertAlmostEqual(float((out[1, 2] - out[1, 0]) / (out[1, 3] - out[1, 1])), 0.5)
        self.assertAlmostEqual(float(out[1, 3] - out[1, 1]), 14)
