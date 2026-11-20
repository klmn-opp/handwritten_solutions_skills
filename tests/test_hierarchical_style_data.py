import unittest

import numpy as np

from research.corpus.hierarchical_style_data import group_conditions


class HierarchicalDataTest(unittest.TestCase):
    def test_source_group_partition_missing_positions_and_duplicate_guard(self):
        rows = [
            {
                "word": "for",
                "position": p,
                "label": "for"[p],
                "source_hash": "actual",
                "source_crop_sha256": h,
                "source_crop": "scan.png",
            }
            for h, p in [("00000005abcdef", 0), ("00000005abcdef", 2), ("00000001abcdef", 1)]
        ]
        a, groups = group_conditions(rows, np.ones((3, 8), np.float32))
        self.assertEqual(len(groups), 2)
        self.assertEqual(int(a["known"].sum()), 3)
        self.assertEqual(int(a["train"].sum()), 1)
        self.assertTrue(np.all(a["codes"][~a["known"]] == 0))
        with self.assertRaises(ValueError):
            group_conditions(rows + [rows[0]], np.ones((4, 8), np.float32))
        with self.assertRaises(ValueError):
            group_conditions([{**rows[0], "label": "x"}, *rows[1:]], np.ones((3, 8), np.float32))
