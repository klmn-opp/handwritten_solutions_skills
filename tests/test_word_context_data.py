import unittest

from research.corpus.word_context_train import complete_indices


class WordContextDataTest(unittest.TestCase):
    def test_partial_rejected_local_crops_do_not_enter_context_training(self):
        rows = [
            {"index": 0, "known_local_positions": 3, "semantic_characters": 3},
            {"index": 1, "known_local_positions": 5, "semantic_characters": 7},
        ]
        self.assertEqual(complete_indices(rows), [0])
        with self.assertRaises(ValueError):
            complete_indices([rows[1]])
        with self.assertRaises(ValueError):
            complete_indices([rows[0], rows[0]])
