import numpy as np

from research.corpus.semantic_raw_data import apply_metrics, class_metrics


def test_train_metrics_ignore_evaluation_and_keep_descender_height():
    tokens = np.array([[[ord("p"), 0, 0]], [[ord("p"), 0, 0]]])
    boxes = np.array([[[0, 0, 10, 20]], [[3, 4, 13, 104]]], dtype=np.float32)
    metrics = class_metrics(tokens, boxes, [0])
    assert metrics["p"] == {"count": 1, "aspect_median": 0.5}
    changed = apply_metrics(tokens, boxes, metrics)
    np.testing.assert_array_equal(changed[..., [1, 3]], boxes[..., [1, 3]])
    np.testing.assert_array_equal(changed[..., 0] + changed[..., 2], boxes[..., 0] + boxes[..., 2])
    np.testing.assert_array_equal(changed[0], boxes[0])
    assert changed[1, 0, 2] - changed[1, 0, 0] == 50
    assert apply_metrics(tokens, boxes, {}).tolist() == boxes.tolist()
