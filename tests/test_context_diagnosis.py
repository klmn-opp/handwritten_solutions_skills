import numpy as np

from research.corpus.context_diagnosis import nearest


def test_nearest_keeps_exact_binary_and_continuous_similarity_separate():
    images = np.zeros((2, 8, 8), dtype=np.float32)
    images[:, 2:6, 3:5] = 0.8
    target = images[:1].copy()
    target[:, 2:6, 3:5] = 1
    report = nearest(images, target)
    assert report["exact_binary_count"] == 2
    assert report["minimum_shape"] > 0
    assert nearest(images, np.zeros((0, 8, 8))) is None


def test_unrelated_crop_does_not_count_as_exact_copy():
    images = np.eye(8, dtype=np.float32)[None]
    target = images[:, ::-1].copy()
    report = nearest(images, target)
    assert report["exact_binary_count"] == 0
    assert report["minimum_shape"] > 0
