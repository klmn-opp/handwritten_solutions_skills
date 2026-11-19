import numpy as np
import pytest
from PIL import Image

from research.corpus.context_evaluation_ink import evaluation_ink
from research.corpus.context_training_data import whole_ink


def test_light_ink_evaluation_preserves_source_and_training_threshold():
    a = np.full((50, 180), 245, dtype=np.uint8)
    a[15:35, 30:36] = 155
    raw = Image.fromarray(a)
    with pytest.raises(ValueError, match="No dark ink"):
        whole_ink(raw)
    ink, meta = evaluation_ink(raw)
    assert ink.max() > 0.9 and (ink > 0.5).sum() > 100
    assert np.array_equal(a, np.asarray(raw))
    assert not meta["training_preprocessing_changed"]
    with pytest.raises(ValueError, match="Insufficient contrast"):
        evaluation_ink(Image.new("L", (80, 30), 245))


def test_dark_evaluation_ink_identical_to_existing_recipe():
    raw = Image.new("L", (120, 40), 255)
    raw.paste(20, (20, 10, 100, 25))
    old, transform = whole_ink(raw)
    new, metadata = evaluation_ink(raw)
    assert np.array_equal(old, new) and transform == metadata
