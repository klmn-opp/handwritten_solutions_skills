import cv2
import numpy as np

from research.corpus.ink_integrity import map_components
from research.corpus.ink_integrity_lines import render_line


def test_one_pass_line_identity_preserves_coverage_and_does_not_duplicate_neighbor():
    low = np.zeros((40, 100), np.float32)
    low[5:35, 15:18] = 1
    low[10:30, 67:80] = 0.8
    high = cv2.resize(low, (400, 160), interpolation=cv2.INTER_NEAREST)
    words = [
        {
            "left": left,
            "right": left + 50,
            "transform": {"bbox": [0, 0, 50, 40], "scale": 2.4, "translation": [196, 16]},
        }
        for left in [0, 50]
    ]
    identity = map_components(np.zeros((2, 5, 17)), guarded=False)
    out = render_line(high, words, [identity, identity], (228, 168))
    expected = np.pad(low, ((64, 64), (64, 64)))
    assert np.allclose(out, expected, atol=1e-6)
    # A translated second word cannot sample the first word as extra ink.
    shifted = {k: v.copy() for k, v in identity.items()}
    shifted["shift"] += 5
    out = render_line(high, words, [identity, shifted], (228, 168))
    assert np.allclose(out[64:104, 64:114], low[:, :50], atol=1e-6)
