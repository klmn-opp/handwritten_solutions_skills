"""CPU sampling contracts for reproducible local-ink optimization."""

import numpy as np


def sample_indices(rng, by_class, batch, *, balanced=False):
    classes = sorted(by_class)
    if balanced:
        if batch != len(classes):
            raise ValueError("Balanced updates require exactly one sample per real class")
        selected = rng.permutation(classes)
    else:
        selected = rng.choice(classes, batch)
    return np.array([rng.choice(by_class[int(c)]) for c in selected])


def sample_points(rng, batch, count):
    if count < 4 or count % 4:
        raise ValueError("Pixel query count must be a positive multiple of four")
    core = 3 * count // 4
    return np.concatenate(
        [rng.integers(8, 88, (batch, core, 2)), rng.integers(0, 96, (batch, count - core, 2))],
        axis=1,
    )
