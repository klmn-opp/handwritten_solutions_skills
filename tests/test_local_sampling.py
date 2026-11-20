import numpy as np
import pytest

from research.corpus.local_sampling import sample_indices, sample_points


def test_default_sampling_exactly_preserves_previous_random_stream():
    old = np.random.default_rng(3518)
    new = np.random.default_rng(3518)
    groups = {97: np.array([0, 2]), 98: np.array([1, 3, 4])}
    expected = np.array([old.choice(groups[int(c)]) for c in old.choice(sorted(groups), 4)])
    assert np.array_equal(sample_indices(new, groups, 4), expected)
    points = np.concatenate([old.integers(8, 88, (4, 768, 2)), old.integers(0, 96, (4, 256, 2))], 1)
    assert np.array_equal(sample_points(new, 4, 1024), points)
    assert old.bit_generator.state == new.bit_generator.state


def test_balanced_step_covers_every_class_once_without_fixed_variant():
    groups = {97: [0, 2], 98: [1, 3], 99: [4, 5]}
    labels = np.array([97, 98, 97, 98, 99, 99])
    rng = np.random.default_rng(3518)
    observed = set()
    for _ in range(20):
        selected = sample_indices(rng, groups, 3, balanced=True)
        assert sorted(labels[selected]) == sorted(groups)
        observed.update(selected)
    assert observed == set(range(6))
    with pytest.raises(ValueError):
        sample_indices(rng, groups, 4, balanced=True)


def test_query_budget_and_domain_follow_registered_ratio():
    points = sample_points(np.random.default_rng(3518), 28, 512)
    assert points.shape == (28, 512, 2)
    assert points[:, :384].min() >= 8 and points[:, :384].max() < 88
    assert points.min() >= 0 and points.max() < 96
    with pytest.raises(ValueError):
        sample_points(np.random.default_rng(0), 4, 7)
