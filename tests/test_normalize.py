import numpy as np
import pytest

from step.normalize import local_normalize


class TestLocalNormalize:
    def test_positive_values(self):
        vec = np.array([2.0, 4.0, 1.0])
        result = local_normalize(vec)
        np.testing.assert_array_almost_equal(result, [0.5, 1.0, 0.25])

    def test_max_becomes_one(self):
        vec = np.array([3.0, 7.0, 5.0])
        result = local_normalize(vec)
        assert result.max() == pytest.approx(1.0)

    def test_zero_vector_unchanged(self):
        vec = np.zeros(5)
        result = local_normalize(vec)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_all_negative_unchanged(self):
        vec = np.array([-1.0, -2.0, -3.0])
        result = local_normalize(vec)
        np.testing.assert_array_equal(result, vec)

    def test_mixed_with_positive_max(self):
        vec = np.array([-1.0, 2.0, 0.0])
        result = local_normalize(vec)
        np.testing.assert_array_almost_equal(result, [-0.5, 1.0, 0.0])
