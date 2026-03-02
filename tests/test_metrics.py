from step.metrics import compute_iou, rolling_mean


class TestComputeIoU:
    def test_identical(self):
        a = frozenset(range(10))
        assert compute_iou(a, a) == 1.0

    def test_disjoint(self):
        a = frozenset(range(0, 10))
        b = frozenset(range(10, 20))
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset(range(0, 10))
        b = frozenset(range(5, 15))
        # intersection=5, union=15
        assert compute_iou(a, b) == 5 / 15

    def test_both_empty(self):
        assert compute_iou(frozenset(), frozenset()) == 1.0

    def test_one_empty(self):
        assert compute_iou(frozenset(range(5)), frozenset()) == 0.0
        assert compute_iou(frozenset(), frozenset(range(5))) == 0.0


class TestRollingMean:
    def test_full_window(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert rolling_mean(values, 3) == 4.0  # mean of [3,4,5]

    def test_window_larger_than_values(self):
        values = [2.0, 4.0]
        assert rolling_mean(values, 100) == 3.0

    def test_empty(self):
        assert rolling_mean([], 10) == 0.0

    def test_single_value(self):
        assert rolling_mean([7.0], 5) == 7.0
