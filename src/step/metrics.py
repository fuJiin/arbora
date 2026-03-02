def compute_iou(a: frozenset[int], b: frozenset[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


def rolling_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    tail = values[-window:]
    return sum(tail) / len(tail)
