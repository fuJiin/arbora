import numpy as np
from numpy.typing import NDArray


def local_normalize(vector: NDArray[np.floating]) -> NDArray[np.floating]:
    max_val = np.max(vector)
    return vector / max_val if max_val > 0 else vector
