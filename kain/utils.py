import numpy as np


def round_within_range(
    value:np.ndarray, vmin:float, vmax:float
    ) -> np.ndarray:
    assert vmin<=vmax
    result = np.round(value)
    result = np.clip(result, vmin, vmax)
    return result.astype(int)