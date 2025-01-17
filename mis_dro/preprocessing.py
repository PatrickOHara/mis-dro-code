"""Function for preprocessing the datasets"""

import numpy as np

def normalise_by_dimension(data: np.ndarray) -> np.ndarray:
    """For each dimension, normalise by subtracting the mean
    and dividing by the standard deviation

    Args:
        data: Shape is (num_observations, dim)

    Returns:
        Normalised dataset
    """
    return np.apply_along_axis(lambda x: (x - np.mean(x))/np.std(x), 0, data)


