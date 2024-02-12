"""Data generation and dataset functions"""

import numpy as np
from sklearn.utils import shuffle
from scipy.stats import expon


def data_generation_outliers(num_observations: int, contamination: float, random_state = None):
    """A contaminated exponential data-generating process (DGP)

    Args:
        num_observations: Number of observations from DGP
        contamination: Ratio of contaminated observations (outliers)
        generator: A numpy random generator, if provided

    Notes:
        Shuffles data to ensure anomalies are not grouped together
    """
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    data = expon.rvs(scale=10, size=n_real, random_state=random_state)
    outl = expon.rvs(scale=70, size=cont_size, random_state=random_state)
    data = np.concatenate((data, outl), axis=0)
    data = shuffle(data, random_state=random_state)
    return data
