"""Data generation and dataset functions"""

from typing import Optional
import numpy as np
from scipy.stats import expon, truncnorm, t


def data_generation_outliers(num_observations: int, contamination: float, random_state: Optional[np.random.Generator] = None):
    """A contaminated exponential data-generating process (DGP)

    Args:
        num_observations: Number of observations from DGP
        contamination: Ratio of contaminated observations (outliers)
        random_state: A numpy random generator, if provided

    Notes:
        Shuffles data to ensure anomalies are not grouped together
    """
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    data = expon.rvs(scale=10, size=n_real, random_state=random_state)
    outl = expon.rvs(scale=70, size=cont_size, random_state=random_state)
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffles the data in-place
    return data

def data_generation_trunc_t(num_observations: int, df: int, random_state: Optional[np.random.Generator] = None):
    """A truncated (above zero) Student-t data-generating process (DGP)
    
    Args:
        num_observations: Number of observations from DGP
        contamination: Degrees of freedom
        random_state: A numpy random generator, if provided
    """
    if not random_state:
        random_state = np.random.default_rng()
        
    # for symmetrical pdfs truncation at zero is equivalent to folding i.e. taking the absolute value of r.v.
    t_samples = np.abs(t.rvs(df, size=num_observations))
    
    return t_samples
    
        
