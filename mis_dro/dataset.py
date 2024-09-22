"""Data generation and dataset functions"""

from typing import Optional
import numpy as np
from scipy.stats import expon, gamma, norm

from bayesian_dro.Bayesian_DRO_continuous import data_generation, DGP_STD_TRUNCATED_NORMAL


def sample_dgp(
    dgp: str,
    num_observations: int,
    contamination: float = 0.0,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample from the DGP"""
    if not generator:
        generator = np.random.default_rng()
    if dgp == "normal":
        return norm.rvs(
            loc=25,
            scale=DGP_STD_TRUNCATED_NORMAL,
            size=num_observations,
            random_state=generator,
        )
    if dgp == "truncated_normal":
        return data_generation(
            num_observations, random_state=generator
        )  # generate new observations
    if dgp == "contaminated_exp":
        # specify contamination level
        return data_generation_outliers(
            num_observations, contamination, random_state=generator
        )
    if dgp == "exponential":
        return expon.rvs(scale=20.0, size=num_observations, random_state=generator)
    if dgp == "gamma":
        return data_generation_gamma(num_observations, a=10, random_state=generator)
    if dgp == "contaminated_normal":
        return contaminated_normal(
            num_observations, contamination, random_state=generator)
    raise ValueError(f"The data-generating process specified is not supported: {dgp}")


def data_generation_outliers(
    num_observations: int,
    contamination: float,
    random_state: Optional[np.random.Generator] = None,
):
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

def contaminated_normal(num_observations: int, contamination: float, random_state: Optional[np.random.Generator] = None):
    """A contaminated Gaussian data-generating process (DGP)

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
    data = norm.rvs(loc=25, scale=DGP_STD_TRUNCATED_NORMAL, size=n_real, random_state=random_state) 
    outl = norm.rvs(loc=75, scale=DGP_STD_TRUNCATED_NORMAL, size=cont_size, random_state=random_state) 
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffles the data in-place
    return data
    

def data_generation_gamma(num_observations: int, a: float, random_state: Optional[np.random.Generator] = None):
    """A Gamma data-generating process (DGP)

    Args:
        num_observations: Number of observations from DGP
        a: shape parameter
        random_state: A numpy random generator, if provided
    """
    if not random_state:
        random_state = np.random.default_rng()

    gamma_samples = gamma.rvs(a, size=num_observations)

    return gamma_samples
