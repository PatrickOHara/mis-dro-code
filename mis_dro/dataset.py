"""Data generation and dataset functions"""

from typing import Optional
import numpy as np

from scipy.stats import expon, gamma, norm, t, multivariate_normal
from sklearn.datasets import make_spd_matrix


from bayesian_dro.Bayesian_DRO_continuous import data_generation, DGP_STD_TRUNCATED_NORMAL


def sample_dgp(
    dgp: str,
    num_observations: int,
    contamination: float = 0.0,
    dim: int = 1,
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
    if dgp == "multivariate_normal":
        # dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0])
        # cov_multiplier = 20.0
        # # NOTE sklearn doesn't seem to accept a Generator
        # sklearn_cov_seed = 1    # NOTE fix the seed, we always want the same covariance
        # sklearn_random_state = np.random.RandomState(seed=sklearn_cov_seed)
        # dgp_cov = cov_multiplier * make_spd_matrix(dim, random_state=sklearn_random_state)
        # return multivariate_normal.rvs(dgp_mean, dgp_cov, size=num_observations, random_state=generator)
        dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0])
        dgp_cov = (DGP_STD_TRUNCATED_NORMAL**2)*np.eye(dim)
        # cov_multiplier = 20.0
        # # NOTE sklearn doesn't seem to accept a Generator
        # sklearn_cov_seed = 1    # NOTE fix the seed, we always want the same covariance
        # sklearn_random_state = np.random.RandomState(seed=sklearn_cov_seed)
        # dgp_cov = cov_multiplier * make_spd_matrix(dim, random_state=sklearn_random_state)
        return multivariate_normal.rvs(dgp_mean, dgp_cov, size=num_observations, random_state=generator)
    if dgp == "cont_multivariate_normal":
        return cont_multivariate_normal(
            num_observations, contamination, random_state=generator)
    if dgp == "contaminated_normal":
        return contaminated_normal(
            num_observations, contamination, random_state=generator)
    if dgp == "student_t":
        return t.rvs(df=3, loc=25, scale=DGP_STD_TRUNCATED_NORMAL, size=num_observations, random_state=generator)
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
    data = expon.rvs(scale=20, size=n_real, random_state=random_state)
    # noise = norm.rvs(loc=1, scale=5, size=cont_size, random_state=random_state)
    outl = norm.rvs(loc=100, scale=0.5, size=cont_size, random_state=random_state) 
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffle the data in-place
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
    
def cont_multivariate_normal(num_observations: int, contamination: float, random_state: Optional[np.random.Generator] = None):
    
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0]) #10
    dgp_mean_outl = dgp_mean + 30
    dgp_cov = (DGP_STD_TRUNCATED_NORMAL**2)*np.eye(5)
    data = multivariate_normal.rvs(dgp_mean, dgp_cov, size=n_real, random_state=random_state)
    outl = multivariate_normal.rvs(dgp_mean_outl, dgp_cov, size=cont_size, random_state=random_state)
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
