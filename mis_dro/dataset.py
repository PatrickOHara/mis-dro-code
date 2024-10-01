"""Data generation and dataset functions"""

from pathlib import Path
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
        dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0])
        cov_multiplier = 20.0
        # NOTE sklearn doesn't seem to accept a Generator
        sklearn_cov_seed = 1    # NOTE fix the seed, we always want the same covariance
        sklearn_random_state = np.random.RandomState(seed=sklearn_cov_seed)
        dgp_cov = cov_multiplier * make_spd_matrix(dim, random_state=sklearn_random_state)
        return multivariate_normal.rvs(dgp_mean, dgp_cov, size=num_observations, random_state=generator)
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

def portfolio_dataset(dgp: str, replication: int, portfolio_dir: Optional[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Gets the training and test datasets for the porfolio problem.
    
    Args:
        dgp: Data-generating process. Either 'dow30' or 'brownian'.
        replication: The ID of the quarter.

    Returns:
        training_data: numpy array of shape (NUM_TRADING_DAYS_IN_YEAR, NUM_STOCKS)
        test_data: numpy array of shape (NUM_TRADING_DAYS_IN_QUARTER, NUM_STOCKS)

    Notes:
        The replication is the ID of the quarter.
        Assume we are given K total quarters and let replciation 0 be Year 1 first quarter.
        Then replication 2 is Year 1 second quarter, 6 is Year 2 third quarter, etc.
    """
    NUM_TRADING_DAYS_IN_YEAR = 0        # NOTE does this change per 
    NUM_TRADING_DAYS_IN_QUARTER = 0
    NUM_STOCKS = 30
    training_data = np.zeros(NUM_TRADING_DAYS_IN_YEAR, NUM_STOCKS)
    test_data = np.zeros(NUM_TRADING_DAYS_IN_QUARTER, NUM_STOCKS)
    if dgp == "dow30":
        # Dow Jones Industrial Average (I think this what we said Dhanush?)
        # TODO get four quarters of training data and 1 quarter of test dataset using the replication
        pass

    elif dgp == "brownian":
        # TODO a synthetic datatset that uses brownian motion - only 10 stocks? keep it small
        # the replication is used to seed the random number generator
        pass

    return training_data, test_data
