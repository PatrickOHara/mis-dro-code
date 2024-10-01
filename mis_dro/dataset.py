"""Data generation and dataset functions"""

import math
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from scipy.stats import expon, gamma, norm, t, multivariate_normal
from sklearn.datasets import make_spd_matrix


from bayesian_dro.Bayesian_DRO_continuous import data_generation, DGP_STD_TRUNCATED_NORMAL
from .constants import IN_SAMPLE_TIME_WINDOW, OUT_OF_SAMPLE_TIME_WINDOW


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

def portfolio_dataset(dgp: str, time_window_id: int, mmc2_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Gets the training and test datasets for the porfolio problem.

    Args:
        dgp: Options include 'DowJones', 'FF49Industries', 'FTSE100', 'NASDAQ100', 'NASDAQComp', 'SP500'
        time_window_id: The ID of the time window.
        mmc2_dir: Path to the directory downloaded from 'data-in-brief' webpage below

    Returns:
        training_data: numpy array of shape (NUM_TRADING_DAYS_IN_YEAR, NUM_STOCKS)
        test_data: numpy array of shape (NUM_TRADING_DAYS_IN_QUARTER, NUM_STOCKS)

    Notes:
        Download data from https://www.data-in-brief.com/article/S2352-3409(16)30399-7/fulltext
    """
    returns_df = pd.read_excel(mmc2_dir / "Datasets" / dgp / f"{dgp}.xlsx", sheet_name="Assets_Returns", header=None)
    num_time_windows = get_num_time_windows(len(returns_df))
    assert time_window_id < num_time_windows
    start_training_week = time_window_id * OUT_OF_SAMPLE_TIME_WINDOW # inclusive
    end_training_week = start_training_week + IN_SAMPLE_TIME_WINDOW # not inclusive
    start_test_week = end_training_week # inclusive
    end_test_week = start_test_week + OUT_OF_SAMPLE_TIME_WINDOW # not inclusive
    training_data = returns_df.iloc[start_training_week: end_training_week].values
    test_data = returns_df.iloc[start_test_week: end_test_week].values
    return training_data, test_data

def get_num_time_windows(num_weeks: int) -> int:
    return math.floor((num_weeks - IN_SAMPLE_TIME_WINDOW) / OUT_OF_SAMPLE_TIME_WINDOW)
