"""Data generation and dataset functions"""

import math
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from scipy.stats import expon, gamma, norm, t, multivariate_normal
from sklearn.datasets import make_spd_matrix


from bayesian_dro.Bayesian_DRO_continuous import data_generation, DGP_STD_TRUNCATED_NORMAL
from .constants import IN_SAMPLE_TIME_WINDOW, OUT_OF_SAMPLE_TIME_WINDOW, DGP_NORMAL_KNOWN_VARIANCE_STD

import pickle

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
            # scale=5,
            size=num_observations,
            random_state=generator,
        ).reshape((num_observations, 1))
    if dgp == "truncated_normal":
        return data_generation(
            num_observations, random_state=generator
        ).reshape((num_observations, 1))  # generate new observations
    if dgp == "contaminated_exp":
        # specify contamination level
        return data_generation_outliers(
            num_observations, contamination, random_state=generator
        ).reshape((num_observations, 1))
    if dgp == "contaminated_exp_large_outliers":
        return data_generation_outliers(
            num_observations, contamination, outlier_mean=1000.0, random_state=generator
        ).reshape((num_observations, 1))
    if dgp == "contaminated_exp_small_outliers":
        return contaminated_exp_small_outliers(num_observations, contamination, random_state=generator).reshape((num_observations, 1))
    if dgp == "exponential":
        return expon.rvs(scale=20.0, size=num_observations, random_state=generator).reshape((num_observations, 1))
    if dgp == "gamma":
        return data_generation_gamma(num_observations, a=10, random_state=generator).reshape((num_observations, 1))
    if dgp == "multivariate_normal_known_cov":
        dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0])
        dgp_cov = (DGP_STD_TRUNCATED_NORMAL**2)*np.eye(dim)
        # cov_multiplier = 20.0
        # # NOTE sklearn doesn't seem to accept a Generator
        # sklearn_cov_seed = 1    # NOTE fix the seed, we always want the same covariance
        # sklearn_random_state = np.random.RandomState(seed=sklearn_cov_seed)
        # dgp_cov = cov_multiplier * make_spd_matrix(dim, random_state=sklearn_random_state)
        return multivariate_normal.rvs(dgp_mean, dgp_cov, size=num_observations, random_state=generator)
    if dgp == "multivariate_normal":
        dgp_mean = np.array([10.0, 20.0, 30.0, 35.0, 22.0])
        cov_multiplier = 20.0
        # NOTE sklearn doesn't seem to accept a Generator
        sklearn_cov_seed = 1    # NOTE fix the seed, we always want the same covariance
        sklearn_random_state = np.random.RandomState(seed=sklearn_cov_seed)
        dgp_cov = cov_multiplier * make_spd_matrix(dim, random_state=sklearn_random_state)
        return multivariate_normal.rvs(dgp_mean, dgp_cov, size=num_observations, random_state=generator)
    if dgp == "cont_multivariate_normal":
        return cont_multivariate_normal(
            num_observations, contamination, random_state=generator)
    if dgp == "portfolio_contaminated_multivariate_normal":
        return portfolio_contaminated_multivariate_normal(num_observations, contamination, random_state=generator)
    if dgp == "contaminated_normal":
        return contaminated_normal(
            num_observations, contamination, random_state=generator)
    if dgp == "student_t":
        return t.rvs(df=3, loc=25, scale=DGP_STD_TRUNCATED_NORMAL, size=num_observations, random_state=generator).reshape((num_observations, 1))
    if dgp == "contaminated_exp_old":
        cont_size = int(np.floor(contamination * num_observations))
        n_real = num_observations - cont_size
        data = expon.rvs(scale=10, size=n_real, random_state=generator)
        outl = expon.rvs(scale=70, size=cont_size, random_state=generator)
        data = np.concatenate((data, outl), axis=0)
        generator.shuffle(data)  # shuffles the data in-place
        return data.reshape((num_observations,1))
    if dgp == "bimodal_multivariate_gaussian":
        cont_size = int(np.floor(contamination * num_observations))
        n_real = num_observations - cont_size
        data_mode1 = multivariate_normal.rvs(mean=np.array([10,20,33,22,25]), cov=5*np.eye(5), size=int(n_real/2), random_state=generator)
        data_mode2 = multivariate_normal.rvs(mean=60*np.ones(5), cov=5*np.eye(5), size=int(n_real/2), random_state=generator)
        outl = multivariate_normal.rvs(mean=90*np.ones(5), cov=5*np.eye(5), size=cont_size, random_state=generator)
        data = np.concatenate((data_mode1, data_mode2, outl), axis=0)
        generator.shuffle(data)  # shuffles the data in-place
        return data
    if dgp == "bimodal_univariate_gaussian":
        cont_size = int(np.floor(contamination * num_observations))
        n_real = num_observations - cont_size
        data_mode1 = norm.rvs(loc=10, scale=5, size=int(n_real/2), random_state=generator)
        data_mode2 = norm.rvs(loc=60, scale=5, size=int(n_real/2), random_state=generator)
        outl = norm.rvs(loc=90, scale=5, size=cont_size, random_state=generator)
        data = np.concatenate((data_mode1, data_mode2, outl), axis=0)
        generator.shuffle(data)  # shuffles the data in-place
        return data.reshape((num_observations,1))
    raise ValueError(f"The data-generating process specified is not supported: {dgp}")


def data_generation_outliers(
    num_observations: int,
    contamination: float,
    outlier_mean: float = 100.0,
    random_state: Optional[np.random.Generator] = None,
):
    """A contaminated exponential data-generating process (DGP)

    Args:
        num_observations: Number of observations from DGP
        contamination: Ratio of contaminated observations (outliers)
        outlier_mean: Location of the mean of the Gaussian to draw contaminated samples from
        random_state: A numpy random generator, if provided

    Notes:
        Shuffles data to ensure anomalies are not grouped together
    """
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    data = expon.rvs(scale=20, size=n_real, random_state=random_state)
    outl = norm.rvs(loc=outlier_mean, scale=0.5, size=cont_size, random_state=random_state) 
    data = np.concatenate((data, outl), axis=0)
    random_state.shuffle(data)  # shuffle the data in-place
    return data

def contaminated_exp_small_outliers(
    num_observations: int,
    contamination: float,
    random_state = None,
):
    """A contaminated exponential data-generating process (DGP) with
    small outliers coming from an exponential distribution with mean 0.01.

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
    outl = expon.rvs(scale=0.01, size=cont_size, random_state=random_state) 
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
    data = norm.rvs(loc=25, scale=DGP_NORMAL_KNOWN_VARIANCE_STD, size=n_real, random_state=random_state) 
    outl = norm.rvs(loc=75, scale=DGP_NORMAL_KNOWN_VARIANCE_STD, size=cont_size, random_state=random_state) 
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

def portfolio_contaminated_multivariate_normal(num_observations: int, contamination: float, random_state: Optional[np.random.Generator] = None):
    """Portfolio contamination dataset"""
    if not random_state:
        random_state = np.random.default_rng()
    cont_size = int(np.floor(contamination * num_observations))
    n_real = num_observations - cont_size
    dgp_mean = np.array([2.5, 0.5, -1.0, -3.0, 3.5])
    # dgp_mean = np.zeros(5)
    dgp_mean_outl = dgp_mean + np.array([2.5, 50.0, 50.0, 50.0, 3.5])
    dgp_cov = np.diag(np.array([5.0, 10.0, 15.0, 20.0, 30.0]))
    # dgp_cov = np.diag(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
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

def portfolio_dataset(
    dgp: str,
    time_window_id: int,
    mmc2_dir: Path,
    in_sample_time_window: int = IN_SAMPLE_TIME_WINDOW,
    out_of_sample_time_window: int = OUT_OF_SAMPLE_TIME_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Gets the training and test datasets for the porfolio problem.

    Args:
        dgp: Options include 'DowJones', 'DowJones-crash', 'FF49Industries', 'FTSE100', 'NASDAQ100', 'NASDAQComp', 'SP500'
        time_window_id: The ID of the time window.
        mmc2_dir: Path to the directory downloaded from 'data-in-brief' webpage below

    Returns:
        training_data: numpy array of shape (NUM_TRADING_DAYS_IN_YEAR, NUM_STOCKS)
        test_data: numpy array of shape (NUM_TRADING_DAYS_IN_QUARTER, NUM_STOCKS)

    Notes:
        Download data from https://www.data-in-brief.com/article/S2352-3409(16)30399-7/fulltext
    """
    returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
    num_time_windows = get_num_time_windows(len(returns_df))
    assert time_window_id < num_time_windows
    start_training_week = time_window_id * 12 # inclusive
    end_training_week = start_training_week + in_sample_time_window # not inclusive
    start_test_week = end_training_week # inclusive
    end_test_week = start_test_week + out_of_sample_time_window # not inclusive
    training_data = returns_df.iloc[start_training_week: end_training_week].values
    test_data = returns_df.iloc[start_test_week: end_test_week].values
    return training_data, test_data

def get_portfolio_returns_df(mmc2_dir: Path, dgp: str) -> pd.DataFrame:
    dgp_temp = dgp
    if dgp == "DowJones-crash":
        dgp_temp = "DowJones"
    return pd.read_excel(mmc2_dir / "Datasets" / dgp_temp / f"{dgp_temp}.xlsx", sheet_name="Assets_Returns", header=None)

def get_num_time_windows(
    num_weeks: int,
    in_sample_time_window: int = IN_SAMPLE_TIME_WINDOW,
    out_of_sample_time_window: int = OUT_OF_SAMPLE_TIME_WINDOW
) -> int:
    return math.floor((num_weeks - in_sample_time_window) / out_of_sample_time_window)

def portfolio_dataset_james(
    time_window_id: int,
    dataset_dir_james
) -> tuple[np.ndarray, np.ndarray]:
    with open(dataset_dir_james, 'rb') as f:
        windows = pickle.load(f)
    num_time_windows = len(windows)
    assert time_window_id < num_time_windows
    window = windows[time_window_id]
    training_data = window[0].values
    test_data = window[1].values
    return training_data, test_data

def get_num_time_windows_james(dataset_dir_james) -> int:
    with open(dataset_dir_james, 'rb') as f:
        windows = pickle.load(f)
    return len(windows)

def get_min_dim_james(dataset_dir_james) -> int:
    with open(dataset_dir_james, 'rb') as f:
        windows = pickle.load(f)
    min_dim = float('inf')
    for window in windows:
        training_data = window[0]
        min_dim = min(min_dim, len(training_data.iloc[0]))
    return min_dim