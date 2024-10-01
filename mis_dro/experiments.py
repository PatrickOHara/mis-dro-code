"""An experiment is a list of dictionaries each containing parameter settings.

Each experiment has an `ExperimentName`.
Use the `get_experiment()` function to get the list of dictionaries associated with an experiment name.
"""

from enum import StrEnum
import itertools
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4
import numpy as np
import pandas as pd
from .constants import (
    BAS_DRO_EPSILON_SET,
    CONTAMINATION_LEVEL,
    NUM_CERTIFY,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
    PORTFOLIO_EPSILON_SET,
    IN_SAMPLE_TIME_WINDOW,
    OUT_OF_SAMPLE_TIME_WINDOW,
)
from .dataset import get_num_time_windows


class ExperimentName(StrEnum):
    """Names of experiments"""

    kl_newsvendor_1d = "kl_newsvendor_1d"
    kl_newsvendor_5d = "kl_newsvendor_5d"
    mmd_newsvendor_1d = "mmd_newsvendor_1d"
    compare_solve = "compare_solve"
    kl_portfolio = "kl_portfolio"

    def is_portfolio(self) -> bool:
        return self in (ExperimentName.kl_portfolio)


def get_experiment(experiment_name: ExperimentName, dataset_dir: Optional[Path] = None) -> List[Dict]:
    """Returns the experiment associated with the name"""
    function_lookup = {
        ExperimentName.kl_newsvendor_1d: kl_newsvendor_1d,
        ExperimentName.kl_newsvendor_5d: kl_newsvendor_5d,
        ExperimentName.mmd_newsvendor_1d: mmd_newsvendor_1d,
        ExperimentName.compare_solve: compare_solve,
    }
    try:
        if experiment_name.is_portfolio():
            # NOTE portfolio setup requires a dataset_dir argument
            return function_lookup[experiment_name](dataset_dir)
        return function_lookup[experiment_name]()
    except KeyError as e:
        raise KeyError(
            f"Please add {experiment_name} as a key in the function lookup dictionary"
        ) from e

def kl_newsvendor_5d() -> List[Dict]:
    """KL univariate newsvendor: compare our Bayesian ambiguity set against Bayesian DRO"""
    experiment = []
    for total_model_samples, algorithm, (dgp, likelihood, posterior), epsilon in itertools.product(
        [25, 100, 900, 2500],
        ["kl_dro_bas", "kl_bdro"],
        [
            ("multivariate_normal", "multivariate_normal", "normal_inverse_wishart"),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if algorithm == "kl_bdro":
            num_posterior_samples = int(np.sqrt(total_model_samples))
            num_likelihood_samples = int(np.sqrt(total_model_samples))
        if algorithm == "kl_dro_bas":
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = total_model_samples
            num_posterior_samples = 1
        contamination = 0.0
        if dgp == "contaminated_exp":
            contamination = CONTAMINATION_LEVEL
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": 5,
            "epsilon": epsilon,
            "inference": "bayes",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def kl_newsvendor_1d() -> List[Dict]:
    """KL univariate newsvendor: compare our Bayesian ambiguity set against Bayesian DRO"""
    experiment = []
    for total_model_samples, algorithm, (dgp, likelihood, posterior), epsilon in itertools.product(
        [25, 100, 900, 2500],
        ["kl_dro_bas", "kl_bdro"],
        [
            ("normal", "normal", "normal_gamma"),
            ("truncated_normal", "normal", "normal_gamma"),
            ("exponential", "exponential", "gamma"),
            ("contaminated_exp", "exponential", "gamma"),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if algorithm == "kl_bdro":
            num_posterior_samples = int(np.sqrt(total_model_samples))
            num_likelihood_samples = int(np.sqrt(total_model_samples))
        if algorithm == "kl_dro_bas":
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = total_model_samples
            num_posterior_samples = 1
        contamination = 0.0
        if dgp == "contaminated_exp":
            contamination = CONTAMINATION_LEVEL
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": 1,
            "epsilon": epsilon,
            "inference": "bayes",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def mmd_newsvendor_1d() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 20     
    num_posterior_samples = 20    
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, inference), contamination, epsilon in itertools.product(
        [
            ("dro_bas_mmd", "contaminated_exp", "exponential", "npl_mmd"),     # misspecified
            ("empirical_mmd", "contaminated_exp", "empirical", "empirical"),            # empirical
            # ("dro_bas_mmd", "exponential", "exponential"),          # well specified
            # ("empirical_mmd", "exponential", "empirical"),                 # empirical
            ("kl_dro_bas", "contaminated_exp", "exponential", "bayes"),
            ("kl_bdro", "contaminated_exp", "exponential", "bayes"),
            # ("dro_bas_mmd", "contaminated_normal", "gaussian_known_var"),     # misspecified
            # ("empirical_mmd", "contaminated_normal", "empirical"),            # empirical
            # ("dro_bas_mmd", "normal", "gaussian_known_var"),          # well specified
            # ("empirical_mmd", "normal", "empirical"),                 # empirical
            # ("dro_bas_mmd", "truncated_normal", "gaussian"),     # misspecified
            # ("empirical_mmd", "truncated_normal", "empirical"),            # empirical
            # ("dro_bas_mmd", "normal", "gaussian"),          # well specified
            # ("empirical_mmd", "normal", "empirical"),                 # empirical
            # ("dro_bas_mmd", "student_t", "gaussian_known_var"),     # misspecified
            # ("empirical_mmd", "student_t", "empirical"),            # empirical
        ],
        [0.0, 0.05, 0.1],
        BAS_DRO_EPSILON_SET,
    ):
        if inference == "bayes":
            posterior = "gamma"
        else:
            posterior = "npl"
        if algorithm == "kl_dro_bas":
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = 400
            num_posterior_samples = 1
        # else:
        #     inference = "npl_mmd"
        # contamination = 0.0
        # if dgp == "contaminated_exp" or dgp == "contaminated_normal":
        #     contamination = CONTAMINATION_LEVEL
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": 1,
            "epsilon": epsilon,
            "inference": inference,
            "lengthscale": -1.0,        
            "likelihood": likelihood,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def kl_portfolio(mmc2_dir: Path) -> List[Dict]:
    """KL Portfolio experiment with DRO-BAS vs BDRO"""
    experiment = []
    for algorithm, dgp, epsilon in itertools.product(
        ["kl_dro_bas", "kl_bdro"],
        ["DowJones"],
        PORTFOLIO_EPSILON_SET,
    ):
        num_likelihood_samples = 1
        num_posterior_samples_list = [1]
        if algorithm == "kl_bdro":
            # likelihood in closed form
            num_posterior_samples_list = [5, 10, 30]

        for num_posterior_samples in num_posterior_samples_list:
            returns_df = pd.read_excel(mmc2_dir / "Datasets" / dgp / f"{dgp}.xlsx", sheet_name="Assets_Returns", header=None)
            num_time_windows = get_num_time_windows(len(returns_df))
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "portfolio",
                "dgp": dgp,
                "dim": 5,
                "epsilon": epsilon,
                "inference": "bayes",
                "lengthscale": -1.0,
                "likelihood": "multivariate_normal",
                "num_likelihood_samples": num_likelihood_samples,
                "num_observations": IN_SAMPLE_TIME_WINDOW,
                "num_posterior_samples": num_posterior_samples,
                "num_replications": num_time_windows,
                "num_test_observations": OUT_OF_SAMPLE_TIME_WINDOW,
                "posterior": "normal_inverse_wishart",
                "uuid": str(uuid4()),  # uniquely identify a run
            }
            experiment.append(params)
    return experiment

def compare_solve() -> List[Dict]:
    """Compares the original grid-search algorithm and cvxpy algorithms"""
    experiment = []
    for algorithm, dgp, epsilon, (posterior, likelihood) in itertools.product(
        ["kl_bdro", "bdro_grid_search", "kl_dro_bas"],
        ["truncated_normal"],
        [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        [("gamma", "exponential"), ("normal_gamma", "normal")],
    ):
        num_posterior_samples = NUM_POSTERIOR_SAMPLES
        if algorithm == "kl_dro_bas" and posterior != "normal_gamma":
            continue  # skip if the posterior doesn't match our algorithm
        elif algorithm == "kl_dro_bas":
            # we calculate the posterior exactly in closed form!
            num_posterior_samples = 1
        params = {
            "algorithm": algorithm,
            "contamination": 0.0,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": 1,
            "epsilon": epsilon,
            "inference": "bayes",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_likelihood_samples": NUM_LIKELIHOOD_SAMPLES,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment
