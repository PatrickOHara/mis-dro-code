"""An experiment is a list of dictionaries each containing parameter settings.

Each experiment has an `ExperimentName`.
Use the `get_experiment()` function to get the list of dictionaries associated with an experiment name.
"""

from enum import StrEnum
import itertools
from typing import Dict, List
from uuid import uuid4
import numpy as np
from .constants import (
    BAS_DRO_EPSILON_SET,
    CONTAMINATION_LEVEL,
    NUM_CERTIFY,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
)


class ExperimentName(StrEnum):
    """Names of experiments"""

    kl_newsvendor_1d = "kl_newsvendor_1d"
    mmd_newsvendor_1d = "mmd_newsvendor_1d"
    compare_solve = "compare_solve"
    mmd_regression_newsvendor_1d = "mmd_regression_newsvendor_1d"


def get_experiment(experiment_name: ExperimentName) -> List[Dict]:
    """Returns the experiment associated with the name"""
    function_lookup = {
        ExperimentName.kl_newsvendor_1d: kl_newsvendor_1d,
        ExperimentName.mmd_newsvendor_1d: mmd_newsvendor_1d,
        ExperimentName.compare_solve: compare_solve,
        ExperimentName.mmd_regression_newsvendor_1d: mmd_regression_newsvendor_1d
    }
    try:
        return function_lookup[experiment_name]()
    except KeyError as e:
        raise KeyError(
            f"Please add {experiment_name} as a key in the function lookup dictionary"
        ) from e

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
            "dgp": dgp,
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
    for (algorithm, dgp, likelihood), epsilon in itertools.product(
        [
            # ("dro_bas_mmd", "contaminated_exp", "exponential"),     # misspecified
            # ("empirical_mmd", "contaminated_exp", "empirical"),            # empirical
            # ("dro_bas_mmd", "exponential", "exponential"),          # well specified
            # ("empirical_mmd", "exponential", "empirical"),                 # empirical
            # ("dro_bas_mmd", "contaminated_normal", "gaussian_known_var"),     # misspecified
            # ("empirical_mmd", "contaminated_normal", "empirical"),            # empirical
            # ("dro_bas_mmd", "normal", "gaussian_known_var"),          # well specified
            # ("empirical_mmd", "normal", "empirical"),                 # empirical
            # ("dro_bas_mmd", "truncated_normal", "gaussian"),     # misspecified
            # ("empirical_mmd", "truncated_normal", "empirical"),            # empirical
            # ("dro_bas_mmd", "normal", "gaussian"),          # well specified
            # ("empirical_mmd", "normal", "empirical"),                 # empirical
            ("dro_bas_mmd", "student_t", "gaussian_known_var"),     # misspecified
            ("empirical_mmd", "student_t", "empirical"),            # empirical
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if likelihood == "empirical":
            inference = "empirical"
        else:
            inference = "npl_mmd"
        contamination = 0.0
        if dgp == "contaminated_exp" or dgp == "contaminated_normal":
            contamination = CONTAMINATION_LEVEL
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dgp": dgp,
            "epsilon": epsilon,
            "inference": inference,
            "lengthscale": -1.0,        # FIXME?
            "likelihood": likelihood,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": "npl",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def mmd_regression_newsvendor_1d() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 20     
    num_posterior_samples = 20      
    # NOTE when using empirical, set likelihood to 'empirical'
    for (algorithm, dgp, likelihood), epsilon in itertools.product(
        [
            ("dro_bas_mmd", "normal_regression", "regression_normal"),     # regression
            # ("empirical_mmd", "normal_regression", "empirical"),            # empirical
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if likelihood == "empirical":
            inference = "empirical"
        else:
            inference = "npl_mmd"
        contamination = 0.0
        if dgp == "contaminated_exp":
            contamination = CONTAMINATION_LEVEL
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dgp": dgp,
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
            "posterior": "npl",
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
            "dgp": dgp,
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
