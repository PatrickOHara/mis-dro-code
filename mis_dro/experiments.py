"""An experiment is a list of dictionaries each containing parameter settings.

Each experiment has an `ExperimentName`.
Use the `get_experiment()` function to get the list of dictionaries associated with an experiment name.
"""

from enum import StrEnum
import itertools
from typing import Dict, List
from uuid import uuid4
from bayesian_dro.Bayesian_DRO_continuous import EPSILON_SET
from .constants import (
    CONTAMINATION_LEVEL,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
)

class ExperimentName(StrEnum):
    """Names of experiments"""

    newsvendor_1d = "newsvendor_1d"
    compare_solve = "compare_solve"

def get_experiment(experiment_name: ExperimentName) -> List[Dict]:
    """Returns the experiment associated with the name"""
    function_lookup = {
        ExperimentName.newsvendor_1d: newsvendor_1d,
        ExperimentName.compare_solve: compare_solve,
    }
    try:
        return function_lookup[experiment_name]()
    except KeyError as e:
        raise KeyError(
            f"Please add {experiment_name} as a key in the function lookup dictionary"
        ) from e


def newsvendor_1d() -> List[Dict]:
    """Vary epsilon and compare Bayesian DRO with Bayes/NPL posterior"""
    # iterate over each of the parameters
    experiment = []
    for algorithm, dgp, epsilon, posterior in itertools.product(
        ["bayesian_dro"],
        ["exponential", "truncated_normal", "contaminated_exp", "gamma"],
        EPSILON_SET,
        ["bayes", "wll", "mmd"],
    ):
        contamination = 0.0
        if dgp == "contaminated_exp":
            contamination = CONTAMINATION_LEVEL
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dgp": dgp,
            "epsilon": epsilon,
            "lengthscale": -1.0,
            "num_likelihood_samples": NUM_LIKELIHOOD_SAMPLES,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def compare_solve() -> List[Dict]:
    """Compares the original grid-search algorithm and cvxpy algorithm"""
    experiment = []
    for algorithm, dgp, epsilon, (posterior, likelihood) in itertools.product(
        ["bayesian_dro", "bdro_grid_search", "normal_gamma_dro"],
        ["truncated_normal"],
        [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        [("gamma", "exponential"), ("normal_gamma", "normal")],
    ):
        num_posterior_samples = NUM_POSTERIOR_SAMPLES
        if algorithm == "normal_gamma_dro" and posterior != "normal_gamma":
            continue    # skip if the posterior doesn't match our algorithm
        elif algorithm == "normal_gamma_dro":
            # we calculate the posterior exactly in closed form!
            num_posterior_samples = 1
        params = {
            "algorithm": algorithm,
            "contamination": 0.0,
            "dgp": dgp,
            "epsilon": epsilon,
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
