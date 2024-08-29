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
    BAS_DRO_EPSILON_SET,
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
    exp_bayes_newsvendor = "exp_bayes_newsvendor"
    normal_bayes_newsvendor = "normal_bayes_newsvendor"


def get_experiment(experiment_name: ExperimentName) -> List[Dict]:
    """Returns the experiment associated with the name"""
    function_lookup = {
        ExperimentName.newsvendor_1d: newsvendor_1d,
        ExperimentName.compare_solve: compare_solve,
        ExperimentName.exp_bayes_newsvendor: exp_bayes_newsvendor,
        ExperimentName.normal_bayes_newsvendor: normal_bayes_newsvendor,
    }
    try:
        return function_lookup[experiment_name]()
    except KeyError as e:
        raise KeyError(
            f"Please add {experiment_name} as a key in the function lookup dictionary"
        ) from e

def exp_bayes_newsvendor() -> List[Dict]:
    """Compare our Bayesian ambiguity set against Bayesian DRO with exponential likelihood"""
    experiment = []
    for n_total_samples_sqrt, algorithm, dgp, epsilon in itertools.product(
        [10, 20, 30, 50, 100],
        ["our_kl_bdro", "kl_bdro"],
        ["exponential", "contaminated_exp", "truncated_normal"],
        EPSILON_SET,
    ):
        if algorithm == "kl_bdro":
            num_posterior_samples = n_total_samples_sqrt
            num_likelihood_samples = n_total_samples_sqrt
        if algorithm == "our_kl_bdro":
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = n_total_samples_sqrt**2  # NOTE temporary experimental value
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
            "likelihood": "exponential",
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": "gamma",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def normal_bayes_newsvendor() -> List[Dict]:
    """Compare our Bayesian ambiguity set against Bayesian DRO with normal likelihood"""
    experiment = []
    for n_total_samples_sqrt, algorithm, dgp, epsilon in itertools.product(
        # [10, 20, 30, 50, 100],
        [10, 20, 30],
        ["our_kl_bdro", "kl_bdro"],
        ["normal", "truncated_normal"],
        BAS_DRO_EPSILON_SET,
    ):
        if algorithm == "kl_bdro":
            num_posterior_samples = n_total_samples_sqrt
            num_likelihood_samples = n_total_samples_sqrt
        if algorithm == "our_kl_bdro":
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = n_total_samples_sqrt**2  # NOTE temporary experimental value
            num_posterior_samples = 1
        contamination = 0.0
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dgp": dgp,
            "epsilon": epsilon,
            "inference": "bayes",
            "lengthscale": -1.0,
            "likelihood": "normal",
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": "normal_gamma",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def newsvendor_1d() -> List[Dict]:
    """Vary epsilon and compare Bayesian DRO with Bayes/NPL inference"""
    # iterate over each of the parameters
    experiment = []
    for algorithm, dgp, epsilon, inference in itertools.product(
        ["kl_bdro"],
        ["exponential", "truncated_normal", "contaminated_exp", "gamma"],
        EPSILON_SET,
        ["bayes", "npl_wlb", "npl_mmd"],
    ):
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
            "likelihood": "exponential",
            "num_likelihood_samples": NUM_LIKELIHOOD_SAMPLES,
            "num_observations": NUM_OBSERVATIONS,
            "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": "gamma",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment


def compare_solve() -> List[Dict]:
    """Compares the original grid-search algorithm and cvxpy algorithms"""
    experiment = []
    for algorithm, dgp, epsilon, (posterior, likelihood) in itertools.product(
        ["kl_bdro", "bdro_grid_search", "our_kl_bdro"],
        ["truncated_normal"],
        [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        [("gamma", "exponential"), ("normal_gamma", "normal")],
    ):
        num_posterior_samples = NUM_POSTERIOR_SAMPLES
        if algorithm == "our_kl_bdro" and posterior != "normal_gamma":
            continue  # skip if the posterior doesn't match our algorithm
        elif algorithm == "our_kl_bdro":
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
