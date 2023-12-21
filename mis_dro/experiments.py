"""An experiment is a list of dictionaries each containing parameter settings"""

import itertools
from uuid import uuid4
import numpy as np
from bayesian_dro.Bayesian_DRO_continuous import EPSILON_SET
from .constants import *

def epsilon_experiment():
    """Vary epsilon and compare Bayesian DRO with Bayes/NPL posterior"""
    # iterate over each of the parameters
    experiment = []
    for algorithm, dgp, epsilon, posterior in itertools.product(
        ["bayesian_dro"],
        ["truncated_normal", "contaminated_exp"],
        EPSILON_SET,
        ["bayes", "npl"],
    ):
        contamination = np.nan
        if dgp == "contaminated_exp":
            contamination = CONTAMINATION_LEVEL
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dgp": dgp,
            "epsilon": epsilon,
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
