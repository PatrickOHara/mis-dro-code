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
    BAS_NUM_REPLICATIONS,
    BAS_TOTAL_MODEL_SAMPLES,
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
    ROBAS_DRO_EPSILON_SET,
)
from .dataset import get_num_time_windows


class ExperimentName(StrEnum):
    """Names of experiments"""

    kl_newsvendor_1d = "kl_newsvendor_1d"
    kl_newsvendor_5d = "kl_newsvendor_5d"
    mmd_newsvendor_1d = "mmd_newsvendor_1d"
    mmd_newsvendor_1d_missp = "mmd_newsvendor_1d_missp"
    compare_solve = "compare_solve"
    mmd_newsvendor_5d = "mmd_newsvendor_5d"
    mmd_portfolio = "mmd_portfolio"
    kl_portfolio = "kl_portfolio"
    kl_newsvendor_exp_1d = "kl_newsvendor_exp_1d"
    mmd_newsvendor_exp_1d = "mmd_newsvendor_exp_1d"

    def is_portfolio(self) -> bool:
        return self in (ExperimentName.kl_portfolio, ExperimentName.mmd_portfolio)



def get_experiment(experiment_name: ExperimentName, dataset_dir: Optional[Path] = None) -> List[Dict]:
    """Returns the experiment associated with the name"""
    function_lookup = {
        ExperimentName.kl_newsvendor_1d: kl_newsvendor_1d,
        ExperimentName.kl_newsvendor_5d: kl_newsvendor_5d,
        ExperimentName.mmd_newsvendor_1d: mmd_newsvendor_1d,
        ExperimentName.mmd_newsvendor_1d_missp: mmd_newsvendor_1d_missp,
        ExperimentName.compare_solve: compare_solve,
        ExperimentName.mmd_newsvendor_5d: mmd_newsvendor_5d,
        ExperimentName.mmd_portfolio: mmd_portfolio,
        ExperimentName.kl_portfolio: kl_portfolio,
        ExperimentName.kl_newsvendor_exp_1d: kl_newsvendor_exp_1d,
        ExperimentName.mmd_newsvendor_exp_1d: mmd_newsvendor_exp_1d
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
    for total_model_samples, algorithm, num_observations, (dgp, likelihood, posterior), epsilon in itertools.product(
        BAS_TOTAL_MODEL_SAMPLES,
        ["kl_dro_bas", "kl_pp", "kl_bdro"],
        [20],
        [
            ("multivariate_normal", "multivariate_normal", "normal_inverse_wishart"),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        params = {
            "algorithm": algorithm,
            "contamination": 0.0,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": 5,
            "epsilon": epsilon,
            "ignore_dpp": True,
            "inference": "bayes",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "njobs": 1,
            "num_likelihood_samples": get_num_likelihood_samples(total_model_samples, algorithm),
            "num_observations": num_observations,
            "num_posterior_samples": get_num_posterior_samples(total_model_samples, algorithm),
            "num_replications": BAS_NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def get_num_likelihood_samples(num_total_samples: int, algorithm: str) -> int:
    if algorithm == "kl_bdro":
        return int(np.sqrt(num_total_samples))
    if algorithm in ("kl_dro_bas", "kl_pp"):
        return num_total_samples
    raise NotImplementedError()

def get_num_posterior_samples(num_total_samples: int, algorithm: str) -> int:
    if algorithm == "kl_bdro":
        return int(np.sqrt(num_total_samples))
    if algorithm in ("kl_dro_bas", "kl_pp"):
        return 1
    raise NotImplementedError()

def kl_newsvendor_1d() -> List[Dict]:
    """KL univariate newsvendor: compare our Bayesian ambiguity set against Bayesian DRO"""
    experiment = []
    for total_model_samples, algorithm, num_observations, (dgp, likelihood, posterior), epsilon in itertools.product(
        BAS_TOTAL_MODEL_SAMPLES,
        ["kl_pp", "kl_dro_bas", "kl_bdro"],
        [5, 20, 100],
        [
            # ("normal", "normal", "normal_gamma"),
            # ("truncated_normal", "normal", "normal_gamma"),
            ("exponential", "exponential", "gamma"),
            # ("contaminated_exp", "exponential", "gamma"),
        ],
        BAS_DRO_EPSILON_SET,
    ):
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
            "ignore_dpp": True,
            "inference": "bayes",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "njobs": 1,
            "num_likelihood_samples": get_num_likelihood_samples(total_model_samples, algorithm),
            "num_observations": num_observations,
            "num_posterior_samples": get_num_posterior_samples(total_model_samples, algorithm),
            "num_replications": BAS_NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def kl_newsvendor_exp_1d() -> List[Dict]:
    experiment = []
    total_model_samples = 900
    num_replications = 100
    num_test_observations = NUM_TEST_OBSERVATIONS
    num_certify_points = 200
    for contamination, num_observations, (algorithm, dgp, likelihood, inference, posterior, dim), epsilon in itertools.product(
        [0.0],
        [20],
        [   

            ("kl_dro_bas", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "bayes", "multivariate_normal_known_cov", 5),
            ("kl_pp", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "bayes", "multivariate_normal_known_cov", 5),
            # ("kl_dro_bas", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_pp", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "npl_mmd", "npl", 5),
            # ("kl_dro_bas", "contaminated_normal", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_pp", "contaminated_normal", "normal_known_var", "bayes",  "normal_known_var", 1),
            # ("kl_bdro", "contaminated_normal", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "contaminated_normal", "normal_known_var", "npl_mmd", "npl", 1),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if algorithm in ["kl_bdro"]:
            num_posterior_samples = 90 #int(np.sqrt(total_model_samples))
            num_likelihood_samples = 10 #int(np.sqrt(total_model_samples))
        if algorithm in ["kl_dro_bas", "kl_pp"]:
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = total_model_samples
            num_posterior_samples = 1
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": dim,
            "epsilon": epsilon,
            "inference": inference,
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": num_replications,
            "num_test_observations": num_test_observations,
            "num_certify_points": num_certify_points,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def mmd_newsvendor_exp_1d() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 10     
    num_posterior_samples = 90  
    num_observations = 20  
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, inference, posterior, dim), contamination, epsilon in itertools.product(
        [
            # ("dro_bas_mmd", "contaminated_normal", "normal_known_var", "npl_mmd", "npl", 1), 
            # ("empirical_mmd", "contaminated_normal", "empirical", "empirical", "npl", 1)
            # ("dro_bas_mmd", "bimodal_univariate_gaussian", "normal_known_var", "npl_mmd", "npl", 1),
            # ("empirical_mmd", "bimodal_univariate_gaussian", "empirical", "empirical", "empirical", 1)
            ("dro_bas_mmd", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "npl_mmd", "npl", 5), 
            ("empirical_mmd", "bimodal_multivariate_gaussian", "empirical", "empirical", "empirical", 5),

        ],
        [0.0],
        ROBAS_DRO_EPSILON_SET,
    ):
        # if inference == "bayes":
        #     posterior = "gamma"
        # else:
        #     posterior = "npl"
        # if algorithm == "kl_dro_bas":
        #     # we calculate the posterior exactly in closed form!
        #     num_likelihood_samples = 900
        #     num_posterior_samples = 1
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "newsvendor",
            "dgp": dgp,
            "dim": dim,
            "epsilon": epsilon,
            "inference": inference,
            "lengthscale": -1.0,        
            "likelihood": likelihood,
            "num_certify_points": 200,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": 100,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def mmd_newsvendor_1d() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 30     
    num_posterior_samples = 30    
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, inference, posterior), contamination, num_observations, epsilon in itertools.product(
        [
            # ("dro_bas_mmd", "contaminated_exp", "exponential", "npl_mmd", "npl"),     # misspecified
            # ("empirical_mmd", "contaminated_exp", "empirical", "empirical", "empirical"),            # empirical
            # ("dro_bas_mmd", "contaminated_exp_large_outliers", "exponential", "npl_mmd", "npl"),     # misspecified
            # ("empirical_mmd", "contaminated_exp_large_outliers", "empirical", "empirical", "empirical"),            # empirical
            # ("dro_bas_mmd", "contaminated_exp_small_outliers", "exponential", "npl_mmd", "npl"),
            # ("empirical_mmd", "contaminated_exp_small_outliers", "empirical", "empirical", "empirical"),
            # ("dro_bas_mmd", "exponential", "exponential", "npl_mmd"),          # well specified
            # ("empirical_mmd", "exponential", "empirical", "empirical"),                 # empirical
            # ("kl_dro_bas", "contaminated_exp", "exponential", "bayes", "gamma"),
            # ("kl_bdro", "contaminated_exp", "exponential", "bayes", "gamma"),
            # ("kl_dro_bas", "contaminated_exp_large_outliers", "exponential", "bayes", "gamma"),
            # ("kl_bdro", "contaminated_exp_large_outliers", "exponential", "bayes", "gamma"),
            # ("kl_dro_bas", "contaminated_exp_small_outliers", "exponential", "bayes", "gamma"),
            # ("kl_bdro", "contaminated_exp_small_outliers", "exponential", "bayes", "gamma"),
            ("kl_pp", "contaminated_exp", "exponential", "bayes", "gamma"),
            ("kl_pp", "contaminated_exp_large_outliers", "exponential", "bayes", "gamma"),
            ("kl_pp", "contaminated_exp_small_outliers", "exponential", "bayes", "gamma"),
            # ("kl_bdro", "contaminated_exp", "exponential", "npl_mmd")
            # ("kl_dro_bas", "exponential", "exponential", "bayes"),
            # ("kl_bdro", "exponential", "exponential", "bayes")
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
        [0.0, 0.1, 0.2],   
        [100],
        ROBAS_DRO_EPSILON_SET,
    ):
        # if inference == "bayes":
        #     posterior = "gamma"
        # else:
        #     posterior = "npl"
        # contamination = 0.0
        # if dgp == "contaminated_exp":
        #     contamination = 0.05
        if algorithm == "kl_dro_bas":
            # we calculate the posterior exactly in closed form!
            num_likelihood_samples = 900
            num_posterior_samples = 1
        if algorithm == "kl_pp":
            num_likelihood_samples = 900
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
            "num_certify_points": 200,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def mmd_newsvendor_1d_missp() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 20     
    num_posterior_samples = 20    
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, inference), epsilon in itertools.product(
        [
            # ("dro_bas_mmd", "contaminated_exp", "exponential", "npl_mmd"),     # misspecified
            # ("empirical_mmd", "contaminated_exp", "empirical", "empirical"),            # empirical
            # ("dro_bas_mmd", "exponential", "exponential", "npl_mmd"),          # well specified
            # ("empirical_mmd", "exponential", "empirical", "empirical"),                 # empirical
            # ("kl_dro_bas", "contaminated_exp", "exponential", "bayes"),
            # ("kl_bdro", "contaminated_exp", "exponential", "bayes"),
            # ("kl_dro_bas", "exponential", "exponential", "bayes"),
            # ("kl_bdro", "exponential", "exponential", "bayes")
            # ("dro_bas_mmd", "contaminated_normal", "gaussian_known_var"),     # misspecified
            # ("empirical_mmd", "contaminated_normal", "empirical"),            # empirical
            # ("dro_bas_mmd", "normal", "gaussian_known_var"),          # well specified
            # ("empirical_mmd", "normal", "empirical"),                 # empirical
            # ("dro_bas_mmd", "truncated_normal", "gaussian"),     # misspecified
            # ("empirical_mmd", "truncated_normal", "empirical"),            # empirical
            # ("dro_bas_mmd", "normal", "gaussian"),          # well specified
            # ("empirical_mmd", "normal", "empirical"),                 # empirical
            # ("dro_bas_mmd", "student_t", "normal", "npl_mmd"),     # misspecified
            # ("empirical_mmd", "student_t", "empirical", "empirical"),            # empirical
            # ("kl_dro_bas", "student_t", "normal", "bayes"),
            ("kl_bdro", "student_t", "normal", "npl_mmd"),   #bayes

        ],
        BAS_DRO_EPSILON_SET,
    ):
        if inference == "bayes":
            posterior = "normal_gamma"
        else:
            posterior = "npl"
        contamination = 0.0
        if dgp == "contaminated_exp":
            contamination = 0.1
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

def mmd_newsvendor_5d() -> List[Dict]:
    """MMD univariate newsvendor: compare our MMD Bayesian ambiguity set against empirical kernel DRO"""
    experiment = []
    num_likelihood_samples = 20     
    num_posterior_samples = 20   
    num_observations = 400 
    # NOTE when using empirical, set likelihood to 'empirical'
    # NOTE do not set up all the below combinations in one experiment to preserve memory
    for (algorithm, dgp, likelihood, contamination), epsilon in itertools.product(
        [
            ("dro_bas_mmd", "cont_multivariate_normal", "multivariate_normal_known_cov", 0.05),     # misspecified
            ("empirical_mmd", "cont_multivariate_normal", "empirical", 0.05),            # empirical
            ("dro_bas_mmd", "cont_multivariate_normal", "multivariate_normal_known_cov", 0.1),     # misspecified
            ("empirical_mmd", "cont_multivariate_normal", "empirical", 0.1),            # empirical
            ("dro_bas_mmd", "multivariate_normal_known_cov", "multivariate_normal_known_cov", 0.0),          # well specified
            ("empirical_mmd", "multivariate_normal_known_cov", "empirical", 0.0),                 # empirical
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if likelihood == "empirical":
            inference = "empirical"
        else:
            inference = "npl_mmd"
        # contamination = 0.0
        # if dgp == "contaminated_exp" or dgp == "contaminated_normal" or dgp == "cont_multivariate_normal":
            # contamination = CONTAMINATION_LEVEL
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dgp": dgp,
            "dim": 5,
            "epsilon": epsilon,
            "inference": inference,
            "lengthscale": -1.0,        
            "likelihood": likelihood,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": num_observations,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": NUM_REPLICATIONS,
            "num_test_observations": NUM_TEST_OBSERVATIONS,
            "posterior": "npl",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def mmd_portfolio(mmc2_dir: Path) -> List[Dict]:
    """MMD portfolio experiment"""
    experiment = []
    dgp = "DowJones"
    num_likelihood_samples = 10  
    num_posterior_samples = 90
    epsilon_set = []
    for epsilon in ROBAS_DRO_EPSILON_SET:
        if epsilon <= 0.2:
            epsilon_set.append(epsilon)
    returns_df = pd.read_excel(mmc2_dir / "Datasets" / dgp / f"{dgp}.xlsx", sheet_name="Assets_Returns", header=None)
    num_time_windows = get_num_time_windows(len(returns_df))
    num_stocks = len(returns_df.columns)
    # NOTE when using empirical, set likelihood to 'empirical'
    for (algorithm, likelihood), epsilon in itertools.product(
        [
            ("dro_bas_mmd", "multivariate_normal"),
            ("empirical_mmd", "empirical"),
        ],
        epsilon_set,
    ):
        if likelihood == "empirical":
            inference = "empirical"
        else:
            inference = "npl_mmd"
        params = {
            "algorithm": algorithm,
            "contamination": 0.0,
            "dataset": "portfolio",
            "dgp": dgp,
            "dim": num_stocks,
            "epsilon": epsilon,
            "inference": inference,
            "lengthscale": -1.0,        
            "likelihood": likelihood,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": IN_SAMPLE_TIME_WINDOW,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": num_time_windows,
            "num_test_observations": OUT_OF_SAMPLE_TIME_WINDOW,
            "posterior": "npl",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def kl_portfolio(mmc2_dir: Path) -> List[Dict]:
    """KL Portfolio experiment with DRO-BAS vs BDRO"""
    experiment = []
    for algorithm, dgp, epsilon in itertools.product(
        ["kl_dro_bas", "kl_bdro", "kl_pp"],
        ["DowJones"],
        PORTFOLIO_EPSILON_SET,
    ):
        num_samples_list = [1]
        if algorithm in ("kl_bdro", "kl_pp"):
            num_samples_list = [100, 400, 900]            

        for num_samples in num_samples_list:
            returns_df = pd.read_excel(mmc2_dir / "Datasets" / dgp / f"{dgp}.xlsx", sheet_name="Assets_Returns", header=None)
            num_time_windows = get_num_time_windows(len(returns_df))
            num_stocks = len(returns_df.columns)

            if algorithm == "kl_dro_bas":
                num_likelihood_samples = 1
                num_posterior_samples = 1
            elif algorithm == "kl_pp":
                num_likelihood_samples = num_samples
                num_posterior_samples = 1
            elif algorithm == "kl_bdro":
                num_posterior_samples = num_samples

            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "portfolio",
                "dgp": dgp,
                "dim": num_stocks,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": "bayes",
                "lengthscale": -1.0,
                "likelihood": "multivariate_normal",
                "njobs": 1,
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
