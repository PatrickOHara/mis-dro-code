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
    SMALL_BAS_DRO_EPSILON_SET,
)
from .dataset import get_num_time_windows, get_portfolio_returns_df, get_num_time_windows_james, get_min_dim_james


class ExperimentName(StrEnum):
    """Names of experiments"""

    kl_newsvendor_1d = "kl_newsvendor_1d"
    kl_newsvendor_5d = "kl_newsvendor_5d"
    mmd_newsvendor_1d = "mmd_newsvendor_1d"
    mmd_newsvendor_1d_missp = "mmd_newsvendor_1d_missp"
    compare_solve = "compare_solve"
    mmd_newsvendor_5d = "mmd_newsvendor_5d"
    mmd_portfolio = "mmd_portfolio"
    mmd_portfolio_synthetic = "mmd_portfolio_synthetic"
    mmd_portfolio_crash = "mmd_portfolio_crash"
    kl_portfolio = "kl_portfolio"
    kl_portfolio_crash = "kl_portfolio_crash"
    kl_portfolio_synthetic = "kl_portfolio_synthetic"
    kl_newsvendor_exp_1d = "kl_newsvendor_exp_1d"
    mmd_newsvendor_exp_1d = "mmd_newsvendor_exp_1d"
    kl_portfolio_james = "kl_portfolio_james"
    # mmd_portfolio_james = "mmd_portfolio_james"
    tv_kl_portfolio_james = "tv_kl_portfolio_james"
    tv_kl_portfolio_james_has_tcosts_in_cost_function = "tv_kl_portfolio_james_has_tcosts_in_cost_function"

    def is_portfolio(self) -> bool:
        return self in (
            ExperimentName.kl_portfolio,
            ExperimentName.mmd_portfolio,
            ExperimentName.kl_portfolio_crash,
            ExperimentName.mmd_portfolio_crash,
            # ExperimentName.tv_kl_portfolio_james  # NOTE: activate under is_james instead, for consistency with kl_portfolio_james regarding whether dim must be supplied
        )
    
    def is_james(self) -> bool:
        return self in (
            ExperimentName.kl_portfolio_james,
            # ExperimentName.mmd_portfolio_james,
            ExperimentName.tv_kl_portfolio_james,
            ExperimentName.tv_kl_portfolio_james_has_tcosts_in_cost_function
        )
    
    def is_temporal_validation(self) -> bool:
        return self in (ExperimentName.tv_kl_portfolio_james, ExperimentName.tv_kl_portfolio_james_has_tcosts_in_cost_function)
    
    def has_tcosts_in_cost_function(self) -> bool:
        return self in (ExperimentName.tv_kl_portfolio_james_has_tcosts_in_cost_function,)



def get_experiment(experiment_name: ExperimentName, dataset_dir_james, risk_free_rates_filename, dataset_dir: Optional[Path] = None, dim: Optional[int] = None, tv_ratio: Optional[str] = None, num_folds_tv: Optional[int] = 1) -> List[Dict]:
    """Returns the experiment associated with the name"""
    function_lookup = {
        ExperimentName.kl_newsvendor_1d: kl_newsvendor_1d,
        ExperimentName.kl_newsvendor_5d: kl_newsvendor_5d,
        ExperimentName.mmd_newsvendor_1d: mmd_newsvendor_1d,
        ExperimentName.mmd_newsvendor_1d_missp: mmd_newsvendor_1d_missp,
        ExperimentName.compare_solve: compare_solve,
        ExperimentName.mmd_newsvendor_5d: mmd_newsvendor_5d,
        ExperimentName.mmd_portfolio: mmd_portfolio,
        ExperimentName.mmd_portfolio_crash: mmd_portfolio_crash,
        ExperimentName.kl_portfolio: kl_portfolio,
        ExperimentName.kl_portfolio_crash: kl_portfolio_crash,
        ExperimentName.kl_portfolio_synthetic: kl_portfolio_synthetic,
        ExperimentName.kl_newsvendor_exp_1d: kl_newsvendor_exp_1d,
        ExperimentName.mmd_newsvendor_exp_1d: mmd_newsvendor_exp_1d,
        ExperimentName.mmd_portfolio_synthetic: mmd_portfolio_synthetic,
        ExperimentName.kl_portfolio_james: kl_portfolio_james,
        # ExperimentName.mmd_portfolio_james: mmd_portfolio_james,
        ExperimentName.tv_kl_portfolio_james: tv_kl_portfolio_james,
        ExperimentName.tv_kl_portfolio_james_has_tcosts_in_cost_function: tv_kl_portfolio_james_has_tcosts_in_cost_function
    }
    try:
        if experiment_name.is_portfolio():
            # NOTE portfolio setup requires a dataset_dir argument
            return function_lookup[experiment_name](dataset_dir)
        elif experiment_name.is_james():
            if experiment_name.is_temporal_validation():
                return function_lookup[experiment_name](dataset_dir_james, risk_free_rates_filename, dim, tv_ratio, num_folds_tv)
            return function_lookup[experiment_name](dataset_dir_james, dim)
        else:
            return function_lookup[experiment_name]()
    except KeyError as e:
        raise KeyError(
            f"Please add {experiment_name} as a key in the function lookup dictionary"
        ) from e

def kl_newsvendor_5d() -> List[Dict]:
    """KL univariate newsvendor: compare our Bayesian ambiguity set against Bayesian DRO"""
    experiment = []
    for algorithm, num_observations, (dgp, likelihood, posterior), epsilon in itertools.product(
        ["kl_pp", "kl_dro_bas", "kl_bdro", "kl_empirical"],
        [NUM_OBSERVATIONS],
        [
            ("multivariate_normal", "multivariate_normal", "normal_inverse_wishart"),
        ],
        BAS_DRO_EPSILON_SET,
    ):
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            total_model_samples_list = BAS_TOTAL_MODEL_SAMPLES
            inference = "bayes"
        for total_model_samples in total_model_samples_list:
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "newsvendor",
                "dgp": dgp,
                "dim": 5,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "lengthscale": -1.0,
                "likelihood": likelihood,
                "njobs": 1,
                "num_likelihood_samples": get_num_likelihood_samples("newsvendor", num_observations, total_model_samples, algorithm),
                "num_observations": num_observations,
                "num_posterior_samples": get_num_posterior_samples("newsvendor", total_model_samples, algorithm),
                "num_replications": BAS_NUM_REPLICATIONS,
                "num_test_observations": NUM_TEST_OBSERVATIONS,
                "posterior": posterior,
                "uuid": str(uuid4()),  # uniquely identify a run
            }
            experiment.append(params)
    return experiment

def get_num_likelihood_samples(dataset: str, num_observations: int, num_total_samples: int, algorithm: str) -> int:
    if algorithm == "kl_bdro" and dataset == "portfolio":
        return 1
    if algorithm == "kl_bdro":
        return int(np.sqrt(num_total_samples))
    if algorithm in ("kl_dro_bas", "kl_pp"):
        return num_total_samples
    if algorithm == "kl_empirical":
        return num_observations
    raise NotImplementedError()

def get_num_posterior_samples(dataset: str, num_total_samples: int, algorithm: str) -> int:
    if algorithm == "kl_bdro" and dataset == "portfolio":
        return num_total_samples
    if algorithm == "kl_bdro":
        return int(np.sqrt(num_total_samples))
    if algorithm in ("kl_dro_bas", "kl_pp"):
        return 1
    if algorithm == "kl_empirical":
        return 1
    raise NotImplementedError()

def kl_newsvendor_1d() -> List[Dict]:
    """KL univariate newsvendor: compare our Bayesian ambiguity set against Bayesian DRO"""
    experiment = []
    for algorithm, num_observations, (dgp, likelihood, posterior), epsilon in itertools.product(
        ["kl_pp", "kl_dro_bas", "kl_bdro", "kl_empirical"],
        [NUM_OBSERVATIONS], # [5, 20, 100],
        [
            ("normal", "normal", "normal_gamma"),
            # ("truncated_normal", "normal", "normal_gamma"),
            ("exponential", "exponential", "gamma"),
            # ("contaminated_exp", "exponential", "gamma"),
        ],
        BAS_DRO_EPSILON_SET,
        # SMALL_BAS_DRO_EPSILON_SET,
    ):
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            # total_model_samples_list = BAS_TOTAL_MODEL_SAMPLES
            total_model_samples_list = [3600, 10000]
            inference = "bayes"
        contamination = 0.0
        if dgp == "contaminated_exp":
            contamination = CONTAMINATION_LEVEL
        for total_model_samples in total_model_samples_list:
            params = {
                "algorithm": algorithm,
                "contamination": contamination,
                "dataset": "newsvendor",
                "dgp": dgp,
                "dim": 1,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "lengthscale": -1.0,
                "likelihood": likelihood,
                "njobs": 1,
                "num_likelihood_samples": get_num_likelihood_samples("newsvendor", num_observations, total_model_samples, algorithm),
                "num_observations": num_observations,
                "num_posterior_samples": get_num_posterior_samples("newsvendor", total_model_samples, algorithm),
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
        [0.0], #, 0.1, 0.2],
        [20],
        [   

            # ("kl_dro_bas", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "bayes", "multivariate_normal_known_cov", 5),
            # ("kl_pp", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "bayes", "multivariate_normal_known_cov", 5),
            ("kl_bdro", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "bayes", "multivariate_normal_known_cov", 5)
            # ("kl_dro_bas", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_pp", "bimodal_univariate_gaussian", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "npl_mmd", "npl", 5),
            # ("kl_dro_bas", "contaminated_normal", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_pp", "contaminated_normal", "normal_known_var", "bayes",  "normal_known_var", 1),
            # ("kl_bdro", "contaminated_normal", "normal_known_var", "bayes", "normal_known_var", 1),
            # ("kl_bdro", "contaminated_normal", "normal_known_var", "npl_mmd", "npl", 1),
            # ("kl_pp", "contaminated_exp", "exponential", "bayes", "gamma", 1),
            # ("kl_dro_bas", "contaminated_exp", "exponential", "bayes", "gamma", 1),
            # ("kl_bdro", "contaminated_exp", "exponential", "bayes", "gamma", 1),
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
            "njobs": 1,
            "epsilon": epsilon,
            "ignore_dpp": True,
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
            # ("dro_bas_mmd", "bimodal_multivariate_gaussian", "multivariate_normal_known_cov", "npl_mmd", "npl", 5), 
            # ("empirical_mmd", "bimodal_multivariate_gaussian", "empirical", "empirical", "empirical", 5),
            ("dro_bas_mmd", "contaminated_exp", "exponential", "npl_mmd", "npl", 1), 
            ("empirical_mmd", "contaminated_exp", "exponential", "empirical", "empirical", 1),

        ],
        [0.0, 0.1, 0.2],
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
            "kernel_name": "k_jax",
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
            "kernel_name": "k_jax",
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
            "kernel_name": "k_jax",
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
            "kernel_name": "k_jax",
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

def kl_portfolio_synthetic() -> List[Dict]:
    """MMD portfolio experiment"""
    experiment = []
    dgp = "portfolio_contaminated_multivariate_normal"
    num_replications = 100
    dim = 5
    epsilon_set = BAS_DRO_EPSILON_SET
    for algorithm, contamination, epsilon, in itertools.product(
        [
            "kl_dro_bas",
            "kl_pp",
            "kl_bdro",
        ],
        [0.0, 0.1, 0.2],
        epsilon_set,
    ):
        inference = "bayes"
        likelihood = "multivariate_normal"
        posterior = "normal_inverse_wishart"
        if algorithm == "kl_dro_bas":
            num_likelihood_samples = 1
            num_posterior_samples = 1
        elif algorithm == "kl_pp":
            num_likelihood_samples = 900
            num_posterior_samples = 1
        elif algorithm == "kl_bdro":
            num_likelihood_samples = 10
            num_posterior_samples = 90
        else:
            raise ValueError(algorithm)
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "portfolio_synthetic",
            "dgp": dgp,
            "dim": dim,
            "epsilon": epsilon,
            "eta": np.nan,
            "ignore_dpp": True,
            "inference": inference,
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "normalise": False,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": 100,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": num_replications,
            "num_test_observations": 100,
            "posterior": posterior,
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def mmd_portfolio_synthetic() -> List[Dict]:
    """MMD portfolio experiment"""
    experiment = []
    dgp = "portfolio_contaminated_multivariate_normal"
    num_likelihood_samples = 10  
    num_posterior_samples = 90
    num_replications = 100
    dim = 5
    # epsilon_set = [0.0001, 0.001, 0.01, 0.1, 1.0]
    epsilon_set = ROBAS_DRO_EPSILON_SET
    for (algorithm, likelihood), contamination, epsilon, in itertools.product(
        [
            ("dro_bas_mmd", "multivariate_normal"),
            ("empirical_mmd", "empirical"),
        ],
        [0.0, 0.1, 0.2],
        epsilon_set,
    ):
        if algorithm == "empirical_mmd":
            inference = "empirical"
            posterior = "empirical"
            eta = np.nan
        elif algorithm == "dro_bas_mmd":
            inference = "npl_mmd"
            posterior = "npl"
            eta = 0.1
        else:
            raise ValueError(algorithm)
        params = {
            "algorithm": algorithm,
            "contamination": contamination,
            "dataset": "portfolio_synthetic",
            "dgp": dgp,
            "dim": dim,
            "epsilon": epsilon,
            "eta": eta,
            "inference": inference,
            "kernel_name": "k_comp",
            "lengthscale": -1.0,
            "likelihood": likelihood,
            "normalise": False,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": 100,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": num_replications,
            "num_test_observations": 100,
            "posterior": posterior,
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
    returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
    num_time_windows = get_num_time_windows(len(returns_df))
    num_stocks = len(returns_df.columns)
    # NOTE when using empirical, set likelihood to 'empirical'
    for (algorithm, likelihood), epsilon, in itertools.product(
        [
            ("dro_bas_mmd", "multivariate_normal"),
            ("empirical_mmd", "empirical"),
        ],
        epsilon_set,
    ):
        if likelihood == "empirical":
            inference = "empirical"
            eta_set = [np.nan]
        else:
            inference = "npl_mmd"
            eta_set = [0.1]
        for eta in eta_set:
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "portfolio",
                "dgp": dgp,
                "dim": num_stocks,
                "epsilon": epsilon,
                "eta": eta,
                "inference": inference,
                "kernel_name": "k_comp",        
                "lengthscale": -1.0,
                "likelihood": likelihood,
                "normalise": False,
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

def mmd_portfolio_crash(mmc2_dir: Path) -> List[Dict]:
    """MMD portfolio experiment"""
    experiment = []
    dgp = "DowJones"
    num_likelihood_samples = 10  
    num_posterior_samples = 90
    dgp = "DowJones-crash"
    epsilon_set = []
    for epsilon in ROBAS_DRO_EPSILON_SET:
        if epsilon <= 0.2:
            epsilon_set.append(epsilon)
    returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
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
            "kernel_name": "k_comp",
            "lengthscale": -1.0,        
            "likelihood": likelihood,
            "num_certify_points": NUM_CERTIFY,
            "num_likelihood_samples": num_likelihood_samples,
            "num_observations": IN_SAMPLE_TIME_WINDOW,
            "num_posterior_samples": num_posterior_samples,
            "num_replications": 1,
            "num_test_observations": IN_SAMPLE_TIME_WINDOW,
            "posterior": "npl",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment

def kl_portfolio_crash(mmc2_dir: Path) -> List[Dict]:
    """Portfolio experiment with a stock crash"""
    experiment = []
    dgp = "DowJones-crash"
    num_samples = 900
    for algorithm, epsilon in itertools.product(
        ["kl_dro_bas", "kl_bdro", "kl_pp"],
        PORTFOLIO_EPSILON_SET,
    ):
        returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
        num_stocks = len(returns_df.columns)
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
            "num_likelihood_samples": get_num_likelihood_samples("portfolio", IN_SAMPLE_TIME_WINDOW, num_samples, algorithm),
            "num_observations": IN_SAMPLE_TIME_WINDOW,
            "num_posterior_samples": get_num_posterior_samples("portfolio", num_samples, algorithm),
            "num_replications": 1,
            "num_test_observations": IN_SAMPLE_TIME_WINDOW,
            "posterior": "normal_inverse_wishart",
            "uuid": str(uuid4()),  # uniquely identify a run
        }
        experiment.append(params)
    return experiment
        
def kl_portfolio(mmc2_dir: Path) -> List[Dict]:
    """KL Portfolio experiment with DRO-BAS vs BDRO"""
    experiment = []
    for algorithm, dgp, epsilon in itertools.product(
        ["kl_dro_bas", "kl_bdro", "kl_pp", "kl_empirical"],
        ["DowJones"],
        PORTFOLIO_EPSILON_SET,
    ):
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            if algorithm == "kl_dro_bas":
                total_model_samples_list = [1]
            elif algorithm == "kl_pp":
                total_model_samples_list = [900, 3600]
            elif algorithm == "kl_bdro":
                total_model_samples_list = [900]
            else:
                raise ValueError("Provide a supported algorithm")
            likelihood = "multivariate_normal"
            posterior = "normal_inverse_wishart"
            inference = "bayes"
        for num_samples in total_model_samples_list:
            returns_df = get_portfolio_returns_df(mmc2_dir, dgp)
            num_time_windows = get_num_time_windows(len(returns_df))
            num_stocks = len(returns_df.columns)
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "portfolio",
                "dgp": dgp,
                "dim": num_stocks,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "likelihood": likelihood,
                "njobs": 1,
                "normalise": False,
                "num_likelihood_samples": get_num_likelihood_samples("portfolio", IN_SAMPLE_TIME_WINDOW, num_samples, algorithm),
                "num_observations": IN_SAMPLE_TIME_WINDOW,
                "num_posterior_samples": get_num_posterior_samples("portfolio", num_samples, algorithm),
                "num_replications": num_time_windows,
                "num_test_observations": OUT_OF_SAMPLE_TIME_WINDOW,
                "posterior": posterior,
                "uuid": str(uuid4()),  # uniquely identify a run
            }
            experiment.append(params)
    return experiment

def kl_portfolio_james(dataset_dir_james, dim: Optional[int]) -> List[Dict]:
    """KL Portfolio experiment with DRO-BAS vs BDRO"""
    if dim:
        assert dim <= get_min_dim_james(dataset_dir_james)
    experiment = []
    for algorithm, dgp, epsilon in itertools.product(
        ["kl_dro_bas", "kl_bdro", "kl_pp", "kl_empirical"],
        ["james"],
        PORTFOLIO_EPSILON_SET,
    ):
        if algorithm == "kl_empirical":
            total_model_samples_list = [0]
            likelihood = "empirical"
            posterior = "empirical"
            inference = "empirical"
        else:
            if algorithm == "kl_dro_bas":
                total_model_samples_list = [1]
            elif algorithm == "kl_pp":
                total_model_samples_list = [900, 3600]
            elif algorithm == "kl_bdro":
                total_model_samples_list = [900]
            else:
                raise ValueError("Provide a supported algorithm")
            likelihood = "multivariate_normal"
            posterior = "normal_inverse_wishart"
            inference = "bayes"
        for num_samples in total_model_samples_list:
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "james",
                "dataset_dir_james": dataset_dir_james,
                "dgp": dgp,
                "dim": dim,
                "epsilon": epsilon,
                "ignore_dpp": True,
                "inference": inference,
                "likelihood": likelihood,
                "njobs": 1,
                "normalise": False, # TODO: double-check this one
                "num_likelihood_samples": get_num_likelihood_samples("portfolio", 52, num_samples, algorithm),
                "num_observations": 52,
                "num_posterior_samples": get_num_posterior_samples("portfolio", num_samples, algorithm),
                "num_replications": get_num_time_windows_james(dataset_dir_james),
                "num_test_observations": 13, # TODO: dynamically measure the length of a test window here and in tv_kl_portfolio_james
                "posterior": posterior,
                "uuid": str(uuid4()),  # uniquely identify a run
            }
            experiment.append(params)
    return experiment

# NOTE: parameters changed
def tv_kl_portfolio_james(dataset_dir_james, risk_free_rates_filename, dim: Optional[int], tv_ratio: Optional[str], NUM_FOLDS: int = 1) -> List[Dict]:
    """Temporal-validation KL univariate portfolio for selecting epsilon"""
    if dim:
        assert dim <= get_min_dim_james(dataset_dir_james)
    experiment = []
    dgp = "james"   # NOTE: changed from DowJones
    for algorithm in ["kl_dro_bas", "kl_bdro", "kl_pp"]:
        likelihood = "multivariate_normal"
        posterior = "normal_inverse_wishart"
        inference = "bayes"

        # NOTE: in what I took from Patrick's branch, total_model_samples was fixed to 900 instead of the below. This doesn't make a difference in the case of kl_dro_bas: num_likelihood_samples is now 1 rather than 900 from it, but this variable isn't actually used due to closed-form, and num_posterior_samples ends up as 1 either way. In the case of kl_pp, it's just the case that 3600 is being left out, which may be good regarding consistency with BDRO. And for BDRO, it's only 900 below anyway.
        if algorithm == "kl_dro_bas":
            total_model_samples_list = [1]
        elif algorithm == "kl_pp":
            total_model_samples_list = [900, 3600]
        elif algorithm == "kl_bdro":
            total_model_samples_list = [900]

        for total_model_samples in total_model_samples_list:
            base_params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "james", # NOTE: changed from portfolio
                "dataset_dir_james": dataset_dir_james, # NOTE: added
                "dgp": dgp,
                "dim": dim,
                "ignore_dpp": True,
                "inference": inference,
                "likelihood": likelihood,
                "njobs": 1,

                # TODO: I have changed the below, but should I let Patrick know that he may need to change "newsvendor" for his results, and are my changes acceptable? More specifically regarding changing "newsvendor" to "portfolio". Seems to affect BDRO alone--30 is used as num_likelihood_samples and num_posterior samples in the last case, now it's 1 and 900 respectively. At least that's now consistent with kl_portfolio_james
                # TODO: should the number of samples and observations be different in the validation runs? Or no because that would change the optimisation method, and we don't want to do that in temporal validation?
                "num_likelihood_samples": get_num_likelihood_samples("portfolio", 52, total_model_samples, algorithm),
                "num_posterior_samples": get_num_posterior_samples("portfolio", total_model_samples, algorithm),

                "num_observations": 52, # NOTE: now hard-coded
                "num_replications": get_num_time_windows_james(dataset_dir_james),  # NOTE: changed
                "num_test_observations": 13,    # TODO: dynamically measure the length of a test window here and in kl_portfolio_james

                "posterior": posterior,
                "do_temporal_validation": True,
                "n_splits": NUM_FOLDS,
                # TODO: note that the below probably won't be needed in runs with use_tv_epsilon as False
                "tv_ratio": tv_ratio,
                "risk_free_rates_filename": risk_free_rates_filename
            }
            tv_uuid_list = []
            for epsilon in (1e-05, 0.001, 1):   # NOTE: changed the epsilons
                for split_idx in range(NUM_FOLDS):
                    fold_params = base_params.copy()
                    fold_params["uuid"] = str(uuid4())
                    fold_params["epsilon"] = epsilon
                    fold_params["n_splits"] = NUM_FOLDS
                    fold_params["split_idx"] = split_idx
                    fold_params["use_tv_epsilon"] = False
                    fold_params["tv_uuid_list"] = []
                    experiment.append(fold_params)
                    tv_uuid_list.append(fold_params["uuid"])
            params = base_params.copy()
            params["uuid"] = str(uuid4())
            params["epsilon"] = None    # this must be calculated later using tv!
            params["split_idx"] = None  # not needed because we will calculate epsilon using all splits
            params["use_tv_epsilon"] = True     # we will exploit the tv epsilon
            params["tv_uuid_list"] = tv_uuid_list   #NOTE point to all the UUIDs across all splits and epsilons 
            experiment.append(params)
    return experiment

# TODO: refactor the below with the above etc.
def tv_kl_portfolio_james_has_tcosts_in_cost_function(dataset_dir_james, risk_free_rates_filename, dim: Optional[int], tv_ratio: Optional[str], NUM_FOLDS: int = 1) -> List[Dict]:
    assert NUM_FOLDS == 1
    if dim:
        assert dim <= get_min_dim_james(dataset_dir_james)
    experiment = []
    dgp = "james"   # NOTE: changed from DowJones
    for algorithm in ["kl_dro_bas", "kl_bdro", "kl_pp"]:
        likelihood = "multivariate_normal"
        posterior = "normal_inverse_wishart"
        inference = "bayes"
        # NOTE: in what I took from Patrick's branch, total_model_samples was fixed to 900 instead of the below. This doesn't make a difference in the case of kl_dro_bas: num_likelihood_samples is now 1 rather than 900 from it, but this variable isn't actually used due to closed-form, and num_posterior_samples ends up as 1 either way. In the case of kl_pp, it's just the case that 3600 is being left out, which may be good regarding consistency with BDRO. And for BDRO, it's only 900 below anyway.
        if algorithm == "kl_dro_bas":
            total_model_samples_list = [1]
        elif algorithm == "kl_pp":
            total_model_samples_list = [900, 3600]
        elif algorithm == "kl_bdro":
            total_model_samples_list = [900]
        NUM_SPLITS = 1
        for total_model_samples in total_model_samples_list:
            params = {
                "algorithm": algorithm,
                "contamination": 0.0,
                "dataset": "james",
                "dataset_dir_james": dataset_dir_james,
                "dgp": dgp,
                "dim": dim,
                "ignore_dpp": True,
                "inference": inference,
                "likelihood": likelihood,
                "njobs": 1,
                "num_likelihood_samples": get_num_likelihood_samples("portfolio", 52, total_model_samples, algorithm),
                "num_posterior_samples": get_num_posterior_samples("portfolio", total_model_samples, algorithm),
                "num_observations": 52,
                "num_replications": get_num_time_windows_james(dataset_dir_james),
                "num_test_observations": 13,
                "posterior": posterior,
                "do_temporal_validation": True,
                "n_splits": NUM_SPLITS,
                "tv_ratio": tv_ratio,
                "risk_free_rates_filename": risk_free_rates_filename,
                "has_tcosts_in_cost_function": True,
                "epsilon_list": [1e-05, 0.001, 1],
                "uuid": str(uuid4()),
            }
            experiment.append(params)
    return experiment

# def mmd_portfolio_james(dim: int) -> List[Dict]:
#     assert dim <= get_min_dim_james()
#     experiment = []
#     dgp = "james"
#     # James: TODO: check the below two, maybe they're specific to the old data?
#     num_likelihood_samples = 10
#     num_posterior_samples = 90
#     epsilon_set = []
#     for epsilon in ROBAS_DRO_EPSILON_SET:
#         if epsilon <= 0.2:  # James: TODO: why this condition? (Taken from mmd_portfolio)
#             epsilon_set.append(epsilon)
#     for (algorithm, likelihood), epsilon, in itertools.product(
#         [
#             ("dro_bas_mmd", "multivariate_normal"),
#             ("empirical_mmd", "empirical"),
#         ],
#         epsilon_set,
#     ):
#         if likelihood == "empirical":
#             inference = "empirical"
#             eta_set = [np.nan]
#         else:
#             inference = "npl_mmd"
#             eta_set = [0.1]
#         for eta in eta_set:
#             params = {
#                 "algorithm": algorithm,
#                 "contamination": 0.0,
#                 "dataset": "james",
#                 "dgp": dgp,
#                 "dim": dim,
#                 "epsilon": epsilon,
#                 "eta": eta,
#                 "inference": inference,
#                 "kernel_name": "k_comp",        
#                 "lengthscale": -1.0,
#                 "likelihood": likelihood,
#                 "normalise": False,
#                 "num_certify_points": NUM_CERTIFY,  # James: TODO: check if num_certify_points should change now that you've appropriated mmd_portfolio
#                 "num_likelihood_samples": num_likelihood_samples,
#                 "num_observations": IN_SAMPLE_TIME_WINDOW,
#                 "num_posterior_samples": num_posterior_samples,
#                 "num_replications": get_num_time_windows_james(),
#                 "num_test_observations": OUT_OF_SAMPLE_TIME_WINDOW,
#                 "posterior": "npl",
#                 "uuid": str(uuid4()),  # uniquely identify a run
#             }
#             experiment.append(params)
#     return experiment

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
