"""Entrypoint app functions"""

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Optional
from uuid import uuid4, UUID
from joblib import Parallel, delayed
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
import typer
from sklearn.model_selection import KFold
import pickle
from typing import Any, Callable

from bayesian_dro.Bayesian_DRO_continuous import main_Bayesian_DRO
from .bayes_conjugates import (
    sample_posterior,
    default_prior_params,
    get_log_partition_constant,
    get_posterior_params,
    derive_analytical_posterior_params,
    posterior_predictive_params,
    sample_posterior_predictive,
)
from .constants import (
    CONTAMINATION_LEVEL,
    IN_SAMPLE_TIME_WINDOW,
    NPL_ETA,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
    NUM_CERTIFY,
    MAX_PARAMS_OOM,
    ROBAS_NEWSVENDOR_NUM_REPLICATIONS,
)
from .dataset import sample_dgp, portfolio_dataset, portfolio_dataset_james, get_num_time_windows
from .epsilon import get_num_observations_in_train_split
from .experiments import ExperimentName, get_experiment
from .likelihood import sample_likelihood, reconstruct_covariance_from_triu
from .newsvendor import newsvendor_cost_cvxpy
from .npl import sample_npl
from .optimise import get_kl_bdro_problem, DRO_BAS_MMD
from .portfolio import get_kl_portfolio_problem, bdro_portfolio_posterior_samples, portfolio_objective_cvxpy, calculate_transaction_cost
from .preprocessing import normalise_by_dimension
from .gaussian_kernel import *
from .results import get_result_df_list, convert_str_to_float_list
from .metrics import calculate_sharpe_ratio, calculate_sortino_ratio

app = typer.Typer(name="misdro")


@app.command(name="setup-kl")
def setup_kl_dro_bas(
    experiment_name: ExperimentName, experiment_dir: Path, batch_size: int, dataset_dir_james, risk_free_rates_filename, dataset_dir: Path = Path("~/datasets/misdro/mmc2"), dim: Optional[int] = None, tv_ratio: Optional[str] = None, overwrite: bool = False
):
    """Setup an experiment in a new directory"""
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get the experiment from the name
    experiment = get_experiment(experiment_name, dataset_dir_james, risk_free_rates_filename, dim=dim, tv_ratio=tv_ratio)

    # write experiment file to JSON
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    if not experiment_name.is_temporal_validation():

        # for the given batch size, how many batches do we need?
        num_batches = math.ceil(float(len(experiment)) / float(batch_size))

        # setup SLURM file
        with open(
            Path(__file__).parent / "kl_dro_bas_template.slurm", "r", encoding="utf-8"
        ) as slurm_file:
            slurm_string = slurm_file.read()
        dgp_string = slurm_string.format(
            experiment_dir=experiment_dir, num_batches_minus_one=num_batches-1, batch_size=batch_size
        )
        (experiment_dir / f"{experiment_name}.slurm").write_text(dgp_string)

    else:

        # separate into two experiments which need separate SLURM files: 
        # A runs all the splits across all epsilons,
        # B uses the epsilons calculated by temporal-validation
        fold_experiment = [params for params in experiment if params["do_temporal_validation"] and not params["use_tv_epsilon"]]

        fold_num_batches = math.ceil(float(len(fold_experiment)) / float(batch_size))
        with open(
            Path(__file__).parent / "kl_dro_bas_template.slurm", "r", encoding="utf-8"
        ) as slurm_file:
            slurm_string = slurm_file.read()
        fold_string = slurm_string.format(
            experiment_dir=experiment_dir, num_batches_minus_one=fold_num_batches-1, batch_size=batch_size
        )
        fold_string += " --do-temporal-validation --no-use-tv-epsilon"
        (experiment_dir / f"do_temporal_validation.slurm").write_text(fold_string)

        use_tv_epsilon_experiment = [params for params in experiment if params["do_temporal_validation"] and params["use_tv_epsilon"]]

        use_tv_epsilon_num_batches = math.ceil(float(len(use_tv_epsilon_experiment)) / float(batch_size))
        with open(
            Path(__file__).parent / "kl_dro_bas_template.slurm", "r", encoding="utf-8"
        ) as slurm_file:
            slurm_string = slurm_file.read()
        use_tv_epsilon_string = slurm_string.format(
            experiment_dir=experiment_dir, num_batches_minus_one=use_tv_epsilon_num_batches-1, batch_size=batch_size
        )
        use_tv_epsilon_string += " --do-temporal-validation --use-tv-epsilon"
        (experiment_dir / f"use_tv_epsilon.slurm").write_text(use_tv_epsilon_string)


@app.command(name="setup-mmd")
def setup_mmd_dro_bas(
    experiment_name: ExperimentName, experiment_dir: Path, npl_samples_dir: Path, batch_size: int, dataset_dir: Path = Path("~/datasets/misdro/mmc2"), overwrite: bool = False, njobs: int = -1,
):
    """Setup an experiment in a new directory"""
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get the experiment from the name
    print("Creating experiment...")
    experiment = get_experiment(experiment_name, dataset_dir=dataset_dir)

    # write experiment file to JSON
    print("Writing experiment to JSON...")
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    # for the given batch size, how many batches do we need?
    num_batches = math.ceil(float(len(experiment)) / float(batch_size))

    # setup SLURM file
    print("Setting up SLURM files for optimization...")
    with open(
        Path(__file__).parent / "mmd_dro_bas_template.slurm", "r", encoding="utf-8"
    ) as slurm_file:
        slurm_string = slurm_file.read()
    slurm_string = slurm_string.format(
        experiment_dir=experiment_dir, njobs=njobs, num_batches_minus_one=num_batches-1, batch_size=batch_size
    )
    slurm_string += f" --npl-samples-dir {npl_samples_dir}"
    (experiment_dir / f"{experiment_name}.slurm").write_text(slurm_string)

    # create an ID for each unique posterior setting and save the IDs to a CSV
    if npl_samples_dir.exists():
        print("Using NPL samples from", npl_samples_dir)
    else:
        print("Setting up SLURM files ready for sampling the NPL on GPUs")
        npl_samples_dir.mkdir(parents=False)
        experiment_df = pd.DataFrame(experiment)
        gb = experiment_df.groupby(POSTERIOR_GB_COLS)
        posterior_settings = []
        for group, _ in gb:
            npl_row = dict(zip(POSTERIOR_GB_COLS, group))
            npl_row["npl_uuid"] = str(uuid4())
            posterior_settings.append(npl_row)
        posterior_settings_df = pd.DataFrame(posterior_settings)
        posterior_settings_df = posterior_settings_df.loc[posterior_settings_df["inference"].isin(["npl_mmd", "npl_wlb"])]
        posterior_settings_df.to_csv(npl_samples_dir / "npl_settings.csv", index=False)

        # then create SLURM file ready to sample the NPL on the cluster
        num_npl_batches = len(posterior_settings_df)
        with open(
            Path(__file__).parent / "sample_npl.slurm", "r", encoding="utf-8"
        ) as npl_slurm_file:
            npl_slurm_string = npl_slurm_file.read()
        npl_slurm_string = npl_slurm_string.format(num_npl_batches_minus_one=num_npl_batches-1, npl_samples_dir=npl_samples_dir, dataset_dir=dataset_dir)
        (npl_samples_dir / f"sample_npl_{experiment_name}.slurm").write_text(npl_slurm_string)

@app.command(name="csv")
def generate_csv(experiment_dir: Path, npl_samples_dir: Optional[Path] = None):
    """Write a CSV file with all the results"""
    experiment_filepath = experiment_dir / "experiment.json"
    with open(experiment_filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    experiment_df = pd.DataFrame(experiment).set_index("uuid")
    print("Loading and concatenating", len(experiment_df), "CSV files into a pandas dataframe...")
    result_df = pd.DataFrame()

    # TODO: replaced stuff with this, make sure no breaking changes and check if get_result_df_list source code must be changed
    result_list = get_result_df_list(experiment_dir, experiment_df.index)

    result_df = pd.concat([result_df] + result_list)
    result_df = result_df.join(experiment_df, on="uuid")
    result_df = result_df.reset_index()
    if npl_samples_dir:
        # load the settings for the NPL sampling
        settings_df = pd.read_csv(npl_samples_dir / "npl_settings.csv")
        # filter df because the empirical method doesn't produce anything and we get an error
        filtered_settings_df = settings_df[settings_df["inference"] == "npl_mmd"]
        # then, for each npl_uuid, load the times taken for each replication
        times_df = pd.concat([pd.read_csv(npl_samples_dir / npl_uuid / f"npl_times_{npl_uuid}.csv") for npl_uuid in filtered_settings_df["npl_uuid"]])
        # now merge the times and the npl_uuids together
        result_df = result_df.merge(settings_df, how="left", on=POSTERIOR_GB_COLS)
        result_df = result_df.merge(times_df, on=["npl_uuid", "replication"], how="left", suffixes=('', '_drop'))
        # finally, replace the incorrect posterior times with the correct ones
        result_df.loc[~(result_df['posterior_time_drop'].isna()), 'posterior_time'] = result_df['posterior_time_drop']
        result_df = result_df.drop("posterior_time_drop", axis=1)
    # save to a CSV file
    print(result_df)
    result_df.to_csv(experiment_dir / "results.csv", index=False)


@app.command(name="experiment")
def run_experiment(
    experiment_dir: Path,
    dgp: str,
    algorithm: str,
    only_missing: bool = False, #NOTE if the experiment runs out of memory set this to true to true 
    njobs: int = -1,
    dataset_dir: Path = Path("~/datasets/misdro/mmc2"),
):
    """When using SLURM, this function is called to run an experiment"""
    print(datetime.now(), "- Running algorithm", algorithm, "with DGP", dgp, "from experiment directory", experiment_dir)
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    # NOTE if only-missing flag, then only run if the results CSV file doesn't exist
    for params in experiment:
        if params["dgp"] == dgp and params["algorithm"] == algorithm and not ((experiment_dir / params["uuid"]).exists() and only_missing):
            run(experiment_dir, **params, njobs=njobs)


@app.command(name="batch")
def batch(experiment_dir: Path, batch_id: int, batch_size: int, only_missing: bool = False, dataset_dir: Path = Path("~/datasets/misdro/mmc2"), npl_samples_dir: Optional[Path] = None, do_temporal_validation: bool = False, use_tv_epsilon: bool = False):
    print(datetime.now(), "Running batch from array index", batch_id)
    print()
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    if do_temporal_validation and not use_tv_epsilon:
        # only keep parameters where do_temporal_validation is set to True
        experiment = [params for params in experiment if params["do_temporal_validation"] and not params["use_tv_epsilon"]]
        print(len(experiment), "params to run in this temporal-validation batch.")
    elif do_temporal_validation and use_tv_epsilon:
        experiment = [params for params in experiment if params["do_temporal_validation"] and params["use_tv_epsilon"]]
        print(len(experiment), "params to run in this 'use_tv_epsilon' batch.")

    start = batch_id * batch_size
    batch_experiment = experiment[start: min(start + batch_size, len(experiment))]
    for params in batch_experiment:
        if not ((experiment_dir / params["uuid"]).exists() and only_missing):
            run(experiment_dir, dataset_dir=dataset_dir, npl_samples_dir=npl_samples_dir, **params)

@app.command(name="uuid")
def run_uuid(experiment_dir: Path, uuid: UUID, dataset_dir: Path = Path("~/datasets/misdro/mmc2"), npl_samples_dir: Optional[Path] = None, njobs: int = -1, verbose: bool = False) -> None:
    """Run DRO for only one specified uuid parameters"""
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    found = False
    for params in experiment:
        if params["uuid"] == str(uuid):
            found = True
            run(experiment_dir, verbose=verbose, dataset_dir=dataset_dir, npl_samples_dir=npl_samples_dir, **params)
    if not found:
        raise ValueError(f"UUID {uuid} not found in {filepath}")

def get_useful_tv_results_shv_all_windows(result_df: pd.DataFrame) -> list[dict]:
    all_tv_results = []
    for _, g in result_df.groupby(level="replication", sort=True):
        tv_results_for_replication = {}
        for _, row in g.iterrows():
            tv_results_for_replication[row["epsilon"]] = {col: row[col] for col in (
                "likelihood_time", "posterior_time", "solve_time", "solution", "out_of_sample_cost"
            )}
        all_tv_results.append(tv_results_for_replication)
    return all_tv_results

def get_stock_figi_list(training_df: pd.DataFrame) -> list[str]:

    # TODO: this whole thing with the use of get_window_of_train_and_test_dataframes is a bit of a fudge--the stock IDs should be saved along with the other results. Moreover, I don't know why I appended _{window index} to each of the FIGIs in the first place, so undo that and get rid of the splitting done below.

    return [col.split("_")[0] for col in training_df.columns]

def choose_risk_free_rates_for_validation(risk_free_rates: list[float], number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation: int, num_test_weeks: int, num_validation_dates_per_window: int, window_index: int) -> list[float]:

    t = number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation
    v = num_validation_dates_per_window

    return risk_free_rates[t: t + num_test_weeks * window_index] + risk_free_rates[t + num_test_weeks * window_index - v: t + num_test_weeks * window_index]

def choose_risk_free_rates_for_testing(risk_free_rates: list[float], number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation: int, num_test_weeks: int, window_index: int) -> list[float]:

    t = number_of_extra_data_at_start_of_risk_free_rates_reserved_for_validation

    return risk_free_rates[t: t + num_test_weeks * (window_index + 1)]

def process_tv_results_shv_for_window(stock_figi_list_this_window: list[str], window_index: int, risk_free_rates: list[float], num_test_observations: int, num_validation_observations_per_window: int, useful_tv_results_shv_all_windows: list[dict[str, Any]], prev_stock_figi_list: list[str], prev_portfolio_weighting: list[float], all_out_of_sample_costs_so_far: list[float], ratio_calculator: Callable[[list[float], list[float], int], float], period_for_test_ratio_in_weeks: int) -> dict[str, Any]:
    # TODO: note that 51 is specific to the risk free returns file in use (because it has 51 weekly risk-free rates up to and including the first rebalance date)
    risk_free_rates_for_validation = choose_risk_free_rates_for_validation(risk_free_rates, 51, num_test_observations, num_validation_observations_per_window, window_index)   
    best_epsilon_this_window, highest_validation_ratio = None, -float("inf")
    total_validation_times = {"posterior": 0, "likelihood": 0, "solve": 0}  # TODO: Maybe save validation times for each epsilon rather than an overall one?
    for epsilon, validation_results in useful_tv_results_shv_all_windows[window_index].items():
        validation_transaction_cost = calculate_transaction_cost(prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list_this_window, validation_results["solution"])
        validation_portfolio_returns = validation_results["out_of_sample_cost"]
        validation_portfolio_returns[0] -= validation_transaction_cost
        all_out_of_sample_costs_so_far_including_validation = all_out_of_sample_costs_so_far + validation_portfolio_returns
        validation_ratio = ratio_calculator(all_out_of_sample_costs_so_far_including_validation, risk_free_rates_for_validation, period_for_test_ratio_in_weeks - 13 + num_validation_observations_per_window)
        if validation_ratio > highest_validation_ratio:
            best_epsilon_this_window, highest_validation_ratio = epsilon, validation_ratio
        for time_type in total_validation_times:
            total_validation_times[time_type] += validation_results[f"{time_type}_time"]
    return {
        "best_epsilon_this_window": best_epsilon_this_window,
        "total_validation_times": total_validation_times,
    }

def process_actual_results_after_tv_shv_for_window(total_validation_times: dict[str, float], results_this_replication: dict[str, Any], prev_stock_figi_list: list[str], prev_portfolio_weighting: list[float], stock_figi_list_this_window: list[str], risk_free_rates: list[float], num_test_observations: int, window_index: int, ratio_calculator: Callable[[list[float], list[float], int], float], all_out_of_sample_costs_so_far: list[float], period_for_test_ratio_in_weeks: int, tv_ratio: str) -> tuple[dict[str, Any], list[float]]:
    for time_type, time in total_validation_times.items():
        results_this_replication[f"total_validation_{time_type}_time"] = time
    actual_transaction_cost = calculate_transaction_cost(prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list_this_window, results_this_replication["solution"])
    actual_portfolio_returns = results_this_replication["out_of_sample_cost"]
    actual_portfolio_returns[0] -= actual_transaction_cost
    results_this_replication["out_of_sample_cost"] = actual_portfolio_returns   # TODO: note somewhere that these will already have transaction costs in case you load them in .ipynb and forget and reapply them
    all_out_of_sample_costs_so_far += results_this_replication["out_of_sample_cost"]
    # TODO: note that 51 is specific to the risk free returns file in use (because it has 51 weekly risk-free rates up to and including the first rebalance date)
    risk_free_rates_for_testing = choose_risk_free_rates_for_testing(risk_free_rates, 51, num_test_observations, window_index)
    actual_ratio = ratio_calculator(all_out_of_sample_costs_so_far, risk_free_rates_for_testing, period_for_test_ratio_in_weeks)
    results_this_replication[tv_ratio] = actual_ratio
    return results_this_replication, all_out_of_sample_costs_so_far

@app.command(name="run")
def run(
    experiment_dir: Path,
    dataset_dir_james,
    risk_free_rates_filename,
    algorithm: str = "kl_bdro",
    contamination: float = CONTAMINATION_LEVEL,
    dataset: str = "newsvendor",
    dataset_dir: Optional[Path] = None,
    dgp: str = "truncated_normal",
    dim: Optional[int] = 1,
    epsilon: Optional[float] = 1.0,
    eta: float = NPL_ETA,
    ignore_dpp: bool = False,
    inference: str = "bayes",
    kernel_name: str = "k_jax",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
    njobs: int = -1,
    normalise: bool = False,
    npl_samples_dir: Optional[Path] = None,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_replications: int = NUM_REPLICATIONS,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    num_certify_points: int = NUM_CERTIFY,
    posterior: str = "gamma",
    do_temporal_validation: bool = False,
    tv_ratio: Optional[str] = None,
    n_splits: Optional[int] = None,
    split_idx: Optional[int] = None,
    use_tv_epsilon: bool = False,
    tv_uuid_list: list[str] = [],
    uuid: str = str(uuid4()),
    verbose: bool = False,
):
    """Run Newsvendor Misspecified Bayesian DRO"""
    if uuid:
        print(uuid)

    if do_temporal_validation and not use_tv_epsilon and n_splits == 1:

        # NOTE: making sure that, when splitting the training data into training and validation for TV, the ratio is the same as for training and testing outside of TV
        num_training_observations = round(num_observations / (num_observations + num_test_observations) * num_observations)
        num_test_observations = num_observations - num_training_observations
        print(f"Doing {n_splits}-fold temporal-validation on split {split_idx}: training/test set size is {num_training_observations}/{num_test_observations}.")
    
    # elif do_cross_validation and n_splits is not None and split_idx is not None and epsilon is not None:

    #     num_training_observations = get_num_observations_in_train_split(n_splits, split_idx, num_observations)
    #     num_test_observations = num_observations - num_training_observations
    #     print(f"Doing {n_splits}-fold cross-validation on split {split_idx}: training/test set size is {num_training_observations}/{num_test_observations}.")

    elif use_tv_epsilon and split_idx is None and do_temporal_validation:
        num_training_observations = num_observations
        # load the results df for each UUID in tv_uuid_list
        result_list = get_result_df_list(experiment_dir, tv_uuid_list)  # NOTE by James: in this Python list, each element is for one UUID (I.E. one epsilon across all replications for the algorithm in question), and the element is the Pandas DataFrame containing, essentially, what is in the UUID's .csv file
        result_df = pd.concat(result_list)  # NOTE by James: combining all the above Pandas DataFrames into a single one, but UUID and replication columns allow easy distinction

        # TODO: this convert_str_to_float_list is from results.py and looks as the one in plot.py originally did; I had to change the latter, so maybe I will need to change the former?
        result_df["out_of_sample_cost"] = result_df["out_of_sample_cost"].map(lambda x: convert_str_to_float_list(x, num_test_observations))    # NOTE by James: just formatting

        # NOTE by James: for each row in result_df (each of which has a distinct UUID/replication combination), this goes to the object in experiment.json with the mathcing UUID and adds all of the "other" parameters in this object to the row
        experiment_filepath = experiment_dir / "experiment.json"
        with open(experiment_filepath, "r", encoding="utf-8") as json_file:
            experiment = json.load(json_file)
        experiment_df = pd.DataFrame(experiment).set_index("uuid")
        result_df = result_df.join(experiment_df, on="uuid")

        assert len(result_df["algorithm"].unique()) == 1

        if n_splits == 1:

            if tv_ratio:
                # [0: {0.00001: {"solution": ..., other result types}, other epsilons}, other replication numbers in order]
                useful_tv_results_shv_all_windows = get_useful_tv_results_shv_all_windows(result_df)
            else:
                # TODO: is this correct? It 
                best_epsilons = (
                    result_df.reset_index()
                            .assign(mean_cost=result_df["out_of_sample_cost"].apply(np.mean))
                            .groupby("replication")
                            .apply(lambda g: g.loc[g["mean_cost"].idxmax(), "epsilon"])
                )

        else:

            # TODO get the best epsilon for each replication from the temporal-validation - store in array

            # NOTE: there was unused logic here featuring epsilons_for_replications

            # NOTE by James: the below results in a row for each epsilon, where its corresponding out-of-sample mean and variance (across all of its replications) is shown
            gb = result_df.groupby(["epsilon"])
            agg_df = gb.agg(
                out_of_sample_mean = pd.NamedAgg(column="out_of_sample_cost", aggfunc=lambda x: np.mean(np.concatenate(x.values))),
                out_of_sample_var = pd.NamedAgg(column="out_of_sample_cost", aggfunc=lambda x: np.var(np.concatenate(x.values), ddof=1)),        
            )

            if dataset == "portfolio" or dataset == "james":
                epsilon = agg_df.loc[agg_df["out_of_sample_mean"] == agg_df["out_of_sample_mean"].max()].index[0]
            print("Epsilon:", epsilon)

            # TODO: there was unused logic here about the Pareto front, see if you should include it

    elif do_temporal_validation:
        raise ValueError("Something went wrong in the previous logic.")
    else:
        num_training_observations = num_observations

    print("DGP:", dgp, " - ALGORITHM:", algorithm, " - NUM LIKELIHOOD SAMPLES:", num_likelihood_samples, " - POSTERIOR:", posterior, "- DATASET:", dataset, "- DIM:", dim)
    if dataset == "james" and (algorithm in ("kl_pp", "kl_empirical") or algorithm in ("kl_bdro", "kl_dro_bas") and likelihood == "multivariate_normal"):
        problem = None
    elif algorithm in ("kl_bdro", "kl_dro_bas", "kl_pp", "kl_empirical") and dataset == "newsvendor":
        problem = get_kl_bdro_problem(
            newsvendor_cost_cvxpy, num_posterior_samples, num_likelihood_samples, dim=dim,
        )
    elif algorithm == "kl_pp" and dataset in ("portfolio", "portfolio_synthetic"):
        problem = get_kl_bdro_problem(portfolio_objective_cvxpy, num_posterior_samples, num_likelihood_samples, dim=dim, is_portfolio=True)
    elif algorithm == "kl_empirical" and dataset in ("portfolio", "portfolio_synthetic"):
        problem = get_kl_bdro_problem(portfolio_objective_cvxpy, 1, num_training_observations, dim=dim, is_portfolio=True)
    elif algorithm in ("kl_bdro", "kl_dro_bas") and dataset in ("portfolio", "portfolio_synthetic") and likelihood == "multivariate_normal":
        problem = get_kl_portfolio_problem(dim, num_posterior_samples)
    elif algorithm in ("dro_bas_mmd", "empirical_mmd"):
        dim_theta = dim
        if algorithm == "dro_bas_mmd":
            n_samples = num_posterior_samples*num_likelihood_samples
        elif algorithm == "empirical_mmd":
            n_samples = num_training_observations
        if dataset == "newsvendor":
            kdro_class = DRO_BAS_MMD(dim_theta, dim, newsvendor_cost_cvxpy)
            problem = kdro_class.get_newsvendor_problem(n_samples, num_certify_points)
        elif dataset in ("portfolio", "portfolio_synthetic", "james"):
            kdro_class = DRO_BAS_MMD(dim_theta, dim, portfolio_objective_cvxpy) # TODO: James: are there cases in which dim_theta and dim should be different?
            problem = kdro_class.get_portfolio_problem(n_samples, num_certify_points)
        else:
            raise ValueError(f"Objective not implemented for dataset '{dataset}'")
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented.")
    # If the number of parameters is small enough, then use Disciplined Parametrized Programming (DPP)
    # to reduce the compilation time in each replication.
    # However, a large number of parameters uses an enormous amout of RAM in the current cvxpy implementation.
    if not ignore_dpp and algorithm in ("kl_bdro", "kl_dro_bas", "dro_bas_mmd", "empirical_mmd"):
        n_parameters = np.sum(np.prod(param.shape) for param in problem.parameters())
        # NOTE whilst DRO-BAS can handle at least 5000 params, BDRO cannot.
        # So, for a fair comparison, we turn off DPP for both DRO-BAS and BDRO.
        if n_parameters >= cp.settings.PARAM_THRESHOLD:
        # if n_parameters >= 1000:
            ignore_dpp = True
            njobs = 1

    params = {
        "algorithm": algorithm,
        "contamination": contamination,
        "dataset": dataset,
        "dataset_dir": dataset_dir,
        "dataset_dir_james": dataset_dir_james,
        "dgp": dgp,
        "dim": dim,
        "epsilon": epsilon,
        "eta": eta,
        "ignore_dpp": ignore_dpp,
        "inference": inference,
        "kernel_name": kernel_name, # James
        "lengthscale": lengthscale,
        "likelihood": likelihood,
        "normalise": normalise,
        "num_certify_points": num_certify_points,
        "num_likelihood_samples": num_likelihood_samples,
        "num_observations": num_observations,
        "num_posterior_samples": num_posterior_samples,
        "num_replications": num_replications,
        "num_test_observations": num_test_observations,
        "posterior": posterior,
        "do_temporal_validation": do_temporal_validation,
        "n_splits": n_splits,
        "split_idx": split_idx,
        "use_tv_epsilon": use_tv_epsilon,
        "uuid": uuid,
        "verbose": verbose,
    }

    if inference in ("npl_wlb", "npl_mmd"):
        posterior_df = pd.read_csv(npl_samples_dir / "npl_settings.csv").set_index(POSTERIOR_GB_COLS)
        npl_params = {key: params[key] for key in POSTERIOR_GB_COLS}
        npl_uuid = get_npl_uuid(posterior_df, npl_params)
        params["npl_uuid_dir"] = npl_samples_dir / npl_uuid
    else:
        params["npl_uuid_dir"] = None
    params.pop("num_replications")  # popped because we don't need to pass this to the run_replication method, but it is needed above for getting the npl_uuid
    params.pop("kernel_name")   # NOTE: James: popped because currently it's used to get npl_uuid but not used in run_replication. TODO: see if that should change
    # TODO: consider popping lengthscale and eta, if they are only used in NPL sampling and this isn't done in run

    # TODO: remember add portfolio vs newsvendor to this boolean and similar ones
    if do_temporal_validation and use_tv_epsilon and tv_ratio:
        with open(dataset_dir_james, "rb") as f:
            stock_figi_lists = [get_stock_figi_list(training_df) for training_df, _ in pickle.load(f)]
        with open(risk_free_rates_filename, "rb") as f:
            risk_free_rates = pickle.load(f)
        all_out_of_sample_costs_so_far = []
        # TODO: maybe enforce for the same index in the below lists to refer to the same stock
        prev_stock_figi_list = []
        prev_portfolio_weighting = []
        ratio_calculator = {"sharpe": calculate_sharpe_ratio, "sortino": calculate_sortino_ratio}[tv_ratio]
        period_for_test_ratio_in_weeks = 156  # TODO: maybe allow this to be chosen dynamically
        num_validation_observations_per_window = len(list(useful_tv_results_shv_all_windows[0].values())[0]["out_of_sample_cost"])

    if njobs == 1:
        all_solve_start = datetime.now()
        list_of_replication_stats = []
        print(all_solve_start, "- Running all replications in series.")
        for j in range(num_replications):
            if use_tv_epsilon and n_splits == 1:
                if tv_ratio:
                    stock_figi_list_this_window = stock_figi_lists[j]
                    processed_validation_results = process_tv_results_shv_for_window(stock_figi_list_this_window, j, risk_free_rates, num_test_observations, num_validation_observations_per_window, useful_tv_results_shv_all_windows, prev_stock_figi_list, prev_portfolio_weighting, all_out_of_sample_costs_so_far, ratio_calculator, period_for_test_ratio_in_weeks)
                    best_epsilon_this_window = processed_validation_results["best_epsilon_this_window"]
                    total_validation_times = processed_validation_results["total_validation_times"]
                else:
                    best_epsilon_this_window = best_epsilons.loc[j]
                params["epsilon"] = best_epsilon_this_window
            results_this_replication = run_replication(j, problem, **params)
            if use_tv_epsilon and n_splits == 1 and tv_ratio:
                results_this_replication, all_out_of_sample_costs_so_far = process_actual_results_after_tv_shv_for_window(total_validation_times, results_this_replication, prev_stock_figi_list, prev_portfolio_weighting, stock_figi_list_this_window, risk_free_rates, num_test_observations, j, ratio_calculator, all_out_of_sample_costs_so_far, period_for_test_ratio_in_weeks, tv_ratio)
                prev_stock_figi_list = stock_figi_list_this_window
                prev_portfolio_weighting = results_this_replication["solution"]
            list_of_replication_stats.append(results_this_replication)
        all_solve_end = datetime.now()
        print(all_solve_end, "- Finished solving all replications in series. Total solve time is", (all_solve_end - all_solve_start).total_seconds())

    else:
        # NOTE we want to do the first replication solve *not* in parallel
        # because it takes a lot of memory to compile a large DPP problem.
        # Subsequent solves don't compile the problem and are solved in parallel
        print("Running replication 0: cvxpy Problem is compiling.")
        first_solve_start = datetime.now()
        replication_stats_0 = run_replication(0, problem, **params)
        print("Compilation + first solve took", (datetime.now() - first_solve_start).total_seconds(), "seconds.")
        all_solve_start = datetime.now()
        print(f"Solving remaining {num_replications - 1} replications in parallel.")
        list_of_replication_stats = Parallel(n_jobs=njobs)(
            delayed(run_replication)(j, problem, **params) for j in range(1, num_replications)
        )
        print("Solving all other replications took", (datetime.now() - all_solve_start).total_seconds(), "seconds.")
        list_of_replication_stats.insert(0, replication_stats_0)
        print(f"Finished running {algorithm} with posterior {posterior} and DGP {dgp}.")

    df = pd.DataFrame(list_of_replication_stats)
    csv_filepath = experiment_dir / f"{uuid}.csv"
    print(f"Writing CSV to {csv_filepath}")
    print()
    df.to_csv(csv_filepath, index=False)

# TODO: consider removing eta and lengthscale from the parameters here, if they are only used in NPL sampling and that's not done in run_replication
def run_replication(
    replication: int,
    problem: Optional[cp.Problem],
    dataset_dir_james,
    algorithm: str = "kl_bdro",
    contamination: float = CONTAMINATION_LEVEL,
    dataset: str = "newsvendor",
    dataset_dir: Optional[Path] = None,
    dgp: str = "truncated_normal",
    dim: Optional[int] = 1,
    epsilon: Optional[float] = 1.0,
    eta: float = NPL_ETA,
    ignore_dpp: bool = False,
    inference: str = "bayes",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
    normalise: bool = False,
    npl_uuid_dir: Optional[Path] = None,
    num_certify_points: int = NUM_CERTIFY,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    posterior: str = "gamma",
    do_temporal_validation: bool = False,
    n_splits: Optional[int] = None,
    split_idx: Optional[int] = None,
    use_tv_epsilon: bool = False,
    uuid: str = str(uuid4()),
    verbose: bool = False,
):

    """Run a single replication where the seed is given by the replication number"""
    # 1. generate dataset
    # Also, get dim in case of James dataset
    dgp_start = datetime.now()
    generator = np.random.default_rng(seed=replication)
    if dataset == "newsvendor" or dataset == "portfolio_synthetic":
        # NOTE if contamination is specified, then only contaminate the training samples (not test samples)
        data = sample_dgp(
            dgp, num_observations, dim=dim, contamination=contamination, generator=generator
        )
        data_eval = sample_dgp(
            dgp, num_test_observations, dim=dim, contamination=0.0, generator=generator
        )
    elif dataset == "james":
        data, data_eval = portfolio_dataset_james(time_window_id=replication, dataset_dir_james=dataset_dir_james)
        dim = data.shape[1]
        if normalise:
            data = normalise_by_dimension(data)
    elif dataset == "portfolio":
        # NOTE shape of data (N, D) where N is number of weeks and D is the number of stocks
        if dgp == "DowJones-crash":
            CRASH_WINDOW_ID = 75    # NOTE use 72 for long term evaluation of crash
            CRASH_OOS_TIME_WINDOW = IN_SAMPLE_TIME_WINDOW  # NOTE use 4 years for long term
            data, data_eval = portfolio_dataset(dgp, CRASH_WINDOW_ID, dataset_dir, out_of_sample_time_window=CRASH_OOS_TIME_WINDOW)
        else:
            data, data_eval = portfolio_dataset(dgp, replication, dataset_dir)
        # NOTE only normalise training data
        if normalise:
            data = normalise_by_dimension(data)
    else:
        raise NotImplementedError(f"Dataset not implemented: {dataset}")
    
    if do_temporal_validation and not use_tv_epsilon:

        if n_splits == 1:

            # TODO: this feels a little bit fudgey re: calculating num_training_observations in run and then again here

            num_training_observations = num_observations - num_test_observations
            train_index = np.arange(num_training_observations)
            test_index = np.arange(num_training_observations, num_training_observations + num_test_observations)
            num_observations = num_training_observations    # NOTE: did this so that, like in Patrick's branch, only num_training_observations is passed to get_kl_bdro_problem when getting the problem for kl_empirical below
            # TODO: the above, however, will also change num_observations for some empirical mmd stuff below, but since that will now be using data which is tied to num_training_observations, I think that should be fine, but check before ever merging these changes with main
    
        # else:

        #     # NOTE we use a different random number generator for CV because we do not want to contaminate the test samples
        #     # and because we want to reproduce the same CV splits for each replication
        #     cv_random_state = np.random.RandomState(seed=replication + 1000)
        #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
        #     train_index, test_index = list(kf.split(data))[split_idx]

        data = data[train_index]
        data_eval = data[test_index]

    dgp_time = (datetime.now() - dgp_start).total_seconds()

    # 2. sample from the posterior
    posterior_start = datetime.now()
    log_partition_constant = 0.0
    if inference == "bayes":
        theta_prior = default_prior_params(posterior, dim=dim)
        theta_posterior = get_posterior_params(posterior, data, theta_prior)
        if algorithm == "kl_dro_bas":
            assert num_posterior_samples == 1
            log_partition_constant = get_log_partition_constant(posterior, theta_posterior)
            theta_sample = derive_analytical_posterior_params(
                posterior, theta_posterior
            )
        elif algorithm == "kl_bdro" and dataset in ("portfolio", "portfolio_synthetic", "james") and posterior == "normal_inverse_wishart":
            mu_post, _, iota_post, Psi_post = theta_posterior
            theta_sample = bdro_portfolio_posterior_samples(num_posterior_samples, mu_post, iota_post, Psi_post, generator=generator)
        elif algorithm == "kl_pp":
            theta_sample = posterior_predictive_params(posterior, theta_posterior)
        else:
            theta_sample = sample_posterior(posterior, theta_posterior, num_posterior_samples, generator=generator)
    elif inference in ("npl_wlb", "npl_mmd"):
        # get the posterior 
        path_to_csv = npl_uuid_dir / f"npl_sample_{replication}.csv"
        theta_sample = pd.read_csv(path_to_csv, index_col=False, header=None).values
        assert num_posterior_samples == theta_sample.shape[0]
        # assert dim == theta_sample.shape[1], f"Dimension dim={dim} not equal to {theta_sample.shape[1]}"
    elif inference == "empirical":
        # empirical does not have a posterior
        theta_sample = np.nan * np.ones(num_posterior_samples)
    else:
        raise ValueError(f"Inference procedure '{inference}' is not supported.")
    assert log_partition_constant >= 0
    # assert log_partition_constant < epsilon
    posterior_time = (datetime.now() - posterior_start).total_seconds()

    # 3. sample from the likelihood
    likelihood_start = datetime.now()
    if inference == "empirical":
        xi = data
    elif inference == "bayes" and algorithm == "kl_pp":
        xi = sample_posterior_predictive(likelihood, posterior, theta_sample, dim, num_likelihood_samples, generator=generator).reshape((1, num_likelihood_samples, dim))
    elif inference == "bayes" and dataset in ("portfolio", "portfolio_synthetic", "james") and likelihood == "multivariate_normal":
        pass    # no need to sample from likelihood cause we have closed form
    else:
        xi = sample_likelihood(
            likelihood,
            theta_sample,
            dim,
            num_likelihood_samples,
            num_posterior_samples,
            generator=generator,
            inference=inference,
        )
    likelihood_time = (datetime.now() - likelihood_start).total_seconds()

    # 4. Instantiate problem object if doing so in each run
    # NOTE: not changing num_posterior_samples or num_likelihood_samples in the case of validation run because that would fundamentally change the optimiser and compromise the integrity of validation itself
    if dataset == "james":
        if algorithm == "kl_pp":
            problem = get_kl_bdro_problem(portfolio_objective_cvxpy, num_posterior_samples, num_likelihood_samples, dim=dim, is_portfolio=True)
        elif algorithm == "kl_empirical":
            problem = get_kl_bdro_problem(portfolio_objective_cvxpy, 1, num_observations, dim=dim, is_portfolio=True)
        elif algorithm in ("kl_bdro", "kl_dro_bas") and likelihood == "multivariate_normal":
            problem = get_kl_portfolio_problem(dim, num_posterior_samples)

    if algorithm == "kl_dro_bas" and (
        dataset == "portfolio" or dataset == "portfolio_synthetic" or dataset == "james"
        or (dataset == "newsvendor" and do_temporal_validation)
    ):
        # NOTE under the above conditions, having values of epsilon just above
        # the constant is benefitial for obtaining a small mean
        epsilon_prime = epsilon
    else:
        # as in Corollary 3.7
        # NOTE for BDRO and BAS-PP this is just equal to epsilon because log_partition_constant is zero
        epsilon_prime = epsilon - log_partition_constant

    # 5. run the chosen DRO algorithm
    solve_start = datetime.now()
    solution = np.nan
    if (
            dataset in ("portfolio" "portfolio_synthetic", "james")
            and algorithm in ("kl_bdro", "kl_dro_bas")
            and likelihood == "multivariate_normal"
    ):
        # if epsilon - log_partition_constant < 0:
        #     # NOTE the optimisation problem is unbounded below
        #     solution = np.inf * np.ones(dim)
        #     solve_time = 0.0
        #     setup_time = 0.0
        # else:
        problem.param_dict["epsilon_minus_constant"].value = np.array([epsilon_prime])
        problem.param_dict["mu_post"].value = theta_sample[0, :dim]
        for i in range(num_posterior_samples):
            # get a PSD covariance from the upper triangular vector
            cov = reconstruct_covariance_from_triu(theta_sample[i, dim:], dim)
            # then take the square root of the covariance and set to parameter value
            problem.param_dict[f"sqrt_cov_post_{i}"].value = sp.linalg.sqrtm(cov)

        # NOTE the MOSEK 'accept_unknown' argument is needed due to https://github.com/cvxpy/cvxpy/pull/2117
        problem.solve(solver=cp.MOSEK, verbose=verbose, ignore_dpp=ignore_dpp, accept_unknown=True)
        solution = problem.var_dict["x"].value
        setup_time = problem.solver_stats.setup_time
    elif algorithm in ("kl_bdro", "kl_dro_bas", "kl_pp", "kl_empirical"):
        if epsilon - log_partition_constant < 0:
            # NOTE the optimisation problem is unbounded below
            solution = np.inf * np.ones(dim)
            solve_time = 0.0
            setup_time = 0.0
        else:
            # set parameters then solve
            problem.param_dict["epsilon_minus_constant"].value = np.array([epsilon_prime])
            xi = xi.reshape((num_posterior_samples, num_likelihood_samples, dim))
            for i in range(num_posterior_samples):
                problem.param_dict[f"xi_{i}"].value = xi[i]
            # NOTE the MOSEK 'accept_unknown' argument is needed due to https://github.com/cvxpy/cvxpy/pull/2117
            problem.solve(solver=cp.MOSEK, verbose=verbose, ignore_dpp=ignore_dpp, accept_unknown=True)
            solution = problem.var_dict["x"].value
            # solve_time = problem.solver_stats.solve_time
            setup_time = problem.solver_stats.setup_time
    elif algorithm in ("dro_bas_mmd", "empirical_mmd"):
        if algorithm == "dro_bas_mmd":
            xi = xi.reshape((num_likelihood_samples*num_posterior_samples,dim))
        elif algorithm == "empirical_mmd":
            xi = data.reshape((num_observations,dim))
        Xcert = np.random.uniform(np.min(xi), np.max(xi), size=[num_certify_points,dim])
        zetai = np.concatenate([xi, Xcert], axis=0)
        l = np.sqrt((1/2)*np.median(distance.cdist(zetai, zetai, 'sqeuclidean')))
        # K = k_jax(zetai, zetai, l)
        K = k_comp(zetai, zetai)
        K_decomp = mat_decomp_jax(K)
        problem.param_dict["Xobs"].value = xi
        problem.param_dict["Xcert"].value = Xcert
        problem.param_dict["K"].value = np.asarray(K)
        problem.param_dict["K_decomposed"].value = np.asarray(K_decomp)
        problem.param_dict["epsilon"].value = np.array([epsilon])
        # NOTE the MOSEK 'accept_unknown' argument is needed due to https://github.com/cvxpy/cvxpy/pull/2117
        problem.solve(cp.MOSEK, verbose=False, ignore_dpp=ignore_dpp, accept_unknown=True)
        solution = problem.var_dict["theta"].value
        # solve_time = problem.solver_stats.solve_time
        setup_time = problem.solver_stats.setup_time
    elif algorithm == "bdro_grid_search":
        solution = main_Bayesian_DRO(xi, epsilon)
        setup_time = 0.0  # can't really measure this easily
    else:
        raise ValueError("Please choose a valid algorithm")
    solve_time = (datetime.now() - solve_start).total_seconds()
    # evaluate the out-of-sample cost
    if (solution == np.inf).any():
        out_of_sample_cost = np.inf * np.ones(num_test_observations)
        # TODO: James: should the solution still be made into a Python list like it is in the else statement?
    else:
        if dataset == "newsvendor":
            out_of_sample_cost = newsvendor_cost_cvxpy(solution, data_eval.reshape((num_test_observations, dim))).value
        elif dataset in ("portfolio", "portfolio_synthetic", "james"):
            out_of_sample_cost = data_eval @ solution
        else:
            raise NotImplementedError(f"Out-of-sample cost for dataset '{dataset}' not implemented")
        solution = list(solution)

    results = {
        "uuid": uuid,
        "replication": replication,
        "solution": solution,
        "dgp_time": dgp_time,
        "likelihood_time": likelihood_time,
        "posterior_time": posterior_time,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "log_partition_constant": log_partition_constant,
        "out_of_sample_cost": list(out_of_sample_cost),
    }

    return results

POSTERIOR_GB_COLS = [
    "contamination",
    "dataset",
    "dgp",
    "dim",
    "eta",
    "inference",
    "kernel_name",
    "lengthscale",
    "likelihood",
    "normalise",
    "num_observations",
    "num_posterior_samples",
    "num_replications",
    "posterior"
]

def get_npl_uuid(posterior_settings_df: pd.DataFrame, params: dict) -> str:
    params_tuple = tuple([params[key] for key in POSTERIOR_GB_COLS])
    param_sr = posterior_settings_df.loc[params_tuple]
    return param_sr["npl_uuid"]

@app.command("npl")
def sample_npl_for_experiment(
    npl_samples_dir: Path,
    batch: int,
    dataset_dir: Optional[Path] = None,
):
    """Run a single replication where the seed is given by the replication number"""

    npl_samples_dir.mkdir(parents=False, exist_ok=True)
    posterior_times = []

    npl_df = pd.read_csv(npl_samples_dir / "npl_settings.csv")
    npl_row = npl_df.iloc[batch]
    npl_dir = npl_samples_dir / npl_row["npl_uuid"]
    npl_dir.mkdir()
    dataset = npl_row["dataset"]
    dgp = npl_row["dgp"]
    npl_uuid = npl_row["npl_uuid"]
    for replication in range(npl_row["num_replications"]):
        # 1. load portfolio dataset
        generator = np.random.default_rng(seed=replication)
        if dataset in ("newsvendor", "portfolio_synthetic"):
            # NOTE if contamination is specified, then only contaminate the training samples (not test samples)
            data = sample_dgp(
                dgp, npl_row["num_observations"], dim=npl_row["dim"], contamination=npl_row["contamination"], generator=generator
            )
        elif dataset == "james":
            raise NotImplementedError("Here, portfolio_dataset_james is being supplied with dim in spite of this no longer being one of its")
            # data, _ = portfolio_dataset_james(time_window_id=replication, dim=npl_row["dim"])
        elif dataset == "portfolio":
            data, _ = portfolio_dataset(dgp, replication, dataset_dir)
            if npl_row["normalise"]:
                data = normalise_by_dimension(data)
        else:
            raise NotImplementedError(f"Dataset not implemented: {dataset}")

        # 2. sample from the posterior
        print()
        npl_start = datetime.now()
        print(npl_start, "- Starting", dataset, "sample NPL for replication", replication)    
        theta_sample = sample_npl(
            data,
            npl_row["inference"],
            npl_row["likelihood"],
            npl_row["num_posterior_samples"],
            seed=replication,
            lengthscale=npl_row["lengthscale"],
            generator=generator,
            dim=npl_row["dim"],
            kernel_name=npl_row["kernel_name"],
            eta=npl_row["eta"],
        )
        npl_finish = datetime.now()
        total_seconds =  (datetime.now() - npl_start).total_seconds()
        print(npl_finish, "- Finished replication", replication, "in", total_seconds, "seconds.")

        df = pd.DataFrame(theta_sample)
        df.to_csv(npl_dir / f"npl_sample_{replication}.csv", index=False, header=False)
        posterior_times.append({
            "replication": replication,
            "npl_uuid": npl_uuid,
            "posterior_time": total_seconds,
        })
    pd.DataFrame(posterior_times).to_csv(npl_samples_dir / npl_uuid / f"npl_times_{npl_uuid}.csv", index=False)

if __name__ == "__main__":
    app()
