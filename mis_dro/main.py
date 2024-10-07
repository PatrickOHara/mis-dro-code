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

from bayesian_dro.Bayesian_DRO_continuous import main_Bayesian_DRO
from .bayes_conjugates import (
    sample_posterior,
    default_prior_params,
    get_log_partition_constant,
    get_posterior_params,
    derive_analytical_posterior_params,
)
from .constants import (
    CONTAMINATION_LEVEL,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
    NUM_CERTIFY,
    MAX_PARAMS_OOM,
)
from .dataset import sample_dgp, portfolio_dataset
from .experiments import ExperimentName, get_experiment
from .likelihood import sample_likelihood, reconstruct_covariance_from_triu
from .newsvendor import newsvendor_cost_cvxpy
from .npl import sample_npl
from .optimise import get_kl_bdro_problem, DRO_BAS_MMD
from .portfolio import get_kl_portfolio_problem, bdro_portfolio_posterior_samples, portfolio_objective_cvxpy
from .gaussian_kernel import *

app = typer.Typer(name="misdro")


@app.command(name="setup-kl")
def setup_kl_dro_bas(
    experiment_name: ExperimentName, experiment_dir: Path, dataset_dir: Path = Path("~/datasets/misdro/mmc2"), overwrite: bool = False
):
    """Setup an experiment in a new directory"""
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get the experiment from the name
    experiment = get_experiment(experiment_name, dataset_dir=dataset_dir)

    # write experiment file to JSON
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    # get unique DGPs
    dgp_algorithm_pairs = set()
    for params in experiment:
        dgp_algorithm_pairs.add((params["dgp"], params["algorithm"]))

    # setup SLURM file
    with open(
        Path(__file__).parent / "kl_dro_bas_template.slurm", "r", encoding="utf-8"
    ) as slurm_file:
        slurm_string = slurm_file.read()
    for dgp, algorithm in dgp_algorithm_pairs:
        dgp_string = slurm_string.format(
            experiment_dir=experiment_dir, dgp=dgp, algorithm=algorithm,
        )
        (experiment_dir / f"{experiment_name}_{dgp}_{algorithm}.slurm").write_text(
            dgp_string
        )


@app.command(name="setup-mmd")
def setup_mmd_dro_bas(
    experiment_name: ExperimentName, experiment_dir: Path, batch_size: int, dataset_dir: Path = Path("~/datasets/misdro/mmc2"), overwrite: bool = False, njobs: int = -1,
):
    """Setup an experiment in a new directory"""
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get the experiment from the name
    experiment = get_experiment(experiment_name, dataset_dir=dataset_dir)

    # write experiment file to JSON
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    # for the given batch size, how many batches do we need?
    num_batches = math.ceil(float(len(experiment)) / float(batch_size))

    # setup SLURM file
    with open(
        Path(__file__).parent / "mmd_dro_bas_template.slurm", "r", encoding="utf-8"
    ) as slurm_file:
        slurm_string = slurm_file.read()
    dgp_string = slurm_string.format(
        experiment_dir=experiment_dir, njobs=njobs, num_batches=num_batches, batch_size=batch_size
    )
    (experiment_dir / f"{experiment_name}.slurm").write_text(dgp_string)


@app.command(name="csv")
def generate_csv(experiment_dir: Path):
    """Write a CSV file with all the results"""
    experiment_filepath = experiment_dir / "experiment.json"
    with open(experiment_filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    df = pd.DataFrame(experiment).set_index("uuid")
    result_df = pd.concat(
        [
            pd.read_csv(
                experiment_dir / f"{uuid}.csv", index_col=["uuid", "replication"]
            )
            for uuid in df.index
            if (experiment_dir / f"{uuid}.csv").exists()
        ]
    )
    result_df = result_df.join(df, on="uuid")
    result_df.to_csv(experiment_dir / "results.csv", index=True)


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
def batch(experiment_dir: Path, start: int, batch_size: int, only_missing: bool = False, dataset_dir: Path = Path("~/datasets/misdro/mmc2")):
    print(datetime.now(), "Running batch from index", start)
    print()
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    batch_experiment = experiment[start: min(start + batch_size, len(experiment))]
    for params in batch_experiment:
        if not ((experiment_dir / params["uuid"]).exists() and only_missing):
            run(experiment_dir, dataset_dir=dataset_dir, **params)

@app.command(name="uuid")
def run_uuid(experiment_dir: Path, uuid: UUID, dataset_dir: Path = Path("~/datasets/misdro/mmc2"), njobs: int = -1, verbose: bool = False) -> None:
    """Run DRO for only one specified uuid parameters"""
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    found = False
    for params in experiment:
        if params["uuid"] == str(uuid):
            found = True
            run(experiment_dir, verbose=verbose, njobs=njobs, dataset_dir=dataset_dir, **params)
    if not found:
        raise ValueError(f"UUID {uuid} not found in {filepath}")



@app.command(name="run")
def run(
    experiment_dir: Path,
    algorithm: str = "kl_bdro",
    contamination: float = CONTAMINATION_LEVEL,
    dataset: str = "newsvendor",
    dataset_dir: Optional[Path] = None,
    dgp: str = "truncated_normal",
    dim: int = 1,
    epsilon: float = 1.0,
    ignore_dpp: bool = False,
    inference: str = "bayes",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
    njobs: int = -1,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_replications: int = NUM_REPLICATIONS,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    num_certify_points: int = NUM_CERTIFY,
    posterior: str = "gamma",
    uuid: str = str(uuid4()),
    verbose: bool = False,
):
    """Run Newsvendor Misspecified Bayesian DRO"""
    if uuid:
        print(uuid)
    print("DGP:", dgp, " - ALGORITHM:", algorithm, " - NUM LIKELIHOOD SAMPLES:", num_likelihood_samples, " - POSTERIOR:", posterior, "- DATASET:", dataset, "- DIM:", dim)
    if algorithm in ("kl_bdro", "kl_dro_bas") and dataset == "newsvendor":
        problem = get_kl_bdro_problem(
            newsvendor_cost_cvxpy, num_posterior_samples, num_likelihood_samples, dim=dim,
        )
    elif algorithm in ("kl_bdro", "kl_dro_bas") and dataset == "portfolio" and likelihood == "multivariate_normal":
        problem = get_kl_portfolio_problem(dim, num_posterior_samples)
    elif algorithm in ("dro_bas_mmd", "empirical_mmd"):
        dim_theta = dim
        if algorithm == "dro_bas_mmd":
            n_samples = num_posterior_samples*num_likelihood_samples
        elif algorithm == "empirical_mmd":
            n_samples = num_observations
        if dataset == "newsvendor":
            kdro_class = DRO_BAS_MMD(dim_theta, dim, newsvendor_cost_cvxpy)
            problem = kdro_class.get_newsvendor_problem(n_samples, num_certify_points)
        elif dataset == "portfolio":
            kdro_class = DRO_BAS_MMD(dim_theta, dim, portfolio_objective_cvxpy)
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
        "dgp": dgp,
        "dim": dim,
        "epsilon": epsilon,
        "experiment_dir": experiment_dir,
        "ignore_dpp": ignore_dpp,
        "inference": inference,
        "lengthscale": lengthscale,
        "likelihood": likelihood,
        "num_certify_points": num_certify_points,
        "num_likelihood_samples": num_likelihood_samples,
        "num_observations": num_observations,
        "num_posterior_samples": num_posterior_samples,
        "num_test_observations": num_test_observations,
        "posterior": posterior,
        "uuid": uuid,
        "verbose": verbose,
    }
    if njobs == 1:
        all_solve_start = datetime.now()
        list_of_replication_stats = []
        print(all_solve_start, "- Running all replications in series.")
        for j in range(num_replications):
            list_of_replication_stats.append(run_replication(j, problem, **params))
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


def run_replication(
    replication: int,
    problem: cp.Problem,
    algorithm: str = "kl_bdro",
    contamination: float = CONTAMINATION_LEVEL,
    dataset: str = "newsvendor",
    dataset_dir: Optional[Path] = None,
    dgp: str = "truncated_normal",
    dim: int = 1,
    epsilon: float = 1.0,
    experiment_dir: Optional[Path] = None,
    ignore_dpp: bool = False,
    inference: str = "bayes",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
    num_certify_points: int = NUM_CERTIFY,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    posterior: str = "gamma",
    uuid: str = str(uuid4()),
    verbose: bool = False,
):
    """Run a single replication where the seed is given by the replication number"""
    # 1. generate dataset
    dgp_start = datetime.now()
    generator = np.random.default_rng(seed=replication)
    if dataset == "newsvendor":
        # NOTE if contamination is specified, then only contaminate the training samples (not test samples)
        data = sample_dgp(
            dgp, num_observations, dim=dim, contamination=contamination, generator=generator
        )
        data_eval = sample_dgp(
            dgp, num_test_observations, dim=dim, contamination=0.0, generator=generator
        )
    elif dataset == "portfolio":
        # NOTE shape of data (N, D) where N is number of weeks and D is the number of stocks
        data, data_eval = portfolio_dataset(dgp, replication, dataset_dir)
    else:
        raise NotImplementedError(f"Dataset not implemented: {dataset}")
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
        elif algorithm == "kl_bdro" and dataset == "portfolio" and posterior == "normal_inverse_wishart":
            mu_post, _, iota_post, Psi_post = theta_posterior
            theta_sample = bdro_portfolio_posterior_samples(num_posterior_samples, mu_post, iota_post, Psi_post, generator=generator)
        else:
            theta_sample = sample_posterior(posterior, theta_posterior, num_likelihood_samples, generator=generator)
    elif inference in ("npl_wlb", "npl_mmd"):
        # theta_sample = sample_npl(
        #     data,
        #     inference,
        #     likelihood,
        #     num_posterior_samples,
        #     seed=replication,
        #     lengthscale=lengthscale,
        #     dim=dim,
        #     generator=generator,
        # )
        # load theta sample from csv files
        if dataset == "newsvendor":
            if contamination == 0.05:
                c = '005'
            if contamination == 0.2:
                c = '02'
            elif contamination == 0.1:
                c = '01'
            elif contamination == 0.0:
                c = '00'
            else:
                raise ValueError(f"There are no npl samples for contamination level {contamination}")
            path_to_csv = Path("/dcs/pg23/u1604520/misdro/npl_samples_N30_exp")
            theta_sample = pd.read_csv(path_to_csv / f"theta_sample_{replication}_cont{c}.csv", header=None).values
        elif dataset == "portfolio":
            path_to_csv = experiment_dir / f"portfolio_theta_sample_{dgp}_{replication}.csv"
            npl_df = pd.read_csv(path_to_csv, index_col=False, header=None)
            theta_sample = npl_df.values
            assert num_posterior_samples == theta_sample.shape[0]
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
    elif inference == "bayes" and dataset == "portfolio" and likelihood == "multivariate_normal":
        pass    # no need to sample from likelihood cause we have closed form
    else:
        xi = sample_likelihood(
            likelihood,
            theta_sample,
            dim,
            num_likelihood_samples,
            generator=generator,
        )
    likelihood_time = (datetime.now() - likelihood_start).total_seconds()

    # 4. run the chosen DRO algorithm
    solve_start = datetime.now()
    solution = np.nan
    if dataset == "portfolio" and algorithm in ("kl_bdro", "kl_dro_bas") and likelihood == "multivariate_normal":
        # if epsilon - log_partition_constant < 0:
        #     # NOTE the optimisation problem is unbounded below
        #     solution = np.inf * np.ones(dim)
        #     solve_time = 0.0
        #     setup_time = 0.0
        # else:
        problem.param_dict["epsilon_minus_constant"].value = np.array([epsilon])
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
    elif algorithm in ("kl_bdro", "kl_dro_bas"):
        if epsilon - log_partition_constant < 0:
            # NOTE the optimisation problem is unbounded below
            solution = np.inf * np.ones(dim)
            solve_time = 0.0
            setup_time = 0.0
        else:
            # set parameters then solve
            problem.param_dict["epsilon_minus_constant"].value = np.array([epsilon - log_partition_constant])
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
    # evaluate the cost
    if (solution == np.inf).any():
        out_of_sample_mean = np.inf
        out_of_sample_var = 0.0
        out_of_sample_costs = np.inf * np.ones(num_test_observations)
    else:
        if dataset == "newsvendor":
            out_of_sample_costs = newsvendor_cost_cvxpy(solution, data_eval).value
        elif dataset == "portfolio":
            # cost is interpreted as negative return (we want to maximise return)
            out_of_sample_costs = data_eval @ solution
        out_of_sample_mean = np.mean(out_of_sample_costs)
        out_of_sample_var = np.var(out_of_sample_costs)
        solution = list(solution)

    return {
        "uuid": uuid,
        "replication": replication,
        "mean_cost": out_of_sample_mean,
        "var_cost": out_of_sample_var,
        "solution": solution,
        "dgp_time": dgp_time,
        "likelihood_time": likelihood_time,
        "posterior_time": posterior_time,
        "solve_time": solve_time,
        "setup_time": setup_time,
        "log_partition_constant": log_partition_constant,
        "out_of_sample_costs": list(out_of_sample_costs),
    }

if __name__ == "__main__":
    app()
