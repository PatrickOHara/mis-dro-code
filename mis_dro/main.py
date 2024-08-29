"""Entrypoint app functions"""

import warnings
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4, UUID
from joblib import Parallel, delayed
import cvxpy as cp
import numpy as np
import pandas as pd
import typer

from bayesian_dro.Bayesian_DRO_continuous import main_Bayesian_DRO
from .bayes_conjugates import (
    sample_posterior,
    default_prior_params,
    get_kl_bdro_constant,
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
    MAX_PARAMS_OOM,
)
from .dataset import sample_dgp
from .experiments import ExperimentName, get_experiment
from .likelihood import sample_likelihood
from .newsvendor import newsvendor_cost_cvxpy
from .npl import sample_npl
from .optimise import get_kl_bdro_problem

app = typer.Typer(name="misdro")


@app.command(name="setup")
def setup(
    experiment_name: ExperimentName, experiment_dir: Path, overwrite: bool = False, njobs: int = -1,
):
    """Setup an experiment in a new directory"""
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get the experiment from the name
    experiment = get_experiment(experiment_name)

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
        Path(__file__).parent / "template.slurm", "r", encoding="utf-8"
    ) as slurm_file:
        slurm_string = slurm_file.read()
    for dgp, algorithm in dgp_algorithm_pairs:
        dgp_string = slurm_string.format(
            experiment_dir=experiment_dir, dgp=dgp, algorithm=algorithm, njobs=njobs,
        )
        (experiment_dir / f"{experiment_name}_{dgp}_{algorithm}.slurm").write_text(
            dgp_string
        )


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
    only_missing: bool = False,
    njobs: int = -1,
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


@app.command(name="uuid")
def run_uuid(experiment_dir: Path, uuid: UUID, njobs: int = -1, verbose: bool = False) -> None:
    """Run DRO for only one specified uuid parameters"""
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    found = False
    for params in experiment:
        if params["uuid"] == str(uuid):
            found = True
            run(experiment_dir, verbose=verbose, njobs=njobs, **params)
    if not found:
        raise ValueError(f"UUID {uuid} not found in {filepath}")


@app.command(name="run")
def run(
    experiment_dir: Path,
    algorithm: str = "kl_bdro",
    contamination: float = CONTAMINATION_LEVEL,
    dgp: str = "truncated_normal",
    epsilon: float = 1.0,
    inference: str = "bayes",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
    njobs: int = -1,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_replications: int = NUM_REPLICATIONS,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    posterior: str = "gamma",
    uuid: str = str(uuid4()),
    verbose: bool = False,
):
    """Run Newsvendor Misspecified Bayesian DRO"""
    if uuid:
        print(uuid)
    print("DGP:", dgp, " - ALGORITHM:", algorithm, " - NUM LIKELIHOOD SAMPLES:", num_likelihood_samples, " - POSTERIOR:", posterior)
    if algorithm in ("kl_bdro", "our_kl_bdro"):
        problem = get_kl_bdro_problem(
            newsvendor_cost_cvxpy, num_posterior_samples, num_likelihood_samples
        )
    elif algorithm == "kdro":
        raise NotImplementedError("Harita's future KDRO code goes here :)")
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented.")

    # If the number of parameters is small enough, then use Disciplined Parametrized Programming (DPP)
    # to reduce the compilation time in each replication.
    # However, a large number of parameters uses an enormous amount of RAM in the current cvxpy implementation.
    ignore_dpp = False
    if algorithm in ("kl_bdro", "our_kl_bdro", "kdro"):
        n_parameters = np.sum(np.prod(param.shape) for param in problem.parameters())
        # NOTE whilst BAS-DRO can handle at least 5000 params, BDRO cannot.
        # So, for a fair comparison, we turn off DPP for both BAS-DRO and BDRO.
        # if n_parameters >= cp.settings.PARAM_THRESHOLD:
        if n_parameters >= 1000:
            ignore_dpp = True
            # njobs = 1

    # BDRO n_parameters ~ >= 900  -> OOM
    # BAS-DRO n_parameters < 10,000 -> still good
    # n_parameters = 2500 -> both BDRO and BAS-DRO, we turn off dpp and njobs = 1

    # on PARROT, we run n_parameters <= 900
    # on PARROT, we run n_parameters >=2500, then turn off dpp for everything

    # DPP is good 
    params = {
        "algorithm": algorithm,
        "contamination": contamination,
        "dgp": dgp,
        "epsilon": epsilon,
        "ignore_dpp": ignore_dpp,
        "inference": inference,
        "lengthscale": lengthscale,
        "likelihood": likelihood,
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
    dgp: str = "truncated_normal",
    epsilon: float = 1.0,
    ignore_dpp: bool = False,
    inference: str = "bayes",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
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
    generator = np.random.default_rng(seed=replication)
    dgp_start = datetime.now()
    # NOTE if contamination is specified, then only contaminate the training samples (not test samples)
    data = sample_dgp(
        dgp, num_observations, contamination=contamination, generator=generator
    )
    data_eval = sample_dgp(
        dgp, num_test_observations, contamination=0.0, generator=generator
    )
    dgp_time = (datetime.now() - dgp_start).total_seconds()

    # 2. sample from the posterior
    posterior_start = datetime.now()
    kl_bdro_constant = 0.0
    if inference == "bayes":
        theta_prior = default_prior_params(posterior)
        theta_posterior = get_posterior_params(posterior, data, theta_prior)
        if algorithm == "our_kl_bdro":
            assert num_posterior_samples == 1
            kl_bdro_constant = get_kl_bdro_constant(posterior, theta_posterior)
            theta_sample = derive_analytical_posterior_params(
                posterior, theta_posterior
            )
        else:
            theta_sample = sample_posterior(
                posterior,
                theta_posterior,
                num_posterior_samples,
                generator=generator,
            )
    elif inference in ("npl_wlb", "npl_mmd"):
        theta_sample = sample_npl(
            data,
            inference,
            posterior,
            num_posterior_samples,
            lengthscale=lengthscale,
            generator=generator,
        )
    else:
        raise ValueError(f"Inference procedure '{inference}' is not supported.")
    assert kl_bdro_constant >= 0
    # assert kl_bdro_constant < epsilon
    posterior_time = (datetime.now() - posterior_start).total_seconds()

    # 3. sample from the likelihood
    likelihood_start = datetime.now()
    xi = sample_likelihood(
        likelihood,
        posterior,
        theta_sample,
        num_likelihood_samples,
        generator=generator,
    )
    likelihood_time = (datetime.now() - likelihood_start).total_seconds()

    # 4. run the chosen DRO algorithm
    solve_start = datetime.now()
    solution = np.nan
    if algorithm in ("kl_bdro", "our_kl_bdro"):
        if epsilon - kl_bdro_constant < 0:
            # NOTE the optimisation problem is unbounded below
            solution = np.inf
            solve_time = 0.0
            setup_time = 0.0
        else:
            # set parameters then solve
            problem.param_dict["epsilon_minus_constant"].value = np.array([epsilon - kl_bdro_constant])
            problem.param_dict["xi"].value = xi
            # NOTE the MOSEK 'accept_unknown' argument is needed due to https://github.com/cvxpy/cvxpy/pull/2117
            problem.solve(solver=cp.MOSEK, verbose=verbose, ignore_dpp=ignore_dpp, accept_unknown=True)
            solution = problem.var_dict["x"].value[0]
            # solve_time = problem.solver_stats.solve_time
            setup_time = problem.solver_stats.setup_time
    elif algorithm == "kdro":
        raise NotImplementedError("Harita's future code goes here :)")
    elif algorithm == "bdro_grid_search":
        solution = main_Bayesian_DRO(xi, epsilon)
        setup_time = 0.0  # can't really measure this easily
    # TODO put in other algorithms here, e.g. main_Bayesian_DRO_epsilon1!
    else:
        raise ValueError("Please choose a valid algorithm")
    solve_time = (datetime.now() - solve_start).total_seconds()
    # evaluate the cost
    if solution == np.inf:
        out_of_sample_mean = np.inf
        out_of_sample_var = 0.0
    else:
        out_of_sample_costs = newsvendor_cost_cvxpy(solution, data_eval).value
        out_of_sample_mean = np.mean(out_of_sample_costs)
        out_of_sample_var = np.var(out_of_sample_costs)
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
    }

if __name__ == "__main__":
    app()
