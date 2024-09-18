"""Entrypoint app functions"""

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
    NUM_CERTIFY
)
from .dataset import sample_dgp
from .experiments import ExperimentName, get_experiment
from .likelihood import sample_likelihood
from .newsvendor import newsvendor_cost_cvxpy
from .npl import sample_npl
from .optimise import get_kl_bdro_problem, DRO_BAS_MMD
from .gaussian_kernel import *

app = typer.Typer(name="misdro")


@app.command(name="setup")
def setup(
    experiment_name: ExperimentName, experiment_dir: Path, overwrite: bool = False
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
        if algorithm in ("kl_bdro"):
            njobs = 1
        else:
            njobs = -1
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
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    # NOTE if only-missing flag, then only run if the results CSV file doesn't exist
    Parallel(n_jobs=njobs)(
        delayed(run)(experiment_dir, **params)
        for params in experiment
        if params["dgp"] == dgp
        and params["algorithm"] == algorithm
        and not ((experiment_dir / params["uuid"]).exists() and only_missing)
    )


@app.command(name="uuid")
def run_uuid(experiment_dir: Path, uuid: UUID, verbose: bool = False) -> None:
    """Run DRO for only one specified uuid parameters"""
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    found = False
    for params in experiment:
        if params["uuid"] == str(uuid):
            found = True
            run(experiment_dir, verbose=verbose, **params)
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
        print("Running", uuid)
    cost = np.zeros(
        (num_replications, num_test_observations)
    )  # init costs for each run
    solutions = np.zeros(num_replications)
    times = {
        "dgp_time": [],
        "posterior_time": [],
        "likelihood_time": [],
        "solve_time": [],
        "setup_time": [],
    }
    if algorithm in ("kl_bdro", "our_kl_bdro"):
        problem = get_kl_bdro_problem(
            newsvendor_cost_cvxpy, num_posterior_samples, num_likelihood_samples
        )
    elif algorithm in ("dro_bas_mmd", "empirical_mmd"):
        dim_theta = 1
        kdro_class = DRO_BAS_MMD(dim_theta, newsvendor_cost_cvxpy)
        if algorithm == "dro_bas_mmd":
            n_samples = num_posterior_samples*num_likelihood_samples
        elif algorithm == "empirical_mmd":
            n_samples = num_observations
        problem = kdro_class.get_problem(n_samples, num_certify_points)

    # If the number of parameters is small enough, then use Disciplined Parametrized Programming (DPP)
    # to reduce the compilation time in each replication.
    # However, a large number of parameters uses an enormous amout of RAM in the current cvxpy implementation.
    if algorithm in ("kl_bdro", "our_kl_bdro", "dro_bas_mmd", "empirical_mmd"):
        ignore_dpp = False
        n_parameters = np.sum(np.prod(param.shape) for param in problem.parameters())
        if n_parameters >= cp.settings.PARAM_THRESHOLD:
            ignore_dpp = True

    for j in range(num_replications):
        # 1. generate dataset
        generator = np.random.default_rng(seed=j)
        dgp_start = datetime.now()
        # NOTE if contamination is specified, then only contaminate the training samples (not test samples)
        data = sample_dgp(
            dgp, num_observations, contamination=contamination, generator=generator
        )
        data_eval = sample_dgp(
            dgp, num_test_observations, contamination=0.0, generator=generator
        )
        times["dgp_time"].append((datetime.now() - dgp_start).total_seconds())

        # 2. sample from the posterior
        posterior_start = datetime.now()
        kl_bdro_constant = 1.0
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
                seed=j,
                l=lengthscale,
                generator=generator,
            )
        elif posterior == "bayes":
            # standard Bayesian posterior sample for theta
            theta_sample = theta_generation(data, num_posterior_samples, random_state=generator)
        times["posterior_time"].append(
            (datetime.now() - posterior_start).total_seconds()
        )

        # 3. sample from the likelihood
        likelihood_start = datetime.now()
        xi = sample_likelihood(
            likelihood,
            posterior,
            theta_sample,
            num_likelihood_samples,
            generator=generator,
        )
        times["likelihood_time"].append(
            (datetime.now() - likelihood_start).total_seconds()
        )

        # 4. run the chosen DRO algorithm
        solve_start = datetime.now()
        if algorithm in ("kl_bdro", "our_kl_bdro"):
            # set parameters then solve
            problem.param_dict["kl_bdro_constant"].value = np.array([kl_bdro_constant])
            problem.param_dict["xi"].value = xi
            problem.param_dict["epsilon"].value = np.array([epsilon])
            problem.solve(solver=cp.MOSEK, verbose=verbose, ignore_dpp=ignore_dpp)
            solutions[j] = problem.var_dict["x"].value
            times["solve_time"].append(problem.solver_stats.solve_time)
            times["setup_time"].append(problem.solver_stats.setup_time)
        elif algorithm in ("dro_bas_mmd", "empirical_mmd"):
            if algorithm == "dro_bas_mmd":
                xi = xi.reshape((num_likelihood_samples*num_posterior_samples,1))
            elif algorithm == "empirical_mmd":
                xi = data.reshape((num_observations,1))
            _, dim_x = xi.shape
            Xcert = np.random.uniform(np.min(xi), np.max(xi), size=[num_certify_points,dim_x])
            zetai = np.concatenate([xi, Xcert])
            l = np.sqrt((1/2)*np.median(distance.cdist(zetai, zetai, 'sqeuclidean')))
            K = k_jax(zetai, zetai, l)
            K_decomp = mat_decomp_jax(K)
            problem.param_dict["Xobs"].value = xi
            problem.param_dict["Xcert"].value = Xcert
            problem.param_dict["K"].value = np.asarray(K)
            problem.param_dict["K_decomposed"].value = np.asarray(K_decomp)
            problem.param_dict["epsilon"].value = np.array([epsilon])
            problem.solve(cp.MOSEK, verbose=False, ignore_dpp=ignore_dpp)
            solutions[j] = problem.var_dict["theta"].value
        elif algorithm == "bdro_grid_search":
            solutions[j] = main_Bayesian_DRO(xi, epsilon)
            times["solve_time"].append((datetime.now() - solve_start).total_seconds())
            times["setup_time"].append(0.0)  # can't really measure this easily
        # TODO put in other algorithms here, e.g. main_Bayesian_DRO_epsilon1!
        else:
            solutions[j] = 0
            raise ValueError("Please choose a valid algorithm")

        # evaluate the cost
        cost[j] = newsvendor_cost_cvxpy(solutions[j], data_eval).value

    # Calculate out-of-sample mean and variance
    mean_cost = np.mean(cost, axis=1)
    var_cost = np.var(cost, axis=1)
    mean_of_means = np.mean(mean_cost)
    mean_of_variances = np.mean(var_cost)
    var_of_means = np.var(mean_cost)

    print(f"Finished running {algorithm} with posterior {posterior} and DGP {dgp}.")
    print(
        f"Out-of-sample mean: {mean_of_means}. Out-of-sample variances: {mean_of_variances + var_of_means}."
    )
    print("Total DGP time:", sum(times["dgp_time"]))
    print("Total posterior time:", sum(times["posterior_time"]))
    print("Total likelihood time:", sum(times["likelihood_time"]))
    print("Total solve time:", sum(times["solve_time"]))

    df = pd.DataFrame(
        {
            "uuid": [uuid] * num_replications,
            "replication": np.arange(num_replications),
            "mean_cost": mean_cost,
            "var_cost": var_cost,
            "solution": solutions,
            "dgp_time": times["dgp_time"],
            "likelihood_time": times["likelihood_time"],
            "posterior_time": times["posterior_time"],
            "solve_time": times["solve_time"],
            "setup_time": times["setup_time"],
        }
    )
    csv_filepath = experiment_dir / f"{uuid}.csv"
    print(f"Writing CSV to {csv_filepath}")
    df.to_csv(csv_filepath, index=False)


if __name__ == "__main__":
    app()
