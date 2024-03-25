"""Entrypoint app functions"""

from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import expon
import typer

from bayesian_dro.Bayesian_DRO_continuous import (
    main_Bayesian_DRO,
    data_generation,
    xi_generation,
    theta_generation,
)
from .constants import (
    CONTAMINATION_LEVEL,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
)
from .dataset import data_generation_outliers, data_generation_gamma
from .experiments import ExperimentName, get_experiment
from .npl import Npl
from .newsvendor import newsvendor_cost
from .models import ExponentialModel
from .optimise import solve_bdro

app = typer.Typer(name="misdro")


@app.command(name="setup")
def setup(experiment_name: ExperimentName, experiment_dir: Path, overwrite: bool = False):
    """Setup an experiment in a new directory"""
    experiment_name = "newsvendor_1d"
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)

    # get the experiment from the name
    experiment = get_experiment(experiment_name)

    # write experiment file to JSON
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    # get unique DGPs
    dgps = set()
    for params in experiment:
        dgps.add(params["dgp"])

    # setup SLURM file
    with open(Path(__file__).parent / "template.slurm", "r", encoding="utf-8") as slurm_file:
        slurm_string = slurm_file.read()
    for dgp in dgps:
        dgp_string = slurm_string.format(experiment_dir=experiment_dir, dgp=dgp)
        (experiment_dir / f"{experiment_name}_{dgp}.slurm").write_text(dgp_string)

@app.command(name="csv")
def generate_csv(experiment_dir: Path):
    """Write a CSV file with all the results"""
    experiment_filepath = experiment_dir / "experiment.json"
    with open(experiment_filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    df = pd.DataFrame(experiment).set_index("uuid")
    result_df = pd.concat(Parallel(n_jobs=-1)(
        delayed(pd.read_csv)(experiment_dir / f"{uuid}.csv", index_col=["uuid", "replication"]) for uuid in df.index if (experiment_dir / f"{uuid}.csv").exists()
    ))
    result_df = result_df.join(df, on="uuid")
    result_df.to_csv(experiment_dir / "results.csv", index=True)


@app.command(name="experiment")
def run_experiment(experiment_dir: Path, dgp: str, only_missing: bool = False):
    """When using SLURM, this function is called to run an experiment"""
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    # NOTE if only-missing flag, then only run if the results CSV file doesn't exist
    Parallel(n_jobs=-1)(
        delayed(run)(experiment_dir, **params)
        for params in experiment
        if params["dgp"] == dgp and not((experiment_dir / params["uuid"]).exists() and only_missing)
    )


@app.command(name="run")
def run(
    experiment_dir: Path,
    algorithm: str = "bayesian_dro",
    contamination: float = CONTAMINATION_LEVEL,
    dgp: str = "gamma",
    epsilon: float = 1.0,
    lengthscale: float = -1.0,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_replications: int = NUM_REPLICATIONS,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    posterior: str = "bayes",
    uuid: str = str(uuid4()),
):
    """Run Newsvendor Misspecified Bayesian DRO"""
    if uuid:
        print("Running", uuid)
    p = 1  # numbers of unknown parameters
    cost = np.zeros((num_replications, num_test_observations))  # init costs for each run
    solutions = np.zeros(num_replications)
    times = {
        "dgp_time": [],
        "posterior_time": [],
        "likelihood_time": [],
        "solve_time": [],
    }

    for j in range(num_replications):
        # generate dataset
        dgp_start = datetime.now()
        generator = np.random.default_rng(seed=j)
        if dgp == "truncated_normal":
            data = data_generation(num_observations, random_state=generator)  # generate new observations
            data_eval = data_generation(num_test_observations, random_state=generator)  # test data
        elif dgp == "contaminated_exp":
            # specify contamination level for train data
            data = data_generation_outliers(num_observations, contamination, random_state=generator)
            # but DO NOT specify any contamination for test data!
            data_eval = data_generation_outliers(num_test_observations, 0.0, random_state=generator)
        elif dgp == "exponential":
            data = expon.rvs(scale=20.0, size=num_observations, random_state=generator)
            data_eval = expon.rvs(scale=20.0, size=num_test_observations, random_state=generator)
        elif dgp == "gamma":
            data = data_generation_gamma(num_observations, a=10, random_state=generator)
            data_eval = data_generation_gamma(num_test_observations, a=10, random_state=generator)
        else:
            raise ValueError(
                f"The data-generating process specified is not supported: {dgp}"
            )
        times["dgp_time"].append((datetime.now() - dgp_start).total_seconds())

        # sample from the posterior
        posterior_start = datetime.now()
        theta_sample = np.zeros((num_posterior_samples, 1))
        if posterior in ("wll", "mmd"):
            # NPL posterior sample for theta
            m = NUM_OBSERVATIONS
            model = ExponentialModel(m)
            npl_toy = Npl(
                data.reshape((num_observations, 1)),
                num_posterior_samples,
                p,
                m,
                model,
                l=lengthscale,
                loss_fn=posterior,
            )
            npl_toy.draw_samples(random_state=generator)
            theta_sample = npl_toy.sample
        elif posterior == "bayes":
            # standard Bayesian posterior sample for theta
            theta_sample = theta_generation(data, num_posterior_samples, random_state=generator)
        times["posterior_time"].append(
            (datetime.now() - posterior_start).total_seconds()
        )

        # sample from the likelihood
        likelihood_start = datetime.now()
        xi = np.zeros([num_posterior_samples, num_likelihood_samples])  # init the xi's
        for i in range(num_posterior_samples):
            xi[i] = xi_generation(theta_sample[i], num_likelihood_samples, random_state=generator)
        times["likelihood_time"].append(
            (datetime.now() - likelihood_start).total_seconds()
        )

        # run the chosen DRO algorithm
        solve_start = datetime.now()
        if algorithm == "bayesian_dro":
            solutions[j], _ = solve_bdro(xi, epsilon)
        elif algorithm == "bdro_grid_search":
            solutions[j] = main_Bayesian_DRO(xi, epsilon)
        # TODO put in other algorithms here, e.g. main_Bayesian_DRO_epsilon1!
        else:
            solutions[j] = 0
            raise ValueError("Please choose a valid algorithm")
        times["solve_time"].append((datetime.now() - solve_start).total_seconds())

        # evaluate the cost
        cost[j] = newsvendor_cost(solutions[j], data_eval)

    # Calculate out-of-sample mean and variance
    mean_cost = np.mean(cost, axis=1)
    var_cost = np.var(cost, axis=1)
    mean_of_means = np.mean(mean_cost)
    mean_of_variances = np.mean(var_cost)
    var_of_means = np.var(mean_cost)

    print(f"Finished running {algorithm} with posterior {posterior} and DGP {dgp}.")
    print(f"Out-of-sample mean: {mean_of_means}. Out-of-sample variances: {mean_of_variances + var_of_means}.")
    print("Total DGP time:", sum(times["dgp_time"]))
    print("Total posterior time:", sum(times["posterior_time"]))
    print("Total likelihood time:", sum(times["likelihood_time"]))
    print("Total solve time:", sum(times["solve_time"]))

    df = pd.DataFrame({
        "uuid": [uuid] * num_replications,
        "replication": np.arange(num_replications),
        "mean_cost": mean_cost,
        "var_cost": var_cost,
        "solution": solutions,
        "dgp_time": times["dgp_time"],
        "likelihood_time": times["likelihood_time"],
        "posterior_time": times["posterior_time"],
        "solve_time": times["solve_time"],
    })
    df.to_csv(experiment_dir / f"{uuid}.csv", index=False)


if __name__ == "__main__":
    app()
