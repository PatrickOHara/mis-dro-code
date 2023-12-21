"""Entrypoint app functions"""

from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import typer

from bayesian_dro.Bayesian_DRO_continuous import (
    main_Bayesian_DRO,
    data_generation,
    xi_generation,
    theta_generation,
)
from .constants import *
from .dataset import data_generation_outliers
from .experiments import epsilon_experiment
from .npl import Npl
from .newsvendor import newsvendor_cost

app = typer.Typer(name="misdro")


@app.command(name="setup")
def setup(
    experiment_dir: Path,
    cpus_per_task: int = 2,
    mem_per_cpu: int = 4000,  # in MB
    overwrite: bool = False,
    partition: str = "cpu-batch",  # name of SLURM partition
    time_limit: int = 8,  # hours
):
    """Setup an experiment in a new directory"""
    if not experiment_dir.exists() or not overwrite:
        experiment_dir.mkdir(parents=False, exist_ok=False)
    experiment = epsilon_experiment()
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)

    # setup SLURM file
    slurm_string = f"""#!/usr/bin/bash
#SBATCH --job-name=misdro
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --time={time_limit}:00:00
#SBATCH --array=0-{len(experiment)-1}
#SBATCH --exclude=emu-01

srun --ntasks=1 --nodes=1 misdro run-index {experiment_dir} $SLURM_ARRAY_TASK_ID\n
"""
    (experiment_dir / "misdro.slurm").write_text(slurm_string)

@app.command(name="csv")
def generate_csv(experiment_dir: Path):
    """Write a CSV file with all the results"""
    experiment_filepath = experiment_dir / "experiment.json"
    with open(experiment_filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    df = pd.DataFrame(experiment)
    df.set_index("uuid", inplace=True)
    result_list = []
    for uuid in df.index:
        result_filepath = experiment_dir / f"{uuid}.json"
        if result_filepath.exists():
            with open(result_filepath, "r", encoding="utf-8") as json_file:
                result = json.load(json_file)
        else:
            result = {"uuid": uuid, "mean_cost": np.nan, "var_cost": np.nan}
        result_list.append(result)
    result_df = pd.DataFrame(result_list)
    result_df.set_index("uuid", inplace=True)
    df = df.join(result_df)
    print(df)
    df.to_csv(experiment_dir / "results.csv", index=True)


@app.command(name="run-index")
def run_index(experiment_dir: Path, index: int):
    """When using SLURM, this function is called"""
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "r", encoding="utf-8") as json_file:
        experiment = json.load(json_file)
    params = experiment[index]
    mean_cost, var_cost = run(**params)
    uuid = params["uuid"]
    results_filepath = experiment_dir / f"{uuid}.json"
    with open(results_filepath, "w", encoding="utf-8") as json_file:
        json.dump(
            {"uuid": uuid, "mean_cost": mean_cost, "var_cost": var_cost},
            json_file,
            indent=4,
        )


@app.command(name="run")
def run(
    algorithm: str = "bayesian_dro",
    contamination: float = CONTAMINATION_LEVEL,
    dgp: str = "truncated_normal",
    epsilon: float = 1.0,
    num_likelihood_samples: int = NUM_LIKELIHOOD_SAMPLES,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_replications: int = NUM_REPLICATIONS,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    posterior: str = "npl",
    uuid: str = "",
):
    """Run Newsvendor Misspecified Bayesian DRO"""
    if uuid:
        print("Running", uuid)
    p = 1  # numbers of unknown parameters
    cost = np.zeros((num_replications, 2))  # init costs for each run

    for j in tqdm(range(num_replications)):
        # generate dataset
        np.random.seed(j)  # set seed for reproducibility
        if dgp == "truncated_normal":
            data = data_generation(num_observations)  # generate new observations
            data_eval = data_generation(num_test_observations)  # test data
        elif dgp == "contaminated_exp":
            # specify contamination level for train data
            data = data_generation_outliers(num_observations, contamination)
            # but DO NOT specify any contamination for test data!
            data_eval = data_generation_outliers(num_test_observations, 0.0)
        else:
            raise ValueError(f"The data-generating process specified is not supported: {dgp}")

        # sample from the posterior
        theta_sample = np.zeros((num_posterior_samples, 1))
        if posterior == "npl":
            # NPL posterior sample for theta
            npl_toy = Npl(
                data.reshape((num_observations, 1)),
                num_posterior_samples,
                p,
                loss_fn="wll",
            )  # can change loss_fn to wll or mmd
            npl_toy.draw_samples()
            theta_sample = npl_toy.sample
        elif posterior == "bayes":
            # standard Bayesian posterior sample for theta
            theta_sample = theta_generation(data, num_posterior_samples)

        # sample from the likelihood
        xi = np.zeros([num_posterior_samples, num_likelihood_samples])  # init the xi's
        for i in range(num_posterior_samples):
            xi[i] = xi_generation(theta_sample[i], num_likelihood_samples)

        # run the chosen DRO algorithm
        if algorithm == "bayesian_dro":
            solution = main_Bayesian_DRO(xi, epsilon)
        # TODO put in other algorithms here, e.g. main_Bayesian_DRO_epsilon1!
        else:
            solution = 0
            raise ValueError("Please choose a valid algorithm")
        # evaluate the cost
        cost_j = newsvendor_cost(solution, data_eval)
        cost[j, :] = cost_j.mean(), cost_j.std() ** 2

    # Calculate out-of-sample mean and variance
    mean_cost = cost[:, 0].mean()
    var_cost = cost[:, 1].mean() + (1 / (num_replications - 1)) * np.sum((cost[:, 0] - mean_cost) ** 2)
    print(
        f"{algorithm} has out-of-sample mean: {mean_cost} and out-of-sample variance: {var_cost}"
    )
    return mean_cost, var_cost


if __name__ == "__main__":
    app()
