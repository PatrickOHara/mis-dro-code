"""Entrypoint app functions"""

from pathlib import Path
import json
from uuid import uuid4
from tqdm import tqdm
import numpy as np
import typer

from bayesian_dro.Bayesian_DRO_continuous import (
    main_Bayesian_DRO, data_generation, xi_generation, theta_generation, EPSILON_SET
)
from .npl import Npl
from .newsvendor import newsvendor_cost

app = typer.Typer(name="misdro")

NUM_OBSERVATIONS: int = 1000,
NUM_POSTERIOR_SAMPLES: int = 1000,
NUM_TEST_OBSERVATIONS: int = 10000,

@app.command(name="setup")
def setup(experiment_dir: Path):
    """Setup an experiment in a new directory"""
    experiment_dir.mkdir(parents=False, exist_ok=False)
    experiment = []
    # iterate over each of the parameters
    for algorithm in ["bayesian_dro"]:  # TODO add more algorithms
        for epsilon in EPSILON_SET:
            for posterior in ["bayes", "npl"]:
                params = {
                    "algorithm": algorithm,
                    "epsilon": epsilon,
                    "num_observations": NUM_OBSERVATIONS,
                    "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
                    "num_test_observations": NUM_TEST_OBSERVATIONS,
                    "posterior": posterior, 
                }
                params["uuid"] = uuid4()    # uniquely identify a run
                experiment.append(params)
    filepath = experiment_dir / "experiment.json"
    with open(filepath, "w", encoding="utf-8") as json_file:
        json.dump(experiment, json_file, indent=4)
    # TODO setup SLURM file

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
        json.dump({"uuid": uuid, "mean_cost": mean_cost, "var_cost": var_cost}, json_file, indent=4)

@app.command(name="run")
def run(
    algorithm: str = "bayesian_dro",
    epsilon: float = 1.0,
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
    num_test_observations: int = NUM_TEST_OBSERVATIONS,
    posterior: str = "npl",
    **kwargs,
):
    """Run Newsvendor Misspecified Bayesian DRO"""
    if "uuid" in kwargs:
        print("Running", kwargs["uuid"])
    K = 200 # number of experiment runs 
    p = 1 # numbers of unknown parameters
    cost = np.zeros((K,2))  # init costs for each run

    for j in tqdm(range(K)):
        # generate dataset
        np.random.seed(j)   # set seed for reproducibility
        data = data_generation(num_observations) # generate new observations
        data_eval = data_generation(num_test_observations) # test data ## when we use the outlier data gen. function remember to specify contamination level for test data 

        # sample from the posterior
        theta_sample = np.zeros((num_posterior_samples, 1))
        if posterior == "npl":
            # NPL posterior sample for theta
            npl_toy = Npl(data.reshape((num_observations,1)),num_posterior_samples, p, loss_fn='wll')  # can change loss_fn to wll or mmd
            npl_toy.draw_samples()
            theta_sample = npl_toy.sample  
        elif posterior == "bayes":
            # standard Bayesian posterior sample for theta
            theta_sample = theta_generation(data, num_posterior_samples) 

        # sample from the likelihood
        xi = np.zeros([num_posterior_samples, num_observations])  # init the xi's 
        for i in range(num_posterior_samples):
            xi[i] = xi_generation(theta_sample[i], num_observations)

        # run the chosen DRO algorithm
        if algorithm == "bayesian_dro":
            solution = main_Bayesian_DRO(xi, epsilon)
        # TODO put in other algorithms here, e.g. main_Bayesian_DRO_epsilon1!
        else:
            solution = 0
            raise ValueError("Please choose a valid algorithm")
        # evaluate the cost
        cost_j = newsvendor_cost(solution, data_eval)
        cost[j,:] = cost_j.mean(), cost_j.std()**2

    # Calculate out-of-sample mean and variance
    mean_cost = cost[:,0].mean()
    var_cost = cost[:,1].mean() + (1/(K-1))*np.sum((cost[:,0]-mean_cost)**2)
    print(f'{algorithm} has out-of-sample mean: {mean_cost} and out-of-sample variance: {var_cost}')
    return mean_cost, var_cost


if __name__=="__main__":
    app()
