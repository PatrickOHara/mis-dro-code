import typer
from mis_dro.npl import *
from mis_dro.constants import NUM_POSTERIOR_SAMPLES
from datetime import datetime
from mis_dro.dataset import *
import pandas as pd

app = typer.Typer(name="portfolio_sample_npl")


def portfolio_sample_npl(
    replication: int,
    dataset_dir: Path,
    experiment_dir: Path,
    dgp: str = "DowJones",
    inference: str = "npl_mmd",
    lengthscale: float = -1.0,
    likelihood: str = "multivariate_normal",
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
):
    """Run a single replication where the seed is given by the replication number"""
    # 1. load portfolio dataset
    generator = np.random.default_rng(seed=replication)
    data, data_eval = portfolio_dataset(dgp, replication, dataset_dir)

    # 2. sample from the posterior
    print()
    print(datetime.now(), "- Starting portfolio sample NPL for replication", replication)    
    theta_sample = sample_npl(
        data,
        inference,
        likelihood,
        num_posterior_samples,
        seed=replication,
        lengthscale=lengthscale,
        generator=generator,
        dim=data.shape[1],
    )

    df = pd.DataFrame(theta_sample)
    df.to_csv(experiment_dir / f"portfolio_theta_sample_{dgp}_{replication}.csv", index=False, header=False)

@app.command()
def main_sample(dataset_dir: Path, experiment_dir: Path, dgp: str = "DowJones"):
    for j in range(200):
        portfolio_sample_npl(
            j,
            dataset_dir,
            experiment_dir,
            dgp=dgp,
            num_posterior_samples=30    # FIXME is this correct?
        )

if __name__ == "__main__":
    app()

