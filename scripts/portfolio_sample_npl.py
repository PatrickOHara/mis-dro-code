from pathlib import Path
import typer
from mis_dro.npl import *
from mis_dro.constants import NUM_POSTERIOR_SAMPLES
from datetime import datetime
from mis_dro.dataset import get_num_time_windows, portfolio_dataset
import pandas as pd

app = typer.Typer(name="portfolio_sample_npl")

@app.command("portfolio-npl")
def portfolio_sample_npl(
    dataset_dir: Path,
    experiment_dir: Path,
    dgp: str = "DowJones",
    inference: str = "npl_mmd",
    lengthscale: float = -1.0,
    likelihood: str = "multivariate_normal",
    num_posterior_samples: int = 90,
):
    """Run a single replication where the seed is given by the replication number"""
    returns_df = pd.read_excel(dataset_dir / "Datasets" / dgp / f"{dgp}.xlsx", sheet_name="Assets_Returns", header=None)
    num_time_windows = get_num_time_windows(len(returns_df))

    for replication in range(num_time_windows):
        # 1. load portfolio dataset
        generator = np.random.default_rng(seed=replication)
        data, data_eval = portfolio_dataset(dgp, replication, dataset_dir)

        # 2. sample from the posterior
        print()
        npl_start = datetime.now()
        print(npl_start, "- Starting portfolio sample NPL for replication", replication)    
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
        npl_finish = datetime.now()
        print(npl_finish, "- Finished replication", replication, "in", (datetime.now() - npl_start).total_seconds(), "seconds.")

        df = pd.DataFrame(theta_sample)
        df.to_csv(experiment_dir / f"portfolio_theta_sample_{dgp}_{replication}.csv", index=False, header=False)


if __name__ == "__main__":
    app()

