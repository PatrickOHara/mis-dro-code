from mis_dro.npl import *
from mis_dro.dataset import sample_dgp
from mis_dro.constants import (
    CONTAMINATION_LEVEL,
    NUM_LIKELIHOOD_SAMPLES,
    NUM_OBSERVATIONS,
    NUM_POSTERIOR_SAMPLES,
    NUM_REPLICATIONS,
    NUM_TEST_OBSERVATIONS,
    NUM_CERTIFY,
    MAX_PARAMS_OOM,
)
from datetime import datetime
from mis_dro.dataset import *
import pandas as pd

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
    )

    df = pd.DataFrame(theta_sample)
    df.to_csv(experiment_dir / f"portfolio_theta_sample_{dgp}_{replication}.csv", index=False, header=False)

if __name__ == "__main__":
    for cont in [0.1, 0.2, 0.0]:
        for j in range(200):
            portfolio_sample_npl(
                j,
                num_posterior_samples=30    # FIXME is this correct?
            )
