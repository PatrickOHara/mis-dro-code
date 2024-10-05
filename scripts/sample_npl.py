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

def run_replication(
    replication: int,
    contamination: float = CONTAMINATION_LEVEL,
    dgp: str = "truncated_normal",
    dim: int = 1,
    inference: str = "npl_mmd",
    lengthscale: float = -1.0,
    likelihood: str = "exponential",
    num_observations: int = NUM_OBSERVATIONS,
    num_posterior_samples: int = NUM_POSTERIOR_SAMPLES,
):
    """Run a single replication where the seed is given by the replication number"""
    # 1. generate dataset
    generator = np.random.default_rng(seed=replication)
    dgp_start = datetime.now()
    # NOTE if contamination is specified, then only contaminate the training samples (not test samples)
    data = sample_dgp(
        dgp, num_observations, dim=dim, contamination=contamination, generator=generator
    )
    dgp_time = (datetime.now() - dgp_start).total_seconds()

    # 2. sample from the posterior
    posterior_start = datetime.now()
    
    theta_sample = sample_npl(
        data,
        inference,
        likelihood,
        num_posterior_samples,
        seed=replication,
        lengthscale=lengthscale,
        generator=generator,
    )

    # experiment_dir = "/dcs/pg23/u1604520/misdro/npl_samples_N30_mvn/"
    experiment_dir = "/dcs/pg23/u1604520/misdro/npl_samples_N90_n20_mvn_bimodal_known_cov/"
    df = pd.DataFrame(theta_sample)
    if contamination == 0.1:
        c = '01'
    elif contamination == 0.2:
        c = '02'
    elif contamination == 0.0:
        c = '00'
    df.to_csv(experiment_dir+f"theta_sample_{replication}_cont{c}.csv", index=False, header=False)

if __name__ == "__main__":
    for cont in [0.1, 0.2, 0.0]:
        for j in range(100):
            run_replication(
                j,
                contamination=cont,
                dgp = "bimodal_multivariate_gaussian",
                # dgp = "contaminated_exp_old",
                dim = 5, # dim = 5,
                likelihood = "multivariate_normal_known_cov",
                # likelihood="exponential",
                num_observations=20,
                num_posterior_samples=90
            )
