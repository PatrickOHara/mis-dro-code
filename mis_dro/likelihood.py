"""Sampling from likelihood"""

from typing import Optional
import numpy as np
import scipy as sp
from bayesian_dro.Bayesian_DRO_continuous import xi_generation

def sample_likelihood(likelihood: str, posterior: str, theta_sample: np.ndarray, num_likelihood_samples: int, generator: Optional[np.random.Generator]) -> np.ndarray:
    """Sample from the likelihood

    Returns:
        ndarray of shape `(num_posterior_samples, num_likelihood_samples)`
    """
    if not generator:
        generator = np.random.default_rng()
    num_posterior_samples = theta_sample.shape[0]
    xi = np.zeros([num_posterior_samples, num_likelihood_samples])
    if likelihood == "exponential":
        for i in range(num_posterior_samples):
            xi[i] = sp.stats.expon.rvs(scale=1 / theta_sample[i], size=num_likelihood_samples, random_state=generator)
    elif likelihood == "normal" and posterior == "normal_gamma":
        for i in range(num_posterior_samples):
            # NOTE numpy normal takes standard deviation as scale parameter - not variance or precision!
            # so when we use normal_gamma (which is on the precision), we need to inverse and take sqrt
            xi[i] = generator.normal(theta_sample[i,0], np.sqrt(1.0 / theta_sample[i,1]), size=num_likelihood_samples)
    else:
        raise NotImplementedError(f"Likelihood {likelihood} with posterior {posterior} not implemented.")
    return xi
