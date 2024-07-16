"""Closed-form expression for Bayesian conjugate models"""

from typing import Optional
import numpy as np

def normal_gamma_posterior(
        data: np.ndarray,
        mu_prior: float,
        kappa_prior: float,
        alpha_prior: float,
        beta_prior: float,
) -> tuple[float, float, float, float]:
    """Get the closed-form expression for normal-gamma posterior

    Returns:
        Posterior parameters for alpha, beta, mu, kappa

    Notes:
        See Section 3.3 of Murphy (2007).

    References:
        Murphy, K. P. (2007). Conjugate bayesian analysis of the gaussian distribution.
        Technical report, Department of Computer Science, The University of British Columbia.
    """
    num_observations = data.shape[0]
    data_mean = np.mean(data)
    print("Data mean:", data_mean)
    print("Num observations:", num_observations)
    mu_posterior = (kappa_prior * mu_prior + num_observations * data_mean) / (num_observations * kappa_prior)
    kappa_posterior = kappa_prior + num_observations
    alpha_posterior = alpha_prior + 0.5 * num_observations
    # beta_posterior = beta_prior + 0.5 * np.sum(np.square(data - data_mean)) + (0.5 * num_observations * kappa_prior * np.square(data_mean - mu_prior)) / (kappa_prior * num_observations)
    beta_posterior = beta_prior + 0.5 * np.mean(np.square(data - data_mean)) + (0.5 * num_observations * kappa_prior * np.square(data_mean - mu_prior)) / (kappa_prior * num_observations)
    return mu_posterior, kappa_posterior, alpha_posterior, beta_posterior

def normal_gamma_rvs(
        num_samples: int,
        mu: float,
        kappa: float,
        alpha: float,
        beta: float,
        generator = None
    ) -> np.ndarray:
    """Normal-gamma random variates"""
    samples = np.zeros((num_samples, 2))
    if not generator:
        generator = np.random.default_rng()
    precision_samples = generator.gamma(alpha, beta, num_samples)
    for i in range(num_samples):
        samples[i, 0] = generator.normal(mu, 1 / (kappa * precision_samples[i]), 1)
    samples[:, 1] = precision_samples
    return samples

