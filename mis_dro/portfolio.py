"""Functions for the portfolio problem"""


from typing import Optional
import numpy as np
import cvxpy as cp
from .bayes_conjugates import sample_posterior, upper_triangular_size

def get_kl_portfolio_problem(num_stocks: int, num_cov_samples: int) -> cp.Problem:
    """Evaluate portfolio cost function with cvxpy assuming a Gaussian likelihood

    Args:
        num_stocks: Number of stocks in portfolio, i.e. dimension of random variable
        num_cov_samples: Number of covariance samples from the posterior

    Returns:
        Cvxpy problem object
    """
    # variables
    x = cp.Variable(num_stocks, name="x")

    # parameters
    epsilon_minus_constant = cp.Parameter(1, name="epsilon_minus_constant", nonneg=True)
    mu_post = cp.Parameter(num_stocks, name="mu_post")
    sqrt_cov_post_samples = [cp.Parameter((num_stocks, num_stocks), name=f"sqrt_cov_post_{i}") for i in range(num_cov_samples)]

    # objective function: maximise return whilst minimising standard deviation
    portfolio_objective = cp.Minimize(- mu_post @ x + cp.sqrt(2 * epsilon_minus_constant) * (1.0 / float(num_cov_samples)) * cp.sum(
        [cp.norm(sqrt_cov_post_samples[i] @ x) for i in range(num_cov_samples)]
    ))

    # constraints
    constraints = [x >= 0, cp.sum(x) == 1]

    return cp.Problem(portfolio_objective, constraints)


def bdro_portfolio_posterior_samples(num_posterior_samples: int, mu_post: np.array, iota_post: float, Psi_post: np.array, generator: Optional[np.random.Generator] = None) -> np.array:
    """With KL BDRO, we only need to sample the covariance using an inverse Wishart.
    We don't need to sample the mean because its available in closed form."""
    # get covariance samples from inverse Wishart
    dim = Psi_post.shape[0]
    inverse_wishart_params = (iota_post, Psi_post)
    vec_triu_cov_samples = sample_posterior("inverse_wishart", inverse_wishart_params, num_posterior_samples, generator=generator)

    # vectorize the covariance matrix
    vec_triu_size = vec_triu_cov_samples.shape[1]

    # return the samples in vectorized format
    theta_sample = np.zeros((num_posterior_samples, dim+vec_triu_size))
    for i, vec_triu_cov in enumerate(vec_triu_cov_samples):
        theta_sample[i, :dim] = mu_post
        theta_sample[i, dim:] = vec_triu_cov
    return theta_sample
