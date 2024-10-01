"""Closed-form expression for Bayesian conjugate models"""

from typing import Optional
import numpy as np
import scipy as sp
from bayesian_dro.Bayesian_DRO_continuous import DGP_STD_TRUNCATED_NORMAL


def default_prior_params(prior: str, dim: int = 1) -> tuple:
    """Get the default prior parameters"""
    prior_params = ()
    if prior == "gamma":
        alpha_prior, beta_prior = 1, 1
        prior_params = (alpha_prior, beta_prior)
    elif prior == "normal_gamma":
        mu_prior, kappa_prior, alpha_prior, beta_prior = 0.0, 1.0, 1.0, 1.0
        prior_params = (mu_prior, kappa_prior, alpha_prior, beta_prior)
    elif prior == "normal_inverse_wishart":
        prior_params = normal_inverse_wishart_prior(dim)
    elif prior == "multivariate_normal_known_cov":
        prior_params = (np.zeros(dim), dim+2)
    else:
        raise NotImplementedError(f"Prior '{prior}' not implemented.")
    return prior_params


def get_posterior_params(
    posterior: str, data: np.ndarray, prior_params: tuple
) -> tuple:
    """Given prior parameters, return the updated posterior parameters"""
    post_params = []
    if posterior == "gamma":
        alpha_prior, beta_prior = prior_params
        alpha_posterior = alpha_prior + data.shape[0]
        beta_posterior = beta_prior + np.sum(data)
        post_params = (alpha_posterior, beta_posterior)
    elif posterior == "normal_gamma":
        mu_prior, kappa_prior, alpha_prior, beta_prior = prior_params
        (
            mu_posterior,
            kappa_posterior,
            alpha_posterior,
            beta_posterior,
        ) = normal_gamma_posterior(data, mu_prior, kappa_prior, alpha_prior, beta_prior)
        post_params = (mu_posterior, kappa_posterior, alpha_posterior, beta_posterior)
    elif posterior == "normal_inverse_wishart":
        post_params = normal_inverse_wishart_posterior(data, *prior_params)
    elif posterior == "multivariate_normal_known_cov":
        mu_prior, kappa_prior = prior_params
        N = data.shape[0]
        xi_mean = np.mean(data, axis=0)
        kappa_posterior = kappa_prior + N
        mu_posterior = (kappa_prior * mu_prior + N * xi_mean) / kappa_posterior
        post_params = (mu_posterior, kappa_posterior)
    else:
        raise NotImplementedError(f"Posterior '{posterior}' is not implemented")
    return post_params


def derive_analytical_posterior_params(
    posterior: str, posterior_params: tuple
) -> np.ndarray:
    """When using our closed form expressions for 'Bayesian ambiguity sets',
    we derive a new distribution from the posterior parameters"""
    if posterior == "gamma":
        alpha_posterior, beta_posterior = posterior_params
        return np.array([alpha_posterior / beta_posterior])
    elif posterior == "normal_gamma":
        theta = np.zeros((1, 2))
        mu_posterior, _, alpha_posterior, beta_posterior = posterior_params
        # we want an analytical form for the precision, which is alpha over beta
        theta[0] = np.array([mu_posterior, alpha_posterior / beta_posterior])
        return theta
    elif posterior == "normal_inverse_wishart":
        mu_post, kappa_post, _, Psi_post = posterior_params
        # Sigma_hat = 0.5 * np.linalg.inv(np.outer(mu_post, mu_post) - (1/kappa_post)*Psi_post)
        # mu_hat = Sigma_hat @ mu_post
        dim = mu_post.shape[0]
        vec_triu_size = upper_triangular_size(dim)
        theta = np.zeros((1, dim+int(vec_triu_size)))
        theta[0, :dim] = mu_post
        theta[0, dim:] = (1/kappa_post) * Psi_post[np.triu_indices(dim)]
        return theta
    elif posterior == "multivariate_normal_known_cov":
        mu_posterior, _ = posterior_params
        dim = mu_posterior.shape[0]
        return mu_posterior.reshape((1,dim))
    else:
        raise NotImplementedError(
            f"We haven't derived an analytical posterior expression for a '{posterior}' posterior"
        )


def sample_posterior(
    posterior: str,
    posterior_params: tuple,
    num_posterior_samples: int,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample from the posterior"""
    if posterior == "gamma":
        alpha_post, beta_post = posterior_params
        return sp.stats.gamma.rvs(
            a=alpha_post,
            scale=1 / beta_post,
            size=num_posterior_samples,
            random_state=generator,
        )
    if posterior == "normal_gamma":
        (
            mu_posterior,
            kappa_posterior,
            alpha_posterior,
            beta_posterior,
        ) = posterior_params
        return normal_gamma_rvs(
            num_posterior_samples,
            mu_posterior,
            kappa_posterior,
            alpha_posterior,
            beta_posterior,
            generator=generator,
        )
    if posterior == "normal_inverse_wishart":
        return normal_inverse_wishart_samples(num_posterior_samples, *posterior_params, generator=generator)
    if posterior == "multivariate_normal_known_cov":
        mu_posterior, kappa_posterior = posterior_params
        return sp.stats.multivariate_normal.rvs(mean=mu_posterior, cov=(DGP_STD_TRUNCATED_NORMAL**2)*np.eye(5), size=num_posterior_samples, random_state=generator)
    else:
        raise NotImplementedError(f"Posterior '{posterior}' is not implemented")


def get_log_partition_constant(posterior: str, posterior_params: list) -> float:
    """Given the posterior params, return the optimization constant for Bayesian DRO"""
    if posterior == "gamma":
        alpha_posterior, _ = posterior_params
        return np.log(alpha_posterior) - sp.special.digamma(alpha_posterior)
    elif posterior == "normal_gamma":
        _, kappa_posterior, alpha_posterior, _ = posterior_params
        return get_normal_gamma_constant(alpha_posterior, kappa_posterior)
    elif posterior == "normal_inverse_wishart":
        return 0.0  # FIXME
    elif posterior == "multivariate_normal_known_cov":
        mu_posterior, kappa_posterior = posterior_params
        dim = mu_posterior.shape[0]
        return dim/(2*kappa_posterior)
    else:
        raise NotImplementedError(f"get_log_partition_constant not implemented for posterior {posterior}")


def get_normal_gamma_constant(alpha: int, kappa: float) -> float:
    """Get the constant for normal-gamma DRO"""
    return 0.5 * (1 / kappa + np.log(alpha) - sp.special.digamma(alpha))


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
    mu_posterior = (kappa_prior * mu_prior + num_observations * data_mean) / (
        num_observations * kappa_prior
    )
    kappa_posterior = kappa_prior + num_observations
    alpha_posterior = alpha_prior + 0.5 * num_observations
    beta_posterior = (
        beta_prior
        + 0.5 * np.sum(np.square(data - data_mean))
        + (0.5 * num_observations * kappa_prior * np.square(data_mean - mu_prior))
        / (kappa_prior * num_observations)
    )
    return mu_posterior, kappa_posterior, alpha_posterior, beta_posterior


def normal_gamma_rvs(
    num_samples: int,
    mu: float,
    kappa: float,
    alpha: float,
    beta: float,
    generator: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Normal-gamma random variates

    Returns:
        mu_samples:
        standard_dev_samples:

    Notes:
        1. Sample precision `lambda` from a Gamma(`alpha`, `beta`) distribution
        2. Sample from a normal distribution with mean `mu` and standard deviation `sqrt(1/(kappa * lambda))`
    """
    samples = np.zeros((num_samples, 2))
    if not generator:
        generator = np.random.default_rng()
    # TODO make these standard deviation samples! By taking inverse sqrt
    precision_samples = generator.gamma(alpha, 1.0 / beta, num_samples)
    standard_devs = 1 / np.sqrt(kappa * precision_samples)
    for i in range(num_samples):
        # NOTE numpy Normal distribution takes standard deviation as a parameter (hence sqrt) - not variance!
        samples[i, 0] = generator.normal(
            mu, standard_devs[i], 1
        )
    samples[:, 1] = standard_devs
    return samples

def normal_inverse_wishart_prior(dim: int):
    """Default normal-inverse-Wishart prior hyperparameters for D dimensions
    
    Args:
        D: dimension

    Returns:
        mu: Prior mean. Vector with shape (D,)
        kappa: Reflects belief in prior mean (positive scalar)
        iota: Reflects belief in prior over covariance (positive scalar)
        Psi: matrix proportional to prior over covariance. Matrix with shape (D,D).
    """
    # since we derived our result via the exponential family, we set kappa = iota + D + 2
    iota = float(dim + 1)   # degrees of freedom must be greater than dim
    kappa = iota + dim + 2
    return np.zeros(dim), kappa, iota, np.identity(dim)

def normal_inverse_wishart_posterior(data, mu_prior, kappa_prior, iota_prior, Psi_prior):
    """Normal-inverse-Wishart posterior hyperparameters for D dimensions

    Args:
        data: Observations with shape (N, D)
        mu: Prior mean. Vector with shape (D,)
        kappa: Reflects belief in prior mean (positive scalar)
        iota: Reflects belief in prior over covariance (positive scalar)
        Psi: matrix for prior over covariance. Matrix with shape (D,D).

    Returns:
        mu: Updated posterior mean. Vector with shape (D,)
        kappa: Updated belief in posterior mean (positive scalar)
        iota: Updated belief in posterior over covariance (positive scalar)
        Psi: Updated matrix for posterior over covariance. Matrix with shape (D,D).

    Notes:
        See Section 3.4.4.3 of Murphy (2023) Probabilistic Machine Learning: Advanced Topics.
    """
    N = data.shape[0]
    xi_mean = np.mean(data, axis=0)
    kappa_post = kappa_prior + N
    mu_post = (kappa_prior * mu_prior + N * xi_mean) / kappa_post
    iota_post = iota_prior + N
    # Psi update comes from eq. (3.172) of Murphy (see notes above)
    Psi_post = Psi_prior + data.T @ data + kappa_prior * np.outer(mu_prior, mu_prior) - kappa_post * np.outer(mu_post, mu_post)
    return mu_post, kappa_post, iota_post, Psi_post

def normal_inverse_wishart_samples(num_samples: int, mu: np.ndarray, kappa: float, iota: float, Psi: np.ndarray, generator: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray]:
    """Get N samples of mean and covariance from the normal-inverse-Wishart distribution.

    Args:
        num_samples: N for short
        mu: Mean hyperparameter. Vector with shape (D,)
        kappa: Positive scalar
        iota: Positive scalar
        Psi: Matrix with shape (D,D)
        generator: Numpy random number generator

    Returns:
        mu_samples: Matrix with shape (N,D)
        cov_samples: Matrix with shape (N, D + D*(D-1)/2 + D).
            The covariance samples are stored as a vector in upper triangular format.
    """
    dim = mu.shape[0]
    assert dim == Psi.shape[0] and dim == Psi.shape[1]
    vec_triu_size = upper_triangular_size(dim)
    samples = np.zeros((num_samples, dim+vec_triu_size))

    # sample from the inverse Wishart
    iw_post = sp.stats.invwishart(iota, Psi, seed=generator)
    cov_samples = iw_post.rvs(num_samples)

    # sample from a multivariate normal given the covariance samples
    idx_triu = np.triu_indices(dim)
    for i, cov in enumerate(cov_samples):
        samples[i, :dim] = sp.stats.multivariate_normal(mu, 1/kappa * cov, seed=generator).rvs()
        samples[i, dim:] = cov[idx_triu]
    return samples

def multivariate_digamma_p(a: float, p: int) -> float:
    """The multivariate digamma function of dimension p

    Notes:
        [1] https://search.r-project.org/CRAN/refmans/CholWishart/html/mvdigamma.html
        [2] https://en.wikipedia.org/wiki/Multivariate_gamma_function#Derivatives
    """
    return np.sum([sp.special.digamma(a + (1-i)/2) for i in range(p)])

def get_normal_inverse_wishart_G_constant(mu_post: np.ndarray, kappa_post: float, Psi_post: np.ndarray) -> float:
    """Returns the constant G(tau, nu) for the normal-inverse-Wishart"""
    dim = mu_post.shape[0]
    Psi_inv = sp.linalg.inv(Psi_post)
    mu_Psi_inv_mu = mu_post.T @ Psi_inv @ mu_post
    term1 = 0.5 * np.log(np.linalg.det(np.outer(mu_post, mu_post) - (1 / kappa_post) * Psi_post))
    term2 = 0.25 * mu_Psi_inv_mu * ((kappa_post**2 * mu_Psi_inv_mu)/(1 - kappa_post * mu_Psi_inv_mu) - kappa_post + 2 * dim + 4)
    term3 = - 0.5 * multivariate_digamma_p(0.5 * (kappa_post - dim - 2), dim)
    term4 = dim/(2*kappa_post)
    term5 = np.log(np.linalg.det(Psi_post))
    return term1 + term2 + term3 + term4 + term5

def upper_triangular_size(dim: int) -> int:
    """Includes the diagonal!"""
    return dim * (dim-1) / 2 + dim
