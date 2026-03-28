import numpy as np
import scipy as sp
from sklearn.model_selection import KFold

from .bayes_conjugates import default_prior_params, get_posterior_params, get_log_partition_constant, derive_analytical_posterior_params, posterior_predictive_params
from .kl_divergence import kl_divergence_gaussian_kde_monte_carlo

def get_num_observations_in_train_split(n_splits: int, split_idx: int, n_observations: int):
    ratio = float(n_observations) / float(n_splits)
    num_splits_with_less_than_max_test_size = n_splits * np.ceil(ratio) - n_observations
    if split_idx < n_splits - num_splits_with_less_than_max_test_size:
        return int(n_observations - np.ceil(ratio))
    else:
        return int(n_observations - np.floor(ratio))
    
def get_kde_epsilon_cross_validation(data: np.array, algorithm: str, posterior: str, likelihood: str, n_splits: int, cv_seed: int) -> float:
    cv_random_state = np.random.RandomState(seed=cv_seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
    dim = data.shape[1]
    epsilon_values = np.zeros(n_splits)
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        fold_train =  data[train_index]
        fold_test = data[test_index]
        theta_prior = default_prior_params(posterior, dim=dim)
        theta_posterior = get_posterior_params(posterior, fold_train, theta_prior)
        if algorithm == "kl_dro_bas":
            log_partition_constant = get_log_partition_constant(posterior, theta_posterior)
            theta_sample = derive_analytical_posterior_params(
                posterior, theta_posterior
            )
            model = get_scipy_likelihood_from_theta(theta_sample[0], likelihood)
        elif algorithm == "kl_pp":
            theta_sample = posterior_predictive_params(posterior, theta_posterior)
            model = get_scipy_posterior_predictive(theta_sample[0], posterior, likelihood)
            log_partition_constant = 0.0
        else:
            raise NotImplementedError()
        epsilon_values[i] = kl_divergence_gaussian_kde_monte_carlo(fold_test, model) + log_partition_constant
    return epsilon_values.mean()

def get_scipy_likelihood_from_theta(theta: np.array, likelihood: str):
    if likelihood == "normal":
        mu, scale = theta
        return sp.stats.norm(loc=mu, scale=scale)
    if likelihood == "exponential":
        return sp.stats.expon(scale=1.0 / theta)
    raise NotImplementedError(f"Likelihood: {likelihood}")

def get_scipy_posterior_predictive(theta: np.array, posterior: str, likelihood: str):
    if likelihood == "normal" and posterior == "normal_gamma":
        mu, scale, df = theta
        return sp.stats.t(df, loc=mu, scale=scale)
    if likelihood == "exponential" and posterior == "gamma":
        shape, scale = theta
        sp.stats.lomax(c=shape, scale=scale)
    raise NotImplementedError(f"Posterior predictive for likelihood {likelihood} and posterior {posterior}.")