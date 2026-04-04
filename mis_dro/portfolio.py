"""Functions for the portfolio problem"""


from typing import Optional, Sequence
import numpy as np
import cvxpy as cp
from .bayes_conjugates import sample_posterior

def portfolio_objective_cvxpy(x, xi):
    """CVXPY portfolio objective"""
    return - xi @ x

def get_kl_portfolio_problem(
        num_stocks: int,
        num_cov_samples: int,
        include_tcosts_in_cost_function: bool = False,
        prev_stock_figi_list: Optional[list[str]] = None,
        prev_portfolio_weighting: Optional[list[float]] = None,
        stock_figi_list_this_window: Optional[list[str]] = None
    ) -> cp.Problem:
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

    if include_tcosts_in_cost_function:
        u = get_cvxpy_transaction_cost_addend(
            prev_stock_figi_list,
            prev_portfolio_weighting,
            stock_figi_list_this_window,
            x
        )
    else:
        u = 0.0

    # objective function: maximise return whilst minimising standard deviation
    portfolio_objective = cp.Minimize(- mu_post @ x + u + cp.sqrt(2 * epsilon_minus_constant) * (1.0 / float(num_cov_samples)) * cp.sum(
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

def calculate_transaction_cost(
        # TODO: now that you've changed the allowed types in this function signature, maybe go to where you are forcing a conversion of an argument to list before passing it to this function and lax such efforts--particularly for prev_portfolio_weighting and new_portfolio_weighting, I think
        prev_stock_figi_list: Sequence[str],
        prev_portfolio_weighting: Sequence[float],
        new_stock_figi_list: Sequence[str],
        new_portfolio_weighting: Sequence[float]
    ) -> float:
    if len(prev_stock_figi_list) != len(prev_portfolio_weighting):
        raise ValueError("prev_stock_figi_list and prev_portfolio_weighting must be same length.")
    if len(new_stock_figi_list) != len(new_portfolio_weighting):
        print(len(new_stock_figi_list), len(new_portfolio_weighting))
        raise ValueError("new_stock_figi_list and new_portfolio_weighting must be same length.")
    prev_map = {figi: float(w) for figi, w in zip(prev_stock_figi_list, prev_portfolio_weighting)}
    new_map  = {figi: float(w) for figi, w in zip(new_stock_figi_list,  new_portfolio_weighting)}
    all_figis = set(prev_map) | set(new_map)
    return 0.005 * sum(abs(new_map.get(figi, 0.0) - prev_map.get(figi, 0.0)) for figi in all_figis)

def compute_drifted_returns_and_final_weights(
    data_eval: np.ndarray,
    solution: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute weekly portfolio returns under buy-and-hold with weight drift.

    Parameters
    ----------
    data_eval : np.ndarray
        Array of shape (m, n), where m is the number of test weeks and
        n is the number of stocks. Each row contains weekly simple stock returns:
            (DACP_t - DACP_{t-1}) / DACP_{t-1}
    solution : np.ndarray
        Initial portfolio weights at the rebalance date, shape (n,) or (n, 1).
        Must be non-negative and sum to 1.

    Returns
    -------
    weekly_portfolio_returns : np.ndarray
        Array of shape (m,), containing the weekly portfolio returns under drift.
    final_weights : np.ndarray
        Array of shape (n,), containing the final drifted portfolio weights
        after the last test week.
    """
    data_eval = np.asarray(data_eval, dtype=float)
    weights = np.asarray(solution, dtype=float).reshape(-1)

    # Checks
    if data_eval.ndim != 2: raise ValueError("data_eval must be a 2D array of shape (m, n).")

    m, n = data_eval.shape

    # Checks
    if weights.shape[0] != n: raise ValueError(f"solution has length {weights.shape[0]}, but data_eval has {n} stocks.")

    weekly_portfolio_returns = np.empty(m, dtype=float)

    for t in range(m):
        stock_returns_t = data_eval[t]  # shape (n,)

        # Weekly portfolio return using start-of-week weights
        portfolio_return_t = weights @ stock_returns_t
        weekly_portfolio_returns[t] = portfolio_return_t

        # Update weights by drift:
        # w_t = [w_{t-1} * (1 + r_t)] / (1 + portfolio_return_t)
        gross_stock_returns_t = 1.0 + stock_returns_t
        gross_portfolio_return_t = 1.0 + portfolio_return_t

        # Check
        if gross_portfolio_return_t <= 0: raise ValueError(f"Portfolio gross return became non-positive at week {t}: {gross_portfolio_return_t}.")

        # Calculate drifted portfolio weighting by the end of this week
        weights = (weights * gross_stock_returns_t) / gross_portfolio_return_t

        # Numerical cleanup
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()

    return weekly_portfolio_returns, weights

# NOTE (pwd): complete the below
def apply_transaction_cost_to_last_portfolio_return(last_portfolio_return: float, transaction_cost: float) -> float:

    return (1 + last_portfolio_return) * (1 - transaction_cost) - 1

def get_cvxpy_transaction_cost_addend(
        prev_stock_figi_list: list[str],
        prev_portfolio_weighting: list[float],
        stock_figi_list_this_window: list[str],
        x: cp.Variable
    ) -> cp.Expression:

    # TODO: James: may be worth refactoring some of the below with what's in portfolio.calculate_transaction_cost
    # TODO: James: consider catching errors regarding prev_stock_figi_list and prev_portfolio_weighting length mistmatches etc.

    prev_map = dict(zip(prev_stock_figi_list, prev_portfolio_weighting))
    y_aligned = np.array(
        [prev_map.get(figi, 0.0) for figi in stock_figi_list_this_window],
        dtype=float,
    )
    prev_only_cost = sum(
        abs(prev_map[figi]) for figi in prev_map if figi not in set(stock_figi_list_this_window)    # TODO: James: abs shouldn't be required in hindsight, but including it shouldn't make a functional difference either
    )
    y = cp.Constant(y_aligned)
    return 0.005 / 13 * (cp.norm1(x - y) + prev_only_cost)
    