"""Optimisation code"""

from typing import List, Iterable, Tuple, Union
import cvxpy as cp
import numpy as np
import scipy as sp

from bayesian_dro.Bayesian_DRO_continuous import LARGEST_X, SMALLEST_X
from .newsvendor import BACKORDER_COST, HOLDING_COST

def newsvendor_cost_cvxpy(x, xi):
    """Evaluate Newsvendor cost function with cvxpy

    Args:
        x: Demand decision variable
        xi: Realised random demand
    """
    return HOLDING_COST * cp.maximum(0, x - xi) + BACKORDER_COST * cp.maximum(0, xi - x)

def __get_epsilon_list(epsilon: Union[float, Iterable[float]]) -> list[float]:
    try:
        iterator = iter(epsilon)
        return list(iterator)
    except TypeError:
        if isinstance(epsilon, float):
            return [epsilon]
        raise TypeError(f"Parameter epsilon should be float or Iterable[float]. Got {type(epsilon)}")

def get_normal_gamma_constant(alpha: int, kappa: float) -> float:
    """Get the constant for normal-gamma DRO"""
    return 0.5 * (1/ kappa + np.log(alpha) - sp.special.digamma(alpha))


def get_kl_bdro_problem(xi: Union[float, np.ndarray], flexi_lambda: bool = True, posterior_constant: float = 1.0) -> Tuple[float, List[float]]:
    """Bayesian DRO as a cvxpy optimisaton problem.
    
    We use an epigraph variable t to upper bound the function G(x, xi).
    That is, we add constraints G(x, xi[i]) <= t[i] for all i = 1,...,num_theta.
    
    The main optimisation trick is then to use the perspective of the log-sum-exp function.

    Args:
        xi: Data sampled from the likelihood
        flexi_lambda: If true, then for each posterior sample, the Lagrangian variable lambda
            can take a difference value. Otherwise, lambda is the same variable for each posterior sample
        posterior_constant

    Returns:
        problem: A cvxpy Problem object
        x: The decision variable
        lam: List of lagrangian variables for each posterior sample

    Notes:
        Requires the MOSEK solver to be installed
    """
    # do type and shape checking of xi
    if isinstance(xi, float):
        num_theta_samples = 1
        num_xi_samples = 1
        xi = np.ndarray([[xi]])
    elif len(xi.shape) == 1:
        # ambiguous shape: is this lots of likelihoods samples for the same posterior sample,
        #   or is it one likelihood sample from lots of posterior samples?
        raise ValueError(f"Shape of xi array is {xi.shape}. Please reshape to (num_theta_samples, num_xi_samples).")
    elif len(xi.shape) == 2:
        num_theta_samples = xi.shape[0]
        num_xi_samples = xi.shape[1]
    else:
        raise ValueError(f"Pass xi as a 2D array or a float. Value of xi is: {xi}")
    assert xi.shape == (num_theta_samples, num_xi_samples)  # quick check

    # declare variables
    x = cp.Variable(1, name="x")
    if flexi_lambda:
        lam = [cp.Variable(1, name=f"lam_{i}", nonneg=True) for i in range(num_theta_samples)]
    else:
        raise NotImplementedError("Non-flexible lambda variable coming soon!")
    t = cp.Variable(xi.shape, name="t")

    # declare parameters
    epsilon_param = cp.Parameter(1, name="epsilon", nonneg=True)
    posterior_constant_param = cp.Parameter(1, name="posterior_constant", nonneg=True)
    xi_param = cp.Parameter((num_theta_samples, num_xi_samples))



    objective = cp.Minimize(
        (1.0/num_theta_samples) * cp.sum([
            lam[i] * epsilon_param
            + lam[i] * cp.log(1.0 / num_xi_samples)
            + cp.perspective(cp.log_sum_exp(t[i]), lam[i])
            for i in range(num_theta_samples)
        ])
    )
    constraints = [
        x >= SMALLEST_X,
        x <= LARGEST_X,
    ] + [newsvendor_cost_cvxpy(x, xi_param[i]) <= t[i] for i in range(num_theta_samples)]

    return cp.Problem(objective, constraints)

