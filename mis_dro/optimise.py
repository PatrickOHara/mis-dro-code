"""Optimisation code"""

from typing import List, Tuple
import cvxpy as cp
import numpy as np

from bayesian_dro.Bayesian_DRO_continuous import LARGEST_X, SMALLEST_X
from .newsvendor import BACKORDER_COST, HOLDING_COST

def newsvendor_cost_cvxpy(x, xi):
    """Evaluate Newsvendor cost function with cvxpy

    Args:
        x: Demand decision variable
        xi: Realised random demand
    """
    return HOLDING_COST * cp.maximum(0, x - xi) + BACKORDER_COST * cp.maximum(0, xi - x)

def solve_bdro(xi: np.typing.ArrayLike, epsilon: float) -> Tuple[float, List[float]]:
    """Bayesian DRO as a cvxpy optimisaton problem.
    
    We use an epigraph variable t to upper bound the function G(x, xi).
    That is, we add constraints G(x, xi[i]) <= t[i] for all i = 1,...,num_theta.
    
    The main optimisation trick is then to use the perspective of the log-sum-exp function.

    Args:
        xi: Data sampled from the likelihood
        epsilon: Size of the KL-ball

    Returns:
        x: The decision variable
        lam: List of lagrangian variables for each posterior sample

    Notes:
        Requires the MOSEK solver to be installed
    """
    num_theta_samples = xi.shape[0]
    num_xi_samples = xi.shape[1]

    x = cp.Variable(1, name="x")
    lam = [cp.Variable(1, name=f"lam_{i}", nonneg=True) for i in range(num_theta_samples)]
    t = cp.Variable(xi.shape, name="t")

    objective = cp.Minimize(
        (1.0/num_theta_samples) * cp.sum([
            lam[i] * epsilon
            + lam[i] * cp.log(1.0 / num_xi_samples)
            + cp.perspective(cp.log_sum_exp(t[i]), lam[i])
            for i in range(num_theta_samples)
        ])
    )
    constraints = [
        x >= SMALLEST_X,
        x <= LARGEST_X,
    ] + [newsvendor_cost_cvxpy(x, xi[i]) <= t[i] for i in range(num_theta_samples)]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver="MOSEK")
    return x.value[0], np.array([lam[i].value for i in range(num_theta_samples)])
