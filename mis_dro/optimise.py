"""Optimisation code"""

import cvxpy as cp
import numpy as np
from scipy.special import logsumexp
from bayesian_dro.Bayesian_DRO_continuous import cost

def bdro(xi, epsilon, num_theta_samples) -> float:
    """Bayesian DRO as a cvxpy optimisaton problem"""
    x = cp.Variable(1, name="x")
    lam = cp.Variable(1, name="lam")

    N_xi = xi.shape[0]

    objective = cp.Minimize(
        (1.0/num_theta_samples) * cp.sum([
            lam * epsilon + lam * np.log(1.0 / N_xi) + logsumexp(cost(x, xi) / lam)
        ])
    )
    constraints = [
        x >= 5,
        x <= 50,
        lam >= 0,
    ]
    problem = cp.Problem(objective, constraints)
    obj_value = problem.solve()
    print("Value of x:", x.value)
    print("Value of lambda:", lam.value)
    print("Objective:", obj_value)
    return obj_value