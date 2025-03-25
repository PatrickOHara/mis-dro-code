"""Functions for the Newsvendor Problem"""

import cvxpy as cp
import numpy as np

BACKORDER_COST = 8  # denoted b
HOLDING_COST = 3  # denoted h

def newsvendor_cost_cvxpy(x, xi):
    """Evaluate Newsvendor cost function with cvxpy

    Args:
        x: Demand decision variable of dimension dim. Shape (dim,)
        xi: N samples of D-dimension random demand. Shape (N,dim).
    
    Returns:
        Vector of shape N
    """
    assert xi.shape[1] == x.shape[0]
    dim = x.shape[0]
    # NOTE we could define a different holding and backorder cost for each product,
    # but I think its simpler (at least for now) to keep the same cost for every product
    h = HOLDING_COST * np.ones(dim)
    b = BACKORDER_COST * np.ones(dim)
    X = cp.vstack([x for _ in range(xi.shape[0])])
    return cp.maximum(0, X - xi) @ h + cp.maximum(0, xi - X) @ b

def empirical_wasserstein_dro_newsvendor(data, epsilon, p: int = 2, b: float = BACKORDER_COST, h: float = HOLDING_COST):
    """Univariate empirical Wasserstein distributionally robust newsvendor problem"""
    len_data = len(data)
    if p == 1:
        # put data in ascending order
        raise NotImplementedError("I don't trust this yet - how is epsilon used?")
        ascend_data = np.sort(data)
        for i in range(1, len_data + 1):
            if (i - 1) / len_data < b / (h + b) and i / len_data >= b / (h + b):
                return ascend_data[i - 1]
    if p > 1:
        Delta = (
            1
            / (h + b)
            * (1 / p) ** (1 / (p - 1))
            * ((p - 1) / p)
            * (b ** (p / (p - 1)) - h ** (p / (p - 1)))
        )
        Lambda = (1 / (h + b)) * (b ** (p / (p - 1)) * h + h ** (p / (p - 1)) * b)
        ascend_data = np.sort(data)
        # NOTE appears to do a bisection search here - wonder where this comes from?
        for i in range(1, len_data + 1):
            if (i - 1) / len_data < b / (h + b) and i / len_data >= b / (h + b):
                temp = ascend_data[i - 1]
                break
        return temp + Delta * p ** (1 / (p - 1)) * epsilon * (1 / Lambda) ** (1 / p)
