"""Functions for the Newsvendor Problem"""

import cvxpy as cp
import numpy as np

BACKORDER_COST = 8  # denoted b
HOLDING_COST = 3  # denoted h


def newsvendor_cost_cvxpy(x, xi, dim: int = 1):
    """Evaluate Newsvendor cost function with cvxpy

    Args:
        x: Demand decision variable
        xi: Realised random demand
        dim: Dimension of random variable and decision variable
    """
    assert x.shape[0] == dim
    assert xi.shape[2] == dim
    # NOTE we could define a different holding and backorder cost for each product,
    # but I think its simpler (at least for now) to keep the same cost for every product
    h = HOLDING_COST * np.ones(dim)
    b = BACKORDER_COST * np.ones(dim)
    return cp.maximum(0, x - xi) @ h + cp.maximum(0, xi - x) @ b

