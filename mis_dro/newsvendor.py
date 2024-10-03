"""Functions for the Newsvendor Problem"""

import cvxpy as cp
import numpy as np

BACKORDER_COST = 8  # denoted b
HOLDING_COST = 3  # denoted h


# def newsvendor_cost_cvxpy(x, xi):
#     """Evaluate Newsvendor cost function with cvxpy

#     Args:
#         x: Demand decision variable
#         xi: Realised random demand
#     """
#     return HOLDING_COST * cp.maximum(0, x - xi) + BACKORDER_COST * cp.maximum(0, xi - x)

def newsvendor_cost_cvxpy_kl(x, xi):
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


