"""Functions for the Newsvendor Problem"""

import numpy as np
import cvxpy as cp

BACKORDER_COST = 8  # denoted b
HOLDING_COST = 3  # denoted h


def newsvendor_cost(x, xi):
    """Evaluate Newsvendor cost function

    Args:
        x: Demand decision variable
        xi: Realised random demand
    """
    return HOLDING_COST * np.maximum(0, x - xi) + BACKORDER_COST * np.maximum(0, xi - x)

#cvxpy-compatible loss
def newsvendor_cost_cvxpy(x, xi):
    """Evaluate Newsvendor cost function in cvxpy

    Args:
        x: Demand decision variable
        xi: Realised random demand
    """
    # zero_vec = cp.Parameter(1, value=[0.0])
    # x_vec = cp.Parameter(1, value=[x - xi])
    # x_vec_= cp.Parameter(1, value=[xi - x])
    # return HOLDING_COST * cp.max(cp.vstack([zero_vec, x_vec])) + BACKORDER_COST * cp.max(cp.vstack([zero_vec, x_vec_]))
    return HOLDING_COST * cp.maximum(0, x - xi) + BACKORDER_COST * cp.maximum(0, xi - x)

