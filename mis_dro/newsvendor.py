"""Functions for the Newsvendor Problem"""

import cvxpy as cp

BACKORDER_COST = 8  # denoted b
HOLDING_COST = 3  # denoted h


def newsvendor_cost_cvxpy(x, xi):
    """Evaluate Newsvendor cost function with cvxpy

    Args:
        x: Demand decision variable
        xi: Realised random demand
    """
    return HOLDING_COST * cp.maximum(0, x - xi) + BACKORDER_COST * cp.maximum(0, xi - x)
