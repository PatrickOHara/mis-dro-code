"""Functions for the Newsvendor Problem"""

import numpy as np

BACKORDER_COST = 8  # denoted b
HOLDING_COST = 3  # denoted h


def newsvendor_cost(x, xi):
    """Evaluate Newsvendor cost function

    Args:
        x: Demand decision variable
        xi: Realised random demand
    """
    return HOLDING_COST * np.maximum(0, x - xi) + BACKORDER_COST * np.maximum(0, xi - x)
