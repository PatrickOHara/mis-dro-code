import pytest
import numpy as np
from bayesian_dro.Bayesian_DRO_continuous import (
    Bayesian_DRO_1,
    Bayesian_DRO_1_lse,
    cost,
)


@pytest.mark.parametrize("xi0", [[709.0, 20.5], [5.1, 2.5], [20.5, 17.2], [19.2, 16.1]])
def test_Bayesian_DRO_1_lse(xi0):
    lam = 0.2
    xi = np.array([xi0])
    x = 2.5
    epsilon = 1.0
    assert np.isclose(
        Bayesian_DRO_1(lam, xi, x, 0, epsilon),
        Bayesian_DRO_1_lse(lam, xi, x, 0, epsilon),
    )
