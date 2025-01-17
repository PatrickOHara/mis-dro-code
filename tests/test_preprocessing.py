"""Test preprocessing functions"""

import pytest
import numpy as np
from mis_dro.dataset import sample_dgp
from mis_dro.preprocessing import normalise_by_dimension


def test_normalise_by_dimension():
    generator = np.random.default_rng(0)
    dim = 5
    data = sample_dgp("multivariate_normal", 20, dim=dim, generator=generator)
    normalised_data = normalise_by_dimension(data)
    for i in range(dim):
        assert np.isclose(normalised_data[:,i], (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])).all()
