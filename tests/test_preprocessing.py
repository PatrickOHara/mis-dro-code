"""Test preprocessing functions"""

import pytest
import numpy as np
from mis_dro.dataset import sample_dgp
from mis_dro.preprocessing import normalise_by_dimension


def test_normalise_by_dimension():
    generator = np.random.default_rng(0)
    dim = 5
    data = sample_dgp("multivariate_normal", 20, dim=dim, generator=generator)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    assert mean.shape[0] == data.shape[1]
    assert std.shape[0] == data.shape[1] 
    normalised_data = normalise_by_dimension(data, mean, std)
    for i in range(dim):
        normalised_dim = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])
        assert normalised_data[:,i].shape == normalised_dim.shape
        assert np.isclose(mean[i], np.mean(data[:,i]))
        assert np.isclose(std[i], np.std(data[:,i]))
        assert np.isclose(normalised_data[:,i], normalised_dim).all()

    data_eval = sample_dgp("multivariate_normal", 5, dim=dim, generator=generator)
    normalised_data_eval = normalise_by_dimension(data_eval, mean, std)
    assert normalised_data_eval.shape == data_eval.shape
