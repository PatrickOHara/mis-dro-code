
import pytest
from mis_dro.bayes_conjugates import upper_triangular_size

@pytest.mark.parametrize(("dim", "expected_size"), [(2, 3), (3, 6), (4, 10)])
def test_upper_triangular_size(dim, expected_size):
    assert upper_triangular_size(dim) == expected_size