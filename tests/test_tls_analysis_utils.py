import numpy as np
import pandas as pd
import pytest

from src.experiments_nodes.tls_detection.generate_tls_masks import (
    density_based_clustering_tls,
)


@pytest.fixture(name="ones")
def get_ones():
    return np.ones((100, 10))


@pytest.fixture(name="zeros")
def get_zeros():
    return np.zeros((110, 10))


def test_density_based_clustering() -> None:
    """test that clustering is applied and cluster column added to the df"""
    output = density_based_clustering_tls(
        pd.DataFrame(np.random.random((1000, 3)), columns=["test", "y", "x"])
    )
    assert "cluster" in output.columns
    assert output.isna().sum().sum() == 0
