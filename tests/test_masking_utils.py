"""masking utils tests"""
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.utils.masking_utils import apply_mask_cells_df, get_tls_mask


@pytest.fixture(name="filters")
def get_filters() -> Dict:
    return {
        "prefix": "test",
        "max_edge_size": 10,
        "clustering": 10,
        "patient": "patient",
        "indication": "blable",
        "datadump": "smth",
    }


@pytest.fixture(name="mask")
def get_mask() -> np.ndarray:
    """mask"""
    return np.array(
        [
            [1, 1, 1, 0, 0, 0, 3, 3, 3],
            [1, 1, 1, 0, 0, 0, 3, 3, 3],
            [1, 1, 1, 0, 0, 0, 3, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 0, 0, 0, 4, 4, 4],
            [2, 2, 2, 0, 0, 0, 4, 4, 4],
            [2, 2, 2, 0, 0, 0, 4, 4, 4],
        ]
    )


@pytest.fixture(name="dataframe")
def get_dataframe() -> pd.DataFrame:
    """df"""
    dataframe = pd.DataFrame(
        [[i, j, "test", 10, 10, "patient"] for i in range(9) for j in range(9)],
        columns=[
            "x",
            "y",
            "prefix",
            "max_edge_size",
            "clustering",
            "patient",
        ],
    )
    dataframe.insert(
        column="cluster",
        loc=0,
        value=[1] * (dataframe.shape[0] // 2)
        + [2] * (dataframe.shape[0] - (dataframe.shape[0] // 2)),
    )
    return dataframe


@pytest.fixture(name="img")
def get_img() -> np.ndarray:
    """img"""
    return np.repeat(
        np.expand_dims(
            np.array(
                [
                    [1, 1, 1, 0, 0, 0, 3, 3, 3],
                    [1, 1, 1, 0, 0, 0, 3, 3, 3],
                    [1, 1, 1, 0, 0, 0, 3, 3, 3],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 2, 2, 0, 0, 0, 4, 4, 4],
                    [2, 2, 2, 0, 0, 0, 4, 4, 4],
                    [2, 2, 2, 0, 0, 0, 4, 4, 4],
                ]
            ),
            -1,
        ),
        3,
        -1,
    )


def test_apply_mask_cells_df(mask: np.ndarray, dataframe: pd.DataFrame) -> None:
    """apply mask to cells_df"""
    masked = apply_mask_cells_df(dataframe, mask)
    assert (masked > 0).sum() == (mask > 0).sum()


def test_get_masks(dataframe: pd.DataFrame) -> None:
    """test get mask functions
    validity of the results is not important since they are morphology operations and the results should be validated qualitatively
    """

    get_tls_mask(dataframe, 1, (1600, 1600))
    get_tls_mask(dataframe, 2, (1000, 1000))
