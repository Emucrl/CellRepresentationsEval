"""test GraphDataset"""

from pathlib import Path

import numpy as np
import torch_geometric
from torch_geometric.data import Data

from src.utils.graph_dataset import GraphDataset

TEST_DATA = Path("tests/test_data/graphs")

FEATURES_LIST = [
    "Hoechst2",
    "Hoechst3",
    "Hoechst4",
    "Hoechst5",
    "Hoechst6",
    "Hoechst7",
    "Hoechst8",
    "Hoechst9",
    "pAKT",
    "HER2",
    "Vimentin",
    "Catenin",
    "NGFR",
    "HCSred",
    "Area",
    "Circ",
    "X",
    "Y",
    "deg",
]


def test_dataset():
    """test the dataset creation (len and features)"""

    dataset1 = GraphDataset(TEST_DATA, features_list=FEATURES_LIST[:2])
    data1 = dataset1[0]
    assert len(dataset1) == 2
    assert data1.x.shape[1] == 2

    dataset2 = GraphDataset(TEST_DATA, features_list=FEATURES_LIST)
    data2 = dataset2[0]
    assert len(dataset2) == 2
    assert isinstance(data2, Data)
    assert data2.x.shape[1] == len(FEATURES_LIST)
    data_adj2 = torch_geometric.utils.to_scipy_sparse_matrix(
        data2.edge_index, data2.edge_weight
    )
    data_adj1 = torch_geometric.utils.to_scipy_sparse_matrix(
        data1.edge_index, data1.edge_weight
    )

    np.testing.assert_almost_equal(data_adj1.A, data_adj2.A, decimal=4)
    np.testing.assert_almost_equal(
        np.array(data2.x)[:, :2], np.array(data1.x), decimal=2
    )


def test_update_dataset_features_list():
    """test the dataset creation (len and features)"""
    dataset = GraphDataset(TEST_DATA, features_list=FEATURES_LIST)

    old_feature_list = dataset.features_list
    data = dataset[0]
    assert data.x.shape[1] == len(old_feature_list)

    dataset.update_features_list(old_feature_list[:2])
    small_data = dataset[0]

    assert dataset.features_list == old_feature_list[:2]
    np.testing.assert_allclose(
        small_data.x.numpy(),
        data.x.numpy()[:, :2],
    )

    dataset.update_features_list(old_feature_list)
    big_data = dataset[0]

    assert dataset.features_list == old_feature_list
    np.testing.assert_allclose(
        big_data.x.numpy(),
        data.x.numpy(),
    )
